#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# ============================================================
# IMPORT LIBRERIE
# ============================================================
import rospy
import cv2          # OpenCV
import numpy as np
import os
import rospkg
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from marta_msgs.msg import SideScanSonar
from marta_msgs.msg import Altitude
from marta_msgs.msg import NavStatus
from sss_package.msg import ImageMetadata
from cv_bridge import CvBridge
from std_msgs.msg import String


# ============================================================
# CLASSE PRINCIPALE NODO WATERFALL CREATOR
# ============================================================
class WaterfallCreatorNode:
    WATERFALL_BUFFER_SIZE = 500

    def __init__(self):

        print("[SSS] waterfall_creator_node.py is active\n")

        # inizializzare publisher
        self.pub_img                = rospy.Publisher('waterfall_image_topic', ImageMetadata, queue_size=100)
        self.pub_realtime_waterfall = rospy.Publisher('waterfall_realtime_topic', ImageMetadata, queue_size=100)

        # inizializzare CvBridge
        self.bridge = CvBridge()

        # definire parametri Zeno
        self.altitude = None
        self.default_altitude = 4.0     # metri
        self.sonar_range = 25.0         # metri
        self.latest_nav = NavStatus()
        self.latest_sss_state = "UNKNOWN"

        # definire parametri waterfall
        self.raw_ping_buffer = []
        self.ping_buffer = []
        self.ping_index  = 0
        self.ping_metadata_buffer = []

        # definire parametri immagini
        self.ping_per_image = 150
        self.ping_overlap   = 75
        self.image_index    = 0
        self.last_image_ping_index = 0

        # definire parametri per plot echo intensity
        self.raw_echo_buffer = []
        self.processed_echo_buffer = []
        self.echo_plot_index = 0



        # creare le cartelle dei risultati
        pkg_path = rospkg.RosPack().get_path('sss_package')
        default_results_dir = os.path.join(pkg_path, 'results')
        results_dir = rospy.get_param('~results_folder', default_results_dir)

        self.raw_image_folder       = os.path.join(results_dir, "0_raw_images")
        self.waterfall_image_folder = os.path.join(results_dir, "1_waterfall_images")
        self.echo_plot_folder       = os.path.join(results_dir, "99_echo_intensity")

        folders_to_create = [
            self.raw_image_folder,
            self.waterfall_image_folder,
            self.echo_plot_folder
        ]

        for folder in folders_to_create:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # salvataggio plot al termine dell'esecuzione
        rospy.on_shutdown(self.save_echo_intensity_plots)

        # inizializzare subscriber
        rospy.Subscriber('/drivers/sss_sim', SideScanSonar, self.waterfall_callback)
        rospy.Subscriber('/drivers/altitude_sim', Altitude, self.altitude_callback)
        rospy.Subscriber('/nav_status', NavStatus, self.navstatus_callback)
        rospy.Subscriber('/phase1/SSS', String, self.sss_state_callback)
        rospy.Subscriber('/phase3/SSS', String, self.sss_state_callback)

      

# ________________________________________________________________________________________________________________________________


    # ========================================================
    # CALLBACK PER CREAZIONE WATERFALL
    # ========================================================
    def waterfall_callback(self, msg):
        # 1. estrarre dati raw
        left, right = self.read_intensity_vectors(msg)

        # 2. unire beam destro e sinistro in singolo profilo (raw ping)
        raw_ping = self.merge_beams(right, left)
        if raw_ping.size == 0:
            rospy.logwarn_throttle(5.0, "Ping SSS ignorato: beam sonar vuoti")
            return

        # 3. preprocessing
        right = self.apply_tvg(right)
        left  = self.apply_tvg(left)
        #right = self.apply_tvg_quadratic(right)
        #left  = self.apply_tvg_quadratic(left)
        #right = self.apply_tvg_logarithmic(right)
        #left  = self.apply_tvg_logarithmic(left)
        #right = self.apply_slant_range_correction(right)
        #left  = self.apply_slant_range_correction(left)
        
        # 4. unire beam destro e sinistro in singolo profilo (ping)
        ping = self.merge_beams(right, left)
        if ping.size == 0:
            rospy.logwarn_throttle(5.0, "Ping SSS ignorato: ping processato vuoto")
            return

        # 5. salvare intensita prima e dopo applicazione TVG per plot
        self.store_echo_intensity(raw_ping, ping)

        # 6. salvare ping corrente nei buffer
        self.store_ping(raw_ping, self.raw_ping_buffer)
        self.store_ping(ping, self.ping_buffer)
        self.ping_index += 1
        self.store_ping_metadata(msg)

        # 7. pubblicare waterfall realtime: usare tutti i ping disponibili nel buffer da 500
        waterfall = self.build_realtime_waterfall()
        self.publish_realtime_waterfall(waterfall)

        # 8. salvare/pubblicare immagini complete: usare finestre da 150 ping con overlap
        raw_image, image, image_metadata = self.build_image_window()
        if image is None:
            return

        self.save_raw_image(raw_image)
        self.save_image(image)
        self.publish_image(image, image_metadata)
        
        self.image_index += 1
        self.last_image_ping_index = self.ping_index
        


    # ========================================================
    # CALLBACK PER ALTITUDE DI ZENO
    # ========================================================
    def altitude_callback(self, msg):
        # estrarre altitudine
        self.altitude = msg.altitude


    # ========================================================
    # CALLBACK PER NAV_STATUS DI ZENO
    # ========================================================
    def navstatus_callback(self, msg):
        # estrarre stato di navigazione
        self.latest_nav = copy.deepcopy(msg)


    # ========================================================
    # CALLBACK PER STATO SSS / CURVA DI ZENO
    # ========================================================
    def sss_state_callback(self, msg):
        # OFF indica ingresso/uscita curva 
        self.latest_sss_state = msg.data
        

# ________________________________________________________________________________________________________________________________

    # ========================================================
    # UTILITIES
    # ========================================================
    def normalize_to_mono8(self, image):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)

    def fill_metadata_msg(self, metadata_msg, metadata_list):
        metadata_msg.ping_indices   = [metadata['ping_index'] for metadata in metadata_list]
        metadata_msg.ping_stamps    = [metadata['ping_stamp'] for metadata in metadata_list]
        metadata_msg.nav_statuses   = [metadata['nav_status'] for metadata in metadata_list]
        metadata_msg.altitudes      = [metadata['altitude'] for metadata in metadata_list]
        # stato SSS e curva sono allineati riga-per-riga ai ping della waterfall
        metadata_msg.sss_states     = [metadata['sss_state'] for metadata in metadata_list]
        metadata_msg.turning_statuses = [metadata['turning_status'] for metadata in metadata_list]
        return metadata_msg


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # LETTURA DATI
    # ========================================================
    def read_intensity_vectors(self, msg):
        # leggere vettori di intensita del beam destro e sinistro
        if isinstance(msg.left_beam.data, str):
            left = np.frombuffer(msg.left_beam.data, dtype=np.uint8).astype(np.float32)
        else:
            left = np.asarray(msg.left_beam.data, dtype=np.uint8).astype(np.float32)

        if isinstance(msg.right_beam.data, str):
            right = np.frombuffer(msg.right_beam.data, dtype=np.uint8).astype(np.float32)
        else:
            right = np.asarray(msg.right_beam.data, dtype=np.uint8).astype(np.float32)

        return left, right

    # ========================================================
    # PREPROCESSING DATI
    # ========================================================
    def apply_tvg(self, beam, strength=0.5):
        # applicare Time Variable Gain:
        # gain = 1 + strength * (range / range_max)^1/2
        beam = np.asarray(beam, dtype=np.float32)
        
        normalized_range = np.linspace(0.0, 1.0, beam.size).astype(np.float32)
        gain = 1.0 + strength * np.sqrt(normalized_range)
        corrected_beam = beam * gain
        return np.clip(corrected_beam, 0, 255).astype(np.float32)

    def apply_tvg_quadratic(self, beam, strength=0.5):
        # applicare Time Variable Gain quadratico:
        # gain = 1 + strength * (range / range_max)^2
        beam = np.asarray(beam, dtype=np.float32)
        
        normalized_range = np.linspace(0.0, 1.0, beam.size).astype(np.float32)
        gain = 1.0 + strength * (normalized_range ** 2)
        corrected_beam = beam * gain
        return np.clip(corrected_beam, 0, 255).astype(np.float32)

    def apply_tvg_logarithmic(self, beam, spreading_gain_db=20.0, absorption_db_per_sample=0.0):
        # applicare Time Variable Gain logaritmico:
        # gain_dB = spreading_gain_db * log10(range) + 2 * alpha * range
        beam = np.asarray(beam, dtype=np.float32)
        
        sample_range = np.arange(1, beam.size + 1, dtype=np.float32)
        gain_db = (spreading_gain_db * np.log10(sample_range) + 2.0 * absorption_db_per_sample * sample_range)
        # normalizzare rispetto al primo bin (0 dB, nessun amplificazione)
        gain_db = gain_db - gain_db[0]
        # convertire da dB a scala lineare
        gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)

        corrected_beam = beam * gain_linear
        return np.clip(corrected_beam, 0, 255).astype(np.float32)

    def apply_slant_range_correction(self, beam):
        # correggere la distorsione geometrica da slant range a ground range
        beam = np.asarray(beam, dtype=np.float32)

        sonar_range    = float(self.sonar_range)
        sonar_altitude = float(self.altitude)

        # costruire una griglia uniforme in ground range, cioe' distanza orizzontale sul fondo
        n_samples = beam.size
        max_ground_range = np.sqrt(sonar_range ** 2 - sonar_altitude ** 2)
        ground_range = np.linspace(0.0, max_ground_range, n_samples).astype(np.float32)

        # convertire ogni punto di ground range nel corrispondente slant range
        slant_range = np.sqrt(ground_range ** 2 + sonar_altitude ** 2)

        # trasformare lo slant range in indice del beam originale e ricampionare l'intensita'
        source_index = (slant_range / sonar_range) * (n_samples - 1)
        sample_index = np.arange(n_samples, dtype=np.float32)
        corrected_beam = np.interp(source_index, sample_index, beam)

        return corrected_beam.astype(np.float32)

    def merge_beams(self, right, left):
        # unire beam destro e sinistro in singolo profilo (ping)
        left = left[::-1]
        ping = np.concatenate([left, right])
        return ping


    # ========================================================
    # COSTRUZIONE WATERFALL E IMMAGINI
    # ========================================================
    def store_ping(self, ping, ping_buffer):
        # aggiungi il nuovo ping in cima alla lista per simulare l'avanzamento
        ping_buffer.insert(0, ping)
        # mantenere solo gli ultimi 500 ping per effetto waterfall
        if len(ping_buffer) > self.WATERFALL_BUFFER_SIZE:
            ping_buffer.pop()

    def store_ping_metadata(self, sonar_msg):
        metadata = {
            'ping_index': int(self.ping_index),
            'ping_stamp': sonar_msg.header.stamp,
            'nav_status': copy.deepcopy(self.latest_nav),
            'altitude': float(self.altitude) if self.altitude is not None else self.default_altitude,
            'sss_state': self.latest_sss_state,
            'turning_status': self.latest_sss_state == "OFF"    # OFF = entrata/uscita da curva
        }
        # aggiungi i nuovi metadata in cima alla lista
        self.ping_metadata_buffer.insert(0, metadata)
        # mantenere solo gli ultimi 500 ping
        if len(self.ping_metadata_buffer) > self.WATERFALL_BUFFER_SIZE:
            self.ping_metadata_buffer.pop()

    def build_realtime_waterfall(self):
        # realtime: immagine aggiornata a ogni ping con massimo 500 righe
        if len(self.ping_buffer) == 0:
            return None
        waterfall = np.vstack(self.ping_buffer).astype(np.float32)
        if waterfall.size == 0 or waterfall.shape[0] == 0 or waterfall.shape[1] == 0:
            return None
        return self.normalize_to_mono8(waterfall)

    def build_image_window(self):
        # creare una nuova immagine solo quando sono disponibili 150 ping
        if len(self.ping_buffer) < self.ping_per_image:
            return None, None, None

        # per realizzare overlap: dopo la prima immagine, aspettare image_step=75 nuovi ping
        image_step = self.ping_per_image - self.ping_overlap            # 150 - 75
        new_pings = self.ping_index - self.last_image_ping_index
        if self.last_image_ping_index > 0 and new_pings < image_step:
            return None, None, None

        # usare le ultime 150 righe per generare immagine
        raw_image = self.normalize_to_mono8(np.vstack(self.raw_ping_buffer[:self.ping_per_image]).astype(np.float32))
        image     = self.normalize_to_mono8(np.vstack(self.ping_buffer[:self.ping_per_image]).astype(np.float32))

        # metadata allineati riga-per-riga con l'immagine processata
        image_metadata = self.ping_metadata_buffer[:self.ping_per_image]
        return raw_image, image, image_metadata

    def publish_realtime_waterfall(self, waterfall):
        if waterfall is None or waterfall.size == 0 or waterfall.shape[0] == 0 or waterfall.shape[1] == 0:
            rospy.logwarn_throttle(5.0, "Waterfall realtime non pubblicata: immagine vuota")
            return

        # convertire la matrice OpenCV mono8 in un messaggio ROS Image
        ros_img = self.bridge.cv2_to_imgmsg(waterfall, encoding='mono8')

        # prendere solo i metadata corrispondenti alle righe presenti nella waterfall realtime
        live_metadata = self.ping_metadata_buffer[:waterfall.shape[0]]

        # inserire immagine e metadata nello stesso messaggio usato dal resto della pipeline
        metadata_msg = ImageMetadata()
        metadata_msg.image = ros_img
        self.fill_metadata_msg(metadata_msg, live_metadata)

        # pubblicare sul topic realtime per la visualizzazione continua
        self.pub_realtime_waterfall.publish(metadata_msg)

    # ========================================================
    # PUBBLICAZIONE IMMAGINE
    # ========================================================
    def save_image(self, image):
        # se non sono stati ancora accumulati abbastanza ping, non salvare immagine
        if image is None:
            return
        # salvare immagine .png
        filename = os.path.join(self.waterfall_image_folder, "sonar_{:03d}.png".format(self.image_index))
        saved = cv2.imwrite(filename, image)


    def save_raw_image(self, raw_image):
        # se non sono stati ancora accumulati abbastanza ping, non salvare immagine
        if raw_image is None:
            return
        # salvare immagine .png
        filename = os.path.join(self.raw_image_folder, "raw_sonar_{:03d}.png".format(self.image_index))
        saved = cv2.imwrite(filename, raw_image)
        

    def publish_image(self, image, image_metadata):
        # se non sono stati ancora accumulati abbastanza ping, non pubblicare immagine
        if image is None or image_metadata is None:
            return

        # convertire in formato ROS Image
        ros_img = self.bridge.cv2_to_imgmsg(image, encoding='mono8')
        ros_img.header.seq = self.image_index
        ros_img.header.stamp = image_metadata[0]['ping_stamp']
        
        metadata_msg = ImageMetadata()
        metadata_msg.header = ros_img.header
        metadata_msg.image = ros_img
        self.fill_metadata_msg(metadata_msg, image_metadata)

        # pubblicare immagine e metadata sul topic waterfall_image_topic
        self.pub_img.publish(metadata_msg)
        print("[SSS] waterfall_image_topic: pubblicata immagine {:03d} con {} ping".format(self.image_index, len(metadata_msg.ping_indices)))

   
# ________________________________________________________________________________________________________________________________

    # ========================================================
    # PLOT INTENSITA
    # ========================================================
    def store_echo_intensity(self, raw_ping, processed_ping):
        self.raw_echo_buffer.append(np.asarray(raw_ping, dtype=np.float32))
        self.processed_echo_buffer.append(np.asarray(processed_ping, dtype=np.float32))

    def plot_echo_intensity_3d(self, waterfall, title, filename):
        # convertire la lista di ping in matrice: x = campioni, y = ping, z = intensita'
        z_intensity = np.asarray(waterfall, dtype=np.float32)
        x_axis = np.arange(z_intensity.shape[1])    # crea vettore di indici da 0 a numero di colonne dell’immagine
        y_axis = np.arange(z_intensity.shape[0])    # crea vettore di indici da 0 a numero di righe   dell’immagine
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)

        # creare una superficie 3D dell'intensita' sonar
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(x_grid, y_grid, z_intensity, cmap='viridis', linewidth=0, antialiased=False)

        # evidenziare il nadir, cioe' il centro tra beam sinistro e destro
        nadir_idx = z_intensity.shape[1] // 2
        ax.plot([nadir_idx] * z_intensity.shape[0], y_axis,
            np.max(z_intensity) * np.ones(z_intensity.shape[0]), color='red', linewidth=2)
        
        ax.set_xlabel('Across-track bins')
        ax.set_ylabel('Ping number')
        ax.set_zlabel('Echo intensity')
        ax.set_title(title)

        # salvare il plot
        fig.colorbar(surface, shrink=0.5, aspect=10)
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return filename

    def plot_echo_intensity_2d(self, raw_waterfall, processed_waterfall, filename):
        # convertire i buffer in matrici: una prima e una dopo il preprocessing
        raw_matrix       = np.asarray(raw_waterfall, dtype=np.float32)
        processed_matrix = np.asarray(processed_waterfall, dtype=np.float32)

        # scegliere un ping casuale da confrontare
        ping_number = np.random.randint(0, raw_matrix.shape[0])

        # plottare il profilo di intensita' dello stesso ping prima e dopo la TVG
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(np.arange(raw_matrix.shape[1]), raw_matrix[ping_number], color='gray', linewidth=1.2, label='Before TVG')
        ax.plot(np.arange(processed_matrix.shape[1]), processed_matrix[ping_number], color='blue', linewidth=1.2, label='After TVG')
        
        # assi e linea verticale sul nadir
        ax.axvline(raw_matrix.shape[1] // 2, color='red', linewidth=1.5, label='Nadir')
        ax.set_xlabel('Across-track bins')
        ax.set_ylabel('Echo intensity')
        ax.set_title('2D Echo Intensity - ping {}'.format(ping_number))
        ax.grid(True, alpha=0.3)
        ax.legend()

        # salvare il plot
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return filename

    def save_echo_intensity_plots(self):
        # se non sono stati raccolti ping, non generare i plot finali
        if len(self.raw_echo_buffer) == 0 or len(self.processed_echo_buffer) == 0:
            return

        print("[SSS] Saving final echo intensity plots")
    
        raw_3d_filename        = os.path.join(self.echo_plot_folder, "echo_intensity_raw_3d_{:03d}.png".format(self.echo_plot_index))
        processed_3d_filename  = os.path.join(self.echo_plot_folder, "echo_intensity_processed_3d_{:03d}.png".format(self.echo_plot_index))
        comparison_2d_filename = os.path.join(self.echo_plot_folder, "echo_intensity_comparison_2d_{:03d}.png".format(self.echo_plot_index))
        
        self.plot_echo_intensity_3d(self.raw_echo_buffer, "Raw - 3D Side Scan Sonar Intensity", raw_3d_filename)
        self.plot_echo_intensity_3d(self.processed_echo_buffer, "After TVG - 3D Side Scan Sonar Intensity", processed_3d_filename)
        self.plot_echo_intensity_2d(self.raw_echo_buffer, self.processed_echo_buffer, comparison_2d_filename)

        self.echo_plot_index += 1




# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # inizializzare nodo ROS
    rospy.init_node('waterfall_creator_node', anonymous=True)
    # istanziare WaterfallCreatorNode
    node = WaterfallCreatorNode()
    # spin ROS
    rospy.spin()
