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
import math
from sss_package.msg import ImageMetadata
from cv_bridge import CvBridge, CvBridgeError


# ============================================================
# CLASSE PRINCIPALE NODO LIST CREATOR
# ============================================================
class ObjectClassificationNode:
    MAX_CLASSIFICATION_SCORE = 7.0

    def __init__(self):

	print("[SSS] object_classification_node.py is active\n")

        # inizializzare subscriber/publisher
        rospy.Subscriber('/waterfall_image_topic', ImageMetadata, self.object_classification_callback)
        self.pub_classified_objects = rospy.Publisher('classified_objects_topic', ImageMetadata, queue_size=200)

        # inizializzare CvBridge
        self.bridge = CvBridge()

        # definire parametri filtri
        self.log_filter_scale        = float(rospy.get_param('~log_filter_scale', 255.0))
        self.median_kernel_size      = int(rospy.get_param('~median_kernel_size', 3))
        self.gaussian_kernel_size    = int(rospy.get_param('~gaussian_kernel_size', 5))
        self.gaussian_sigma          = float(rospy.get_param('~gaussian_sigma', 0.0))
        self.clahe_clip_limit        = float(rospy.get_param('~clahe_clip_limit', 2.0))
        self.clahe_tile_grid_size    = int(rospy.get_param('~clahe_tile_grid_size', 8))
        
        # definire parametri morfologici
        self.bright_morph_first_open_kernel_size  = int(rospy.get_param('~bright_morph_first_open_kernel_size', 3))	 # pulizia iniziale leggera
        self.bright_morph_first_close_kernel_size = int(rospy.get_param('~bright_morph_first_close_kernel_size', 9)) # chiusura piccoli buchi all'interno degli oggetti
        self.bright_morph_final_open_kernel_size  = int(rospy.get_param('~bright_morph_final_open_kernel_size', 5))	 # pulizia finale piu' forte
        self.bright_morph_final_close_kernel_size = int(rospy.get_param('~bright_morph_final_close_kernel_size', 9)) # riconnessione frammenti dopo la pulizia finale
        self.bright_morph_iterations        	  = int(rospy.get_param('~bright_morph_iterations', 1))

        self.dark_morph_first_close_kernel_width  = int(rospy.get_param('~dark_morph_first_close_kernel_width', 5))	 # connette frammenti prima della pulizia
        self.dark_morph_first_close_kernel_height = int(rospy.get_param('~dark_morph_first_close_kernel_height', 3))
        self.dark_morph_first_open_kernel_size    = int(rospy.get_param('~dark_morph_first_open_kernel_size', 3))	 # rimozione piccoli blob isolati dopo aver riconnesso le ombre
        self.dark_morph_iterations   		      = int(rospy.get_param('~dark_morph_iterations', 1))

        # definire parametri saliency
        self.cfar_train_cells      = rospy.get_param('~cfar_train_cells', 18)
        self.cfar_guard_cells      = rospy.get_param('~cfar_guard_cells', 10)
        self.cfar_rank_percentile  = rospy.get_param('~cfar_rank_percentile', 80.0)
        self.cfar_threshold_scale  = rospy.get_param('~cfar_threshold_scale', 1.00)
        self.cfar_threshold_offset = rospy.get_param('~cfar_threshold_offset', 2.0)

        # definire parametri binarizzazione
        self.saliency_percentile = rospy.get_param('~saliency_percentile', 60.0)
        self.bright_percentile   = rospy.get_param('~bright_percentile', 85.0)
        self.dark_percentile     = rospy.get_param('~dark_percentile', 25.0)

        # definire parametri detection e classification
        self.sonar_range_m                 = float(rospy.get_param('~sonar_range_m', 25.0))
        self.sonar_bins_per_side           = int(rospy.get_param('~sonar_bins_per_side', 1000))
        self.sss_frequency_hz              = float(rospy.get_param('~sss_frequency_hz', 5.0))
        self.alongtrack_beam_angle_deg     = float(rospy.get_param('~alongtrack_beam_angle_deg', 0.3))
        self.alongtrack_beam_angle_rad     = math.radians(self.alongtrack_beam_angle_deg)

        self.min_object_area_px            = int(rospy.get_param('~min_object_area_px', 30))				# 20 << minima area dell oggetto stimata
        self.min_shadow_area_px            = int(rospy.get_param('~min_shadow_area_px', 30))				# 20 << minima area dell ombra stimata
        self.max_shadow_alongtrack_gap_m   = float(rospy.get_param('~max_shadow_alongtrack_gap_m', 2.5))
        self.max_shadow_acrosstrack_gap_m  = float(rospy.get_param('~max_shadow_acrosstrack_gap_m', 8.0))
        self.min_classification_confidence = float(rospy.get_param('~min_classification_confidence', 0.35))
        self.tie_buoy_max_area_m2          = float(rospy.get_param('~tie_buoy_max_area_m2', 0.45))
        self.tie_buoy_max_dimension_m      = float(rospy.get_param('~tie_buoy_max_dimension_m', 1.6))
        self.tie_buoy_max_aspect_ratio     = float(rospy.get_param('~tie_buoy_max_aspect_ratio', 3.2))
        self.tie_tube_min_area_m2          = float(rospy.get_param('~tie_tube_min_area_m2', 0.55))
        self.tie_tube_min_dimension_m      = float(rospy.get_param('~tie_tube_min_dimension_m', 2.0))
        self.tie_tube_min_aspect_ratio     = float(rospy.get_param('~tie_tube_min_aspect_ratio', 3.2))
        self.tube_shadow_min_area_m2       = float(rospy.get_param('~tube_shadow_min_area_m2', 4.0))
        self.buoy_far_shadow_gap_m         = float(rospy.get_param('~buoy_far_shadow_gap_m', 2.0))


        # creazione cartelle per risultati 
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('sss_package') 

        default_results_dir = os.path.join(pkg_path, 'results')
        results_dir = rospy.get_param('~results_folder', default_results_dir)

        self.filtered_image_folder = os.path.join(results_dir, "2_filtered_images")
        self.saliency_image_folder = os.path.join(results_dir, "3_saliency_images")
        self.bright_saliency_image_folder = os.path.join(self.saliency_image_folder, "bright_maps")
        self.dark_saliency_image_folder = os.path.join(self.saliency_image_folder, "dark_maps")
        self.saliency_binary_image_folder = os.path.join(results_dir, "4_saliency_binary_images")
        self.bright_saliency_binary_image_folder = os.path.join(self.saliency_binary_image_folder, "bright_maps")
        self.dark_saliency_binary_image_folder = os.path.join(self.saliency_binary_image_folder, "dark_maps")

        self.binary_maps_image_folder = os.path.join(results_dir, "5_binary_maps_images")
        self.bright_map_folder = os.path.join(self.binary_maps_image_folder, "bright_maps")
        self.dark_map_folder = os.path.join(self.binary_maps_image_folder, "dark_maps")

        self.binary_and_salient_maps_image_folder = os.path.join(results_dir, "6_binary_and_salient_maps_images")
        self.bright_and_salient_map_folder = os.path.join(self.binary_and_salient_maps_image_folder, "bright_maps")
        self.dark_and_salient_map_folder = os.path.join(self.binary_and_salient_maps_image_folder, "dark_maps")

        self.morph_images_folder = os.path.join(results_dir, "7_morph_images")
        self.bright_morph_image_folder = os.path.join(self.morph_images_folder, "bright_maps")
        self.dark_morph_image_folder = os.path.join(self.morph_images_folder, "dark_maps")

        self.classification_image_folder = os.path.join(results_dir, "8_classification_images")
        self.classification_text_folder = os.path.join(results_dir, "8_classification_texts")

        folders_to_create = [
            self.filtered_image_folder,
            self.saliency_image_folder,
            self.bright_saliency_image_folder,
            self.dark_saliency_image_folder,
            self.saliency_binary_image_folder,
            self.bright_saliency_binary_image_folder,
            self.dark_saliency_binary_image_folder,
            self.binary_maps_image_folder,
            self.bright_map_folder,
            self.dark_map_folder,
            self.binary_and_salient_maps_image_folder,
            self.bright_and_salient_map_folder,
            self.dark_and_salient_map_folder,
            self.morph_images_folder,
            self.bright_morph_image_folder,
            self.dark_morph_image_folder,
            self.classification_image_folder,
            self.classification_text_folder
        ]

        for folder in folders_to_create:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.image_index = 0

# ________________________________________________________________________________________________________________________________



    # ========================================================
    # CALLBACK PER OBJECT CLASSIFICATION
    # ========================================================
    def object_classification_callback(self, msg):
        # 1. convertire immagine
        image = self.ros_image_to_cv2(msg.image)

        # 2. filtering pipeline	
        #image = self.apply_log_filter(image)
        #image = self.apply_gaussian_filter(image)
        #image = self.apply_median_filter(image)
        #image = self.apply_clahe_filter(image)

        # 3. saliency
        # opzione 1) OpenCV spectral residual saliency
        #saliency_map = self.compute_spectral_residual_saliency(image)

        # opzione 2) OS-CFAR 1D saliency separata per oggetti luminosi e ombre scure
        bright_saliency_map, dark_saliency_map = self.compute_os_cfar_1d_saliency_maps(image)   # dark_saliency scartato per eccessiva perdita informazioni

        # 4. bright/dark binary maps e binary saliency maps
        bright_map, dark_map       = self.create_binary_maps(image)
        bright_saliency_binary_map = self.create_saliency_binary_map(bright_saliency_map)
        #dark_saliency_binary_map   = self.create_saliency_binary_map(dark_saliency_map)            

        # 5. bright/dark binary AND salient maps
        bright_and_salient_map = cv2.bitwise_and(bright_map, bright_saliency_binary_map)
        #dark_and_salient_map   = cv2.bitwise_and(dark_map, dark_saliency_binary_map)              

        # 6. morphological cleaning
        bright_and_salient_morph_map = self.morphological_cleaning_bright(bright_and_salient_map)
        dark_morph_map               = self.morphological_cleaning_dark(dark_map)
        #dark_and_salient_morph_map   = self.morphological_cleaning_dark(dark_and_salient_map)


        # 7. risoluzione across-track
        self.meters_per_pixel_x = self.sonar_range_m / float(self.sonar_bins_per_side)            # Ry = 25 m / 1000 bin

        # 8. risoluzione along-track
        nav_status = msg.nav_statuses[int(image.shape[0] / 2)]
        self.auv_velocity_mps = math.sqrt((nav_status.ned_speed.x * nav_status.ned_speed.x) + (nav_status.ned_speed.y * nav_status.ned_speed.y))
        self.alongtrack_sampling_m       = self.auv_velocity_mps / self.sss_frequency_hz          # s = v / fp
        self.alongtrack_beam_footprint_m = self.sonar_range_m * self.alongtrack_beam_angle_rad    # Rx = R * ampiezza angolare along-track
        self.meters_per_pixel_y = max(self.alongtrack_sampling_m, self.alongtrack_beam_footprint_m)

        # 9. classificazione oggetti usando oggetto luminoso + ombra scura
        classifications = self.detect_and_classify_objects(bright_and_salient_morph_map, dark_morph_map)
        #classifications = self.detect_and_classify_objects(bright_and_salient_morph_map, dark_and_salient_morph_map)


        # 10. salvataggio risultati
        image_index = self.image_index
        self.save_debug_image(self.filtered_image_folder, "filtered", image, image_index)
        self.save_debug_image(self.bright_saliency_image_folder, "bright_saliency", bright_saliency_map, image_index)
        self.save_debug_image(self.dark_saliency_image_folder, "dark_saliency", dark_saliency_map, image_index)
        self.save_debug_image(self.bright_saliency_binary_image_folder, "bright_saliency_binary", bright_saliency_binary_map, image_index)
        #self.save_debug_image(self.dark_saliency_binary_image_folder, "dark_saliency_binary", dark_saliency_binary_map, image_index)
        self.save_debug_image(self.bright_map_folder, "bright", bright_map, image_index)
        self.save_debug_image(self.dark_map_folder, "dark", dark_map, image_index)
        self.save_debug_image(self.bright_and_salient_map_folder, "bright_and_salient", bright_and_salient_map, image_index)
        #self.save_debug_image(self.dark_and_salient_map_folder, "dark_and_salient", dark_and_salient_map, image_index)
        self.save_debug_image(self.bright_morph_image_folder, "bright_and_salient_morph", bright_and_salient_morph_map, image_index)
        #self.save_debug_image(self.dark_morph_image_folder, "dark_and_salient_morph", dark_and_salient_morph_map, image_index)
        self.save_debug_image(self.dark_morph_image_folder, "dark_morph", dark_morph_map, image_index)
        self.add_turning_status_to_classifications(msg, classifications)
        self.save_classification_image(image, classifications, image_index)
        self.save_classification_text(classifications, image_index)

        classified_msg = self.build_classified_objects_message(msg, classifications)
        self.pub_classified_objects.publish(classified_msg)
        print("[SSS] classified_objects_topic: pubblicata immagine {:03d} con {} oggetti classificati".format(image_index, len(classifications)))
        if len(classifications) == 0:
            print("[SSS] object_classification_node: nessun oggetto classificato nell'immagine {:03d}".format(image_index))

        self.image_index += 1
        return classifications

    def build_classified_objects_message(self, source_msg, classifications):
        detected_msg = ImageMetadata()
        detected_msg.header = source_msg.header
        detected_msg.image = source_msg.image
        detected_msg.ping_indices = source_msg.ping_indices
        detected_msg.ping_stamps = source_msg.ping_stamps
        detected_msg.nav_statuses = source_msg.nav_statuses
        detected_msg.altitudes = source_msg.altitudes
        detected_msg.sss_states = source_msg.sss_states
        detected_msg.turning_statuses = source_msg.turning_statuses

        detected_msg.object_classes = []
        detected_msg.object_confidences = []
        detected_msg.object_centroid_x_px = []
        detected_msg.object_centroid_y_px = []
        detected_msg.object_bbox_x_px = []
        detected_msg.object_bbox_y_px = []
        detected_msg.object_bbox_width_px = []
        detected_msg.object_bbox_height_px = []

        for detection in classifications:
            obj = detection['object']
            bbox = obj['bbox_px']
            centroid = obj['centroid_px']
            detected_msg.object_classes.append(detection['classification'])
            detected_msg.object_confidences.append(float(detection['confidence']))
            detected_msg.object_centroid_x_px.append(float(centroid[0]))
            detected_msg.object_centroid_y_px.append(float(centroid[1]))
            detected_msg.object_bbox_x_px.append(int(bbox[0]))
            detected_msg.object_bbox_y_px.append(int(bbox[1]))
            detected_msg.object_bbox_width_px.append(int(bbox[2]))
            detected_msg.object_bbox_height_px.append(int(bbox[3]))

        return detected_msg

# ________________________________________________________________________________________________________________________________

    # ========================================================
    # UTILITIES
    # ========================================================

    def add_turning_status_to_classifications(self, source_msg, classifications):
        for detection in classifications:
            # la coordinata y del centroide identifica la riga/ping della detection
            row = int(round(detection['object']['centroid_px'][1]))
            row = max(0, min(row, len(source_msg.ping_indices) - 1))

            detection['sss_state'] = source_msg.sss_states[row]
            detection['isturning'] = bool(source_msg.turning_statuses[row])

    def make_positive_odd(self, value):
        value = int(value)
        if value < 1:
            value = 1
        if value % 2 == 0:
            value += 1
        return value


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # PREPROCESSING IMMAGINE
    # ========================================================
    def ros_image_to_cv2(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as exc:
            print("Errore conversione ROS Image -> OpenCV: {}".format(exc))
            return None
        return np.asarray(image, dtype=np.uint8)

    def normalize_uint8(self, image):
        image = np.asarray(image)
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32)
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    def apply_log_filter(self, image):
        image = self.normalize_uint8(image).astype(np.float32)
        log_image = np.log1p(image)
        max_log   = float(np.max(log_image))
        if max_log <= 0.0:
            return np.zeros_like(image, dtype=np.uint8)
        return np.clip((log_image / max_log) * self.log_filter_scale, 0, 255).astype(np.uint8)

    def apply_median_filter(self, image):
        kernel_size = self.make_positive_odd(self.median_kernel_size)
        return cv2.medianBlur(self.normalize_uint8(image), kernel_size)

    def apply_gaussian_filter(self, image):
        kernel_size = self.make_positive_odd(self.gaussian_kernel_size)
        return cv2.GaussianBlur(self.normalize_uint8(image), (kernel_size, kernel_size), self.gaussian_sigma)

    def apply_clahe_filter(self, image):
        tile_grid_size = self.make_positive_odd(self.clahe_tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit = self.clahe_clip_limit, tileGridSize = (tile_grid_size, tile_grid_size))
        return clahe.apply(self.normalize_uint8(image))



    # ========================================================
    # SALIENCY MAP
    # ========================================================
    def compute_spectral_residual_saliency(self, image):
	    # normalizzare immagine in uint8 [0, 255]
        image = self.normalize_uint8(image)

	    # creare mappa di salienza
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(image)
        if not success:
            return None
        # convertire da float [0,1] a uint8 [0,255]
        saliency_map = (saliency_map * 255).astype(np.uint8)
        return saliency_map


    def compute_os_cfar_1d_saliency_maps(self, image):
	    # normalizzare immagine in uint8 [0, 255]
        image = self.normalize_uint8(image)

        # calcolare la risposta CFAR sugli oggetti luminosi 
        bright_response = self.compute_os_cfar_1d_response(image)

        # invertire l'immagine: le ombre diventano valori alti rilevabili
        inverted_image = 255 - image

        # calcolare la risposta CFAR sulle ombre diventate luminose
        dark_response = self.compute_os_cfar_1d_response(inverted_image)

        return bright_response, dark_response

    def compute_os_cfar_1d_response(self, image):
        # portare l'immagine in float32 per calcolare soglie e differenze
        image_float = self.normalize_uint8(image).astype(np.float32)
        height, width = image_float.shape

        # parametri della finestra CFAR: train|guard|CUT|guard|train
        train = max(1, int(self.cfar_train_cells))
        guard = max(0, int(self.cfar_guard_cells))
        radius = train + guard

        # rank = indice corrispondente al percentile
        rank_percentile = float(self.cfar_rank_percentile)
        rank_percentile = max(0.0, min(100.0, rank_percentile))
        # convertire il percentile in un indice di un array per usarlo in partition()
        rank = int(round((rank_percentile / 100.0) * ((2 * train) - 1)))
        # il massimo indice valido e' (2 * train) - 1
        rank = max(0, min((2 * train) - 1, rank))

        # parametri della soglia adattiva
        threshold_scale  = float(self.cfar_threshold_scale)
        threshold_offset = float(self.cfar_threshold_offset)
        across_response = np.zeros_like(image_float, dtype=np.float32)
        along_response = np.zeros_like(image_float, dtype=np.float32)

        # applicare CFAR riga per riga (across-track)
        for row in range(height):
            data = image_float[row, :]
            padded = np.pad(data, (radius, radius), mode='edge')

            for col in range(width):		# col = CUT
                center = col + radius

                # celle training a sinistra e destra, escludendo le celle guard vicino al CUT
                left_train  = padded[center - guard - train:center - guard]
                right_train = padded[center + guard + 1:center + guard + train + 1]
                training_cells = np.concatenate((left_train, right_train))

                # soglia locale: campione ordinato scelto dal rank, scalato e traslato
                noise_estimate = np.partition(training_cells, rank)[rank]
                threshold = noise_estimate * threshold_scale + threshold_offset

                # se il pixel e' maggiore della soglia, salvare quanto piu' forte e'
                if data[col] > threshold:
                    across_response[row, col] = data[col] - threshold

        # applicare CFAR colonna per colonna (along-track)
        for col in range(width):
            data = image_float[:, col]
            padded = np.pad(data, (radius, radius), mode='edge')
        
            for row in range(height):		# row = CUT
                center = row + radius
        
                # celle training sopra e sotto, escludendo le celle guard vicino al CUT
                left_train  = padded[center - guard - train:center - guard]
                right_train = padded[center + guard + 1:center + guard + train + 1]
                training_cells = np.concatenate((left_train, right_train))
        
                # soglia locale: campione ordinato scelto dal rank, scalato e traslato
                noise_estimate = np.partition(training_cells, rank)[rank]
                threshold = noise_estimate * threshold_scale + threshold_offset
        
                # controllare se l'intensita' del CUT e' maggiore della soglia
                # se il pixel e' maggiore della soglia, salvare quanto piu' forte e'
                if data[row] > threshold:
                    along_response[row, col] = data[row] - threshold

        # combinare le due direzioni senza richiedere che entrambe rilevino lo stesso pixel
        response = np.maximum(across_response, along_response)

        # se nessun pixel e stato rilevato, restituire una mappa nera
        max_response = float(np.max(response))
        if max_response <= 0.0:
            return np.zeros_like(image_float, dtype=np.uint8)

        # normalizzare la risposta in [0, 255] per salvarla/combinarla come immagine uint8
        return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # ========================================================
    # BINARY MAPS
    # ========================================================
    def create_binary_maps(self, image):
	    # normalizzare immagine in uint8 [0, 255]
        image = self.normalize_uint8(image)
        # calcolare una soglia dinamica
        bright_threshold = np.percentile(image, self.bright_percentile)
        dark_threshold   = np.percentile(image, self.dark_percentile)
        # creare maschera binaria: 1 dove la salienza e' sopra la soglia, 0 altrove
        bright_map = (image >= bright_threshold).astype(np.uint8) * 255
        dark_map   = (image <= dark_threshold).astype(np.uint8) * 255

        return bright_map, dark_map


    def create_saliency_binary_map(self, saliency_map):
	    # normalizzare immagine in uint8 [0, 255]
        saliency_map = self.normalize_uint8(saliency_map)
        # considerare solo i pixel con salienza non nulla
        nonzero_saliency = saliency_map[saliency_map > 0]
        if nonzero_saliency.size == 0:
            return np.zeros_like(saliency_map, dtype=np.uint8)
        # calcolare una soglia dinamica solo sui pixel salienti non nulli
        salient_threshold = np.percentile(nonzero_saliency, self.saliency_percentile)
        # creare una maschera binaria: 1 dove la salienza e' sopra la soglia, 0 altrove
        saliency_binary = (saliency_map > salient_threshold).astype(np.uint8) * 255

        return saliency_binary


    def morphological_cleaning_bright(self, binary_map):
        # open-close-open-close: pulire rumore, riconnettere oggetti, ripulire e chiudere frammenti rimasti
        first_open_kernel_size  = self.make_positive_odd(self.bright_morph_first_open_kernel_size)
        first_close_kernel_size = self.make_positive_odd(self.bright_morph_first_close_kernel_size)
        final_open_kernel_size  = self.make_positive_odd(self.bright_morph_final_open_kernel_size)
        final_close_kernel_size = self.make_positive_odd(self.bright_morph_final_close_kernel_size)

        # generare kernel
        first_open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (first_open_kernel_size, first_open_kernel_size))
        first_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (first_close_kernel_size, first_close_kernel_size))
        final_open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_open_kernel_size, final_open_kernel_size))
        final_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_close_kernel_size, final_close_kernel_size))
        # 1) opening: eliminare piccoli blob bianchi (rumore) e separare regioni sottilmente collegate
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, first_open_kernel, iterations=1)
        # 2) closing: chiudere piccoli buchi neri e collegare regioni vicine
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, first_close_kernel, iterations=4)
        # 3) opening finale: rimuovere piccoli blob bianchi creati o rimasti dopo il closing
        #binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, final_open_kernel, iterations=1)
        # 4) closing finale: riconnettere frammenti utili rimasti dopo l opening finale
        #binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, final_close_kernel, iterations=1)
        return binary_map


    def morphological_cleaning_dark(self, binary_map):
        # close-open: riconnettere ombre e pulire rumore
        first_close_kernel_width  = self.make_positive_odd(self.dark_morph_first_close_kernel_width)
        first_close_kernel_height = self.make_positive_odd(self.dark_morph_first_close_kernel_height)
        first_open_kernel_size    = self.make_positive_odd(self.dark_morph_first_open_kernel_size)

        # generare kernel
        first_open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (first_open_kernel_size, first_open_kernel_size))
        first_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (first_close_kernel_width, first_close_kernel_height))
        # 1) closing: chiudere piccoli buchi neri e collegare regioni vicine
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, first_close_kernel, iterations=1)
        # 2) opening: rimuovere piccoli blob bianchi creati o rimasti dopo il closing
        binary_map  = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, first_open_kernel, iterations=1)
        return binary_map



    # ========================================================
    # CLASSIFICAZIONE OGGETTI
    # ========================================================
    def extract_blobs(self, binary_map, min_area_px, blob_type):
	    # normalizzare immagine in uint8 [0, 255]
        binary_map = self.normalize_uint8(binary_map)
        _, binary_map = cv2.threshold(binary_map, 0, 255, cv2.THRESH_BINARY)

        # etichettare tutte le regioni bianche connesse nella mappa binaria
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)	# 8 = pixel connesso anche in diagonale

        blobs = []

        # saltare label 0 perche' rappresenta lo sfondo nero
        for label in range(1, num_labels):
            # scartare componenti troppo piccole per essere oggetti/ombre
            area = int(stats[label, cv2.CC_STAT_AREA])							# CC_STAT_AREA = area in pixel, restituita da connectedComponentsWithStats()
            if area < min_area_px:                                              # filtro veloce che conta quanti pixel compongono area
                continue

            # creare una maschera isolata per la componente corrente
            component_mask = np.zeros_like(binary_map, dtype=np.uint8)			# matrice di zeri
            component_mask[labels == label] = 255								# inserire 255 nella matrice di zeri in corrispondenza del blob

            # estrarre il (primo) contorno della componente per calcolare geometria, area e rettangolo minimo
            contours = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # C_A_S = considera solo i vertici
            contours = contours[0] if len(contours) == 2 else contours[1]
            if not contours:
                continue
            contour = contours[0]

            # area del contorno in pixel quadrati
            contour_area_px = float(cv2.contourArea(contour))                   # misura area in pixel all'interno del contorno
            if contour_area_px <= 0.0:
                continue

            # rettangolo orientato in pixel: centro, lati e angolo inclinazione
            pixel_rect = cv2.minAreaRect(contour)
            (cx_px, cy_px), (w_px, h_px), angle_px = pixel_rect
            # salvare dimensioni massime e minime del rettangolo in pixel
            max_dimension_px = float(max(w_px, h_px))
            min_dimension_px = float(min(w_px, h_px))
            if min_dimension_px <= 0.0:
                continue

            # con bbox inclinate e risoluzioni diverse sui due assi, la conversione pixel->metro non e' immediata
            # trasformare il contorno in coordinate reali (metri)
            metric_contour = contour.astype(np.float32).copy()
            metric_contour[:, :, 0] *= self.meters_per_pixel_x
            metric_contour[:, :, 1] *= self.meters_per_pixel_y

            # rettangolo orientato in metri: centro, lati e angolo inclinazione
            metric_rect = cv2.minAreaRect(metric_contour)
            (_, _), (w_m, h_m), angle_m = metric_rect
            # salvare dimensioni massime e minime del rettangolo in metri
            max_dimension_m = float(max(w_m, h_m))
            min_dimension_m = float(min(w_m, h_m))
            if min_dimension_m <= 0.0:
                continue

            # area del contorno in metri quadrati
            contour_area_m2 = float(cv2.contourArea(metric_contour))

            # salvare tutte le misure utili
            blobs.append({
                'id': int(label),
                'type': blob_type,
                'area_m2': contour_area_m2,
                'area_px': contour_area_px,
                'centroid': (float(cx_px), float(cy_px)),
                'bbox': (
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT])
                ),
                'pixel_rect': pixel_rect,
                'metric_rect': metric_rect,
                'max_dimension_px': max_dimension_px,
                'min_dimension_px': min_dimension_px,
                'max_dimension_m': max_dimension_m,
                'min_dimension_m': min_dimension_m,
                'aspect_ratio': max_dimension_m / min_dimension_m,
                'angle': float(angle_m),
                'contour': contour
            })
        return blobs

    def detect_and_classify_objects(self, bright_map, dark_map):
        # ricavare larghezza immagine e assumere il nadir al centro
        image_width  = bright_map.shape[1]
        nadir_column = image_width / 2.0

        # estrarre blob come possibili oggetti e come possibili ombre
        objects = self.extract_blobs(bright_map, self.min_object_area_px, 'object')	# array blob oggetti
        shadows = self.extract_blobs(dark_map, self.min_shadow_area_px, 'shadow')	# array blob ombre

        detections = []
        for obj in objects:
            # associare all'oggetto l'ombra piu' plausibile
            best_shadow = None
            best_pair_score = -1.0
            best_alongtrack_gap_m = 0.0
            best_acrosstrack_gap_m = 0.0

            for candidate_shadow in shadows:
                # leggere i centroidi in pixel di oggetto e ombra
                obj_x, obj_y       = obj['centroid']
                shadow_x, shadow_y = candidate_shadow['centroid']

                # distanza across-track dal nadir
                obj_range    = abs(obj_x - nadir_column)
                shadow_range = abs(shadow_x - nadir_column)

                # scartare ombra se non e' piu' lontana dal nadir e se non e' sullo stesso lato dell'oggetto
                if shadow_range <= obj_range:
                    continue
                if (obj_x - nadir_column) * (shadow_x - nadir_column) < 0.0:
                    continue

                # distanza tra oggetto e ombra in metri
                alongtrack_gap_m  = abs(shadow_y - obj_y) * self.meters_per_pixel_y
                acrosstrack_gap_m = (shadow_range - obj_range) * self.meters_per_pixel_x

                # scartare coppie troppo lontane per essere un oggetto e la sua ombra
                if alongtrack_gap_m > self.max_shadow_alongtrack_gap_m:
                    continue
                if acrosstrack_gap_m > self.max_shadow_acrosstrack_gap_m:
                    continue

                # punteggio: 1 punto per ogni condizione buona della coppia oggetto-ombra
                pair_score = 0.0
                if alongtrack_gap_m <= 0.5 * self.max_shadow_alongtrack_gap_m:
                    pair_score += 1.0
                if max(0.0, acrosstrack_gap_m) <= 0.5 * self.max_shadow_acrosstrack_gap_m:
                    pair_score += 1.0

                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_shadow = candidate_shadow
                    best_alongtrack_gap_m = alongtrack_gap_m
                    best_acrosstrack_gap_m = acrosstrack_gap_m

            shadow = best_shadow
            pair_score = best_pair_score
            alongtrack_gap_m = best_alongtrack_gap_m
            acrosstrack_gap_m = best_acrosstrack_gap_m
            if shadow is None:
                continue

            shadow_ratio = shadow['max_dimension_m'] / max(obj['max_dimension_m'], 0.001)
            #aspect_ratio : max_dimension_m / min_dimension_m

            # tubo atteso: lungo, stretto e con ombra coerente con un oggetto allungato
            tube_score = pair_score
            if 2.0 <= obj['max_dimension_m'] <= 4.0:
                tube_score += 1.0
            if 0.4 <= obj['min_dimension_m'] <= 1.1:
                tube_score += 1.0
            if 2.0 <= obj['aspect_ratio'] <= 10.0:
                tube_score += 1.0
            if 0.8 <= shadow_ratio <= 4.0:
                tube_score += 1.0
            if shadow['area_m2'] >= self.tube_shadow_min_area_m2:
                tube_score += 1.0

            # boa attesa: piu' compatta e con ombra abbastanza lunga rispetto all'oggetto
            buoy_score = pair_score
            if 0.3 <= obj['max_dimension_m'] <= 2.7:
                buoy_score += 1.0
            if 0.3 <= obj['min_dimension_m'] <= 1.2:
                buoy_score += 1.0
            if 1.0 <= obj['aspect_ratio'] <= 4.0:
                buoy_score += 1.0
            if 1.0 <= shadow_ratio <= 5.0:
                buoy_score += 1.0
            if acrosstrack_gap_m >= self.buoy_far_shadow_gap_m:
                buoy_score += 1.0

            # usare il punteggio migliore come base per la confidenza finale
            max_score = max(tube_score, buoy_score)
            confidence = min(max_score / self.MAX_CLASSIFICATION_SCORE, 1.0)

            # sotto soglia la detection resta ignota, sopra soglia vince il punteggio maggiore
            if confidence < self.min_classification_confidence:
                classification = 'unknown'
                tie_breaker = 'below_confidence_threshold'
            elif tube_score > buoy_score:
                classification = 'tube'
                tie_breaker = 'not_needed'
            elif buoy_score > tube_score:
                classification = 'buoy'
                tie_breaker = 'not_needed'
            else:
                classification, tie_breaker = self.resolve_classification_tie(obj)

            # salvare oggetto, ombra e punteggi
            object_data = {
                'id': obj['id'],
                'centroid_px': [round(obj['centroid'][0], 2), round(obj['centroid'][1], 2)],
                'bbox_px': list(obj['bbox']),
                'rotated_bbox_px': self.rotated_bbox_points(obj['pixel_rect']),
                'max_dimension_m': round(obj['max_dimension_m'], 3),
                'min_dimension_m': round(obj['min_dimension_m'], 3),
                'aspect_ratio': round(obj['aspect_ratio'], 3),
                'area_m2': round(obj['area_m2'], 3),
                'angle_deg': round(obj['angle'], 3)
            }

            shadow_data = {
                'id': shadow['id'],
                'centroid_px': [round(shadow['centroid'][0], 2), round(shadow['centroid'][1], 2)],
                'bbox_px': list(shadow['bbox']),
                'rotated_bbox_px': self.rotated_bbox_points(shadow['pixel_rect']),
                'max_dimension_m': round(shadow['max_dimension_m'], 3),
                'min_dimension_m': round(shadow['min_dimension_m'], 3),
                'aspect_ratio': round(shadow['aspect_ratio'], 3),
                'area_m2': round(shadow['area_m2'], 3),
                'angle_deg': round(shadow['angle'], 3)
            }

            detections.append({
                'classification': classification,
                'confidence': round(float(confidence), 3),
                'tube_score': round(float(tube_score), 3),
                'buoy_score': round(float(buoy_score), 3),
                'pair_score': round(float(pair_score), 3),
                'alongtrack_gap_m': round(float(alongtrack_gap_m), 3),
                'acrosstrack_gap_m': round(float(acrosstrack_gap_m), 3),
                'tie_breaker': tie_breaker,
                'object': object_data,
                'shadow': shadow_data
            })

        return detections

    def resolve_classification_tie(self, obj):
        tube_tie_score = 0.0
        buoy_tie_score = 0.0

        # punteggi per boa
        if obj['area_m2'] <= self.tie_buoy_max_area_m2:
            buoy_tie_score += 1.0
        if obj['max_dimension_m'] <= self.tie_buoy_max_dimension_m:
            buoy_tie_score += 1.0
        if obj['aspect_ratio'] <= self.tie_buoy_max_aspect_ratio:
            buoy_tie_score += 1.0

        # punteggi per tubo
        if obj['area_m2'] >= self.tie_tube_min_area_m2:
            tube_tie_score += 1.0
        if obj['max_dimension_m'] >= self.tie_tube_min_dimension_m:
            tube_tie_score += 1.0
        if obj['aspect_ratio'] >= self.tie_tube_min_aspect_ratio:
            tube_tie_score += 1.0

        if tube_tie_score > buoy_tie_score:
            return 'tube', 'tube_geometry_area'
        if buoy_tie_score > tube_tie_score:
            return 'buoy', 'buoy_geometry_area'

        return 'unknown', 'unresolved_equal_score'

    def rotated_bbox_points(self, pixel_rect):
        # convertire il rettangolo orientato di OpenCV nei quattro vertici
        box = cv2.boxPoints(pixel_rect)
        return [[int(round(point[0])), int(round(point[1]))] for point in box]


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # SALVATAGGIO IMMAGINI
    # ========================================================

    def save_debug_image(self, folder, prefix, image, image_index):
        filename = os.path.join(folder, "{}_{:03d}.png".format(prefix, image_index))
        saved = cv2.imwrite(filename, self.normalize_uint8(image))
        if not saved:
            print("[SSS] Impossibile salvare immagine: {}".format(filename))
        return filename

    def save_classification_image(self, image, classifications, image_index):
        output = cv2.cvtColor(self.normalize_uint8(image), cv2.COLOR_GRAY2BGR)

        for detection in classifications:
            obj = detection['object']
            shadow = detection['shadow']
            label = "{} {:.2f}".format(detection['classification'], detection['confidence'])

            # disegnare rettangoli orientati, allineati alla forma del blob
            obj_box = np.asarray(obj['rotated_bbox_px'], dtype=np.int32)
            shadow_box = np.asarray(shadow['rotated_bbox_px'], dtype=np.int32)
            cv2.drawContours(output, [obj_box], 0, (0, 255, 0), 2)
            cv2.drawContours(output, [shadow_box], 0, (255, 0, 0), 2)

            text_x = int(obj['bbox_px'][0])
            text_y = max(15, int(obj['bbox_px'][1]) - 6)
            cv2.putText(output, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return self.save_debug_image(self.classification_image_folder, "classification", output, image_index)

    def save_classification_text(self, classifications, image_index):
        # salvare su file tutte le informazioni numeriche prodotte dalla classificazione
        filename = os.path.join(self.classification_text_folder, "classification_{:03d}.txt".format(image_index))

        try:
            with open(filename, 'w') as text_file:
                text_file.write("image_index: {}\n".format(image_index))
                text_file.write("detections_count: {}\n".format(len(classifications)))
                text_file.write("\nPROCESSING PARAMETERS\n")
                text_file.write("log_filter_scale: {:.3f}\n".format(self.log_filter_scale))
                text_file.write("median_kernel_size: {}\n".format(self.median_kernel_size))
                text_file.write("gaussian_kernel_size: {}\n".format(self.gaussian_kernel_size))
                text_file.write("gaussian_sigma: {:.3f}\n".format(self.gaussian_sigma))
                text_file.write("clahe_clip_limit: {:.3f}\n".format(self.clahe_clip_limit))
                text_file.write("clahe_tile_grid_size: {}\n".format(self.clahe_tile_grid_size))
                text_file.write("bright_morph_first_open_kernel_size: {}\n".format(self.bright_morph_first_open_kernel_size))
                text_file.write("bright_morph_first_close_kernel_size: {}\n".format(self.bright_morph_first_close_kernel_size))
                text_file.write("bright_morph_final_open_kernel_size: {}\n".format(self.bright_morph_final_open_kernel_size))
                text_file.write("bright_morph_final_close_kernel_size: {}\n".format(self.bright_morph_final_close_kernel_size))
                text_file.write("bright_morph_iterations: {}\n".format(self.bright_morph_iterations))
                text_file.write("dark_morph_first_open_kernel_size: {}\n".format(self.dark_morph_first_open_kernel_size))
                text_file.write("dark_morph_first_close_kernel_width: {}\n".format(self.dark_morph_first_close_kernel_width))
                text_file.write("dark_morph_first_close_kernel_height: {}\n".format(self.dark_morph_first_close_kernel_height))
                text_file.write("dark_morph_iterations: {}\n".format(self.dark_morph_iterations))
                text_file.write("saliency_percentile: {:.3f}\n".format(self.saliency_percentile))
                text_file.write("cfar_train_cells: {}\n".format(self.cfar_train_cells))
                text_file.write("cfar_guard_cells: {}\n".format(self.cfar_guard_cells))
                text_file.write("cfar_rank_percentile: {:.3f}\n".format(self.cfar_rank_percentile))
                text_file.write("cfar_threshold_scale: {:.3f}\n".format(self.cfar_threshold_scale))
                text_file.write("cfar_threshold_offset: {:.3f}\n".format(self.cfar_threshold_offset))
                text_file.write("bright_percentile: {:.3f}\n".format(self.bright_percentile))
                text_file.write("dark_percentile: {:.3f}\n".format(self.dark_percentile))
                text_file.write("sonar_range_m: {:.3f}\n".format(self.sonar_range_m))
                text_file.write("sonar_bins_per_side: {}\n".format(self.sonar_bins_per_side))
                text_file.write("sss_frequency_hz: {:.3f}\n".format(self.sss_frequency_hz))
                text_file.write("alongtrack_beam_angle_deg: {:.3f}\n".format(self.alongtrack_beam_angle_deg))
                text_file.write("auv_velocity_mps: {:.3f}\n".format(self.auv_velocity_mps))
                text_file.write("alongtrack_sampling_m: {:.6f}\n".format(self.alongtrack_sampling_m))
                text_file.write("alongtrack_beam_footprint_m: {:.6f}\n".format(self.alongtrack_beam_footprint_m))
                text_file.write("meters_per_pixel_x: {:.6f}\n".format(self.meters_per_pixel_x))
                text_file.write("meters_per_pixel_y: {:.6f}\n".format(self.meters_per_pixel_y))
                text_file.write("nadir_column: center_of_image\n")
                text_file.write("min_object_area_px: {}\n".format(self.min_object_area_px))
                text_file.write("min_shadow_area_px: {}\n".format(self.min_shadow_area_px))
                text_file.write("max_shadow_alongtrack_gap_m: {:.3f}\n".format(self.max_shadow_alongtrack_gap_m))
                text_file.write("max_shadow_accrosstrack_gap_m: {:.3f}\n".format(self.max_shadow_acrosstrack_gap_m))
                text_file.write("min_classification_confidence: {:.3f}\n".format(self.min_classification_confidence))
                text_file.write("tie_buoy_max_area_m2: {:.3f}\n".format(self.tie_buoy_max_area_m2))
                text_file.write("tie_buoy_max_dimension_m: {:.3f}\n".format(self.tie_buoy_max_dimension_m))
                text_file.write("tie_buoy_max_aspect_ratio: {:.3f}\n".format(self.tie_buoy_max_aspect_ratio))
                text_file.write("tie_tube_min_area_m2: {:.3f}\n".format(self.tie_tube_min_area_m2))
                text_file.write("tie_tube_min_dimension_m: {:.3f}\n".format(self.tie_tube_min_dimension_m))
                text_file.write("tie_tube_min_aspect_ratio: {:.3f}\n".format(self.tie_tube_min_aspect_ratio))
                text_file.write("tube_shadow_min_area_m2: {:.3f}\n".format(self.tube_shadow_min_area_m2))
                text_file.write("buoy_far_shadow_gap_m: {:.3f}\n".format(self.buoy_far_shadow_gap_m))

                if len(classifications) == 0:
                    text_file.write("\nNessuna detection trovata.\n")
                    return filename

                for detection_index, detection in enumerate(classifications):
                    obj = detection['object']
                    shadow = detection['shadow']

                    text_file.write("\nDETECTION {}\n".format(detection_index + 1))
                    text_file.write("classification: {}\n".format(detection['classification']))
                    text_file.write("confidence: {:.3f}\n".format(detection['confidence']))
                    text_file.write("tube_score: {:.3f}\n".format(detection['tube_score']))
                    text_file.write("buoy_score: {:.3f}\n".format(detection['buoy_score']))
                    text_file.write("pair_score: {:.3f}\n".format(detection['pair_score']))
                    text_file.write("alongtrack_gap_m: {:.3f}\n".format(detection['alongtrack_gap_m']))
                    text_file.write("acrosstrack_gap_m: {:.3f}\n".format(detection['acrosstrack_gap_m']))
                    text_file.write("tie_breaker: {}\n".format(detection.get('tie_breaker', 'not_available')))
                    text_file.write("sss_state: {}\n".format(detection.get('sss_state', 'UNKNOWN')))
                    text_file.write("isturning: {}\n".format(str(detection.get('isturning', False)).lower()))

                    text_file.write("object_id: {}\n".format(obj['id']))
                    text_file.write("object_centroid_px: [{:.2f}, {:.2f}]\n".format(obj['centroid_px'][0], obj['centroid_px'][1]))
                    text_file.write("object_bbox_px: {}\n".format(obj['bbox_px']))
                    text_file.write("object_rotated_bbox_px: {}\n".format(obj['rotated_bbox_px']))
                    text_file.write("object_max_dimension_m: {:.3f}\n".format(obj['max_dimension_m']))
                    text_file.write("object_min_dimension_m: {:.3f}\n".format(obj['min_dimension_m']))
                    text_file.write("object_aspect_ratio: {:.3f}\n".format(obj['aspect_ratio']))
                    text_file.write("object_area_m2: {:.3f}\n".format(obj['area_m2']))
                    text_file.write("object_angle_deg: {:.3f}\n".format(obj['angle_deg']))

                    text_file.write("shadow_id: {}\n".format(shadow['id']))
                    text_file.write("shadow_centroid_px: [{:.2f}, {:.2f}]\n".format(shadow['centroid_px'][0], shadow['centroid_px'][1]))
                    text_file.write("shadow_bbox_px: {}\n".format(shadow['bbox_px']))
                    text_file.write("shadow_rotated_bbox_px: {}\n".format(shadow['rotated_bbox_px']))
                    text_file.write("shadow_max_dimension_m: {:.3f}\n".format(shadow['max_dimension_m']))
                    text_file.write("shadow_min_dimension_m: {:.3f}\n".format(shadow['min_dimension_m']))
                    text_file.write("shadow_aspect_ratio: {:.3f}\n".format(shadow['aspect_ratio']))
                    text_file.write("shadow_area_m2: {:.3f}\n".format(shadow['area_m2']))
                    text_file.write("shadow_angle_deg: {:.3f}\n".format(shadow['angle_deg']))
        except IOError as exc:
            print("[SSS] Impossibile salvare report classificazione: {} ({})".format(filename, exc))

        return filename

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # inizializzare nodo ROS
    rospy.init_node('object_classification_node', anonymous=True)
    # istanziare ObjectClassificationNode
    node = ObjectClassificationNode()
    # spin ROS
    rospy.spin()
