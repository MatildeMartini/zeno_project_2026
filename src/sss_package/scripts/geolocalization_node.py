#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# ============================================================
# IMPORT LIBRERIE
# ============================================================
import math
import os
import csv
import re
import rospkg
import json
import yaml
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import String
from sss_package.msg import ImageMetadata
from sss_package.msg import GeolocatedObject
from geodetic_functions import ll2ne 
from geodetic_functions import ne2ll 


# conversione tra le classi pubblicate dal classificatore e le etichette usate
# nella lista finale SSS
CLASSIFICATION_TYPES = {
    'buoy': 'boa_probabile',
    'tube': 'tubo_probabile'
}

# impostato a True scarta dalla lista finale gli oggetti con
# classificazione diversa, che hanno coordinate vicine, aventi 
# livello di confidenza e object_count inferiore
FILTER_CROSS_CLASS_CONFLICTS = True


# ============================================================
# CLASSE PRINCIPALE NODO GEOLOCALIZATION
# ============================================================
class GeolocalizationNode:

    def __init__(self):

	print("[SSS] geolocalization_node.py is active\n")

        # inizializzare subscriber/publisher
        rospy.Subscriber('/classified_objects_topic', ImageMetadata, self.classified_objects_callback)
        self.pub_object_list = rospy.Publisher('list_topic', String, queue_size=20)

        # definire parametri Zeno e sensore
        self.sonar_range_m     = float(rospy.get_param('~sonar_range_m', 25.0))
        self.sensor_x_offset_m = float(rospy.get_param('~sensor_x_offset_m', 0.063))
        self.sensor_y_offset_m = float(rospy.get_param('~sensor_y_offset_m', 0.354))
        self.sensor_z_offset_m = float(rospy.get_param('~sensor_z_offset_m', 0.096))

        # definire parametri per lista finale
        self.object_match_distance_m         = float(rospy.get_param('~object_match_distance_m', 3.0))
        self.resolve_cross_class_conflicts   = bool(FILTER_CROSS_CLASS_CONFLICTS)
        self.cross_class_conflict_distance_m = float(rospy.get_param('~cross_class_conflict_distance_m', 2.0))
        self.final_map_filename              = rospy.get_param('~final_map_filename', 'final_detection_map.png')
        self.real_objects_csv                = rospy.get_param('~real_objects_csv', '')
        self.real_reference_objects          = self.load_real_reference_objects()
        self.safezone_polygon_points         = self.load_safezone_polygon_points()


        # definire parametri oggetti
        self.list_text_index = 0
        self.object_list = []
        self.next_object_id = 1
        self.auv_trajectory_points = []
        self.auv_trajectory_keys = set()

        # creazione cartelle per i risultati
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('sss_package') 
        default_folder = os.path.join(pkg_path, 'results', '9_list_texts')
        self.list_text_folder = rospy.get_param('~list_text_folder', default_folder)
        if not os.path.exists(self.list_text_folder):
            os.makedirs(self.list_text_folder)
        self.object_list_json_filename = os.path.join(self.list_text_folder, "SSS_object_list.json")
        rospy.on_shutdown(self.save_final_detection_map_with_trajectory)



# ________________________________________________________________________________________________________________________________


    # ========================================================
    # CALLBACK PER GEOLOCALIZZAZIONE
    # ========================================================
    def classified_objects_callback(self, msg):
        # 1. preparare la lista degli oggetti geolocalizzati per un'immagine
        geolocated_objects = []
        self.update_auv_trajectory(msg)

        # 2. individuare il nadir
        image_width = int(msg.image.width)
        localization_infos = []

        # 3. geolocalizzare ogni oggetto rilevato
        for object_index in range(len(msg.object_classes)):
            result = self.geolocalize_detection(msg, object_index, image_width)
            if result is not None:
                geolocated_object, localization_info = result
                geolocated_objects.append(geolocated_object)
                localization_infos.append(localization_info)

	# 4. aggiornare e pubblicare solo la lista finale degli oggetti
        self.update_object_list(geolocated_objects)
        self.filter_cross_class_conflicts()
        self.publish_object_list()
        self.save_geolocated_list_text(geolocated_objects, localization_infos)
        self.save_detection_map()



# ________________________________________________________________________________________________________________________________

    # ========================================================
    # TRAIETTORIA AUV
    # ========================================================
    def update_auv_trajectory(self, msg):
        # Salvare le posizioni durante la simulazione; la traiettoria viene
        # disegnata solo nella mappa finale, quando il nodo si chiude.
        for row_index in range(len(msg.nav_statuses)):
            nav_status = msg.nav_statuses[row_index]
            latitude = float(nav_status.position.latitude)
            longitude = float(nav_status.position.longitude)
            stamp_sec = None

            if row_index < len(msg.ping_stamps):
                stamp_sec = float(msg.ping_stamps[row_index].to_sec())
                trajectory_key = ('stamp', round(stamp_sec, 9))
            elif row_index < len(msg.ping_indices):
                trajectory_key = ('ping', int(msg.ping_indices[row_index]))
            else:
                trajectory_key = ('ll', round(latitude, 10), round(longitude, 10))

            if trajectory_key in self.auv_trajectory_keys:
                continue

            self.auv_trajectory_keys.add(trajectory_key)
            self.auv_trajectory_points.append({
                'lat': latitude,
                'lon': longitude,
                'stamp': stamp_sec
            })

    def save_final_detection_map_with_trajectory(self):
        filename = self.save_detection_map(draw_trajectory=True)
        if filename is not None:
            rospy.loginfo("[SSS] Mappa finale salvata con traiettoria AUV: {}".format(filename))


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # GEOLOCALIZZAZIONE
    # ========================================================

    def geolocalize_detection(self, msg, object_index, image_width):
        # xc e' la coordinata across-track; yc identifica il ping dell'immagine waterfall
        centroid_x = float(msg.object_centroid_x_px[object_index])
        centroid_y = float(msg.object_centroid_y_px[object_index])
        row_index  = int(round(centroid_y))

        # convertire la coordinata x del centroide in distanza orizzontale sul fondale
        altitude_m = float(msg.altitudes[row_index])
        nadir_column  = image_width / 2.0
        bins_per_side = image_width / 2.0
        if bins_per_side <= 0.0:
            return None

        range_bin = abs(float(centroid_x) - nadir_column)

        # calcolo della distanza (orizzontale) del centroide dalla terna sensore
        meters_per_pixel_slant = self.sonar_range_m / bins_per_side     # 25 m / 1000 bin
        slant_range_m = range_bin * meters_per_pixel_slant              # 25 m : 1000 bin = slant_range_m m : range_bin bin
        if slant_range_m <= altitude_m:
            rospy.logwarn("Detection in water-column/blind-zone: slant={:.3f} altitude={:.3f}".format(slant_range_m, altitude_m))
            return None

        # proiezione sul fondale: ground^2 = slant^2 - altitude^2
        ground_range_m = math.sqrt((slant_range_m * slant_range_m) - (altitude_m * altitude_m))

        # calcolo delle coordinate del centroide rispetto alla terna body:
        # nel body frame, x e' l'offset longitudinale del sensore, y e' la distanza
        # laterale dal nadir piu l'offset fisico del sonar
        side = -1.0 if float(centroid_x) < nadir_column else 1.0

        body_position = [
            self.sensor_x_offset_m,
            side * (self.sensor_y_offset_m + ground_range_m),
            self.sensor_z_offset_m
        ]

        # conversione da body a NED usando roll, pitch e yaw disponibili nel messaggio di navigazione
        nav_status = msg.nav_statuses[row_index]

        x_body, y_body, z_body = body_position

        roll  = float(nav_status.orientation.roll)
        pitch = float(nav_status.orientation.pitch)
        yaw   = float(nav_status.orientation.yaw)

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        north_m = (cy * cp) * x_body + (cy * sp * sr - sy * cr) * y_body + (cy * sp * cr + sy * sr) * z_body
        east_m  = (sy * cp) * x_body + (sy * sp * sr + cy * cr) * y_body + (sy * sp * cr - cy * sr) * z_body
        down_m  = (-sp) * x_body + (cp * sr) * y_body + (cp * cr) * z_body

        # conversione da NED a coordinate assolute (latitude, longitude)
        object_latitude, object_longitude = ne2ll(
            [float(nav_status.position.latitude), float(nav_status.position.longitude)],
            [north_m, east_m]
        )

        # risultati della geolocalizzazione
        output = GeolocatedObject()
        output.object_class  = msg.object_classes[object_index]
        output.confidence    = float(msg.object_confidences[object_index])
        output.latitude      = float(object_latitude)
        output.longitude     = float(object_longitude)
        output.ping_index    = int(msg.ping_indices[row_index])
        output.ping_stamp    = msg.ping_stamps[row_index]
        output.centroid_x_px = centroid_x
        output.centroid_y_px = centroid_y

        if object_index < len(msg.object_bbox_x_px):
            output.bbox_x_px = int(msg.object_bbox_x_px[object_index])
            output.bbox_y_px = int(msg.object_bbox_y_px[object_index])
            output.bbox_width_px  = int(msg.object_bbox_width_px[object_index])
            output.bbox_height_px = int(msg.object_bbox_height_px[object_index])

        localization_info = {
            'object_index': int(object_index),
            'row_index': int(row_index),
            'image_width': int(image_width),
            'auv_latitude': float(nav_status.position.latitude),
            'auv_longitude': float(nav_status.position.longitude),
            'auv_yaw_rad': float(nav_status.orientation.yaw),
            'altitude_m': float(altitude_m),
            'slant_range_m': float(slant_range_m),
            'ground_range_m': float(ground_range_m),
            'body_x_m': float(body_position[0]),
            'body_y_m': float(body_position[1]),
            'body_z_m': float(body_position[2]),
            'north_offset_m': float(north_m),
            'east_offset_m': float(east_m),
            'down_offset_m': float(down_m)
        }

        return output, localization_info

# ________________________________________________________________________________________________________________________________

    # ========================================================
    # LISTA
    # ========================================================
    def update_object_list(self, geolocated_objects):
        # aggiornare la lista finale: filtrare target fuori safezone e fondere detection vicine
        # dello stesso tipo nello stesso oggetto osservato piu' volte
        for geolocated_object in geolocated_objects:
            object_type = CLASSIFICATION_TYPES.get(geolocated_object.object_class)
            if object_type is None:
                continue
            
            # escludere target rilevati fuori dalla safezone
            if not self.is_point_inside_safezone(
                float(geolocated_object.latitude),
                float(geolocated_object.longitude)
            ):
                rospy.loginfo("[SSS] Detection esclusa dalla lista finale: fuori safezone lat={:.10f} lon={:.10f}".format(
                    float(geolocated_object.latitude),
                    float(geolocated_object.longitude)
                ))
                continue

            # riconoscere oggetti inquadrati piu' volte
            existing_object = self.find_matching_object(geolocated_object, object_type)
            if existing_object is None:
                self.object_list.append({
                    'id': self.next_object_id,
                    'confidence': float(geolocated_object.confidence),
                    'obs_count': 1,
                    'lon': float(geolocated_object.longitude),
                    'lat': float(geolocated_object.latitude),
                    'type': object_type
                })
                self.next_object_id += 1
            else:
                self.update_existing_object(existing_object, geolocated_object)

    def is_point_inside_safezone(self, latitude, longitude):
        # se la safezone non e' disponibile, non bloccare le detection
        if len(self.safezone_polygon_points) < 3:
            return True

        # ogni attraversamento del bordo alterna lo stato dentro/fuori.
        # i punti esattamente sul bordo sono considerati validi
        inside = False
        point_lat = float(latitude)
        point_lon = float(longitude)
        previous_lat = float(self.safezone_polygon_points[-1][0])
        previous_lon = float(self.safezone_polygon_points[-1][1])

        # scorrere ogni lato del poligono, usando il punto precedente e quello corrente
        for point in self.safezone_polygon_points:
            current_lat = float(point[0])
            current_lon = float(point[1])

            # controllare prima se il punto cade esattamente sul lato corrente.
            # epsilon evita problemi numerici con coordinate geografiche molto vicine
            epsilon = 1e-12
            cross_product = (
                (point_lon - previous_lon) * (current_lat - previous_lat) -
                (point_lat - previous_lat) * (current_lon - previous_lon)
            )
            if abs(cross_product) <= epsilon:
                min_lat = min(previous_lat, current_lat) - epsilon
                max_lat = max(previous_lat, current_lat) + epsilon
                min_lon = min(previous_lon, current_lon) - epsilon
                max_lon = max(previous_lon, current_lon) + epsilon
                if min_lat <= point_lat <= max_lat and min_lon <= point_lon <= max_lon:
                    return True

            # se una semiretta orizzontale dal punto interseca il lato, alternare
            # inside. Alla fine, inside=True significa punto dentro al poligono
            crosses_latitude = ((current_lat > point_lat) != (previous_lat > point_lat))
            if crosses_latitude:
                lon_at_point_lat = (
                    (previous_lon - current_lon) * (point_lat - current_lat) /
                    (previous_lat - current_lat)
                ) + current_lon
                if point_lon < lon_at_point_lat:
                    inside = not inside

            previous_lat = current_lat
            previous_lon = current_lon

        return inside

    def find_matching_object(self, geolocated_object, object_type):
        # cercare l'oggetto gia' salvato piu vicino alla nuova detection.
        # la distanza viene calcolata in metri convertendo lat/lon in Nord-Est locale
        best_object = None
        best_distance = None

        for stored_object in self.object_list:
            if stored_object['type'] != object_type:
                continue

            north_m, east_m = ll2ne(
                [stored_object['lat'], stored_object['lon']],
                [float(geolocated_object.latitude), float(geolocated_object.longitude)]
            )
            distance = math.sqrt((north_m * north_m) + (east_m * east_m))
            if distance <= self.object_match_distance_m:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_object = stored_object

        return best_object

    def update_existing_object(self, stored_object, geolocated_object):
        # aggiornare posizione e confidenza senza conservare tutte le detection precedenti dell'oggetto
        old_count = int(stored_object['obs_count'])
        new_count = old_count + 1

        # media incrementale
        stored_object['confidence'] = ((stored_object['confidence'] * old_count) + float(geolocated_object.confidence)) / float(new_count)
        stored_object['lat']        = ((stored_object['lat'] * old_count) + float(geolocated_object.latitude))  / float(new_count)
        stored_object['lon']        = ((stored_object['lon'] * old_count) + float(geolocated_object.longitude)) / float(new_count)
        stored_object['obs_count']  = new_count

# ________________________________________________________________________________________________________________________________

    # ========================================================
    # LISTA FILTRATA
    # ========================================================
    def filter_cross_class_conflicts(self):
        # filtro opzionale: se disattivato, mantenere la lista finale originale;
        # se attivato e se due oggetti appartenenti a classi differenti risultano troppo vicini tra loro,
        # viene mantenuto esclusivamente quello con score maggiore, mentre l'altro viene eliminato
        if not self.resolve_cross_class_conflicts:
            return
        if len(self.object_list) < 2:
            return

        # Ordinare gli oggetti dal piu affidabile al meno affidabile.
        # Lo score combina confidenza e numero di osservazioni, cosi un oggetto
        # rivisto molte volte pesa piu di una detection isolata ad alta confidenza.
        ranked_objects = sorted(
            self.object_list,
            key=lambda obj: self.compute_final_object_score(obj),
            reverse=True
        )

        kept_objects = []
        removed_ids = []
        for candidate in ranked_objects:
            is_conflict = False
            for kept_object in kept_objects:
                # confrontare solo classi diverse: oggetti dello stesso tipo sono gia'
                # gestiti dalla funzione update_object_list()
                if candidate['type'] == kept_object['type']:    # la prima iterazione, l'array e' vuoto e l'oggetto con maggiore confidenza viene inserito
                    continue                                    # successivamente, viene confrontato con oggetti con confidenza inferiore

                # se due oggetti di classe diversa sono troppo vicini, tenere quello
                # gia' accettato perche' ha score maggiore nella lista ordinata

                # convertire le due coordinate lat/lon in un frame locale Nord-Est
                # e calcolare la distanza euclidea in metri
                north_m, east_m = ll2ne(
                    [float(candidate['lat']), float(candidate['lon'])],
                    [float(kept_object['lat']), float(kept_object['lon'])]
                )
                distance_m = math.sqrt((north_m * north_m) + (east_m * east_m))
                if distance_m <= self.cross_class_conflict_distance_m:
                    is_conflict = True
                    removed_ids.append(candidate['id'])
                    break
            # se oggetti non vanno in conflitto, mantenerli
            if not is_conflict:
                kept_objects.append(candidate)

        # se non e' stato rimosso nulla, evitare di riscrivere inutilmente la lista
        if len(removed_ids) == 0:
            return

        # mantenere l'ordine originale della lista finale, rimuovendo solo gli id scartati
        kept_ids = set([obj['id'] for obj in kept_objects])
        self.object_list = [obj for obj in self.object_list if obj['id'] in kept_ids]

    def compute_final_object_score(self, stored_object):
        confidence = float(stored_object.get('confidence', 0.0))
        obs_count = int(stored_object.get('obs_count', 1))
        return confidence * math.log10(obs_count + 1)


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # PUBBLICAZIONE LISTA
    # ========================================================
    def publish_object_list(self):
        # costruire la struttura del JSON pubblicato sul topic
        output = OrderedDict()
        for stored_object in self.object_list:
            output[str(stored_object['id'])] = OrderedDict([
                ('confidence', round(float(stored_object['confidence']), 3)),
                ('obs_count', int(stored_object['obs_count'])),
                ('lon', float(stored_object['lon'])),
                ('lat', float(stored_object['lat'])),
                ('type', stored_object['type'])
            ])

        json_text = json.dumps(output, indent=4)
        self.pub_object_list.publish(String(data=json_text))
        rospy.loginfo("[SSS] list_topic: pubblicata lista finale con {} oggetti unici".format(len(self.object_list)))

        # salvare la stessa lista finale nel file JSON
        file_output = OrderedDict([
            ('final_list', output)
        ])
        try:
            with open(self.object_list_json_filename, 'w') as json_file:
                json_file.write(json.dumps(file_output, indent=4))
                json_file.write("\n")
        except IOError as exc:
            rospy.logwarn("[SSS] Impossibile salvare lista finale JSON: {} ({})".format(
                self.object_list_json_filename,
                exc
            ))

    def normalize_map_object_type(self, object_type):
        # Uniforma nomi italiani/inglesi e classi probabili prima di separare
        # boe e tubi nella mappa finale.
        object_type = str(object_type).strip().lower()
        if object_type in ['boa', 'buoy', 'boa_probabile', 'buoy_probabile']:
            return 'boa'
        if object_type in ['tubo', 'tube', 'tubo_probabile', 'tube_probabile']:
            return 'tubo'
        return None

    def resolve_real_objects_csv_path(self):
        csv_path = str(self.real_objects_csv).strip()
        if csv_path == '':
            return ''

        csv_path = os.path.expanduser(os.path.expandvars(csv_path))
        if not os.path.isabs(csv_path):
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads', csv_path)
            if os.path.exists(downloads_path):
                csv_path = downloads_path

        return csv_path

    def load_real_reference_objects(self):
        csv_path = self.resolve_real_objects_csv_path()
        if csv_path == '':
            rospy.loginfo("[SSS] Nessun CSV oggetti reali configurato (~real_objects_csv vuoto)")
            return []

        if not os.path.exists(csv_path):
            rospy.logwarn("[SSS] CSV oggetti reali non trovato: {}".format(csv_path))
            return []

        reference_objects = []
        try:
            with open(csv_path, 'r') as csv_file:
                sample = csv_file.read(2048)
                csv_file.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)

                if has_header:
                    reader = csv.DictReader(csv_file, dialect=dialect)
                    for row_index, row in enumerate(reader, start=2):
                        normalized_row = {}
                        for key, value in row.items():
                            if key is None:
                                continue
                            normalized_key = str(key).strip().lower()
                            normalized_row[normalized_key] = '' if value is None else str(value).strip()

                        object_type = None
                        for key in ['type', 'class', 'object_class', 'object', 'descrizione', 'description', 'nome', 'name']:
                            object_type = self.normalize_map_object_type(normalized_row.get(key, ''))
                            if object_type is not None:
                                break
                        if object_type is None:
                            continue

                        latitude = None
                        longitude = None
                        wkt_text = normalized_row.get('wkt', '')
                        point_match = re.search(r'POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)', wkt_text, re.IGNORECASE)
                        if point_match is not None:
                            longitude = float(point_match.group(1))
                            latitude = float(point_match.group(2))
                        else:
                            for key in ['lat', 'latitude', 'latitudine']:
                                if key in normalized_row:
                                    latitude = float(normalized_row[key])
                                    break
                            for key in ['lon', 'lng', 'long', 'longitude', 'longitudine']:
                                if key in normalized_row:
                                    longitude = float(normalized_row[key])
                                    break

                        if latitude is None or longitude is None:
                            rospy.logwarn("[SSS] Riga CSV oggetti reali {} ignorata: coordinate non valide".format(row_index))
                            continue

                        reference_objects.append({
                            'id': normalized_row.get('nome', normalized_row.get('name', row_index)),
                            'lat': float(latitude),
                            'lon': float(longitude),
                            'type': object_type
                        })
                else:
                    reader = csv.reader(csv_file, dialect=dialect)
                    for row_index, row in enumerate(reader, start=1):
                        values = [str(value).strip() for value in row if str(value).strip() != '']
                        if len(values) == 0:
                            continue

                        object_type = None
                        for value in values:
                            object_type = self.normalize_map_object_type(value)
                            if object_type is not None:
                                break
                        if object_type is None:
                            continue

                        latitude = None
                        longitude = None
                        for value in values:
                            point_match = re.search(r'POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)', value, re.IGNORECASE)
                            if point_match is not None:
                                longitude = float(point_match.group(1))
                                latitude = float(point_match.group(2))
                                break

                        if latitude is None or longitude is None:
                            numeric_values = []
                            for value in values:
                                try:
                                    numeric_values.append(float(value))
                                except ValueError:
                                    pass

                            if len(numeric_values) < 2:
                                rospy.logwarn("[SSS] Riga CSV oggetti reali {} ignorata: coordinate non valide".format(row_index))
                                continue

                            first_value = numeric_values[0]
                            second_value = numeric_values[1]
                            if abs(first_value) <= 20.0 and abs(second_value) > 20.0:
                                latitude = float(second_value)
                                longitude = float(first_value)
                            else:
                                latitude = float(first_value)
                                longitude = float(second_value)

                        reference_objects.append({
                            'id': row_index,
                            'lat': float(latitude),
                            'lon': float(longitude),
                            'type': object_type
                        })

            rospy.loginfo("[SSS] Caricati {} oggetti reali da {}".format(
                len(reference_objects),
                csv_path
            ))
        except Exception as exc:
            rospy.logwarn("[SSS] Impossibile leggere CSV oggetti reali {}: {}".format(csv_path, exc))
            return []

        return reference_objects


# ________________________________________________________________________________________________________________________________

    # ========================================================
    # MAPPA
    # ========================================================
    def save_detection_map(self, draw_trajectory=False):
        # salvare un'immagine con safezone, oggetti classifcati e oggetti reali
        final_boas = []
        final_tubos = []
        reference_boas = []
        reference_tubos = []

        # separare gli oggetti finali rilevati in boe e tubi:
        # boa = cerchio verde
        # tubo = quadrato blu
        for stored_object in self.object_list:
            object_type = self.normalize_map_object_type(stored_object.get('type', ''))
            map_object = {
                'id': stored_object.get('id', ''),
                'lat': float(stored_object['lat']),
                'lon': float(stored_object['lon'])
            }
            if object_type == 'boa':
                final_boas.append(map_object)
            elif object_type == 'tubo':
                final_tubos.append(map_object)

        # separare allo stesso modo gli oggetti reali letti dal CSV:
        # boa = cerchio nero
        # tubo = quadrato nero
        for stored_object in self.real_reference_objects:
            object_type = self.normalize_map_object_type(stored_object.get('type', ''))
            map_object = {
                'id': stored_object.get('id', ''),
                'lat': float(stored_object['lat']),
                'lon': float(stored_object['lon'])
            }
            if object_type == 'boa':
                reference_boas.append(map_object)
            elif object_type == 'tubo':
                reference_tubos.append(map_object)

        filename = os.path.join(self.list_text_folder, self.final_map_filename)
        fig = None

        try:
            fig, ax = plt.subplots(figsize=(9, 8))
            fig.subplots_adjust(right=0.78)

            # disegnare il poligono safezone
            if len(self.safezone_polygon_points) > 0:
                polygon_lats = [float(point[0]) for point in self.safezone_polygon_points]
                polygon_lons = [float(point[1]) for point in self.safezone_polygon_points]
                polygon_lats.append(float(self.safezone_polygon_points[0][0]))
                polygon_lons.append(float(self.safezone_polygon_points[0][1]))
                ax.plot(polygon_lons, polygon_lats, color='black', linewidth=1.8, label='safezone', zorder=2)

            # disegnare la traiettoria AUV solo nella mappa finale di shutdown
            if draw_trajectory and len(self.auv_trajectory_points) > 0:
                trajectory_points = sorted(
                    self.auv_trajectory_points,
                    key=lambda point: float(point['stamp']) if point['stamp'] is not None else 0.0
                )
                trajectory_lats = [float(point['lat']) for point in trajectory_points]
                trajectory_lons = [float(point['lon']) for point in trajectory_points]
                ax.plot(
                    trajectory_lons,
                    trajectory_lats,
                    color='lightskyblue',
                    linewidth=1.8,
                    alpha=0.9,
                    label='traiettoria AUV',
                    zorder=3
                )

            # disegnare sulla stessa mappa detection SSS e oggetti reali.
            for object_group, color, marker, label, label_ids, facecolors in [
                (final_boas, 'green', 'o', 'SSS final boa', True, None),
                (final_tubos, 'blue', 's', 'SSS final tubo', True, None),
                (reference_boas, 'black', 'o', 'boa nota', False, 'none'),
                (reference_tubos, 'black', 's', 'tubo noto', False, 'none')
            ]:
                if len(object_group) == 0:
                    continue

                lats = [obj['lat'] for obj in object_group]
                lons = [obj['lon'] for obj in object_group]

                ax.scatter(lons, lats, s=70, c=color if facecolors is None else None, marker=marker,
                    edgecolors=color,facecolors=facecolors, linewidths=1.5, label=label, zorder=4)

                if label_ids:
                    for obj in object_group:
                        ax.text(
                            obj['lon'],
                            obj['lat'],
                            str(obj['id']),
                            fontsize=8,
                            color=color,
                            ha='left',
                            va='bottom',
                            zorder=5
                        )

            # Se disponibile, usare la safezone per fissare i limiti della mappa.
            # La traiettoria AUV puo uscire dal poligono e non deve allargare gli assi.
            axis_points = []

            for point in self.safezone_polygon_points:
                axis_points.append((float(point[0]), float(point[1])))

            if len(axis_points) == 0:
                for object_group in [final_boas, final_tubos, reference_boas, reference_tubos]:
                    for map_object in object_group:
                        axis_points.append((float(map_object['lat']), float(map_object['lon'])))

            if len(axis_points) > 0:
                lats = [point[0] for point in axis_points]
                lons = [point[1] for point in axis_points]
                min_lat = min(lats)
                max_lat = max(lats)
                min_lon = min(lons)
                max_lon = max(lons)

                lat_span = max_lat - min_lat
                lon_span = max_lon - min_lon
                min_span = 0.0001
                lat_padding = min_span / 2.0 if lat_span < min_span else lat_span * 0.15
                lon_padding = min_span / 2.0 if lon_span < min_span else lon_span * 0.15

                ax.set_xlim(min_lon - lon_padding, max_lon + lon_padding)
                ax.set_ylim(min_lat - lat_padding, max_lat + lat_padding)

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('SSS final object map')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

            fig.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close(fig)
            return filename
        except Exception as exc:
            # chiudere la figura
            rospy.logwarn("Impossibile salvare mappa detection: {} ({})".format(filename, exc))
            if fig is not None:
                plt.close(fig)
            return None


# ________________________________________________________________________________________________________________________________

    def parse_wkt_polygon_points(self, wkt_text):
        polygon_match = re.search(r'POLYGON\s*\(\(\s*(.*?)\s*\)\)', str(wkt_text), re.IGNORECASE)
        if polygon_match is None:
            return []

        polygon_points = []
        for coordinate_text in polygon_match.group(1).split(','):
            values = coordinate_text.strip().split()
            if len(values) < 2:
                continue

            try:
                longitude = float(values[0])
                latitude = float(values[1])
            except ValueError:
                continue

            polygon_points.append((float(latitude), float(longitude)))

        if len(polygon_points) > 1:
            first_point = polygon_points[0]
            last_point = polygon_points[-1]
            if abs(first_point[0] - last_point[0]) < 1e-12 and abs(first_point[1] - last_point[1]) < 1e-12:
                polygon_points = polygon_points[:-1]

        return polygon_points

    def read_csv_dialect_and_header(self, csv_file):
        sample = csv_file.read(2048)
        csv_file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True

        return dialect, has_header

    def load_safezone_polygon_points_from_csv(self):
        csv_path = self.resolve_real_objects_csv_path()
        if csv_path == '':
            rospy.loginfo("[SSS] Nessun CSV oggetti reali configurato: safezone CSV non disponibile")
            return []

        if not os.path.exists(csv_path):
            rospy.logwarn("[SSS] CSV per safezone non trovato: {}".format(csv_path))
            return []

        try:
            with open(csv_path, 'r') as csv_file:
                dialect, has_header = self.read_csv_dialect_and_header(csv_file)

                if has_header:
                    reader = csv.DictReader(csv_file, dialect=dialect)
                    for row_index, row in enumerate(reader, start=2):
                        row_values = ['' if value is None else str(value).strip() for value in row.values()]
                        row_text = ' '.join(row_values).lower()

                        if 'safezone' not in row_text and 'safe zone' not in row_text:
                            continue

                        for value in row_values:
                            polygon_points = self.parse_wkt_polygon_points(value)
                            if len(polygon_points) >= 3:
                                rospy.loginfo("[SSS] Safezone caricata dal CSV {} riga {} con {} vertici".format(
                                    csv_path,
                                    row_index,
                                    len(polygon_points)
                                ))
                                return polygon_points
                else:
                    reader = csv.reader(csv_file, dialect=dialect)
                    for row_index, row in enumerate(reader, start=1):
                        row_values = [str(value).strip() for value in row]
                        row_text = ' '.join(row_values).lower()

                        if 'safezone' not in row_text and 'safe zone' not in row_text:
                            continue

                        for value in row_values:
                            polygon_points = self.parse_wkt_polygon_points(value)
                            if len(polygon_points) >= 3:
                                rospy.loginfo("[SSS] Safezone caricata dal CSV {} riga {} con {} vertici".format(
                                    csv_path,
                                    row_index,
                                    len(polygon_points)
                                ))
                                return polygon_points
        except Exception as exc:
            rospy.logwarn("[SSS] Impossibile leggere safezone dal CSV {}: {}".format(csv_path, exc))
            return []

        rospy.logwarn("[SSS] Nessuna riga safezone POLYGON trovata nel CSV: {}".format(csv_path))
        return []

    def load_safezone_polygon_points(self):
        polygon_points = []

        try:
            # rospkg trova il pacchetto zeno_mission, poi sono letti i vertici NED del poligono da region_params.yaml
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('zeno_mission')
            yaml_file = os.path.join(pkg_path, 'config', 'region_params.yaml')

            with open(yaml_file, 'r') as safezone_file:
                data = yaml.safe_load(safezone_file)
            if data is None:
                raise ValueError("region_params.yaml vuoto")

            origin = data['origin']['coordinates']
            map_origin = (
                float(origin['latitude']),
                float(origin['longitude'])
            )
            vertices_ned = data['polygon_vertices']['original']

            # per lavorare e disegnare in lat/lon, i vertici [North, East] vengono convertiti 
            for vertex in vertices_ned:
                north = float(vertex[0])
                east = float(vertex[1])
                latitude, longitude = ne2ll(map_origin, (north, east))
                polygon_points.append((float(latitude), float(longitude)))

            rospy.loginfo("[SSS] Safezone caricata da region_params.yaml con {} vertici".format(
                len(polygon_points)
            ))
            return polygon_points
        except Exception as exc:
            rospy.logwarn("[SSS] Impossibile leggere safezone da region_params.yaml: {}".format(exc))
            return self.load_safezone_polygon_points_from_csv()


    def save_geolocated_list_text(self, geolocated_objects, localization_infos):
        # salvare su file la lista finale con coordinate e dettagli della geolocalizzazione
        filename = os.path.join(self.list_text_folder, "geolocated_list_{:03d}.txt".format(self.list_text_index))
        self.list_text_index += 1

        try:
            with open(filename, 'w') as text_file:

                text_file.write("geolocated_objects_count: {}\n".format(len(geolocated_objects)))
                text_file.write("sonar_range_m: {:.3f}\n".format(self.sonar_range_m))
                text_file.write("sensor_x_offset_m: {:.3f}\n".format(self.sensor_x_offset_m))
                text_file.write("sensor_y_offset_m: {:.3f}\n".format(self.sensor_y_offset_m))
                text_file.write("sensor_z_offset_m: {:.3f}\n".format(self.sensor_z_offset_m))

                if len(geolocated_objects) == 0:
                    text_file.write("\nNessun oggetto geolocalizzato.\n")
                    return filename

                for index, geolocated_object in enumerate(geolocated_objects):
                    info = localization_infos[index]
                    text_file.write("\nOBJECT {}\n".format(index + 1))
                    text_file.write("classification: {}\n".format(geolocated_object.object_class))
                    text_file.write("confidence: {:.3f}\n".format(geolocated_object.confidence))
                    text_file.write("latitude: {:.10f}\n".format(geolocated_object.latitude))
                    text_file.write("longitude: {:.10f}\n".format(geolocated_object.longitude))
                    text_file.write("ping_index: {}\n".format(geolocated_object.ping_index))
                    text_file.write("ping_stamp: {:.9f}\n".format(geolocated_object.ping_stamp.to_sec()))
                    text_file.write("row_index: {}\n".format(info['row_index']))
                    text_file.write("centroid_px: [{:.2f}, {:.2f}]\n".format(geolocated_object.centroid_x_px, geolocated_object.centroid_y_px))
                    text_file.write("bbox_px: [{}, {}, {}, {}]\n".format(
                        geolocated_object.bbox_x_px,
                        geolocated_object.bbox_y_px,
                        geolocated_object.bbox_width_px,
                        geolocated_object.bbox_height_px
                    ))
                    text_file.write("auv_latitude: {:.10f}\n".format(info['auv_latitude']))
                    text_file.write("auv_longitude: {:.10f}\n".format(info['auv_longitude']))
                    text_file.write("auv_yaw_rad: {:.6f}\n".format(info['auv_yaw_rad']))
                    text_file.write("altitude_m: {:.3f}\n".format(info['altitude_m']))
                    text_file.write("slant_range_m: {:.3f}\n".format(info['slant_range_m']))
                    text_file.write("ground_range_m: {:.3f}\n".format(info['ground_range_m']))
                    text_file.write("body_position_m: [{:.3f}, {:.3f}, {:.3f}]\n".format(
                        info['body_x_m'],
                        info['body_y_m'],
                        info['body_z_m']
                    ))
                    text_file.write("ned_offset_m: [{:.3f}, {:.3f}, {:.3f}]\n".format(
                        info['north_offset_m'],
                        info['east_offset_m'],
                        info['down_offset_m']
                    ))
        except IOError as exc:
            rospy.logwarn("Impossibile salvare lista geolocalizzata: {} ({})".format(filename, exc))

        return filename

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # inizializzare nodo ROS
    rospy.init_node('geolocalization_node', anonymous=True)
    # istanziare GeolocalizationNode
    node = GeolocalizationNode()
    # spin ROS
    rospy.spin()
