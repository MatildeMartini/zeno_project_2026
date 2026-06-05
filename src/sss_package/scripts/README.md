# Script Side-Scan Sonar

Questa cartella contiene i quattro nodi principali della pipeline SSS. Il loro
compito e' trasformare i ping grezzi del side-scan sonar in immagini waterfall,
rilevare e classificare gli oggetti, geolocalizzarli e visualizzare in tempo
reale il risultato.

## Flusso Della Pipeline

Il flusso principale dei topic e' il seguente:


/drivers/sss_sim
/drivers/altitude_sim
/nav_status
        |
        v
waterfall_creator_node.py
        |
        | /waterfall_image_topic
        | /waterfall_realtime_topic
        v
object_classification_node.py
        |
        | /classified_objects_topic
        v
geolocalization_node.py
        |
        | /list_topic
        v
SSS_object_list.json


`realtime_visualizer_node.py` esegue in parallelo: mostra la waterfall
realtime e le detection classificate, in una finestra OpenCV con le
bounding box sovrapposte.

La pipeline puo essere avviata con:
roslaunch sss_package sss.launch


## `waterfall_creator_node.py`

`waterfall_creator_node.py` e' il primo nodo della pipeline SSS. Legge i dati
grezzi del sonar da `/drivers/sss_sim`, l'altitudine da `/drivers/altitude_sim`, 
lo stato di navigazione da `/nav_status` e lo stato SSS da `/phase1/SSS` o 
`/phase3/SSS`.

Per ogni ping, il nodo estrae i beam sinistro e destro, inverte il beam
sinistro, li unisce in un unico profilo across-track e applica una correzione
TVG (Time Variable Gain) per compensare la perdita di intensita' con la distanza.
Ogni ping viene poi inserito in cima a un buffer waterfall, insieme ai metadati
allineati riga per riga: indice del ping, timestamp, posizione e orientazione del
veicolo, altitudine, stato SSS e informazione di curva.Questa informazione viene
usata poi dal nodo addetto alla geolocalizzazione.

Il nodo pubblica due messaggi `sss_package/ImageMetadata` in:

- `/waterfall_realtime_topic`: waterfall aggiornata a ogni ping, con al massimo
  gli ultimi 500 ping, usata dal visualizzatore realtime.
- `/waterfall_image_topic`: immagini waterfall complete da 150 ping con overlap
  di 75 ping, usate dal nodo di classificazione.

Gli output vengono salvati in `src/sss_package/results/`:

- `0_raw_images/`: immagini raw prima della TVG.
- `1_waterfall_images/`: immagini waterfall processate (dopo TVG).
- `99_echo_intensity/`: plot 2D e 3D dell'intensita sonar, salvati alla chiusura
  del nodo.

 

## `object_classification_node.py`

`object_classification_node.py` riceve le immagini da `/waterfall_image_topic` e
pubblica su `/classified_objects_topic` un nuovo messaggio `ImageMetadata` che
mantiene immagine e metadati originali, ma aggiunge le informazioni sugli
oggetti rilevati.

Per ogni detection vengono salvati:

- classe (`buoy`, `tube` o `unknown`);
- confidenza;
- centroide in pixel;
- bounding box in pixel.

La classificazione e' basata su elaborazione di immagine e considerazioni
geometriche. Il nodo calcola mappe di salienza OS-CFAR per evidenziare ritorni
luminosi, crea mappe binarie bright/dark tramite soglie percentili, applica 
pulizia morfologica e poi estrae componenti connesse. Ogni blob luminoso viene
confrontato con le ombre candidate: una coppia oggetto-ombra e' considerata 
valida se l'ombra e' piu' lontana dal nadir, si trova dallo stesso
lato dell'oggetto e rispetta limiti di distanza along-track e across-track.

La decisione tra `buoy` e `tube` usa punteggi geometrici. Per esempio, un tubo e'
favorito da una forma piu' allungata, dimensioni maggiori e ombra coerente; una
boa e' favorita da una forma piu' compatta e da un rapporto oggetto-ombra piu'
compatibile con un target piccolo. Se i punteggi sono troppo bassi, la detection
viene marcata come `unknown`; se i punteggi sono pari, vengono usati criteri su area,
dimensione massima e aspect ratio.

Il nodo salva output di debug in `src/sss_package/results/`:

- `2_filtered_images/`
- `3_saliency_images/bright_maps/`
- `3_saliency_images/dark_maps/`
- `4_saliency_binary_images/`
- `5_binary_maps_images/`
- `6_binary_and_salient_maps_images/`
- `7_morph_images/`
- `8_classification_images/`
- `8_classification_texts/`



## `geolocalization_node.py`

`geolocalization_node.py` riceve le detection classificate da
`/classified_objects_topic`, converte ogni detection da coordinate immagine a
coordinate geografiche e mantiene la lista finale degli oggetti SSS pubblicata
su `/list_topic`.

Il procedimento usa direttamente i metadati salvati nella waterfall. La
coordinata `centroid_y_px` identifica la riga dell'immagine, quindi il ping
corrispondente, la posizione del veicolo, l'assetto e l'altitudine. La
coordinata `centroid_x_px` viene invece confrontata con il nadir, cioe la colonna
centrale della waterfall, per stimare la distanza laterale dal veicolo.

La distanza in pixel viene convertita in slant range usando il range del sonar.
Poi viene calcolata la ground range sfruttando l'altitudine e Pitagora.

La posizione dell'oggetto viene espressa prima nel frame body del veicolo,
tenendo conto degli offset fisici del sensore SSS. Successivamente viene ruotata
nel frame NED usando roll, pitch e yaw di `NavStatus`, e infine convertita in
latitudine e longitudine tramite `ne2ll`.


Il nodo costruisce una lista di oggetti unici. Se una nuova detection ha
la stessa classe di un oggetto gia' salvato ed e' vicina a esso, viene
fusa con quell'oggetto: latitudine, longitudine e confidenza vengono 
aggiornate con una media incrementale, mentre `obs_count` viene aumentato.
Le detection fuori dalla safe zone vengono escluse.



L'output principale e:
src/sss_package/results/9_list_texts/SSS_object_list.json


Il file contiene `final_list`, cioe la lista finale degli oggetti SSS con
confidenza, numero di osservazioni, latitudine, longitudine e tipo. Alla chiusura
del nodo viene salvata anche una mappa finale con la traiettoria dell'AUV; se
viene fornito il parametro `real_objects_csv`, la mappa puo includere anche
oggetti reali di riferimento.


## `realtime_visualizer_node.py`

`realtime_visualizer_node.py` serve solo per monitoraggio live. Si iscrive a
`/waterfall_realtime_topic` e `/classified_objects_topic`, converte la waterfall
in immagine OpenCV e disegna sopra bounding box, centroidi, classe e confidenza
delle detection.

Le detection vengono memorizzate usando gli indici dei ping e non solo le righe
dell'immagine. In questo modo le bounding box restano allineate mentre la
waterfall realtime scorre.

Il nodo mostra anche lo stato di curva di Zeno usando i metadati SSS:

- stato sconosciuto;
- Zeno in curva;
- Zeno non in curva.

Questo nodo non genera la lista finale e non salva risultati di analisi: serve a
controllare visivamente se waterfall, detection e stato di curva sono coerenti
durante l'esecuzione.


## Note Pratiche

- Gli script sono scritti in ROS Python 2.7.
- Le cartelle `results/` contengono output generati.
- Il messaggio custom usato tra i nodi e' `sss_package/ImageMetadata`.
- La lista `SSS_object_list.json` viene poi usata da `src/comparazione.py` per
  fondere la lista finale SSS con quella FLS.
