# Listing B.1 : Feature extraction
# Script que extrae características de videos de fútbol usando YOLO
# Detecta personas, sus keypoints corporales y el balón para crear una matriz de datos

# Importación de bibliotecas necesarias
from ultralytics import YOLO  # Framework YOLO para detección de objetos y pose
import cv2  # OpenCV para procesamiento de video
from math import *  # Funciones matemáticas
import os
import json  # Para leer archivos de etiquetas
import numpy as np
import tempfile
from collections import defaultdict  # Diccionarios con valores por defecto
import re  # Expresiones regulares
import pandas as pd  # Para crear DataFrames y exportar CSV

# Cargar modelos YOLO pre-entrenados
pose = YOLO('yolov8m-pose.pt')  # Modelo para detección de pose humana (keypoints)
obj_det = YOLO('yolov8x.pt')  # Modelo para detección de objetos (persona y balón)

# Rutas a las carpetas de entrada
video_folder_path = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session 25"
json_folder_path  = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Json\\Session 25 json"
video_folder = os.listdir(video_folder_path)  # Lista de archivos de video

# Matriz principal para almacenar todas las características extraídas
panda_matrix = []

# Codificación one-hot para las clases de toque
one_hot_enc_class = {"RightF":[1,0,0,0,0,0],  # Pie derecho
                     "LeftF":[0,1,0,0,0,0],   # Pie izquierdo
                     "RightT":[0,0,1,0,0,0],  # Muslo derecho
                     "LeftT":[0,0,0,1,0,0],   # Muslo izquierdo
                     "Chest":[0,0,0,0,1,0],   # Pecho
                     "Other":[0,0,0,0,0,1]}   # Otro (sin toque)

# Contadores para videos con problemas
no_touch_counter = 0  # Videos sin toques etiquetados
ball_missing = 0  # Videos donde se pierde el balón
people_overload = 0  # Videos con más de una persona
bad_video_list = []  # Lista de IDs de videos con datos problemáticos


# Contador de videos procesados
video_counter = 1

# Bucle principal: procesar cada video en la carpeta
for video_path in video_folder:
    
    # Filtrar solo archivos .avi
    if not video_path.endswith('.avi'):
        continue
    print(video_path)
    
    # Cargar archivo JSON con las etiquetas del video
    json_name = '.'.join([video_path.split('.')[0],'json'])
    json_file_path = os.path.join(json_folder_path, json_name)
    json_file = open(json_file_path, 'r')
    json_data = json.load(json_file)
    button_presses = json_data['button_presses']  # String con formato "Clase:Frame;Clase:Frame;..."
    
    # Saltar videos sin toques etiquetados
    if button_presses == "" or button_presses == " ":
        no_touch_counter += 1
        continue
    
    # Extraer información del primer y último toque
    first_touch_string = button_presses.split(";")[0]
    frame_of_first_touch = int(first_touch_string.split(":")[1])  # Frame del primer toque
    class_of_first_touch = first_touch_string.split(":")[0]  # Clase del primer toque
    last_touch_string = button_presses.split(";")[-1]
    frame_of_last_touch = int(last_touch_string.split(":")[1])  # Frame del último toque
    
    # Crear diccionario {frame: clase} con todos los toques
    list_of_clss_frame = re.split(':|;', button_presses)
    dict_class_frame = dict()
    for i in range(1, len(list_of_clss_frame), 2):
                dict_class_frame[int(list_of_clss_frame[i])] = list_of_clss_frame[i-1]


    # Abrir el video para procesamiento frame por frame
    cap = cv2.VideoCapture(os.path.join(video_folder_path,video_path))

    # Variables para tracking de objetos a lo largo del video
    track_history_ball = defaultdict(lambda: [])  # Historial de posiciones del balón por ID
    track_history_person = []  # Historial de detecciones de persona
    active_ball = None  # ID del balón activo (el que interactúa con la persona)
    bad_video = False  # Flag para marcar videos con problemas
    active_ball_dict = dict()  # Diccionario {ball_id: frame_inicio}
    ball_lost_counter = 0  # Contador de frames sin detectar el balón activo
    
    # Bucle de procesamiento frame por frame
    while cap.isOpened():
        # Leer siguiente frame del video
        success, frame = cap.read()
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames
        first_active_ball_found = False

        if success:
            # Ejecutar detección y tracking de objetos (persona clase=0, balón clase=32)
            results = obj_det.track(frame, persist=True, classes = [0,32], verbose = False, conf = 0.4)
            clss = results[0].boxes.cls.cpu().tolist()  # Lista de clases detectadas
            
            # Ejecutar detección de pose humana (17 keypoints corporales)
            human_pose = pose.predict(frame, verbose=False)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Número del frame actual

            # Extraer coordenadas de keypoints corporales
            keypoints = human_pose[0].keypoints.xy.cpu().numpy().tolist()

            # Inicializar posiciones de partes del cuerpo con valor por defecto (2000,2000)
            # Este valor indica que no se detectó esa parte del cuerpo
            right_foot = (2000,2000)  # Pie derecho
            left_foot = (2000,2000)   # Pie izquierdo
            right_thigh = (2000,2000) # Muslo derecho
            left_thigh = (2000,2000)  # Muslo izquierdo
            chest  = (2000,2000)      # Pecho
            
            # Si se detectaron todos los keypoints necesarios
            if len(keypoints[0]) == 17 and [0.0,0.0] not in keypoints[0][11:17] and [0.0,0.0] not in keypoints[0][5:7]:
                right_foot = keypoints[0][16]  # Keypoint 16: tobillo derecho
                left_foot = keypoints[0][15]   # Keypoint 15: tobillo izquierdo
                # Muslo derecho: promedio entre rodilla (14) y cadera (12)
                right_thigh = [(keypoints[0][14][0] + keypoints[0][12][0])/2, (keypoints[0][14][1] + keypoints[0][12][1])/2]
                # Muslo izquierdo: promedio entre rodilla (13) y cadera (11)
                left_thigh = [(keypoints[0][13][0] + keypoints[0][11][0])/2, (keypoints[0][13][1] + keypoints[0][11][1])/2]
                # Pecho: promedio entre hombros (6 y 5)
                chest = [(keypoints[0][6][0] + keypoints[0][5][0])/2, (keypoints[0][6][1] + keypoints[0][5][1])/2]
            
            person_found = False  # Flag para indicar si ya se encontró una persona
            indx = 0  # Índice para iterar sobre las detecciones
            # if active_ball != None and current_frame <= frame_of_last_touch and active_ball not in results[0].boxes.id.int().cpu().tolist():
            #     bad_video = True
            #     ball_missing += 1
            #     cap.release()
            #     break
            active_ball_found = False  # Flag para saber si se detectó el balón activo en este frame
            
            # Procesar cada objeto detectado en el frame
            for obj in clss:
                try:
                    # Si es una persona (clase 0) y aún no se ha encontrado ninguna
                    if obj == 0 and not person_found:
                        keypoints = human_pose[0].keypoints.xy.cpu().numpy().tolist()
                        person_height = results[0].boxes.xywh.cpu().tolist()[indx][3]  # Altura del bounding box
                        # Guardar keypoints, altura y frame en el historial
                        track_history_person.append((keypoints[0], person_height, current_frame))
                        person_found = True
                        
                    # Si es un balón (clase 32)
                    elif obj == 32:
                        boxes = results[0].boxes.xywh.cpu()[indx]  # Coordenadas del bounding box
                        track_ids = results[0].boxes.id.int().cpu().tolist()[indx]  # ID único del tracking
                        
                        # Verificar si es el balón activo
                        if track_ids == active_ball:
                            active_ball_found = True
                            
                        x, y, w, h = boxes  # Centro (x,y), ancho y alto del bounding box
                        track = track_history_ball[track_ids]
                        track.append((float(x), float(y), float(h), current_frame))  # Guardar posición y frame
                        
                    # Si se detecta una segunda persona en el primer frame, marcar video como malo
                    elif obj == 0 and person_found and current_frame == 1:
                        bad_video = True
                        people_overload += 1
                        cap.release()
                        break
                except:
                    pass
                indx+=1  # Incrementar índice para siguiente detección
                
            if not active_ball_found and first_active_ball_found:
                last_ball_id =  track_history_ball.keys()[-1]
                ball_lost_counter += 1
                if last_ball_id != active_ball and track_history_ball[last_ball_id][0][-1] > track_history_ball[active_ball][-1][-1]:
                    new_ball_added_x = track_history_ball[last_ball_id][-1][0]
                    new_ball_added_y = track_history_ball[last_ball_id][-1][1]
                    new_ball_added_h = track_history_ball[last_ball_id][-1][2]
                    old_ball_added_x = track_history_ball[active_ball][-1][0]
                    old_ball_added_y = track_history_ball[active_ball][-1][1]
                    old_ball_added_h = track_history_ball[active_ball][-1][2]
                    if abs(new_ball_added_x - old_ball_added_x)<10 and abs(new_ball_added_y - old_ball_added_y)<10 and abs(new_ball_added_h - old_ball_added_h)<5:
                        active_ball = last_ball_id
                        active_ball_dict[active_ball] = current_frame
                        ball_lost_counter = 0
                
                if ball_lost_counter == 10:
                    ball_missing += 1
                    bad_video = True
                    cap.release()
                    break
            # En el frame del primer toque, identificar cuál balón es el activo
            if current_frame == frame_of_first_touch:
                min_distance = 820  # Distancia mínima inicial (valor grande)
                
                # Determinar qué parte del cuerpo tocó el balón
                if 'RightF' in class_of_first_touch:
                    first_touch = right_foot
                elif 'LeftF' in class_of_first_touch:
                    first_touch = left_foot
                elif 'RightT' in class_of_first_touch:
                    first_touch = right_thigh
                elif 'LeftT' in class_of_first_touch:
                    first_touch = left_thigh
                elif 'Chest' in class_of_first_touch:
                    first_touch = chest
                
                # Si solo hay un balón detectado, ese es el activo
                if len(list(track_history_ball.keys())) == 1:
                    active_ball = list(track_history_ball.keys())[0]
                else:
                    # Si hay múltiples balones, encontrar el más cercano al punto de toque
                    for key in track_history_ball.keys():
                        # Ignorar balones que no se han visto recientemente
                        if track_history_ball[key][-1][-1] < current_frame-3:
                            continue
                        # Calcular distancia euclidiana entre balón y punto de toque
                        x_diff = track_history_ball[key][-1][0] - first_touch[0]
                        y_diff = track_history_ball[key][-1][1] - first_touch[1]
                        distance = sqrt(x_diff**2 + y_diff**2)
                        # El balón más cercano es el activo
                        if distance < min_distance:
                            active_ball = key
                            min_distance = distance
                            first_active_ball_found = True
                
                # Si no se pudo identificar el balón activo, marcar video como malo
                if active_ball == None:
                    ball_missing += 1
                    bad_video = True
                    cap.release()
                    break
                else:
                    active_ball_dict[active_ball] = current_frame  # Registrar inicio del balón activo
        else:
            # Break the loop if the end of the video is reached
            break
        
    cap.release()
    
    # Si el video tuvo problemas, saltarlo y continuar con el siguiente
    if bad_video:
        continue

    # Variables para construir la matriz de características frame por frame
    person_counter = 0  # Índice actual en track_history_person
    ball_counter = 0    # Índice actual en track_history_ball para el balón activo
    last_ball_frame = 0
    bad_data_touch_counter = 0  # Contador de frames con toque pero sin datos
    keys = list(active_ball_dict.keys())  # Lista de IDs de balones activos
    active_ball_position = 0  # Índice del balón activo actual
    active_ball = keys[active_ball_position]  # ID del balón activo actual
    
    # Construir matriz de características para cada frame del video
    for frame_counter in range(1, frame_total+1):
        if active_ball != keys[-1] and frame_counter == active_ball_dict[keys[active_ball_position+1]]:
            active_ball_position += 1
            active_ball = keys[active_ball_position]
            ball_counter = 0
        # Obtener datos del balón y persona para el frame actual
        ball_data = track_history_ball[active_ball][ball_counter]
        person_data = track_history_person[person_counter]
        latest_active_ball_frame = ball_data[-1]  # Frame del último dato del balón
        latest_active_person_frame = person_data[-1]  # Frame del último dato de la persona
        
        # Inicializar variables de características con valores por defecto
        x_center_ball = None
        y_center_ball = None
        ball_height = None
        person_height = None
        dist_ball_rfoot = 820   # Distancia balón-pie derecho (820 = no disponible)
        dist_ball_lfoot = 820   # Distancia balón-pie izquierdo
        dist_ball_rthigh = 820  # Distancia balón-muslo derecho
        dist_ball_lthigh = 820  # Distancia balón-muslo izquierdo
        dist_ball_chest = 820   # Distancia balón-pecho
        height_ratio = 0        # Ratio altura persona/altura balón
        
        # Si tenemos datos válidos tanto del balón como de la persona en este frame
        if latest_active_ball_frame == frame_counter and latest_active_person_frame == frame_counter and len(person_data[0]) == 17 and [0.0,0.0] not in person_data[0][11:17] and [0.0,0.0] not in person_data[0][5:7]:
            # Extraer coordenadas del balón
            x_center_ball = ball_data[0]
            y_center_ball = ball_data[1]
            ball_height = ball_data[2]
            
            # Extraer keypoints relevantes de la persona
            right_foot_kp = person_data[0][16]
            left_foot_kp = person_data[0][15]
            right_thigh_kp = [(person_data[0][14][0] + person_data[0][12][0])/2, (person_data[0][14][1] + person_data[0][12][1])/2]
            left_thigh_kp = [(person_data[0][13][0] + person_data[0][11][0])/2, (person_data[0][13][1] + person_data[0][11][1])/2]
            chest_kp = [(person_data[0][6][0] + person_data[0][5][0])/2, (person_data[0][6][1] + person_data[0][5][1])/2]

            # Calcular distancias euclidianas entre el balón y cada parte del cuerpo
            dist_ball_rfoot = int(sqrt((x_center_ball-right_foot_kp[0])**2 + (y_center_ball-right_foot_kp[1])**2))
            dist_ball_lfoot = int(sqrt((x_center_ball-left_foot_kp[0])**2 + (y_center_ball-left_foot_kp[1])**2))
            dist_ball_rthigh = int(sqrt((x_center_ball-right_thigh_kp[0])**2 + (y_center_ball-right_thigh_kp[1])**2))
            dist_ball_lthigh = int(sqrt((x_center_ball-left_thigh_kp[0])**2 + (y_center_ball-left_thigh_kp[1])**2))
            dist_ball_chest = int(sqrt((x_center_ball-chest_kp[0])**2 + (y_center_ball-chest_kp[1])**2))
            
            # Calcular ratio de alturas (normalización)
            height_ratio = person_data[1]/ball_height

        # Verificar si hay un toque en un rango de ±3 frames del frame actual
        recognition_of_touch = False
        for i in range(-3,4):
            recognition_of_touch = recognition_of_touch or (frame_counter+i in dict_class_frame.keys())
            if recognition_of_touch:
                break
        
        # Si hay toque pero no hay datos válidos, incrementar contador de mal dato
        if recognition_of_touch and dist_ball_rfoot == 820:
            bad_data_touch_counter += 1
            # Si hay 7 frames consecutivos con este problema, marcar video como malo
            if bad_data_touch_counter == 7:
                bad_video_list.append(video_counter)
        else:
            bad_data_touch_counter = 0
        
        # Añadir fila a la matriz principal con las características y la etiqueta
        if recognition_of_touch:
            # Si hay toque, usar la clase etiquetada
            panda_matrix.append([video_counter, frame_counter, dist_ball_rfoot, dist_ball_lfoot, dist_ball_rthigh, dist_ball_lthigh, dist_ball_chest, height_ratio] + one_hot_enc_class[(dict_class_frame[frame_counter + i]).strip(" ")])
        else:
            # Si no hay toque, etiquetar como "Other"
            panda_matrix.append([video_counter, frame_counter, dist_ball_rfoot, dist_ball_lfoot, dist_ball_rthigh, dist_ball_lthigh, dist_ball_chest, height_ratio] + one_hot_enc_class["Other"])


        if latest_active_person_frame == frame_counter and person_counter < len(track_history_person)-1:
            person_counter += 1
        if latest_active_ball_frame == frame_counter and ball_counter < len(track_history_ball[active_ball])-1:
            ball_counter += 1    

        
    print("video number: ", video_counter, " has finished succesfully")
    video_counter += 1  # Incrementar contador de videos

    
# Crear DataFrame de pandas con todas las características extraídas
# Columnas: ID de video, número de frame, distancias a cada parte del cuerpo,
# ratio de alturas, y codificación one-hot de la clase
column_names = ["video_id", "frame_no", "distance_to_RF","distance_to_LF","distance_to_RT", "distance_to_LT","distance_to_CH", "person_ball_H_rt", "RightF", "LeftF", "RightT", "LeftT", "Chest", "Other"]
data_frame = pd.DataFrame(panda_matrix, columns=column_names)

# Filtrar videos con datos problemáticos
df_filtered = data_frame[~data_frame['video_id'].isin(bad_video_list)]

# Exportar DataFrame a archivo CSV
df_filtered.to_csv("C:\\Users\\Domagoj\\Desktop\\Diplomski\\Codes\\New_data_Session25_chest_filtered.csv", index=False)

# Imprimir estadísticas de videos problemáticos
print("bad video conts: ", "ball lost: ", ball_missing, "\npeople overload: ", people_overload, "\nno touches: ", no_touch_counter, "\ntouch happend but data is missing:", len(bad_video_list))
