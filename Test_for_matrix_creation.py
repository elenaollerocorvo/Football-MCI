# Script de prueba para validar extracción de características usando YOLO
# Realiza pruebas de detección de pose y objetos en imágenes/videos

# Importación de bibliotecas
from ultralytics import YOLO  # Framework YOLO para detección
import cv2  # OpenCV para procesamiento de imágenes
import torch  # PyTorch (backend de YOLO)
from math import *  # Funciones matemáticas

# Cargar modelos YOLO
pose = YOLO('yolov8m-pose.pt')  # Modelo de detección de pose humana
objekt_lopta = YOLO('yolov8x.pt')  # Modelo de detección de objetos

# Rutas a archivos de prueba
put_do_videa = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session 0\\jetsonCatch02_002320.avi"  # Video
put_do_framea = "C:\\Users\\Domagoj\\Desktop\\Yolo_data\\frame43.jpg"  # Imagen individual

# PRUEBA: Detección de pose en una imagen
inferece_obj = pose.predict(source=put_do_framea, verbose=False, save=True)
keypoints = inferece_obj[0].keypoints.xy.cpu().numpy().tolist()  # Extraer keypoints

# Verificar si se detectaron todos los keypoints necesarios
# 17 keypoints en total, verificar que no haya [0.0, 0.0] en posiciones críticas
if len(keypoints[0]) == 17 and [0.0,0.0] not in keypoints[0][10:17] and [0.0,0.0] not in keypoints[0][5:7]:
   print(keypoints, len(keypoints[0]))  # Imprimir keypoints válidos

# Líneas comentadas para otras pruebas
#inferece_pose = pose.predict(put_do_videa, verbose=False, save= True)  # Detección de pose en video
#inferece_lopte = objekt_lopta.predict(source=put_do_framea, conf=0.4, classes=[0,32])  # Detección de objetos
#tracking = objekt_lopta.track(source=put_do_videa, show=True, name='Tracking_test', persist=True, classes=[0,32], conf = 0.4, save=True)  # Tracking en video

exit()  # Salir del script

# SECCIÓN DE PRUEBAS EXTENDIDAS (requiere descomentar líneas anteriores)
# Extraer datos de detecciones
keypoints = inferece_pose[0].keypoints.xy.cpu().numpy().tolist()  # Keypoints de pose
coord_obj = inferece_lopte[0].boxes.xywh.cpu().tolist()  # Coordenadas de objetos (x, y, w, h)
bbx_wh = inferece_lopte[0].boxes.xywh.cpu().tolist()  # Dimensiones de bounding boxes
clss = inferece_lopte[0].boxes.cls.cpu().tolist()  # Clases de objetos detectados

# Clases de referencia en YOLO
reference_class_ball = 32.0  # Clase 32 = balón deportivo
reference_class_person = 0.0  # Clase 0 = persona

indx = 0  # Índice para iterar sobre detecciones
ball_found = False  # Flag para indicar si se encontró el balón

# Extraer keypoints específicos de la pierna derecha
skup_koordinata_desna_noga = keypoints[0][16]  # Tobillo derecho
skup_koordinata_desno_koljeno = keypoints[0][14]  # Rodilla derecha

# Procesar cada objeto detectado
for klasa in clss:
    # Si es un balón
    if klasa == reference_class_ball:
      ball_found = True
      coord_lopte = coord_obj[indx]  # Coordenadas del balón
      x_center_ball = coord_lopte[0]  # Centro X del balón
      y_center_ball = coord_lopte[1]  # Centro Y del balón
      visina_lopte = bbx_wh[indx]  # Dimensiones del balón
      
    # Si es una persona
    elif klasa == reference_class_person:
       visina_covj = bbx_wh[indx]  # Altura de la persona
       
    # Si llegamos al final sin encontrar balón
    elif not ball_found and len(clss) <= indx+1:
       print('nema lopte na slici', len(clss), indx)  # No hay balón en la imagen
    
    indx+=1  # Incrementar índice
    
# VISUALIZACIÓN: Preparar colores y coordenadas para dibujo
dot_radius = 10  # Radio del punto (no usado actualmente)
crvena = (0, 0, 255)  # Color rojo en formato BGR
plava = (255, 0, 0)   # Color azul en formato BGR
zelena = (0, 255, 0)  # Color verde en formato BGR

# Extraer coordenadas X, Y de keypoints
x_cord_noga = skup_koordinata_desna_noga[0]    # X del tobillo
y_cord_noga = skup_koordinata_desna_noga[1]    # Y del tobillo
x_cord_koljeno = skup_koordinata_desno_koljeno[0]  # X de la rodilla
y_cord_koljeno = skup_koordinata_desno_koljeno[1]  # Y de la rodilla

# Calcular punto medio del muslo (entre rodilla y cadera)
x_cord_tigh = (x_cord_koljeno + keypoints[0][12][0])/2
y_cord_tigh = (y_cord_koljeno + keypoints[0][12][1])/2

# Calcular punto del pecho (entre hombros)
chest_kp = [(keypoints[0][6][0] + keypoints[0][5][0])/2, (keypoints[0][6][1] + keypoints[0][5][1])/2]

# Calcular distancia euclidiana entre balón y tobillo
dist_ball_touch = int(sqrt((x_center_ball-x_cord_noga)**2 + (y_center_ball-y_cord_noga)**2))

# Calcular ratio de alturas (persona/balón)
ball_to_person_ratio = visina_covj[3]/visina_lopte[3]

# Obtener resolución del video
video_capture = cv2.VideoCapture(put_do_videa)
original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_capture.release()

print(original_width, original_height)

# VISUALIZACIÓN: Dibujar líneas sobre la imagen para verificar detecciones
image = cv2.imread(put_do_framea)

# Línea roja: del balón al pie
cv2.line(image, (int(x_center_ball), int(y_center_ball)), (int(x_cord_noga), int(y_cord_noga)), crvena, 2)

# Línea azul: del pecho al pie
cv2.line(image, (int(chest_kp[0]), int(chest_kp[1])), (int(x_cord_noga), int(y_cord_noga)), plava, 2)

# Línea verde: altura de la persona
cv2.line(image, (int(visina_covj[0]), int(visina_covj[1]- visina_covj[3]/2)), (int(visina_covj[0]), int(visina_covj[1]+ visina_covj[3]/2)), zelena, 2)

# Línea verde: altura del balón
cv2.line(image, (int(visina_lopte[0]), int(visina_lopte[1]- visina_lopte[3]/2)), (int(visina_lopte[0]), int(visina_lopte[1]+ visina_lopte[3]/2)), zelena, 2)

# Mostrar imagen con anotaciones
cv2.imshow("Image with Dot", image)
cv2.waitKey(0)  # Esperar tecla para cerrar
cv2.destroyAllWindows()  # Cerrar ventanas

#print(bbx_wh)

#print(x_cord, '\n',y_cord)


# vidcap = cv2.VideoCapture(put_do_videa)
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1


