# Test script to validate feature extraction using YOLO
# Performs pose and object detection tests on images/videos

# Import libraries
from ultralytics import YOLO  # YOLO framework for detection
import cv2  # OpenCV for image processing
import torch  # PyTorch (YOLO backend)
from math import * # Mathematical functions

# Load YOLO models
pose = YOLO('yolov8m-pose.pt')  # Human pose detection model
objekt_lopta = YOLO('yolov8x.pt')  # Object detection model

# Paths to test files
put_do_videa = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session 0\\jetsonCatch02_002320.avi"  # Video
put_do_framea = "C:\\Users\\Domagoj\\Desktop\\Yolo_data\\frame43.jpg"  # Single image

# TEST: Pose detection on an image
inferece_obj = pose.predict(source=put_do_framea, verbose=False, save=True)
keypoints = inferece_obj[0].keypoints.xy.cpu().numpy().tolist()  # Extract keypoints

# Check if all necessary keypoints were detected
# 17 keypoints in total, verify that there are no [0.0, 0.0] in critical positions
if len(keypoints[0]) == 17 and [0.0,0.0] not in keypoints[0][10:17] and [0.0,0.0] not in keypoints[0][5:7]:
   print(keypoints, len(keypoints[0]))  # Print valid keypoints

# Commented lines for other tests
#inferece_pose = pose.predict(put_do_videa, verbose=False, save= True)  # Pose detection in video
#inferece_lopte = objekt_lopta.predict(source=put_do_framea, conf=0.4, classes=[0,32])  # Object detection
#tracking = objekt_lopta.track(source=put_do_videa, show=True, name='Tracking_test', persist=True, classes=[0,32], conf = 0.4, save=True)  # Tracking in video

exit()  # Exit the script

# EXTENDED TESTS SECTION (requires uncommenting previous lines)
# Extract data from detections
keypoints = inferece_pose[0].keypoints.xy.cpu().numpy().tolist()  # Pose keypoints
coord_obj = inferece_lopte[0].boxes.xywh.cpu().tolist()  # Object coordinates (x, y, w, h)
bbx_wh = inferece_lopte[0].boxes.xywh.cpu().tolist()  # Bounding boxes dimensions
clss = inferece_lopte[0].boxes.cls.cpu().tolist()  # Detected object classes

# Reference classes in YOLO
reference_class_ball = 32.0  # Class 32 = sports ball
reference_class_person = 0.0  # Class 0 = person

indx = 0  # Index to iterate over detections
ball_found = False  # Flag to indicate if the ball was found

# Extract specific keypoints of the right leg
skup_koordinata_desna_noga = keypoints[0][16]  # Right ankle
skup_koordinata_desno_koljeno = keypoints[0][14]  # Right knee

# Process each detected object
for klasa in clss:
    # If it is a ball
    if klasa == reference_class_ball:
      ball_found = True
      coord_lopte = coord_obj[indx]  # Ball coordinates
      x_center_ball = coord_lopte[0]  # Ball X center
      y_center_ball = coord_lopte[1]  # Ball Y center
      visina_lopte = bbx_wh[indx]  # Ball dimensions
      
    # If it is a person
    elif klasa == reference_class_person:
       visina_covj = bbx_wh[indx]  # Person height
       
    # If we reach the end without finding a ball
    elif not ball_found and len(clss) <= indx+1:
       print('nema lopte na slici', len(clss), indx)  # No ball in the image
    
    indx+=1  # Increment index
    
# VISUALIZATION: Prepare colors and coordinates for drawing
dot_radius = 10  # Dot radius (currently not used)
crvena = (0, 0, 255)  # Red color in BGR format
plava = (255, 0, 0)   # Blue color in BGR format
zelena = (0, 255, 0)  # Green color in BGR format

# Extract X, Y coordinates of keypoints
x_cord_noga = skup_koordinata_desna_noga[0]    # Ankle X
y_cord_noga = skup_koordinata_desna_noga[1]    # Ankle Y
x_cord_koljeno = skup_koordinata_desno_koljeno[0]  # Knee X
y_cord_koljeno = skup_koordinata_desno_koljeno[1]  # Knee Y

# Calculate middle point of the thigh (between knee and hip)
x_cord_tigh = (x_cord_koljeno + keypoints[0][12][0])/2
y_cord_tigh = (y_cord_koljeno + keypoints[0][12][1])/2

# Calculate chest point (between shoulders)
chest_kp = [(keypoints[0][6][0] + keypoints[0][5][0])/2, (keypoints[0][6][1] + keypoints[0][5][1])/2]

# Calculate Euclidean distance between ball and ankle
dist_ball_touch = int(sqrt((x_center_ball-x_cord_noga)**2 + (y_center_ball-y_cord_noga)**2))

# Calculate height ratio (person/ball)
ball_to_person_ratio = visina_covj[3]/visina_lopte[3]

# Get video resolution
video_capture = cv2.VideoCapture(put_do_videa)
original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_capture.release()

print(original_width, original_height)

# VISUALIZATION: Draw lines on the image to verify detections
image = cv2.imread(put_do_framea)

# Red line: from ball to foot
cv2.line(image, (int(x_center_ball), int(y_center_ball)), (int(x_cord_noga), int(y_cord_noga)), crvena, 2)

# Blue line: from chest to foot
cv2.line(image, (int(chest_kp[0]), int(chest_kp[1])), (int(x_cord_noga), int(y_cord_noga)), plava, 2)

# Green line: person height
cv2.line(image, (int(visina_covj[0]), int(visina_covj[1]- visina_covj[3]/2)), (int(visina_covj[0]), int(visina_covj[1]+ visina_covj[3]/2)), zelena, 2)

# Green line: ball height
cv2.line(image, (int(visina_lopte[0]), int(visina_lopte[1]- visina_lopte[3]/2)), (int(visina_lopte[0]), int(visina_lopte[1]+ visina_lopte[3]/2)), zelena, 2)

# Show image with annotations
cv2.imshow("Image with Dot", image)
cv2.waitKey(0)  # Wait for key to close
cv2.destroyAllWindows()  # Close windows

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
