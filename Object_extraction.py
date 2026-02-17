# Listing B.1 : Feature extraction
# Script that extracts features from football (soccer) videos using YOLO
# Detects people, their body keypoints, and the ball to create a data matrix

# Import necessary libraries
from ultralytics import YOLO  # YOLO framework for object and pose detection
import cv2  # OpenCV for video processing
from math import * # Mathematical functions
import os
import json  # To read label files
import numpy as np
import tempfile
from collections import defaultdict  # Dictionaries with default values
import re  # Regular expressions
import pandas as pd  # To create DataFrames and export to CSV

# Load pre-trained YOLO models
pose = YOLO('yolov8m-pose.pt')  # Model for human pose detection (keypoints)
obj_det = YOLO('yolov8x.pt')  # Model for object detection (person and ball)

# Paths to input folders
video_folder_path = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Videos\\Session 25"
json_folder_path  = "C:\\Users\\Domagoj\\Desktop\\Diplomski\\Json\\Session 25 json"
video_folder = os.listdir(video_folder_path)  # List of video files

# Main matrix to store all extracted features
panda_matrix = []

# One-hot encoding for touch classes
one_hot_enc_class = {"RightF":[1,0,0,0,0,0],  # Right foot
                     "LeftF":[0,1,0,0,0,0],   # Left foot
                     "RightT":[0,0,1,0,0,0],  # Right thigh
                     "LeftT":[0,0,0,1,0,0],   # Left thigh
                     "Chest":[0,0,0,0,1,0],   # Chest
                     "Other":[0,0,0,0,0,1]}   # Other (no touch)

# Counters for problematic videos
no_touch_counter = 0  # Videos with no labeled touches
ball_missing = 0  # Videos where the ball is lost
people_overload = 0  # Videos with more than one person
bad_video_list = []  # List of video IDs with problematic data

# Processed video counter
video_counter = 1

# Main loop: process each video in the folder
for video_path in video_folder:
    
    # Filter only .avi files
    if not video_path.endswith('.avi'):
        continue
    print(video_path)
    
    # Load JSON file with video labels
    json_name = '.'.join([video_path.split('.')[0],'json'])
    json_file_path = os.path.join(json_folder_path, json_name)
    json_file = open(json_file_path, 'r')
    json_data = json.load(json_file)
    button_presses = json_data['button_presses']  # String formatted as "Class:Frame;Class:Frame;..."
    
    # Skip videos with no labeled touches
    if button_presses == "" or button_presses == " ":
        no_touch_counter += 1
        continue
    
    # Extract information about the first and last touch
    first_touch_string = button_presses.split(";")[0]
    frame_of_first_touch = int(first_touch_string.split(":")[1])  # Frame of the first touch
    class_of_first_touch = first_touch_string.split(":")[0]  # Class of the first touch
    last_touch_string = button_presses.split(";")[-1]
    frame_of_last_touch = int(last_touch_string.split(":")[1])  # Frame of the last touch
    
    # Create dictionary {frame: class} with all touches
    list_of_clss_frame = re.split(':|;', button_presses)
    dict_class_frame = dict()
    for i in range(1, len(list_of_clss_frame), 2):
        dict_class_frame[int(list_of_clss_frame[i])] = list_of_clss_frame[i-1]


    # Open the video for frame-by-frame processing
    cap = cv2.VideoCapture(os.path.join(video_folder_path,video_path))

    # Variables for object tracking throughout the video
    track_history_ball = defaultdict(lambda: [])  # History of ball positions by ID
    track_history_person = []  # History of person detections
    active_ball = None  # ID of the active ball (the one interacting with the person)
    bad_video = False  # Flag to mark problematic videos
    active_ball_dict = dict()  # Dictionary {ball_id: start_frame}
    ball_lost_counter = 0  # Counter for frames without detecting the active ball
    
    # Frame-by-frame processing loop
    while cap.isOpened():
        # Read next frame of the video
        success, frame = cap.read()
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
        first_active_ball_found = False

        if success:
            # Run object detection and tracking (person class=0, ball class=32)
            results = obj_det.track(frame, persist=True, classes=[0,32], verbose=False, conf=0.4)
            clss = results[0].boxes.cls.cpu().tolist()  # List of detected classes
            
            # Run human pose detection (17 body keypoints)
            human_pose = pose.predict(frame, verbose=False)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame number

            # Extract body keypoint coordinates
            keypoints = human_pose[0].keypoints.xy.cpu().numpy().tolist()

            # Initialize body part positions with default value (2000,2000)
            # This value indicates that the body part was not detected
            right_foot = (2000,2000)  # Right foot
            left_foot = (2000,2000)   # Left foot
            right_thigh = (2000,2000) # Right thigh
            left_thigh = (2000,2000)  # Left thigh
            chest  = (2000,2000)      # Chest
            
            # If all necessary keypoints were detected
            if len(keypoints[0]) == 17 and [0.0,0.0] not in keypoints[0][11:17] and [0.0,0.0] not in keypoints[0][5:7]:
                right_foot = keypoints[0][16]  # Keypoint 16: right ankle
                left_foot = keypoints[0][15]   # Keypoint 15: left ankle
                # Right thigh: average between knee (14) and hip (12)
                right_thigh = [(keypoints[0][14][0] + keypoints[0][12][0])/2, (keypoints[0][14][1] + keypoints[0][12][1])/2]
                # Left thigh: average between knee (13) and hip (11)
                left_thigh = [(keypoints[0][13][0] + keypoints[0][11][0])/2, (keypoints[0][13][1] + keypoints[0][11][1])/2]
                # Chest: average between shoulders (6 and 5)
                chest = [(keypoints[0][6][0] + keypoints[0][5][0])/2, (keypoints[0][6][1] + keypoints[0][5][1])/2]
            
            person_found = False  # Flag indicating if a person has already been found
            indx = 0  # Index to iterate over detections
            # if active_ball != None and current_frame <= frame_of_last_touch and active_ball not in results[0].boxes.id.int().cpu().tolist():
            #     bad_video = True
            #     ball_missing += 1
            #     cap.release()
            #     break
            active_ball_found = False  # Flag to know if the active ball was detected in this frame
            
            # Process each detected object in the frame
            for obj in clss:
                try:
                    # If it's a person (class 0) and none has been found yet
                    if obj == 0 and not person_found:
                        keypoints = human_pose[0].keypoints.xy.cpu().numpy().tolist()
                        person_height = results[0].boxes.xywh.cpu().tolist()[indx][3]  # Bounding box height
                        # Save keypoints, height, and frame to history
                        track_history_person.append((keypoints[0], person_height, current_frame))
                        person_found = True
                        
                    # If it's a ball (class 32)
                    elif obj == 32:
                        boxes = results[0].boxes.xywh.cpu()[indx]  # Bounding box coordinates
                        track_ids = results[0].boxes.id.int().cpu().tolist()[indx]  # Unique tracking ID
                        
                        # Check if it's the active ball
                        if track_ids == active_ball:
                            active_ball_found = True
                            
                        x, y, w, h = boxes  # Center (x,y), width, and height of bounding box
                        track = track_history_ball[track_ids]
                        track.append((float(x), float(y), float(h), current_frame))  # Save position and frame
                        
                    # If a second person is detected in the first frame, mark video as bad
                    elif obj == 0 and person_found and current_frame == 1:
                        bad_video = True
                        people_overload += 1
                        cap.release()
                        break
                except:
                    pass
                indx+=1  # Increment index for next detection
                
            if not active_ball_found and first_active_ball_found:
                last_ball_id = list(track_history_ball.keys())[-1]
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
                
            # In the frame of the first touch, identify which ball is active
            if current_frame == frame_of_first_touch:
                min_distance = 820  # Initial minimum distance (large value)
                
                # Determine which body part touched the ball
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
                
                # If only one ball is detected, that's the active one
                if len(list(track_history_ball.keys())) == 1:
                    active_ball = list(track_history_ball.keys())[0]
                else:
                    # If there are multiple balls, find the closest one to the touch point
                    for key in track_history_ball.keys():
                        # Ignore balls that haven't been seen recently
                        if track_history_ball[key][-1][-1] < current_frame-3:
                            continue
                        # Calculate Euclidean distance between ball and touch point
                        x_diff = track_history_ball[key][-1][0] - first_touch[0]
                        y_diff = track_history_ball[key][-1][1] - first_touch[1]
                        distance = sqrt(x_diff**2 + y_diff**2)
                        # The closest ball is the active one
                        if distance < min_distance:
                            active_ball = key
                            min_distance = distance
                            first_active_ball_found = True
                
                # If the active ball couldn't be identified, mark video as bad
                if active_ball == None:
                    ball_missing += 1
                    bad_video = True
                    cap.release()
                    break
                else:
                    active_ball_dict[active_ball] = current_frame  # Register start of the active ball
        else:
            # Break the loop if the end of the video is reached
            break
        
    cap.release()
    
    # If the video had issues, skip it and continue to the next one
    if bad_video:
        continue

    # Variables to build the feature matrix frame by frame
    person_counter = 0  # Current index in track_history_person
    ball_counter = 0    # Current index in track_history_ball for the active ball
    last_ball_frame = 0
    bad_data_touch_counter = 0  # Counter for frames with touch but no data
    keys = list(active_ball_dict.keys())  # List of active ball IDs
    active_ball_position = 0  # Index of current active ball
    active_ball = keys[active_ball_position]  # ID of current active ball
    
    # Build feature matrix for each frame of the video
    for frame_counter in range(1, frame_total+1):
        if active_ball != keys[-1] and frame_counter == active_ball_dict[keys[active_ball_position+1]]:
            active_ball_position += 1
            active_ball = keys[active_ball_position]
            ball_counter = 0
            
        # Get ball and person data for the current frame
        ball_data = track_history_ball[active_ball][ball_counter]
        person_data = track_history_person[person_counter]
        latest_active_ball_frame = ball_data[-1]  # Frame of the last ball data
        latest_active_person_frame = person_data[-1]  # Frame of the last person data
        
        # Initialize feature variables with default values
        x_center_ball = None
        y_center_ball = None
        ball_height = None
        person_height = None
        dist_ball_rfoot = 820   # Ball-right foot distance (820 = unavailable)
        dist_ball_lfoot = 820   # Ball-left foot distance
        dist_ball_rthigh = 820  # Ball-right thigh distance
        dist_ball_lthigh = 820  # Ball-left thigh distance
        dist_ball_chest = 820   # Ball-chest distance
        height_ratio = 0        # Person height / ball height ratio
        
        # If we have valid data for both the ball and person in this frame
        if latest_active_ball_frame == frame_counter and latest_active_person_frame == frame_counter and len(person_data[0]) == 17 and [0.0,0.0] not in person_data[0][11:17] and [0.0,0.0] not in person_data[0][5:7]:
            # Extract ball coordinates
            x_center_ball = ball_data[0]
            y_center_ball = ball_data[1]
            ball_height = ball_data[2]
            
            # Extract relevant person keypoints
            right_foot_kp = person_data[0][16]
            left_foot_kp = person_data[0][15]
            right_thigh_kp = [(person_data[0][14][0] + person_data[0][12][0])/2, (person_data[0][14][1] + person_data[0][12][1])/2]
            left_thigh_kp = [(person_data[0][13][0] + person_data[0][11][0])/2, (person_data[0][13][1] + person_data[0][11][1])/2]
            chest_kp = [(person_data[0][6][0] + person_data[0][5][0])/2, (person_data[0][6][1] + person_data[0][5][1])/2]

            # Calculate Euclidean distances between ball and each body part
            dist_ball_rfoot = int(sqrt((x_center_ball-right_foot_kp[0])**2 + (y_center_ball-right_foot_kp[1])**2))
            dist_ball_lfoot = int(sqrt((x_center_ball-left_foot_kp[0])**2 + (y_center_ball-left_foot_kp[1])**2))
            dist_ball_rthigh = int(sqrt((x_center_ball-right_thigh_kp[0])**2 + (y_center_ball-right_thigh_kp[1])**2))
            dist_ball_lthigh = int(sqrt((x_center_ball-left_thigh_kp[0])**2 + (y_center_ball-left_thigh_kp[1])**2))
            dist_ball_chest = int(sqrt((x_center_ball-chest_kp[0])**2 + (y_center_ball-chest_kp[1])**2))
            
            # Calculate height ratio (normalization)
            height_ratio = person_data[1]/ball_height

        # Check if there is a touch within a range of ±3 frames of the current frame
        recognition_of_touch = False
        for i in range(-3,4):
            recognition_of_touch = recognition_of_touch or (frame_counter+i in dict_class_frame.keys())
            if recognition_of_touch:
                break
        
        # If there is a touch but no valid data, increment bad data counter
        if recognition_of_touch and dist_ball_rfoot == 820:
            bad_data_touch_counter += 1
            # If there are 7 consecutive frames with this problem, mark video as bad
            if bad_data_touch_counter == 7:
                bad_video_list.append(video_counter)
        else:
            bad_data_touch_counter = 0
        
        # Add row to the main matrix with features and label
        if recognition_of_touch:
            # If there's a touch, use the labeled class
            panda_matrix.append([video_counter, frame_counter, dist_ball_rfoot, dist_ball_lfoot, dist_ball_rthigh, dist_ball_lthigh, dist_ball_chest, height_ratio] + one_hot_enc_class[(dict_class_frame[frame_counter + i]).strip(" ")])
        else:
            # If there's no touch, label as "Other"
            panda_matrix.append([video_counter, frame_counter, dist_ball_rfoot, dist_ball_lfoot, dist_ball_rthigh, dist_ball_lthigh, dist_ball_chest, height_ratio] + one_hot_enc_class["Other"])


        if latest_active_person_frame == frame_counter and person_counter < len(track_history_person)-1:
            person_counter += 1
        if latest_active_ball_frame == frame_counter and ball_counter < len(track_history_ball[active_ball])-1:
            ball_counter += 1    

        
    print("video number: ", video_counter, " has finished succesfully")
    video_counter += 1  # Increment video counter

    
# Create pandas DataFrame with all extracted features
# Columns: video ID, frame number, distances to each body part,
# height ratio, and one-hot encoding of the class
column_names = ["video_id", "frame_no", "distance_to_RF","distance_to_LF","distance_to_RT", "distance_to_LT","distance_to_CH", "person_ball_H_rt", "RightF", "LeftF", "RightT", "LeftT", "Chest", "Other"]
data_frame = pd.DataFrame(panda_matrix, columns=column_names)

# Filter videos with problematic data
df_filtered = data_frame[~data_frame['video_id'].isin(bad_video_list)]

# Export DataFrame to CSV file
df_filtered.to_csv("C:\\Users\\Domagoj\\Desktop\\Diplomski\\Codes\\New_data_Session25_chest_filtered.csv", index=False)

# Print statistics of problematic videos
print("bad video conts: ", "ball lost: ", ball_missing, "\npeople overload: ", people_overload, "\nno touches: ", no_touch_counter, "\ntouch happend but data is missing:", len(bad_video_list))
