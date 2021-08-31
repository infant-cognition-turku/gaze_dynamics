# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Visualize a set of 2D facial landmark points and head pose given the CSV file output from OpenFace
and the corresponding videos.

"""

import csv
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm


##############################################################################################
################################# The configuration settings #################################
##############################################################################################

# The directory of the OpenFace's output CSV files
csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output_interpolated/'

# The directory of the videos that the test participants were looking at. HAS TO BE in the same framerate as
# the fixations CSV files (e.g. 120 Hz with some Tobii eye trackers)
video_file_dir = 'C:/Work/code/python_files/ml_pipeline/videos_same_resolution_upsampled/'

# Define the output directory of the visualization videos (will be automatically
# created if it does not already exist)
output_dir = './head_pose_and_landmarks_visualization'

# Define the indices of our desired facial landmarks and yaw, pitch, and roll from the
# OpenFace's output CSV file
face_x_indices = np.arange(299,367)
face_y_indices = np.arange(367,435)
yaw_index = 297
pitch_index = 296
roll_index = 298

##############################################################################################
##############################################################################################
##############################################################################################




# Create the directory if it does not already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Make some error handling
try:
    filenames_csv = os.listdir(csv_file_dir)
except FileNotFoundError:
    sys.exit('Given CSV file directory does not exist!')
    
try:
    filenames_video = os.listdir(video_file_dir)
except FileNotFoundError:
    sys.exit('Given video file directory does not exist!')
    
csv_files = [filename for filename in filenames_csv if filename.endswith('.csv')]
video_files = [filename for filename in filenames_video if filename.endswith('.mp4') or filename.endswith('.MP4')]

# Check if there are no files in the given directories. Also check file number consistency
if len(csv_files) == 0:
    sys.exit('There are no CSV files in the given CSV file directory: ' + csv_file_dir)
    
if len(video_files) == 0:
    sys.exit('There are no video files in the given video file directory: ' + video_file_dir)
    
if len(csv_files) != len(video_files):
    sys.exit('There are a different number of CSV files and video files in the given CSV and video file directories.')


# Transpose the face index vector
face_indices = np.vstack((face_x_indices, face_y_indices)).T

# Read CSV files, store the data into a list
for filename_csv in tqdm(csv_files):
    csv_file = os.path.join(csv_file_dir, filename_csv)
    csv_data = []
    
    # Read the input CSV file
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # Skip the header row
        header_row = next(csvreader)
        
        for row in csvreader:
            csv_data.append(row)

    filename_video = filename_csv.split('.')[0] + '.mp4'
    
    # Get the CSV data in Numpy format
    csv_data = np.array(csv_data, dtype=np.float32)
    
    video_file = os.path.join(video_file_dir, filename_video)
    
    # Start capturing the video
    cap = cv2.VideoCapture(video_file)
    
    # Check if instantiation was successful
    if not cap.isOpened():
        filename_video = filename_csv.split('.')[0] + '.MP4'
        video_file = os.path.join(video_file_dir, filename_video)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise Exception("Could not open video file: " + video_file + " (tested both .mp4 and .MP4 formats)")
    
    # Get the framerate, width and height of the video
    fps_v = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Write the new video
    write_name = os.path.join(output_dir, filename_video + '_visualization_pose_landmarks.mp4')
    out = cv2.VideoWriter(write_name, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width, height))
    
    frame_index = 0
    
    while True:
        _, frame = cap.read()
        
        if frame is None:
            break
        
        # Get the features of the current frame
        frame_features = csv_data[frame_index]
        
        # Find the coordinates of the given points
        points_face = []
        for index in face_indices:
            points_face.append((int(frame_features[index[0]]), int(frame_features[index[1]])))
        
        # Draw the points
        for point in points_face:
            cv2.drawMarker(frame, point, color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
            
        # Get the yaw, pitch, and roll values from the OpenFace CSV data (in radians). Draw
        # x-, y-, and z-axis that are projected based on the yaw, pitch, and roll angles
        yaw = frame_features[yaw_index]
        pitch = frame_features[pitch_index]
        roll = frame_features[roll_index]
        
        # Convert the angles to degrees for later use
        yaw_degrees = yaw * (180/np.pi)
        pitch_degrees = pitch * (180/np.pi)
        roll_degrees = roll * (180/np.pi)
        
        # Turn two of the three arrows around (in their own axis) for a better visualization
        yaw = -yaw
        pitch = -pitch
        
        # Place the arrows to the top right of the screen
        tdx = (9/11) * width
        tdy = (1/13) * height
        
        # The length of the arrows
        size_arrow = width / 10
        
        # x-axis, drawn in red
        x_1 = size_arrow * (np.cos(yaw) * np.cos(roll)) + tdx
        y_1 = size_arrow * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
        
        # y-axis, drawn in green
        x_2 = size_arrow * (-np.cos(yaw) * np.sin(roll)) + tdx
        y_2 = size_arrow * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
        
        # z-axis, drawn in blue
        x_3 = size_arrow * (np.sin(yaw)) + tdx
        y_3 = size_arrow * (-np.cos(yaw) * np.sin(pitch)) + tdy
        
        # Draw the arrows
        cv2.arrowedLine(frame, (int(tdx), int(tdy)), (int(x_1), int(y_1)), color=(0,0,255), thickness=4)
        cv2.arrowedLine(frame, (int(tdx), int(tdy)), (int(x_2), int(y_2)), color=(0,255,0), thickness=4)
        cv2.arrowedLine(frame, (int(tdx), int(tdy)), (int(x_3), int(y_3)), color=(255,0,0), thickness=4)
        
        # Add text about the values of yaw, pitch, and roll
        yaw_text = 'Yaw: ' + str(round(yaw_degrees)) + ' degrees'
        pitch_text = 'Pitch: ' + str(round(pitch_degrees)) + ' degrees'
        roll_text = 'Roll: ' + str(round(roll_degrees)) + ' degrees'
        video_frame = cv2.putText(frame, yaw_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
        video_frame = cv2.putText(frame, pitch_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
        video_frame = cv2.putText(frame, roll_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
        
        out.write(frame)
        frame_index += 1
        
    # Close capture and video write
    cap.release()
    out.release()