# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Visualizes fixations of a specific person of a specific video file into a new video file.
Note that the video files and the fixation CSV files should be in the same framerate! Also
note that this script only requires having fixation data and their respective videos
WITHOUT the need for OpenFace features.

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

# The ID of the specific test subject that we want to visualize
test_subject_id = '0209'

# The ID of the video that we want to visualize
visualization_video_id = 'facevideo12s.mp4'

# The directory of the fixation CSV files
fixations_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/fixations/'

# The directory of the videos that the test participants were looking at. HAS TO BE in the same framerate as
# the fixations CSV files (e.g. 120 Hz with some Tobii eye trackers)
video_file_dir = 'C:/Work/code/python_files/ml_pipeline/videos_same_resolution_upsampled/'

# Define the output directory of the visualization videos (will be automatically
# created if it does not already exist)
output_dir = './fixations_visualization_specific_person'

##############################################################################################
##############################################################################################
##############################################################################################




# Create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Make some error handling
try:
    filenames_csv = os.listdir(fixations_csv_file_dir)
except FileNotFoundError:
    sys.exit('Given CSV file directory does not exist!')
    
try:
    filenames_video = os.listdir(video_file_dir)
except FileNotFoundError:
    sys.exit('Given video file directory does not exist!')
    
csv_files = [filename for filename in filenames_csv if filename.endswith('.csv')]

# Delete filenames_csv to save memory
del filenames_csv

video_files = [filename for filename in filenames_video if filename.endswith('.mp4') or filename.endswith('.MP4')]

# Check if there are no files in the given directories. Also check file number consistency
if len(csv_files) == 0:
    sys.exit('There are no CSV files in the given CSV file directory: ' + fixations_csv_file_dir)
    
if len(video_files) == 0:
    sys.exit('There are no video files in the given video file directory: ' + video_file_dir)

# Read CSV files, process each one by one
for filename in csv_files:
    
    # Only take into account those cases when the specific test subject is present
    if test_subject_id not in filename:
        continue
    
    csv_file = os.path.join(fixations_csv_file_dir, filename)
    
    csv_data = []
    
    # Read the input CSV file
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # Skip the header row
        header_row = next(csvreader)
        
        for row in csvreader:
            csv_data.append(row)
    
    # We skip the iterations in which there is no gaze data available
    if len(csv_data) == 0:
        continue
    
    # Go through all the gaze data
    gazedata = np.array(csv_data)
    trial_name = (gazedata[0][0]).split('.')[0]
    
    # Find out the video file of the gaze data
    videofile_name = os.path.join(video_file_dir, gazedata[0][1])
    
    # Only take into account the video files with the specific video ID
    if visualization_video_id not in videofile_name:
        continue
    
    # Start capturing the video
    cap = cv2.VideoCapture(videofile_name)
    
    # Get the framerate of the video
    fps_v = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Check if instantiation was successful
    if not cap.isOpened():
        raise Exception("Could not open video file: " + videofile_name)
        
    # A list containing all of the frames in the video
    video_data = []
    
    # Append each video frame into the list
    while True:
        _, frame = cap.read()
        
        if frame is None:
            break
        
        video_data.append(frame)
    
    height, width, _ = video_data[0].shape
    
    # Close capture
    cap.release()
    
    # Start writing the video file with gaze visualizations
    write_name = os.path.join(output_dir, trial_name + '_visualization_fixations.mp4')
    out = cv2.VideoWriter(write_name, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width, height))
    
    # Start going through the video and gaze data, and add gaze points to desired frames
    video_frame_index = 1
    
    for data_row in tqdm(gazedata):
        
        fixation_x_position = int(float(data_row[10]))
        fixation_y_position = int(float(data_row[11]))
        
        fixation_first_frame_index = int(float(data_row[5]))
        fixation_last_frame_index = int(float(data_row[6]))
        
        # Perform some error prevention
        if (fixation_first_frame_index - 1) >= len(video_data):
            break
        
        if (fixation_last_frame_index - 1) >= len(video_data):
            fixation_last_frame_index = len(video_data) - 1
        
        fixation_frame_indices = np.arange(fixation_first_frame_index, fixation_last_frame_index + 1)
        
        # We are behind the first frame index of the fixation, so we catch up
        if video_frame_index < fixation_first_frame_index:
            while video_frame_index < fixation_first_frame_index:
                video_frame = video_data[video_frame_index-1]
                test_subject_id_text = 'Subject ID: ' + test_subject_id
                video_frame = cv2.putText(video_frame, test_subject_id_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
                out.write(video_frame)
                video_frame_index += 1
        
        # We should not be ahead of the first frame index, so we raise an error if we are.
        if video_frame_index > fixation_first_frame_index:
            sys.exit('Something went wrong when processing the file.')
            
        # Go through the frames in the fixation and add gaze points to the video
        for video_frame_index in fixation_frame_indices:
            # Handle rounding errors of last video frames in the CSV file
            if (video_frame_index - 1) < len(video_data):
                video_frame = video_data[video_frame_index-1]
            else:
                video_frame = video_data[len(video_data)-1]
            cv2.drawMarker(video_frame, (fixation_x_position,fixation_y_position), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=6)
            test_subject_id_text = 'Subject ID: ' + test_subject_id
            video_frame = cv2.putText(video_frame, test_subject_id_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            out.write(video_frame)
    
    out.release()
    
    # Delete the original video list to save memory
    del video_data


