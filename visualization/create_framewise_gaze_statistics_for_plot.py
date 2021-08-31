# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Creates a file containing framewise gaze statistics for the following script:
  -create_face_prediction_plot.py

"""


import csv
import os
import sys
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':
    
    ##############################################################################################
    ################################# The configuration settings #################################
    ##############################################################################################
    
    # The directory of the gaze direction CSV files
    gaze_directions_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/gaze_directions'
    
    # The directory of the output files that are going to be used by the script
    # create_face_prediction_plot.py (will be automatically created if it doesn't already exist)
    output_dir = './ml_model_prediction_visualization_plot_data'
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    
    # Create the output directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make some error handling
    try:
        filenames_csv = os.listdir(gaze_directions_csv_file_dir)
    except FileNotFoundError:
        sys.exit('Given CSV file directory does not exist!')
        
    csv_files = [filename for filename in filenames_csv if filename.endswith('.csv')]
        
    # Delete filenames_csv to save memory
    del filenames_csv
    
    # Check if there are no files in the given directories. Also check file number consistency
    if len(csv_files) == 0:
        sys.exit('There are no CSV files in the given CSV file directory: ' + gaze_directions_csv_file_dir)
    
    # Go through the files
    population_data = {}
    for filename in tqdm(csv_files):
        csv_file = os.path.join(gaze_directions_csv_file_dir, filename)
        
        # Read the input CSV file
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row = next(csvreader)
            
            for row in csvreader:
                frame_number = row[0]
                video_id = row[1]
                gaze_label = row[-1]
                
                if video_id in population_data:
                    if frame_number not in population_data[video_id]:
                        # Add the frame number to the dictionary
                        population_data[video_id][frame_number] = []
                else:
                    # Add the video to the dictionary population_data
                    population_data[video_id] = {}
                    population_data[video_id][frame_number] = []
                
                population_data[video_id][frame_number].append(gaze_label)    
        
    
    counts_roi = {}
    for video_id in tqdm(population_data):
        
        # We only take the long videos into account
        if 'break' in video_id:
            continue
        
        counts_roi[video_id] = {}
        
        for frame_number in population_data[video_id]:
            counts_roi[video_id][frame_number] = {}
            
            other_count = 0
            not_available_count = 0
            nose_count = 0
            eyes_count = 0
            mouth_count = 0
            face_count = 0
            
            for label in population_data[video_id][frame_number]:
                if label == 'other':
                    other_count += 1
                elif label == 'not_available':
                    not_available_count += 1
                elif label == 'nose':
                    nose_count += 1
                elif label == 'left_eye' or label == 'right_eye':
                    eyes_count += 1
                elif label == 'mouth':
                    mouth_count += 1
                elif label == 'face':
                    face_count += 1
                else:
                    sys.exit('Element ' + label + ' not a valid label.')
                    
            counts_roi[video_id][frame_number]['other_count'] = other_count
            counts_roi[video_id][frame_number]['not_available_count'] = not_available_count
            counts_roi[video_id][frame_number]['nose_count'] = nose_count
            counts_roi[video_id][frame_number]['eyes_count'] = eyes_count
            counts_roi[video_id][frame_number]['mouth_count'] = mouth_count
            counts_roi[video_id][frame_number]['face_count'] = face_count
    
    
    # Finally, after we have collected the gaze counts for each frame of each long video, we compute the
    # probability of the gaze at the face for each frame and save it in a Numpy array
    for video_id in tqdm(counts_roi):
        write_name = os.path.join(output_dir, video_id.split('.')[0] + '_occurrence_prob_framewise.npy')
        
        # A list where all probabilities of looking at the face are added. We do not take the last 15 frames
        # into account since usually there is gaze data from only a few test subjects in those frames
        num_last_frames_removed = 15
        face_probs_framewise = []
        for frame_number in range(1, len(counts_roi[video_id])-num_last_frames_removed):
            other_count_frame = counts_roi[video_id][str(frame_number)]['other_count']
            nose_count_frame = counts_roi[video_id][str(frame_number)]['nose_count']
            eyes_count_frame = counts_roi[video_id][str(frame_number)]['eyes_count']
            mouth_count_frame = counts_roi[video_id][str(frame_number)]['mouth_count']
            face_count_frame = counts_roi[video_id][str(frame_number)]['face_count']
            
            total_count = other_count_frame + nose_count_frame + eyes_count_frame + mouth_count_frame + face_count_frame
            
            if total_count != 0:
                onface_count = nose_count_frame + eyes_count_frame + mouth_count_frame + face_count_frame
                onface_prob = onface_count / total_count
            else:
                onface_prob = 0
            face_probs_framewise.append(onface_prob)
        
        face_probs_framewise = np.array(face_probs_framewise, dtype=np.float32)
        np.save(write_name, face_probs_framewise)
        