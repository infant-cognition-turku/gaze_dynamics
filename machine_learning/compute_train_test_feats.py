# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Compute the training and test features from OpenFace's CSV output for a machine-learning model
that takes the temporal relation (90 frames) of the features into account. Only perform the
computation for long videos (names do not include "break").

"""


import csv
import os
import sys
import numpy as np
from tqdm import tqdm



# Normalize the 3d input features to have zero mean and unit variance (each column
# represents each distinct feature) -> the dimensions of the input are (element_id, frame_index, feature_index)
def normalize_3d_features(feats) -> np.ndarray:
    
    feats_unrolled = np.squeeze(np.expand_dims(np.reshape(feats, (-1, feats.shape[2])), axis=0))
    feat_mean = feats_unrolled.mean(axis=0)
    feat_std = feats_unrolled.std(axis=0)
    
    # Go through each element in the features and normalize the elements' features
    h, _, _ = feats.shape
    for i in tqdm(range(h)):
        feats[i,:,:] = (feats[i,:,:] - feat_mean) / feat_std
        feats[i,:,:] = np.nan_to_num(feats[i,:,:]) # Remove NaN values by converting them to zero
    
    return feats



# Find out the video ID of the gaze CSV file
def find_out_video_id(csv_file):
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        _ = next(csvreader)
        for row in csvreader:
            video_id = row[1]
            break
    
    return video_id



if __name__ == '__main__':

    ##############################################################################################
    ################################# The configuration settings #################################
    ##############################################################################################
    
    # The directory of the gaze direction CSV files
    gaze_directions_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/gaze_directions'
    
    # The directory of OpenFace's output CSV files
    openface_output_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output_interpolated'
    
    # The directory where the computed features will be stored (will be automatically created if
    # it doesn't already exist)
    output_feat_dir = './feats'
    
    # The names of the saved training and testing features and their respective labels
    savename_train_feats = os.path.join(output_feat_dir, 'train_feats_temporal_90_frames.npy')
    savename_train_labels = os.path.join(output_feat_dir, 'train_labels_temporal_90_frames.npy')
    savename_test_feats = os.path.join(output_feat_dir, 'test_feats_temporal_90_frames.npy')
    savename_test_labels = os.path.join(output_feat_dir, 'test_labels_temporal_90_frames.npy')
    
    # The ratio in which we split the data into a training and test set
    train_test_ratio = 0.80
    
    # The random seed for splitting the data to ensure repeatability 
    random_seed = 222
    
    # The number of frames in each sample
    features_num_frames = 90 # With 120 FPS equals to 0.75 seconds
    
    # The indices of the desired features from OpenFace's CSV output
    desired_openface_feature_indices = np.concatenate((np.arange(293,299), np.arange(679,696), np.arange(714,716)))
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    
    # Create output_feat_dir if it doesn't exist
    if not os.path.exists(output_feat_dir):
        os.makedirs(output_feat_dir)
    
    # Make some error handling
    try:
        filenames_csv = os.listdir(gaze_directions_csv_file_dir)
    except FileNotFoundError:
        sys.exit('Given CSV file directory ' + gaze_directions_csv_file_dir + ' does not exist!')
        
    csv_files_gazedir = [filename for filename in filenames_csv if filename.endswith('.csv')]
        
    # Delete filenames_csv to save memory
    del filenames_csv
    
    # Check if there are no files in the given directories. Also check file number consistency
    if len(csv_files_gazedir) == 0:
        sys.exit('There are no CSV files in the given CSV file directory: ' + gaze_directions_csv_file_dir)
    
    # The same for OpenFace's CSV output files
    try:
        filenames_csv_openface = os.listdir(openface_output_csv_file_dir)
    except FileNotFoundError:
        sys.exit('Given CSV file directory ' + openface_output_csv_file_dir + ' does not exist!')
        
    csv_files_openface = [filename for filename in filenames_csv_openface if filename.endswith('.csv')]
        
    # Delete filenames_csv to save memory
    del filenames_csv_openface
    
    # Check if there are no files in the given directories. Also check file number consistency
    if len(csv_files_openface) == 0:
        sys.exit('There are no CSV files in the given CSV file directory: ' + openface_output_csv_file_dir)
    
    
    # Go through each OpenFace CSV file, gather the features and the labels into a list
    all_features = []
    all_labels = []
    for openface_filename in tqdm(csv_files_openface):
        # Only take the long videos into account
        if 'break' in openface_filename:
            continue
        
        openface_csv_file = os.path.join(openface_output_csv_file_dir, openface_filename)
        openface_csv_data = []
        
        # Read the OpenFace CSV file
        with open(openface_csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row_openface = next(csvreader)
            
            for row in csvreader:
                openface_csv_data.append(row)
        
        # Convert the features into Numpy format for easier handling
        openface_feats = np.array(openface_csv_data, dtype=np.float32)
        
        # Go through each gaze direction CSV file
        for filename in csv_files_gazedir:
            csv_file = os.path.join(gaze_directions_csv_file_dir, filename)
            video_id = find_out_video_id(csv_file)
            
            # Read the gaze direction CSV file only if it concerns the current OpenFace CSV file
            if (video_id.split('.')[0] + '.csv') == openface_filename:
                # Gather all gaze labels from the file into a list
                gaze_labels_csv_file = []
                with open(csv_file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    header_row = next(csvreader)
                    for row in csvreader:
                        gaze_label = row[-1]
                        gaze_labels_csv_file.append(gaze_label)
                        
                # Gather all the sections with a length of features_num_frames frames as features
                for i in range(len(gaze_labels_csv_file) - features_num_frames):
                    features = []
                    first_index = i
                    last_index = i + features_num_frames
                    last_frame_of_section_label = gaze_labels_csv_file[last_index]
                    
                    # We skip the sections whose last frame's label is 'not_available'
                    if last_frame_of_section_label != 'not_available':
                        for j in range(first_index, last_index):
                            features.append(openface_feats[j][desired_openface_feature_indices])
                        features = np.array(features)
                        all_features.append(features)
                        
                        if last_frame_of_section_label == 'other':
                            label = np.array([1,0])
                        elif last_frame_of_section_label == 'nose':
                            label = np.array([0,1])
                        elif last_frame_of_section_label == 'left_eye' or last_frame_of_section_label == 'right_eye':
                            label = np.array([0,1])
                        elif last_frame_of_section_label == 'mouth':
                            label = np.array([0,1])
                        elif last_frame_of_section_label == 'face':
                            label = np.array([0,1])
                        else:
                            sys.exit(last_frame_of_section_label + ' is not a valid label.')
                            
                        all_labels.append(label)
    
    # Delete variables that are not needed to save memory
    del openface_feats
    del openface_csv_data
    del csv_files_openface
    del features
    del gaze_labels_csv_file
    
    
    # Convert the features and the labels into Numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # Normalize the features
    all_features = normalize_3d_features(all_features)
    
    # Split the features and labels into a train and test set
    np.random.seed(random_seed)
    mask = np.random.rand(len(all_labels)) <= train_test_ratio
    train_features = all_features[mask]
    test_features = all_features[~mask]
    train_labels = all_labels[mask]
    test_labels = all_labels[~mask]
    
    # Delete large variables to save memory
    del all_features
    del all_labels
    del mask
    
    # Save the features and labels
    np.save(savename_train_feats, train_features)
    np.save(savename_train_labels, train_labels)
    np.save(savename_test_feats, test_features)
    np.save(savename_test_labels, test_labels)




