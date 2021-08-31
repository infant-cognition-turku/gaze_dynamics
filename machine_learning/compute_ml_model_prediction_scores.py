# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Compute the model prediction score for each test subject, a separate score all the videos
for each test subject (i.e., 1-4 scores per each test subject, depending on the number of
videos that the test subject is involved in). A higher average score means that the child
deviates from the average model less, while a lower average score means that the child 
deviates from the average model more. Take the min-max normalized scores (video level
normalization) are also provided for each video separately.

"""

import csv
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.nn import Module, Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Dropout, ELU
from torch import cuda, no_grad, from_numpy, Tensor, load, flatten
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from typing import Tuple
from scipy.special import softmax
import matplotlib.pyplot as plt



# Normalize the 3d input features to have zero mean and unit variance (each column
# represents each distinct feature) -> the dimensions of the input are (element_id, frame_index, feature_index)
def normalize_3d_features(feats) -> np.ndarray:
    
    feats_unrolled = np.squeeze(np.expand_dims(np.reshape(feats, (-1, feats.shape[2])), axis=0))
    feat_mean = feats_unrolled.mean(axis=0)
    feat_std = feats_unrolled.std(axis=0)
    
    # Go through each element in the features and normalize the elements' features
    h, _, _ = feats.shape
    for i in range(h):
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


class TemporalDNN(Module):

    def __init__(self,
                 conv_1_in_dim: int,
                 conv_1_out_dim: int,
                 num_norm_features_1: int,
                 mp_1_kernel_size: Tuple[int,int],
                 conv_2_in_dim: int,
                 conv_2_out_dim: int,
                 num_norm_features_2: int,
                 mp_2_kernel_size: Tuple[int,int],
                 conv_3_in_dim: int,
                 conv_3_out_dim: int,
                 num_norm_features_3: int,
                 mp_3_kernel_size: Tuple[int,int],
                 conv_4_in_dim: int,
                 conv_4_out_dim: int,
                 num_norm_features_4: int,
                 mp_4_kernel_size: Tuple[int,int],
                 linear_1_input_dim: int,
                 linear_1_output_dim: int,
                 linear_2_input_dim: int,
                 linear_2_output_dim: int,
                 linear_3_input_dim: int,
                 linear_3_output_dim: int,
                 zero_padding: int,
                 kernel_size: int,
                 dropout: float) \
            -> None:

        super().__init__()
        
        self.conv_layer_1 = Conv2d(in_channels=conv_1_in_dim, out_channels=conv_1_out_dim,
                                   kernel_size=kernel_size, padding=zero_padding)
        
        self.batch_normalization_1 = BatchNorm2d(num_norm_features_1)
        self.maxpooling_1 = MaxPool2d(mp_1_kernel_size) # Default value of stride is kernel_size
        
        self.conv_layer_2 = Conv2d(in_channels=conv_2_in_dim, out_channels=conv_2_out_dim,
                                   kernel_size=kernel_size, padding=zero_padding)
        
        self.batch_normalization_2 = BatchNorm2d(num_norm_features_2)
        self.maxpooling_2 = MaxPool2d(mp_2_kernel_size) # Default value of stride is kernel_size
        
        self.conv_layer_3 = Conv2d(in_channels=conv_3_in_dim, out_channels=conv_3_out_dim,
                                   kernel_size=kernel_size, padding=zero_padding)
        
        self.batch_normalization_3 = BatchNorm2d(num_norm_features_3)
        self.maxpooling_3 = MaxPool2d(mp_3_kernel_size) # Default value of stride is kernel_size
        
        self.conv_layer_4 = Conv2d(in_channels=conv_4_in_dim, out_channels=conv_4_out_dim,
                                   kernel_size=kernel_size, padding=zero_padding)
        
        self.batch_normalization_4 = BatchNorm2d(num_norm_features_4)
        self.maxpooling_4 = MaxPool2d(mp_4_kernel_size) # Default value of stride is kernel_size
        
        #################################################################
        
        self.linear_layer_1 = Linear(in_features=linear_1_input_dim,
                              out_features=linear_1_output_dim)
        
        self.linear_layer_2 = Linear(in_features=linear_2_input_dim,
                              out_features=linear_2_output_dim)
                              
        self.linear_layer_3 = Linear(in_features=linear_3_input_dim,
                              out_features=linear_3_output_dim)
        
        self.non_linearity_relu = ReLU()
        self.non_linearity_elu = ELU()
        self.dropout = Dropout(dropout)
        
        
        
    def forward(self, X: Tensor) -> Tensor:
        
        # Make the batches of size [batch_size, num_frames, num_features] into size
        # [batch_size, 1, num_frames, num_features] by adding a dummy dimension
        X = X.unsqueeze(1)
        
        # The convolutional layers
        X = self.non_linearity_relu(self.batch_normalization_1(self.conv_layer_1(X)))
        X = self.dropout(self.maxpooling_1(X))
        X = self.non_linearity_relu(self.batch_normalization_2(self.conv_layer_2(X)))
        X = self.dropout(self.maxpooling_2(X))
        X = self.non_linearity_relu(self.batch_normalization_3(self.conv_layer_3(X)))
        X = self.dropout(self.maxpooling_3(X))
        X = self.non_linearity_relu(self.batch_normalization_4(self.conv_layer_4(X)))
        X = self.dropout(self.maxpooling_4(X))
        
        # Flatten before the linear layers
        X = flatten(X,start_dim=1,end_dim=-1)
        
        # The linear layers
        X = self.non_linearity_elu(self.linear_layer_1(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_2(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_3(X))
        
        # Return the output
        return X



class temporal_dataset(Dataset):

    def __init__(self, X):
        super().__init__()    
        self.feats = from_numpy(X)
        
    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index]









if __name__ == '__main__':
    
    ##############################################################################################
    ################################# The configuration settings #################################
    ##############################################################################################
    
    # The directory of the gaze direction CSV files
    gaze_directions_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/gaze_directions'
    
    # The directory of OpenFace's output CSV files
    openface_output_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output_interpolated'
    
    # The name of the output CSV file containing the test subject scores
    output_csv_file_name = './test_subject_scores_video_level_normalized.csv'
    
    # The directory of the output visualization plots (will be automatically created if it doesn't
    # already exist)
    output_visualization_dir = './test_subject_score_visualizations'
    
    # The number of frames in each sample
    features_num_frames = 90 # With 120 FPS equals to 0.75 seconds
    
    # The indices of the desired features from OpenFace's CSV output
    desired_openface_feature_indices = np.concatenate((np.arange(293,299), np.arange(679,696), np.arange(714,716)))
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    
    
    
    #############################################################################################################
    # Neural network related parameters
    
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}\n\n')    

    # Define hyper-parameters to be used.
    dropout = 0.1
    
    num_input_channels = 1    # Number of frames in the input features
    conv_out_dim_1 = 8
    conv_out_dim_2 = 16
    conv_out_dim_3 = 32
    conv_out_dim_4 = 64
    kernel_size_mp_1 = (2,1)
    kernel_size_mp_2 = (3,1)
    kernel_size_mp_3 = (3,1)
    kernel_size_mp_4 = (1,5)
    input_dim = 25
    linear_1_input_dim = 1600
    linear_1_output_dim = 512
    linear_2_input_dim = 512
    linear_2_output_dim = 256
    linear_3_input_dim = 256
    output_dim = 2    # The output dimension. We have 2 different labels.
    
    kernel_size = 3
    zero_pad_size = 1

    # Instantiate our DNN
    DNN = TemporalDNN(conv_1_in_dim = num_input_channels,
                 conv_1_out_dim = conv_out_dim_1,
                 num_norm_features_1 = conv_out_dim_1,
                 mp_1_kernel_size = kernel_size_mp_1,
                 conv_2_in_dim = conv_out_dim_1,
                 conv_2_out_dim = conv_out_dim_2,
                 num_norm_features_2 = conv_out_dim_2,
                 mp_2_kernel_size = kernel_size_mp_2,
                 conv_3_in_dim = conv_out_dim_2,
                 conv_3_out_dim = conv_out_dim_3,
                 num_norm_features_3 = conv_out_dim_3,
                 mp_3_kernel_size = kernel_size_mp_3,
                 conv_4_in_dim = conv_out_dim_3,
                 conv_4_out_dim = conv_out_dim_4,
                 num_norm_features_4 = conv_out_dim_4,
                 mp_4_kernel_size = kernel_size_mp_4,
                 linear_1_input_dim = linear_1_input_dim,
                 linear_1_output_dim = linear_1_output_dim,
                 linear_2_input_dim = linear_2_input_dim,
                 linear_2_output_dim = linear_2_output_dim,
                 linear_3_input_dim = linear_3_input_dim,
                 linear_3_output_dim = output_dim,
                 zero_padding = zero_pad_size,
                 kernel_size = kernel_size,
                 dropout = dropout)
    
    
    # Pass DNN to the available device.
    DNN = DNN.to(device)

    # Give the parameters of our DNN to an optimizer.
    optimizer = Adam(params=DNN.parameters(), lr=1e-3)
    
    # Load the model
    DNN.load_state_dict(load('best_model_temporal_adv_90_frames.pt', map_location=device))
    
    #############################################################################################################
    
    # Create the output directory if it does not already exist
    if not os.path.exists(output_visualization_dir):
        os.makedirs(output_visualization_dir)
    
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
    
    # Go through the files to get the unique test subject IDs
    test_subject_ids = []
    for filename in csv_files:
        csv_file = os.path.join(gaze_directions_csv_file_dir, filename)
        
        # The ID of the test subject is the first four letters of filename
        id_test_subject = filename[0:4]
        test_subject_ids.append(id_test_subject)
    
    # Get the unique test subjects in sorted ID order
    test_subject_ids.sort()
    test_subject_ids_unique = list(dict.fromkeys(test_subject_ids))
    
    
    all_video_scores_dict = {}
    test_subject_scores_dict = {}
    # Start going through each test subject
    for test_subject in tqdm(test_subject_ids_unique):
        features_test_subject = {}
        labels_test_subject = {}
        # Go through all gaze direction CSV files that concern the specific test subject. Only take
        # into account the longer videos
        for filename_csv in csv_files:
            if 'break' not in filename_csv:
                if test_subject in filename_csv:
                    features_test_subject_video = []
                    labels_test_subject_video = []
                    # Find out the filename of the CSV-file-specific video file
                    csv_file = os.path.join(gaze_directions_csv_file_dir, filename_csv)
                    video_id = find_out_video_id(csv_file)
                    
                    # Get the OpenFace's output CSV file related to the specific video
                    openface_csv_file = os.path.join(openface_output_csv_file_dir, video_id.split('.')[0] + '.csv')
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
                    
                    # Gather all gaze labels from the gaze direction CSV file into a list
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
                            features_test_subject_video.append(features)
                            
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
                                
                            labels_test_subject_video.append(label)
                            
                    if len(labels_test_subject_video) != 0:
                        features_test_subject[video_id] = features_test_subject_video
                        labels_test_subject[video_id] = labels_test_subject_video
        
        # We skip the processing of features if there are no features available for the test subject
        if len(labels_test_subject) == 0:
            continue
        
        # Compute the score for each video separately
        for video in features_test_subject:
            feats_video = np.array(features_test_subject[video])
            feats_video = normalize_3d_features(feats_video)
            labels_video = np.array(labels_test_subject[video])
        
            # Now that we have all video-specific labels of the test subject, and also all video-specific
            # features gathered and normalized, we analyze the features frame-by-frame using our model, and
            # then compute the mean score based on the differences between the model predictions and the
            # actual gaze labels
        
            # Parameters for testing
            params_test = {'batch_size': feats_video.shape[0],
                      'shuffle': False,
                      'drop_last': False}
            
            # Initialize the test set
            test_set = temporal_dataset(feats_video)
            test_data_loader = DataLoader(test_set, **params_test)
            
            DNN.eval()
            with no_grad():
                        
                for test_data in test_data_loader:
                            
                    X_input = test_data
                    X_input = X_input.to(device)
        
                    y_hat = DNN(X_input.float())  # Cast the double tensor to float
                    
                    # Convert to Numpy arrays
                    y_hat = y_hat.cpu().numpy()
                    prediction_confidences = softmax(y_hat, axis=1)
                    
            prediction_confidences = np.array(prediction_confidences, dtype=np.float32)
            test_subject_score_video = np.mean(np.sum(prediction_confidences * labels_video, axis=1))
            
            # Add the test subject score of the specific video to the dictionary of test subject scores
            if test_subject not in test_subject_scores_dict:
                test_subject_scores_dict[test_subject] = {video: test_subject_score_video}
            else:
                test_subject_scores_dict[test_subject][video] = test_subject_score_video
            
            # Add the score to a list of all scores in order to determine the overall minimum and
            # maximum score values for min-max normalization
            if video not in all_video_scores_dict:
                all_video_scores_dict[video] = [test_subject_score_video]
            else:
                all_video_scores_dict[video].append(test_subject_score_video)
            
    
    # Write the data from test_subject_scores_dict into a CSV file. Also, find the minimum and maximum
    # value of test subject scores, and add the min-max normalized scores as well to the CSV file
    new_header = ['test_subject_id', 'video_id', 'score', 'normalized_score']
    with open(output_csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(new_header)
        for subject_id in test_subject_scores_dict:
            for video_id in test_subject_scores_dict[subject_id]:
                min_score = np.amin(np.array(all_video_scores_dict[video_id]))
                max_score = np.amax(np.array(all_video_scores_dict[video_id]))
                score = test_subject_scores_dict[subject_id][video_id]
                normalized_score = (score - min_score) / (max_score - min_score)
                csv_row = [subject_id, video_id, str(score), str(normalized_score)]
                writer.writerow(csv_row)
                
    # Plot the normalized video scores for each video
    for video in all_video_scores_dict:
        video_scores = np.array(all_video_scores_dict[video])
        min_score = np.amin(video_scores)
        max_score = np.amax(video_scores)
        normalized_scores = (video_scores - min_score) / (max_score - min_score)
        plt.figure(figsize=(19, 9.5))
        plt.plot(normalized_scores, marker='.', markersize=20, linestyle="None")
        title_text = 'Min-max normalized scores for ' + video.split('.')[0]
        plt.title(title_text, fontsize=30)
        plt.ylabel('Score', fontsize=20)
        plt.xlabel('Test subject index', fontsize=20)
        plt.ylim((-0.02,1.02))
        
        write_name = os.path.join(output_visualization_dir, video.split('.')[0] + '_scores.pdf')
        plt.savefig(write_name)
        plt.close()
    
