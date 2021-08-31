# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Visualize the gaze-at-face probability output of the temporal machine-learning model
in the given set of video files.

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
from tensorflow.keras.utils import to_categorical
import cv2



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
    
    # The directory of OpenFace's output CSV files
    openface_output_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output_interpolated'
    
    # The directory of the videos that are going to be visualized
    video_dir = './videos_same_resolution_upsampled'
    
    # The output directory where the visualizations are saved
    output_dir = './ml_model_prediction_visualizations'
    
    # The number of frames in each sample
    features_num_frames = 90 # With 120 FPS equals to 0.75 seconds
    
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # The indices of the desired features from OpenFace's CSV output
    desired_openface_feature_indices = np.concatenate((np.arange(293,299), np.arange(679,696), np.arange(714,716)))
    
    # Get the names for OpenFace's CSV output files
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
    for openface_filename in tqdm(csv_files_openface):
        all_features_video = []
        
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
        
        # Go through each frame, gather all the sections with a length of features_num_frames frames as features
        for i in range(openface_feats.shape[0] - features_num_frames):
            features = []
            first_index = i
            last_index = i + features_num_frames
            
            for j in range(first_index, last_index):
                features.append(openface_feats[j][desired_openface_feature_indices])
            features = np.array(features)
            all_features_video.append(features)
        
        all_features_video = np.array(all_features_video)
        all_features_video = normalize_3d_features(all_features_video)
        
        # Now that we have all features gathered and normalized, we analyze the video frame-by-frame
        # using our model, and then save the visualization
        
        # Parameters for testing
        params_test = {'batch_size': all_features_video.shape[0],
                  'shuffle': False,
                  'drop_last': False}
        
        # Initialize the test set
        test_set = temporal_dataset(all_features_video)
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
                predictions = to_categorical(np.argmax(prediction_confidences, axis=1))
                
        prediction_confidences = np.array(prediction_confidences, dtype=np.float32)
        predictions = np.array(predictions, dtype=np.float32)
        
        
        # Get the name of the video. By default, the name of the video is the same as
        # the name of the OpenFace output's CSV file
        videofile_name = os.path.join(video_dir, openface_filename.split('.')[0] + '.mp4')
        
        # Start capturing the video
        cap = cv2.VideoCapture(videofile_name)
        
        # Check if instantiation was successful
        if not cap.isOpened():
            videofile_name = os.path.join(video_dir, openface_filename.split('.')[0] + '.MP4')
            cap = cv2.VideoCapture(videofile_name)
            if not cap.isOpened():
                raise Exception("Could not open video file: " + videofile_name + " (tested both .mp4 and .MP4 formats)")
            
        # Get basic information from the video
        fps_v = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # As a verification, check whether the number of frames is consistent with the
        # OpenFace CSV data
        if num_frames_video != len(openface_csv_data):
            sys.exit('Something is wrong, the number of frames in ' + videofile_name + ' (' + str(num_frames_video)
                     + ' frames) is different from the number of frames in ' + openface_csv_file + ' ('
                     + str(len(openface_csv_data)) + ' frames)')
        
        # The basic information about the new video-to-be-written with blurred faces
        write_name = os.path.join(output_dir, openface_filename.split('.')[0] + '_model_prediction_visualization.mp4')
        out = cv2.VideoWriter(write_name, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width, height))
        
        # Go through each frame in the video
        for video_frame_index in range(num_frames_video - 1):
            _, frame = cap.read()
            
            # For the first (features_num_frames - 1) frames we don't have a prediction
            if video_frame_index < (features_num_frames - 1):
                frame = cv2.putText(frame, 'No prediction', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            else:
                prediction = np.argmax(predictions[video_frame_index - features_num_frames + 1])
                confidence = prediction_confidences[video_frame_index - features_num_frames + 1][prediction]
                
                if prediction == 0:
                    gaze_label = 'other'
                else:
                    gaze_label = 'face'
                    
                prediction_text = 'Model prediction: ' + gaze_label
                confidence_text = 'Confidence: ' + str(confidence)
                frame = cv2.putText(frame, prediction_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
                frame = cv2.putText(frame, confidence_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            
            out.write(frame)
            
        out.release()

