# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Using the random subset feature selection method, we rank each feature's importance and
give the feature an importance score. In the process, a feature that is "turned off"
is randomized. Please note that the algorithm runs indefinitely, so check every now and
then the output text file (by default 'random_subset_selection_log.txt') whether the
feature importance order has converged.

"""


import numpy as np
from torch import cuda, no_grad, from_numpy, Tensor, load, flatten
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Dropout, ELU
from typing import Tuple
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from tensorflow.keras.utils import to_categorical
import sys
import random




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





class dataset_random_subset_selection(Dataset):

    def __init__(self, all_feature_indices, feature_subset, train_val_test='test', feature_path='./feats', 
                 train_val_ratio=0.75, random_state=22):
        super().__init__()
        
        # train_val_test has three options: 'train', 'validation' and 'test'
        if train_val_test == 'train':
            # We split the training data into two sets and take the larger set
            # as our training set
            features = np.load(os.path.join(feature_path, 'train_feats_temporal_90_frames.npy'))
            labels = np.load(os.path.join(feature_path, 'train_labels_temporal_90_frames.npy'))
            X, _, y, _ = train_test_split(features, labels, train_size=train_val_ratio, random_state=random_state)
            
                
        elif train_val_test == 'validation':
            # We split the training data into two sets and take the larger set
            # as our validation set
            features = np.load(os.path.join(feature_path, 'train_feats_temporal_90_frames.npy'))
            labels = np.load(os.path.join(feature_path, 'train_labels_temporal_90_frames.npy'))
            _, X, _, y = train_test_split(features, labels, train_size=train_val_ratio, random_state=random_state)
            
        else:
            # We use the test set
            X = np.load(os.path.join(feature_path, 'test_feats_temporal_90_frames.npy'))
            y = np.load(os.path.join(feature_path, 'test_labels_temporal_90_frames.npy'))
            
        # Create a randomized tensor that is the same shape as X
        np.random.seed(random_state)
        randomized_X = np.random.rand(X.shape[0], X.shape[1], X.shape[2])
        randomized_X = normalize_3d_features(randomized_X)
        
        # Determine which features are going to be randomized
        features_to_be_randomized = []
        for feat_index in all_feature_indices:
            if feat_index not in feature_subset:
                features_to_be_randomized.append(feat_index)
                
        # We input the selected random features from randomized_X and to X
        for feat_index in features_to_be_randomized:
            for i in range(X.shape[0]):
                X[i,:,feat_index] = randomized_X[i,:,feat_index]
        
        # Delete the unnecessary variable to save memory
        del randomized_X
        
        self.feats = from_numpy(X)
        self.labels = from_numpy(y)
        

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]





if __name__ == '__main__':
    
    feature_labels = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                      'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                      'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                      'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
                      'eye_aperture_right', 'eye_aperture_left']
    
    name_of_textfile = 'random_subset_selection_log.txt'
    file = open(name_of_textfile, 'w')
    file.close()
    
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(name_of_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')

    # Define hyper-parameters to be used.
    dropout = 0.1
    batch_size = 1024
    
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
    optimizer = Adam(params=DNN.parameters(), lr=1e-4)
    
    # Load the model
    DNN.load_state_dict(load('best_model_temporal_adv_90_frames.pt', map_location=device))
    
    # Parameters for testing
    params_test = {'batch_size': batch_size,
              'shuffle': False,
              'drop_last': False}
    
    # We have 25 different features
    all_feature_indices = np.arange(input_dim)
    
    # Initialize the score dicts
    dict_of_feature_scores = {}
    dict_of_feature_scores_normalized = {}
    for feat_index in all_feature_indices:
        dict_of_feature_scores[feat_index] = []
        dict_of_feature_scores_normalized[feat_index] = []
    
    # We continue our process infinitely
    iteration_number = 1
    while True:
        with open(name_of_textfile, 'a') as f:
            f.write('#####################################################################\n')
            f.write(f'Number of iteration: {iteration_number} \n\n')
            
        dict_of_feature_scores_current_iteration = {}
        
        # Make sure that we have the same randomized features in each iteration
        random_integer = random.randint(1, 2**30)
        
        # Randomize the feature indices and split the feature indices into subsets of equal size (size=5)
        randomized_feature_indices = np.arange(input_dim)
        np.random.shuffle(randomized_feature_indices)
        randomized_feature_subsets = np.split(randomized_feature_indices, 5)
        
        with open(name_of_textfile, 'a') as f:
            f.write(f'Order of randomized feature indices: {randomized_feature_indices} \n\n')
        
        # Go through all the feature subsets and evaluate the performance of the subset
        for feature_subset in randomized_feature_subsets:
            with open(name_of_textfile, 'a') as f:
                f.write(f'Now testing with subset: {feature_subset} \n')
                
            # Initialize the data loader
            test_set = dataset_random_subset_selection(all_feature_indices, feature_subset, random_state=random_integer)
            test_data_loader = DataLoader(test_set, **params_test)
            
            # Apply the DNN to the test data.
            testing_scores = []
            testing_accuracies = []
            DNN.eval()
            with no_grad():
                        
                for test_data in test_data_loader:
                            
                    X_input, y_output = [i.to(device) for i in test_data]
        
                    y_hat = DNN(X_input.float())  # Cast the double tensor to float
                    
                    # Convert to Numpy arrays
                    y_output = y_output.cpu().numpy()
                    y_hat = y_hat.cpu().numpy()
                    y_hat_softmax = softmax(y_hat, axis=1)
                    y_hat_max_indices = to_categorical(np.argmax(y_hat_softmax, axis=1), num_classes=2)
                    testing_score_batch = y_output * y_hat_softmax
                    testing_accuracy_batch = y_output * y_hat_max_indices
        
                    testing_scores.append(testing_score_batch)
                    testing_accuracies.append(testing_accuracy_batch)
                
                testing_score = np.mean(np.sum(np.concatenate(testing_scores, axis=0), axis=1))
                testing_accuracy = np.mean(np.sum(np.concatenate(testing_accuracies, axis=0), axis=1))
                
                # Delete unnecessary variables to save memory
                del test_set
                del test_data_loader
                
                # Add the score of the subset a feature participated in as the score of the
                # feature for the current iteration
                for feat_index in feature_subset:
                    dict_of_feature_scores_current_iteration[feat_index] = testing_score
                
                with open(name_of_textfile, 'a') as f:
                    f.write(f'  Testing score: {testing_score:7.5f} \n')
                    f.write(f'  Testing accuracy: {testing_accuracy:7.5f} \n\n')
        
            
        if len(dict_of_feature_scores_current_iteration) != input_dim:
            with open(name_of_textfile, 'a') as f:
                f.write(f'Something went wrong! len(dict_of_feature_scores_current_iteration) is not {input_dim}')
            sys.exit()
            
        # Go through all scores of the iteration to find out the min and max value
        # for min-max normalization
        scores_current_iteration = []
        for feat_index in dict_of_feature_scores_current_iteration:
            scores_current_iteration.append(dict_of_feature_scores_current_iteration[feat_index])
            
        min_score = np.amin(np.array(scores_current_iteration))
        max_score = np.amax(np.array(scores_current_iteration))
        
        # Add the score and the normalized score of each feature index to their respective dicts
        for feat_index in all_feature_indices:
            score_feature = dict_of_feature_scores_current_iteration[feat_index]
            normalized_score_feature = (score_feature - min_score) / (max_score - min_score)
            dict_of_feature_scores[feat_index].append(score_feature)
            dict_of_feature_scores_normalized[feat_index].append(normalized_score_feature)
            
        # Print out the results. First sort the indices into a descending order based on the mean score
        # of the feature. Do the same for both unnormalized and normalized scores.
        mean_scores = []
        for feat_index in dict_of_feature_scores:
            mean_scores.append(np.mean(np.array(dict_of_feature_scores[feat_index])))
        mean_scores = np.array(mean_scores)
        sort_indices = mean_scores.argsort()
        mean_scores_sorted = mean_scores[sort_indices[::-1]]
        all_feature_indices_sorted = all_feature_indices[sort_indices[::-1]]
        
        with open(name_of_textfile, 'a') as f:
            f.write('The results of the random subset feature selection process (unnormalized scores): \n')
        for i in range(input_dim):
            feature_index = all_feature_indices_sorted[i]
            with open(name_of_textfile, 'a') as f:
                f.write(f'{i+1}. Feature index {feature_index} ({feature_labels[feature_index]});\n')
                f.write(f'        Mean score: {mean_scores_sorted[i]}\n')
        
        
        mean_scores_normalized = []
        for feat_index in dict_of_feature_scores_normalized:
            mean_scores_normalized.append(np.mean(np.array(dict_of_feature_scores_normalized[feat_index])))
        mean_scores_normalized = np.array(mean_scores_normalized)
        sort_indices_normalized = mean_scores_normalized.argsort()
        mean_scores_normalized_sorted = mean_scores_normalized[sort_indices_normalized[::-1]]
        all_feature_indices_normalized_sorted = all_feature_indices[sort_indices_normalized[::-1]]
        
        with open(name_of_textfile, 'a') as f:
            f.write('\nThe results of the random subset feature selection process (normalized scores): \n')
        for i in range(input_dim):
            feature_index = all_feature_indices_normalized_sorted[i]
            with open(name_of_textfile, 'a') as f:
                f.write(f'{i+1}. Feature index {feature_index} ({feature_labels[feature_index]});\n')
                f.write(f'        Mean normalized score: {mean_scores_normalized_sorted[i]}\n')
        
        iteration_number += 1
            
        


