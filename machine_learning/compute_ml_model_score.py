# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Compute the model prediction score and the model prediction accuracy using the
trained machine-learning model that takes into account the previous 90 frames
in each timestep. For each input sample, the model outputs the probabilities for
each output class (the probabilities sum up to 1).

"""


from copy import deepcopy
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

    def __init__(self, train_val_test='train', feature_path='./feats', 
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
        
        
        self.feats = from_numpy(X)
        self.labels = from_numpy(y)
        

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]





if __name__ == '__main__':
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')        

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
    best_model = deepcopy(DNN.state_dict())
    
    # Parameters for testing
    params_test = {'batch_size': batch_size,
              'shuffle': False,
              'drop_last': False}
    
    # Initialize the data loader
    print('Initializing test set...')
    test_set = temporal_dataset(train_val_test='test')
    test_data_loader = DataLoader(test_set, **params_test)
    print('Done!')
    
    # Apply the DNN to the test data.
    print('Starting testing', end=' | ')
            
    # Load the best version of the model
    DNN.load_state_dict(best_model)
            
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
            y_hat_max_indices = to_categorical(np.argmax(y_hat_softmax, axis=1))
            testing_score_batch = y_output * y_hat_softmax
            testing_accuracy_batch = y_output * y_hat_max_indices

            testing_scores.append(testing_score_batch)
            testing_accuracies.append(testing_accuracy_batch)
        
        testing_score = np.mean(np.sum(np.concatenate(testing_scores, axis=0), axis=1))
        testing_accuracy = np.mean(np.sum(np.concatenate(testing_accuracies, axis=0), axis=1))
        
        #testing_score = np.array(testing_score, dtype=np.float32)
        print(f'Testing score: {testing_score:7.5f} | Testing_accuracy: {testing_accuracy:7.5f}')


