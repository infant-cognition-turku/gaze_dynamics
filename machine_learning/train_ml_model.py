# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Train a machine-learning model to predict the gaze label (face/non-face) for each video frame
based on the OpenFace features using the training data. After training, test the performance of the
model on the test data (i.e., compute the loss on the test set). This model creates dependencies for
90 adjacent frames by using a convolutional neural network (CNN) model.

"""


from copy import deepcopy
import numpy as np
from torch import cuda, no_grad, from_numpy, Tensor, max, save, load, flatten
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Dropout, ELU, CrossEntropyLoss
from typing import Tuple
from sys import exit
import os
from sklearn.model_selection import train_test_split
import time


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
    
    name_of_textfile = 'adv_cnn_90_trainlog.txt'
    file = open(name_of_textfile, 'w')
    file.close()
    
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(name_of_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')    

    # Define hyper-parameters to be used.
    max_epochs = 10000
    patience = 80
    dropout = 0.1
    batch_size = 4096
    
    load_model = 0
    
    
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

    # Instantiate our loss function as a class.
    loss_function = CrossEntropyLoss()

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    if load_model:
        DNN.load_state_dict(load('best_model_temporal_adv_90_frames.pt', map_location=device))
        best_model = deepcopy(DNN.state_dict())
    else:
        best_model = None
    
    # The parameters for training and validation
    params_train = {'batch_size': batch_size,
              'shuffle': True,
              'drop_last': False}
    
    # Parameters for testing
    params_test = {'batch_size': batch_size,
              'shuffle': False,
              'drop_last': False}
    
    # Initialize the data loaders
    with open(name_of_textfile, 'a') as f:
        f.write('Initializing training set...\n')
    training_set = temporal_dataset(train_val_test='train')
    train_data_loader = DataLoader(training_set, **params_train)
    with open(name_of_textfile, 'a') as f:
        f.write('Done!\n')
        f.write('Initializing validation set...\n')
    validation_set = temporal_dataset(train_val_test='validation')
    validation_data_loader = DataLoader(validation_set, **params_train)
    with open(name_of_textfile, 'a') as f:
        f.write('Done!\n')
        f.write('Initializing test set...\n')
    test_set = temporal_dataset(train_val_test='test')
    test_data_loader = DataLoader(test_set, **params_test)
    with open(name_of_textfile, 'a') as f:
        f.write('Done!\n')
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = 1

    # Start training.
    for epoch in range(1,max_epochs+1):
        
        start_time = time.time()

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        DNN.train()
        
        # Loop through every batch of our training data
        for train_data in train_data_loader:

            # Zero the gradient of the optimizer
            optimizer.zero_grad()

            # Get the batches
            X_input, y_output = [i.to(device) for i in train_data]

            # Get the predictions of our model
            y_hat = DNN(X_input.float())  # Cast the double tensor to float
            
            # Calculate the loss of our model. In order to calculate the loss 
            # using cross-entropy, we have to turn the ground truth (y_output)
            # to a vector.
            y_output = max(y_output, 1)[1]
            loss = loss_function(input=y_hat, target=y_output)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode, so (e.g.) dropout
        # will **not** function
        DNN.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():
            
            # Loop through every batch of our validation data
            for validation_data in validation_data_loader:

                # Get the batches
                X_input, y_output = [i.to(device) for i in validation_data]

                # Get the predictions of the model
                y_hat = DNN(X_input.float())  # Cast the double tensor to float

                # Calculate the loss
                y_output = max(y_output, 1)[1]
                loss = loss_function(input=y_hat, target=y_output)

                # Log the validation loss
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(DNN.state_dict())
            best_validation_epoch = epoch
            save(best_model, 'best_model_temporal_adv_90_frames.pt')
        else:
            patience_counter += 1
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        with open(name_of_textfile, 'a') as f:
            f.write(f'Epoch: {epoch:04d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss: {epoch_loss_validation:7.4f} (lowest: {lowest_validation_loss:7.4f}) | '
              f'Duration: {epoch_time:7.4f} seconds\n')
        
        # If patience counter is fulfilled, stop the training and do the testing
        if patience_counter >= patience:
            max_epochs_reached = 0
            break
        
        
    
    if max_epochs_reached:
        with open(name_of_textfile, 'a') as f:
            f.write('\nMax number of epochs reached, stopping training\n\n')
    else:
        with open(name_of_textfile, 'a') as f:
            f.write('\nExiting due to early stopping\n\n')
    
    if best_model is None:
        with open(name_of_textfile, 'a') as f:
            f.write('\nNo best model. The criteria for the lowest acceptable validation loss not satisfied!\n\n')
        exit()
    else:
        # Process similar to validation.
        with open(name_of_textfile, 'a') as f:
            f.write(f'\nBest epoch {best_validation_epoch} with loss {lowest_validation_loss}\n\n')
            f.write('Starting testing | ')
                
        # Load the best version of the model
        DNN.load_state_dict(best_model)
                
        testing_loss = []
        DNN.eval()
        with no_grad():
                    
            for test_data in test_data_loader:
                        
                X_input, y_output = [i.to(device) for i in test_data]

                y_hat = DNN(X_input.float())  # Cast the double tensor to float
                
                y_output = max(y_output, 1)[1]
                loss = loss_function(input=y_hat, target=y_output)

                testing_loss.append(loss.item())

            testing_loss = np.array(testing_loss).mean()
            with open(name_of_textfile, 'a') as f:
                f.write(f'Testing loss: {testing_loss:7.4f}')

