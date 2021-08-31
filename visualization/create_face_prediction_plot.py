# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Create a plot for the different ML models' output probabilities, and also the actual
probability of the population for each frame in each video. Uses data that is computed
in the following scripts:
  -create_framewise_gaze_statistics_for_plot.py
  -create_model_predictions_for_plot.py


"""


import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2






if __name__ == '__main__':

    ##############################################################################################
    ################################# The configuration settings #################################
    ##############################################################################################
    
    # The directory of the videos whose plots are visualized
    video_dir = './videos_same_resolution_upsampled'
    
    # The directory where the outputs of the scripts create_framewise_gaze_statistics_for_plot.py
    # and create_model_predictions_for_plot.py are located
    prediction_file_dir = './ml_model_prediction_visualization_plot_data'
    
    # The directory of the output visualization plots (will be automatically created if it doesn't
    # already exist)
    output_dir = './ml_model_prediction_and_prior_prob_plots'
    
    # The number of frames in each sample
    features_num_frames = 90 # With 120 FPS equals to 0.75 seconds
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    
    # Create the output directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the names for the video files
    try:
        filenames_video = os.listdir(video_dir)
    except FileNotFoundError:
        sys.exit('Given video file directory ' + video_dir + ' does not exist!')
        
    video_files = [filename for filename in filenames_video if filename.endswith('.mp4')]
        
    # Delete filenames_video to save memory
    del filenames_video
    
    # Check if there are no files in the given directories. Also check file number consistency
    if len(video_files) == 0:
        sys.exit('There are no MP4 files in the given video file directory: ' + video_dir)
    
    # Go through each video file
    for video_filename in tqdm(video_files):
        
        # Only take the long videos into account
        if 'break' in video_filename:
            continue
        
        # Get the name of the video
        videofile_name = os.path.join(video_dir, video_filename.split('.')[0] + '.mp4')
        
        # Start capturing the video
        cap = cv2.VideoCapture(videofile_name)
        
        # Check if instantiation was successful
        if not cap.isOpened():
            videofile_name = os.path.join(video_dir, video_filename.split('.')[0] + '.MP4')
            cap = cv2.VideoCapture(videofile_name)
            if not cap.isOpened():
                raise Exception("Could not open video file: " + videofile_name + " (tested both .mp4 and .MP4 formats)")
            
        # Get basic information from the video
        fps_v = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Get the name of the temporal and framewise models' prediction files + actual population probabilities and load them.
        face_probs_framewise = np.load(os.path.join(prediction_file_dir, video_filename.split('.')[0] + '_occurrence_prob_framewise.npy'))
        predictions_temp = np.load(os.path.join(prediction_file_dir, video_filename.split('.')[0] + '_model_predictions.npy'))
        
        # Make all the vectors of equal length
        predictions_temp = predictions_temp[0:len(face_probs_framewise)]
        
        # The basic information about the new plot-to-be-written
        write_name = os.path.join(output_dir, video_filename.split('.')[0] + '_predictions_and_actual_probabilities_plot.pdf')
        
        x_points = np.arange(len(predictions_temp)) / fps_v
        plt.figure(figsize=(19, 9.5))
        plt.plot(x_points, face_probs_framewise, linewidth=3)
        plt.plot(x_points, predictions_temp, linewidth=3)
        title_text = 'Model predictions for ' + video_filename.split('.')[0]
        plt.title(title_text, fontsize=30)
        plt.ylabel('OnFace probability', fontsize=20)
        plt.xlabel('Seconds', fontsize=20)
        plt.xlim((0,len(predictions_temp)/fps_v))
        plt.ylim((0,1))
        plt.legend(['Actual OnFace probability', 'Temporal model'], fontsize=16)
        plt.savefig(write_name)
        plt.close()

