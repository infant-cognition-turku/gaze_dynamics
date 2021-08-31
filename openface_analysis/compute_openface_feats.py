# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Perform OpenFace video analysis for video files in a given directory. In addition,
compute eye aperture and add it to OpenFace's output CSV file (if selected).

"""

import os
import sys
import platform
import csv
from tqdm import tqdm


##############################################################################################
################################# The configuration settings #################################
##############################################################################################

#######################################################
# Initial directories (modify according to your need) #
#######################################################

# The directory that contains the .mp4 videos we want to analyze
video_file_dir = 'C:/Work/code/python_files/ml_pipeline/test_videos'

# The directory where the output of OpenFace will be stored (will be automatically created if it doesn't already exist)
openface_output_dir = 'C:/Work/code/python_files/ml_pipeline/test_openface_output'

# The directory where the OpenFace executable is stored
openface_dir = 'C:/Work/code/openface/OpenFace'



#############################################################
# Analysis-related settings (modify according to your need) #
#############################################################

# Do we want to compute eye aperture and add it to the CSV file output of OpenFace?
#
# Options: 1: We compute the eye aperture
#          0: We do not compute the eye aperture

compute_eye_aperture = 1

# Do we want to visualize the output during the process? The more is visualized, the
# slower the process is. Also note that if any part of the process is visualized, the
# visualizations will pop-up and close for each input video separately for the entire
# duration of the processing, so only visualize the output when you are NOT working on
# something else during the processing and you really want to have a look at the
# output in real-time.
#
# Options: 'all': visualizes all output
#          'tracked_face': visualizes the tracked face
#          'hog_features': visualizes the HOG features
#          'aligned_faces': visualizes the aligned faces
#          'action_units': visualizes the action units
#          'nothing': does not visualize anything

visualize = 'nothing'


# After the OpenFace video analysis, do we want to delete some unnecessary files?
#
# Options: 1: We delete the given files
#          0: We do not delete the given files

delete_aligned_faces = 1   # Do we delete the folders which contain the images of tracked faces?
delete_tracked_face_video = 1  # Do we delete the .AVI videos of tracked faces?
delete_hog_features = 1    # Do we delete the HOG features?
delete_meta_files = 1  # Do we delete the metadata files?


##############################################################################################
##############################################################################################
##############################################################################################




###############################################################
# Step 1: Perform OpenFace analysis for the given video files #
###############################################################

# Create output_video_dir if it doesn't exist
if not os.path.exists(openface_output_dir):
    os.makedirs(openface_output_dir)

# Find out the list of video files in the given directory
try:
    filenames_video = os.listdir(video_file_dir)
except FileNotFoundError:
    sys.exit('Given CSV file directory does not exist!')
    
video_files = [filename for filename in filenames_video if filename.endswith('.mp4') or filename.endswith('.MP4')]
    
# Delete filenames_csv to save memory
del filenames_video

# Check if there are no files in the given directories. Also check file number consistency
if len(video_files) == 0:
    sys.exit('There are no MP4 files in the given video file directory: ' + video_file_dir)
    

# Find out the name of the OpenFace executable (depends on the OS)
if platform.system() == 'Windows':
    executable = os.path.join(openface_dir, 'x64/Release/FeatureExtraction.exe')
else:
    executable = os.path.join(openface_dir, 'build/bin/FeatureExtraction')


# Run OpenFace video analysis for each of the videos
for video_file in tqdm(video_files):
    
    # Get the name of the video file
    video_file_name = os.path.join(video_file_dir, video_file)
    
    print('\nRunning OpenFace analysis for video :', video_file_name)
    
    # Run OpenFace for the video
    if visualize == 'all':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -verbose -wild'
    elif visualize == 'tracked_face':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -vis-track -wild'
    elif visualize == 'hog_features':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -vis-hog -wild'
    elif visualize == 'aligned_faces':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -vis-align -wild'
    elif visualize == 'action_units':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -vis-aus -wild'
    elif visualize == 'nothing':
        command = executable + ' -f "' + video_file_name + '" -out_dir "' + openface_output_dir + '" -wild'
    else:
        sys.exit('Check the parameter "visualize" for an incorrect value!')
        
    os.system(command)



#########################################################################
# Step 2: Perform some cleaning for the output directory (if necessary) #
#########################################################################

if delete_aligned_faces:
    print('\n\nDeleting aligned faces...')
    
    # Find all the subdirectories in the output directory
    subdirectories = [f.path for f in os.scandir(openface_output_dir) if f.is_dir()]
    
    # Go through all the subdirectories
    for subdirectory in subdirectories:
        
        # Find all the images in the directory
        filenames_images = os.listdir(subdirectory)
        image_files = [filename for filename in filenames_images if filename.endswith('.bmp')]
        
        # Go through all the image files and delete them
        for image in image_files:
            image_fullname = os.path.join(subdirectory, image)
            os.remove(image_fullname)
            
        # Remove the empty directory
        filenames_images = os.listdir(subdirectory)
        if len(filenames_images) == 0:
            os.rmdir(subdirectory)
    
    print('Done!')
    

if delete_tracked_face_video:
    print('\n\nDeleting tracked face videos...')
    
    # Find all the video files in the output directory
    filenames_openface_output_dir = os.listdir(openface_output_dir)
    video_files_tracked_faces = [filename for filename in filenames_openface_output_dir if filename.endswith('.avi')]
    
    # Delete all the videos
    for video in video_files_tracked_faces:
        video_fullname = os.path.join(openface_output_dir, video)
        os.remove(video_fullname)
        
    print('Done!')
    

if delete_hog_features:
    print('\n\nDeleting HOG features...')
    
    # Find all the HOG files in the output directory
    filenames_openface_output_dir = os.listdir(openface_output_dir)
    hog_files = [filename for filename in filenames_openface_output_dir if filename.endswith('.hog')]
    
    # Delete all the HOG files
    for hog_file in hog_files:
        hog_file_fullname = os.path.join(openface_output_dir, hog_file)
        os.remove(hog_file_fullname)
        
    print('Done!')


if delete_meta_files:
    print('\n\nDeleting metadata files...')
    
    # Find all the metadata files in the output directory
    filenames_openface_output_dir = os.listdir(openface_output_dir)
    metadata_files = [filename for filename in filenames_openface_output_dir if filename.endswith('.txt')]
    
    # Delete all the metadata files
    for metadata_file in metadata_files:
        metadata_file_fullname = os.path.join(openface_output_dir, metadata_file)
        os.remove(metadata_file_fullname)
        
    print('Done!')





##################################################################################
# Step 3: Compute eye aperture and add it to the OpenFace CSV file (if selected) #
##################################################################################

if compute_eye_aperture:
    print('\n\nComputing eye aperture...')
    
    # Find all the CSV files in the output directory
    filenames_openface_output_dir = os.listdir(openface_output_dir)
    csv_files = [filename for filename in filenames_openface_output_dir if filename.endswith('.csv')]
    
    # Go through all the CSV files
    for filename in tqdm(csv_files):
        csv_file = os.path.join(openface_output_dir, filename)
        
        # Read the CSV file
        csv_data = []
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row_original = next(csvreader)
            
            for row in csvreader:
                # Just in case, remove all '"' characters from each row
                row_cleaned = [s.replace('"', '') for s in row]
                csv_data.append(row_cleaned)
                
        
        # Now that we have the data collected from the CSV file, we compute the eye apertures for both eyes
        eye_apertures_right_vector = []
        eye_apertures_left_vector = []
        
        # Go through the values in the CSV file.
        for k in range(len(csv_data)):
            line = csv_data[k]
            
            # If face recognition confidence is above a certain threshold, we compute the eye aperture for both eyes.
            confidence_threshold = 0.5
            confidence = float(line[3])
            if confidence > confidence_threshold:
                # We compute the eye apertures. The facial landmark (68 keypoints)
                # indices we are interested of are indeces 36-47.
                
                # The x and y coordinates of the right eye
                x_36 = float(line[335])
                x_37 = float(line[336])
                x_38 = float(line[337])
                x_39 = float(line[338])
                x_40 = float(line[339])
                x_41 = float(line[340])
                right_eye_points_x = [x_36, x_37, x_38, x_39, x_40, x_41]
                
                y_36 = float(line[403])
                y_37 = float(line[404])
                y_38 = float(line[405])
                y_39 = float(line[406])
                y_40 = float(line[407])
                y_41 = float(line[408])
                right_eye_points_y = [y_36, y_37, y_38, y_39, y_40, y_41]

                # The x and y coordinates of the left eye
                x_42 = float(line[341])
                x_43 = float(line[342])
                x_44 = float(line[343])
                x_45 = float(line[344])
                x_46 = float(line[345])
                x_47 = float(line[346])
                left_eye_points_x = [x_42, x_43, x_44, x_45, x_46, x_47]
                
                y_42 = float(line[409])
                y_43 = float(line[410])
                y_44 = float(line[411])
                y_45 = float(line[412])
                y_46 = float(line[413])
                y_47 = float(line[414])
                left_eye_points_y = [y_42, y_43, y_44, y_45, y_46, y_47]
                
                # Compute right eye aperture
                initial_sum_right_eye = 0
                for a in range(5):
                    initial_sum_right_eye = initial_sum_right_eye + abs(right_eye_points_x[a]*right_eye_points_y[a+1] \
                                                                        - right_eye_points_y[a]*right_eye_points_x[a+1])
                
                sum_right_eye = initial_sum_right_eye + abs(right_eye_points_x[5]*right_eye_points_y[0] \
                                                            - right_eye_points_y[5]*right_eye_points_x[0])
                eye_aperture_right = 0.5 * sum_right_eye
                
                # Compute left eye aperture
                initial_sum_left_eye = 0
                for a in range(5):
                    initial_sum_left_eye = initial_sum_left_eye + abs(left_eye_points_x[a]*left_eye_points_y[a+1] \
                                                                        - left_eye_points_y[a]*left_eye_points_x[a+1])
                
                sum_left_eye = initial_sum_left_eye + abs(left_eye_points_x[5]*left_eye_points_y[0] \
                                                            - left_eye_points_y[5]*left_eye_points_x[0])
                eye_aperture_left = 0.5 * sum_left_eye
                
            else:
                # We give values of 0 for the eye apertures (not enough confidence).
                eye_aperture_right = 0
                eye_aperture_left = 0
                
            # Add the computed eye apertures to their respective vectors
            eye_apertures_right_vector.append(eye_aperture_right)
            eye_apertures_left_vector.append(eye_aperture_left)
        
        
        # Delete the old CSV file
        os.remove(csv_file)
        
        # Create a new CSV file which is similar to the original CSV file but it has eye apertures appended.
        new_header = header_row_original.copy()
        new_header.extend(['eye_aperture_right', 'eye_aperture_left'])
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(new_header)
            for i in range(len(csv_data)):
                new_row = csv_data[i]
                new_row.extend([eye_apertures_right_vector[i], eye_apertures_left_vector[i]])
                writer.writerow(new_row)
        
    print('Done!')
 





