# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Remove poorly detected facial features based on a predetermined threshold, and
perform interpolation for the rest of the features. Originally meant for features
from OpenFace, but can be used for other features with some minor modifications.

"""

from scipy.interpolate import interp1d
import numpy as np
import csv
import os
import sys
from tqdm import tqdm
from importlib.machinery import SourceFileLoader


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python feature_interpolation.py \nOR \n2) python feature_interpolation.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
else:
    try:
        import conf_feature_interpolation as conf
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python feature_interpolation.py \nOR \n2) python feature_interpolation.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_feature_interpolation.py" in the same directory 
        as "feature_interpolation.py"''')




def interpolate(interp_start_point, interp_end_point, interp_start_time, interp_end_time):
    """
    Returns a 1-D interpolation function based on the start and end points of the interpolation
    interval.
    
    interp_start_point: The starting point (parameter value) of the interpolation range.
    interp_end_point: The ending point (parameter value) of the interpolation range.
    interp_start_time: The time of the starting point.
    interp_end_time: The time of the ending point.

    """
    
    points = np.array([interp_start_point, interp_end_point])
    t = np.array([interp_start_time, interp_end_time])
    interp_function = interp1d(t, points)
    
    return interp_function




if __name__ == '__main__':

    # Take parameter values from the configuration file
    confidence_threshold = conf.confidence_threshold
    openface_csv_file_dir = conf.openface_csv_file_dir
    output_dir = conf.output_dir
    
    # Create the output directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Make some error handling
    try:
        filenames_csv = os.listdir(openface_csv_file_dir)
    except FileNotFoundError:
        sys.exit('Given CSV file directory does not exist!')
        
    csv_files = [filename for filename in filenames_csv if filename.endswith('.csv')]
    
    # Delete filenames_csv to save memory
    del filenames_csv
    
    # Check if there are no files in the given directories. Also check file number consistency
    if len(csv_files) == 0:
        sys.exit('There are no CSV files in the given CSV file directory: ' + openface_csv_file_dir)
        
    # Read CSV files, process each one by one
    csv_counter = 1
    for filename in csv_files:
        csv_file = os.path.join(openface_csv_file_dir, filename)
        csv_data = []
        print('\nInterpolating features for file: ' + filename + ', file ' + str(csv_counter) + '/' + str(len(csv_files)))
        csv_counter += 1
        
        # Read the input CSV file
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row = next(csvreader)
            
            # Add information of whether the frame is interpolated or not to the CSV header
            header_row.append('interpolated')
            
            for row in csvreader:
                csv_data.append(row)
            
            # Make thresholding based on the confidence threshold to determine which
            # frames are going to get replaced by interpolated versions.
            for frame in csv_data:
                confidence = float(frame[3])
                if confidence < confidence_threshold:
                    interpolated_flag = '1'
                else:
                    interpolated_flag = '0'
                frame.append(interpolated_flag)
            
            # Convert the CSV data into Numpy format for easier handling
            csv_data = np.array(csv_data, dtype=np.float32)
            
            # Initialize parameter values
            num_frames, num_items_per_frame = csv_data.shape
            interpolation_start_point = None
            interpolation_end_point = None
            interpolation_at_beginning_flag = 0
            
            # Skip the first five features ('frame', 'face_id', 'timestamp', 'confidence', 'success')
            # and the last feature ('interpolated').
            interpolation_feature_indices = np.arange(5, num_items_per_frame - 1)
                            
            # Start going through the data frame by frame. Initialize a progress bar using the tqdm library.
            for i in tqdm(range(num_frames)):
                frame_data = csv_data[i]
                
                # We try to identify the start and end points of the frames-to-be-interpolated
                if frame_data[-1] == 1:
                    if interpolation_start_point == None:
                        interpolation_start_point = i - 1
                        
                        # Tackle cases when there are erroneous frames at the beginning of the video
                        if interpolation_start_point < 1:
                            interpolation_start_point = 0
                            interpolation_at_beginning_flag = 1
                else:
                    if interpolation_start_point != None:
                        interpolation_end_point = i
                
                # If we have the interpolation start and end points known, we start interpolating
                if interpolation_start_point != None:
                    if interpolation_end_point != None:
                        if interpolation_at_beginning_flag == 1:
                            # Copy end point value to start
                            frame_data_interp_end = csv_data[interpolation_end_point]
                            interpolation_frame_indices = np.arange(interpolation_start_point + 1, interpolation_end_point)
                            
                            # Go through each feature-to-be-interpolated.
                            for interp_feature_index in interpolation_feature_indices:    
                                for interp_frame_index in interpolation_frame_indices:
                                    csv_data[interp_frame_index][interp_feature_index] = frame_data_interp_end[interp_feature_index]
                            
                            interpolation_at_beginning_flag = 0
                        else:
                            # Perform interpolation based on start and end points
                            frame_data_interp_start = csv_data[interpolation_start_point]
                            frame_data_interp_end = csv_data[interpolation_end_point]
                            interpolation_frame_indices = np.arange(interpolation_start_point + 1, interpolation_end_point)
                            
                            # Go through each feature-to-be-interpolated.
                            for interp_feature_index in interpolation_feature_indices:    
                                interp_function = interpolate(frame_data_interp_start[interp_feature_index],
                                                              frame_data_interp_end[interp_feature_index], 
                                                              frame_data_interp_start[2], frame_data_interp_end[2])
                                for interp_frame_index in interpolation_frame_indices:
                                    query_time = csv_data[interp_frame_index][2]
                                    csv_data[interp_frame_index][interp_feature_index] = interp_function(query_time)
                            
                        interpolation_start_point = None
                        interpolation_end_point = None
                        
            # Tackle cases when there are erroneous frames at the end of the video
            if interpolation_start_point != None:
                # Copy start point value to end
                frame_data_interp_start = csv_data[interpolation_start_point]
                interpolation_frame_indices = np.arange(interpolation_start_point + 1, num_frames)
                
                # Go through each feature-to-be-interpolated.
                for interp_feature_index in interpolation_feature_indices:    
                    for interp_frame_index in interpolation_frame_indices:
                        csv_data[interp_frame_index][interp_feature_index] = frame_data_interp_start[interp_feature_index]
                        
        # Initial information for the CSV file with interpolations
        csv_file_save_name = os.path.join(output_dir, filename.split('.')[0] + '.csv')
        csv_data = csv_data.tolist() # Convert Numpy array back to a list
        
        # Round everything to six digits to save disk space. Convert the first two
        # columns ('frame', 'face_id') and the last column ('interpolated') into integers
        for j in range(len(csv_data)):
            csv_data[j] = [round(num, 6) for num in csv_data[j]]
            for i in [0,1,-1]:
                csv_data[j][i] = int(csv_data[j][i])
        
        # Save a new version of the CSV file with interpolations performed
        with open(csv_file_save_name, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header_row)
            for row in csv_data:
                writer.writerow(row)
        
