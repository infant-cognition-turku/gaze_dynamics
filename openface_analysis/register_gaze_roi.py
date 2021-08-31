# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Register whether the gaze is pointed at the left/right eye, the nose, the
mouth, the face, or the background. Report the gaze direction in a CSV file.
If the script is suspended, it will continue computation from where it last
left off the next time it starts running again.

"""

import csv
import os
import sys
import numpy as np
import cv2
import warnings
from tqdm import tqdm
from importlib.machinery import SourceFileLoader



# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python register_gaze_roi.py \nOR \n2) python register_gaze_roi.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
else:
    try:
        import conf_register_gaze_roi as conf
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python register_gaze_roi.py \nOR \n2) python register_gaze_roi.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_register_gaze_roi.py" in the same directory 
        as "register_gaze_roi.py"''')




def compute_max_pixel_dist(hull, pixel_indices):
    """
    Returns the maximum pixel distance value between a set of pixels and a hull that
    the pixels are compared against. A larger pixel distance value means that the pixel
    is closer to the center of the hull.
    
    _______________________________________________________________________________________
    Input:
        
    hull: The visual hull that the given pixel indices are compared against. Must be in the
          format created by the function cv2.convexHull().
    pixel_indices: The pixel indices that are tested against the visual hull.
    _______________________________________________________________________________________
    Output:
        
    max_pixel_dist: The maximum pixel distance value.

    """
    
    max_pixel_dist = -np.inf
    for pixel in pixel_indices:
        
        # Test how close the pixel is to the given hull. If pixel_dist < 0, then the
        # tested pixel is not inside the hull (smaller value -> further away from the
        # hull). If pixel_dist == 0, then the tested pixel is on the border of the hull.
        # If pixel_dist > 0, then the tested pixel is inside the hull (larger value ->
        # closer to the hull center).
        pixel_dist = cv2.pointPolygonTest(hull, (int(pixel[1]), int(pixel[0])), True)
        
        if pixel_dist > max_pixel_dist:
            max_pixel_dist = pixel_dist
    
    return max_pixel_dist



def compute_convex_hull(facial_landmark_indices, frame_features):
    """
    Returns the visual hull of a set of facial landmark points given the OpenFace features
    of a video frame and the (x,y)-indices of the desired facial landmark points.
    
    _______________________________________________________________________________________
    Input:
        
    facial_landmark_indices: The (x,y)-indices of the desired facial landmark points
    frame_features: The OpenFace features of the given video frame
    
    _______________________________________________________________________________________
    Output:
        
    hull: The visual hull of the set of given facial landmark points

    """
    
    points = []
    for index in facial_landmark_indices:
        points.append((int(frame_features[index[0]]), int(frame_features[index[1]])))
    hull = cv2.convexHull(np.array(points))
    
    return hull



def get_two_additional_facial_landmarks(chin_point, nose_point, frame_features, previous_coefficients_line_1, 
                                        previous_coefficients_line_2):
    """
    Get two additional points for the 68 facial landmarks to make the face ROI larger (we take 
    the forehead into consideration). These two points are computed based on the facial landmarks
    with indices 8 (tip of chin) and 27 (nose bridge).
    
    ______________________________________________________________________________________________
    Input:
    
    chin_point: The (x,y)-coordinate of landmark 8
    nose_point: The (x,y)-coordinate of landmark 27
    frame_features: The OpenFace features of the given video frame
    previous_coefficients_line_1: The previous coefficients of line 1
    previous_coefficients_line_2: The previous coefficients of line 2
    ______________________________________________________________________________________________
    Output:
    
    x_1, x_2: The x-coordinates of the two additional facial landmarks
    y_1, y_2: The y-coordinates of the two additional facial landmarks
    coefficients_line_1: The coefficients of line 1
    coefficients_line_2: The coefficients of line 2
    
    """
    
    coefficients_line_1 = previous_coefficients_line_1
    coefficients_line_2 = previous_coefficients_line_2
    
    # Get the line equations for the lines passing the points, first get the actual points
    face_x_line_1_points = frame_features[face_x_line_1_point_indices].astype(int)
    face_y_line_1_points = frame_features[face_y_line_1_point_indices].astype(int)
    face_x_line_2_points = frame_features[face_x_line_2_point_indices].astype(int)
    face_y_line_2_points = frame_features[face_y_line_2_point_indices].astype(int)
    
    # Some error handlings are needed to ensure that the line equation can also be created
    # if the points are on top of each other
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            coefficients_line_1 = np.polyfit(face_x_line_1_points, face_y_line_1_points, 1)
        except np.RankWarning:
            if coefficients_line_1 is not None:
                pass
            
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            coefficients_line_2 = np.polyfit(face_x_line_2_points, face_y_line_2_points, 1)
        except np.RankWarning:
            if coefficients_line_2 is not None:
                pass
    
    
    # Determine height-to-be-added based on facial landmarks (indices 0 and 8) in order
    # to take the size of the head and the height of the frame into account
    diff_points_8_and_27 = abs(chin_point[1] - nose_point[1])
    diff_points_adjusted_to_frame_height = diff_points_8_and_27/height
    scaling_factor = 2.0
    num_higher_pixels = int(diff_points_adjusted_to_frame_height*diff_points_8_and_27*scaling_factor)
    
    
    # Find the two points that are num_higher_pixels pixels higher and on the two lines
    y_1 = int(face_y_line_1_points[0] - num_higher_pixels)
    if y_1 < 0:
        y_1 = 0
    if coefficients_line_1 is not None:
        x_1 = int((y_1 - coefficients_line_1[1])/coefficients_line_1[0])
    else:
        x_1 = int(face_x_line_1_points[0])
    
    y_2 = int(face_y_line_2_points[1] - num_higher_pixels)
    if y_2 < 0:
        y_2 = 0
    if coefficients_line_2 is not None:
        x_2 = int((y_2 - coefficients_line_2[1])/coefficients_line_2[0])
    else:
        x_2 = int(face_x_line_2_points[1])
    
    
    
    return x_1, x_2, y_1, y_2, coefficients_line_1, coefficients_line_2



if __name__ == '__main__':

    # Take parameter values from the configuration file
    fixation_validity_threshold = conf.fixation_validity_threshold
    x_dir_error = conf.x_dir_error
    y_dir_error = conf.y_dir_error
    fixations_csv_file_dir = conf.fixations_csv_file_dir
    openface_output_csv_file_dir = conf.openface_output_csv_file_dir
    video_file_dir = conf.video_file_dir
    output_dir = conf.output_dir
    
    right_eye_x_indices = conf.right_eye_x_indices
    right_eye_y_indices = conf.right_eye_y_indices
    left_eye_x_indices = conf.left_eye_x_indices
    left_eye_y_indices = conf.left_eye_y_indices
    nose_x_indices = conf.nose_x_indices
    nose_y_indices = conf.nose_y_indices
    outer_mouth_x_indices = conf.outer_mouth_x_indices
    outer_mouth_y_indices = conf.outer_mouth_y_indices
    face_x_indices = conf.face_x_indices
    face_y_indices = conf.face_y_indices
    face_x_line_1_point_indices = conf.face_x_line_1_point_indices
    face_y_line_1_point_indices = conf.face_y_line_1_point_indices
    face_x_line_2_point_indices = conf.face_x_line_2_point_indices
    face_y_line_2_point_indices = conf.face_y_line_2_point_indices
    chin_x_point_index = conf.chin_x_point_index
    chin_y_point_index = conf.chin_y_point_index
    nose_bridge_x_point_index = conf.nose_bridge_x_point_index
    nose_bridge_y_point_index = conf.nose_bridge_y_point_index
    
    # Create the directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Make some error handling
    try:
        filenames_csv = os.listdir(fixations_csv_file_dir)
    except FileNotFoundError:
        sys.exit('Given CSV file directory does not exist!')
        
    try:
        filenames_video = os.listdir(video_file_dir)
    except FileNotFoundError:
        sys.exit('Given video file directory does not exist!')
        
    csv_files = [filename for filename in filenames_csv if filename.endswith('.csv')]
    
    # Delete filenames_csv to save memory
    del filenames_csv
    
    video_files = [filename for filename in filenames_video if filename.endswith('.mp4') or filename.endswith('.MP4')]
    
    # Check if there are no files in the given directories.
    if len(csv_files) == 0:
        sys.exit('There are no CSV files in the given CSV file directory: ' + fixations_csv_file_dir)
        
    if len(video_files) == 0:
        sys.exit('There are no video files in the given video file directory: ' + video_file_dir)
    
    # Transpose the index vectors
    right_eye_indices = np.vstack((right_eye_x_indices, right_eye_y_indices)).T
    left_eye_indices = np.vstack((left_eye_x_indices, left_eye_y_indices)).T
    nose_indices = np.vstack((nose_x_indices, nose_y_indices)).T
    outer_mouth_indices = np.vstack((outer_mouth_x_indices, outer_mouth_y_indices)).T
    face_indices = np.vstack((face_x_indices, face_y_indices)).T
    
    # Read CSV files, process each one by one
    csv_counter = 1
    for filename in csv_files:
        csv_file = os.path.join(fixations_csv_file_dir, filename)
        csv_data = []
        print('\nComputing gaze direction for file: ' + filename + ', file ' + str(csv_counter) + '/' + str(len(csv_files)))
        csv_counter += 1
        
        # Read the input CSV file
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row = next(csvreader)
            
            for row in csvreader:
                csv_data.append(row)
        
        # We skip the iterations in which there is no gaze data available
        if len(csv_data) == 0:
            print('    Skipped analysis of the file ' + filename + ' since there is no valid fixation data in the file')
            continue
        
        # Convert the gaze data into Numpy format and retreive basic information
        gazedata = np.array(csv_data)
        trial_name = (gazedata[0][0]).split('.')[0]
        video_id = gazedata[0][1]
        video_total_valid_time = gazedata[0][2]
        video_total_trial_time = gazedata[0][3]
        
        # We skip the iterations in which the CSV fails to exceed the validity threshold
        if (float(video_total_valid_time)/float(video_total_trial_time)) < fixation_validity_threshold:
            print('    Skipped analysis of the file ' + filename + ' based on the validity threshold of ' + str(fixation_validity_threshold))
            continue
        
        # The basic information for the new CSV file with gaze directions
        write_name = os.path.join(output_dir, trial_name + '_gazedirs.csv')
        new_header = ['frame', 'video_id', 'total_valid_time', 'total_trial_time', 'fixation_x', 'fixation_y', 
                      'fixation_x_error', 'fixation_y_error', 'gaze_label']
        
        # If the new CSV file with gaze directions already exists in the output directory,
        # we skip the analysis for the file
        if os.path.isfile(write_name):
            print('    Skipped analysis of file ' + filename + ' since its analysis file already exists in ' + output_dir)
            continue
        
        # Find out the video file of the gaze data
        videofile_name = os.path.join(video_file_dir, gazedata[0][1])
        
        # Start capturing the video
        cap = cv2.VideoCapture(videofile_name)
        
        # Check if instantiation was successful
        if not cap.isOpened():
            raise Exception("Could not open video file: " + videofile_name)
            
        # Get the number of frames in video
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get the height and width of the video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Close capture
        cap.release()
        
        openface_csv_file = os.path.join(openface_output_csv_file_dir, gazedata[0][1].split('.')[0] + '.csv')
        openface_csv_data = []
        
        # Read the OpenFace CSV file
        with open(openface_csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            
            # Skip the header row
            header_row_openface = next(csvreader)
            
            for row in csvreader:
                openface_csv_data.append(row)
        
        # We skip the iterations in which there are no OpenFace features available
        if len(openface_csv_data) == 0:
            print('    Skipped analysis of the file ' + filename + ' since there are no valid OpenFace CSV features for the file')
            continue
        
        # Convert the data into Numpy format for easier handling
        openface_data = np.array(openface_csv_data, dtype=np.float32)
        
        # Initialize line coefficients
        coefficients_line_1 = None
        coefficients_line_2 = None
        
        # Initialize the list containing the data for the output gaze CSV file
        output_data = []
        
        # Start going through the video and gaze data, and add gaze points to desired frames
        video_frame_index = 0
        
        # Initialize a progress bar using the tqdm library
        for data_row in tqdm(gazedata):
            
            fixation_x_position = int(float(data_row[10]))
            fixation_y_position = int(float(data_row[11]))
            
            # Take measurement error into account
            fixation_x_left_position = fixation_x_position - x_dir_error
            fixation_x_right_position = fixation_x_position + x_dir_error
            fixation_y_upper_position = fixation_y_position - y_dir_error
            fixation_y_lower_position = fixation_y_position + y_dir_error
            
            fixation_first_frame_index = int(float(data_row[5]))
            fixation_last_frame_index = int(float(data_row[6]))
            
            # Perform some error prevention
            if (fixation_first_frame_index - 1) >= num_frames_video:
                # Create the new CSV file with gaze directions
                with open(write_name, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(new_header)
                    for row in output_data:
                        writer.writerow(row)
                break
            
            if (fixation_last_frame_index - 1) >= num_frames_video:
                fixation_last_frame_index = num_frames_video - 1
                
            fixation_frame_indices = np.arange(fixation_first_frame_index, fixation_last_frame_index + 1)
            
            # We are behind the first frame index of the fixation, so we catch up
            if video_frame_index < (fixation_first_frame_index - 1):
                while video_frame_index < (fixation_first_frame_index - 1):
                    # The frame does not have gaze data available
                    gaze_label = 'not_available'
                    video_frame_index += 1
                    output_data.append([str(video_frame_index), video_id, video_total_valid_time, video_total_trial_time,
                                        str(None), str(None), str(x_dir_error), str(y_dir_error), gaze_label])
            
            # We should not be ahead of the first frame index, so we raise an error if we are.
            if video_frame_index > fixation_first_frame_index:
                sys.exit('Something went wrong when processing the file.')
                
            # Go through the frames in the fixation and add gaze points to the video
            for video_frame_index in fixation_frame_indices:
                # Handle rounding errors of last video frames in the CSV file
                if (video_frame_index - 1) < num_frames_video:
                    frame_features = openface_data[video_frame_index-1]
                else:
                    frame_features = openface_data[num_frames_video-1]
                    
                # Find convex hull for given points
                hull_right_eye = compute_convex_hull(right_eye_indices, frame_features)
                hull_left_eye = compute_convex_hull(left_eye_indices, frame_features)
                hull_nose = compute_convex_hull(nose_indices, frame_features)
                hull_outer_mouth = compute_convex_hull(outer_mouth_indices, frame_features)
                
                points_face = []
                for index in face_indices:
                    points_face.append((int(frame_features[index[0]]), int(frame_features[index[1]])))
                
                # Get two additional points for points_face to make the face ROI larger (take also forehead
                # into consideration)
                chin_point = np.array((int(frame_features[chin_x_point_index]), int(frame_features[chin_y_point_index])))
                nose_point = np.array((int(frame_features[nose_bridge_x_point_index]), int(frame_features[nose_bridge_y_point_index])))
                
                x_1, x_2, y_1, y_2, coefficients_line_1, coefficients_line_2 = get_two_additional_facial_landmarks(
                    chin_point, nose_point, frame_features, coefficients_line_1, coefficients_line_2)
                
                # Add the two points to the face points and compute the convex hull for the face
                points_face.append((x_1, y_1))
                points_face.append((x_2, y_2))
                hull_face = cv2.convexHull(np.array(points_face))
                
                polygon_points_gaze = np.array([[fixation_x_left_position,fixation_y_position], [fixation_x_position,fixation_y_upper_position], \
                                           [fixation_x_right_position,fixation_y_position], [fixation_x_position,fixation_y_lower_position]])
                hull_gaze = cv2.convexHull(polygon_points_gaze)
                    
                # Paint the convex hull with white to find the indices of the white pixels in the next step
                black_frame_gaze = np.zeros((height, width, 3)).astype(np.uint8)
                cv2.fillPoly(black_frame_gaze, [hull_gaze], (255, 255, 255))
                
                # Create a mask by using numpy boolean indexing, it will produce a mask with
                # True/False values inside, it will be True is the pixel value is white
                mask_gaze = black_frame_gaze[:,:,0] == 255
                
                # Get all pixel indices of the gaze region
                gaze_pixel_indices_temp = np.where(mask_gaze == True)
                gaze_pixel_indices = np.vstack((gaze_pixel_indices_temp[0], gaze_pixel_indices_temp[1])).T
                
                if len(gaze_pixel_indices) == 0:
                    # Handle cases when the test subject's gaze is validly registered, but the gaze is directed
                    # towards a point that is outside of the screen.
                    gaze_label = 'other'
                else:
                    # Go through all the gaze pixels. Find the one closest to the given hull.
                    pixel_dist_right_eye = compute_max_pixel_dist(hull_right_eye, gaze_pixel_indices)
                    pixel_dist_left_eye = compute_max_pixel_dist(hull_left_eye, gaze_pixel_indices)
                    pixel_dist_nose = compute_max_pixel_dist(hull_nose, gaze_pixel_indices)
                    pixel_dist_mouth = compute_max_pixel_dist(hull_outer_mouth, gaze_pixel_indices)
                    pixel_dist_face = compute_max_pixel_dist(hull_face, gaze_pixel_indices)
                    
                    # Find out if the gaze is aimed at the eyes, the nose, or the mouth
                    pixel_dist_vector = np.array([pixel_dist_right_eye, pixel_dist_left_eye, pixel_dist_nose, pixel_dist_mouth])
                    is_facial_roi_watched = pixel_dist_vector >= 0
                    facial_roi_watched_indices = np.where(is_facial_roi_watched == True)[0]
                    
                    if len(facial_roi_watched_indices) == 0:
                        # Facial ROIs are not looked at, so we will find out whether the gaze
                        # is aimed at the face in general
                        if pixel_dist_face < 0:
                            # The face is not looked at either
                            gaze_label = 'other'
                        else:
                            # The face is looked at
                            gaze_label = 'face'
                    else:
                        # At least one of the facial ROIs is looked at, so we try to identify
                        # the one with the minimum distance (i.e. maximum value)
                        facial_roi_min_dist_index = np.argmax(pixel_dist_vector)
                        
                        if facial_roi_min_dist_index == 0:
                            gaze_label = 'right_eye'
                        elif facial_roi_min_dist_index == 1:
                            gaze_label = 'left_eye'
                        elif facial_roi_min_dist_index == 2:
                            gaze_label = 'nose'
                        else:
                            gaze_label = 'mouth'
                        
                output_data.append([str(video_frame_index), video_id, video_total_valid_time, video_total_trial_time,
                                        str(fixation_x_position), str(fixation_y_position), str(x_dir_error), 
                                        str(y_dir_error), gaze_label])
                        
        # Create the new CSV file with gaze directions
        with open(write_name, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(new_header)
            for row in output_data:
                writer.writerow(row)
                
