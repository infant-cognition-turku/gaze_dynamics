# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

The configuration file for register_gaze_roi.py.

"""



import numpy as np


# The validity threshold (between 0 and 1) is the threshold which determines for which fixation CSV
# files we skip analysis based on the ratio of valid eye tracking video frames and the total number
# of video frames in the trial. For example, if the threshold is 0.5, we skip all fixation CSV files
# where the number of valid eye tracking frames is below 50% of the total number of frames in the trial.
fixation_validity_threshold = 0.85

# The calibration errors (in pixels) of the eye tracking measurements
x_dir_error = 24
y_dir_error = 36

# The directory of the fixation CSV files
fixations_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/fixations'

# The directory of the OpenFace's output CSV files
openface_output_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output_interpolated/'

# The directory of the videos that the test participants were looking at. HAS TO BE in the same framerate as
# the fixations CSV files (e.g. 120 Hz with some Tobii eye trackers)
video_file_dir = 'C:/Work/code/python_files/ml_pipeline/videos_same_resolution_upsampled/'

# Define the output directory of the gaze direction CSV files (will be automatically
# created if it does not already exist)
output_dir = './gaze_directions'




# Declare the indices of the desired facial landmarks from the OpenFace CSV file. If you don't modify
# OpenFace's source code, these do not need to be modified
right_eye_x_indices = np.arange(335,341) # 335, ..., 340
right_eye_y_indices = np.arange(403,409)
left_eye_x_indices = np.arange(341, 347)
left_eye_y_indices = np.arange(409,415)

nose_x_indices = np.arange(326,335)
nose_y_indices = np.arange(394, 403)

outer_mouth_x_indices = np.arange(347,359)
outer_mouth_y_indices = np.arange(415,427)

face_x_indices = np.arange(299,367)
face_y_indices = np.arange(367,435)

face_x_line_1_point_indices = np.array([299,300])
face_y_line_1_point_indices = np.array([367,368])
face_x_line_2_point_indices = np.array([314,315])
face_y_line_2_point_indices = np.array([382,383])

# The facial landmark point of the chin and nose bridge is needed for computing the height of the
# area that is added to the upper face (it is proportional to the size of the face)
chin_x_point_index = np.array([307])
chin_y_point_index = np.array([375])
nose_bridge_x_point_index = np.array([326])
nose_bridge_y_point_index = np.array([394])