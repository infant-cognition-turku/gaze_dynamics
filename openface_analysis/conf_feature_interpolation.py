# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

The configuration file for feature_interpolation.py.

"""


# The minimum confidence of the face detections. Video frames with a lower confidence
# will be substituted with interpolated values. This is based on the confidence value
# of OpenFace's output.
confidence_threshold = 0.92

# The directory of OpenFace's output CSV files whose low confidence features are going to
# be interpolated
openface_csv_file_dir = 'C:/Work/code/python_files/ml_pipeline/openface_output/'

# The output directory of the interpolated CSV files (will be automatically created if it
# does not already exist). This SHOULD NOT be the same directory as openface_csv_file_dir
# if you do not want to overwrite the previous CSV files (as the names of the interpolated
# output CSV files will be the same as the names of the input CSV files).
output_dir = './openface_output_interpolated'

