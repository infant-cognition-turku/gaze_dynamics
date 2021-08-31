# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

The configuration file for pyfeat_blurface.py.

"""


# Defines what face we want to keep in the video (other faces are blurred).
# Options: 'keep_lower', 'keep_upper', 'keep_leftmost', 'keep_rightmost'
mode = 'keep_lower'

# Define the minimum confidence threshold for face detections, a value between 0 and 1.
# A high threshold will get rid of all false face detections, but might also get rid of
# proper face detections.
min_confidence = 0.5

# Define the blur type of unwanted faces.
# Options: 'black', 'white', 'gaussian' (aka a typical blur)
blur_type = 'gaussian'

# Define the directory of the input videos containing faces
video_dir = './blurface_test_video'

# Define the output directory of the videos with faces blurred (will be automatically
# created if it does not already exist)
output_dir = './output_blurface'

# Define the face detection model for PyFeat
# Options: 'retinaface', 'faceboxes', 'mtcnn'
face_model = 'retinaface'

