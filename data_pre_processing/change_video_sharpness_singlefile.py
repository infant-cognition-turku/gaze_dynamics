# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Change the sharpness of a single video file using ffmpeg.

"""

import os
import platform


##############################################################################################
################################# The configuration settings #################################
##############################################################################################

# The video that we want to sharpen
input_video_file = 'C:/Work/code/python_files/ml_pipeline/videos_same_resolution/facebreak9.mp4'

# The directory where the sharpened video file will be stored (will be automatically created if
# it doesn't already exist)
output_video_file_dir = 'C:/Work/code/python_files/ml_pipeline/test_sharpen'

# The name of the sharpened video file
output_video_file_name = 'sharpened.mp4'

# ONLY FOR WINDOWS USERS, Mac and Linux users don't need to care about the variable "ffmpeg_dir"
# (Mac and Linux users, check online how to install ffmpeg software if you don't have it yet).
# The directory of ffmpeg executable (ffmpeg.exe), can be either built by yourself (check online 
# how to do so), or the much easier route is to download Shutter Encoder software and use the 
# executable (ffmpeg.exe) from there. Note that the directory where the executable is CANNOT
# contain any whitespaces, or else ffmpeg won't work.
ffmpeg_dir = 'E:/Shutter_Encoder/Library'

##############################################################################################
##############################################################################################
##############################################################################################


# Create output_video_dir if it doesn't exist
if not os.path.exists(output_video_file_dir):
    os.makedirs(output_video_file_dir)

# Perform the sharpening operation
video_write_name = os.path.join(output_video_file_dir, output_video_file_name)
if platform.system() == 'Windows':
    command = ffmpeg_dir + '/ffmpeg -i ' + input_video_file + ' -vf unsharp=5:5:5.0:5:5:5.0 -c:a copy ' + video_write_name
else:
    command = 'ffmpeg -i ' + input_video_file + ' -vf unsharp=5:5:5.0:5:5:5.0 -c:a copy ' + video_write_name
os.system(command)

