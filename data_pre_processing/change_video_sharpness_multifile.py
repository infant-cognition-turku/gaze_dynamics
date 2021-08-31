# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Change the sharpness of all video files in a given directory using ffmpeg.

"""

import os
import sys
import platform


##############################################################################################
################################# The configuration settings #################################
##############################################################################################

# The directory of the video files that we want to sharpen
input_video_dir = 'C:/Work/code/python_files/ml_pipeline/videos_same_resolution'

# The directory where the sharpened video files will be stored (will be automatically created if
# it doesn't already exist)
output_video_dir = 'C:/Work/code/python_files/ml_pipeline/test_sharpen'

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


# Make sure that the input and output video directories are not the same
if input_video_dir == output_video_dir:
    sys.exit('Input and output video directories are the same! Change them to be different from each other.')

# Create output_video_dir if it doesn't exist
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)

# Find out the list of video files in the given directory
try:
    filenames_video = os.listdir(input_video_dir)
except FileNotFoundError:
    sys.exit('Given CSV file directory does not exist!')
    
video_files = [filename for filename in filenames_video if filename.endswith('.mp4') or filename.endswith('.MP4')]
    
# Delete filenames_csv to save memory
del filenames_video

# Check if there are no files in the given directories. Also check file number consistency
if len(video_files) == 0:
    sys.exit('There are no MP4 files in the given video file directory: ' + input_video_dir)
    
# Go through all video files in the given directory
for video_file in video_files:
    
    # Get the name of the video file
    video_file_name = os.path.join(input_video_dir, video_file)
    
    # Create a name for the new video-to-be-written
    video_write_name = os.path.join(output_video_dir, video_file)
    
    # Perform the operation
    if platform.system() == 'Windows':
        command = ffmpeg_dir + '/ffmpeg -i ' + video_file_name + ' -vf unsharp=5:5:5.0:5:5:5.0 -c:a copy ' + video_write_name
    else:
        command = 'ffmpeg -i ' + video_file_name + ' -vf unsharp=5:5:5.0:5:5:5.0 -c:a copy ' + video_write_name
    os.system(command)

