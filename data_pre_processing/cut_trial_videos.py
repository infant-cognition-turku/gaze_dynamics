# -*- coding: utf-8 -*-
"""
Einari Vaaras, UTU

Given a set of gaze data (trial events), cut the videos into trials using ffmpeg. Please note
that, if necessary, the input videos (containing multiple trials) need to be manually cut first,
so that the video starts from the first video frame of the first trial. Tools such as Shutter
Encoder can be used to first cut the videos, if necessary.

"""

import os
import sys
import csv
import platform
from tqdm import tqdm


##############################################################################################
################################# The configuration settings #################################
##############################################################################################

# The directory of the CSV files
csv_file_dir = './timestamp_csv_files'

# The directory of the input videos (uncut)
input_video_dir = './uncut_videos'

# The directory of the cut videos
output_video_dir = './cut_videos'

# The length of the clips we want to extract (in seconds)
clip_length_seconds = 4.00

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


# Find out the list of CSV files in the given directory
try:
    filenames_csv = os.listdir(csv_file_dir)
except FileNotFoundError:
    sys.exit('Given CSV file directory does not exist!')
    
csv_files = [filename for filename in filenames_csv if filename.endswith('.csv') or filename.endswith('.CSV')]
    
# Delete filenames_csv to save memory
del filenames_csv

# Check if there are no files in the given directories. Also check file number consistency
if len(csv_files) == 0:
    sys.exit('There are no CSV files in the given CSV file directory: ' + csv_file_dir)

# Perform the operation for all CSV files found in the given directory.
for csv_file in tqdm(csv_files):
    # Get the name of the CSV file
    filename_csv_file = os.path.join(csv_file_dir, csv_file)
    
    # Read the CSV file row by row, skip the header row
    data_array = []
    with open(filename_csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # Skip the header row
        header_row_original = next(csvreader)
        
        for row in csvreader:
            # Just in case, remove all '"' characters from each row
            row_cleaned = [s.replace('"', '') for s in row]
            data_array.append(row_cleaned)
    
    # We have two videos for each CSV file, one for trial stages 1 and 2, and another for trial stages 3 and 4.
    target_stages = [1, 2, 3, 4]
    
    for target_stage in target_stages:
        # Get the name of the video file. The format of the name of the CSV files is x_trialEvents.csv, and the 
        # format of the names of the video files is MiTrack_x_ETy.mp4, where in both x is the id of the test subject,
        # and in the video file y is either 1 or 2. If y is 1, then the video contains the trial stages 1 and 2, and
        # if y is 2, then the video contains the trial stages 3 and 4.
        x = csv_file.split('.')[0].split('_')[0]
        
        if target_stage < 3:
            y = '1'
        else:
            y = '2'
        
        video_namebase = 'MiTrack_' + x + '_ET' + y
        video_name = os.path.join(input_video_dir, video_namebase + '.mp4')
        video_write_name = os.path.join(output_video_dir, video_namebase)
        
        # Go through the values in the CSV file, split the videos accordingly.
        for row in data_array:
            trial = int(row[2])
            print(trial)
            stage = int(row[3])
            combination = row[8]
            
            # Find out the time instant of the first video frame. This needs to be determined for both original
            # videos (stages 1&2 and stages 3&4) separately. We have 32 trials in each CSV file altogether,
            # divided into four stages (trials 1-8 for stage 1, trials 9-16 for stage 2, trials 17-24 for
            # stage 3, and trials 25-32 for stage 4). Stages 1&2 were presented in a row, and stages 3&4 were
            # also presented in a row --> the time instants of the first frames are determined by the onset times
            # of trials 1 and 17.
            if trial == 1 or trial == 17:
                first_frame_time_instant = int(row[5])
            
            # We are only interested in the stage target_stage, so we omit other stages. Also, if combination
            # is -1, this means that we have not been able to reliably evaluate the time point at which the 
            # gaze of the test subject moved. Therefore, all cases where combination is -1 are left out, too.
            if stage == target_stage and combination != '-1':
                stimulation_filename = row[4]
                stimulation = stimulation_filename.split('.')[0]
                
                # Convert the start time from milliseconds to seconds. We want to extract clips of
                # length clip_length_seconds, starting from the start time.
                start_time = str((int(row[5]) - first_frame_time_instant) / 1000)
                
                # Perform the command (i.e., split the videos)
                if platform.system() == 'Windows':
                    command = ffmpeg_dir + '/ffmpeg -ss ' + start_time + ' -i ' + video_name + ' -t ' \
                              + str(clip_length_seconds) + ' -c:v libx264 -c:a aac -strict experimental -b:a 128k ' \
                              + video_write_name + '_' + str(trial) + '_' + stimulation + '.mp4'
                else:
                    command = 'ffmpeg -ss ' + start_time + ' -i ' + video_name + ' -t ' + str(clip_length_seconds) \
                              + ' -c:v libx264 -c:a aac -strict experimental -b:a 128k ' + video_write_name + '_' \
                              + str(trial) + '_' + stimulation + '.mp4'
                
                os.system(command)
    
    
    
    
