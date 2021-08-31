# -*- coding: utf-8 -*-

"""
Einari Vaaras, UTU

In a video, keep only one face (leftmost/rightmost/uppest/lowest) and blur
the rest of the faces. Utilizes PyFeat for face detection. See the website
https://py-feat.org/content/intro.html on how to install PyFeat.

"""

from feat import Detector
import os
import glob
import numpy as np
import cv2
import sys
import copy
import time
from tqdm import tqdm
from importlib.machinery import SourceFileLoader


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python pyfeat_blurface.py \nOR \n2) python pyfeat_blurface.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
else:
    try:
        import conf_pyfeat_blurface as conf
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python pyfeat_blurface.py \nOR \n2) python pyfeat_blurface.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_pyfeat_blurface.py" in the same directory 
        as "pyfeat_blurface.py"''')


if __name__ == '__main__':

    # Take parameter values from the configuration file
    mode = conf.mode
    min_confidence = conf.min_confidence
    blur_type = conf.blur_type
    video_dir = conf.video_dir
    output_dir = conf.output_dir
    face_model = conf.face_model
    
    # Create the directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the face detector
    detector = Detector(face_model = face_model, landmark_model = 'mobilenet', au_model = 'logistic', emotion_model = 'svm')
    
    # Find all videos in the given video file directory
    videofiles = glob.glob(os.path.join(video_dir, '*.mp4'))
    
    if len(videofiles) == 0:
        videofiles = glob.glob(os.path.join(video_dir, '*.MP4'))
        if len(videofiles) == 0:
            sys.exit('There are no MP4 video files in the given directory, check the parameter "video_dir"!')
    
    
    # We go through all the videos in the given directory
    video_counter = 1
    for videofile in videofiles:
        
        print('\nBlurring unwanted faces for file: ' + videofile + ', file ' + str(video_counter) + '/' + str(len(videofiles)))
        video_counter += 1
        
        start_time = time.time()
        
        # Get the face predictions from PyFeat. Convert them to Numpy format
        print('Detecting faces from each video frame... This may take a while...')
        video_predictions = detector.detect_video(videofile)
        print('Done!')
        face_predictions = video_predictions.to_numpy()
        print('Blurring faces...')
        
        # Delete large variable to save memory
        del video_predictions
        
        # Remove the last row of face_predictions (PyFeat falsely includes a row of NaN
        # values as the last frame)
        face_predictions = np.delete(face_predictions, -1, axis=0)
        
        # Delete the weakest face detections to avoid having false detections
        deletable_rows = []
        i = 0
        for prediction in face_predictions:
            face_prediction_condifence = prediction[5]
            if face_prediction_condifence < min_confidence:
                deletable_rows.append(i)
            i += 1
        
        if len(deletable_rows) > 0:
            face_predictions = np.delete(face_predictions, deletable_rows, axis=0)
        
        # Start video capture
        cap = cv2.VideoCapture(videofile)
        
        # Get basic information from the video
        fps_v = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # The basic information about the new video-to-be-written with blurred faces
        write_name = os.path.join(output_dir, (videofile.split(os.sep)[-1]).split('.')[0] + '_blurred.mp4')
        out = cv2.VideoWriter(write_name, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width, height))
        
        # Go through the video frame by frame. Initialize a progress bar using the tqdm library.
        for video_frame_index in tqdm(range(num_frames_video)):
            _, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = copy.deepcopy(frame_rgb)
            
            # Compute the blurred version of the image
            if blur_type == 'gaussian':
                blurred_frame = cv2.GaussianBlur(frame_rgb, (21,21), cv2.BORDER_DEFAULT)
            
            # Extract from face_predictions only those detections which involve the
            # present video frame
            frame_predictions = []
            for element in face_predictions:
                index = element[0]
                if index == video_frame_index:
                    frame_predictions.append(element)
            frame_predictions = np.array(frame_predictions)
                
            shape_predictions = frame_predictions.shape
            
            # If there are face detections in the present frame, leave the leftmost/rightmost/topmost/lowest
            # face detections and blur the rest. Find out the face we want to keep unblurred.
            if shape_predictions[0] > 0:
                if mode == 'keep_lower':
                    # Find the lowest face and blur the rest of the faces. Note that in Python,
                    # pixel indexing starts from the top left corner of the image.
                    face_rectangle_bottomleft_corners_y = frame_predictions[:,2] + frame_predictions[:,4]
                    index_desired_face = np.argmax(face_rectangle_bottomleft_corners_y)
                
                elif mode == 'keep_upper':
                    # Find the highest face and blur the rest of the faces. Note that in Python,
                    # pixel indexing starts from the top left corner of the image.
                    face_rectangle_topleft_corners_y = frame_predictions[:,2]
                    index_desired_face = np.argmin(face_rectangle_topleft_corners_y)
                    
                elif mode == 'keep_leftmost':
                    # Find the leftmost face and blur the rest of the faces. Note that in Python,
                    # pixel indexing starts from the top left corner of the image.
                    face_rectangle_topleft_corners_x = frame_predictions[:,1]
                    index_desired_face = np.argmin(face_rectangle_topleft_corners_x)
                    
                elif mode == 'keep_rightmost':
                    # Find the rightmost face and blur the rest of the faces. Note that in Python,
                    # pixel indexing starts from the top left corner of the image.
                    face_rectangle_topright_corners_x = frame_predictions[:,1] + frame_predictions[:,3]
                    index_desired_face = np.argmax(face_rectangle_topright_corners_x)
                    
                else:
                    sys.exit('Wrong argument for the parameter "mode"!')
                    
                # Blur all faces except the face with the index index_desired_face
                i = 0
                for prediction in frame_predictions:
                    if i != index_desired_face:
                        try:
                            face_rectangle_x = int(prediction[1])
                            face_rectangle_y = int(prediction[2])
                            face_rectangle_width = int(prediction[3])
                            face_rectangle_height = int(prediction[4])
                            
                            if blur_type == 'gaussian':
                                # Assign a designated area of the blurred image to the original image
                                frame_rgb[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :] = \
                                blurred_frame[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :]
                            elif blur_type == 'black':
                                # First find out the minimum value of the image data type, and then assign that
                                # for the designated pixels
                                min_pixel_value = np.iinfo(frame_rgb.dtype).min
                                frame_rgb[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :] = min_pixel_value
                            elif blur_type == 'white':
                                # First find out the maximum value of the image data type, and then assign that
                                # for the designated pixels
                                max_pixel_value = np.iinfo(frame_rgb.dtype).max
                                frame_rgb[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :] = max_pixel_value
                            else:
                                sys.exit('Wrong argument for the parameter "blur_type"!')
                        except ValueError:
                            pass
                        
                    i += 1
                
                # We want to have the desired face completely visible, so if the blurred faces overlap with
                # that face, we ensure that the original face is visible
                i = 0
                for prediction in frame_predictions:
                    if i == index_desired_face:
                        try:
                            face_rectangle_x = int(prediction[1])
                            face_rectangle_y = int(prediction[2])
                            face_rectangle_width = int(prediction[3])
                            face_rectangle_height = int(prediction[4])
                            frame_rgb[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :] = \
                                original_frame[face_rectangle_y:(face_rectangle_y + face_rectangle_height), face_rectangle_x:(face_rectangle_x + face_rectangle_width), :]
                        except ValueError:
                            pass
                    i += 1
            
            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Close capture and video write
        cap.release()
        out.release()
        
        # Delete large variable to save memory
        del face_predictions
        
        end_time = time.time() - start_time
        print_text = 'Elapsed time for processing ' + videofile + ':'
        print(print_text, round(end_time, 2), 'seconds')
