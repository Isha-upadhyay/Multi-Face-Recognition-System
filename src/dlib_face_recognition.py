# -*- coding: utf-8 -*-


#importing the required libraries
import pickle  #for loading the encodings
import time   #for time calculations
import cv2   #for image processing
import face_recognition     # for face recognition and face detection   
import numpy as np  #for numerical calculations and array operations  

from database_pandas import store_inferred_face_in_dataframe, df_inferred_faces #for storing the inferred faces in the dataframe  
from parameters import NUMBER_OF_TIMES_TO_UPSAMPLE, \
                       DLIB_FACE_ENCODING_PATH, \
                       FRAME_HEIGHT, FRAME_WIDTH, \
                       FACE_MATCHING_TOLERANCE, \
                       BATCH_SIZE, \
                       INFERENCE_BUFFER_SIZE  
from custom_logging import logger
from locks import lock #for thread synchronization  

# load the known faces and embeddings
logger.info("[INFO] loading encodings...")  
data = pickle.loads(open(DLIB_FACE_ENCODING_PATH,"rb").read())   #load the encodings from the file

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]  
known_face_names = data["names"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []   
all_face_encodings = []  
all_face_names = [] 
all_processed_frames = []   


def single_frame_face_recognition(frame, frame_downsample, number_of_times_to_upsample, model, face_matching_tolerance):  
    '''Single frame face recognition function
    
    Arguments:
        frame {numpy array} -- frame to be processed
        frame_downsample {bool} -- whether to downsample the frame or not
        number_of_times_to_upsample
        model -- face detection model
        face_matching_tolerance -- tolerance for face matching
    
    Returns:
        a processed frame
    '''

    if frame_downsample:  #true
        #resize the current frame to 1/4 size to proces faster
        #current_frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        
        #resize the frame to FRAME_WIDTH*FRAME_HEIGHT to display the video if frame is too big
        # frame.shape[1] is width and frame.shape[0] is height
        if frame.shape[1] > FRAME_WIDTH or frame.shape[0] > FRAME_HEIGHT:
            current_frame_small = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
        else:
            current_frame_small = frame
    else:
        #consider the frame as it is
        current_frame_small = frame

    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample,model) 
        
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)  #face_encodings returns a list of 128-dimensional face encodings (one for each face in the image)

    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings): #zip is used to iterate over two lists simultaneously
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        if frame_downsample: 
            #change the position maginitude to fit the actual size video frame
            '''top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4'''
            pass
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        # Find the best match (smallest distance to a known face)
        best_match_index = np.argmin(all_matches) #returns the index of the smallest element in the array
        # If the best match is within tolerance, use the name of the known face
        if all_matches[best_match_index] <= face_matching_tolerance: #0.45 is the tolerance
            name_of_person = known_face_names[best_match_index] #get the name of the person
            #save the name of the person in the dataframe
            store_inferred_face_in_dataframe(name_of_person, all_matches[best_match_index]) #store the inferred face in the dataframe
        else:
            name_of_person = 'Unknown face' #if the face is not recognized then it is an unknown face
            #save the name of the person in the dataframe as unknown face

        # For known face use green color and for unknown face use red color
        if name_of_person == 'Unknown face':
            color = (0,0,255) #Red
        else:
            color = (0,255,0) #Green
        
        #draw rectangle around the face    
        cv2.rectangle(current_frame_small,(left_pos,top_pos),(right_pos,bottom_pos),color,5) #draw rectangle around the face
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX #font style
        cv2.putText(current_frame_small, name_of_person, (left_pos,bottom_pos), font, 1, (255,255,255),2) #put text in the image
    
    yield current_frame_small #return the processed frame to the main thread



def multi_frame_face_recognition(frames_buffer, frame_downsample=True, number_of_times_to_upsample=1, model='hog', face_matching_tolerance=0.45): #hog is used for CPU and cnn is used for GPU
    '''
        Multi frame face recognition function

        Arguments:
            frames_buffer {list} -- list of frames to be processed
            frame_downsample {bool} -- whether to downsample the frame or not
            number_of_times_to_upsample
            model {string} -- model to be used for face detection
            face_matching_tolerance {float} -- tolerance for face matching

        Yields:
            processed frames
    '''
    #pop first frame from frames_buffer to get the first frame
    while True: # Loop until we have a frame to process
        if len(frames_buffer) > 0: #check if there are frames in the buffer
            _ = frames_buffer.pop(0) #pop the first frame from the buffer to get the first frame  
            break   #break the loop if we have successfully read one frame from stream

    # Continue looping until there are no more frames to process.
    while True: # Loop until we have a frame to process 
        
        #if is_stream:
          # check if there are frames in the buffer

        if len(frames_buffer) > 0: #check if there are frames in the buffer
          # Read the next frame from the buffer
          #pop first frame from frames_buffer 
          img0 = frames_buffer.pop(0) #pop the first frame from the buffer to get the first frame
          if img0 is None: #if the frame is empty then
            continue 
          ret = True #we have successfully read one frame from stream 
          if len(frames_buffer) >= 10: #if the buffer has more than 10 frames then
            frames_buffer.clear() #clear the buffer if it has more than 10 frames to avoid memory overflow
        else:
          # buffer is empty, nothing to do
          continue

        #if we are able to read a frame then process it
        if ret:
            #yield the processed frame to the main thread
            yield from single_frame_face_recognition(frame= img0,  #current frame
                                                     frame_downsample=frame_downsample,  #downsample the frame
                                                     number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,  #number of times to upsample the image looking for faces
                                                     model=model,   #face detection model
                                                     face_matching_tolerance=face_matching_tolerance) #face matching tolerance
            

def batched_frame_face_recognition(frames_buffer):
    '''This function is to be used for GPU based face recognition. It performs face detection on a batch of frames at a time.
    
    Arguments:
        frames_buffer {list} -- list of frames to be processed
    
    Returns:
        None
    '''
    while True:
    
        # Wait until there are at least BATCH_SIZE frames in frames_buffer
        while len(frames_buffer) < BATCH_SIZE: #check if the buffer has BATCH_SIZE frames to process 1
            time.sleep(0.01) #wait for 0.01 seconds

        #Find start time for batch processing 
        tick = time.time() #get the current time ---- time recorded in seconds
            
        # Slice first BATCH frames from frames_buffer
        with lock: #lock the buffer
            batched_frame_buffer = frames_buffer[:BATCH_SIZE] #get the first BATCH_SIZE frames from the buffer
            # Remove first BATCH_SIZE frames from frames_buffer to avoid processing them again
            del frames_buffer[:BATCH_SIZE] #delete the first BATCH_SIZE frames from the buffer

        #extract batch of frames, cam names and cam ips from the batched_frame_buffer 
        batch_of_frames = [batch[0] for batch in batched_frame_buffer] #get the frames from the batched_frame_buffer
        batch_of_cam_names = [batch[1] for batch in batched_frame_buffer] #get the camera names from the batched_frame_buffer
        batch_of_cam_ips = [batch[2] for batch in batched_frame_buffer] #get the camera ips from the batched_frame_buffer

        # Use exception handling to catch any errors that might occur
        try:
            batch_of_face_locations = face_recognition.batch_face_locations(batch_of_frames, 
                                                                        number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                                        batch_size=BATCH_SIZE)
        
            for i, all_face_locations_single_frame in enumerate(batch_of_face_locations):

                #detect face encodings for all the faces detected
                all_face_encodings_single_frame = face_recognition.face_encodings(batch_of_frames[i],all_face_locations_single_frame)

                #looping through the face locations and the face embeddings
                for _, current_face_encoding in zip(all_face_locations_single_frame,all_face_encodings_single_frame):
                    #splitting the tuple to get the four position values of current face
                    #top_pos,right_pos,bottom_pos,left_pos = current_face_location
                    
                    '''if frame_downsample:
                        #change the position maginitude to fit the actual size video frame
                        top_pos = top_pos*4
                        right_pos = right_pos*4
                        bottom_pos = bottom_pos*4
                        left_pos = left_pos*4
                        pass'''
                    
                    #find all the matches and get the list of matches
                    all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
                    # Find the best match (smallest distance to a known face)
                    best_match_index = np.argmin(all_matches)
                    # If the best match is within tolerance, use the name of the known face
                    if all_matches[best_match_index] <= FACE_MATCHING_TOLERANCE:
                        name_of_person = known_face_names[best_match_index]
                        #save the person details in the dataframe
                        store_inferred_face_in_dataframe(name_of_person, all_matches[best_match_index], batch_of_cam_names[i], batch_of_cam_ips[i])
                        print(df_inferred_faces)
        
        except Exception as e:
            logger.error('Error in batch_face_locations: {}'.format(e))

        # Find end time for batch processing
        tock = time.time()

        # Calculate time taken for batch processing
        time_taken = tock - tick
        logger.info(f'Time taken for batch processing of {BATCH_SIZE} frames = {time_taken} seconds')

        # Todo: Purge frames_buffer if it has more than INFERENCE_BUFFER_SIZE frames to avoid memory overflow
        frames_buffer_size = len(frames_buffer)
        if frames_buffer_size > INFERENCE_BUFFER_SIZE:
            with lock:
                logger.warning(f'Frames buffer size: {frames_buffer_size}. Purging frames_buffer...')
                frames_buffer.clear()
