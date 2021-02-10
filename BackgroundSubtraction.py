import os
import cv2
import numpy as np
import re


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    '''
    Returns the text in the natural numberical assending order
    '''
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]

def dilation(frame):
    '''
    Takes the frame  and perform dilation
    Inputs:
        frame: numpy array
    Parameters:
        dilated numpy array
    '''
    frame = cv2.dilate(frame, None, iterations=3)
    return frame

def gaussian_blur(frame):
    '''
    Takes the frames  and perform gaussian blur
    Parameters:
        frame: numpy array
    Returns:
        blured numpy array
    '''
    frame=cv2.GaussianBlur(frame, (21, 21), 0)
    return frame


def background_subtraction(frame1,frame2):
    '''
    Takes two frames  and perform background subtraction on frame2 based on frame1
    Parameters:
        frame1: numpy array
        frame2: numpy array
    Returns:
        binarized frame
    '''
    difference = cv2.absdiff(frame1, frame2)
    thresh = cv2.threshold(difference, 255/2, 255, cv2.THRESH_BINARY)[1]
    return thresh


def subtract_background(read_folder,write_folder):
    '''
    Reads the frames in gray scale from the folder, applies blur and saves new 
    frame in the folder
    
    Parameters:
        read_folder: folder to read frames from
        write_folder: folder to save frames after applying blur
    '''
    files = [file for file in os.listdir(read_folder)]
    sorted_files = sorted(files, key=natural_sort_key)
    
    frames = []
    for file in sorted_files:
        frame = cv2.imread(read_folder+file,0)
        frame = gaussian_blur(frame)
        frames.append(frame)
        
        
    first_frame = frames[0]
    frames_except_first = np.array(frames[1:])
    i = 0
    for frame in frames_except_first:
        frame = background_subtraction(frame,first_frame)
        frame = dilation(frame)
        cv2.imwrite(write_folder+str(i)+'.jpg',frame)
        i = i+1


#subtract_background( 'E:/FYP/fyp/Activity Recognition/frames/','E:/FYP/fyp/Activity Recognition/frames_bg/')