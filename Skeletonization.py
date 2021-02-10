import os
import cv2
import numpy as np
import re

def skeletonize(frame):
    '''
    Applies cross morphological structuring on image   
    Parameters:
        frame: numpy array
    Returns:
        skel: numpy array after skeletonization
    '''
    size = np.size(frame)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    skel=np.zeros(frame.shape,np.uint8)
    while( not done):
        eroded = cv2.erode(frame,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(frame,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = eroded.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True
    return skel

    
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    '''
    Returns the text in the natural numberical assending order
    '''
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]



def frames_skeletonization(read_folder,write_folder):
    '''
    Applies skeletonization on each frame in read folder and saves skeletonized 
    frame in the folder
    
    Parameters:
        read_folder: folder to read frames from
        write_folder: folder to save frames after skeletonization
    '''
    files = [file for file in os.listdir(read_folder)]
    sorted_files = sorted(files, key=natural_sort_key)
    
    i = 0
    for file in sorted_files:
        frame = cv2.imread(read_folder+file,0)
        skel = skeletonize(frame)
        cv2.imwrite(write_folder+str(i)+'.jpg',skel)
        i = i+1
        
#frames_skeletonization('E:/FYP/fyp/Activity Recognition/frames_bg/', 'E:/FYP/fyp/Activity Recognition/frames_skel/')

