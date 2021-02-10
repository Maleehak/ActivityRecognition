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


def hough_transform(skeleton):
    '''
    Applies hough transform on skeletonized image   
    Parameters:
        frame:  skeletonized numpy array
    Returns:
        new_img: new image with hough transformations
    '''
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    new_img = np.zeros((skeleton.shape[0],skeleton.shape[1]))
    v = np.median(skeleton)
    sigma = 0.1

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v)) 

    #canny edge
    edges = cv2.Canny(skeleton, lower, upper)
    dilated_edges = cv2.dilate(edges,element)
    lines = cv2.HoughLinesP(dilated_edges, rho=2, theta=np.pi/180,
                            threshold=120, minLineLength=10, maxLineGap=56)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return new_img


def hough_transformation(read_folder,write_folder):
    '''
    Applies hough transform on each frame in read folder and saves skeletonized 
    frame in the folder
    
    Parameters:
        read_folder: folder to read frames from
        write_folder: folder to save frames after hough transform
    '''
    files = [file for file in os.listdir(read_folder)]
    sorted_files = sorted(files, key=natural_sort_key)
    i = 0
    for file in sorted_files:
        frame = cv2.imread(read_folder+file,0)
        skel = hough_transform(frame)
        cv2.imwrite(write_folder+str(i)+'.jpg',skel)
        i = i+1
        
#hough_transformation( 'E:/FYP/fyp/Activity Recognition/frames_skel/', 'E:/FYP/fyp/Activity Recognition/frames_hough/')