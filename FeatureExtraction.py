import os
import cv2
import math
import re
import numpy as np



def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    '''
    Returns the text in the natural numberical assending order
    '''
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]

def compute_points(image):
    '''
    Calculates head starting,left leg and right leg ending pointss
    Parameters:
        image: hough transformed image
    Returns:
        {'upper': upper,
        'lower_left': lower_left,
        'lower_right': lower_right}
    '''
    
    #upper most point
    upper = (0,0)
    key = 0
    for i in range(0,600):
        for j in range(0,800):
            #print(x,y, new_img[i][j] )
            if image[i][j] == 255.0:
                upper = (j,i)
                key =1
                break 
        if key == 1:
            break
    
    #lower left leg point
    lower_left = (0,0)
    key = 0

    for i in range(599,0,-1):
        for j in range(799,0,-1):
            if(image[i][j] == 255.0):
                lower_left =(j,i)
                key = 1
                break 
        if key == 1:
            break    
    
    # lower right leg point
    lower_right =(0,0)
    key = 0

    for i in range(599,upper[1],-1):
        for j in range(799,upper[0],-1):
            #print(i,j,new_img[i][j])
            if(image[i][j] == 255.0):
                lower_right =(j,i)
                key = 1
                break
        if key == 1:
            break
    
    return {
        'upper': upper,
        'lower_left': lower_left,
        'lower_right': lower_right
    }

def distance(p1, p2):
    '''
    Calculates distance between 2 point
    Parameters:
        p1: line with starting and ending point
        p2:
    Returns:
        magnitude of the line created by 2 points
    
    '''
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)


def ang(lineA, lineB):
    '''
    Calculates angle between 2 lines
    Parameters:
        lineA: line with starting and ending point
        lineB: line with starting and ending point
    Returns:
        angle: angle in degrees
    
    '''
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    angle = np.math.atan(np.linalg.det([vA,vB]),np.dot(vA,vB))
    return np.degrees(angle)

def feature_vector(upper , lower_left, lower_right):
    '''
    Given the 3 points, computes the centroid, angle,length of left leg and 
    length of right leg
    
    Parameters:
        upper: head starting point,
        lower_left: lower most point of left leg,
        lower_right:lower most point of right leg
    Returns: 
        [centroidX,centroidY,angle,length_left_leg,length_right_leg]
    '''
    centroid = int((upper[0]+ lower_left[0] + lower_right[0])/3) , int((upper[1]+ lower_left[1] + lower_right[1])/3)
    centroidX = centroid[0]
    centroidY = centroid[1]
    left_leg = [(centroid[0], centroid[1]), (lower_left[0], lower_left[1])]
    right_leg = [(centroid[0], centroid[1]), (lower_right[0], lower_right[1])]
    angle = ang(left_leg, right_leg)
    length_left_leg = distance(centroid,lower_left)
    length_right_leg = distance(centroid,lower_right)
    return [centroidX,centroidY,angle, length_left_leg, length_right_leg]


def feature_extraction(read_folder):
    '''
    extracts features(centroid,angle b/w legs, left leg length, right leg length)
    from each frame in the folder
    
    Parameters:
        read_folder: folder to read frames from
    Returns:
        features_per_video: list of all feature vectors in a video
    '''
    files = [file for file in os.listdir(read_folder)]
    sorted_files = sorted(files, key=natural_sort_key)
    i = 0
    features_per_video = []
    feature_header = ["centroidX", "centroidY","angleBetweenLegs", "lengthOfLeftLeg", "lengthOfRightLeg"]
    for file in sorted_files:
        frame = cv2.imread(read_folder+file,0)
        points = compute_points(frame)
        upper  = points['upper']
        lower_left  = points['lower_left']
        lower_right  = points['lower_right']
        if  all(upper) & all(lower_right) & all(lower_right):
            feature_vec = feature_vector(upper , lower_left, lower_right)
            print(i,feature_vec)
            features_per_video.append(feature_vec)
        i = i+1
    return features_per_video, feature_header

#features_per_video, feature_header = feature_extraction('E:/FYP/fyp/Activity Recognition/frames_hough/')
#print(features_per_video)