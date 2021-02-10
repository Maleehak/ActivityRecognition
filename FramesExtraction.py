import cv2

def capture_frames(video , folder):
    '''
    Captures all the frames from the video and stores them in a folder
    Parameters:
        video : path of video
        folder : path of folder
    Returns:
        None
    '''
    i = 0
    cap = cv2.VideoCapture(video)
    scaling_factorx=5
    scaling_factory=5
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.resize(frame,None,fx=scaling_factorx,fy=scaling_factory,interpolation=cv2.INTER_AREA)
        cv2.imwrite(folder+str(i)+'.jpg',frame)
        i = i+1 

#Example 
#capture_frames('E:/FYP/fyp/Activity Recognition/walking.avi' , 'E:/FYP/fyp/Activity Recognition/frames/')