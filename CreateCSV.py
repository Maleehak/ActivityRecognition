import numpy as np
import FeatureExtraction as fe
import pandas as pd



def create_csv(feature_vector,feature_header, file_name, class_name):
    '''
   create feature file from the given feature vector
    
    Parameters:
        feature_vector: numpy array
        feature_header: list of feature names
        file_name: path and filename to store vector in
        class_name: class of the features extracted
    Returns:
       None
    '''
    feature_vector = np.array(feature_vector)
    np.savetxt(file_name, feature_vector, delimiter=",")
    df = pd.read_csv(file_name, header=None)
    
    df.columns = feature_header
    df['class'] = class_name
    df.to_csv(file_name)
    

feature_vector, feature_header = fe.feature_extraction('E:/FYP/fyp/Activity Recognition/frames_hough/')
file_name = 'features.csv'
class_name = 'walking'
create_csv(feature_vector,feature_header, file_name, class_name)