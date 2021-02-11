import pandas as pd
import numpy as np


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
    

#feature_header = ["ok","ok","ok","ok","ok"]
#feature_vector = [
#     [795, 346, 0.0, 22.02271554554524, 22.02271554554524],
#     [795, 346, 0.0, 22.02271554554524, 22.02271554554524]]
#class_name = "notOk"
#file_name = "areyouok.csv"

create_csv(feature_vector,feature_header, file_name, class_name)