from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 
from sklearn.model_selection import train_test_split

os.getcwd()
os.listdir(os.getcwd())

class CREATE_DATASET():
    def __init__(self):
        self.is_data = True
    
    def __read_csv__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        return data_set
    
    def __obtain_data__(self, path, number_features, number_labels):
        data_set = pd.read_csv(path, low_memory=False)
        input_x = data_set.iloc[ : , number_labels:(number_features+number_labels)]
        input_y = data_set.iloc[ : , 0:number_labels]
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = 0)

        return input_x, x_train, x_test, y_train, y_test
