from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 

os.getcwd()
os.listdir(os.getcwd())

class CREATE_DATASET():
    def __init__(self):
        self.is_data = True

    def __obtain_data_set(self, path):
        input_x = []
        input_y = []
        return input_x, input_y

    def __obtain_train_test_set(self, input_x, input_y):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        return x_train, x_test, y_train, y_test

    