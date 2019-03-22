from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 

os.getcwd()
os.listdir(os.getcwd())

class GENERATE_MODEL():
    def __init__(self):
        self.is_model = True

    #RNN Classifier Model Generation
    def __generate__model_RNN__(self):
        metrics = []
        model = []
        return metrics, model

    #DNN Classifier Model Generation
    def __generate_model_DNN__(self):
        pass
    
    def __generate__Regressor_Classifier(self):
        pass

    #Evaluate RNN Classifier
    def __evaluate_RNN__(self):
        metrics = []
        return metrics
    
    #Evaluate CNN Classifier
    def __evaluate_CNN__(self):
        metrics = []
        return metrics

    #Evaluate DNN CLassifier
    def __evaluate_DNN__(self, input_x ,features_train ,features_test, labels_train , labels_test, batch_size):
        estimator = tf.estimator.DNNClassifier(
                    feature_columns=input_x,
                    hidden_units=[1024, 512, 256, 64],
                    optimizer=tf.train.ProximalAdagradOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=0.001
                    ))
        

        def input_fn_train(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset
            
        
        def input_fn_eval(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset


        def input_fn_predict(ftrs, lbls, b_sze):
            dataset = tf.data.Dataset.from_tensor_slice((dict(ftrs)), lbls)
            dataset = dataset.shuffle(1000).repeat().batch(b_sze)
            return dataset
        
        input_fn_train = input_fn_train(features_train, labels_train, batch_size)
        input_fn_eval = input_fn_eval(features_test, labels_test, batch_size)
        input_fn_predict = input_fn_predict(features_test, labels_test, batch_size)

        estimator.train(input=input_fn_train)
        metrics = estimator.evaluate(input_fn=input_fn_eval)
        predictions = estimator.predict(input_fn=input_fn_predict)
        return metrics, predictions