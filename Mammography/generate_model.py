from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 

class GENERATE_MODEL():
    def __init__(self):
        self.is_model = True
    
    #CNN Classifier Model Generation Which makes use of Images
    def __generate__model_CNN__(self, features, labels ,image_height, image_width, mode):


        #Input Layer
        input_layer = tf.reshape(features["x"], [-1, image_height, image_width, 1])

        #Convulutional Layer #1
        conv1 = tf.layers.conv2d(
            input=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu
        )

        #Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        #Convulutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            input=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu
        )

        #Pooling Layer #2
        pool2 - tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

        #Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training_mode == tf.estimator.ModeKeys.Train
        )

        predictions = {
            #Generate predicitions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            #Add `softmax_tensor` to the graph. It is used for PREDICT and the 
            #`logging_hook`.
            "probabilites": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Trainiing Op (for TRAIN Modde)
          if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        return metrics, model

    #RNN Classifier Model Generation
    def __generate_model_RNN__(self):
        metrics = []
        model = []
        return metrics, model

    #Evaluate CNN CLassifier
    def __evaluate__CNN__(self):
        metrics = []
        return metrics
    
    #Evaluate RNN CLassifier
    def __evaluate__RNN__(self):
        metrics = []
        return metrics