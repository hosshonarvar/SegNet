#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:11:03 2019

@author: hosseinhonarvar
"""

""" This module includes all the operations needed for SegNet """

import tensorflow as tf

### Convolutional layer

def conv2d_layer(input_layer, name, weights_layer, biases_layer, train_flag):
    
    # Perform 2D convolution
    output_layer = tf.nn.conv2d(
            input_layer, weights_layer, strides=[1, 1, 1, 1], padding='SAME')
    
    # Add biases to the output of 2D convolution
    output_layer = tf.nn.bias_add(output_layer, biases_layer)
    
    # Define the dynamic shape to handle the batch_size place holder with 
    # unknown (?) value
    shape_dyn = [tf.shape(output_layer)[k] for k in range(4)]
    output_layer = tf.reshape(output_layer, [shape_dyn[0], 
                                             output_layer.shape[1], 
                                             output_layer.shape[2], 
                                             output_layer.shape[3]]) 
    
    # Perfrom batch normalization depending on the training flag
    batch_norm_true = tf.keras.layers.BatchNormalization(trainable=True)(
            output_layer, training=True)
    batch_norm_false = tf.keras.layers.BatchNormalization(trainable=False)(
            output_layer, training=False)
    output_layer = tf.cond(train_flag, lambda: batch_norm_true, 
                           lambda: batch_norm_false)
    
    # Evaluate the ReLu activation function
    output_layer = tf.nn.relu(output_layer)
    
    return output_layer

### Max-pool layer and indices
    
def maxpool2d_indices_layer(input_layer):
    
    output_layer, output_layer_indices = tf.nn.max_pool_with_argmax(
            input_layer, ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1], padding='SAME', include_batch_in_index=True)
    
    return output_layer, output_layer_indices

### Upsample layer
    
def upsample2d_layer(input_layer, input_layer_indices, output_layer_shape):
    
    input_layer = tf.reshape(input_layer, [-1])
    input_layer_indices = tf.expand_dims(tf.reshape(input_layer_indices, [-1]), 
                                         axis=1)
    
    # Get the output shape by flattening the dynamic shape 
    shape_temp = [ tf.shape(output_layer_shape)[k] for k in range(4)]      
    output_layer_shape_flat = tf.reshape(output_layer_shape, [
            shape_temp[0]*shape_temp[1]*shape_temp[2]*shape_temp[3]])
    output_layer_shape_flat_dyn = tf.cast(tf.shape(output_layer_shape_flat),
                                          tf.int64)
    
    # Scatter the the data
    output_layer = tf.scatter_nd(input_layer_indices, input_layer, 
                                 output_layer_shape_flat_dyn)
    
    # Reshape the scattered data from 1D to 4D
    output_layer = tf.reshape(output_layer, [shape_temp[0], 
                                             output_layer_shape.shape[1],
                                             output_layer_shape.shape[2],
                                             output_layer_shape.shape[3]])
    
    return output_layer

### Define the evaluations: parameters optimizer, predictions, and metrics

def evaluations(Y, logits, num_classes):
    
    logits_reshape = tf.reshape(logits, [-1, num_classes])
    Y_flat = tf.cast(tf.reshape(Y, [-1]), tf.int64)
    Y_one_hot = tf.one_hot(Y_flat, depth=num_classes)
    
    # Softmax classifier and loss (cross entropy)
    loss_minibatch= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y_one_hot, logits=logits_reshape))
    
    # Training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        # Optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer()
        
        # Minimize the loss
        train_op = optimizer.minimize(loss_minibatch)
        
    # Predictions
    predictions_flat = tf.math.argmax(logits_reshape, axis=1)

    # Global accuracy 
    global_acc_minibatch = tf.compat.v1.metrics.accuracy(
            labels=Y_flat, predictions=predictions_flat)
    
    # Class accuracy 
    mIOU_minibatch = tf.compat.v1.metrics.mean_iou(
            labels=Y_flat, predictions=predictions_flat, 
            num_classes=num_classes)
    
    return train_op, predictions_flat, loss_minibatch, global_acc_minibatch, \
           mIOU_minibatch