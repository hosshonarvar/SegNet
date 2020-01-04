#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:40:30 2019

@author: hosseinhonarvar
"""

''' This module includes the functions needed to create the dataset from 
input
'''

import tensorflow as tf

### Make lists for X and Y paths

def read_paths_file(paths_file):
    X_paths, Y_paths = [], []
    
    # Read the file with paths of images and labels 
    data = open(paths_file, 'r').read().splitlines()
    num_examples = 0
    for d in data:
        num_examples +=1
        X_paths.append(d.split(' ')[0]) # images paths
        Y_paths.append(d.split(' ')[1]) # labels paths
        
    return X_paths, Y_paths, tf.constant(num_examples, tf.int64)
    
### Parse every element of dataset (X,Y)
    
def parse_dataset(X_elem_path, Y_elem_path, input_height, input_width, 
                  num_input_channels, num_classes):
    
    # Read the content of each image
    X_elem_content = tf.read_file(X_elem_path)
    Y_elem_content = tf.read_file(Y_elem_path)
    
    # Decode png files to pixels
    X_elem_decoded = tf.image.decode_png(X_elem_content, 
                                         channels=num_input_channels)
    Y_elem_decoded = tf.image.decode_png(Y_elem_content, channels=1) # greyscale
    
    # Resize each decoded image
    X_elem_resized = tf.image.resize_images(
            X_elem_decoded, [input_height, input_width])
    Y_elem_resized = tf.squeeze(tf.image.resize_images(
            Y_elem_decoded, [input_height, input_width]))
    
    # Change the data type
    X_elem = tf.cast(X_elem_resized, tf.float32)
    Y_elem = tf.cast(Y_elem_resized, tf.int32)
    
    return X_elem, Y_elem

### Create the batches of dataset (X, Y) 

def create_dataset(placeholder_X, placeholder_Y, input_height, input_width, 
                   num_input_channels, num_classes, batch_size):
    
    dataset_paths = tf.data.Dataset.from_tensor_slices((placeholder_X, 
                                                        placeholder_Y))
    # Create the X and Y dataset
    dataset = dataset_paths.map(lambda X_elem, Y_elem: parse_dataset(
            X_elem, Y_elem, input_height, input_width, 
            num_input_channels, num_classes))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset