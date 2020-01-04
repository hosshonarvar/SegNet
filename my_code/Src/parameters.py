#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:07:51 2019

@author: hosseinhonarvar
"""

''' This module includes the functions needed to initialize and restore the 
    parameters of convolutional layers in the neural network
'''

import tensorflow as tf
import numpy as np
import os

# List of convolutional layers for each box in encoder and decoder
# conv_layers = {box number: number of conv layers}
conv_layers = {1:2, 2:2, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:2, 10:3}
num_boxes = len(conv_layers)+1

### Initialize parameters of each convolutional layer

def initialize_parameters_layer(num_classes):
        
    # Dictionary of filters for each conv layer
    filters_layers = {}
    for box_b in range(1, num_boxes):
        for layer_l in range(1, conv_layers[box_b]+1):
            
            filt_name = 'f_'+str(box_b)+'_'+str(layer_l)
        
            # No filter for pretrained VGG16 layers in encoder
            if box_b<=5:
                filters_layers.update([(filt_name, [])])
            
            # 13 filters for conv layers in decoder
            if box_b==6:
                filters_layers.update([(filt_name, [3, 3, 512, 512])])
            elif box_b==7:
                if layer_l<=2:
                    filters_layers.update([(filt_name, [3, 3, 512, 512])])
                elif layer_l==3:
                    filters_layers.update([(filt_name, [3, 3, 512, 256])])
            elif box_b==8:
                if layer_l<=2:
                    filters_layers.update([(filt_name, [3, 3, 256, 256])])
                elif layer_l==3:
                    filters_layers.update([(filt_name, [3, 3, 256, 128])])
            elif box_b==9:
                if layer_l==1:
                    filters_layers.update([(filt_name, [3, 3, 128, 128])])
                elif layer_l==2:
                    filters_layers.update([(filt_name, [3, 3, 128, 64])])
            elif box_b==10:
                if layer_l<=2:
                    filters_layers.update([(filt_name, [3, 3, 64, 64])])
                elif layer_l==3:
                    filters_layers.update([(filt_name, 
                                            [1, 1, 64, num_classes])])  
    
    return filters_layers

### Parameters of each convolutional layer in encoder
### Restore VGG16 pretrained parameters for encoder
    
def encoder_parameters_layer():
    
    # Load the VGG16 parameters 
    VGG16_layers = np.load('vgg16.npy', allow_pickle=True, 
                           encoding='latin1').item()
    
    # Dictionary of parameters for each conv layer
    # parameters_layers = {layer_name: [weights, biases]}
    parameters_layers = {}
    for box_b in range(1, num_boxes):
        for layer_l in range(1, conv_layers[box_b]+1):
                
            l_name = 'conv_'+str(box_b)+'_'+str(layer_l)
            l_name_VGG16 = 'conv'+str(box_b)+'_'+str(layer_l)
            w_name = 'w_'+str(box_b)+'_'+str(layer_l)
            b_name = 'b_'+str(box_b)+'_'+str(layer_l)
            
            # Weights (w) and biases (b)for pretrained VGG16 layers in encoder
            if box_b<=5:
                w_l = tf.Variable(VGG16_layers[l_name_VGG16][0], 
                                      trainable=False, name=w_name)
                b_l = tf.Variable(VGG16_layers[l_name_VGG16][1], 
                                      trainable=False, name=b_name)
                parameters_layers.update([(l_name, [w_l, b_l])]) 
    
    return parameters_layers

### Parameters of each convolutional layer in decoder
### Use trainable parameters for decoder
    
def decoder_parameters_layer(filters_layers, parameters_layers):
    
    # Dictionary of parameters for each conv layer
    # parameters_layers = {layer_name: [weights, biases]}
    for box_b in range(1, num_boxes):
        for layer_l in range(1, conv_layers[box_b]+1):
                        
            # Weights (w) and biases (b) to be trained in decoder
            if box_b>=6:
                
                l_name = 'conv_'+str(box_b)+'_'+str(layer_l)
                w_name = 'w_'+str(box_b)+'_'+str(layer_l)
                b_name = 'b_'+str(box_b)+'_'+str(layer_l)
                
                # Weights 
                filt_name = 'f_'+str(box_b)+'_'+str(layer_l)
                f_l = filters_layers[filt_name]
                w_l = tf.compat.v1.get_variable(name=w_name, 
                                      shape=f_l, 
                                      initializer=tf.initializers.he_normal(),
                                      trainable=True)
                
                # Biases 
                f_l_out_channels = f_l[-1]
                b_l = tf.compat.v1.get_variable(name=b_name, 
                                      shape=[f_l_out_channels], 
                                      initializer=tf.zeros_initializer,
                                      trainable=True)
                
                parameters_layers.update([(l_name, [w_l, b_l])])
    
    return parameters_layers

### Get parameters of all layers
    
def get_parameters(num_classes):
    
    # Initialize parameters of each layer
    filters_layers = initialize_parameters_layer(num_classes)
        
    # Parameters of each layer in encoder
    parameters_layers= encoder_parameters_layer()
    
    # Parameters of each layer in decoder
    parameters_layers = decoder_parameters_layer(
            filters_layers, parameters_layers)
    
    return parameters_layers

### Restore parameters of each convolutional layer in encoder and decoder
    
def restore_parameters_layer(model_save_dir, model_version):
            
    # Load the pretrained SegNet parameters 
    params_path = os.path.join(model_save_dir, 'Data', 'SegNet.npy')
    SegNet_layers = np.load(params_path, allow_pickle=True, encoding='latin1').item()
    
    # Dictionary of parameters for each conv layer
    # parameters_layers = {layer_name: [weights, biases]}
    parameters_layers = {}
    for box_b in range(1, num_boxes):
        for layer_l in range(1, conv_layers[box_b]+1):
                
            l_name = 'conv_'+str(box_b)+'_'+str(layer_l)
            w_name = 'w_'+str(box_b)+'_'+str(layer_l)
            b_name = 'b_'+str(box_b)+'_'+str(layer_l)
            
            # Weights (w) and biases (b)for pretrained VGG16 layers in encoder
            w_l = tf.Variable(SegNet_layers[l_name][0], 
                              trainable=False, name=w_name)
            b_l = tf.Variable(SegNet_layers[l_name][1], 
                              trainable=False, name=b_name)
            parameters_layers.update([(l_name, [w_l, b_l])])
            
    
    return parameters_layers