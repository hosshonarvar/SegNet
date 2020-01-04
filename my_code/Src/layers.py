#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:49:52 2019

@author: hosseinhonarvar
"""

import Src.operations as net_ops
import tensorflow as tf

def network_layers(input_layer, parameters_layers, train_flag):
    
    # Normalize the input images
    input_layer = tf.nn.lrn(input_layer, depth_radius=5, bias=1.0, 
                            alpha=0.0001, beta=0.75)
    
### Create encoder layers
    
    ### Box 1

    conv_1_1 = net_ops.conv2d_layer(input_layer, 'conv_1_1',
                                parameters_layers['conv_1_1'][0], 
                                parameters_layers['conv_1_1'][1], 
                                train_flag)
    
    conv_1_2 = net_ops.conv2d_layer(conv_1_1, 'conv_1_2',
                                parameters_layers['conv_1_2'][0], 
                                parameters_layers['conv_1_2'][1], 
                                train_flag)
    
    pool_1_1, pool_1_1_indices = net_ops.maxpool2d_indices_layer(conv_1_2)
    
    ### Box 2
            
    conv_2_1 = net_ops.conv2d_layer(pool_1_1, 'conv_2_1',
                                parameters_layers['conv_2_1'][0], 
                                parameters_layers['conv_2_1'][1], 
                                train_flag)
    
    conv_2_2 = net_ops.conv2d_layer(conv_2_1, 'conv_2_2',
                                parameters_layers['conv_2_2'][0], 
                                parameters_layers['conv_2_2'][1], 
                                train_flag)
    
    pool_2_1, pool_2_1_indices = net_ops.maxpool2d_indices_layer(conv_2_2)
    
    ### Box 3
            
    conv_3_1 = net_ops.conv2d_layer(pool_2_1, 'conv_3_1',
                                parameters_layers['conv_3_1'][0], 
                                parameters_layers['conv_3_1'][1], 
                                train_flag)
    
    conv_3_2 = net_ops.conv2d_layer(conv_3_1, 'conv_3_2',
                                parameters_layers['conv_3_2'][0], 
                                parameters_layers['conv_3_2'][1], 
                                train_flag)
    
    conv_3_3 = net_ops.conv2d_layer(conv_3_2, 'conv_3_3',
                                parameters_layers['conv_3_3'][0], 
                                parameters_layers['conv_3_3'][1], 
                                train_flag)
    
    pool_3_1, pool_3_1_indices = net_ops.maxpool2d_indices_layer(conv_3_3)
    
    ### Box 4
            
    conv_4_1 = net_ops.conv2d_layer(pool_3_1, 'conv_4_1',
                                parameters_layers['conv_4_1'][0], 
                                parameters_layers['conv_4_1'][1], 
                                train_flag)
    
    conv_4_2 = net_ops.conv2d_layer(conv_4_1, 'conv_4_2',
                                parameters_layers['conv_4_2'][0], 
                                parameters_layers['conv_4_2'][1], 
                                train_flag)
    
    conv_4_3 = net_ops.conv2d_layer(conv_4_2, 'conv_4_3',
                                parameters_layers['conv_4_3'][0], 
                                parameters_layers['conv_4_3'][1], 
                                train_flag)
    
    pool_4_1, pool_4_1_indices = net_ops.maxpool2d_indices_layer(conv_4_3)
    
    ### Box 5
            
    conv_5_1 = net_ops.conv2d_layer(pool_4_1, 'conv_5_1', 
                                parameters_layers['conv_5_1'][0], 
                                parameters_layers['conv_5_1'][1], 
                                train_flag)
    
    conv_5_2 = net_ops.conv2d_layer(conv_5_1, 'conv_5_2',
                                parameters_layers['conv_5_2'][0], 
                                parameters_layers['conv_5_2'][1], 
                                train_flag)
    
    conv_5_3 = net_ops.conv2d_layer(conv_5_2, 'conv_5_3',
                                parameters_layers['conv_5_3'][0], 
                                parameters_layers['conv_5_3'][1], 
                                train_flag)
    
    pool_5_1, pool_5_1_indices = net_ops.maxpool2d_indices_layer(conv_5_3)
    
### Create decoder layers
    
    ### Box 6
    
    upsample_6_1 = net_ops.upsample2d_layer(pool_5_1, pool_5_1_indices, 
                                            conv_5_3)
    
    conv_6_1 = net_ops.conv2d_layer(upsample_6_1, 'conv_6_1',
                                parameters_layers['conv_6_1'][0], 
                                parameters_layers['conv_6_1'][1], 
                                train_flag)
    
    conv_6_2 = net_ops.conv2d_layer(conv_6_1, 'conv_6_2',
                                parameters_layers['conv_6_2'][0], 
                                parameters_layers['conv_6_2'][1], 
                                train_flag)
    
    conv_6_3 = net_ops.conv2d_layer(conv_6_2, 'conv_6_3',
                                parameters_layers['conv_6_3'][0], 
                                parameters_layers['conv_6_3'][1], 
                                train_flag)
    ### Box 7
    
    upsample_7_1 = net_ops.upsample2d_layer(conv_6_3, pool_4_1_indices, 
                                            conv_4_3)
    
    conv_7_1 = net_ops.conv2d_layer(upsample_7_1, 'conv_7_1',
                                parameters_layers['conv_7_1'][0], 
                                parameters_layers['conv_7_1'][1], 
                                train_flag)
    
    conv_7_2 = net_ops.conv2d_layer(conv_7_1, 'conv_7_2',
                                parameters_layers['conv_7_2'][0], 
                                parameters_layers['conv_7_2'][1], 
                                train_flag)
    
    conv_7_3 = net_ops.conv2d_layer(conv_7_2, 'conv_7_3',
                                parameters_layers['conv_7_3'][0], 
                                parameters_layers['conv_7_3'][1], 
                                train_flag)
    
    ### Box 8
    
    upsample_8_1 = net_ops.upsample2d_layer(conv_7_3, pool_3_1_indices, 
                                            conv_3_3)
    
    conv_8_1 = net_ops.conv2d_layer(upsample_8_1, 'conv_8_1', 
                                parameters_layers['conv_8_1'][0], 
                                parameters_layers['conv_8_1'][1], 
                                train_flag)
    
    conv_8_2 = net_ops.conv2d_layer(conv_8_1, 'conv_8_2', 
                                parameters_layers['conv_8_2'][0], 
                                parameters_layers['conv_8_2'][1], 
                                train_flag)
    
    conv_8_3 = net_ops.conv2d_layer(conv_8_2, 'conv_8_3', 
                                parameters_layers['conv_8_3'][0], 
                                parameters_layers['conv_8_3'][1], 
                                train_flag)
    
    ### Box 9
    
    upsample_9_1 = net_ops.upsample2d_layer(conv_8_3, pool_2_1_indices, 
                                            conv_2_2)
    
    conv_9_1 = net_ops.conv2d_layer(upsample_9_1, 'conv_9_1', 
                                parameters_layers['conv_9_1'][0], 
                                parameters_layers['conv_9_1'][1], 
                                train_flag)
    
    conv_9_2 = net_ops.conv2d_layer(conv_9_1, 'conv_9_2', 
                                parameters_layers['conv_9_2'][0], 
                                parameters_layers['conv_9_2'][1], 
                                train_flag)
    
    ### Box 10
    
    upsample_10_1 = net_ops.upsample2d_layer(conv_9_2, pool_1_1_indices, 
                                             conv_1_2)
    
    conv_10_1 = net_ops.conv2d_layer(upsample_10_1, 'conv_10_1', 
                                 parameters_layers['conv_10_1'][0], 
                                 parameters_layers['conv_10_1'][1], 
                                 train_flag)

    conv_10_2 = net_ops.conv2d_layer(conv_10_1, 'conv_10_2', 
                                 parameters_layers['conv_10_2'][0], 
                                 parameters_layers['conv_10_2'][1], 
                                 train_flag)
    
    ### Logits
    
    conv_10_3 = tf.nn.conv2d(conv_10_2, parameters_layers['conv_10_3'][0], 
                             strides=[1, 1, 1, 1], padding='SAME')
    conv_10_3 = tf.nn.bias_add(conv_10_3, parameters_layers['conv_10_3'][1])
    logits = conv_10_3
    
    return logits