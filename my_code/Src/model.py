#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:16:17 2019

@author: hosseinhonarvar
"""

""" This module includes the SegNet class for training and testing the model.
"""

import json
import os
import tensorflow as tf
import Src.input as net_input
import Src.parameters as net_params
import Src.layers as net_layers
import Src.operations as net_ops
import numpy as np

class SegNet:
    
    def __init__(self, cofig_file):
        
        ### Read configuration file

        with open(cofig_file, 'r') as f:
            config = json.load(f)    
        self.train_paths_file = config['TRAIN_PATHS_FILE']
        self.val_paths_file = config['VAL_PATHS_FILE']
        self.test_paths_file = config['TEST_PATHS_FILE']
        self.model_save_dir = config['MODEL_SAVE_DIR']
        self.model_version = config['MODEL_VERSION']
        self.input_h = config['INPUT_HEIGHT']
        self.input_w = config['INPUT_WIDTH']
        self.input_c = config['INPUT_CHANNELS']
        self.num_classes = config['NUM_CLASSES']
        self.VGG_flag = config['VGG_FLAG']
        self.VGG_file = config['VGG_FILE']
        self.tb_logs = config['TB_LOGS']
        self.batch_size = config['BATCH_SIZE']
        self.opti_algo = config['OPTI_ALGO']
        self.num_epochs = config['NUM_EPOCHS']
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            
            ### Create dataset of inputs X and outputs Y
                
            # Placeholder for X and Y paths to create paths dataset 
            self.X_pl = tf.placeholder(tf.string)
            self.Y_pl = tf.placeholder(tf.string)
            self.is_train_pl = tf.placeholder(tf.bool)
            self.batch_size_pl = tf.placeholder(tf.int64, shape=[])
            
            dataset = net_input.create_dataset(self.X_pl, self.Y_pl, 
                                               self.input_h, self.input_w, 
                                               self.input_c, self.num_classes, 
                                               self.batch_size_pl)
            
            # Iterator for dataset
            self.iterator = tf.compat.v1.data.make_initializable_iterator(
                    dataset)
            self.X, self.Y = self.iterator.get_next()
            
            ### Get the parameters of each layer
        
            self.parameters_layers = net_params.get_parameters(
                    self.num_classes)
            
            ### Compute the network logits
            
            self.logits = net_layers.network_layers(
                    self.X, self.parameters_layers, 
                    train_flag=self.is_train_pl)
            
    def train(self):
        
        with self.graph.as_default():
            
            ### Execute computational graph for training
            
            with self.sess.as_default():
                
                # Get lists of X and Y paths from paths files
                X_train_paths, Y_train_paths, num_examples= \
                net_input.read_paths_file(paths_file=self.train_paths_file)
                print('Number of examples for training:', \
                      self.sess.run(num_examples))
                
                num_batches = tf.math.floor(tf.math.divide(num_examples, 
                                                           self.batch_size_pl))
                
                train_op, predictions_flat, loss_minibatch, \
                global_acc_minibatch, mIOU_minibatch = net_ops.evaluations(
                        self.Y, self.logits, self.num_classes)
                
                tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                
                # Initialize all tf variables 
                init_g = tf.compat.v1.global_variables_initializer()
                init_l = tf.compat.v1.local_variables_initializer()
                self.sess.run([init_g, init_l])
                
                train_cost, train_acc, train_mIOU = [], [], []
                            
                # Loop over number of epochs
                for i_epoch in range(self.num_epochs):
                    
                    # Initialize iterator with training data paths
                    self.sess.run(self.iterator.initializer, feed_dict={
                            self.X_pl: X_train_paths, self.Y_pl: Y_train_paths,
                            self.batch_size_pl: self.batch_size})
                    
                    cost_batch_tot = 0.
                    global_acc_minibatch_tot = 0.
                    mIOU_minibatch_tot = 0.
    
                    # Tarin the model
                    try:
                        
                        # Run over number of batches till dataset exhausted
                        while True:
                            
                            fetches=[train_op, loss_minibatch, 
                                     global_acc_minibatch, mIOU_minibatch, 
                                     self.X, self.Y, predictions_flat, 
                                     num_batches]
                            feed_dict={self.batch_size_pl: self.batch_size,
                                       self.is_train_pl: True}
                        
                            _, loss_minibatch_val, glob_acc_minibatch_val, \
                            mIOU_minibatch_val, X_val, Y_val, pred_val, \
                            num_batches_val = self.sess.run(fetches, feed_dict)
                            
                            cost_batch_tot += loss_minibatch_val/ \
                            num_batches_val
                            global_acc_minibatch_tot += \
                            glob_acc_minibatch_val[0]/num_batches_val
                            mIOU_minibatch_tot += mIOU_minibatch_val[0]/ \
                            num_batches_val
                                
                    except tf.errors.OutOfRangeError:
                        pass
                    
                    train_cost.append(cost_batch_tot)
                    print("Epoch:", i_epoch+1, ", cost=", "{:.5f}".format(
                            cost_batch_tot))
                    
                    train_acc.append(global_acc_minibatch_tot)
                    print("Epoch:", i_epoch+1, ", acc=", "{:.5f}".format(
                            global_acc_minibatch_tot))
                    
                    train_mIOU.append(mIOU_minibatch_tot)
                    print("Epoch:", i_epoch+1, ", mIOU=", "{:.5f}".format(
                            mIOU_minibatch_tot))
                    
                # Save the model
                saver = tf.compat.v1.train.Saver()
                checkpoint_path = os.path.join(self.model_save_dir, 
                                               'model.ckpt')
                saver.save(self.sess, checkpoint_path, 
                           global_step=self.model_version)
                np.save(os.path.join(self.model_save_dir, 'Data', 
                                     'train_cost'), train_cost)
                np.save(os.path.join(self.model_save_dir, 'Data', 
                                     'train_acc'), train_acc)
                np.save(os.path.join(self.model_save_dir, 'Data', 
                                     'train_mIOU'), train_mIOU)
                
    def inference(self):
                
        with self.graph.as_default():
                    
            ### Execute computational graph for testing

            with self.sess.as_default():
                
                # Get lists of X and Y paths from paths files
                X_test_paths, Y_test_paths, num_examples= \
                net_input.read_paths_file(paths_file=self.test_paths_file)
                print('Number of examples for testing:', \
                      self.sess.run(num_examples))
                
                num_batches = tf.math.floor(tf.math.divide(num_examples, 
                                                           self.batch_size_pl))
                
                
                _, predictions_flat, _, global_acc_minibatch, mIOU_minibatch \
                = net_ops.evaluations(self.Y, self.logits, self.num_classes)
                
                # Initialize local tf variables 
                init_l = tf.compat.v1.local_variables_initializer()
                self.sess.run([init_l])
                
                
                # Initialize iterator with test data paths
                self.sess.run(self.iterator.initializer, feed_dict={
                        self.X_pl:X_test_paths, 
                        self.Y_pl:Y_test_paths, 
                        self.batch_size_pl: 1})
                
                global_acc_minibatch_tot = 0.
                mIOU_minibatch_tot = 0.
                        
                # Test the model
                try:
                    
                    # Run over number of batches till dataset exhausted
                    while True:
                        fetches=[global_acc_minibatch, mIOU_minibatch, 
                                 self.X, self.Y, predictions_flat, num_batches]
                        feed_dict={self.batch_size_pl: 1,  
                                   self.is_train_pl: False}
                        
                        glob_acc_minibatch_val, \
                        mIOU_minibatch_val, X_val, Y_val, pred_val, \
                        num_batches_val = self.sess.run(fetches, feed_dict)
                        global_acc_minibatch_tot += glob_acc_minibatch_val[0] \
                        /num_batches_val
                        mIOU_minibatch_tot += mIOU_minibatch_val[0] \
                        /num_batches_val
                            
                except tf.errors.OutOfRangeError:
                    pass
                    
            print(", acc=", "{:.5f}".format(global_acc_minibatch_tot))
            print(", mIOU=", "{:.5f}".format(mIOU_minibatch_tot))
