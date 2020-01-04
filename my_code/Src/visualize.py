#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:52:27 2019

@author: hosseinhonarvar
"""

''' This module includes the functions needed to visualize the metrics and 
    results.
'''

import matplotlib.pyplot as plt 
import numpy as np
import os

def plot_metrics(model_save_dir):
    
    plot_files = ['train_cost.npy', 'train_acc.npy', 'train_mIOU.npy']
    plot_names = ['Cost', 'Global Accuracy', 'Mean Intersection over Union']

    plt.figure()
    for idx, i_file in enumerate(plot_files):
        plot_path = os.path.join(model_save_dir, 'Data', i_file)
        train_y = np.load(plot_path, allow_pickle=True, encoding='latin1')
        num_epochs = list(range(1, len(train_y)+1))
        plt.subplot(1, 3, idx+1)
        plt.plot(num_epochs, train_y, lw=2)
        plt.ylabel(plot_names[idx])
        plt.xlabel('Epoch')
        
    plt.rcParams['figure.figsize'] = (15,3)
    plt.show()