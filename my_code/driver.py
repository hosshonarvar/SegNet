#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:01:48 2019

@author: hosseinhonarvar
"""

import os
from Src.model import SegNet
import Src.visualize as net_visual

class Driver:

    ### Set the configuration file path
    
    # Set the experiment type path (e.g., 'Batch_size/1') if running 
    # experiments
    # Set the experiment type to None if not running experiments
    
    def __init__(self, experiment_type = 'Batch_size/1'):
    
        if experiment_type:
            self.config_path = os.path.join('./', 'Experiments', 
                                            experiment_type, 
                                            'configuration.json')
            print('Running the experiment in', experiment_type)
        else:
            self.config_path = './configuration.json'
            print('Running the trained model in current directory')
    
        ### Run and visualize
                
        # Instantiate the SegNet class
        S = SegNet(self.config_path)
        
        # Train the model
        S.train()
        
if __name__ == '__main__': 
    
    Driver()
    # Plot the metrics if running the trained model
    net_visual.plot_metrics('./Model')
    