#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 10:24:19 2025

@author: benjamintung
"""
import numpy as np
import torch
import sys
import cupy

sys.path.append("../../../../") # added - to import applications and simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch # added
from simulator.algorithms.dnn.torch.convert import analog_modules

from build_tiny_imagenet import tiny_imagenet
from data_loader_tiny_imagenet import get_test_loader # added

import statistics

useGPU = True
N_images = 10000 # number of images
batch_size = 64
Nruns = 1
print_progress = True
n=5
depth = 6*n+2 #Depth 32
data_dir = None

#SONOS model used (uncomment to use)
# SONOS_MODEL = "SONOS_TID_202312" 
SONOS_MODEL = "SONOS_TID" 

#Loads the test set data
test_loader = get_test_loader( 
    data_dir=data_dir,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False
    )

# =============================================================================
# This function loads the CrossSim parameters
# Inputs:
#   TID: TID level of the device
#   useGPU: True if we use GPU
# Outputs:
#   params - params of the CrossSim device
# =============================================================================

def create_params(TID=0, SONOS_MODEL = SONOS_MODEL, useGPU=useGPU):

    params = CrossSimParameters()
    
    params.xbar.device.drift_error.model = SONOS_MODEL
    params.core.weight_bits = 8
    params.xbar.device.cell_bits = 7
    params.core.style = "BALANCED"
    params.xbar.device.Rmin = 3.75e4
    params.xbar.device.Rmax = 6e12
    params.xbar.device.error_model = "SONOS"
    params.xbar.device.programming_error.enable = True #change to false
    params.xbar.device.drift_error.enable = True #change to false
    params.core.balanced.subtract_current_in_xbar = True
    params.xbar.device.time = TID
    params.simulation.useGPU = useGPU

    return params

# =============================================================================
# This function loads a trained ResNet32 model with CrossSim parameters at a specified TID level
# Inputs:
#   model_name: Model name and path
#   TID: TID level of the device
#   n: number of Basic Blocks
#.  depth: depth level of the model
# Outputs:
#   cross_sim_model: the cross sim version of the model
# =============================================================================

def model_creation(model_name, TID=0, n=n, depth=depth):

    params = create_params(TID)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model = tiny_imagenet(n)
    model = model.to(device)
    model.load_state_dict(torch.load(model_name, weights_only = True))
    
    cross_sim_model = from_torch(model, params)
    cross_sim_model = cross_sim_model.eval()
    return cross_sim_model

# =============================================================================
# This function graphs the percent error, standard deviation, and mean error of 
# CrossSim over different radiation levels
# Inputs:
#   model_name: Model and path name
#   TID: TID level of the device
#   v: v_cal test voltage for alpha calculation
#   N_images: Number of images in test set
#   test_loader: the images required
#   n: number of Basic Blocks
#   depth: depth level of the model
#   use_alpha: If alpha correction factor is used
# Outputs:
#   tops_1: Accuracy of the model
# =============================================================================

def inference(model_name, TID, v=1, N_images=N_images, test_loader=test_loader, n=n, depth=depth, use_alpha=True):
    
    
    cross_sim_model_1 = model_creation(model_name, TID)             #Creates the model at the TID

    if use_alpha: 
        cross_sim_model_0 = model_creation(model_name)            #Creates a base model with no TID
        layers0 = analog_modules(cross_sim_model_0)     #Gets the layers from the model
        layers1 = analog_modules(cross_sim_model_1)
        
        for j in range(0, len(layers0)):

            #Get the analog cores
            if (j < 33):
                core0 = layers0[j].core.core
                core1 = layers1[j].core.core
            else:
                core0 = layers0[j].core
                core1 = layers1[j].core

            v_cal = np.full(core0.shape[1], v)

            #Get the matrix
            matrix0 = core0.get_matrix()
            matrix1 = core1.get_matrix()
            if type(matrix0) == cupy.ndarray:
                matrix0 = matrix0.get()
                matrix1 = matrix1.get()
                
            
            #Calculate alpha value
            alpha_denom_abs = np.sum(abs(matrix0 @ v_cal)) 
            alpha_numer_abs = np.sum(abs(matrix1 @ v_cal))

            alpha = alpha_numer_abs / alpha_denom_abs

            core1.alpha = alpha


    y_pred, y, k = np.zeros(N_images), np.zeros(N_images), 0
    
    for inputs, labels in test_loader:
        # Execute model
        output = cross_sim_model_1(inputs)
        # Collect outputs
        y_pred_k = output.data.detach().cpu().numpy()
        batch_size_k = y_pred_k.shape[0]
        y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
        y[k:(k+batch_size_k)] = labels.detach().numpy()
        k += batch_size_k
        print("Image {:d}/{:d}, accuracy so far = {:.2f}%".format(
            k, N_images, 100*np.sum(y[:k] == y_pred[:k])/k), end="\r")

    # Evaluate accuracy
    top1 = np.sum(y == y_pred)/len(y)
    print("\n=================")
    print("TID:", TID)
    print('Final accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N_images),N_images))
    
    return top1

#Model to evaluate
model_name = "TinyImageNet32NoiseAltAlpha_v2_Adam_30_0_10000.pth"

#Creates TID range to iterate through
TID_begin = np.arange(0,10e3,1e3)
TID_middle = np.arange(10e3, 100e3, 10e3)
TID_end = np.arange(100e3, 1e6, 100e3)
TID = np.concatenate((TID_begin, TID_middle, TID_end))
TID = TID_middle

accuracy_average = []
std = []
for j in range(0, len(TID)): #Sweeping across a TID range
    accuracy = []
    for k in range(0,1): #10 trials
        accuracy.append(inference(model_name, TID[j], 1, N_images, test_loader, n, depth, True))
    # std.append(statistics.stdev(accuracy))
    accuracy_average.append(sum(accuracy) / len(accuracy))

accuracy_array = np.array(accuracy_average)
# std_array = np.array(std)
np.save("Acc_NoAlt_Alpha.npy", accuracy_array)
# np.save("Std_NoAlt_Alpha.npy",std_array)

print(accuracy_average)
# print(std)