#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:29:24 2024

@author: benjamintung
"""

import sys
import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)
sys.path.append("../../../../") # added - to import applications and simulator
from simulator import AnalogCore
from simulator import CrossSimParameters
import matplotlib.pyplot as plt

IMAGE_CNT = 10000

#Import matrices 
W_dense = np.load("TIN_classifier_data/W_dense.npy")
bias_dense = np.load("TIN_classifier_data/bias_dense.npy")
x_classifier = np.load("TIN_classifier_data/x_classifier.npy")
y_test = np.load("TIN_classifier_data/y_test.npy")

# dot_products_SONOS = np.load("dot_products_all/dot_products_all_SONOS.npy")
# dot_products_ideal = np.load("dot_products_all/dot_products_all_ideal.npy")
# TID_images = np.load("dot_products_all/TID_images_all.npy")


#setting the parameters
params = CrossSimParameters()

params.xbar.device.drift_error.model = "SONOS_TID"
# params.xbar.device.drift_error.model = "SONOS_TID_202312"

params.core.weight_bits = 8
params.xbar.device.cell_bits = 7
params.core.style = "BALANCED"
params.xbar.device.Rmin = 3.75e4
params.xbar.device.Rmax = 6e12
params.xbar.device.error_model = "SONOS"
params.xbar.device.programming_error.enable = True
params.xbar.device.drift_error.enable = True
params.core.balanced.subtract_current_in_xbar = True

# #=============================================================================
#This portion graphs the conductance array given different TID values:

def conductance(TID):
    for i in range(0,6):
        
        params.xbar.device.time = TID[i]
        radiation_core = AnalogCore(W_dense, params = params)   #Creates the analog_core with radiation effects
    
        G_pos = radiation_core.cores[0][0].core_pos.matrix / params.xbar.device.Rmin    #Get the Conductances
        G_neg = radiation_core.cores[0][0].core_neg.matrix / params.xbar.device.Rmin
        
        G_pos_flatten = G_pos.flatten()
        G_neg_flatten = G_neg.flatten()
        
        G = 1e6 * np.concatenate([G_pos_flatten, G_neg_flatten])
        
        #Plot the conductances for a specific TID
        ax = plt.subplot(3,2,i+1)
        
        plt.hist(G, bins = 40)
        plt.title(str(TID[i]/1000) + "krad")
        ax.set_yscale("log")
        ax.set_xlim([0,60])
        ax.set_ylim([1,10e5])
        
    plt.tight_layout()  
    plt.text(-5, 0.0001, "Conductance (uS)", ha='center', fontsize = "large")
    plt.text(-90, 100000000000000, 'Number of SONOS cells', va='center', rotation='vertical', fontsize = "large")
    plt.show()
    plt.close()
# #=============================================================================


#This portion compares the accuracy between numpy and CrossSim at different TID levels

def dot_products(TID):
    for i in range(0,4):
        x = []
        y = []
        
        params.xbar.device.time = TID[i]
        radiation_core = AnalogCore(W_dense, params = params)   #Create the device
    
    
        for j in range(0, IMAGE_CNT):                           #Iterates through every image
            output_rad = x_classifier[j,:] @ radiation_core     #Gets the output of the radiation/numpy multiplication
            output_numpy = x_classifier[j,:] @ W_dense
            
            output_rad_bias = output_rad + bias_dense           #Add the bias vector
            output_numpy_bias = output_numpy + bias_dense
               
            x.append(output_numpy_bias)
            y.append(output_rad_bias)
            
            
        x = np.concatenate(x).ravel()
        y = np.concatenate(y).ravel()
        
        a, b = np.round(np.polyfit(x,y,1),2)
            
        plt.subplot (2,2,i+1)   
        plt.scatter(x, y, 1, c="blue")   
    
        plt.plot([-100,100],[-100,100])
        plt.title(str(TID[i]/1000) + "krad")
        plt.axis("square")
        plt.xlim(-100,100)
        plt.ylim(-100,100)
        
        x_fit = np.arange(-100,100, 0.1)
        plt.plot(x_fit, a*x_fit + b, label = "y = " + str(a) + "x + " + str(b))
        plt.legend(loc = "lower right")
    
    plt.tight_layout()
    plt.text(-225, -175, "Ideal Dot Product", ha='center', fontsize = "large")
    plt.text(-650, 125, 'SONOS Dot Product', va='center', rotation='vertical', fontsize = "large")
    plt.show()
    plt.close()

# This portion compares the accuracy between numpy and CrossSim at different TID levels from the paper
# TID_index = [0, 307, 1507, 9108]
# for i in range (0,4):
#     x = dot_products_ideal[TID_index[i],:]
#     y = dot_products_SONOS[TID_index[i],:]
#     TID_value = TID_images[TID_index[i]]
#     a, b = np.round(np.polyfit(x,y,1),2)
#     plt.subplot (2,2,i+1)
#     plt.scatter(x, y)
#     plt.plot([-100,100],[-100,100], label = "y = x")
#     plt.title(str(np.round(TID_value/1000,0))+"krad")
#     plt.xlabel("Software Dot Product")
#     plt.ylabel("SONOS Dot Product")
#     plt.axis("square")
#     plt.xlim(-100,100)
#     plt.ylim(-100,100)
    
#     x_fit = np.arange(-100,100, 0.1)
#     plt.plot(x_fit, a*x_fit + b, label = "y = " + str(a) + "x + " + str(b))
#     plt.legend()

# plt.tight_layout()
# plt.show()


#=============================================================================
#This portion graphs the accuracy of the CrossSim with radiation effects over different TID levels

def accuracy_comparison(TID):
#Create the TID array and accuracy arrays
    accuracy_rad = []
    accuracy_ideal = []
    
    for i in range(0,len(TID)):
        
        #Create a device at 0 radiation and a device with radiation
        params.xbar.device.time = TID[i]
        radiation_core = AnalogCore(W_dense, params = params)
        params.xbar.device.time = TID[0]
        ideal_radiation_core = AnalogCore(W_dense, params = params)
    
        num_correct_rad = 0
        num_correct_ideal = 0
    
        for j in range(0, IMAGE_CNT):
            output_rad = x_classifier[j,:] @ radiation_core                 #Multiplies image array by crosssim array
            output_ideal = x_classifier[j,:] @ ideal_radiation_core
            
            output_rad_bias = output_rad + bias_dense                       #Add bias vector
            output_ideal_bias = output_ideal + bias_dense
            
            classification_rad = np.argmax(output_rad_bias)                 #Gets maximum of output vector
            classification_numpy = np.argmax(output_ideal_bias)
              
            if y_test[j] == classification_rad:                             #If correctly classified add 1 to num correct
                num_correct_rad += 1
            if y_test[j] == classification_numpy:                           
                num_correct_ideal += 1
                
        accuracy_rad.append(num_correct_rad / IMAGE_CNT)                    #Stores the accuracy in an array
        accuracy_ideal.append(num_correct_ideal / IMAGE_CNT)
        
    #plots the accuracy arrays
    plt.plot(TID,accuracy_rad)
    plt.plot(TID,accuracy_ideal)
    plt.xlabel("Average TID of batch (rad)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend(["Simulation","Ideal"])
    plt.show()
    plt.close()
#=============================================================================

TID = [0, 110e3, 550e3, 1e6, 2e6, 3.2e6]
conductance(TID)

TID = [0, 103e3, 522e3, 3.2e6]  #TID array
dot_products(TID)

TID = np.arange(0,3.25e6,50e3)
accuracy_comparison(TID)




        




