

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:20:17 2024

@author: benjamintung
"""

import sys
import numpy as np
sys.path.append("../../../../") # added - to import applications and simulator
from simulator import AnalogCore
from simulator import CrossSimParameters
import matplotlib.pyplot as plt
import statistics

IMAGE_CNT = 10000
IMAGE_SIZE = 256
COLUMN_CNT = 200    
# SONOS_MODEL = "SONOS_TID_202312" 
SONOS_MODEL = "SONOS_TID"           

#Import matrices 
W_dense = np.load("TIN_classifier_data/W_dense.npy")                            #Weight Matrix
bias_dense = np.load("TIN_classifier_data/bias_dense.npy")                      #Bias Matrix
x_classifier = np.load("TIN_classifier_data/x_classifier.npy")                  #Classifier Matrix (Image Data)
y_test = np.load("TIN_classifier_data/y_test.npy")                              #Correct Labels


#setting the parameters
params = CrossSimParameters()

params.xbar.device.drift_error.model = SONOS_MODEL
params.core.weight_bits = 8
params.xbar.device.cell_bits = 7
params.core.style = "BALANCED"
params.xbar.device.Rmin = 3.75e4
params.xbar.device.Rmax = 6e12
params.xbar.device.error_model = "SONOS"
params.xbar.device.programming_error.enable = True
params.xbar.device.drift_error.enable = True
params.core.balanced.subtract_current_in_xbar = True

# =============================================================================
# Device_creation creates a CrossSim device
# Inputs:
#   weight_matrix  - weight matrix to be programmed
#   params         - parameters of the device
#   TID            - radiation of the device in krad
# Outputs:
#   radiation_core - the CrossSim model of the device
# =============================================================================
def device_creation (weight_matrix, params=params, TID=0):
    params.xbar.device.time = TID                                               #Set TID level
    radiation_core = AnalogCore(W_dense, params = params)                       #Create the device
    return radiation_core

# =============================================================================
# Alpha_calc calculates either the numerator or denominator of the alpha function
# The denominator is calculated with TID=0, while the numerator is calculated with 
# the current TID level
# Inputs:
#   v             - test voltage V_cal to be applied to the device
#   l             - number of columns used in the device (set between 0-199)
#   weight_matrix - weight matrix to be programmed
#   params        - parameters of the device
#   TID           - radiation of the device in krad
# Outputs:
#   np.sum(alpha) - either the denominator of alpha if TID is set to 0, or 
#                   the numerator with current TID level
# =============================================================================

def alpha_calc (v, l, weight_matrix, params=params, TID=0):
    radiation_core = device_creation(weight_matrix, params, TID)
    v_cal = np.full(IMAGE_SIZE, v)                                              #Create a 1x256 array for V_cal                                      
    alpha = v_cal @ radiation_core[:,0:l]                                       #Multiply V_cal by CrossSim core
    return np.sum(alpha)

# =============================================================================
# Percent_error calculates the percent error between two arrays
# Inputs:
#   observed_array - the array observed during calculations
#   expected_array - the array with the expected answers
# Outputs:
#   percent error - the percent error between the two arrays
# =============================================================================

def percent_error(observed_array, expected_array):
    percent_error = abs(np.divide(observed_array - expected_array,expected_array)) * 100
    return sum(percent_error) / len(percent_error)

# =============================================================================
# This function compares the dot product accuracy between numpy and CrossSim at different TID levels using alpha
# Inputs:
#   v - test voltage V_cal to be used for alpha
#   l - number of columns used for alpha (between 0-199)
#   W_dense - weight_matrix used in the device
#   params - params of the CrossSim device
#   TID - list of up to four different TID values to compare numpy and CrossSim dot products
# =============================================================================

def dot_product_comparison(v, l, W_dense=W_dense, params=params, TID=0):
    
    denom_sum = alpha_calc(v, l, W_dense, params)                               #Calculate denominator of alpha
    
    # G_pos = radiation_core.cores[0][0].core_pos.matrix                              #Alternatively, use conductance to evaluate V * G
    # G_neg = radiation_core.cores[0][0].core_neg.matrix
    
    # scaling = radiation_core.cores[0][0].mvm_out_scale
    
    # G_sum = scaling * (np.sum(G_pos[:,0:L]) - np.sum(G_neg[:,0:L]))
    
    # denom_G = V * G_sum
    
    
    for i in range(0,len(TID)):                                                 #Iterates through TID levels
        
        radiation_core = device_creation(W_dense, params, TID[i])               #Create device at that TID level
        current_sum = alpha_calc(v, l, W_dense, params, TID[i])                 #Calculate numerator of alpha
        
        x = []
        y = []
        
        for j in range(0, IMAGE_CNT):                                               #Iterates through every image
            
            alpha = current_sum / denom_sum                                         #Calculate alpha
        
            output_rad = (1 / alpha) * ((x_classifier[j,:] @ radiation_core))       #Gets the output of the radiation/numpy multiplication with alpha
            output_numpy = x_classifier[j,:] @ W_dense                              #Calculates numpy value to compare
            
            output_rad_bias = output_rad + bias_dense                               #Add the bias vector
            output_numpy_bias = output_numpy + bias_dense
            
            x.append(output_numpy_bias)
            y.append(output_rad_bias)
        
            
        x = np.concatenate(x).ravel()                                               #Flatten the matrix
        y = np.concatenate(y).ravel()
        
        #plots the scatterplot of the outputs
        plt.subplot (2,2,i+1)   
        plt.scatter(x, y, 1, c="blue")
        plt.plot([-100,100],[-100,100])
        
        #Set graph limits/title
        plt.title(str(TID[i]/1000) + "krad")
        plt.axis("square")
        plt.xlim(-100,100)
        plt.ylim(-100,100)
        
        
        #Plot line of best fit
        a, b = np.round(np.polyfit(x,y,1),2)
        x_fit = np.linspace(-100,100)
        plt.plot(x_fit, a*x_fit + b, label = "y = " + str(a) + "x + " + str(b))
        plt.legend(loc = "lower right")
        
    #Add labels
    plt.tight_layout()
    plt.text(-225, -175, "Ideal Dot Product", ha='center', fontsize = "large")
    plt.text(-650, 125, 'SONOS Dot Product', va='center', rotation='vertical', fontsize = "large")
    plt.show()
    plt.close()



# =============================================================================
# This function graphs the percent error, standard deviation, and mean error of 
# CrossSim over different radiation levels
# Inputs:
#   v - test voltage V_cal to be used for alpha
#   l - number of columns used for alpha (between 0-199)
#   W_dense - weight_matrix used in the device
#   params - params of the CrossSim device
#   TID - list of TID values to compare
# =============================================================================

def percent_mean_error(v, l, W_dense=W_dense, params=params,TID=0):
    
    denom_sum = alpha_calc(v, l, W_dense, params)                               #Calculate denominator of alpha
    
    accuracy_TID = []
    mean_error = []
    stddev = []
    accuracy_TID_alpha = []
    
    for i in range(0,len(TID)):
        
        #Create a device with radiation
        radiation_core = device_creation(W_dense, params, TID[i])
        pe_avg = []
        pe_alpha_avg = []
        error_avg = []
        
        current_sum = alpha_calc(v, l, W_dense, params, TID[i])
        
        alpha = current_sum / denom_sum                                         #Calculate alpha value
    
        for j in range(0, IMAGE_CNT):
            
            output_numpy = x_classifier[j,:] @ W_dense
            output_rad = (x_classifier[j,:] @ radiation_core)                   #Gets the output of the radiation/numpy multiplication
            output_rad_alpha = (1 / alpha) * ((x_classifier[j,:] @ radiation_core))
            
            output_numpy_bias = output_numpy + bias_dense
            output_rad_bias = output_rad + bias_dense                           #Add the bias vector
            output_rad_alpha_bias = output_rad_alpha + bias_dense
            
            pe_avg.append(percent_error(output_rad_bias, output_numpy_bias))    #Calculate percent error with/without alpha
            pe_alpha_avg.append(percent_error(output_rad_alpha_bias, output_numpy_bias))
            
            error = np.divide(output_rad_alpha_bias - output_numpy_bias,output_numpy_bias) * 100    #Calculate mean error 
            error_avg.append(sum(error) / len(error))
            
        avg_accuracy = statistics.fmean(pe_avg)                                 #Calculate average percent error for the TID level and store it
        avg_accuracy_alpha = statistics.fmean(pe_alpha_avg)
        accuracy_TID.append(avg_accuracy)
        accuracy_TID_alpha.append(avg_accuracy_alpha)
        
        stddev.append(statistics.stdev(error_avg))                              #Calculate standard deviation of mean error
        mean_error.append(statistics.fmean(error_avg))
        
        

    
    #Plots a zoomed in percent error
    plt.subplot(2,1,1)
    plt.plot(TID[0:10],accuracy_TID_alpha[0:10], label = "With Alpha", c = "orange")
    plt.xlabel("Average TID of batch (rad)")
    plt.ylabel("Percent Error")
    plt.legend()
    plt.tight_layout

    #Plots the percent error graphs
    plt.subplot(2,1,2)
    plt.plot(TID,accuracy_TID, label = "No Alpha")
    plt.plot(TID,accuracy_TID_alpha, label = "With Alpha", c = "orange")
    plt.xlabel("Average TID of batch (rad)")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation of Percent Error")
    plt.show()
    plt.close()
    
    #Plots a mean error with standard deviation
    plt.subplot(2,1,1)
    plt.errorbar(TID, mean_error, stddev, label = "Mean Error")
    plt.title("Mean Error and Standard Deviation")
    plt.ylabel("Mean Error/Stdev")
    plt.xlabel("Average TID of batch (rad)")
    
    #Plots just the mean error
    plt.subplot(2,1,2)
    plt.plot(TID, mean_error)
    plt.ylabel("Mean Error")
    plt.xlabel("Average TID of batch (rad)")
    plt.tight_layout
    plt.show()
    plt.close()


# =============================================================================
# This function sweeps how many columns you need for optimal accuracy at different TID levels
# Inputs:
#   v - test voltage V_cal to be used for alpha
#   W_dense - weight_matrix used in the device
#   params - params of the CrossSim device
#   TID - list of TID values to compare
# =============================================================================

def column_count(v, W_dense=W_dense, params=params, TID=0):
    
    for i in range(0, len(TID)):                                                #Sweep through TID values
        radiation_core = device_creation(W_dense, params, TID[i])
        
        
        accuracy_TID_alpha = []
        
        for l in range(1, COLUMN_CNT):                                          #Sweep columns from 1 to 200
            pe_alpha_avg = []
            
            denom_sum = alpha_calc(v, l, W_dense, params)                       #Calculate alpha based on column value
            current_sum = alpha_calc(v, l, W_dense, params, TID[i])
            alpha = current_sum / denom_sum
        
            for j in range(0, IMAGE_CNT):
            
                output_numpy = x_classifier[j,:] @ W_dense
                output_rad_alpha = (1 / alpha) * ((x_classifier[j,:] @ radiation_core))
                
                output_numpy_bias = output_numpy + bias_dense
                output_rad_alpha_bias = output_rad_alpha + bias_dense
                
                pe_alpha_avg.append(percent_error(output_rad_alpha_bias, output_numpy_bias))
                
            avg_accuracy_alpha = statistics.fmean(pe_alpha_avg)
            accuracy_TID_alpha.append(avg_accuracy_alpha)
            
        #plots the accuracy arrays
        # plt.plot(range(1,200),accuracy_TID, label = "No Alpha")
        plt.plot(range(1,200),accuracy_TID_alpha, label = str(TID[i]/1000) + "krad")
        
    plt.xlabel("Number of Columns")
    plt.ylabel("Percent Error")
    plt.ylim([0,100])
    plt.title("Percent Error")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


# =============================================================================
# This function compares the global and local alpha accuracy
# Inputs:
#   v - test voltage V_cal to be used for alpha
#   l - number of columns used for alpha (between 0-199)
#   W_dense - weight_matrix used in the device
#   params - params of the CrossSim device
#   TID - list of TID values to compare
# =============================================================================

def local_alpha(v, l, W_dense, params, TID=0):
    
    v_cal = np.full(IMAGE_SIZE, v) 
    
    normal_core = device_creation(W_dense, params)                              #Creates device with 0 TID
    denom_sum = alpha_calc(v, l, W_dense, params)                               #Calculate denominator of alpha

    accuracy_TID_local = []
    accuracy_TID_global = []
    
   
    inverse_alpha_matrix = np.zeros(200)
    
    for i in range(0,len(TID)):
        
        pe_avg_local = []
        pe_avg_global = []
        
        radiation_core = device_creation(W_dense, params, TID[i])               #Create device at that TID level
        current_sum = alpha_calc(v, l, W_dense, params, TID[i])                 #Calculate numerator of alpha
        
        alpha_global = current_sum / denom_sum
        
        for k in range(0,200):                                                  #Calculates a local alpha factor
                                                                                #Stores it in a matrix
            current_local = v_cal @ radiation_core[:,k]
            current_sum_local = np.sum(current_local)
        
            denom_local = v_cal @ normal_core[:,k]
            denom_sum_local = np.sum(denom_local)
            
            inverse_alpha_matrix[k] = denom_sum_local / current_sum_local
    
        for j in range(0, IMAGE_CNT):
        
            output_numpy = x_classifier[j,:] @ W_dense                          #Calculates output with local/global alpha
            output_local = np.multiply((x_classifier[j,:] @ radiation_core),inverse_alpha_matrix)
            output_global = (1 / alpha_global) * ((x_classifier[j,:] @ radiation_core))
            
            output_numpy_bias = output_numpy + bias_dense                       #Add the bias vector
            output_local_bias = output_local + bias_dense           
            output_global_bias = output_global + bias_dense
            
            pe_avg_local.append(percent_error(output_local_bias, output_numpy_bias))    #Calculate percent error with/without alpha
            pe_avg_global.append(percent_error(output_global_bias, output_numpy_bias))
            
        avg_accuracy_local = statistics.fmean(pe_avg_local)                             #Calculate average percent error for the TID level and store it
        avg_accuracy_global = statistics.fmean(pe_avg_global)
        # print("local alpha", avg_accuracy_local)
        # print("global alpha", avg_accuracy_global)
        accuracy_TID_local.append(avg_accuracy_local)
        accuracy_TID_global.append(avg_accuracy_global)
        
    #plots the accuracy lists
    plt.plot(TID,accuracy_TID_local, label = "Local Alpha")
    plt.plot(TID,accuracy_TID_global, label = "Global Alpha")
    plt.xlabel("Average TID of batch (rad)")
    plt.ylabel("Percent Error")
    plt.title("Percent Error")
    plt.legend()
    plt.show()
    plt.close()

# =============================================================================
# This function compares image recognition accuracy over TID levels
# Inputs:
#   v - test voltage V_cal to be used for alpha
#   l - number of columns used for alpha (between 0-199)
#   W_dense - weight_matrix used in the device
#   params - params of the CrossSim device
#   TID - list of TID values to compare
# =============================================================================

def accuracy_comparison(v, l, W_dense=W_dense, params=params, TID=0):
    accuracy_rad = []
    accuracy_ideal = []
    accuracy_alpha = []
    
    denom_sum = alpha_calc(v, l, W_dense, params)

    for i in range(0,len(TID)):
        
        # x = []
        # y = []
        
        #Create a device at 0 radiation and a device with radiation
        radiation_core = device_creation(W_dense, params, TID[i])
        ideal_radiation_core = device_creation(W_dense, params, TID[0])
        

        num_correct_rad = 0
        num_correct_alpha = 0
        num_correct_ideal = 0
        
        current_sum = alpha_calc(v, l, W_dense, params, TID[i])
        
        alpha = current_sum / denom_sum

        for j in range(0, IMAGE_CNT):
            output_rad = x_classifier[j,:] @ radiation_core                 #Multiplies image array by crosssim array
            output_ideal = x_classifier[j,:] @ ideal_radiation_core
            output_alpha = (1 / alpha) * (x_classifier[j,:] @ radiation_core)
            
            output_rad_bias = output_rad + bias_dense                       #Add bias vector
            output_ideal_bias = output_ideal + bias_dense
            output_alpha_bias = output_alpha + bias_dense
            
            classification_rad = np.argmax(output_rad_bias)                 #Gets maximum of output vector
            classification_numpy = np.argmax(output_ideal_bias)
            classification_alpha = np.argmax(output_alpha_bias)
              
            if y_test[j] == classification_rad:                             #If correctly classified add 1 to num correct
                num_correct_rad += 1
            if y_test[j] == classification_numpy:                           
                num_correct_ideal += 1
            if y_test[j] == classification_alpha:
                num_correct_alpha += 1
            
            # x.append(output_ideal_bias)
            # y.append(output_alpha_bias)
        
            
        # x = np.concatenate(x).ravel()                                               #Flatten the matrix
        # y = np.concatenate(y).ravel()
        
        
        # #plots the scatterplot of the outputs
        # plt.subplot (2,2,i+1)   
        # plt.scatter(x, y, 1, c="blue")
        # plt.plot([-100,100],[-100,100])
        
        # #Set graph limits/title
        # plt.title(str(TID[i]/1000) + "krad")
        # plt.axis("square")
        # plt.xlim(-100,100)
        # plt.ylim(-100,100)
                
        accuracy_rad.append(num_correct_rad / IMAGE_CNT)                    #Stores the accuracy in an array
        accuracy_ideal.append(num_correct_ideal / IMAGE_CNT)
        accuracy_alpha.append(num_correct_alpha / IMAGE_CNT)
    
    # plt.show()
    # plt.close()
    #plots the accuracy arrays
    plt.plot(TID,accuracy_rad, label="Simulation")
    # plt.plot(TID,accuracy_ideal, label="Ideal")
    plt.plot(TID, accuracy_alpha, label="Alpha")
    plt.xlabel("Average TID of batch (rad)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.show()
    plt.close()
    
    # print("Radiation Accuracy", accuracy_rad)
    # print("Alpha Accuracy", accuracy_alpha)
    

v = 1
l = 199

TID = [0, 103e3, 522e3, 3.2e6]
# dot_product_comparison(v, l, W_dense, params, TID)

TID = np.arange(0,1e6,50e3)
# percent_mean_error(v, l, W_dense, params, TID)
# local_alpha(v, l, W_dense, params, TID)
# accuracy_comparison(v, l, W_dense, params, TID)

TID = [0, 110e3, 550e3, 1e6, 2e6, 3.2e6]
TID = [0, 4e3, 10e3, 20e3, 50e3, 110e3]
column_count(v, W_dense, params, TID)
