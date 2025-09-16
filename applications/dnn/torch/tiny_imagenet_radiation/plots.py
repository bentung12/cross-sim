#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:34:55 2025

@author: benjamintung
"""

import matplotlib.pyplot as plt
import numpy as np

#Loading the data
base_accuracy = np.load("data/alpha_data/base_accuracy.npy")
base_accuracy_std = np.load("data/alpha_data/base_accuracy_std.npy")

alpha = np.load("data/alpha_data/alpha.npy")
alpha_std = np.load("data/alpha_data/alpha_std.npy")

alt_2k = np.load("data/noise_data/alt_2k.npy")
alt_3k = np.load("data/noise_data/alt_3k.npy")
alt_4k = np.load("data/noise_data/alt_4k.npy")
alt_5k = np.load("data/noise_data/alt_5k.npy")
alt_10k = np.load("data/noise_data/alt_10k.npy")

dist_1k = np.load("data/noise_data/dist_1k.npy")
dist_4k = np.load("data/noise_data/dist_4k.npy")
alt_dist_4k = np.load("data/noise_data/alt_dist_4k.npy")

alpha_noise_4k = np.load("data/alpha_loop_data/alpha_noise_4k.npy")
alpha_loop_4k = np.load("data/alpha_loop_data/alpha_loop_4k.npy")
alpha_loop_10k = np.load("data/alpha_loop_data/alpha_loop_10k.npy")
alpha_loop_15k = np.load("data/alpha_loop_data/alpha_loop_15k.npy")
alpha_loop_20k = np.load("data/alpha_loop_data/alpha_loop_20k.npy")


alpha_loop_10k_noalt = [np.float64(0.54915), np.float64(0.54985), np.float64(0.5485), np.float64(0.5507500000000001), np.float64(0.549), np.float64(0.5480499999999999), np.float64(0.5487), np.float64(0.5488500000000001), np.float64(0.5492), np.float64(0.54755), np.float64(0.5469999999999999), np.float64(0.53775), np.float64(0.5259499999999999), np.float64(0.51505), np.float64(0.5092), np.float64(0.50865), np.float64(0.4965), np.float64(0.4858), np.float64(0.4704), np.float64(0.4435), np.float64(0.36795), np.float64(0.2727), np.float64(0.21639999999999998), np.float64(0.15355000000000002), np.float64(0.09145), np.float64(0.07394999999999999), np.float64(0.054900000000000004), np.float64(0.060899999999999996)]

# =============================================================================
# Uncomment/Comment Relevant Plots for graphs
# =============================================================================

# =============================================================================
# Base Accuracy (No CrossSim training or Alpha application)
# =============================================================================

plt.errorbar(base_accuracy[0], base_accuracy[1], base_accuracy_std[1], label = "Base Accuracy")

# =============================================================================
# Applying Alpha on top
# =============================================================================

# plt.errorbar(alpha[0], alpha[1], alpha_std[1], label = "Alpha Correction")

# =============================================================================
# Training at alternating TID levels
# =============================================================================

# plt.plot(alt_2k[0], alt_2k[1], label = "2k Alt")
# plt.plot(alt_3k[0], alt_3k[1], label = "3k Alt")
plt.plot(alt_4k[0], alt_4k[1], label = "4k Alt")
# plt.plot(alt_5k[0], alt_5k[1], label = "5k Alt")
# plt.plot(alt_10k[0], alt_10k[1], label = "10k Alt")

# =============================================================================
# Training at a distribution
# =============================================================================

# plt.plot(dist_1k[0], dist_1k[1], label = "1k Dist")
# plt.plot(dist_4k[0], dist_4k[1], label = "4k Dist")
# plt.plot(alt_dist_4k[0], alt_dist_4k[1], label = "4k Alt Dist")

# =============================================================================
# Applying Alpha on top of the Noise Model
# =============================================================================

plt.plot(alpha_noise_4k[0], alpha_noise_4k[1], label = "4k Alpha Noise")

# =============================================================================
# Training with Alpha in the Loop
# =============================================================================

# plt.plot(alpha_loop_4k[0], alpha_loop_4k[1], label = "4k Alpha Loop")
plt.plot(alpha_loop_10k[0], alpha_loop_10k[1], label = "10k Alpha Loop")
# plt.plot(alpha_loop_15k[0], alpha_loop_15k[1], label = "15k Alpha Loop")
# plt.plot(alpha_loop_20k[0], alpha_loop_20k[1], label = "20k Alpha Loop")

#Plot Parameters
plt.title("TID vs Accuracy")
plt.legend()
plt.xscale("log")
plt.xlabel("TID (rad)")
plt.ylabel("Accuracy")

# =============================================================================
# This function takes the graphs and prints relevant parameters
# Inputs: 
#     data: the graph that you want parameters for
#     degree: the degree of the polynomial you want to fit to the graph for roll-off parameters
# Outputs:
#     None
# =============================================================================


def roll_off(data, degree):
    x = data[0][0:20]
    y = data[1][0:20]
    TID0_acc = y[0]
    max_acc = np.max(y)
    max_loc = x[np.argmax(y)]
    
    coefficients = np.polyfit(x, y, 3)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, y_fit)
    
    target_y = max_acc - 0.05
    coeff_0 = [coefficients[0], coefficients[1], coefficients[2], coefficients[3] - target_y]
    target_y = max_acc - 0.01
    coeff_1 = [coefficients[0], coefficients[1], coefficients[2], coefficients[3] - target_y]
    target_y = 0.49335
    coeff_2 = [coefficients[0], coefficients[1], coefficients[2], coefficients[3] - target_y]
    
    print("Roll Off Parameters      ")
    print("Accuracy at 0 TID:       ", TID0_acc)
    print("Max Accuracy:            ", max_acc)
    print("Max Accuracy Location:   ", max_loc)
    print("5% Accuracy Loss:        ", np.roots(coeff_0).real)
    print("1% Accuracy Loss:        ", np.roots(coeff_1).real)
    print("Reaches 49.335% Accuracy:", np.roots(coeff_2).real)

#Calls the roll_off function
roll_off(alpha_loop_10k, 3)

plt.show()
plt.close()





