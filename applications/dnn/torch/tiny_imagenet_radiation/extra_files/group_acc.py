#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:46:44 2025

@author: benjamintung
"""

import matplotlib.pyplot as plt
import numpy as np

TID_begin = np.arange(0,20e3,1e3)
TID_end = np.arange(20e3, 100e3, 10e3)
TID = np.concatenate((TID_begin, TID_end))

TID_begin = np.arange(0,10e3,1e3)
TID_middle = np.arange(10e3, 100e3, 10e3)
TID_end = np.arange(100e3, 1e6, 100e3)
TID_2 = np.concatenate((TID_begin, TID_middle, TID_end))
TID_3 = np.arange(0, 20e3, 1e3)

accuracy_groups = np.load("data/Acc_Alpha_Mod_Grp.npy")
std_groups = np.load("data/Std_Alpha_Mod_Grp.npy")

alpha_abs =  np.load("data/Accuracy_Alpha_Abs.npy")
alpha_std_abs = np.load("data/Std_Alpha_Abs.npy")

accuracy_alpha = np.load("data/Accuracy_Alpha.npy")
std_alpha = np.load("data/Std_Alpha.npy") #Only absolute value on first layer

accuracy_groups_2 = np.load("data/Accuracy_Alpha_Grp_2.npy")
std_groups = np.load("data/Std_Alpha_Grp_2.npy")

acc_alpha_loop_begin = np.load("data/Acc_Alpha_Noise_Loop_B.npy")
acc_alpha_loop_end = np.load("data/Acc_Alpha_Noise_Loop.npy")

alpha_loop = np.concatenate((acc_alpha_loop_begin.flatten(), acc_alpha_loop_end.flatten()))
print(alpha_loop)

no_alpha = [0.54335, 0.54251, 0.52953, 0.5121,  0.4993,  0.47186, 0.44414, 0.4142,  0.37953,
  0.34284, 0.31102, 0.26503, 0.22962 ,0.19341, 0.16005, 0.12864 ,0.10596, 0.08067,
  0.06124, 0.04609, 0.03481, 0.00501 ,0.00551, 0.00538, 0.00521 ,0.00504, 0.00514,
  0.00501]
no_alpha_std = [0.00402251, 0.00208084, 0.00418012, 0.00948999, 0.00698697, 0.01338566,
  0.00654832, 0.01006302, 0.00844144, 0.01218279, 0.01410916, 0.01269699,
  0.01155468, 0.01046926, 0.01113006, 0.00875483, 0.00840954, 0.00730632,
  0.00398976, 0.00498563, 0.00175464, 0.00124316, 0.0009279 , 0.000668,
  0.00055867, 0.00045753, 0.00016465, 0.00015239]


alpha_noise = [0.53676667, 0.53473333, 0.53676667, 0.53743333, 0.5352,     0.5363,
  0.537 ,     0.53556667, 0.53363333, 0.53566667, 0.5286    , 0.51876667,
  0.5002,     0.47486667, 0.43406667, 0.413     , 0.3969    , 0.36186667,
  0.3419,     0.28936667, 0.23003333, 0.13093333, 0.10346667, 0.05116667,
  0.03316667, 0.0263    , 0.01853333, 0.02706667]

four_k = [0.5464,  0.5368,  0.5429,  0.54325, 0.54175, 0.53695, 0.53395 ,0.52635, 0.51125,
 0.50625, 0.4848 , 0.4808 , 0.467 ,  0.4499,  0.43225, 0.42025, 0.396  , 0.39095,
 0.3665 , 0.33235]

ten_k_loop = [0.55204, 0.55177, 0.55007, 0.55237, 0.55277, 0.55058, 0.55189, 0.5499,  0.54703,
  0.54788, 0.54688, 0.54045, 0.52975, 0.52278, 0.51263, 0.50379, 0.49305, 0.48108,
  0.47665, 0.45445, 0.36839, 0.29509, 0.22287, 0.17057, 0.14166, 0.10679, 0.07836,
  0.05627]

twenty_k_loop = [0.4861,  0.48565 ,0.4877,  0.48315 ,0.4866,  0.4846,  0.48065, 0.48755, 0.48625,
  0.4872,  0.4827,  0.47885, 0.4707 , 0.4679,  0.4586 , 0.4375,  0.44465, 0.41775,
  0.41875, 0.41315, 0.3336 , 0.26545, 0.18615, 0.12615, 0.10335, 0.0776,  0.06695,
  0.0364]

fifteen_k_loop = [0.5397,     0.54113333, 0.54466667, 0.53783333, 0.5418,     0.54103333,
  0.54053333, 0.54056667, 0.5417    , 0.5366,     0.54153333, 0.5346,
  0.52183333, 0.51533333, 0.5061    , 0.503 ,     0.48563333, 0.4774,
  0.4654,     0.45533333, 0.37406667, 0.25213333, 0.18733333, 0.08423333,
  0.07256667, 0.05186667, 0.03723333, 0.02536667]

base_acc_array = np.array([TID, no_alpha])
base_acc_std_array = np.array([TID, no_alpha_std])

base_accuracy = np.load("data/alpha_data/base_accuracy.npy")
base_accuracy_std = np.load("data/alpha_data/base_accuracy_std.npy")

alpha = np.load("data/alpha_data/alpha.npy")
alpha_std = np.load("data/alpha_data/alpha_std.npy")

np.save("alpha_noise_4k.npy", np.array([TID_2, alpha_noise]))
np.save("alpha_loop_4k.npy", np.array([TID_2, alpha_loop]))
np.save("alpha_loop_10k.npy", np.array([TID_2, ten_k_loop]))
np.save("alpha_loop_15k.npy", np.array([TID_2, fifteen_k_loop]))
np.save("alpha_loop_20k.npy", np.array([TID_2, twenty_k_loop]))

plt.errorbar(base_accuracy[0], base_accuracy[1], base_accuracy_std[1], label = "Base Accuracy")
plt.errorbar(alpha[0], alpha[1], alpha_std[1], label = "Alpha Correction")
plt.plot(TID_3, four_k, label="4k TID")
plt.plot (TID_2, alpha_noise, label="4k Alpha No Loop")
# plt.plot(TID_2, alpha_loop, label="4K Alpha Loop")
plt.plot(TID_2, ten_k_loop, label="10k Alpha Loop")
# plt.plot(TID_2, fifteen_k_loop, label = "15k Alpha Loop")
# plt.plot(TID_2, twenty_k_loop, label="20k Alpha Loop")


# for i in range(0,7):
#     plt.errorbar(TID, accuracy_groups[i], label = str(2**(i))+" Groups")
    
# for i in range(0,7):
#     plt.errorbar(TID_2, accuracy_groups_2[i], label = str(i+1)+" Groups")
    
# plt.errorbar(TID_2, accuracy_alpha[0], std_alpha[0], label="Alpha (No Abs)")

def roll_off(x, y, target_y, degree):
    coefficients = np.polyfit(x, y, 3)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, y_fit)
    coeff_1 = [coefficients[0], coefficients[1], coefficients[2], coefficients[3] - target_y]
    return np.roots(coeff_1)

# print(roll_off(TID_2[0:25], ten_k_loop[0:25], 0.49335, 3))

plt.title("TID vs Accuracy")
plt.legend()
plt.xscale("log")
plt.xlabel("TID (rad)")
plt.ylabel("Accuracy")
plt.show()