#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:46:59 2024

@author: benjamintung
"""

from PIL import Image
import os
import numpy as np
import statistics

def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, "r")
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "L":
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values, width, height


test_dir = "/Users/benjamintung/Desktop/ASU/Research/Marinella/CrossSim/cross-sim-pytorch/applications/dnn/data/datasets/tiny-imagenet-200/test/images"
train_dir = "/Users/benjamintung/Desktop/ASU/Research/Marinella/CrossSim/cross-sim-pytorch/applications/dnn/data/datasets/tiny-imagenet-200/train"

def welford_std(data):
    n = 0
    mean = 0.0
    M2 = 0.0

    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)

    if n < 2:
        return 0.0

    return (M2 / (n - 1)) ** 0.5           

def mean_std(folder_dir):

    R = []
    G = []
    B = []
    num_images = 0
    
    for subdir, dirs, files in os.walk(folder_dir):
        
        for file in files:
            path = os.path.join(subdir, file)
            if (path.endswith("JPEG")):
                pix, width, height = get_image(path)
                
                for i in range(0,width):
                    for j in range(0, height):
                        R.append(pix[i,j][0])
                        if np.shape(pix)[2] != 1:
                            G.append(pix[i,j][1])
                            B.append(pix[i,j][2])
                            
                        else:
                            G.append(pix[i,j][0])
                            B.append(pix[i,j][0])
                
    R_mean = sum(R) / len(R)
    G_mean = sum(G) / len(G)
    B_mean = sum(B) / len(B)
    
    R_std = welford_std(R)
    G_std = welford_std(G)
    B_std = welford_std(B)
    
    print("R_mean", R_mean / 256)
    print("G_mean", G_mean / 256)
    print("B_mean", B_mean / 256)
    
    print("R_std", R_std / 256)
    print("G_std", G_std / 256)
    print("B_std", B_std / 256)

mean_std(test_dir)
mean_std(train_dir)
        
# test:
# R_mean 0.46993770093917847
# G_mean 0.43982608642578125
# B_mean 0.39056721649169923
# R_std 0.27460691872852344
# G_std 0.2669988428565931
# B_std 0.27867653995811315

#train_dir
        
        
        


                    
                    

            
              
                        
                
                
                    
                    
                
        
