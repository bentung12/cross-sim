# module load cuda-12.5.0-gcc-12.1.0
import numpy as np
import torch
import pickle
import tensorflow as tf
import sys
import cupy

sys.path.append("../../../../") # added - to import applications and simulator
from simulator import AnalogCore
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, synchronize, reinitialize # added
from simulator.algorithms.dnn.torch.convert import analog_modules

from build_tiny_imagenet import tiny_imagenet
from data_loader_tiny_imagenet import get_train_valid_loader, get_test_loader # added

import matplotlib.pyplot as plt
import statistics
import torch.nn as nn

#Sweep TID across 0 to 10Krad

useGPU = True # use GPU?
N_images = 10000 # number of images
batch_size = 64
Nruns = 1
print_progress = True
n=5
data_dir= None

# SONOS_MODEL = "SONOS_TID_202312"
SONOS_MODEL = "SONOS_TID"

depth = 6*n+2

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

def model_creation(TID=0, n=n, depth=depth):

    params = create_params(TID)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_name = "TinyImageNet"+str(depth)
        
    model = tiny_imagenet(n)
    model = model.to(device)
    model.load_state_dict(torch.load("TinyImageNet32NoiseAltAlpha_v2_Adam_30_0_4000.pth", weights_only = True))
    
    cross_sim_model = from_torch(model, params)
    cross_sim_model = cross_sim_model.eval()
    return cross_sim_model

test_loader = get_test_loader( 
    data_dir=data_dir,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False
    )

# TID_begin = np.arange(0,10e3,1e3)
# TID_middle = np.arange(10e3, 100e3, 10e3)
# TID_end = np.arange(100e3, 1e6, 100e3)
# TID = np.concatenate((TID_begin, TID_middle, TID_end))

TID = np.arange(0, 20e3, 1e3)


accuracy_avg = []
for i in range(0, len(TID)):
    accuracy=[]
    for j in range(0, 10): #trials
        cross_sim_model = model_creation(TID[i])
    
        y_pred, y, k = np.zeros(N_images), np.zeros(N_images), 0
        
        for inputs, labels in test_loader:
            # Execute model
            output = cross_sim_model(inputs)
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
        accuracy.append(top1)
        print("\n=================")
        print("TID:", TID[i])
        print('Final accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N_images),N_images))
    accuracy_avg.append(sum(accuracy) / len(accuracy))

print("Final Accuracy", accuracy_avg)
np.save("Acc_10k_Avg.npy", accuracy_avg)