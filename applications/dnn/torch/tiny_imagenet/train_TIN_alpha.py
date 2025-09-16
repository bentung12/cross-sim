#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:55:01 2025

@author: benjamintung
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)
import time
import copy
import sys
import cupy
sys.path.append("../../../../") # added - to import applications and simulator
from simulator.algorithms.dnn.torch.convert import synchronize, from_torch, analog_modules
from simulator import CrossSimParameters
from data_loader_tiny_imagenet import get_train_valid_loader, get_test_loader # added
from build_tiny_imagenet import tiny_imagenet

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("CUDA",torch.cuda.is_available())

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Maximum number of epochs to train for
num_epochs = 30

# L2 regularization of the kernel weights
l2_penalty = 1e-4

data_dir = None

# SONOS_MODEL = "SONOS_TID_202312"
SONOS_MODEL = "SONOS_TID"
useGPU = True

n = 5 #Training Resnet 32
depth = 6*n + 2

# =============================================================================
# This function loads the CrossSim parameters
# Inputs:
#   TID: TID level of the device
#   useGPU: True if we use GPU
# Outputs:
#   params - params of the CrossSim device
# =============================================================================

def create_params(TID=0, SONOS_MODEL = SONOS_MODEL, useGPU=useGPU, enable_drift = True):

    params = CrossSimParameters()
    
    params.xbar.device.drift_error.model = SONOS_MODEL
    params.core.weight_bits = 8
    params.xbar.device.cell_bits = 7
    params.core.style = "BALANCED"
    params.xbar.device.Rmin = 3.75e4
    params.xbar.device.Rmax = 6e12
    params.xbar.device.error_model = "SONOS"
    params.xbar.device.programming_error.enable = True #change to false
    params.xbar.device.drift_error.enable = enable_drift
    params.core.balanced.subtract_current_in_xbar = True
    params.xbar.device.time = TID
    params.simulation.useGPU = useGPU

    return params

# =============================================================================
# This function creates ResNet32 model with CrossSim parameters at a specified TID level
# Inputs:
#   TID: TID level of the device
#   n: number of Basic Blocks
#.  depth: depth level of the model
# Outputs:
#   cross_sim_model: the cross sim version of the model
# =============================================================================

def model_creation(TID=0, n=n, depth=depth):

    params = create_params(TID)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_name = "TinyImageNet"+str(depth)
        
    model = tiny_imagenet(n)
    model = model.to(device)
    
    cross_sim_model = from_torch(model, params)
    cross_sim_model = cross_sim_model.eval()
    return cross_sim_model

# =============================================================================
# This function trains a CrossSim model at a certain TID level with alpha in the loop
# Important Inputs:
#   TID: TID level of the device its trained at (and validated at)
#   alt: Alternates training between 0 TID and a specified TID every epoch
#   dist: Trains at a Gaussian distribution of TIDs throughout the epochs
#   mean: The mean of the Gaussian distribution
#   std_dev: The standard deviation of the Gaussian distribution
#   use_alpha: Set to True if training with alpha in the loop
# Outputs:
#   model: The model with the best accuracy
# =============================================================================
    
def train_model(model, dataloaders, criterion, optimizer, reducer, scheduler, 
    optimizer_name, model_name, TID, alt, dist, mean, std_dev, num_epochs=30, l2_penalty=0, use_alpha=True):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if dist==True:
        #Creates a distribution of TIDs to train at
        TID_sample = abs(np.random.normal(mean, std_dev, num_epochs)) 
        print("TID", TID_sample)

    for epoch in range(0,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # changed
            if phase == 'train':
                model.train()  # Set model to training mode (training sets)
            else:
                model.eval()   # Set model to evaluate mode (validation and testing sets)

            if epoch % 2 == 0 and alt==True:
                training_TID = 0
            elif epoch % 2 == 1 and dist==True:
                training_TID = TID_sample[epoch]
            else:
                training_TID = TID

            print("Training at TID:", training_TID)
            
            alpha_list = []
            if use_alpha: 
                v = 1
                j=0
                
                for layer in analog_modules(model):
                    layer.params = create_params(TID=0)
                    if (j < 33):
                        core0 = layer.core.core
                    else:
                        core0 = layer.core
                    
                    v_cal = np.full(core0.shape[1], v)

                    #Get the matrix from the layers at 0 TID
                    matrix0 = core0.get_matrix()
                    if type(matrix0) == cupy.ndarray:
                        matrix0 = matrix0.get()
                        
                    layer.params = create_params(TID=training_TID)
                    if (j < 33):
                        core1 = layer.core.core
                    else:
                        core1 = layer.core

                    #Get the matrix from the layers at the training TID
                    matrix1 = core1.get_matrix()
                    if type(matrix1) == cupy.ndarray:
                        matrix1 = matrix1.get()
                        
                    alpha_denom_abs = np.sum(abs(matrix0 @ v_cal)) 
                    alpha_numer_abs = np.sum(abs(matrix1 @ v_cal))
        
                    alpha = alpha_numer_abs / alpha_denom_abs

                    #Calculate alpha and create a list of alpha values
                    alpha_list.append(alpha)
                    
                    j += 1
                    
            print("Alpha Values", alpha_list)

            for layer in analog_modules(model):

                layer.params = create_params(TID=training_TID)

                #During validation, calculate accuracy at TID level
                if phase == 'val':
                    layer.params = create_params(TID=TID)

            layers = analog_modules(model)
            for j in range(0, len(layers)):
                if (j < 33):
                    core = layers[j].core.core
                else:
                    core = layers[j].core
                core.alpha = alpha_list[j] #Set the core parameter alpha to the correct alpha value
    
            running_loss = 0.0
            running_corrects = 0
            running_corrects_val = 0

            # Iterate over data.
        
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    synchronize(model)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                        

                    l2_reg = 0.0
                    for param in model.parameters():
                        # Regularize the convolution kernels only
                        # Exclude dense kernel, all bias, and batch norm
                        if len(param.shape) == 4:
                            l2_reg += torch.norm(param, 2)

                    loss += l2_penalty * l2_reg

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == "val":
                        #Calculate the accuracy of the model at TID=0 as well and average it
                        for layer in analog_modules(model):
                            layer.params = create_params(TID=0)
                        #Set alpha=1 when evaluating at TID=0
                        for j in range(0, len(layers)):
                            if (j < 33):
                                core = layers[j].core.core
                            else:
                                core = layers[j].core
                                core.alpha = 1
                        outputs_val = model(inputs)
                        _, preds_val = torch.max(outputs_val, 1)

                        
                        for layer in analog_modules(model):
                            layer.params = create_params(TID=TID)

                        #Reset proper alpha values for the specified TID levels
                        layers = analog_modules(model)
                        for j in range(0, len(layers)):
                            if (j < 33):
                                core = layers[j].core.core
                            else:
                                core = layers[j].core
                            core.alpha = alpha_list[j]        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == "val":
                    running_corrects_val += torch.sum(preds_val == labels.data)
                
            if phase == 'train':
                epoch_loss = running_loss / (len(dataloaders[phase].dataset)*.9)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset)*.9)
                comb_acc = 0
            else:
                epoch_loss = running_loss / (len(dataloaders[phase].dataset)*.1)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset)*.1)
                opposite_acc = running_corrects_val.double() / (len(dataloaders[phase].dataset)*.1)

                #Combined accuracy averages the accuracy at 0 TID and the initial model TID
                comb_acc = (epoch_acc + opposite_acc) / 2

            print('{} Loss: {:.4f} Acc: {:.4f} Comb: {:.4f}'.format(phase, epoch_loss, epoch_acc, comb_acc))
            

            # deep copy the model
            if phase == 'val' and comb_acc > best_acc:
                best_acc = comb_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print("Current LR: " + str(optimizer.param_groups[0]["lr"]))
        print()
        reducer.step(running_loss)
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    for i in range(3,20): #Runs the training 20 times
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        depth = 6*n+2 #Depth 32
        model_name = "TinyImageNet"+str(depth)
    
        model = tiny_imagenet(n)
        model = model.to(device)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: "+str(params))
        print(summary(model,(3, 64, 64))) # Tiny Image Net image size - may need to change this

        # model.load_state_dict(torch.load("TinyImageNet32Noise_v2_Adam_30_0_10000.pth", weights_only = True))
        #changed
        params = CrossSimParameters()

        TID = 10e3
        alt = True
        dist = False
        mean = 0
        std_dev = 1e3
        
        cross_sim_model = model_creation(TID)
    
        # Using Adam optimizer and learning rate scheduler
        optimizer = "Adam"
        if optimizer == "Adam":
            print("Using Adam optimizer.")
            optimizer_ft = optim.Adam(cross_sim_model.parameters(), lr=1e-3, eps=1e-7)
        reducer = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ft,mode='min',factor=np.sqrt(0.1),patience=5,min_lr=0.5e-6)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer_ft,
            milestones=[20,25],
            gamma=0.1)
    
        # Set up the loss function
        criterion = nn.CrossEntropyLoss()
        print("Training " + str(model_name))
        train_loader, valid_loader = get_train_valid_loader(
            data_dir=data_dir,
            batch_size=batch_size,
            random_seed=int(np.random.rand()*100),
            augment=True
        )
    
        dataloaders_dict = { # added
            'train': train_loader,
            'val': valid_loader,
        }
            
        cross_sim_model = train_model(
            cross_sim_model, #changed
            dataloaders_dict,
            criterion,
            optimizer_ft,
            reducer,
            scheduler,
            optimizer,
            model_name,
            TID,
            alt, 
            dist,
            mean,
            std_dev,
            num_epochs=num_epochs,
            l2_penalty=l2_penalty,
        )
    
        test_loader = get_test_loader( # added
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False
        )
    
        cross_sim_model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = cross_sim_model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        test_loss = running_loss / (len(test_loader.dataset))
        test_acc = running_corrects.double() / (len(test_loader.dataset))
        with open(f"outputTinyImageNetNoise{6*n+2}_TestAccAltAlpha_{TID}.txt", "a") as f:  #Writes the accuracy to a file
            f.write("\nIteration: "+str(i)+"\n")
            f.write('Test accuracy: {:4f}'.format(test_acc))
            
    
        torch.save(cross_sim_model.state_dict(), model_name + "NoiseAltAlpha" + "_v2_" + str(optimizer) + "_" + str(num_epochs) + "_" + str(i) + "_" + str(int(TID)) + ".pth") #Saves the model weights
