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

n = 5 #Training Resnet 32

def train_model(model, dataloaders, criterion, optimizer, reducer, scheduler, 
    optimizer_name, model_name, num_epochs=30, l2_penalty=0):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(0,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # changed
            if phase == 'train':
                model.train()  # Set model to training mode (training sets)
            else:
                model.eval()   # Set model to evaluate mode (validation and testing sets)
                

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
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

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                epoch_loss = running_loss / (len(dataloaders[phase].dataset)*.9)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset)*.9)
            else:
                epoch_loss = running_loss / (len(dataloaders[phase].dataset)*.1)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset)*.1)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
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
    for i in range(0,20): #Runs the training 20 times
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
    
        # Using Adam optimizer and learning rate scheduler
        optimizer = "Adam"
        if optimizer == "Adam":
            print("Using Adam optimizer.")
            optimizer_ft = optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
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
            
        model = train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            reducer,
            scheduler,
            optimizer,
            model_name,
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
    
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        test_loss = running_loss / (len(test_loader.dataset))
        test_acc = running_corrects.double() / (len(test_loader.dataset))
        with open(f"outputTinyImageNet{6*n+2}_v2_TestAcc.txt", "a") as f:  #Writes the accuracy to a file
            f.write("\nIteration: "+str(i)+"\n")
            f.write('Test accuracy: {:4f}'.format(test_acc))
            
    
        torch.save(model.state_dict(), model_name + "_v2_" + str(optimizer) + "_" + str(num_epochs) + "_" + str(i) + ".pth") #Saves the model weights