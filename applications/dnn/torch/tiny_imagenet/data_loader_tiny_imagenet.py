#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:55:56 2025

@author: benjamintung
"""

"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision.transforms.autoaugment import AutoAugment

################################################################################
# This is a custom dataset class created to apply transforms                   #
# Parameters:                                                                  #
#    data             - the image data                                         #
#    target           - the image labels                                       #
#    transform        - image transforms                                       #
#    target_transform - image label transforms                                 # 
# Functions:                                                                   #
#    len: returns number of images in dataset                                  #
#    getitem: transforms the image data/labels and returns them                #
################################################################################

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y





def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           augment=True,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    #Define the training set transforms
    if augment:
        print('Using real-time data augmentation.')
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1))
        ])

    #Load the pre-processed data
    x_train = np.transpose(np.load("TIN_dataset/x_train.npy"), (0,3,1,2))
    y_train = np.load("TIN_dataset/y_train.npy")

    #Convert the data into a dataset
    train_dataset = MyDataset(data = torch.Tensor(x_train), target = torch.Tensor(y_train).long(), transform = train_transform)
    valid_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
    

    #Split the training set into 90% training and 10% validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print("train_idx", len(train_idx))
    print("valid_idx", len(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    #Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler = train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler = valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
                                  
    return train_loader, valid_loader


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    #Load the pre-processed test set data
    x_test = np.transpose(np.load("TIN_dataset/x_test.npy"), (0,3,1,2))
    y_test = np.load("TIN_dataset/y_test.npy")

    #Create the dataset
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())

    #Create the dataloader
    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader