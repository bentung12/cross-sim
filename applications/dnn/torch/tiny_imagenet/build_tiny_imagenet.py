#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:59:01 2025

@author: benjamintung
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import sys

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes,stride)
        self.bn1 = norm_layer(planes, momentum=0.01, eps=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=0.01, eps=0.001, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class tiny_imagenet(nn.Module):

    def __init__(self, num_blocks, in_channels=3, num_classes=200): #modified number of classes 200, num_blocks should be 5
        super().__init__()
        self.in_planes = 64 #changed to 64 from 16
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=0.01, eps=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self._make_layer(64, num_blocks, stride=1)
        self.res2 = self._make_layer(128, num_blocks, stride=2)
        self.res3 = self._make_layer(256, num_blocks, stride=2)
        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
        #                                 nn.Flatten(), 
        #                                 nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Linear(256, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, num_classes))
        
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
        
        self.conv1.apply(init_kernel)
        self.res1.apply(init_kernel)
        self.res2.apply(init_kernel)
        self.res3.apply(init_kernel)
        self.classifier.apply(init_kernel)


    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * BasicBlock.expansion, stride)
            )
        layers = []
        layers.append(
            BasicBlock(self.in_planes, planes*BasicBlock.expansion, stride, downsample))
        self.in_planes = planes*BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.classifier(out)
        return out

def init_kernel(m):
    if isinstance(m, nn.Conv2d):
        # Initialize kernels of Conv2d layers as kaiming normal
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # Initialize biases of Conv2d layers at 0
        nn.init.zeros_(m.bias)