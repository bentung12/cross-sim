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

model = tiny_imagenet(5)
for name, layer in model.named_modules():
    print(name, "->", layer)

model.load_state_dict(torch.load("models/TinyImageNet32_v2_Adam_30_1.pth", weights_only = True))

linear_layer = model.classifier[2].weight.detach().numpy()
print("Size", linear_layer.shape)
np.save("linear_layer.npy", linear_layer)