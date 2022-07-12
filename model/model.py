use_cuda = True

import os
import numpy as np
import torch
import time

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #for gradient descent


# Classifier
class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(256 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 70)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


