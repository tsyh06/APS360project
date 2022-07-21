# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import itertools
import cv2
import splitfolders

import os
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

from io import StringIO

# set up directory path 
data_dir = os.getcwd()+"/data/classes"
split_dir = os.getcwd()+"/split"
train_dir = split_dir + "/train"
val_dir = split_dir + "/val"
test_dir = split_dir + "/test"

!unzip 'v2_processed_data_transition.zip' -d data

# split the images into 3 sub folder for train, validation and test
splitfolders.ratio(data_dir, output=split_dir, seed=1337, ratio=(0.8, 0.1,0.1))

# load the images as 1d array
train_img = []
train_labels = []
for labels_name in listdir(train_dir):
  img_dir = join(train_dir,labels_name)
  for image_name in listdir(img_dir):
    img_path = join(img_dir, image_name)
    if isfile(img_path):
      image_rgb = cv2.imread(img_path)
      image_rgb = cv2.resize(image_rgb, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
      image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
      image_gray = image_gray.tolist()
      pixel_list = list(itertools.chain.from_iterable(image_gray))
      train_img.append(pixel_list)

      train_labels.append(labels_name)

# Train the random forest classifier
model = RandomForestClassifier()
model.fit(train_img, train_labels)

# Print the test accuracy for individual class 
test_image = []
test_label = []

for labels_name in listdir(test_dir):
  img_dir = join(test_dir,labels_name)
  img = []
  label =[]
  for image_name in listdir(img_dir):
    img_path = join(img_dir, image_name)
    if isfile(img_path):
      image_rgb = cv2.imread(img_path)
      image_rgb = cv2.resize(image_rgb, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
      image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
      image_gray = image_gray.tolist()
      pixel_list = list(itertools.chain.from_iterable(image_gray))
      img.append(pixel_list)
      label.append(labels_name)
      
      test_image.append(pixel_list)
      test_label.append(labels_name)   

  score = model.score(img, label)
  print(labels_name, " : ", score)

# get the general test accuracy of the model
model.score(test_image, test_label)

# check the architecture of this random forest classifier
model.estimators_[0].tree_.max_depth
i=0
deep=0
for trees in model.estimators_:
  deep+=trees.tree_.max_depth
  i+=1
print("i: ",i)
print ("avg: ", deep/i)

# export the first 2 depth of the first decision tree as a clear png
export_graphviz(model.estimators_[0],
                out_file='tree.dot',
                max_depth=2,
               filled = True)
os.system('dot -Tpng tree.dot -o tree.png')
