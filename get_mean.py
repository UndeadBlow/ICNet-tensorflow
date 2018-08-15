from __future__ import print_function

import sys
from pathlib import Path
sys.path.append('../')
sys.path.append('./')
sys.path.append('./datasets')

import argparse
import os
import copy

import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from tensorlayer_nets import *
from model import *
from tools import decode_labels

import train
from train import INPUT_SIZE, IMG_MEAN, NUM_CLASSES

import cv2

def GetAllFilesListRecusive(path, extensions):
    
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all

# vistas valid [120.31763023 117.31701579 107.41887195]
# vistas train [119.74690619 116.83774316 106.77015853]
# apollo train [139.85109202 135.58997361 125.24092723]
# total [126.63854281333334, 123.24824418666667, 113.14331923666667]
files = GetAllFilesListRecusive('/mnt/Data/Datasets/Segmentation/Apollo/remapped', ['.jpg'])

w = int(INPUT_SIZE.split(',')[0])
h = int(INPUT_SIZE.split(',')[1])
mean_img = np.zeros((w, h, 3))
i = 0

for file in files:
    print('img ', i, len(files))
    i = i + 1
    mean_img = mean_img + cv2.resize(cv2.imread(file), (w, h))

print(np.mean(mean_img / len(files), axis = (0, 1)))