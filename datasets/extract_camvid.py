import tensorflow as tf
import numpy as np
import cv2

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import shutil
import uuid
import copy
import os 
import cv2

def pure_name(name):
    name = name[name.rfind('/') + 1 : name.rfind('.')]
    name = name.replace('_L', '')
    return name

from pathlib import Path
sys.path.append('/home/undead/reps/tf_models/object_detection/datasets/')

import navmii_dataset_utils as navmii_utils

def rename_and_move(dist_dir, out_dir, start_frame, pattern, format_num = '5'):
    mask_files = navmii_utils.GetAllFilesListRecusive(out_dir, ['.png'])
    files = navmii_utils.GetAllFilesListRecusive(dist_dir, ['.jpeg'])
    labels_output = ''

    for name in files:
        if pattern in name:
            pur_name = pure_name(name).replace(pattern, '')
            pur_name = int(pur_name) + start_frame - 1
            pur_name = pattern + ('{:0' + format_num + '}').format(pur_name)

            filename = [f for f in mask_files if pure_name(f) == pur_name]

            if len(filename) == 1:
                im = cv2.imread(name)
                new_name = out_dir + '/' + pur_name + '.png'
                cv2.imwrite(new_name, im)
                labels_output = labels_output + new_name + ' ' + filename[0] + '\n'
            if len(filename) > 1:
                print('WTF????!!!', name, filename)

    with open('/mnt/Data/Datasets/Segmentation/Camvid/list.txt', 'a') as file:
        file.write(labels_output)


# 0001TP_ 6660 6
# 0006R0_f -1 5
# 0016E5_ -1 5
# Seq05VD_f -1 5
rename_and_move('/mnt/Data/Datasets/Segmentation/Camvid/video_and_frames', '/mnt/Data/Datasets/Segmentation/Camvid/dataset', 6660, '0001TP_', '6')
rename_and_move('/mnt/Data/Datasets/Segmentation/Camvid/video_and_frames', '/mnt/Data/Datasets/Segmentation/Camvid/dataset', -1, '0006R0_f', '5')
rename_and_move('/mnt/Data/Datasets/Segmentation/Camvid/video_and_frames', '/mnt/Data/Datasets/Segmentation/Camvid/dataset', -1, '0016E5_', '5')
rename_and_move('/mnt/Data/Datasets/Segmentation/Camvid/video_and_frames', '/mnt/Data/Datasets/Segmentation/Camvid/dataset', -1, 'Seq05VD_f', '5')