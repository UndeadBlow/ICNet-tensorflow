import tensorflow as tf
import numpy as np
import cv2

import sys

from pathlib import Path
sys.path.append('../')
sys.path.append('./')

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
import shutil
import uuid
import copy
import os 
import cv2
import json

import argparse
import os
import time
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet, ICNet_BN
from tools import decode_labels

from inference import load, load_img, preprocess, check_input
from train import INPUT_SIZE, IMG_MEAN

def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all


def pure_name(name):
    name = name[name.rfind('/') + 1 : name.rfind('.')]
    name = name.replace('_L', '')
    return name

def run_on_video(video_filename, out_filename, model_path, num_classes, save_to = 'simple', canvas_size = (1600, 800), alpha = 0.8, beta = 0.2, output_size = (1280, 720)):
    '''
    save_to: simple, double_screen or weighted
    '''
    input_size = (int(INPUT_SIZE.split(',')[0]), int(INPUT_SIZE.split(',')[0]))
    x = tf.placeholder(dtype=tf.float32, shape=(int(input_size[0]), int(input_size[1]), 3))
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)
    
    net = ICNet_BN({'data': img_tf}, num_classes = num_classes)
    
    raw_output = net.layers['conv6_cls']
    
    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
    #raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, INPUT_SIZE[0], INPUT_SIZE[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()
    
    print('model_path', model_path)
    ckpt = tf.train.latest_checkpoint(model_path)
    if len(ckpt):
        loader = tf.train.Saver(var_list = restore_var)
        load(loader, sess, ckpt)



######
    cap = cv2.VideoCapture(video_filename)

    out_cap = None
    if save_to == 'double_screen':
        out_cap = cv2.VideoWriter(out_filename.replace('.mp4', '.avi'), 
                cv2.VideoWriter_fourcc(*"MJPG"), 20, (canvas_size[0], canvas_size[1]))
    elif save_to == 'weighted':
        out_cap = cv2.VideoWriter(out_filename.replace('.mp4', '.avi'), 
                cv2.VideoWriter_fourcc(*"MJPG"), 20, (output_size[0], output_size[1]))

    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Error opening video stream or file")
        return


    frame_num = 0
    while(cap.isOpened()):
        frame_num = frame_num + 1
        print('Processing frame', frame_num)
        
        # Capture frame-by-frame
        ret, image = cap.read()

        if out_cap == None:
            out_cap = cv2.VideoWriter(out_filename.replace('.mp4', '.avi'), 
                cv2.VideoWriter_fourcc(*"MJPG"), 20, (image.shape[0], image.shape[1]))

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size[0], input_size[1]))

        preds = sess.run(pred, feed_dict={x: image})
        msk = decode_labels(preds, num_classes=num_classes)
        frame = msk[0]

        #cv2.imshow('1', frame)
        #cv2.waitKey(0)
        #print(frame.shape)
        
        

        if save_to == 'double_screen':

            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype = np.uint8)
            #frame_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_orig = cv2.resize(image, (int(canvas_size[0] / 2), int(canvas_size[1])))

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(canvas_size[0] / 2), int(canvas_size[1])))

            canvas[:, 0 : int(canvas_size[0] / 2), :] = frame_orig
            canvas[:, int(canvas_size[0] / 2) :, : ] = frame
            #cv2.imshow('1', frame)
            #cv2.waitKey(0)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            print('canvas shape', canvas.shape)

            out_cap.write(canvas)



        elif save_to == 'simple':

            frame = cv2.resize(frame, (image.shape[0], image.shape[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_cap.write(frame)



        elif save_to == 'weighted':

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, output_size)
            image = cv2.resize(image, output_size)

            frame = cv2.addWeighted(image, alpha, frame, beta, 0)
            
            #cv2.imshow('1', frame)
            #cv2.waitKey(0)

            out_cap.write(frame)



    cap.release()
    out_cap.release()


if __name__ == '__main__':
    run_on_video('/home/undead/Downloads/lanes/lanes3.mp4', '/home/undead/Downloads/lanes/lanes3_out_w_2.mp4',
                 '/home/undead/reps/ICNetUB/best_models/miou_0.4328', num_classes = 3, save_to = 'weighted', alpha = 0.5, beta = 0.5)