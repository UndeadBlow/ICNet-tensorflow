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
import zipfile
import argparse
import os
import time
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet, ICNet_BN
from tools import decode_labels

from inference import load, load_img, preprocess, check_input, GetAllFilesListRecusive
from train import INPUT_SIZE, IMG_MEAN

import matplotlib.pyplot as plt

def pure_name(name):
    name = name[name.rfind('/') + 1 : name.rfind('.')]
    name = name.replace('_L', '')
    return name

def run_on_video(video_filename, out_filename, model_path, num_classes, save_to = 'simple', canvas_size = (1600, 800), alpha = 0.8, beta = 0.2, output_size = (1280, 720), step = 1):
    '''
    save_to: simple, double_screen or weighted
    '''
    input_size = (int(INPUT_SIZE.split(',')[0]), int(INPUT_SIZE.split(',')[0]))
    x = tf.placeholder(dtype=tf.float32, shape=(int(input_size[0]), int(input_size[1]), 3))
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)
    
    net = ICNet_BN({'data': img_tf}, num_classes = num_classes)
    
    raw_output = net.layers['conv6']
    
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
                cv2.VideoWriter_fourcc(*"MJPG"), 60, (canvas_size[0], canvas_size[1]))
    elif save_to == 'weighted':
        out_cap = cv2.VideoWriter(out_filename.replace('.mp4', '.avi'), 
                cv2.VideoWriter_fourcc(*"MJPG"), 60, (output_size[0], output_size[1]))

    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Error opening video stream or file")
        return


    frame_num = 0
    zf = None
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, image = cap.read()


        frame_num = frame_num + 1
        if frame_num % step != 0:
            continue
        print('Processing frame', frame_num)
        
        if out_cap == None and save_to != 'images':
            out_cap = cv2.VideoWriter(out_filename.replace('.mp4', '.avi'), 
                cv2.VideoWriter_fourcc(*'MJPG'), 60, (image.shape[1], image.shape[0]))
        elif save_to == 'images' and zf == None:
            zipfile_name = out_filename.replace('.avi', '.zip')

        original_shape = image.shape
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

            frame = cv2.resize(frame, (original_shape[1], original_shape[0]), interpolation = cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow('1', frame)
            #cv2.waitKey(0)
            out_cap.write(frame)

        
        elif save_to == 'images':

            frame = cv2.resize(frame, (original_shape[1], original_shape[0]), interpolation = cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation = cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/tmp/1.png', frame)
            #cv2.imwrite('/tmp/1_orig.png', image)
            zf = zipfile.ZipFile(zipfile_name, "a", zipfile.ZIP_DEFLATED)
            name = 'frame_' + '%08d' % frame_num + '.png'
            orig_name = 'frame_orig_' + '%08d' % frame_num + '.png'
            zf.write('/tmp/1.png', name)
            #zf.write('/tmp/1_orig.png', orig_name)
            zf.close()

        elif save_to == 'weighted':

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, output_size)
            image = cv2.resize(image, output_size)

            frame = cv2.addWeighted(image, alpha, frame, beta, 0)
            

            out_cap.write(frame)

        elif save_to == 'perspective':

            preds = preds.squeeze()

            img, mask = getCutedRoad(image, preds)

            mask = np.expand_dims(mask, axis = 0)
            mask = np.expand_dims(mask, axis = 3)

            msk = decode_labels(mask, num_classes=num_classes)
            f = msk[0]

            # h, w = frame.shape[:2]


            # src = np.float32([[x1, y1],    # br
            #           [x0, y1],    # bl
            #           [x0, y0],   # tl
            #           [x1, y0]])  # tr

            # dst = np.float32([[w, h],       # br
            #                 [0, h],       # bl
            #                 [0, 0],       # tl
            #                 [w, 0]])      # tr


            # M = cv2.getPerspectiveTransform(src, dst)
            # Minv = cv2.getPerspectiveTransform(dst, src)

            # warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

            # cv2.imshow('1', warped)

            #resized = cv2.resize(img[y0 : y1, x0 : x1], input_size, interpolation = cv2.INTER_NEAREST)

            print((preds == 2).sum() / (preds.shape[0] * preds.shape[1]))
            mask = np.array(mask)
            mask = mask.squeeze()
            print((mask == 2).sum() / (mask.shape[0] * mask.shape[1]))

            cv2.imshow('2', cv2.resize(img, input_size, interpolation = cv2.INTER_NEAREST))
            cv2.imshow('3', cv2.resize(f, input_size, interpolation = cv2.INTER_NEAREST))
            cv2.imshow('4', image)
            cv2.waitKey(0)
            #quit()


    cap.release()
    out_cap.release()
    zf.close()

def getCutedRoad(img, mask):

    x1, y1, x0, y0 = getRoadCoords(mask)
    return img[y0 : y1, x0 : x1], mask[y0 : y1, x0 : x1]

def getRoadCoords(mask, road_index = 1):
    indx = np.argwhere(mask == road_index)

    # sort by smallest x
    indx = sorted(indx, key=lambda x: x[0], reverse=False)

    y0, y1 = indx[0][0], indx[-1][0]

    indx = sorted(indx, key=lambda x: x[1], reverse=False)

    x0, x1 = indx[0][1], indx[-1][1]

    return x1, y1, x0, y0


if __name__ == '__main__':
    #run_on_video('/home/undead/Downloads/lanes/nick.avi', '/home/undead/Downloads/lanes/nick_out_800.mp4',
    #             '/home/undead/reps/ICNetUB/best_models/miou_0.4607', num_classes = 3, save_to = 'images', alpha = 0.5, beta = 0.5, step = 1)
    run_on_video('/home/undead/segment/lkas.mp4', '/home/undead/segment/lkas_out.avi',
                 '/home/undead/reps/ICNetUB/miou_0.4648_1280', num_classes = 3, save_to = 'weighted', alpha = 0.5, beta = 0.5, step = 1)