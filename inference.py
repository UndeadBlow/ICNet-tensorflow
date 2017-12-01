from __future__ import print_function

import sys
from pathlib import Path
sys.path.append('../')
sys.path.append('./')
sys.path.append('./datasets')

import argparse
import os

import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet_BN
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
num_classes = NUM_CLASSES

snapshot_dir = './snapshots/'

SAVE_DIR = './output/'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.",
                        required=True)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--snapshots-dir", type=str, default=snapshot_dir,
                        help="Path to checkpoints.")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="If true, will output weighted images")


    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    
    shape = INPUT_SIZE.split(',')
    img = cv2.resize(img, (int(shape[0]), int(shape[1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('input image shape: ', img.shape)

    return img, filename

def preprocess(img):
    # Convert RGB to BGR
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(img, dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    
    img = tf.expand_dims(img, dim=0)

    return img

def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h/32) + 1) * 32
        new_w = (int(ori_w/32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

def main():
    args = get_arguments()
    
    if args.img_path[-4] != '.':
        files = GetAllFilesListRecusive(args.img_path, ['.jpg', '.jpeg', '.png'])
    else:
        files = [args.img_path]


    shape = INPUT_SIZE.split(',')
    shape = (int(shape[0]), int(shape[1]), 3)

    x = tf.placeholder(dtype = tf.float32, shape = shape)
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)

    # Create network.
    net = ICNet_BN({'data': img_tf}, is_training = False, num_classes = num_classes)
    
    # Predictions.
    raw_output = net.layers['conv6_cls']
    output = tf.image.resize_bilinear(raw_output, tf.shape(img_tf)[1:3,])
    output = tf.argmax(output, dimension = 3)
    pred = tf.expand_dims(output, dim = 3)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.snapshots_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)

    for path in files:

        img, filename = load_img(path)

        preds = sess.run(pred, feed_dict={x: img})

        msk = decode_labels(preds, num_classes=num_classes)
        im = msk[0]

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if args.weighted:
            indx = (im == [0, 0, 0])
            im = cv2.addWeighted(im, 0.7, img, 0.7, -15)
            im[indx] = img[indx]

        cv2.imwrite(args.save_dir + filename.replace('.jpg', '.png'), im)

if __name__ == '__main__':
    main()
