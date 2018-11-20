r"""Saves out a GraphDef containing the architecture of the model."""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

import sys
from pathlib import Path
sys.path.append('./bisenet')
from bisenet.models.BiSeNet import build_bisenet
import bisenet.utils.helpers as helpers

from model import *
from tensorlayer_nets import *

from tools import decode_labels, prepare_label, inv_preprocess
from image_reader import ImageReader
from inference import preprocess, check_input

from hyperparams import *

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

#tf.app.flags.DEFINE_integer(
#    'image_size', None,
#    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)

        x = tf.placeholder(name = 'input', dtype = tf.float32, shape = (1, shape[0], shape[1], 3))

        img_tf = tf.cast(x, dtype = tf.float32)
        # Extract mean.
        img_tf -= IMG_MEAN

        print(img_tf)
        # Create network.
        #net = psp_net({'inputs': img_tf}, is_training = False, num_classes = NUM_CLASSES)
        #net = unext(img_tf, is_train = False, n_out = NUM_CLASSES)
        #net = ICNext({'data': img_tf}, is_training = False, num_classes = NUM_CLASSES)
        net, init_fn = build_bisenet(img_tf, NUM_CLASSES, pretrained_dir = './bisenet/utils/models', is_training = False)

        #raw_output = net.outputs
        #raw_output = net.layers['conv6']
        output = tf.image.resize_bilinear(net, shape[:2], name = 'raw_output')
        output = tf.argmax(output, dimension = 3)
        pred = tf.expand_dims(output, dim = 3, name = 'indices')

        # Adding additional params to graph. It is necessary also to point them as outputs in graph freeze conversation, otherwise they will be cuted
        tf.constant(label_colours, name = 'label_colours')
        tf.constant(label_names, name = 'label_names')
        
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), 3)
        tf.constant(shape, name = 'input_size')
        tf.constant(['indices'], name = 'output_name')

        graph_def = graph.as_graph_def()

        for node in graph_def.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
            print('Successfull written to', FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
