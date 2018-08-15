from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer_nets import *
from tools import decode_labels, prepare_label, inv_preprocess
from image_reader import ImageReader

from hyperparams import *

SNAPSHOT_DIR = './snapshots/'
RESTORE_FROM = ''

def get_arguments():
    parser = argparse.ArgumentParser(description="U-Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
#    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
#                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
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

def labels_to_one_hot(ground_truth, num_classes = NUM_CLASSES):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Uniform',  num_classes = NUM_CLASSES):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, num_classes)

    if weight_map is not None:
        n_classes = num_classes
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [n_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, n_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights,  tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score

def main():
    # lr_decay = 0.5
    # decay_every = 100
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    coord = tf.train.Coordinator()
    
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # config.intra_op_parallelism_threads = 1
    sess = tf.Session(config = config)
    net = unext(image_batch, is_train = True, reuse = False, n_out = NUM_CLASSES)
    
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_output = net.outputs
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), dtype = tf.int32)
    prediction = tf.gather(raw_prediction, indices)
                                                                                            
    main_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = gt)

    t_vars = tf.trainable_variables()
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in t_vars if 'kernel' in v.name]
    #reduced_loss = 0.5 * tf.reduce_mean(main_loss) + generalised_dice_loss(prediction, gt) + tf.add_n(l2_losses)
    reduced_loss = tf.reduce_mean(main_loss) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension = 3)
    pred = tf.expand_dims(raw_output_up, dim = 3)
    
    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    loss_summary = tf.summary.scalar('TotalLoss', reduced_loss)
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # Using Poly learning rate policy 
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.train.exponential_decay(base_lr, step_ph, args.num_steps, args.power)

    lr_summary = tf.summary.scalar('LearningRate', learning_rate)
    #train_op = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(reduced_loss, var_list = t_vars)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(reduced_loss, var_list = t_vars)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 10)

    ckpt = tf.train.get_checkpoint_state(SNAPSHOT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        #restore_vars = list([t for t in tf.global_variables() if not 'uconv1' in t.name])
        loader = tf.train.Saver(var_list = tf.global_variables())
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    # Iterate over training steps.
    save_summary_every = 10
    for step in range(args.num_steps):
        start_time = time.time()
        
        feed_dict = {step_ph: step}
        if not step % args.save_pred_every == 0:
            loss_value, _, l_summary, lr_summ = sess.run([reduced_loss, train_op, loss_summary, lr_summary], feed_dict=feed_dict)
            duration = time.time() - start_time
        elif step % args.save_pred_every == 0:
            loss_value, _, summary, l_summary, lr_summ = sess.run([reduced_loss, train_op, total_summary, loss_summary, lr_summary], feed_dict=feed_dict)
            duration = time.time() - start_time
            save(saver, sess, args.snapshot_dir, step)
            summary_writer.add_summary(summary, step)

        if step % save_summary_every == 0:
    
            summary_writer.add_summary(l_summary, step)
            summary_writer.add_summary(lr_summ, step)
        
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
        
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
