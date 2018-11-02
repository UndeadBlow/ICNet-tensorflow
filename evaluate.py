from __future__ import print_function
import argparse
import os
import sys
import time
import shutil
import zipfile

from PIL import Image
import tensorflow as tf
import numpy as np

from model import *
from tensorlayer_nets import *
from tools import decode_labels
from image_reader import ImageReader
import logging
from inference import GetAllFilesListRecusive

from train import IMG_MEAN, NUM_CLASSES, INPUT_SIZE, IGNORE_LABEL

def calc_size(filename):
    size = 0
    with open(filename, 'r') as f:
        for line in f:
            size = size + 1

    return size

SAVE_DIR = './output/'

DATA_LIST_PATH = '/mnt/Data/Datasets/Autovision/v0beta/test.txt'

snapshot_dir = './snapshots'
best_models_dir = './best_models'

num_classes = NUM_CLASSES

num_steps = calc_size(DATA_LIST_PATH) # numbers of images in validation set
time_list = []
INTERVAL = 120
INPUT_SIZE = INPUT_SIZE.split(',')
INPUT_SIZE = [int(INPUT_SIZE[0]), int(INPUT_SIZE[1])]
IGNORE_LABEL = IGNORE_LABEL
batch_size = 1


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Text file with pairs image-answer")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--snapshot-dir", type=str, default=snapshot_dir,
                        help="Path to load")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--repeated-eval", action="store_true",
                        help="Run repeated evaluation for every checkpoint.")
    parser.add_argument("--ignore-zero", action="store_true",
                        help="If true, zero class will be ignored for total score")
    parser.add_argument("--best-models-dir", type=str, default='',
                        help="If set, best mIOU checkpoint will be saved in that dir in .zip format")
    parser.add_argument("--eval-interval", type=int, default=INTERVAL,
                        help="How often to evaluate model, seconds")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Size of batch")



    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def save_model(step, iou, checkpint_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    files = os.listdir(checkpint_dir)

    files = [os.path.abspath(checkpint_dir + '/' + f) for f in files]

    filename = list([f for f in files if str(step) in f])

    if len(filename) != 3 and len(filename) != 4:
        return

    iou = '{0:.4f}'.format(iou)
    zipfile_name = output_dir + '/miou_{0}.zip'.format(iou)

    print('Saving stpe {} with mIOU {} in file {}'.format(step, iou, zipfile_name))

    zf = zipfile.ZipFile(zipfile_name, "w", zipfile.ZIP_DEFLATED)
    for f in filename:
        zf.write(f, os.path.basename(f))
    zf.close()

def load_last_best_iou(dir):

    if not os.path.exists(dir):
        return 0.0

    files = os.listdir(dir)

    best_iou = 0.0
    for f in files:

        iou = float(f[f.rfind('miou_') + 5 : f.rfind('.')])
        if iou > best_iou:
            best_iou = iou

    return best_iou

def evaluate_checkpoint(model_path, args):
    coord = tf.train.Coordinator()

    tf.reset_default_graph()

    reader = ImageReader(
            args.data_list,
            INPUT_SIZE,
            random_scale = False,
            random_mirror = False,
            ignore_label = IGNORE_LABEL,
            img_mean = IMG_MEAN,
            coord = coord,
            train = False)
    image_batch, label_batch = reader.dequeue(args.batch_size)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    # Create network.
    net = ICNext({'data': image_batch}, is_training = False, num_classes = num_classes)
    #net = unext(image_batch, is_train = False, n_out = NUM_CLASSES)

    # Predictions.
    raw_output = net.layers['conv6']
    #raw_output = net.outputs

    raw_output_up = tf.image.resize_bilinear(raw_output, size = INPUT_SIZE, align_corners = True)
    raw_output_up = tf.argmax(raw_output_up, dimension = 3)
    pred = tf.expand_dims(raw_output_up, dim = 3)

    # mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, IGNORE_LABEL)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    iou_metric, iou_op = tf.metrics.mean_iou(pred, gt, num_classes = num_classes)
    acc_metric, acc_op = tf.metrics.accuracy(pred, gt)

    # Summaries
    iou_summ_op = tf.summary.scalar('mIOU', iou_metric)
    acc_summ_op = tf.summary.scalar('Accuracy', acc_metric)
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                        time.gmtime()))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(var_list = tf.global_variables())
    load(saver, sess, model_path)

    for step in range(int(num_steps / batch_size)):
        preds, _, _ = sess.run([raw_output_up, iou_op, acc_op])

        if step % int(100 / batch_size) == 0:
            print('Finish {0}/{1}'.format(step + 1, int(num_steps / batch_size)))

    iou, iou_summ, acc, acc_summ = sess.run([iou_metric, iou_summ_op, acc_metric, acc_summ_op])

    sess.close()

    coord.request_stop()
    #coord.join(threads)

    return iou, iou_summ, acc, acc_summ

# def evaluate_checkpoint(model_path, args):
#     # Set placeholder
#     image_filename = tf.placeholder(dtype=tf.string)
#     anno_filename = tf.placeholder(dtype=tf.string)
#
#     # Read & Decode image
#     img = tf.image.decode_jpeg(tf.read_file(image_filename), channels=3)
#     anno = tf.image.decode_png(tf.read_file(anno_filename), channels=1)
#
#     ori_shape = tf.shape(img)
#     img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
#     img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
#
#     # Extract mean.
#     img = tf.image.resize_images(img, INPUT_SIZE)
#     img = img - IMG_MEAN
#     img = tf.expand_dims(img, dim = 0)
#     h, w = INPUT_SIZE
#     img.set_shape([1, h, w, 3])
#     anno = tf.image.resize_nearest_neighbor(tf.expand_dims(anno, 0), INPUT_SIZE)
#     anno = tf.squeeze(anno, squeeze_dims=[0])
#     anno.set_shape([h, w, 1])
#     net = ICNet_BN({'data': img}, is_training = False, num_classes = num_classes)
#
#     # Predictions.
#     raw_output = net.layers['conv6']
#
#     raw_output_up = tf.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
#     raw_output_up = tf.argmax(raw_output_up, axis=3)
#     raw_pred = tf.expand_dims(raw_output_up, dim=3)
#
#     # mIoU
#     pred_flatten = tf.reshape(raw_pred, [-1,])
#     raw_gt = tf.reshape(anno, [-1,])
#
#     #indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)
#     mask = tf.less_equal(raw_gt, num_classes - 1)
#     indices = tf.squeeze(tf.where(mask), 1)
#     gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
#     pred = tf.gather(pred_flatten, indices)
#
#     mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=NUM_CLASSES)
#     miou_op = tf.summary.scalar('mIOU', mIoU)
#
#     # Set up tf session and initialize variables.
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     init = tf.global_variables_initializer()
#     local_init = tf.local_variables_initializer()
#
#     sess.run(init)
#     sess.run(local_init)
#
#     restore_var = tf.global_variables()
#     saver = tf.train.Saver(var_list = restore_var)
#     load(saver, sess, model_path)
#
#     imgs = []
#     labels = []
#     with open(DATA_LIST_PATH, 'r') as f:
#         for line in f:
#             if line.strip():
#                 imgs.append(line.split(' ')[0].strip())
#                 labels.append(line.split(' ')[1].strip())
#     if len(imgs) != len(labels):
#         print('WTF imgs != labels')
#         quit()
#
#     for i in range(len(imgs)):
#         if i % 100:
#             print('Finish {0}/{1}'.format(i + 1, len(imgs)))
#
#         feed_dict = {image_filename: imgs[i], anno_filename: labels[i]}
#         _ = sess.run(update_op, feed_dict=feed_dict)
#
#
#     iou, summ = sess.run([mIoU, miou_op])
#
#     return summ, iou
#########################################################


def main():
    args = get_arguments()

    if args.repeated_eval:

        last_evaluated_model_path = None

        while True:
            start = time.time()

            best_iou = load_last_best_iou(args.best_models_dir)

            model_path = tf.train.latest_checkpoint(args.snapshot_dir)

            if not model_path:
                logging.info('No model found')
            elif model_path == last_evaluated_model_path:
                logging.info('Found already evaluated checkpoint. Will try again in %d '
                    'seconds', args.eval_interval)
            else:
                global_step = int(os.path.basename(model_path).split('-')[1])
                last_evaluated_model_path = model_path
                number_of_evaluations = 0

                eval_path = args.snapshot_dir + '/eval'
                if not (os.path.exists(eval_path)):
                    os.mkdir(eval_path)

                summary_writer = tf.summary.FileWriter(eval_path)

                iou, iou_summ, acc, acc_summ  = evaluate_checkpoint(last_evaluated_model_path, args)
                print('Step', global_step, ', mIOU:', iou)
                print('Step', global_step, ', Accuracy:', acc)

                if iou > best_iou:
                    if len(args.best_models_dir):
                        save_model(global_step, iou, args.snapshot_dir, args.best_models_dir)
                    best_iou = iou

                print('Best for now mIOU: {}'.format(best_iou))

                summary_writer.add_summary(iou_summ, global_step)
                summary_writer.add_summary(acc_summ, global_step)
                number_of_evaluations += 1

                ########################

                time_to_next_eval = start + args.eval_interval - time.time()

                if time_to_next_eval > 0:

                    time.sleep(time_to_next_eval)

    # run once. Not tested yet
    else:

        model_path = tf.train.latest_checkpoint(args.snapshot_dir)
        global_step = int(os.path.basename(model_path).split('-')[1])
        summ, iou = evaluate_checkpoint(model_path, args)
        print('Step', global_step, ', mIOU:', iou)


if __name__ == '__main__':
    main()