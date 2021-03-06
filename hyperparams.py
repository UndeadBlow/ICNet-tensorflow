import numpy as np
import cv2
import math

#  Mean taken from Mapilary Vistas dataset
IMG_MEAN = np.array((126.63854281333334, 123.24824418666667, 113.14331923666667), dtype = np.float32)

BATCH_SIZE = 4
DATA_LIST_PATH = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/plain_train_no_apollo.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '560,560'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 18
NUM_STEPS = 200000
POWER = 0.001
RANDOM_SEED = 4321
WEIGHT_DECAY = 1e-4
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 400

USE_CLASS_WEIGHTS = False
#CLASS_WEIGHTS = [1.0, 1.0, 1.0]

#############################
# Augmentations
CROP_PROB = 0.5
MIN_CROP = 0.6
MAX_CROP = 1.0

PAD_PROB = 0.05
MIN_PAD = 1.1
MAX_PAD = 1.6

# FOCAL_PROB = 1.0
# MIN_FOCAL = 600
# MAX_FOCAL = 1000
##############################
##############################

# For ICNet, Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0

#label_colours = [(0, 0, 0), (128, 64, 128), (250, 0, 0)]
                # 0 void label, 1 = road, 2 = road mark

label_names = ['unlabeled', 'ground', 'road', 'sidewalk', 'rail track', 'building',
               'wall', 'fence', 'bridge', 'tunnel', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'person', 'car', 'motorcycle']
label_colours = [(0,  0, 0), (128, 64,128), (244, 35,232), (250,170,160), (70, 70, 70), (102, 102,156),
                 (190,153,153), (180,165,180), (150,120, 90), (153,153,153), (250,170, 30),
                 (220,220,  0), (107,142, 35), (152,251,152), (80, 150, 250), (255,  0,  0),
                 (0, 60,100), (0,  0,  255)]


# label_names = ['unlabeled', 'ground', 'road', 'sidewalk', 'rail track', 'building',
#                'wall', 'fence', 'bridge', 'tunnel', 'pole', 'traffic light',
#                'traffic sign', 'vegetation', 'terrain', 'person', 'car', 'truck', 'bus',
#                'train', 'motorcycle']
# label_colours = [(0,  0, 0), (128, 64,128), (244, 35,232), (250,170,160), (70, 70, 70), (102, 102,156),
#                  (190,153,153), (180,165,180), (150,120, 90), (153,153,153), (250,170, 30),
#                  (220,220,  0), (107,142, 35), (152,251,152), (80, 150, 250), (255,  0,  0),
#                  (0, 60,100), (0,  0, 90), (0,  0,230), (119, 11, 32),  (0,  0,  255)]


def draw_color_scheme():
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1
    display_str_height = cv2.getTextSize('t', font, font_scale, thickness)[0][1]

    label_color_size = (150, 70)
    w_margin = 80
    h_margin = display_str_height + 40
    w = int(4 * (label_color_size[0] + w_margin) + w_margin / 2)
    h = int(round(len(label_colours) / 4) * (label_color_size[1] + h_margin) + h_margin)
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas.fill(255)
    print(canvas.shape)

    current_x = 0
    current_y = 0
    for name, color in zip(label_names, label_colours):
        print(name, (int(color[2]), int(color[1]), int(color[0])))
        cv2.rectangle(canvas, (int(current_x + w_margin / 2), int(current_y + h_margin)),
                              (int(current_x + w_margin + label_color_size[0]), int(current_y + h_margin / 2 + label_color_size[1])),
                              (color[2], color[1], color[0]), cv2.FILLED)

        cv2.putText(canvas,
                    name,
                    (int(current_x + w_margin / 2), int(current_y + h_margin + label_color_size[1])),
                    font, font_scale, (0, 0, 0), thickness = thickness)

        current_x = current_x + w_margin + label_color_size[0]
        if current_x + w_margin / 2 + label_color_size[0] >= w:
            current_x = 0
            current_y = current_y + h_margin / 2 + label_color_size[1]

    cv2.imshow('Color scheme', canvas)
    cv2.waitKey(0)


if __name__ == '__main__':

    draw_color_scheme()
