import numpy as np


#  Mean taken from Mapilary Vistas dataset
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype = np.float32)

BATCH_SIZE = 1
DATA_LIST_PATH = '/mnt/Data/Datasets/Segmentation/mapillary-vistas-dataset_public_v1.0/cityscaped.txt'
IGNORE_LABEL = 0
INPUT_SIZE = '360,360'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 25
NUM_STEPS = 200000
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 20

USE_CLASS_WEIGHTS = False
#CLASS_WEIGHTS = [1.0, 1.0, 1.0]

#############################
# Augmentations
CROP_PROB = 0.8
MIN_CROP = 0.9
MAX_CROP = 1.0

PAD_PROB = 0.05
MIN_PAD = 1.0
MAX_PAD = 1.1
##############################
##############################

# For ICNet, Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0

# Pairs epoch - learning rate. Set to {} to remove LR shedule
LR_SHEDULE = {}

#label_colours = [(0, 0, 0), (128, 64, 128), (250, 0, 0)]
                # 0 void label, 1 = road, 2 = road mark
label_names = ['unlabeled', 'ground', 'road', 'sidewalk', 'rail track', 'building',
               'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
               'trailer', 'train', 'motorcycle']
label_colours = [(0,  0, 0), (128, 64,128), (244, 35,232), (250,170,160), (70, 70, 70), (102,102,156), 
                 (190,153,153), (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (250,170, 30),
                 (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (80, 150, 250), (255,  0,  0), (  0,  0,142),
                 (  0,  0, 70), (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32) ]


                 