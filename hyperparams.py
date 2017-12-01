import numpy as np


#  Mean taken from Mapilary Vistas dataset
IMG_MEAN = np.array((106.33906592, 116.77648721, 119.91756518), dtype = np.float32)

BATCH_SIZE = 8
DATA_LIST_PATH = '/mnt/Data/Datasets/Segmentation/vistas_no_pp/list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '800,800'
LEARNING_RATE = 5e-2
MOMENTUM = 0.95
NUM_CLASSES = 3
NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.000001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 8
SAVE_PRED_EVERY = 200

USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.0, 3.0]


# Crop will be re-tried until at least one pixel of that class is on image. Set to -1 to disable
CROP_MUSTHAVE_CLASS_INDEX = 2

# For ICNet, Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.4
LAMBDA2 = 0.4
LAMBDA3 = 1.0

# Pairs epoch - learning rate. Set to {} to remove LR shedule
LR_SHEDULE = {0 : 5e-2, 20000 : 3e-3, 60000 : 1e-4}