# ICNet_tensorflow

This a fork from (https://github.com/UndeadBlow/ICNet-tensorflow) repository. Some features were added, some removed. I've created that fork for myself, but maybe you will find it usefull too, because that version is more for training from scratch, while original repo more for inference.

## Differences with original (Important!!!)

Here is only list of changes. How to use them see below or in code directly.

- [x] Works with Python3 now
- [x] Added classes weights to training procedure. You can disable it.
- [x] Added "--not-restore-last" option in training, so now you can load pre-trained .npy or .tf checkpoint model to use it with different number of classes. Before you could use only 19.
- [x] Added loss, learning rate and images summaries, so now you can monitor training with Tensorboard.
- [x] All hardcoded parameters, like input size, is taken now from train.py and not duplicating. Please note, that for now inference.py works with pre-defined input size and not let to resize network to input image. That gives better perfomance.
- [x] You can evaluate now with "--repeated-eval" option. In that case evaluation will run endless and will write mIOU results to summary, so you can monitor it with Tensorboard.
- [x] You can use now "--best-models-dir". If it set, best model will be zipped on each iteration and saved to that path.
- [x] Added option "--ignore-zero" in evaluation. With that option results of zero class will be ignored for mIOU. It is very usefull in tasks where most place on image is taken with zero class.
- [x] Some minor changes and improvements. For example, size of validation set is calculating automatically from size of list.txt. It's not good to set it by hand every time.
- [ ] "--measure-time" not working for now. I plan to improve that procedure, in original code it was not fully correct.
- [ ] Some features was removed, for example you can't use evaluate.py with .npy models now. I don't think it's good to use two different formats. You can only load from .npy for training and use after that only .tf format of checkpoints.

## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on my own data, so don't forget to change labels and number of classes. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.
  
## Update

#### 2017/11/23:
1. Forked by me initial changes was done (described above in differences section). Models now stored in repository.

#### 2017/11/15:
1. Support `training phase`, you can train on your own dataset. Please read the guide below.

#### 2017/11/13:
1. Add `bnnomerge model` which reparing for training phase. Choose different model using flag `--model=train, train_bn, trainval, trainval_bn` (Upload model in google drive).
2. Change `tf.nn.batch_normalization` to `tf.layers.batch_normalization`.

#### 2017/11/07:
`Support every image size larger than 128x256` by changing the avg pooling ksize and strides in the pyramid module. If input image size cannot divided by 32, it will be padded in to mutiple of 32.


## Install
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyadTdHejU1Q1lfRkU?usp=sharing
) and put into `model` directory.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png
```

Inference time:  ~0.02s, I have no idea why it's faster than caffe implementation 

## Evaluation
Perform in single-scaled model on the cityscapes validation datase.

| Model | Accuracy |  Missing accuracy |
|:-----------:|:----------:|:---------:|
| train_30k   | **65.56/67.7** | **2.14%** |
| trainval_90k| **78.44%**    | None |

Example of usag:
```
python evaluate.py --model other --snapshot-dir=test --repeated-eval --data-list some_valid_list.txt --best-models-dir ./best_models
```

## Image Result
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/test_1024x2048.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/test_1024x2048.png)

## Training on your own dataset
> Note: This implementation is different from the details descibed in ICNet paper, since I did not re-produce model compression part. Instead, we train on the half kernel directly.

### Step by Step
**1. Change the `DATA_LIST_PATH`** in line 22, make sure the list contains the absolute path of your data files, in `list.txt`:
```
/ABSOLUTE/PATH/TO/image /ABSOLUTE/PATH/TO/label
```
**2. Set Hyperparameters** (line 21-35) in `train.py`
```
IMG_MEAN = np.array((106.33906592, 116.77648721, 119.91756518), dtype = np.float32)
BATCH_SIZE = 8
DATA_LIST_PATH = '/mnt/Data/Datasets/Segmentation/mapillary_vistas_3_class/list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '800,800'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 100000
POWER = 0.96
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './test/'
SAVE_NUM_IMAGES = 8
SAVE_PRED_EVERY = 150

Please note, that some of that parameters, like IMG_MEAN, will be used in inference and evaluation procedure.

Also change labels in tools.py with respect to your classes.
```
Also **set the loss function weight** (line 38-40) descibed in the paper:
```
# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.4
LAMBDA2 = 0.4
LAMBDA3 = 1.0
```
**3.** Run following command and **decide whether to update mean/var or train beta/gamma variable**. Remember to choose `--model=others`.
```
python train.py --update-mean-var --train-beta-gamma --not-restore-last
```
After training the dataset, you can run following command to get the result:  
```
python inference.py --img-path=YOUR_OWN_IMAGE
```

Also you can run model on video with utils.py script in dataset directory, but it is a little hardcoded for myself, so success is on your own.
### Result ( inference with my own data )

Input                      |  Output
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor1.jpg)  |  ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/output/indoor1.jpg)
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor3.jpg)  |  ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/output/indoor3.jpg)


## Citation
    @article{zhao2017icnet,
      author = {Hengshuang Zhao and
                Xiaojuan Qi and
                Xiaoyong Shen and
                Jianping Shi and
                Jiaya Jia},
      title = {ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
      journal={arXiv preprint arXiv:1704.08545},
      year = {2017}
    }
