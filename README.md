# ICNet Tensorflow

This is fork from (https://github.com/hellochick/ICNet-tensorflow) repository. Some features were added, some removed. I've created that fork for myself, but maybe you will find it usefull too, because that version is more for training from scratch, while original repo more for inference.
I tried to create system that looks more like Tensorflow Models Object Detection API.

ICNet original paper: https://arxiv.org/abs/1704.08545. Also check citation in the end.

## Features of that repository

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
- [x] "--measure-time" for inference works fine now and measures quite objective.
- [x] Added scripts for exporting and quantizing graph for production usage. That works also similar to Tensorflow Object Detection API pipeline.
- [ ] Some features was removed, for example you can't use evaluate.py with .npy models now. I don't think it's good to use two different formats. You can only load from .npy for training and use after that only .tf format of checkpoints.

**Please, note that for now that repo is quite imperfect, because was created for one specific task. But mostly I mean interface and I tried to adapt it for common segmentation tasks with any number of classes. Main problem now is that a half of parameters are set in hyperparams.py, while the rest you must set by command line arguments. So for now, please, prefer to use hyperparams.py for changing hyperparameters, especially input size. 
Also try to check Update section to be sure you know last changes**

## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on my own data, so don't forget to change labels and number of classes. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.
  
## Change list

#### 2017/12/10
1. Moved --measure-time option to inference.py script. More suitable here.
2. Added export_inference_graph.py that creates frozen model (from checkpoints). Outputs .pb file that can be used for production.
3. Added example of usage of exporting graph. Check export_model.sh. In that script used all pipeline: first creating frozen graph without weights, then unfrozen with weights and then cleaned and quantized graph, if you wish.

#### 2017/12/01
1. Added --measure-time for evaluation. Pass it to evaluation to get time measures instead of validation.
2. Fixed classes weights, now that feature must works much better.
3. Moved all hyperparams to one place: hyperparams.py. Control them now from this file.
4. Added PSPNet model, you can use train_psp.py to train it. Not all fatures works for that model.
5. Added --weighted for inference, with that option output will be a combination of input and output, very very illustrative.
6. Now inference will work on directory (will inference all images in dir) if you will pass path to directory in --img-path argument.
7. Random crop was fully rewritten. Random pad and random scale now not working (coming soon), but crop works correctly now. In original it worked only with random scale (incorrectly) and always was with padding. You can control crop rate with minval and maxval in image_reader.py:
```
h_rate = tf.random_uniform(shape = [1], minval = 0.5, maxval = 1.0, dtype = tf.float32)
w_rate = tf.random_uniform(shape = [1], minval = 0.5, maxval = 1.0, dtype = tf.float32)
```
8. Added CROP_MUSTHAVE_CLASS_INDEX in hyperparams. If it is -1 nothing will change. But if you will set it to some class index, crop operation will return only crops where that class is presented (at least one pixel, but usually that's enough. In future I will implement also parameter to control pixels part of needed class).
9. Removed excess image summaries (only original + network output now, no branches output) for now. Will thinks about it more in future.
10. Added --evaluate-once to evaluation.
11. Fixed summaries writing, it was incorrect before.

#### 2017/11/23:
1. Forked by me. Initial changes was done (described above in differences section). Models now stored in repository.

#### 2017/11/15:
1. Support `training phase`, you can train on your own dataset. Please read the guide below.

#### 2017/11/13:
1. Add `bnnomerge model` which reparing for training phase. Choose different model using flag `--model=train, train_bn, trainval, trainval_bn` (Upload model in google drive).
2. Change `tf.nn.batch_normalization` to `tf.layers.batch_normalization`.

#### 2017/11/07:
`Support every image size larger than 128x256` by changing the avg pooling ksize and strides in the pyramid module. If input image size cannot divided by 32, it will be padded in to mutiple of 32.


## Install
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyadTdHejU1Q1lfRkU?usp=sharing) and put into `model` directory.

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

If you want to use it for training validation, use parameter --repeated-eval. In that case model will run endless and will evaluate one time per --eval-interval period if new checkpoint comes.
Also if you want to select best valid model, you need to setup --best-models-dir.

Example of usage:
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
**1. Change the `DATA_LIST_PATH`** train.py, make sure the list contains the absolute path of your data files, in `list.txt`:
```
/ABSOLUTE/PATH/TO/image /ABSOLUTE/PATH/TO/label
```
**2. Set Hyperparameters** 

In `hyperparams.py`:
```
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
```

If you want to use classes weights, you need to pass argument --use-class-weights and setup class weights value in train.py:
```
CLASS_WEIGHTS = [1.0, 1.0, 2.0]
```

**3.** Run following command and **decide whether to update mean/var or train beta/gamma variable**.
```
python train.py --update-mean-var --train-beta-gamma --random-mirror --use-class-weights --not-restore-last
```
After training the dataset, you can run following command to get the result:  
```
python inference.py --img-path=YOUR_OWN_IMAGE
```

Example of validation run:
```
python evaluate.py --snapshot-dir=snapshots --repeated-eval --best-models-dir ./best_models --eval-interval 120 --ignore-zero
```
Example of training process in Tensorboard summaries:
![TB](https://raw.githubusercontent.com/UndeadBlow/ICNet-tensorflow/master/tensorboard.png)


Also you can run model on video with utils.py script in dataset directory, but it is a little hardcoded for myself, so success is on your own.

## Export model for production

Check file export_model.sh for example of exporting model.

In that .sh file please change all paths and names to yours.

Then run it with bash.
```
bash export_model.sh
```

First script will convert your model to inference graph - without weights. So called unfrozen graph.
After that it will convert it to frozen graph, with weights.

Example of how to run that model after exporting see in inference.py:

```
def load_from_pb(shape, path):
    segment_graph = tf.Graph()
    with segment_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            
            x = tf.placeholder(dtype = tf.float32, shape = shape)
            img_tf = preprocess(x)
            img_tf, n_shape = check_input(img_tf)
            
            tf.import_graph_def(seg_graph_def, {'input:0': img_tf}, name = '')

            raw_output = segment_graph.get_tensor_by_name('conv6_cls/BiasAdd:0')
            output = tf.image.resize_bilinear(raw_output, tf.shape(img_tf)[1:3,])
            output = tf.argmax(output, dimension = 3)
            pred = tf.expand_dims(output, dim = 3)

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            config.allow_soft_placement = True
            config.log_device_placement = False

            sess = tf.Session(graph = segment_graph, config = config)

    return sess, pred, x
```

You can also run inference.py on that exported .pb model with flas --pb-file.
```
python3 inference.py --img-path ../1.jpg --pb-file ../frozen_icnet.pb
```

You can also try to quantize weights and transform graph to smaller size if you want. Check how to do that [here](https://www.tensorflow.org/performance/quantization).

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
