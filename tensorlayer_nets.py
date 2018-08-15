import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from hyperparams import *

def u_net(x, is_train = False, reuse = False, n_out = 1, pad='SAME'):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init=tf.random_normal_initializer(1., 0.02)


        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        conv1 = Conv2d(inputs, 64, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        pool1 = BatchNormLayer(pool1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn1')

        conv2 = Conv2d(pool1, 128, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        pool2 = BatchNormLayer(pool2, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(pool2, 256, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        pool3 = BatchNormLayer(pool3, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(pool3, 512, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        pool4 = BatchNormLayer(pool4, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(pool4, 1024, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5_2')
        conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn4_1')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn5')

        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn6')

        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn7')

        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv1_2')

        b_init = tf.random_uniform_initializer(minval = 0, maxval = n_out)
        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='uconv1')
        
    return conv1

def light_deform_u_net(x, is_train = False, reuse = False, n_out = 1, pad='SAME', filter_size_scale = 1.0):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev = 0.03)
        b_init = tf.constant_initializer(value = 0.01)

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        conv1 = Conv2d(inputs, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv1_1', W_init=w_init)
        conv1 = Conv2d(conv1, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv1_2', W_init=w_init)
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv2_1', W_init=w_init)
        conv2 = Conv2d(conv2, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv2_2', W_init=w_init)
        conv2 = ElementwiseLayer([pool1, conv2], tf.add, name = 'add1')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv3_1', W_init=w_init)
        conv3 = Conv2d(conv3, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv3_2', W_init=w_init)
        conv3 = ElementwiseLayer([pool2, conv3], tf.add, name = 'add2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv4_1', W_init=w_init)
        conv4 = Conv2d(conv4, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='conv4_2', W_init=w_init)
        conv4 = ElementwiseLayer([pool3, conv4], tf.add, name = 'add3')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv5_1', W_init=w_init)
        #conv5 = Conv2d(conv5, 32 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv5_2', W_init=w_init)
        #conv5 = ElementwiseLayer([pool4, conv5], tf.add, name = 'add4')
        offset1 = Conv2d(conv5, 18, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='offset2')
        conv5 = DeformableConv2d(conv5, offset1, int(128 * filter_size_scale), (3, 3), act=tf.nn.leaky_relu, W_init=w_init, b_init=None, name = 'deform_conv2')

        up4 = DeConv2d(conv5, int(32 * filter_size_scale), (3, 3), (nx/8, ny/8), (2, 2), name='deconv4', W_init=w_init)
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv4_1', W_init=w_init)
        conv4 = Conv2d(conv4, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv4_2', W_init=w_init)
        conv4 = ElementwiseLayer([up4, conv4], tf.add, name = 'add5')
        up3 = DeConv2d(conv4, int(32 * filter_size_scale), (3, 3), (nx/4, ny/4), (2, 2), name='deconv3', W_init=w_init)
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv3_1', W_init=w_init)
        conv3 = Conv2d(conv3, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv3_2', W_init=w_init)
        conv3 = ElementwiseLayer([up3, conv3], tf.add, name = 'add6')
        up2 = DeConv2d(conv3, int(32 * filter_size_scale), (3, 3), (nx/2, ny/2), (2, 2), name='deconv2', W_init=w_init)
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu,  name='uconv2_1', W_init=w_init)
        conv2 = Conv2d(conv2, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv2_2', W_init=w_init)
        conv2 = ElementwiseLayer([up2, conv2], tf.add, name = 'add7')
        up1 = DeConv2d(conv2, int(32 * filter_size_scale), (3, 3), (nx/1, ny/1), (2, 2), name='deconv1', W_init=w_init)
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv1_1', W_init=w_init)
        conv1 = Conv2d(conv1, 64 * filter_size_scale, (3, 3), act=tf.nn.leaky_relu, name='uconv1_2', W_init=w_init)
        conv1 = ElementwiseLayer([up1, conv1], tf.add, name = 'add8')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1')
        
    return conv1

def deform_u_net(x, is_train = False, reuse = False, n_out = 1, pad='SAME'):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.leaky_relu, name='conv1_1', W_init=w_init)
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.leaky_relu, name='conv1_2', W_init=w_init)
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.leaky_relu, name='conv2_1', W_init=w_init)
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.leaky_relu, name='conv2_2', W_init=w_init)
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.leaky_relu, name='conv3_1', W_init=w_init)
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.leaky_relu, name='conv3_2', W_init=w_init)
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.leaky_relu, name='conv4_1', W_init=w_init)
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.leaky_relu, name='conv4_2', W_init=w_init)
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        offset1 = Conv2d(pool4, 18, (3, 3), (1, 1), act=None, padding='SAME', name='offset1')
        conv5 = DeformableConv2d(pool4, offset1, 1024, (3, 3), act=tf.nn.leaky_relu, W_init=w_init, b_init=None, name = 'deform_conv1')
        offset1 = Conv2d(conv5, 18, (3, 3), (1, 1), act=None, padding='SAME', name='offset2')
        conv5 = DeformableConv2d(conv5, offset1, 1024, (3, 3), act=tf.nn.leaky_relu, W_init=w_init, b_init=None, name = 'deform_conv2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4', W_init=w_init)
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.leaky_relu, name='uconv4_1', W_init=w_init)
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.leaky_relu, name='uconv4_2', W_init=w_init)
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3', W_init=w_init)
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.leaky_relu, name='uconv3_1', W_init=w_init)
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.leaky_relu, name='uconv3_2', W_init=w_init)
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2', W_init=w_init)
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.leaky_relu,  name='uconv2_1', W_init=w_init)
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.leaky_relu, name='uconv2_2', W_init=w_init)
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1', W_init=w_init)
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.leaky_relu, name='uconv1_1', W_init=w_init)
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.leaky_relu, name='uconv1_2', W_init=w_init)

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1')
        
    return conv1

# lambda x : tf.nn.leaky_relu(x, alpha = 5.5)
# check what is faster, 3x1 - 1x3 or 3x3?

# 0.5 3x1 - 1x3 plain:   3 102 145, av. time on 1000 runs: 46 ms
# 0.75 3x1 - 1x3 plain:  6 971 417, av. time on 1000 runs: 77 ms
# 0.75 v2 3x1 - 1x3 plain:  5 797 433, av. time on 1000 runs: 72 ms
# 0.5 partly 3x3 plain: 19 610 817, av. time on 1000 runs: 96 ms ????
# 1.0 3x1 - 1x3 plain:  12 386 161, av. time on 1000 runs: 101 ms
# 0.5 3x3 plain:         3 108 865, av. time on 1000 runs: 47 ms
# 0.5 3x1 - 1x3 deform:  4 870 209, av. time on 1000 runs: 296 ms

# 0.5 v3 3x1 - 1x3 deform:  2 610 105, 720: 119 ms, 560: 75 ms

def unextv3(x, is_train = False, reuse = False, n_out = 1, pad='SAME', activation = tf.nn.leaky_relu, depth = 0.5):

    def ResNextBlock(input, n_inputs, name, add = True, reduced = False):

        # The idea from mobilenet that bottleneck layer should be linear
        conv0 = Conv2d(input, n_inputs / 2, (1, 1), act = None, name = name + 'conv1_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv0, n_inputs / 4, (1, 1), act = None, name = name + 'conv1_2', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_inputs / 2, (3, 1), act = activation, name = name + 'conv1_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, n_inputs / 2, (1, 3), act = activation, name = name + 'conv1_4', W_init = w_init, padding = 'SAME')

        if reduced:
            conv1 = ElementwiseLayer([conv0, conv1], tf.add, name = name + 'add1_1')
            return conv1
        else:
            conv1 = Conv2d(conv1, n_inputs, (1, 1), act = activation, name = name + 'conv1_5', W_init = w_init, padding = 'SAME')
            if add:
                conv1 = ElementwiseLayer([conv1, input], tf.add, name = name + 'add1_1')
            else:
                conv1 = ConcatLayer([conv1, input], name = name + 'concat1_1')

        return conv1

    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("unext", reuse = reuse):

        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        b_init = tf.constant_initializer(value = 0.0)

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        conv1 = Conv2d(inputs, 64 * depth, (3, 1), act = activation, name = 'conv0_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv0_2', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (3, 1), act = activation, name = 'conv0_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv0_4', W_init = w_init, padding = 'SAME')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = ResNextBlock(pool1, 64 * depth, 'block3', add = False)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block4')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = ResNextBlock(pool2, 128 * depth, 'block5', add = False)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block6')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = ResNextBlock(pool3, 256 * depth, 'block7', add = False)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block8')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = ResNextBlock(pool4, 512 * depth, 'block9', add = False)
        conv5 = ResNextBlock(conv5, 1024 * depth, 'block10')
        print(conv5.outputs.shape)

        up4 = DeConv2d(conv5, int(512 * depth), (3, 1),(nx / 8, ny / 8), (2, 1), act = None, name='deconv4_1', W_init=w_init)
        up4 = DeConv2d(up4, int(512 * depth), (1, 3), (nx / 8, ny / 8), (1, 2), act = None, name='deconv4_2', W_init=w_init)
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = ResNextBlock(up4, 1024 * depth, 'block11', reduced = True)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block12')
        up3 = DeConv2d(conv4, int(256 * depth),(3, 1), (nx / 4, ny / 4), (2, 1), act = None, name='deconv3_1', W_init=w_init)
        up3 = DeConv2d(up3, int(256 * depth), (1, 3), (nx / 4, ny / 4), (1, 2),  act = None, name='deconv3_2', W_init=w_init)
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = ResNextBlock(up3, 512 * depth, 'block13', reduced = True)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block14')
        up2 = DeConv2d(conv3, int(128 * depth), (3, 1), (nx / 2, ny / 2), (2, 1), act = None, name='deconv2_1', W_init=w_init)
        up2 = DeConv2d(up2, int(128 * depth), (1, 3), (nx / 2, ny / 2), (1, 2), act = None, name='deconv2_2', W_init=w_init)
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = ResNextBlock(up2, 256 * depth, 'block15', reduced = True)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block16')
        up1 = DeConv2d(conv2, int(64 * depth), (3, 1), (nx / 1, ny / 1), (2, 1), act = None, name='deconv1_1', W_init=w_init)
        up1 = DeConv2d(up1, int(64 * depth), (1, 3), (nx / 1, ny / 1), (1, 2), act = None, name='deconv1_2', W_init=w_init)
        up1 = ConcatLayer([up1, conv1], 3, name='concat1')
        conv1 = Conv2d(up1, 128 * depth, (3, 1), act = activation, name = 'conv17_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 128 * depth, (1, 3), act = activation, name = 'conv17_2', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (3, 1), act = activation, name = 'conv17_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv17_4', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1')

        #print(conv1.outputs.shape)
        print('Params :', conv1.count_params())
        # quit()

    return conv1
    
def lunextv3(x, is_train = False, reuse = False, n_out = 1, pad='SAME', activation = tf.nn.selu, depth = 0.5):
    
    def ResNextBlock(input, n_inputs, name, add = True, reduced = False):

        # The idea from mobilenet that bottleneck layer should be linear
        conv0 = Conv2d(input, n_inputs / 2, (1, 1), act = None, name = name + 'conv1_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv0, n_inputs / 4, (1, 1), act = None, name = name + 'conv1_2', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_inputs / 2, (3, 1), act = activation, name = name + 'conv1_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, n_inputs / 2, (1, 3), act = activation, name = name + 'conv1_4', W_init = w_init, padding = 'SAME')

        if reduced:
            conv1 = ElementwiseLayer([conv0, conv1], tf.add, name = name + 'add1_1')
            return conv1
        else:
            conv1 = Conv2d(conv1, n_inputs, (1, 1), act = activation, name = name + 'conv1_5', W_init = w_init, padding = 'SAME')
            if add:
                conv1 = ElementwiseLayer([conv1, input], tf.add, name = name + 'add1_1')
            else:
                conv1 = ConcatLayer([conv1, input], name = name + 'concat1_1')

        return conv1

    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("unext", reuse = reuse):

        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        b_init = tf.constant_initializer(value = 0.0)

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        conv1 = Conv2d(inputs, 64 * depth, (3, 3), (1, 1), act = activation, name = 'conv0_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (3, 3), (1, 1), act = activation, name = 'conv0_1', W_init = w_init, padding = 'SAME')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = ResNextBlock(pool1, 64 * depth, 'block3', add = False)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block4')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = ResNextBlock(pool2, 128 * depth, 'block5', add = False)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block6')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = ResNextBlock(pool3, 256 * depth, 'block7', add = False)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block8')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = ResNextBlock(pool4, 512 * depth, 'block9')
        conv5 = ResNextBlock(conv5, 512 * depth, 'block10')
        print(conv5.outputs.shape)


        up4 = DeConv2d(conv5, int(512 * depth), (3, 3),(nx / 8, ny / 8), (2, 2), act = None, name='deconv4_1', W_init=w_init)
        #up4 = DeConv2d(up4, int(512 * depth), (1, 3), (nx / 8, ny / 8), (1, 2), act = None, name='deconv4_2', W_init=w_init)
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = ResNextBlock(up4, 1024 * depth, 'block11', reduced = True)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block12')
        up3 = DeConv2d(conv4, int(256 * depth),(3, 3), (nx / 4, ny / 4), (2, 2), act = None, name='deconv3_1', W_init=w_init)
        #up3 = DeConv2d(up3, int(256 * depth), (1, 3), (nx / 4, ny / 4), (1, 2),  act = None, name='deconv3_2', W_init=w_init)
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = ResNextBlock(up3, 512 * depth, 'block13', reduced = True)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block14')
        up2 = DeConv2d(conv3, int(128 * depth), (3, 3), (nx / 2, ny / 2), (2, 2), act = None, name='deconv2_1', W_init=w_init)
        #up2 = DeConv2d(up2, int(128 * depth), (1, 3), (nx / 2, ny / 2), (1, 2), act = None, name='deconv2_2', W_init=w_init)
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = ResNextBlock(up2, 256 * depth, 'block15', reduced = True)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block16')
        up1 = DeConv2d(conv2, int(64 * depth), (3, 3), (nx / 1, ny / 1), (2, 2), act = None, name='deconv1_1', W_init=w_init)
        #up1 = DeConv2d(up1, int(64 * depth), (1, 3), (nx / 1, ny / 1), (1, 2), act = None, name='deconv1_2', W_init=w_init)
        up1 = ConcatLayer([up1, conv1], 3, name='concat1')
        conv1 = Conv2d(up1, 64 * depth, (3, 1), act = activation, name = 'conv17_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv17_2', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (3, 1), act = activation, name = 'conv17_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv17_4', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1')

        print('Params :', conv1.count_params())

        return conv1

def unext(x, is_train = False, reuse = False, n_out = 1, pad='SAME', activation = tf.nn.relu, depth = 0.5):
    # leaky_rely speed: 57.4 ms
    # relu speed:       48.9 ms
    # selu speed:       50.0 ms
    def ResNextBlock(input, n_inputs, name, add = True, reduced = False):

        # The idea from mobilenet that bottleneck layer should be linear
        conv0 = Conv2d(input, n_inputs / 2, (1, 1), act = None, name = name + 'conv1_1', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv0, n_inputs / 4, (1, 1), act = None, name = name + 'conv1_2', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_inputs / 2, (3, 1), act = activation, name = name + 'conv1_3', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, n_inputs / 2, (1, 3), act = activation, name = name + 'conv1_4', W_init = w_init, padding = 'SAME')

        if reduced:
            conv1 = ElementwiseLayer([conv0, conv1], tf.add, name = name + 'add1_1')
            return conv1
        else:
            conv1 = Conv2d(conv1, n_inputs, (1, 1), act = activation, name = name + 'conv1_5', W_init = w_init, padding = 'SAME')
            if add:
                conv1 = ElementwiseLayer([conv1, input], tf.add, name = name + 'add1_1')
            else:
                conv1 = ConcatLayer([conv1, input], name = name + 'concat1_1')

        return conv1

    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("unext", reuse = reuse):

        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        b_init = tf.constant_initializer(value = 0.0)

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')

        # speed with 1x3 3x1 first layer 67.4 ms
        # speed with 3x3 first layer 65.8 ms
        # cudnn speed up?
        # check the same on cpu to exclude cudnn effect:
        # 3x1 1x3: 853 ms (1964217)
        # 3x3:     859 ms <- but here are less parameters (1961689)

        # speed with 3x1 1x3 deconvs - 67.4 (1964217)
        # with 3x3 - 67.0 (2353881)
        # same on cpu:
        # 3x1 1x3: 853 ms
        # 3x3:     916 ms 
        conv1 = Conv2d(inputs, 64 * depth, (3, 3), (1, 1), act = activation, name = 'conv0_1', W_init = w_init, padding = 'SAME')
        # +1 layers here 
        # without 1955417 params and 54 ms
        # with 1955417 params and 57.4 ms
        conv1 = Conv2d(conv1, 64 * depth, (3, 3), (1, 1), act = activation, name = 'conv0_1', W_init = w_init, padding = 'SAME')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = ResNextBlock(pool1, 64 * depth, 'block3', add = False)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block4')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = ResNextBlock(pool2, 128 * depth, 'block5', add = False)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block6')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = ResNextBlock(pool3, 256 * depth, 'block7', add = False)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block8')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = ResNextBlock(pool4, 512 * depth, 'block9', add = False)
        conv5 = ResNextBlock(conv5, 1024 * depth, 'block10')
        print(conv5.outputs.shape)


        up4 = DeConv2d(conv5, int(512 * depth), (3, 1),(nx / 8, ny / 8), (2, 1), act = None, name='deconv4_1', W_init=w_init)
        up4 = DeConv2d(up4, int(512 * depth), (1, 3), (nx / 8, ny / 8), (1, 2), act = None, name='deconv4_2', W_init=w_init)
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = ResNextBlock(up4, 1024 * depth, 'block11', reduced = True)
        conv4 = ResNextBlock(conv4, 512 * depth, 'block12')
        up3 = DeConv2d(conv4, int(256 * depth),(3, 1), (nx / 4, ny / 4), (2, 1), act = None, name='deconv3_1', W_init=w_init)
        up3 = DeConv2d(up3, int(256 * depth), (1, 3), (nx / 4, ny / 4), (1, 2),  act = None, name='deconv3_2', W_init=w_init)
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = ResNextBlock(up3, 512 * depth, 'block13', reduced = True)
        conv3 = ResNextBlock(conv3, 256 * depth, 'block14')
        up2 = DeConv2d(conv3, int(128 * depth), (3, 1), (nx / 2, ny / 2), (2, 1), act = None, name='deconv2_1', W_init=w_init)
        up2 = DeConv2d(up2, int(128 * depth), (1, 3), (nx / 2, ny / 2), (1, 2), act = None, name='deconv2_2', W_init=w_init)
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = ResNextBlock(up2, 256 * depth, 'block15', reduced = True)
        conv2 = ResNextBlock(conv2, 128 * depth, 'block16')
        up1 = DeConv2d(conv2, int(64 * depth), (3, 1), (nx / 1, ny / 1), (2, 1), act = None, name='deconv1_1', W_init=w_init)
        up1 = DeConv2d(up1, int(64 * depth), (1, 3), (nx / 1, ny / 1), (1, 2), act = None, name='deconv1_2', W_init=w_init)
        up1 = ConcatLayer([up1, conv1], 3, name='concat1')
        # 3x1 1x3 here last layers -> 1943193 params
        # 3x3 -> 1955417 params
        conv1 = Conv2d(up1, 64 * depth, (3, 3), act = activation, name = 'conv17_1', W_init = w_init, padding = 'SAME')
        #conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv17_2', W_init = w_init, padding = 'SAME')
        conv1 = Conv2d(conv1, 64 * depth, (3, 3), act = activation, name = 'conv17_3', W_init = w_init, padding = 'SAME')
        #conv1 = Conv2d(conv1, 64 * depth, (1, 3), act = activation, name = 'conv17_4', W_init = w_init, padding = 'SAME')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1')

        #print(conv1.outputs.shape)
        print('Params :', conv1.count_params())
        # quit()

    return conv1

def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out

def psp_net(x, is_training = False, reuse = False, num_classes = 1, pad='VALID'):
    
    is_train = is_training
    n_out = num_classes
    
    w_init = tf.truncated_normal_initializer(stddev=0.03)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    def AddLayer(input1, input2, io_size, middle_size, pad_size, is_train = True, pad='VALID', name = '', arate = 2, conv_type = 'atrous'):
        
        el = ElementwiseLayer([input1, input2], tf.add, name = 'el' + name)
        relu = PReluLayer(el, name = 'relu0' + name)
        conv = Conv2d(relu, middle_size, (1, 1), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)

        if pad_size >  0:
            conv = ZeroPad2d(conv, pad_size, name = 'pad0' + name)
        if conv_type == 'atrous':
            conv1 = AtrousConv2dLayer(conv, middle_size, (3, 3), arate, act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('atrous', (conv1.count_params()))
        elif conv_type == 'plain':
            conv1 = Conv2d(conv, middle_size, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('plain', (conv1.count_params()))
        elif conv_type == 'deform':
            offset1 = Conv2d(conv, 18, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='offset1')
            conv1 = DeformableConv2d(conv, offset1, int(middle_size), (3, 3), act=tf.nn.leaky_relu, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('deform', (conv1.count_params() + offset1.count_params()), conv1.outputs.shape)

        conv = Conv2d(conv1, io_size, (1, 1), act=None, padding=pad, W_init=w_init, name = 'conv3' + name)
        bn = BatchNormLayer(conv, act=None, is_train=is_train, gamma_init=gamma_init, name = 'bn2' + name)

        return relu, bn

    def SimpleLayer(input, input_size, middle_size, output_size, pad_size, is_train = True, pad='VALID', name = '', conv_type = 'atrous', arate = 2, ker_size = 1):
        
        bn = Conv2d(input, input_size, (1, 1), (ker_size, ker_size), padding=pad, W_init=w_init, name = 'conv0' + name)

        if pad_size != 0:
            bn = ZeroPad2d(bn, pad_size, name = 'pad0' + name)
        if conv_type == 'atrous':
            conv = AtrousConv2dLayer(bn, middle_size, (3, 3), arate, act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        elif conv_type == 'plain':
            conv = Conv2d(bn, middle_size, (3, 3), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        elif conv_type == 'deform':
            offset1 = Conv2d(bn, 18, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', name='offset1')
            conv = DeformableConv2d(bn, offset1, int(middle_size), (3, 3), act=None, W_init=w_init, b_init=None, name = 'conv1' + name)

        conv = Conv2d(conv, output_size, (1, 1), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, name = 'conv2' + name)
        bn = BatchNormLayer(conv, act=None, is_train=is_train, gamma_init=gamma_init, name = 'bn2' + name)

        return bn

    def SmallLayer(input1, input2, input_size, out_size, is_train = True, pad='VALID', name = '', ker_size = 1):
        
        el = ElementwiseLayer([input1, input2], tf.add, name = 'el' + name)
        relu = PReluLayer(el, name = 'relu0' + name)
        conv = Conv2d(relu, out_size, (1, 1), (ker_size, ker_size), act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        bn = BatchNormLayer(conv, act=None, is_train=is_train, gamma_init=gamma_init, name = 'bn1' + name)

        return relu, bn

    def BilinearLayer(input, pool_size, shape, is_train = True, pad='VALID', name = '', channels = 512):
        
        pool = MeanPool2d(input, (pool_size, pool_size), (pool_size, pool_size), name = 'pool0' + name)
        conv = Conv2d(pool, channels, (1, 1), act=None, padding=pad, W_init=w_init, name = 'conv0' + name)
        bn = BatchNormLayer(conv, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'bn0' + name)
        up = UpSampling2dLayer(bn, shape, name = 'up0' + name, is_scale = False)

        return up

    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):

        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name = 'inputs')
        channels_rate = 0.5

        conv1 = Conv2d(inputs, 64 * channels_rate, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='conv1_1')
        bn1 = PReluLayer(conv1, name = 'layer1_relu')
        conv1 = Conv2d(bn1, 64 * channels_rate, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=None, name='conv1_3')
        bn1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn1_1')
        conv1 = Conv2d(bn1, 128 * channels_rate, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='conv1_4')
        pool1 = MaxPool2d(conv1, (3, 3), (2, 2), padding='SAME', name='pool2')
        conv1 = Conv2d(pool1, 256 * channels_rate, (1, 1), act=None, padding=pad, W_init=w_init, b_init=None, name='conv1_5')
        bn1 = BatchNormLayer(conv1, act=None, is_train=is_train, gamma_init=gamma_init, name='bn1_3')

        bn2 = SimpleLayer(pool1, 64 * channels_rate, 64 * channels_rate, 256 * channels_rate, pad_size = 1, name = 'layer2', conv_type = 'plain', is_train = is_train)

        conv3_0, bn3 = AddLayer(bn1, bn2, 256 * channels_rate, 64 * channels_rate, pad_size = 1, name = 'layer3', conv_type = 'plain', is_train = is_train)
        conv4_0, bn4 = AddLayer(conv3_0, bn3, 256 * channels_rate, 64 * channels_rate, pad_size = 1, name = 'layer4', conv_type = 'plain', is_train = is_train)

        rel5, bn5 = SmallLayer(conv4_0, bn4, 256 * channels_rate, 512 * channels_rate, name = 'layer5', ker_size = 1, is_train = is_train)
        bn6 = SimpleLayer(rel5, 128 * channels_rate, 128 * channels_rate, 512 * channels_rate, pad_size = 1, name = 'layer6', ker_size = 1, conv_type = 'plain', is_train = is_train)

        conv7_0, bn7 = AddLayer(bn5, bn6, 512 * channels_rate, 128 * channels_rate, pad_size = 1, name = 'layer7', conv_type = 'plain', is_train = is_train)
        conv8_0, bn8 = AddLayer(conv7_0, bn7, 512 * channels_rate, 128 * channels_rate, pad_size = 1, name = 'layer8', conv_type = 'plain', is_train = is_train)
        conv9_0, bn9 = AddLayer(conv8_0, bn8, 512 * channels_rate, 128 * channels_rate, pad_size = 1, name = 'layer9', conv_type = 'plain', is_train = is_train)

        rel10, bn10 = SmallLayer(conv9_0, bn9, 512 * channels_rate, 1024 * channels_rate, name = 'layer10', ker_size = 1, is_train = is_train)
        bn11 = SimpleLayer(rel10, 256 * channels_rate, 256 * channels_rate, 1024 * channels_rate, pad_size = 2, name = 'layer11', arate = 2, is_train = is_train)

        conv12_0, bn12 = AddLayer(bn11, bn10, 1024 * channels_rate, 256 * channels_rate, pad_size = 2, name = 'layer12', arate = 2, is_train = is_train)
        conv13_0, bn13 = AddLayer(conv12_0, bn12, 1024 * channels_rate, 256 * channels_rate, pad_size = 2, name = 'layer13', arate = 2, is_train = is_train)
        conv14_0, bn14 = AddLayer(conv13_0, bn13, 1024 * channels_rate, 256 * channels_rate, pad_size = 2, name = 'layer14', arate = 2, is_train = is_train)
        conv15_0, bn15 = AddLayer(conv14_0, bn14, 1024 * channels_rate, 256 * channels_rate, pad_size = 2, name = 'layer15', arate = 2, is_train = is_train)
        conv16_0, bn16 = AddLayer(conv15_0, bn15, 1024 * channels_rate, 256 * channels_rate, pad_size = 2, name = 'layer16', arate = 2, is_train = is_train)

        rel17, bn17 = SmallLayer(conv16_0, bn16, 1024 * channels_rate, 1024 * channels_rate, name = 'layer17', is_train = is_train)
        bn18 = SimpleLayer(rel17, 256 * channels_rate, 256 * channels_rate, 1024 * channels_rate, pad_size = 4, name = 'layer18', arate = 4, is_train = is_train)

        conv19_0, bn19 = AddLayer(bn17, bn18, 1024 * channels_rate, 1024 * channels_rate, pad_size = 4, name = 'layer19', arate = 4, is_train = is_train)
        conv20_0, bn20 = AddLayer(conv19_0, bn19, 1024 * channels_rate, 1024 * channels_rate, pad_size = 4, 
                                  name = 'layer20', arate = 4, is_train = is_train, conv_type = 'atrous')

        el21 = ElementwiseLayer([conv20_0, bn20], tf.add, name = 'layer21')
        pool12 = MaxPool2d(bn2, (3, 3), (1, 1), padding='SAME', name='pool12')
        el21_1 = ConcatLayer([pool12, el21], name = 'layer21_1')
        el21_1 = Conv2d(el21_1, 1024 * channels_rate, (1, 1), act=None, padding=pad, W_init=w_init, b_init=None, name='conv21_1')
        conv_21 = PReluLayer(el21_1, name = 'layer21_relu')

        shape = tuple([int(t) for t in conv_21.outputs.shape])
        shape = shape[1:3]

        bi22 = BilinearLayer(conv_21, 60, shape, channels = 512 * channels_rate, name = 'layer23', is_train = is_train)
        bi23 = BilinearLayer(conv_21, 30, shape, channels = 512 * channels_rate, name = 'layer24', is_train = is_train)
        bi24 = BilinearLayer(conv_21, 20, shape, channels = 512 * channels_rate, name = 'layer25', is_train = is_train)
        bi25 = BilinearLayer(conv_21, 10, shape, channels = 512 * channels_rate, name = 'layer26', is_train = is_train)

        concat26 = ConcatLayer([conv_21, bi22, bi23, bi24, bi25], concat_dim = -1, name = 'layer27')
        conv27 = Conv2d(concat26, 512 * channels_rate, (3, 3), act=None, padding='SAME', W_init=w_init, name = 'layer28')
        bn28 = BatchNormLayer(conv27, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'layer29')
        conv29 = Conv2d(bn28, n_out, (1, 1), act=None, padding=pad, b_init=tf.random_uniform_initializer(minval=0, maxval=n_out), W_init=w_init, name = 'layer30')
        print('output shape: ', conv29.outputs.shape)


    return conv29

if __name__ == "__main__":
    pass
