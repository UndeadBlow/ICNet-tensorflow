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

        #b_init = tf.random_uniform_initializer(minval = 0, maxval = 25)
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.leaky_relu, padding=pad, W_init=w_init, b_init=b_init, name='uconv1')
    return conv1

def deform_unet(x, is_train = False, reuse = False, n_out = 1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.leaky_relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.leaky_relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.leaky_relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.leaky_relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.leaky_relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.leaky_relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.leaky_relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.leaky_relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.leaky_relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.leaky_relu, name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.leaky_relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.leaky_relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.leaky_relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.leaky_relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.leaky_relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.leaky_relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.leaky_relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.leaky_relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
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

def psp_net(x, is_train = False, reuse = False, n_out = 1, pad='VALID'):
    
    w_init = tf.truncated_normal_initializer(stddev=0.03)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)

    def AddLayer(input1, input2, io_size, middle_size, pad_size, is_train = True, pad='VALID', name = '', arate = 2, conv_type = 'atrous'):
        
        el = ElementwiseLayer([input1, input2], tf.add, name = 'el' + name)
        relu = PReluLayer(el, name = 'relu0' + name)
        conv = Conv2d(relu, middle_size, (1, 1), act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        #bn = BatchNormLayer(conv, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'bn0' + name)
        print('bn', bn.count_params())
        if pad_size >  0:
            bn = ZeroPad2d(bn, pad_size, name = 'pad0' + name)
        if conv_type == 'atrous':
            conv1 = AtrousConv2dLayer(bn, middle_size, (3, 3), arate, act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('atrous', (conv1.count_params()))
        elif conv_type == 'plain':
            conv1 = Conv2d(bn, middle_size, (3, 3), (1, 1), act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('plain', (conv1.count_params()))
        elif conv_type == 'deform':
            offset1 = Conv2d(bn, 18, (3, 3), (1, 1), act=None, padding='SAME', name='offset1')
            conv1 = DeformableConv2d(bn, offset1, int(middle_size), (3, 3), act=None, W_init=w_init, b_init=None, name = 'conv1' + name)
            print('deform', (conv1.count_params() + offset1.count_params()), conv1.outputs.shape)

        bn = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'bn1' + name)
        conv = Conv2d(bn, io_size, (1, 1), act=None, padding=pad, W_init=w_init, name = 'conv3' + name)
        bn = BatchNormLayer(conv, act=None, is_train=is_train, gamma_init=gamma_init, name = 'bn2' + name)

        return relu, bn

    def SimpleLayer(input, input_size, middle_size, output_size, pad_size, is_train = True, pad='VALID', name = '', conv_type = 'atrous', arate = 2, ker_size = 1):
        
        conv = Conv2d(input, input_size, (1, 1), (ker_size, ker_size), act=None, padding=pad, W_init=w_init, name = 'conv0' + name)
        bn = BatchNormLayer(conv, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'bn0' + name)
        if pad_size != 0:
            bn = ZeroPad2d(bn, pad_size, name = 'pad0' + name)
        if conv_type == 'atrous':
            conv = AtrousConv2dLayer(bn, middle_size, (3, 3), arate, act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        elif conv_type == 'plain':
            conv = Conv2d(bn, middle_size, (3, 3), act=None, padding=pad, W_init=w_init, b_init=None, name = 'conv1' + name)
        elif conv_type == 'deform':
            offset1 = Conv2d(bn, 18, (3, 3), (1, 1), act=None, padding='SAME', name='offset1')
            conv = DeformableConv2d(bn, offset1, int(middle_size), (3, 3), act=None, W_init=w_init, b_init=None, name = 'conv1' + name)

        bn = BatchNormLayer(conv, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name = 'bn1' + name)
        conv = Conv2d(bn, output_size, (1, 1), act=None, padding=pad, W_init=w_init, name = 'conv2' + name)
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

        conv1 = Conv2d(inputs, 64 * channels_rate, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None, name='conv1_1')
        bn1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        bn1 = PReluLayer(bn1, name = 'layer1_relu')
        conv1 = Conv2d(bn1, 64 * channels_rate, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=None, name='conv1_3')
        bn1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn1_1')
        conv1 = Conv2d(bn1, 128 * channels_rate, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=None, name='conv1_4')
        bn1 = BatchNormLayer(conv1, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn1_2')
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
        conv20_0, bn20 = AddLayer(conv19_0, bn19, 1024 * channels_rate, 1024 * channels_rate, pad_size = 0, 
                                  name = 'layer20', arate = 4, is_train = is_train, conv_type = 'deform')

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
