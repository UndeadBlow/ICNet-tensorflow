import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'
layer_name = []
BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}
                
def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        layer_name.append(name)
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.is_training = is_training
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')

        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False, ignore_layers = []):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:

            # Pass unrestorable layers
            if len([f for f in ignore_layers if f in op_name]):
                continue

            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]

                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def get_layer_name(self):
        return layer_name
    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
    @layer
    def zero_padding(self, input, paddings, name):
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return tf.pad(input, paddings=pad_mat, name=name)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.selu(output, name=scope.name)
            return output

    def conv_next(self,
                k_h,
                k_w,
                c_o,
                s_h,
                s_w,
                name,
                relu=True,
                padding=DEFAULT_PADDING,
                group=1,
                biased=True):
        
        self.validate_padding(padding)
        input = self.terminals[0]
        
        c_i = input.get_shape()[-1]
        output = self.conv(1, 1, int(c_i) / 2, 1, 1, name = name + '_conv0', relu = False, biased = False, padding = padding)\
                .conv(1, 1, int(c_i) / 4, 1, 1, name = name + '_conv1', relu = False, biased = False, padding = padding)\
                .conv(3, 1, int(c_i) / 2, s_h, 1, name = name + '_conv2', relu = True, biased = False, padding = padding)\
                .conv(1, 3, int(c_i) / 2, 1, s_w, name = name + '_conv3', relu = True, biased = False, padding = padding)\
                .conv(1, 1, int(c_i) / 2, 1, 1, name = name + '_conv4', relu = True, biased = False, padding = padding)
        

        return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.selu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.selu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)

        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.selu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        """
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.selu(output)
            return output
        """
        output = tf.layers.batch_normalization(
                    input,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=self.is_training,
                    name=name
                )

        if relu:
            output = tf.nn.selu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)

    @layer
    def interp(self, input, factor = -1, name = '', size = [], force_resize = False):
        if factor < 1.0 and factor > 0.0 or force_resize:
            ori_h, ori_w = input.get_shape().as_list()[1:3]
            if factor > 0:
                resize_shape = [(int)(ori_h * factor), (int)(ori_w * factor)]
            else:
                resize_shape = [int(size[0]), int(size[1])]
            return tf.image.resize_bilinear(input, size = resize_shape, align_corners = True, name = name)

        else:

            if factor < 0.0:
                ori_h, ori_w = input.get_shape().as_list()[1:3]
                batch = input.get_shape().as_list()[0]
                channel = input.get_shape().as_list()[3]
                res_h = int(size[0])
                res_w = int(size[1])
                factor = int(res_h / ori_h)
                pad = 'VALID'
            else:
                ori_h, ori_w = input.get_shape().as_list()[1:3]
                batch = input.get_shape().as_list()[0]
                channel = input.get_shape().as_list()[3]
                res_h = (int)(ori_h * factor)
                res_w = (int)(ori_w * factor)
                pad = 'SAME'
                
            with tf.variable_scope(name) as scope:
                kernel = self.make_var('weights', shape = [3, 3, channel, channel])
                #print('deconv else', batch, res_h, res_w, channel, size, factor)
                output = tf.nn.selu(input, name = scope.name)
                output = tf.nn.conv2d_transpose(output, kernel, 
                    [batch, res_h, res_w, channel], [1, int(factor), int(factor), 1], padding = pad, name = scope.name)

                return output

        #return tf.image.resize_bilinear(input, size=resize_shape, align_corners=True, name=name)

