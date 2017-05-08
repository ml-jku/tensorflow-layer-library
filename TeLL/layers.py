# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017
Different classes and utility functions for stack-able network layers

See architectures/sample_architectures.py for some usage examples

"""

# ------------------------------------------------------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------------------------------------------------------
from collections import OrderedDict
from itertools import zip_longest
import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------------------------------------------------------
#  Functions
# ------------------------------------------------------------------------------------------------------------------
def tof(i, shape):
    """Check whether i is tensor or initialization function; return tensor or initialized tensor;
    
    Parameters
    -------
    i : tensor or function
        Tensor or function to initialize tensor
    shape : list or tuple
        Shape of tensor to initialize
    
    Returns
    -------
    : tensor
        Tensor or initialized tensor
    """
    if callable(i):
        return i(shape)
    else:
        return i


def tofov(i, shape=None, var_params=None):
    """Check whether i is tensor or initialization function or tf.Variable; return tf.Variable;
    
    Parameters
    -------
    i : tensor or function or tf.Variable
        Tensor or function to initialize tensor
    shape : list or tuple or None
        Shape of tensor to initialize
    var_params : dict or None
        Dictionary with additional parameters for tf.Variable, e.g. dict(trainable=True); Defaults to empty dict;
    
    Returns
    -------
    : tf.Variable
        Tensor or initialized tensor or tf.Variable as tf.Variable
    """
    
    if isinstance(i, tf.Variable):
        # i is already a tf.Variable -> nothing to do
        return i
    else:
        # i is either a tensor or initializer -> turn it into a tensor with tof()
        i = tof(i, shape)
        # and then turn it into a tf.Variable
        if var_params is None:
            var_params = dict()
        return tf.Variable(i, **var_params)


def dot_product(tensor_nd, tensor_2d):
    """Broadcastable version of tensorflow dot product between tensor_nd ad tensor_2d
    
    Parameters
    -------
    tensor_nd : tensor
        Tensor with 1, 2 or more dimensions; Dot product will be performed on last dimension and broadcasted over other
        dimensions
    tensor_2d : tensor
        Tensor with 1 or 2 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for dot product result
    """
    # Get shape and replace unknown shapes (None) with -1
    shape_nd = tensor_nd.get_shape().as_list()
    shape_nd = [s if isinstance(s, int) else -1 for s in shape_nd]
    shape_2d = tensor_2d.get_shape().as_list()
    if len(shape_2d) > 2:
        raise ValueError("tensor_2d must be a 1D or 2D tensor")
    
    if len(shape_2d) == 1:
        tensor_2d = tf.expand_dims(tensor_2d, 0)
    if len(shape_nd) == 1:
        shape_nd = tf.expand_dims(shape_nd, 0)
    
    if len(shape_nd) > 2:
        # collapse axes except for ones to multiply and perform matmul
        dot_prod = tf.matmul(tf.reshape(tensor_nd, [-1, shape_nd[-1]]), tensor_2d)
        # reshape to correct dimensions
        dot_prod = tf.reshape(dot_prod, shape_nd[:-1] + shape_2d[-1:])
    elif len(shape_nd) == 2:
        dot_prod = tf.matmul(tensor_nd, tensor_2d)
    else:
        dot_prod = tf.matmul(tf.expand_dims(tensor_nd, 0), tensor_2d)
    
    return dot_prod


def conv2d(x, W, strides=(1, 1, 1, 1), padding='SAME', dilation_rate=(1, 1), name='conv2d'):
    """Broadcastable version of tensorflow 2D convolution with weight mask, striding, zero-padding, and dilation
    
    For dilation the tf.nn.convolution function is used. Otherwise the computation will default to the (cudnn-
    supported) tf.nn.conv2d function.
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    W : tensor
        Kernel to perform convolution with; Shape: [x_dim, y_dim, input_features, output_features]
    dilation_rate : tuple of int
        Defaults to (1, 1) (i.e. normal 2D convolution). Use list of integers to specify multiple dilation rates;
        only for spatial dimensions -> len(dilation_rate) must be 2;
    
    Returns
    -------
    : tensor
        Tensor for convolution result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    if dilation_rate == (1, 1):
        def conv_fct(inp):
            return tf.nn.conv2d(input=inp, filter=W, padding=padding, strides=strides, name=name)
    else:
        if (strides[0] != 1) or (strides[-1] != 1):
            raise AttributeError("Striding in combination with dilation is only possible along the spatial dimensions,"
                                 "i.e. strides[0] and strides[-1] have to be 1.")
        
        def conv_fct(inp):
            return tf.nn.convolution(input=inp, filter=W, dilation_rate=dilation_rate,
                                     padding=padding, strides=strides[1:3], name=name)
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    with tf.variable_scope(name):
        if len(x_shape) > 4:
            if x_shape[0] == -1:
                x_flat = tf.reshape(x, [-1] + x_shape[2:])
            else:
                x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
            conv = conv_fct(x_flat)
            conv = tf.reshape(conv, x_shape[:2] + conv.get_shape().as_list()[1:])
        else:
            conv = conv_fct(x)
    return conv


def conv2d_transpose2d(x, W, output_shape, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC'):
    """Deconvolution/upscaling/transposed convolution based on tf.nn.conv2d_transpose but with broadcasting
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for transposed convolution result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    if len(x_shape) > 4:
        if x_shape[0] == -1:
            x_flat = tf.reshape(x, [-1] + x_shape[2:])
            output_shape = [-1] + output_shape[2:]
        else:
            x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
            output_shape = [output_shape[0] * output_shape[1]] + output_shape[2:]
        
        deconv = tf.nn.conv2d_transpose(x_flat, W, output_shape, strides=strides, padding=padding,
                                        data_format=data_format)
        # , data_format=data_format) - tensorflow version issues >.<
        deconv = tf.reshape(deconv, x_shape[:2] + deconv.get_shape().as_list()[1:])
    else:
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding=padding, data_format=data_format)
        # , data_format=data_format)
    return deconv


def avgpool2D(x, ksize, strides, padding, data_format):
    """Broadcastable version of tensorflow max_pool
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for avgpooling result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    if len(x_shape) > 4:
        if x_shape[0] == -1:
            x_flat = tf.reshape(x, [-1] + x_shape[2:])
        else:
            x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
        avgpool = tf.nn.avg_pool(x_flat, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
        avgpool = tf.reshape(avgpool, x_shape[:2] + avgpool.get_shape().as_list()[1:])
    else:
        avgpool = tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return avgpool


def maxpool2D(x, ksize, strides, padding, data_format):
    """Broadcastable version of tensorflow max_pool
    
    Parameters
    -------
    x : tensor
        Input tensor to be convoluted with weight mask; Shape can be [samples, x_dim, y_dim, features] or
        [samples, timesteps, x_dim, y_dim, features]; Convolution is performed over last 3 dimensions;
    
    Returns
    -------
    : tensor
        Tensor for maxpooling result
    """
    x_shape = x.get_shape().as_list()
    x_shape = [s if isinstance(s, int) else -1 for s in x_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    if len(x_shape) > 4:
        if x_shape[0] == -1:
            x_flat = tf.reshape(x, [-1] + x_shape[2:])
        else:
            x_flat = tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:])
        maxpool = tf.nn.max_pool(x_flat, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
        maxpool = tf.reshape(maxpool, x_shape[:2] + maxpool.get_shape().as_list()[1:])
    else:
        maxpool = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return maxpool


def resize2d(images, size, method=0, align_corners=False, name='resize2d'):
    """Broadcastable of tf.image.resize_images()"""
    images_shape = images.get_shape().as_list()
    images_shape = [s if isinstance(s, int) else -1 for s in images_shape]
    
    # Flatten matrix in first dimensions if necessary (join samples and sequence positions)
    with tf.variable_scope(name):
        if len(images_shape) > 4:
            if images_shape[0] == -1:
                images_flat = tf.reshape(images, [-1] + images_shape[2:])
            else:
                images_flat = tf.reshape(images, [images_shape[0] * images_shape[1]] + images_shape[2:])
            resized = tf.image.resize_images(images_flat, size, method=method, align_corners=align_corners)
            resized = tf.reshape(resized, images_shape[:2] + resized.get_shape().as_list()[1:])
        else:
            resized = tf.image.resize_images(images, size, method=method, align_corners=align_corners)
    return resized


def get_input(incoming):
    """Get input from Layer class or tensor
    
    Check if input is available via get_output() function or turn tensor into lambda expressions instead; Also
    try to fetch shape of incoming via get_output_shape();
    
    Returns
    -------
    : tensor
        Tensor with input
    : list
        Shape of input tensor
    """
    try:
        return incoming.get_output, incoming.get_output_shape()
    except AttributeError:
        return lambda **kwargs: incoming, [d if isinstance(d, int) else -1 for d in incoming.get_shape().as_list()]


def fire_module(incoming, s1x1, e1x1, e3x3, w_init, a=tf.nn.elu, name='fire'):
    """Fire module as proposed in SqueezeNet https://arxiv.org/abs/1602.07360
    
    This function builds and returns a fire module.
    
    Parameters
    -------
    incoming : Layer
        Input layer to fire module
    s1x1 : int
        Number of output features for squeeze-part with 1x1 kernel
    e1x1 : int
        Number of output features for expand-part with 1x1 kernel
    e3x3 : int
        Number of output features for expand-part with 3x3 kernel
    w_init : function
        Initialization function for kernel weights
    a : function
        Activation function to be used in s1x1, e1x1, and e3x3
    
    Returns
    -------
    : list of Layer
        List containing all layers used in fire module with last list element being the output layer of fire module
    """
    layers = list()
    with tf.name_scope(name):
        # Squeeze (s1x1)
        layers.append(ConvLayer(incoming=incoming, W=w_init([1, 1, incoming.get_output_shape()[-1], s1x1]),
                                padding='SAME', name='{}_s1x1'.format(name), a=a))
        # Expand (e1x1)
        layers.append(ConvLayer(incoming=layers[0], W=w_init([1, 1, layers[0].get_output_shape()[-1], e1x1]),
                                padding='SAME', name='{}_e1x1'.format(name), a=a))
        # Expand (e3x3)
        layers.append(ConvLayer(incoming=layers[0], W=w_init([3, 3, layers[0].get_output_shape()[-1], e3x3]),
                                padding='SAME', name='{}_e3x3'.format(name), a=a))
        # Concat
        layers.append(ConcatLayer([layers[-2], layers[-1]], name='concat'))
    
    return layers


def refinement_module(incoming_f, incoming_m, w_init, num_feat_out, size_out=None, a=tf.nn.elu, name='refinement'):
    """Refinement module as proposed in https://arxiv.org/abs/1603.08695 (refactored version)
    
    This function builds and returns a refinement module taking an input incoming_f and incoming_m with
    incoming_f >= incoming_m in terms of features.
    
    Parameters
    -------
    incoming_f : Layer
        Input layer to refinement module with larger number of features
    incoming_m : Layer
        Input layer to refinement module with smaller number of features
    w_init : function
        Initialization function for kernel weights
    num_feat_out : int
        Number of output features
    size_out : tuple or list of int or None
        A 1-D int32 Tensor of 2 elements new_height and new_width used for optional resize of output;
    a : function
        Activation function to be used on output convolution
    
    Returns
    -------
    : list of Layer
        List containing all layers used in refinement module with last list element being the output layer
    """
    layers = list()
    f_shape = incoming_f.get_output_shape()
    m_shape = incoming_m.get_output_shape()
    
    if f_shape[-1] < m_shape[-1]:
        raise ValueError("Dimensions should be F>=M for refinement module but are F={} and M={}".format(f_shape[-1],
                                                                                                        m_shape[-1]))
    
    with tf.name_scope(name):
        
        # Refine F to S
        if f_shape[-1] > m_shape[-1]:
            layers.append(ConvLayer(incoming=incoming_f, W=w_init([3, 3, incoming_f.get_output_shape()[-1],
                                                                   m_shape[-1]]),
                                    padding='SAME', name='{}Sr'.format(name), a=a))
            incoming_f = layers[-1]
        
        # Refine S to S*
        layers.append(ConvLayer(incoming=incoming_f, W=w_init([3, 3, incoming_f.get_output_shape()[-1], num_feat_out]),
                                padding='SAME', name='{}S'.format(name), a=tf.identity))
        
        # Refine M to M*
        layers.append(ConvLayer(incoming=incoming_m, W=w_init([3, 3, incoming_m.get_output_shape()[-1], num_feat_out]),
                                padding='SAME', name='{}M'.format(name), a=tf.identity))
        
        # Creation of new M
        layers.append(SumLayer(incomings=layers[-2:], a=a))
        
        # Upscaling
        if size_out is not None:
            layers.append(ScalingLayer(layers[-1], size=size_out, method=0, align_corners=False))
    
    return layers


def calc_next_seq_pos(layers, max_seq_len=None, tickersteps=0, tickerstep_nodes=False):
    """Convenience function: Updates the network state for the next sequence position (incl. all lower layers)
    
    This function is only needed to explicitly compute the network state for the next sequence position. Calling
    .get_output() makes this function obsolete.
    
    Parameters
    -------
    layers : list of layers
        List of LAyers representing the network
    max_seq_len : int or None
        optional limit for number of sequence positions in input (will be cropped at end)
    tickersteps : int
        Number of tickersteps to run after final position
    tickerstep_nodes : bool
        Activate tickerstep input nodes? (True: add tickerstep-bias to activations)
    
    Returns
    -------
    """
    
    for layer in layers:
        try:
            _ = layer.compute_nsp(max_seq_len=max_seq_len, tickersteps=tickersteps, tickerstep_nodes=tickerstep_nodes)
        except AttributeError:
            pass


def multi_dilation_conv(incoming, dilation_rates, join='mean', a=tf.nn.elu, name="multi_dilation_conv", **kwargs):
    """Convenience function for multiple dilation masks within the same convolutional layer; see ConvLayer for full
    parameter explainations;
    
    Parameters
    -------
    incoming : layer, tensorflow tensor, or placeholder
        Input of shape (samples, sequence_positions, features) or (samples, features) or (samples, ..., features);
    dilation_rates: list of tuples of int
        list of tuples of length 2 (see dilation_rate in ConvLayer)
    join : str
        String defining method to elementwise join the output of the layers with varying dilation rates; can be 'sum'
        or 'mean';
    a : function
        Activation function applied for joined layers
    
    Returns
    -------
    list of created layers, with output layer as last list element
    
    """
    conv_layers = []
    
    with tf.name_scope(name):
        for dilation_rate in dilation_rates:
            conv_layers.append(ConvLayer(incoming, **kwargs, a=tf.identity, dilation_rate=dilation_rate))
        if join == 'sum':
            output = SumLayer(conv_layers, a=a)
        elif join == 'mean':
            output = MeanLayer(conv_layers, a=a)
        else:
            raise AttributeError("join must be 'sum' or 'mean' but is '{}'".format(join))
    
    return conv_layers + [output]


# ------------------------------------------------------------------------------------------------------------------
#  Classes
# ------------------------------------------------------------------------------------------------------------------
class Layer(object):
    def __init__(self):
        """Template class for all layers
        
        Parameters
        -------
        
        Returns
        -------
        
        Attributes
        -------
        .out : tensor or None
            Current output of the layer (does not trigger computation)
        """
        self.layer_scope = None
        self.out = None
    
    def get_output(self, **kwargs):
        """Calculate and return output of layer"""
        return self.out
    
    def get_output_shape(self):
        """Return shape of output (preferably without calculating the output)"""
        return []


class RNNInputLayer(Layer):
    def __init__(self, incoming, name='RNNInputLayer'):
        """
        Input layer for RNNs that takes an update-able input tensor "incoming" and provides it as a layer
                
        Use this layer to create an input for the network that can be updated (e.g. at every sequence position)

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Incoming layer
        
        Returns
        -------
        """
        super(RNNInputLayer, self).__init__()
        self.incoming, self.incoming_shape = get_input(incoming)
        with tf.variable_scope(name) as self.layer_scope:
            self.out = self.incoming()
            self.name = name
    
    def update(self, incoming):
        """Update the input (e.g. for the next timestep)
        
        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input to use as update for layer
        """
        self.incoming, self.incoming_shape = get_input(incoming)
        with tf.variable_scope(self.layer_scope):
            self.out = self.incoming()
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shape
    
    def get_output(self, **kwargs):
        """Calculate and return output of layer"""
        with tf.variable_scope(self.layer_scope):
            return self.out


class DropoutLayer(Layer):
    def __init__(self, incoming, prob, noise_shape=None, name='DropoutLayer'):
        """ Dropout layer using tensorflow dropout

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Incoming layer
        prob : float or False
            Probability to drop out an element
        noise_shape : list or None
            Taken from tensorflow documentation: By default, each element is kept or dropped independently. If
            noise_shape is specified, it must be broadcastable to the shape of x, and only dimensions with
            noise_shape[i] == shape(x)[i] will make independent decisions. For example, if shape(x) = [k, l, m, n] and
            noise_shape = [k, 1, 1, n], each batch and channel component will be kept independently and each row and
            column will be kept or not kept together.
            If None: drop out last dimension of input tensor consistently (i.e. drop out features);

        Returns
        -------
        """
        super(DropoutLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            if noise_shape is None:
                noise_shape = np.append(np.ones(len(self.incoming_shape) - 1, dtype=np.int32),
                                        [self.incoming_shape[-1]])
            else:
                self.noise_shape = noise_shape
            
            self.prob = prob
            self.noise_shape = noise_shape
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.prob is not False:
                    self.out = tf.nn.dropout(incoming, keep_prob=1. - self.prob, noise_shape=self.noise_shape)
                else:
                    self.out = incoming
        
        return self.out


class MeanLayer(Layer):
    def __init__(self, incomings, a=tf.identity, name='MeanLayer'):
        """Elementwise mean of multiple layers
        
        Parameters
        -------
        incomings : list of layers, tensorflow tensors, or placeholders
            Input of shape (samples, sequence_positions, features) or (samples, features) or (samples, ..., features);
        a : tf function
            Activation function
            
        Returns
        -------
        """
        super(MeanLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incomings = []
            self.incoming_shapes = []
            
            for incoming in incomings:
                incoming, incoming_shape = get_input(incoming)
                self.incomings.append(incoming)
                self.incoming_shapes.append(incoming_shape)
            
            self.a = a
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shapes[0]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        
        if self not in prev_layers:
            prev_layers += [self]
            incomings = [incoming(prev_layers=prev_layers, **kwargs) for incoming in self.incomings]
            with tf.variable_scope(self.layer_scope):
                self.out = tf.add_n(incomings) / len(incomings)
        
        return self.out


class SumLayer(Layer):
    def __init__(self, incomings, a=tf.identity, name='SumLayer'):
        """Elementwise sum of multiple layers
        
        Parameters
        -------
        incomings : list of layers, tensorflow tensors, or placeholders
            Input of shape (samples, sequence_positions, features) or (samples, features) or (samples, ..., features);
        a : tf function
            Activation function
            
        Returns
        -------
        """
        super(SumLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incomings = []
            self.incoming_shapes = []
            
            for incoming in incomings:
                incoming, incoming_shape = get_input(incoming)
                self.incomings.append(incoming)
                self.incoming_shapes.append(incoming_shape)
            
            self.a = a
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shapes[0]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        
        if self not in prev_layers:
            prev_layers += [self]
            incomings = [incoming(prev_layers=prev_layers, **kwargs) for incoming in self.incomings]
            with tf.variable_scope(self.layer_scope):
                self.out = tf.add_n(incomings)
        
        return self.out


class DenseLayer(Layer):
    def __init__(self, incoming, n_units, flatten_input=False, W=tf.zeros, b=tf.zeros, a=tf.sigmoid, name='DenseLayer'):
        """ Dense layer, flexible enough to broadcast over time series

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input of shape (samples, sequence_positions, features) or (samples, features) or (samples, ..., features);
        n_units : int
            Number of dense layer units
        flatten_input : bool
            True: flatten all inputs (samples[, ...], features) to shape (samples, -1); i.e. fully connect to everything
            per sample
            False: flatten all inputs (samples[, sequence_positions, ...], features) to shape
            (samples, sequence_positions, -1); i.e. fully connect to everything per sample or sequence position;
        W : initializer or tensor or tf.Variable
            Weights W either as initializer or tensor or tf.Variable; Will be used as learnable tf.Variable in any case;
        b : initializer or tensor or tf.Variable or None
            Biases b either as initializer or tensor or tf.Variable; Will be used as learnable tf.Variable in any case;
            No bias will be applied if b=None;
        a : function
            Activation function
        name : string
            Name of individual layer; Used as tensorflow scope;
            
        Returns
        -------
        """
        super(DenseLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            if (len(self.incoming_shape) > 2) and flatten_input:
                incoming_shape = [self.incoming_shape[0], np.prod(self.incoming_shape[1:])]
            else:
                incoming_shape = self.incoming_shape
            
            # Set init for W
            W = tofov(W, shape=[incoming_shape[-1], n_units], var_params=dict(name='W_dense'))
            
            # Set init for b
            if b is not None:
                b = tofov(b, [n_units], var_params=dict(name='b_dense'))
            
            self.a = a
            self.b = b
            self.W = W
            
            self.n_units = n_units
            self.flatten_input = flatten_input
            self.incoming_shape = incoming_shape
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shape[:-1] + [self.n_units]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.flatten_input:
                    # Flatten all but first dimension (e.g. flat seq_pos and features)
                    X = tf.reshape(incoming, self.incoming_shape)
                else:
                    X = incoming
                net = dot_product(X, self.W)
                if self.b is not None:
                    net += self.b
                self.out = self.a(net)
        
        return self.out
    
    def get_weights(self):
        """Return list with all layer weights"""
        return [self.W]
    
    def get_biases(self):
        """Return list with all layer biases"""
        if self.b is None:
            return []
        else:
            return [self.b]


class LSTMLayer(Layer):
    def __init__(self, incoming, n_units,
                 W_ci=tf.zeros, W_ig=tf.zeros, W_og=tf.zeros, W_fg=tf.zeros,
                 b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                 a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.identity,
                 c_init=tf.zeros, h_init=tf.zeros, learn_c_init=False, learn_h_init=False, forgetgate=True,
                 output_dropout=False, store_states=False, return_states=False, precomp_fwds=False,
                 tickerstep_biases=None, learn_tickerstep_biases=True, name='LSTM'):
        """LSTM layer for different types of sequence predictions with inputs of shape [samples, sequence positions,
        features] or typically [samples, 1, features] in combination with RNNInputLayer
        
        Parameters
        -------
        incoming : tensorflow tensor or placeholder
            Input layer to LSTM layer of shape [samples, sequence positions, features] or typically
            [samples, 1, features] in combination with RNNInputLayer
        n_units : int
            Number of LSTM units in layer;
        W_ci, W_ig, W_og, W_fg : (list of) initializer or (list of) tensor or (list of) tf.Variable
            Initial values or initializers for cell input, input gate, output gate, and forget gate weights; Can be list
            of 2 elements as [W_fwd, W_bwd] to define different weight initializations for forward and recurrent
            connections; If single element, forward and recurrent connections will use the same initializer/tensor;
            Shape of weights is [n_inputs, n_outputs];
        b_ci, b_ig, b_og, b_fg : tensorflow initializer or tensor or tf.Variable
            Initial values or initializers for bias for cell input, input gate, output gate, and forget gate;
        a_ci,  a_ig, a_og, a_fg, a_out :  tensorflow function
            Activation functions for cell input, input gate, output gate, forget gate, and LSTM output respectively;
        c_init : tensorflow initializer or tensor or tf.Variable
            Initial values for cell states; By default not learnable, see learn_c_init;
        h_init : tensorflow initializer or tensor or tf.Variable
            Initial values for hidden states; By default not learnable, see learn_h_init;
        learn_c_init : bool
            Make c_init learnable?
        learn_h_init : bool
            Make h_init learnable?
        forgetgate : bool
            Flag to disable the forget gate (i.e. to always set its output to 1)
        output_dropout : float or False
            Dropout rate for LSTM output dropout (i.e. dropout of whole LSTM unit with rescaling of the remaining
            units); This also effects the recurrent connections;
        store_states : bool
            True: Store hidden states and cell states in lists self.h and self.c
        return_states : bool
            True: Return all hidden states (continuous prediction); this forces store_states to True;
            False: Only return last hidden state (single target prediction)
        precomp_fwds : bool
            True: Forward inputs are precomputed over all sequence positions at once
        tickerstep_biases : initializer, tensor, tf.Variable or None
            not None: Add this additional bias to the forward input if tickerstep_nodes=True for get_output();
        learn_tickerstep_biases : bool
            Make tickerstep_biases learnable?
        name : string
            Name of individual layer; Used as tensorflow scope;

        Returns
        -------
        """
        super(LSTMLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            self.n_units = n_units
            self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
            if return_states:
                store_states = True
            
            #
            # Initialize weights and biases
            #
            
            # Turn W inits into lists [forward_pass, backward_pass]
            W_ci, W_ig, W_og, W_fg = [v[:2] if isinstance(v, list) else [v, v] for v in [W_ci, W_ig, W_og, W_fg]]
            
            # Make W and b tf variables
            W_ci, W_ig, W_og, W_fg = [
                [tofov(v[0], shape=[self.incoming_shape[-1], n_units], var_params=dict(name=n + '_fwd')),
                 tofov(v[1], shape=[n_units, n_units], var_params=dict(name=n + '_bwd'))]
                for v, n in zip([W_ci, W_ig, W_og, W_fg], ['W_ci', 'W_ig', 'W_og', 'W_fg'])]
            b_ci, b_ig, b_og, b_fg = [tofov(v, shape=[n_units], var_params=dict(name=n)) for v, n in
                                      zip([b_ci, b_ig, b_og, b_fg], ['b_ci', 'b_ig', 'b_og', 'b_fg'])]
            
            # Pack weights for fwd and bwd connections
            W_fwd_conc = tf.concat(axis=1, values=[W[0] for W in [W_ci, W_ig, W_og, W_fg]])
            W_bwd_conc = tf.concat(axis=1, values=[W[1] for W in [W_ci, W_ig, W_og, W_fg]])
            
            if not forgetgate:
                def a_fg(x):
                    return tf.ones(x.get_shape().as_list())
            
            # Initialize bias for tickersteps
            if tickerstep_biases is not None:
                self.W_tickers = OrderedDict(zip_longest(self.lstm_inlets,
                                                         [tofov(tickerstep_biases, shape=[n_units],
                                                                var_params=dict(name='W_tickers_' + g,
                                                                                trainable=learn_tickerstep_biases))
                                                          for g in self.lstm_inlets]))
            else:
                self.W_tickers = None
            
            #
            # Create mask for output dropout
            # apply dropout to n_units dimension of outputs, keeping dropout mask the same for all samples,
            # sequence positions, and pixel coordinates
            #
            output_shape = self.get_output_shape()
            if output_dropout:
                out_do_mask = tf.ones(shape=[output_shape[0], output_shape[-1]],
                                      dtype=tf.float32)
                out_do_mask = tf.nn.dropout(out_do_mask, keep_prob=1. - output_dropout,
                                            noise_shape=[1, output_shape[-1]])
            
            def out_do(x):
                """Function for applying dropout mask to outputs"""
                if output_dropout:
                    return out_do_mask * x
                else:
                    return x
            
            # Redefine a_out to include dropout (sneaky, sneaky)
            a_out_nodropout = a_out
            
            def a_out(x):
                return a_out_nodropout(out_do(x))
            
            #
            # Handle initializations for h (hidden states) and c (cell states) as Variable
            #
            h_init = out_do(tofov(h_init, shape=[output_shape[0], output_shape[-1]],
                                  var_params=dict(name='h_init', trainable=learn_h_init)))
            c_init = tofov(c_init, shape=[output_shape[0], output_shape[-1]],
                           var_params=dict(name='h_init', trainable=learn_c_init))
            
            # Initialize lists to store LSTM activations and cell states later
            h = [h_init]
            c = [c_init]
            
            self.precomp_fwds = precomp_fwds
            self.store_states = store_states
            self.return_states = return_states
            
            self.W_fwd = OrderedDict(zip(self.lstm_inlets, [W[0] for W in [W_ci, W_ig, W_og, W_fg]]))
            self.W_bwd = OrderedDict(zip(self.lstm_inlets, [W[1] for W in [W_ci, W_ig, W_og, W_fg]]))
            
            self.W_fwd_conc = W_fwd_conc
            self.W_bwd_conc = W_bwd_conc
            self.a = OrderedDict(zip(self.lstm_inlets, [a_ci, a_ig, a_og, a_fg]))
            self.a['out'] = a_out
            self.b = OrderedDict(zip(self.lstm_inlets, [b_ci, b_ig, b_og, b_fg]))
            self.h = h
            self.c = c
            self.external_rec = None
            
            self.out = tf.expand_dims(h_init, 1)
            self.name = name
    
    def add_external_recurrence(self, incoming):
        """Use this Layer as recurrent connections for the LSTM instead of LSTM hidden activations; In this case the
        weight initialization for the recurrent weights has to be done by hand, e.g.
        W_ci = [w_init, w_init([n_recurrents, n_lstm])]
        
        Parameters
        -------
        incoming : layer class, tensorflow tensor or placeholder
            Incoming external recurrence for LSTM layer as layer class or tensor of shape
            (samples, 1, features)
        
        Example
        -------
        >>> # Example for LSTM that uses its own hidden state and the output of a higher convolutional layer as
        >>> # recurrent connections
        >>> lstm = LSTMLayer(...)
        >>> dense_1 = DenseLayer(incoming=lstm, ...)
        >>> modified_recurrence = ConcatLayer(lstm, dense_1)
        >>> lstm.add_external_recurrence(modified_recurrence)
        >>> lstm.get_output()
        """
        self.external_rec, _ = get_input(incoming)
    
    def comp_net_fwd(self, incoming, start_at=0):
        """Compute and yield relative sequence position and net_fwd per real sequence position

        Compute net_fwd as specified by precomp_fwds (precompute it for all inputs or compute it online for each
        seq_pos) for incoming with shape: (samples, sequence positions, features);

        Parameters
        -------
        incoming : tensor
            Incoming layer
        start_at : int
            Start at this sequence position in the input sequence; If negative, the unknown sequence positions
            at the beginning will be padded with the first sequence position:

        Returns
        -------
        seq_pos : int
            Relative sequence position (starting at 0)
        net_fwd : tf.tensor
            Yields the net_fwd at the current sequence position with shape (samples, features)
        """
        precomp_fwds = self.precomp_fwds
        W_fwd = self.W_fwd_conc
        
        if precomp_fwds:
            # Pre compute net input
            net_fwd_precomp = dot_product(incoming, W_fwd)
        else:
            net_fwd_precomp = None
        
        # loop through sequence positions and return respective net_fwd
        for relative_seq_pos, real_seq_pos in enumerate(range(start_at, self.incoming_shape[1])):
            # If first sequence positions are unknown, pad beginning of sequence with first sequence position
            if real_seq_pos < 0:
                index_seq_pos = 0
            else:
                index_seq_pos = real_seq_pos
            
            # Yield the relative sequence position and net_fwd by either indexing the precomputed net_fwd_precomp or
            # calculating the net_fwd online per sequence position
            if precomp_fwds:
                cur_net_fwd = net_fwd_precomp[:, index_seq_pos, :]
            else:
                cur_net_fwd = dot_product(incoming[:, index_seq_pos, :], W_fwd)
            
            yield relative_seq_pos, cur_net_fwd
    
    def get_output_shape(self):
        """Return shape of output"""
        # Get shape of output tensor(s), which will be [samples, n_seq_pos+n_tickersteps, n_units]
        return [self.incoming_shape[0], -1] + [self.n_units]
    
    def get_output(self, prev_layers=None, max_seq_len=None, tickersteps=0,
                   tickerstep_nodes=False, comp_next_seq_pos=True, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        max_seq_len : int or None
            Can be used to artificially hard-clip the sequences at length max_seq_len
        tickersteps : int or None
            Tickersteps to apply after the sequence end; Tickersteps use 0 input and a trainable bias; not suitable for
            variable sequence lengths;
        tickerstep_nodes : bool
            True: Current sequence positions will be treated as ticker steps (tickerstep-bias is added to activations)
        comp_next_seq_pos : bool
            True: Cell state and hidden state for next sequence position will be computed
            False: Return current hidden state without computing next sequence position
        """
        if prev_layers is None:
            prev_layers = list()
        if (self not in prev_layers) and comp_next_seq_pos:
            prev_layers += [self]
            self.compute_nsp(prev_layers=prev_layers, max_seq_len=max_seq_len, tickersteps=tickersteps,
                             tickerstep_nodes=tickerstep_nodes, **kwargs)
        if self.return_states:
            if len(self.h) == 1:
                self.out = tf.expand_dims(self.h[-1], 1)  # add empty dimension for seq_pos
            else:
                self.out = tf.stack(self.h[1:], axis=1)  # pack states but omit initial state
        else:
            self.out = tf.expand_dims(self.h[-1], 1)  # add empty dimension for seq_pos
        return self.out
    
    def compute_nsp(self, prev_layers=None, max_seq_len=None, tickersteps=0, tickerstep_nodes=False, **kwargs):
        """
        Computes next sequence position
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        max_seq_len : int or None
            optional limit for number of sequence positions in input (will be cropped at end)
        tickersteps : int
            Number of tickersteps to run after final position
        tickerstep_nodes : bool
            Activate tickerstep input nodes? (True: add tickerstep-bias to activations)
        """
        incoming = self.incoming(prev_layers=prev_layers, comp_next_seq_pos=True, max_seq_len=max_seq_len,
                                 tickersteps=tickersteps, tickerstep_nodes=tickerstep_nodes, **kwargs)
        
        external_rec = None
        if self.external_rec is not None:
            external_rec = self.external_rec(prev_layers=prev_layers, max_seq_len=max_seq_len, tickersteps=tickersteps,
                                             tickerstep_nodes=tickerstep_nodes, comp_next_seq_pos=True,
                                             **kwargs)[:, -1, :]
        
        act = OrderedDict(zip_longest(self.lstm_inlets, [None]))
        
        with tf.variable_scope(self.name) as scope:
            # Make sure tensorflow can reuse the variable names
            scope.reuse_variables()
            
            # Handle restriction on maximum sequence length
            if max_seq_len is not None:
                incoming = incoming[:, :max_seq_len, :]
            
            #
            # Compute LSTM cycle at each sequence position in 'incoming'
            #
            
            # Loop through sequence positions and get corresponding net_fwds
            for seq_pos, net_fwd in self.comp_net_fwd(incoming):
                
                # Calculate net for recurrent connections at current sequence position
                if self.external_rec is None:
                    net_bwd = dot_product(self.h[-1], self.W_bwd_conc)
                else:
                    net_bwd = dot_product(external_rec, self.W_bwd_conc)
                
                # Sum up net from forward and recurrent connections
                act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=1, num_or_size_splits=4,
                                                                      value=net_fwd + net_bwd)
                
                act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=1, num_or_size_splits=4,
                                                                      value=net_fwd + net_bwd)
                
                # peepholes could be added here #
                
                # Calculate activations
                if tickerstep_nodes and (self.W_tickers is not None):
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g] + self.W_tickers[g])
                                                             for g in self.lstm_inlets]))
                else:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g])
                                                             for g in self.lstm_inlets]))
                
                # Calculate new cell state
                if self.store_states:
                    self.c.append(act['ci'] * act['ig'] + self.c[-1] * act['fg'])
                else:
                    self.c[-1] = act['ci'] * act['ig'] + self.c[-1] * act['fg']
                
                # Calculate new output with new cell state
                if self.store_states:
                    self.h.append(self.a['out'](self.c[-1]) * act['og'])
                else:
                    self.h[-1] = self.a['out'](self.c[-1]) * act['og']
            
            # Process tickersteps
            for _ in enumerate(range(tickersteps)):
                # The forward net input during the ticker steps is 0 (no information is added anymore)
                # ticker_net_fwd = 0
                
                # Calculate net for recurrent connections at current sequence position
                if self.external_rec is None:
                    net_bwd = dot_product(self.h[-1], self.W_bwd_conc)
                else:
                    net_bwd = dot_product(external_rec, self.W_bwd_conc)
                
                # Split net from recurrent connections
                act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=1, num_or_size_splits=4, value=net_bwd)
                
                # Calculate activations including ticker steps
                if self.W_tickers is not None:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g] + self.W_tickers[g])
                                                             for g in self.lstm_inlets]))
                else:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g])
                                                             for g in self.lstm_inlets]))
                
                # Calculate new cell state
                if self.store_states:
                    self.c.append(act['ci'] * act['ig'] + self.c[-1] * act['fg'])
                else:
                    self.c[-1] = act['ci'] * act['ig'] + self.c[-1] * act['fg']
                
                # Calculate new output with new cell state
                if self.store_states:
                    self.h.append(self.a['out'](self.c[-1]) * act['og'])
                else:
                    self.h[-1] = self.a['out'](self.c[-1]) * act['og']
    
    def get_weights(self):
        """Return list with all layer weights"""
        if self.W_tickers is not None:
            return [w for w in [self.W_fwd_conc, self.W_bwd_conc] + list(self.W_tickers.values())
                    if w is not None]
        else:
            return [w for w in [self.W_fwd_conc, self.W_bwd_conc] if w is not None]
    
    def get_biases(self):
        """Return list with all layer biases"""
        return list(self.b.values())


class ConvLayer(Layer):
    def __init__(self, incoming, W=None, b=tf.zeros, ksize: int = None, num_outputs: int = None,
                 weight_initializer=None, a=tf.nn.elu, strides=(1, 1, 1, 1), padding='ZEROPAD', dilation_rate=(1, 1),
                 name='ConvLayer'):
        """ Convolutional layer, flexible enough to broadcast over timeseries

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input of shape (samples, sequence_positions, array_x, array_y, features) or
            (samples, array_x, array_y, features);
        W : tensorflow tensor or tf.Variable
            Initial value for weight kernel of shape (kernel_x, kernel_y, n_input_channels, n_output_channels)
        b : tensorflow initializer or tensor or tf.Variable or None
            Initial values or initializers for bias; None if no bias should be applied;
        ksize : int
            Kernel size; only used in conjunction with num_outputs and weight_initializer
        num_outputs : int
            Number of output feature maps; only used in conjunction with ksize and weight_initializer
        weight_initializer : initializer function
            Function for initialization of weight kernels; only used in conjunction with ksize and num_outputs
        a :  tensorflow function
            Activation functions for output
        strides : tuple
            Striding to use (see tensorflow convolution for further details)
        padding : str or tuple of int
            Padding method for image edges (see tensorflow convolution for further details); If specified as
            tuple or list of integer tf.pad is used to symmetrically zero-pad the x and y dimensions of the input.
            Furthermore supports TensorFlow paddings "VALID" and "SAME" in addition to "ZEROPAD" which symmetrically
            zero-pads the input so output-size = input-size / stride (taking into account strides and dilation;
            comparable to Caffe and Theano).
        dilation_rate : tuple of int or list of int
            Defaults to (1, 1) (i.e. normal 2D convolution). Use list of integers to specify multiple dilation rates;
            only for spatial dimensions -> len(dilation_rate) must be 2;
        
        Returns
        -------
        """
        super(ConvLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            # Set init for W and b
            if all(p is not None for p in [weight_initializer, ksize, num_outputs]):
                W = tofov(weight_initializer, shape=(ksize, ksize, incoming.get_output_shape()[-1], num_outputs),
                          var_params=dict(name='W_conv'))
            else:
                W = tofov(W, shape=None, var_params=dict(name='W_conv'))
                ksize = W.get_shape()[0].value
            if b is not None:
                b = tofov(b, shape=W.get_shape().as_list()[-1], var_params=dict(name='b_conv'))
            
            if padding == "ZEROPAD":
                if len(self.incoming_shape) == 5:
                    s = strides[1:3]
                    i = (int(self.incoming_shape[2] / s[0]), int(self.incoming_shape[3] / s[1]))
                elif len(self.incoming_shape) == 4:
                    s = strides[1:3]
                    i = (int(self.incoming_shape[1] / s[0]), int(self.incoming_shape[2] / s[1]))
                else:
                    raise ValueError("invalid input shape")
                # --
                padding_x = int(np.ceil((i[0] - s[0] - i[0] + ksize + (ksize - 1) * (dilation_rate[0] - 1)) / (s[0] * 2)))
                padding_y = int(np.ceil((i[1] - s[1] - i[1] + ksize + (ksize - 1) * (dilation_rate[1] - 1)) / (s[1] * 2)))
            elif (isinstance(padding, list) or isinstance(padding, tuple)) and len(padding) == 2:
                padding_x = padding[0]
                padding_y = padding[1]
            
            if padding == "SAME" or padding == "VALID":
                self.padding = padding
            else:
                if len(self.incoming_shape) == 5:
                    self.padding = [[0, 0], [0, 0], [padding_x, padding_x], [padding_y, padding_y], [0, 0]]
                elif len(self.incoming_shape) == 4:
                    self.padding = [[0, 0], [padding_x, padding_x], [padding_y, padding_y], [0, 0]]
            
            self.a = a
            self.b = b
            self.W = W
            self.strides = strides
            self.dilation_rate = dilation_rate
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        # TODO: return shape without construction of graph
        return self.get_output(comp_next_seq_pos=False).get_shape().as_list()
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                # Perform convolution
                if isinstance(self.padding, list):
                    conv = conv2d(tf.pad(incoming, self.padding, "CONSTANT"), self.W, strides=self.strides,
                                  padding="VALID", dilation_rate=self.dilation_rate)
                else:
                    conv = conv2d(incoming, self.W, strides=self.strides, padding=self.padding,
                                  dilation_rate=self.dilation_rate)
                
                # Add bias
                if self.b is not None:
                    conv += self.b
                
                # Apply activation function
                self.out = self.a(conv)
        
        return self.out
    
    def get_weights(self):
        """Return list with all layer weights"""
        return [self.W]
    
    def get_biases(self):
        """Return list with all layer biases"""
        if self.b is None:
            return []
        else:
            return [self.b]


class AvgPoolingLayer(Layer):
    def __init__(self, incoming, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC',
                 name='MaxPoolingLayer'):
        """Average-pooling layer, capable of broadcasing over timeseries
        
        see tensorflow nn.avg_pool function for further details on parameters
        
        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            input to layer
            
        Returns
        -------
        """
        super(AvgPoolingLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.ksize = ksize
            self.strides = strides
            self.padding = padding
            self.data_format = data_format
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        # TODO: return shape without construction of graph
        return self.get_output(comp_next_seq_pos=False).get_shape().as_list()
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                self.out = avgpool2D(incoming, ksize=self.ksize, strides=self.strides, padding=self.padding,
                                     data_format=self.data_format)
        return self.out


class MaxPoolingLayer(Layer):
    def __init__(self, incoming, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC',
                 name='MaxPoolingLayer'):
        """Max pooling layer, capable of broadcasting over time series
        
        see tensorflow max pooling function for further details on parameters
        
        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            input to layer
            
        Returns
        -------
        """
        super(MaxPoolingLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.ksize = ksize
            self.strides = strides
            self.padding = padding
            self.data_format = data_format
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        # TODO: return shape without construction of graph
        return self.get_output(comp_next_seq_pos=False).get_shape().as_list()
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                self.out = maxpool2D(incoming, ksize=self.ksize, strides=self.strides, padding=self.padding,
                                     data_format=self.data_format)
        return self.out


class DeConvLayer(Layer):
    def __init__(self, incoming, W=None, b=tf.zeros, ksize: int = None, num_outputs: int = None,
                 weight_initializer=None, a=tf.nn.elu, output_shape=None, strides=(1, 2, 2, 1), padding='SAME',
                 data_format='NHWC',
                 name='DeConvLayer'):
        """Deconvolution (upscaling) layer, based on tf.nn.conv2d_transpose but capable of broadcasting over time series


        see tensorflow tf.nn.conv2d_transpose function for further details on parameters

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Incoming layer
        W : 4D tensor
            [height, width, output_channels, in_channels]
        Returns
        -------
        """
        super(DeConvLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            # Set init for W and b
            if all(p is not None for p in [weight_initializer, ksize, num_outputs]):
                W = tofov(weight_initializer,
                          shape=(ksize, ksize, num_outputs, incoming.get_output_shape()[-1]),
                          var_params=dict(name='W_deconv'))
            else:
                W = tofov(W, shape=None, var_params=dict(name='W_deconv'))
            b = tofov(b, shape=W.get_shape().as_list()[-2], var_params=dict(name='b_deconv'))
            
            if output_shape is None:
                if padding == 'SAME' and strides[0] == 1:
                    if len(self.incoming_shape) == 5:
                        output_shape = [self.incoming_shape[0], self.incoming_shape[1],
                                        self.incoming_shape[2] * strides[1], self.incoming_shape[3] * strides[2],
                                        W.get_shape().as_list()[-2] * strides[3]]
                    else:
                        output_shape = [self.incoming_shape[0], self.incoming_shape[1] * strides[1],
                                        self.incoming_shape[2] * strides[2], W.get_shape().as_list()[-2] * strides[3]]
                else:
                    raise AttributeError("Automatic output_shape calculation not implemented for strides!=1 in "
                                         "first dimension")
            
            if isinstance(padding, int):
                if len(self.incoming_shape) == 5:
                    self.padding = [[0, 0], [0, 0], [padding, padding], [padding, padding], [0, 0]]
                elif len(self.incoming_shape) == 4:
                    self.padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
                else:
                    raise ValueError("invalid input shape")
            else:
                self.padding = padding
            
            self.a = a
            self.b = b
            self.W = W
            
            self.output_shape = output_shape
            self.strides = strides
            
            self.data_format = data_format
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.output_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                self.out = self.a(conv2d_transpose2d(incoming, W=self.W, output_shape=self.output_shape,
                                                     strides=self.strides, padding=self.padding,
                                                     data_format=self.data_format) + self.b)
        return self.out


class ScalingLayer(Layer):
    def __init__(self, incoming, size, method=0, align_corners=False, name='ScalingLayer'):
        """Scaling layer for images and frame sequences (broadcastable version of tf.image.resize_images)

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            input to layer
        size : tuple or list of int
            A 1-D int32 Tensor of 2 elements: new_height, new_width
        method : str
            A string to set the upscaling method to be used by tf.image.resize_images
        Returns
        -------
        """
        super(ScalingLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            if len(self.incoming_shape) > 4:
                self.output_shape = self.incoming_shape[:2] + list(size) + self.incoming_shape[4:]
            else:
                self.output_shape = self.incoming_shape[:1] + list(size) + self.incoming_shape[3:]
            
            self.scale_size = size
            self.method = method
            self.align_corners = align_corners
            
            self.out = None
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.output_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.incoming_shape == self.scale_size:
                    self.out = incoming
                else:
                    self.out = resize2d(incoming, size=self.scale_size, method=self.method,
                                        align_corners=self.align_corners)
        
        return self.out


class ConcatLayer(Layer):
    def __init__(self, incomings, name='ConcatLayer'):
        """Concatenate outputs of multiple layers at last dimension (e.g. for skip-connections)
        
        Parameters
        -------
        incomings : list of layers, tensorflow tensors, or placeholders
            List of incoming layers to be concatenated
        
        Returns
        -------
        """
        super(ConcatLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incomings = []
            self.incoming_shapes = []
            
            for incoming in incomings:
                incoming, incoming_shape = get_input(incoming)
                self.incomings.append(incoming)
                self.incoming_shapes.append(incoming_shape)
            self.name = name
    
    def get_output_shape(self):
        """Return shape of output"""
        return self.incoming_shapes[0][:-1] + [sum([s[-1] for s in self.incoming_shapes])]
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        """
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incomings = [incoming(prev_layers=prev_layers, **kwargs) for incoming in self.incomings]
            with tf.variable_scope(self.layer_scope):
                self.out = tf.concat(axis=len(self.incoming_shapes[0]) - 1, values=incomings)
        
        return self.out


class ConvLSTMGate(object):
    def __init__(self, W=None, b=tf.zeros, ksize: int = None, num_outputs: int = None, weight_initializer=None,
                 a=tf.nn.elu, name='ConvLSTMInlet'):
        pass


class ConvLSTMLayer(Layer):
    def __init__(self, incoming, n_units, W_ci, W_ig, W_og, W_fg,
                 b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                 a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                 c_init=tf.zeros, h_init=tf.zeros, learn_c_init=False, learn_h_init=False, forgetgate=True, comb='add',
                 store_states=False, return_states=False, output_dropout=False, precomp_fwds=True, W_red_rec=None,
                 a_reduce_recurrents=None, tickerstep_biases=None, learn_tickerstep_biases=True,
                 dilation_rate=(1, 1), name='ConvLSTMLayer'):
        """ ConvLSTM layer for different types of pixlewise sequence predictions with inputs of shape
        (samples, sequence positions, array_x, array_y, features).
        
        Parameters
        -------
        incoming : layer class, tensorflow tensor, or placeholder
            Input layer to convLSTM layer or tensor of shape (samples, sequence positions or ? or None, array_x,
            array_y, features)
        n_units : int
            Number of convLSTM units in layer (=number of output features)
        W_ci, W_ig, W_og, W_fg : list of tensorflow tensors or list of tf.Variables
            Initial values for cell input, input gate, output gate, and forget gate weight kernels; List of 2
            elements as [W_fwd, W_bwd] to define different weight initializations for forward and recurrent connections;
            Shape fwd: (kernel_x, kernel_y, input_features, n_units); Shape bwd: (kernel_x, kernel_y, n_units, n_units);
        b_ci, b_ig, b_og, b_fg : tensorflow initializer or tensor or tf.Variable
            Initial values or initializers for bias for cell input, input gate, output gate, and forget gate;
        a_ci, a_ig, a_og, a_fg, a_out : tensorflow function
            Activation functions for cell input, input gate, output gate, forget gate, and LSTM output respectively;
        c_init : tensorflow initializer or tensor or tf.Variable
            Initial values for cell states; By default not learnable, see learn_c_init;
        h_init : tensorflow initializer or tensor or tf.Variable
            Initial values for hidden states; By default not learnable, see learn_h_init;
        learn_c_init : bool
            Make c_init learnable?
        learn_h_init : bool
            Make h_init learnable?
        forgetgate : bool
            Flag to disable the forget gate (i.e. to always set its output to 1)
        comb : {'add', 'mul'}, None
            Method to combine forward and recurrent connections in the LSTM inlets; Can be 'add' for standard convLSTM
            (i.e. elementwise addition) or 'mul' for additional combination via elementwise multiplication
            ('mul' not implemented yet)
        store_states : bool
            True: Cell state and output at each sequence position is stored and accessible via .caf and .haf;
            False: Only prediction at last sequence position (+tickersteps) is returned;
        return_states : bool
            True: Return all hidden states (continuous prediction); this forces store_states to True;
            False: Only return last hidden state (single target prediction)
        output_dropout : float or False
            Dropout rate for convLSTM output dropout (i.e. dropout of whole convLSTM unit with rescaling of the
            remaining units); This also effects the recurrent connections; Dropout mask is kept the same over all
            sequence positions in current incoming;
        precomp_fwds : bool
            True: Forward inputs are precomputed for all sequence positions
        W_red_rec : tensor
            see a_reduce_recurrents;
        a_reduce_recurrents : tf function or None
            If not None: Use W_red_rec kernel to reduce number of internal recurrent features
            (see https://arxiv.org/abs/1603.08695) and apply activation function a_reduce_recurrents;
        tickerstep_biases : initializer, tensor, tf.Variable or None
            not None: Add this additional bias to the forward input if tickerstep_nodes=True for get_output();
        learn_tickerstep_biases : bool
            Make tickerstep_biases learnable?
        dilation_rate : tuple of int or list of int
            Defaults to (1, 1) (i.e. normal 2D convolution). Use list of integers to specify multiple dilation rates;
            only for spatial dimensions -> len(dilation_rate) must be 2;
        name : string
            Name of individual layer; Used as tensorflow scope;
        
        Returns
        -------
        """
        super(ConvLSTMLayer, self).__init__()
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            self.n_units = n_units
            self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
            if return_states:
                store_states = True
            
            #
            # Initialize weights and biases
            #
            
            # Turn W inits into lists [forward_pass, backward_pass]
            W_ci, W_ig, W_og, W_fg = [v[:2] if isinstance(v, list) else [v, v] for v in [W_ci, W_ig, W_og, W_fg]]
            
            # Check if feature dimensions to produce agree for all weight windows with n_units
            W_dims = [[w[0].get_shape().as_list()[3], w[1].get_shape().as_list()[3]] for w in [W_ci, W_ig, W_og, W_fg]]
            W_dims = np.array(W_dims).flatten()
            if np.any(W_dims != n_units):
                raise ValueError("Feature dimensions to produce must agree with n_units!")
            
            # TODO: make all gates optional (list with keys and function for splitting)
            if not forgetgate:
                def a_fg(x):
                    return tf.ones(x.get_shape().as_list())
            
            # Make W and b tf variables
            W_ci, W_ig, W_og, W_fg = [[tofov(w, shape=None, var_params=dict(name=n + suffix)) for w, suffix in
                                       zip(v, ['_fwd', '_bwd'])] for v, n in
                                      zip([W_ci, W_ig, W_og, W_fg], ['W_ci', 'W_ig', 'W_og', 'W_fg'])]
            b_ci, b_ig, b_og, b_fg = [tofov(b, shape=[n_units], var_params=dict(name=n)) for b, n in
                                      zip([b_ci, b_ig, b_og, b_fg], ['b_ci', 'b_ig', 'b_og', 'b_fg'])]
            
            # Pack weights for fwd and bwd connections by concatenating them at sliding mask feature dimension
            # TODO: enable parallel calculation on multi-gpu
            W_fwd_conc = tf.concat(axis=3, values=[W[0] for W in [W_ci, W_ig, W_og, W_fg]], name='W_fwd_conc')
            W_bwd_conc = tf.concat(axis=3, values=[W[1] for W in [W_ci, W_ig, W_og, W_fg]], name='W_bwd_conc')
            
            # Initialize kernel for reducing recurrent features
            self.reduce_recurrents = None
            self.W_red_rec = W_red_rec
            if a_reduce_recurrents is not None:
                self.W_red_rec = tofov(W_red_rec, var_params=dict(name='W_red_rec'))
                
                def reduce_recurrents(h_prev):
                    """Reduces features of internal recurrent connections h_prev"""
                    return a_reduce_recurrents(conv2d(h_prev, self.W_red_rec))
                
                self.reduce_recurrents = reduce_recurrents
            
            # Initialize bias for tickersteps
            if tickerstep_biases is not None:
                self.W_tickers = OrderedDict(zip_longest(self.lstm_inlets,
                                                         [tofov(tickerstep_biases, shape=[n_units],
                                                                var_params=dict(name='W_tickers_' + g,
                                                                                trainable=learn_tickerstep_biases))
                                                          for g in self.lstm_inlets]))
            else:
                self.W_tickers = None
            
            #
            # Create mask for output dropout
            # apply dropout to n_units dimension of outputs, keeping dropout mask the same for all samples,
            # sequence positions, and pixel coordinates
            #
            output_shape = self.get_output_shape()
            if output_dropout:
                out_do_mask = tf.ones(shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[4]],
                                      dtype=tf.float32, name='out_do_mask')
                out_do_mask = tf.nn.dropout(out_do_mask, keep_prob=1. - output_dropout,
                                            noise_shape=[1, 1, 1, output_shape[4]])
            
            def out_do(x):
                """Function for applying dropout mask to outputs"""
                if output_dropout:
                    return out_do_mask * x
                else:
                    return x
            
            # Redefine a_out to include dropout (sneaky, sneaky)
            a_out_nodropout = a_out
            
            def a_out(x):
                return a_out_nodropout(out_do(x))
            
            #
            # Handle initializations for h (outputs) and c (cell states) as Variable if overwriteable or tensor if not
            # shape=(samples, x, y, n_units)
            #
            h_init = out_do(tofov(h_init, shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[4]],
                                  var_params=dict(trainable=learn_h_init)))
            c_init = tofov(c_init, shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[4]],
                           var_params=dict(trainable=learn_c_init))
            
            # Initialize lists to store LSTM activations and cell states
            h = [h_init]  # [-1, x, y, n_units]
            c = [c_init]  # [-1, x, y, n_units]
            
            self.precomp_fwds = precomp_fwds
            self.store_states = store_states
            self.return_states = return_states
            
            self.W_fwd = OrderedDict(zip(self.lstm_inlets, [W[0] for W in [W_ci, W_ig, W_og, W_fg]]))
            self.W_bwd = OrderedDict(zip(self.lstm_inlets, [W[1] for W in [W_ci, W_ig, W_og, W_fg]]))
            
            self.W_fwd_conc = W_fwd_conc
            self.W_bwd_conc = W_bwd_conc
            self.a = OrderedDict(ci=a_ci, ig=a_ig, og=a_og, fg=a_fg, out=a_out)
            self.b = OrderedDict(ci=b_ci, ig=b_ig, og=b_og, fg=b_fg)
            self.h = h
            self.c = c
            self.comb = comb
            self.max_seq_len = None
            self.external_rec = None
            
            self.dilation_rate = dilation_rate
            
            self.out = tf.expand_dims(h_init, 1)
            self.name = name
    
    def add_external_recurrence(self, incoming):
        """Use this Layer as recurrent connections for the conLSTM instead of convLSTM hidden activations
        
        Don't forget to make sure that the bias and weight shapes fit to new shape!
        
        Parameters
        -------
        incoming : layer class, tensorflow tensor or placeholder
            Incoming external recurrence for convLSTM layer as layer class or tensor of shape
            (samples, 1, array_x, array_y, features)
        
        Example
        -------
        >>> # Example for convLSTM that uses its own hidden state and the output of a higher convolutional layer as
        >>> # recurrent connections
        >>> conv_lstm = ConvLSTMLayer(...)
        >>> conv_1 = ConvLayer(incoming=conv_lstm, ...)
        >>> modified_recurrence = ConcatLayer(conv_lstm, conv_1)
        >>> conv_lstm.add_external_recurrence(modified_recurrence)
        >>> conv_lstm.get_output()
        """
        self.external_rec, _ = get_input(incoming)
    
    def comp_net_fwd(self, incoming, start_at=0):
        """Compute and yield relative sequence position and net_fwd per real sequence position
        
        Compute net_fwd as specified by precomp_fwds (precompute it for all inputs or compute it online for each
        seq_pos) for incoming with shape: (samples, sequence positions, array_x, array_y, features);
        
        Parameters
        -------
        incoming : tensor
            Incoming layer
        start_at : int
            Start at this sequence position in the input sequence; If negative, the unknown sequence positions
            at the beginning will be padded with the first sequence position:
        
        Returns
        -------
        seq_pos : int
            Relative sequence position (sarting at 0)
        net_fwd : tf.tensor
            Yields the net_fwd at the current sequence position with shape (samples, array_x, array_y, features)
        """
        precomp_fwds = self.precomp_fwds
        W_fwd = self.W_fwd_conc
        
        if precomp_fwds:
            # Precompute convolution over all sequence positions for forward inputs
            net_fwd_precomp = conv2d(incoming, W_fwd, dilation_rate=self.dilation_rate, name="net_fwd_precomp")
        else:
            net_fwd_precomp = None
        
        # loop through sequence positions and return respective net_fwd
        for relative_seq_pos, real_seq_pos in enumerate(range(start_at, self.incoming_shape[1])):
            # If first sequence positions are unknown, pad beginning of sequence with first sequence position
            if real_seq_pos < 0:
                index_seq_pos = 0
            else:
                index_seq_pos = real_seq_pos
            
            # Yield the relative sequence position and net_fwd by either indexing the precomputed net_fwd_precomp or
            # calculating the net_fwd online per sequence position
            if precomp_fwds:
                cur_net_fwd = net_fwd_precomp[:, index_seq_pos, :]
            else:
                cur_net_fwd = conv2d(incoming[:, index_seq_pos, :], W_fwd, dilation_rate=self.dilation_rate,
                                     name="net_fwd")
            
            yield relative_seq_pos, cur_net_fwd
    
    def get_output_shape(self):
        """Return shape of output"""
        # Get shape of output tensor(s), which will be [samples, n_seq_pos+n_tickersteps, x, y, n_units]
        return [self.incoming_shape[0], -1] + self.incoming_shape[2:-1] + [self.n_units]
    
    def get_output(self, prev_layers=None, max_seq_len=None, tickersteps=0,
                   tickerstep_nodes=False, comp_next_seq_pos=True, **kwargs):
        """Calculate and return output of layer

        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        max_seq_len : int or None
            Can be used to artificially hard-clip the sequences at length max_seq_len
        tickersteps : int or None
            Tickersteps to apply after the sequence end; Tickersteps use 0 input and a trainable bias; not suitable for
            variable sequence lengths;
        tickerstep_nodes : bool
            True: Current sequence positions will be treated as ticker steps (tickerstep-bias is added to activations)
        comp_next_seq_pos : bool
            True: Cell state and hidden state for next sequence position will be computed
            False: Return current hidden state without computing next sequence position
        """
        if self.W_tickers is None and tickerstep_nodes:
            raise AttributeError("Required additional tickerstep_nodes but no tickerstep nodes were defined when "
                                 "creating the ConvLSTMLayer! (see arguments tickerstep_biases and "
                                 "learn_tickerstep_biases")
        
        if prev_layers is None:
            prev_layers = list()
        if (self not in prev_layers) and comp_next_seq_pos:
            prev_layers += [self]
            self.compute_nsp(prev_layers=prev_layers, max_seq_len=max_seq_len, tickersteps=tickersteps,
                             tickerstep_nodes=tickerstep_nodes, **kwargs)
        if self.return_states:
            if len(self.h) == 1:
                self.out = tf.expand_dims(self.h[-1], 1)  # add empty dimension for seq_pos
            else:
                self.out = tf.stack(self.h[1:], axis=1)  # pack states but omit initial state
        else:
            self.out = tf.expand_dims(self.h[-1], 1)  # add empty dimension for seq_pos
        return self.out
    
    def compute_nsp(self, prev_layers=None, max_seq_len=None, tickersteps=0, tickerstep_nodes=False, **kwargs):
        """
        Computes next sequence position
        
        Parameters
        -------
        prev_layers : list of Layer or None
            List of layers that have already been processed (i.e. whose outputs have already been (re)computed and/or
            shall not be computed again)
        max_seq_len : int or None
            optional limit for number of sequence positions in input (will be cropped at end)
        tickersteps : int
            Number of tickersteps to run after final position
        tickerstep_nodes : bool
            Activate tickerstep input nodes? (True: add tickerstep-bias to activations)
        """
        incoming = self.incoming(prev_layers=prev_layers, max_seq_len=max_seq_len, tickersteps=tickersteps,
                                 tickerstep_nodes=tickerstep_nodes, comp_next_seq_pos=True, **kwargs)
        
        external_rec = None
        if self.external_rec is not None:
            external_rec = self.external_rec(prev_layers=prev_layers, max_seq_len=max_seq_len, tickersteps=tickersteps,
                                             tickerstep_nodes=tickerstep_nodes, comp_next_seq_pos=True,
                                             **kwargs)[:, -1, :]
        
        with tf.variable_scope(self.name) as scope:
            
            act = OrderedDict(zip_longest(self.lstm_inlets, [None]))
            
            # Make sure tensorflow can reuse the variable names
            scope.reuse_variables()
            
            # Handle restriction on maximum sequence length
            if max_seq_len is not None:
                incoming = incoming[:, :max_seq_len, :]
            
            #
            # Compute LSTM cycle at each sequence position in 'incoming'
            #
            
            # Loop through sequence positions and get corresponding net_fwds
            for seq_pos, net_fwd in self.comp_net_fwd(incoming):
                # Get previous output/hidden state
                h_prev = self.h[-1]
                
                # Reduce nr of recurrent features
                if self.reduce_recurrents is not None:
                    h_prev = self.reduce_recurrents(h_prev)
                
                # Calculate net for recurrent connections at current sequence position
                if self.external_rec is None:
                    net_bwd = conv2d(h_prev, self.W_bwd_conc, dilation_rate=self.dilation_rate, name="net_bwd")
                else:
                    net_bwd = conv2d(external_rec, self.W_bwd_conc, dilation_rate=self.dilation_rate, name="net_bwd")
                
                # Combine net from forward and recurrent connections
                if self.comb == 'mul':
                    # TODO: Implement correct version of multiplication combination
                    act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=3, num_or_size_splits=4,
                                                                          value=net_fwd * net_bwd, name="net_input")
                elif self.comb == 'add':
                    act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=3, num_or_size_splits=4,
                                                                          value=net_fwd + net_bwd, name="net_input")
                else:
                    raise ValueError("Combination method {} unknown".format(self.comb))
                
                # Calculate activations
                if tickerstep_nodes and (self.W_tickers is not None):
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g] + self.W_tickers[g])
                                                             for g in self.lstm_inlets]))
                else:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g])
                                                             for g in self.lstm_inlets]))
                
                # ci, ig, og, fg = [self.a['ci'](ci + self.b['ci']), self.a['ig'](ig + self.b['ig']),
                #                   self.a['og'](og + self.b['og']), self.a['fg'](fg + self.b['fg'])]
                
                # Calculate new cell state
                if self.store_states:
                    self.c.append(act['ci'] * act['ig'] + self.c[-1] * act['fg'])
                else:
                    self.c[-1] = act['ci'] * act['ig'] + self.c[-1] * act['fg']
                
                # Calculate new output with new cell state
                if self.store_states:
                    self.h.append(self.a['out'](self.c[-1]) * act['og'])
                else:
                    self.h[-1] = self.a['out'](self.c[-1]) * act['og']
            
            # Process tickersteps
            for _ in enumerate(range(tickersteps)):
                # The forward net input during the ticker steps is 0 (no information is added anymore)
                # ticker_net_fwd = 0
                
                # Get previous output/hidden state
                h_prev = self.h[-1]
                
                # Reduce nr of recurrent features
                if self.reduce_recurrents is not None:
                    h_prev = self.reduce_recurrents(h_prev)
                
                # Calculate net for recurrent connections at current sequence position
                if self.external_rec is None:
                    net_bwd = conv2d(h_prev, self.W_bwd_conc, dilation_rate=self.dilation_rate, name="net_bwd")
                else:
                    net_bwd = conv2d(external_rec, self.W_bwd_conc, dilation_rate=self.dilation_rate, name="net_bwd")
                
                # Combine net from forward and recurrent connections
                if self.comb == 'mul':
                    act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=3, num_or_size_splits=4, value=net_bwd,
                                                                          name="net_input")
                elif self.comb == 'add':
                    act['ci'], act['ig'], act['og'], act['fg'] = tf.split(axis=3, num_or_size_splits=4, value=net_bwd,
                                                                          name="net_input")
                else:
                    raise ValueError("Combination method {} unknown".format(self.comb))
                
                # Calculate activations including ticker steps
                if self.W_tickers is not None:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g] + self.W_tickers[g])
                                                             for g in self.lstm_inlets]))
                else:
                    act = OrderedDict(zip(self.lstm_inlets, [self.a[g](act[g] + self.b[g])
                                                             for g in self.lstm_inlets]))
                
                # ci, ig, og, fg = [a_ci(ci + b_ci + W_tci), a_ig(ig + b_ig + W_tig), a_og(og + b_og + W_tog),
                #                   a_fg(fg + b_fg + W_tfg)]
                
                # Calculate new cell state
                if self.store_states:
                    self.c.append(act['ci'] * act['ig'] + self.c[-1] * act['fg'])
                else:
                    self.c[-1] = act['ci'] * act['ig'] + self.c[-1] * act['fg']
                
                # Calculate new output with new cell state
                if self.store_states:
                    self.h.append(self.a['out'](self.c[-1]) * act['og'])
                else:
                    self.h[-1] = self.a['out'](self.c[-1]) * act['og']
    
    def get_weights(self):
        """Return list with all layer weights"""
        if self.W_tickers is not None:
            return [w for w in [self.W_fwd_conc, self.W_bwd_conc, self.W_red_rec] + list(self.W_tickers.values())
                    if w is not None]
        else:
            return [w for w in [self.W_fwd_conc, self.W_bwd_conc, self.W_red_rec] if w is not None]
    
    def get_biases(self):
        """Return list with all layer biases"""
        return list(self.b.values())
    
    def get_plots_w(self, max_num_inp, max_num_out):
        """Prepare to plot weights from first 'max_num_inp' and 'max_num_out' features
        (W has shape [kx, ky, n_inputfeats, n_outputfeats])
        """
        plotsink = list()
        plot_dict = dict()
        plot_range_dict = dict()
        
        # Get all weights , W_fwd2=tf.split(3, 4, self.W_fwd_conc)
        weight_dict = OrderedDict(W_fwd=self.W_fwd, W_bwd=self.W_bwd, W_red_rec=self.W_red_rec,
                                  W_tickers=self.W_tickers)
        lstm_weights = [w for w in list(weight_dict.keys()) if (weight_dict[w] is not None) and (w is not 'W_tickers')]
        for weight_name in lstm_weights:
            weight = weight_dict[weight_name]
            # Open a plotsink entry for each weight
            plotsink.append([])
            
            # In each plotsink entry plot all inlets, specified input features, and specified output features
            if weight_name == 'W_red_rec':
                num_inp_f = min(weight.get_shape().as_list()[2], max_num_inp)
                num_out_f = min(weight.get_shape().as_list()[3], max_num_out)
                
                for out_f in range(num_out_f):
                    for inp_f in range(num_inp_f):
                        plot_dict['{}_{}_i{}_o{}'.format(self.name, weight_name, inp_f, out_f)] = \
                            weight[None, :, :, inp_f, out_f]
                        plotsink[-1].append('{}_{}_i{}_o{}'.format(self.name, weight_name, inp_f, out_f))
                continue
            
            for lstm_inlet in self.lstm_inlets:
                num_inp_f = min(weight[lstm_inlet].get_shape().as_list()[2], max_num_inp)
                num_out_f = min(weight[lstm_inlet].get_shape().as_list()[3], max_num_out)
                
                for out_f in range(num_out_f):
                    for inp_f in range(num_inp_f):
                        plot_dict['{}_{}_{}_i{}_o{}'.format(self.name, weight_name, lstm_inlet, inp_f, out_f)] = \
                            weight[lstm_inlet][None, :, :, inp_f, out_f]
                        plotsink[-1].append('{}_{}_{}_i{}_o{}'.format(self.name, weight_name, lstm_inlet, inp_f, out_f))
        
        return plot_dict, plotsink, plot_range_dict
    
    def get_plots_state(self):
        """Prepare to plot states"""
        plotsink = list()
        plot_dict = dict()
        plot_range_dict = dict()
        
        # Hidden and cell state
        if self.store_states:
            plot_dict['{}_h'.format(self.name)] = self.h
            plot_dict['{}_c'.format(self.name)] = self.c
        
        return plot_dict, plotsink, plot_range_dict
    
    def get_plots_out(self, sample=0, frames=slice(0, None)):
        """Prepare to plot outputs for sample 'sample' with 'frames' frames"""
        plotsink = list()
        plot_dict = dict()
        plot_range_dict = dict()
        
        # Prepare output for plotting (plot_dict value is [tensor, [min, max]]
        plot_dict['{}_out'.format(self.name)] = tf.arg_max(self.out[sample, frames, :, :, :], 3)
        plot_range_dict['{}_out'.format(self.name)] = [0, self.n_units]
        plotsink.append(['{}_out'.format(self.name)])
        
        return plot_dict, plotsink, plot_range_dict


class AdditiveNoiseLayer(Layer):
    def __init__(self, incoming, noisefct=tf.random_normal, noiseparams=None, backprop_noise=False,
                 name='AdditiveNoiseLayer'):
        """ Noise layer which adds noise to the incoming tensor
        
        Adds tensor created by noisefct to the incoming layer. Evaluation of noisefct is done dynamically, i.e.
        noisefct is called each time the network output is computed via get_output().

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input;
        noisefct : function
            Function for generating tensor/variable containing the noise of form noisefct(shape, **kwargs);
        noiseparams : dict or None
            Dictionary to use as kwargs for noisefct;
        backprop_noise : bool
            True: Enable backpropagation through noisefct
            False: Clip backpropagation at noisefct
            
        """
        super(AdditiveNoiseLayer, self).__init__()
        
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.out = None
            self.name = name
            self.backprop_noise = backprop_noise
            self.noisefct = noisefct
            if noiseparams is not None:
                self.noiseparams = noiseparams
            else:
                self.noiseparams = dict()
    
    def get_output_shape(self):
        return self.incoming_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        """
        
        noise = self.noisefct(shape=tf.shape(self.incoming()), **self.noiseparams)
        
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.backprop_noise:
                    self.out = incoming + noise
                else:
                    self.out = incoming + tf.stop_gradient(noise)
        
        return self.out


class MultiplicativeNoiseLayer(Layer):
    def __init__(self, incoming, noisefct=tf.random_normal, noiseparams=None, backprop_noise=False,
                 name='MultiplicativeNoiseLayer'):
        """ Layer which uses the the incoming tensor to gate noise.

        Uses tensor created by noisefct to gate the incoming layer. Evaluation of noisefct is done dynamically, i.e.
        noisefct is called each time the network output is computed via get_output().

        Parameters
        -------
        incoming : layer, tensorflow tensor, or placeholder
            Input;
        noisefct : function
            Function for generating tensor/variable containing the noise of form noisefct(shape, **kwargs);
        noiseparams : dict or None
            Dictionary to use as kwargs for noisefct;
        backprop_noise : bool
            True: Enable backpropagation through noisefct
            False: Clip backpropagation at noisefct

        """
        super(MultiplicativeNoiseLayer, self).__init__()
        
        with tf.variable_scope(name) as self.layer_scope:
            self.incoming, self.incoming_shape = get_input(incoming)
            
            self.out = None
            self.name = name
            self.backprop_noise = backprop_noise
            self.noisefct = noisefct
            if noiseparams is not None:
                self.noiseparams = noiseparams
            else:
                self.noiseparams = dict()
    
    def get_output_shape(self):
        return self.incoming_shape
    
    def get_output(self, prev_layers=None, **kwargs):
        """Calculate and return output of layer
        """
        
        noise = self.noisefct(shape=tf.shape(self.incoming()), **self.noiseparams)
        
        if prev_layers is None:
            prev_layers = list()
        if self not in prev_layers:
            prev_layers += [self]
            incoming = self.incoming(prev_layers=prev_layers, **kwargs)
            with tf.variable_scope(self.layer_scope):
                if self.backprop_noise:
                    self.out = incoming * noise
                else:
                    self.out = incoming * tf.stop_gradient(noise)
        
        return self.out
