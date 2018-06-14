# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2018

Misc. objects that import tensorflow
"""
import tensorflow as tf
import TeLL.layers
from TeLL.utility.misc import get_rec_attr, import_object


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def layers_from_specs(incoming, layerspecs, verbose=True):
    """Create a single-branch-network from a list of layer specifications; beta-state convenience function;

    Create a network from a list of layer specifications (dictionaries); does only work for simple
    single-branch-networks, for more flexible designs, please create your network directly; layer in list will be
    created one-by-one, each with previous layer as input;
    If weight initializers ('W' or 'weight_initializer') are iterables with length>1, the first element is treated as
    function and the others as arguments for the function; otherwise, weight initializers is treated as function;
    This function has not been tested for all layers, use carefully/adapt to your needs;


    Parameters
    ----------
    incoming : layer, tensorflow tensor, or placeholder
        Incoming layer
    layerspecs : list of dicts or dict-likes
        Network design as list of dictionaries, where each dict represents the layer parameters as kwargs and requires
        the additional key "type" with a string value that is the TeLL layer-class name.

    Returns
    ----------
    list of TeLL Layer objects
        Network as list of connected TeLL layers

    Example
    -------
    >>> # Creating a network containing a convolutional layer, followed by maxpooling and another convolutional layer
    >>> batchsize, x_dim, y_dim, channels = (5, 28, 28, 3)
    >>> network_input = tf.placeholder(shape=(batchsize, x_dim, y_dim, channels), dtype=tf.float32)
    >>> layerspecs = [{"type": "ConvLayer", "name": "c1", "num_outputs": 32, "ksize": 3, "a": "tensorflow.nn.relu",
    >>>                "weight_initializer": "0.1:tensorflow.orthogonal_initializer"},
    >>>                {"type": "MaxPoolingLayer", "name": "mp1", "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]},
    >>>                {"type": "ConvLayer", "name": "c2", "num_outputs": 64, "ksize": 3, "a": "tensorflow.nn.relu",
    >>>                "weight_initializer": "tensorflow.orthogonal_initializer"}]
    >>> tell_network = layers_from_specs(incoming=network_input, layerspecs=layerspecs)
    >>> output_tensor = tell_network[-1].get_output()
    """
    layers = [incoming]
    for l_i, layerspec in enumerate(layerspecs):
        layertype = layerspec['type']
        if verbose:
            print("\t\tbuilding {}...".format(layertype), end='')
        layer = get_rec_attr(TeLL.layers, layertype)
        
        # Replace strings with corresponding functions and remove the "type" field
        layerspec = layerspec.copy()
        if 'a' in layerspec.keys():
            layerspec['a'] = import_object(layerspec['a'])
        if 'W' in layerspec.keys():
            W = layerspec['W'].split(':')
            if len(W) == 1:
                layerspec['W'] = import_object(W[0])()
            else:
                winit_fct = import_object(W[-1])
                winit_args = [float(a) for a in W[:-1]]
                layerspec['W'] = winit_fct(*winit_args)
        
        if 'weight_initializer' in layerspec.keys():
            weight_initializer = layerspec['weight_initializer'].split(':')
            if len(weight_initializer) == 1:
                layerspec['weight_initializer'] = import_object(weight_initializer[0])()
            else:
                winit_fct = import_object(weight_initializer[-1])
                winit_args = [float(a) for a in weight_initializer[:-1]]
                layerspec['weight_initializer'] = winit_fct(*winit_args)
        
        if layerspec.get('name', None) is None:
            layerspec['name'] = "{}_{}".format(l_i, layerspec['type'])
        del layerspec['type']
        
        layers.append(layer(layers[-1], **layerspec))
        if verbose:
            print(" in {} / out {}".format(layers[-2].get_output_shape(), layers[-1].get_output_shape()))
    return layers[1:]


def tensor_shape_with_flexible_dim(tensor, dim):
    """Return shape of tensor with dimension dim as a flexible dimension

    Parameters
    ----------
    tensor : tensor
        Tensor of which to return shape with a flexible dimension

    dim : int
        Index of dimension to replace with a flexible dimension
    """
    shape_list = tensor.shape.as_list()
    shape_list = [s if s_i != dim else None for s_i, s in enumerate(shape_list)]
    return tf.TensorShape(shape_list)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class TriangularValueEncoding(object):
    def __init__(self, max_value: int, triangle_span: int):
        """Encodes an integer value with range [0, max_value] as multiple activations between 0 and 1 via triangles of
        width triangle_span;

        LSTM profits from having an integer input with large range split into multiple input nodes; This class encodes
        an integer as multiple nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the integer range such that 2 triangles
        overlap by 1/2 width and the whole integer range is covered; For each integer to encode, the high of the
        triangle at this integer position is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each
        integer value;

        Values are encoded via encode_value(value) and returned as float32 tensorflow tensor of length self.n_nodes;

        Parameters
        ----------
        max_value : int
            Maximum value to encode
        triangle_span : int
            Width of each triangle
        """
        # round max_value up to a multiple of triangle_span
        if max_value % triangle_span != 0:
            max_value = ((max_value // triangle_span) + 1) * triangle_span
        
        # Calculate number of overlapping triangle nodes
        n_nodes_half = int(max_value / triangle_span)
        n_nodes = n_nodes_half * 2 + 1
        
        # Template for tensor
        coding = tf.zeros((n_nodes,), dtype=tf.float32, name='ingametime')
        
        self.n_nodes_python = n_nodes
        self.n_nodes = tf.constant(n_nodes, dtype=tf.int32)
        self.n_nodes_half = tf.constant(n_nodes_half, dtype=tf.int32)
        self.max_value = tf.constant(max_value, dtype=tf.int32)
        self.triangle_span = tf.constant(triangle_span, dtype=tf.int32)
        self.triangle_span_float = tf.constant(triangle_span, dtype=tf.float32)
        self.half_triangle_span = tf.cast(self.triangle_span / 2, dtype=tf.int32)
        self.half_triangle_span_float = tf.cast(self.triangle_span / 2, dtype=tf.float32)
        self.coding = coding
    
    def encode_value(self, value):
        """Encode value as multiple triangle node activations
        
        Parameters
        ----------
        value : int tensor
            Value to encode as integer tensorflow tensor
        
        Returns
        ----------
        float32 tensor
            Encoded value as float32 tensor of length self.n_nodes
        """
        value_float = tf.cast(value, dtype=tf.float32)
        
        index = tf.cast(value / self.triangle_span, dtype=tf.int32)
        act = (tf.constant(0.5, dtype=tf.float32)
               - (tf.mod(tf.abs(value_float - self.half_triangle_span_float), self.triangle_span_float)
                  / self.triangle_span_float)) * tf.constant(2, dtype=tf.float32)
        coding1 = tf.one_hot(index, on_value=act, depth=self.n_nodes, dtype=tf.float32)
        
        index = tf.cast((value + self.half_triangle_span) / self.triangle_span, dtype=tf.int32) + self.n_nodes_half
        act = (tf.mod(tf.abs(value_float - self.half_triangle_span_float), self.triangle_span_float)
               / self.triangle_span_float) * tf.constant(2, dtype=tf.float32)
        coding2 = tf.one_hot(index, on_value=act, depth=self.n_nodes, dtype=tf.float32)
        
        return coding1 + coding2
