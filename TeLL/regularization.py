# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Functions for regularization and convenience wrappers for tensorflow regularization functions

"""
import numbers

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

from .layers import get_input


def regularize(layers, l1=1e-6, l2=1e-3, regularize_weights=True, regularize_biases=True):
    """ Regularize weights and/or biases of given layers if they offer
        a function getWeigthts/getBiases respectively to retrieve them.
        
        Parameters
        -------
        layers : objects implementing getWeights and/or getBiases
            Array of layers to regularize
        l1 : float
            Weight of L1 regularization (default = 1e-6)
        l2 : float
            Weight of L2 regularization (default = 1e-3)
        regularize_weights : bool
            Regularize only layer weights (default = True)
        regularize_biases : bool
            Regularize only layer biases (default = True)
        
        Returns
        -------
        Returns combined regularization penalty.
    
    """
    penalty = 0
    for layer in layers:
        get_weights = getattr(layer, "get_weights", None)
        get_biases = getattr(layer, "get_biases", None)
        if regularize_weights and callable(get_weights):
            weights = get_weights()
            for w in weights:
                if l1 != 0:
                    penalty += l1 * tf.reduce_sum(tf.abs(w))
                if l2 != 0:
                    penalty += l2 * tf.nn.l2_loss(w)
        if regularize_biases and callable(get_biases):
            biases = get_biases()
            for b in biases:
                if l1 != 0:
                    penalty += l1 * tf.reduce_sum(tf.abs(b))
                if l2 != 0:
                    penalty += l2 * tf.nn.l2_loss(b)
    return penalty


def __covar_l1norm(hiddens, enum_dims, feature_dims, n_features):
    """ Compute the L1 norm of the covariance matrix of hiddens. """
    enum_dims = list(set(range(len(hiddens.shape.as_list()))) - set(feature_dims))
    centered = hiddens - tf.reduce_mean(hiddens, enum_dims, keep_dims=True)
    
    # quick-fix
    # data2d = tf.reshape(tf.transpose(centered, enum_dims + feature_dims), [-1, n_features])
    data2d = tf.reshape(centered, [-1, n_features])
    
    covar_scaled_matrix = tf.matmul(data2d, data2d, transpose_a=True)
    covar_l1norm = tf.reduce_sum(tf.abs(covar_scaled_matrix))
    covar_num = tf.to_float(tf.shape(data2d)[0])
    
    return covar_l1norm, covar_num


def decor_penalty(layer, labels, n_classes, feature_dims, weight=1e-3, is_one_hot=True):
    """ Compute layer's decorrelation penalty conditioned on labels.
        
        For every class in labels, this function computes the L1-norm of the covariance 
        matrix of the hiddens belonging to that class and returns the sum of these norms 
        multiplied by weight. 
        
        layer : The layer to regularize.
        labels : An integer tensor representing the labels of layer's output on which the 
            decorrelation should be conditioned. The first tf.rank(labels) dimensions 
            of labels and layer's output must match. If not one-hot encoded, the values of 
            labels must be in [0 .. n_classes-1]. 
        n_classes : The number of distinct classes in labels. 
        feature_dims : The dimensions (as list) of layer's output representing the features 
            which are to be decorrelated. 
        weight : A factor by which to multiply the final score. 
        is_one_hot : Whether the labels are one-hot or integer encoded. 
    """
    
    # TODO: maybe enable for a list of layers
    # TODO: make default arg for feature_dims
    
    get_hiddens, h_shape = get_input(layer)
    hiddens = get_hiddens()
    
    labels_rank = len(labels.shape.as_list())
    hiddens_rank = len(h_shape)
    
    if is_one_hot:
        labels_rank -= 1
    
    # ensure label dims come first, feature dims come later
    assert (all(labels_rank <= i and i < hiddens_rank for i in feature_dims))
    
    # initial score
    score = tf.zeros([1])
    denom = tf.zeros([1])
    
    # all non-feature dims enumerate the feature instances
    enum_dims = list(set(range(len(h_shape))) - set(feature_dims))
    
    # compute the total number of features
    n_features = 1
    for i in feature_dims:
        n_features *= h_shape[i]
    
    # make sure there was no -1 in the product
    assert (n_features > 0)
    
    for i in range(n_classes):
        # make boolean mask indicating membership to class i
        if is_one_hot:
            class_i_mask = tf.equal(labels[:, i], 1)
        else:
            class_i_mask = tf.equal(labels, i)
        # determine if class i is empty
        empty = tf.equal(tf.reduce_sum(tf.to_float(class_i_mask)), 0)
        
        # add _covar_l1norm to score if not empty
        l1norm, num = tf.cond(empty, lambda: (tf.zeros([1]), tf.zeros([1])), lambda: __covar_l1norm(
            tf.boolean_mask(hiddens, class_i_mask), enum_dims, feature_dims, n_features))
        
        score += l1norm
        denom += num
    
    score /= denom
    return tf.reduce_sum((score / tf.to_float(n_classes)) * weight)


def dropout_selu(x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""
    
    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        
        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        
        if tensor_util.constant_value(keep_prob) == 1:
            return x
        
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)
        
        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean, 2) + fixedPointVar)))
        
        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret
    
    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))
