# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Functions for initializing tensorflow variables and wrappers for tensorflow initializers

"""

import numpy as np
import tensorflow as tf


def constant(shape=None, init=0.1, dtype=tf.float32):
    if isinstance(init, (float, int)):
        initial = tf.constant(init, shape=shape, dtype=dtype)
    else:
        initial = tf.constant(init, dtype=dtype)
    
    return initial


def orthogonal(shape, gain=np.sqrt(2)):
    """ orthogonal initialization method

     Parameters
     -------
     shape : array
        the shape of the weight matrix. Frist dimension contains the width of the layer below.
     """
    return tf.initializers.orthogonal(gain=gain)(shape=shape)


def scaled_elu_initialization(shape, truncated=True):
    """ Preferred variable initialization method for the scaled ELU activation function.

     Parameters
     -------
     shape : array
        the shape of the weight matrix. Frist dimension contains the width of the layer below.
     truncated : boolean
        Whether the truncated normal distribution should be used.
     """
    if len(shape) == 4:
        f_in = int(np.prod(shape[:-1]))
    else:
        f_in = shape[0]
    
    if truncated:
        return tf.truncated_normal(shape=shape, stddev=tf.cast(tf.sqrt(1 / f_in), tf.float32))
    else:
        return tf.random_normal(shape=shape, stddev=tf.cast(tf.sqrt(1 / f_in), tf.float32))


def scaled_elu_initialization_rec(shape, truncated=True):
    """ Preferred variable initialization method for the scaled ELU activation function.

     Parameters
     -------
     shape : array
        the shape of the weight matrix. Frist dimension contains the width of the layer below.
     truncated : boolean
        Whether the truncated normal distribution should be used.
     """
    if len(shape) == 4:
        f_in = int(np.prod(shape[:-1]))
    else:
        f_in = shape[0]
    
    if truncated:
        return tf.truncated_normal(shape=shape, stddev=tf.cast(tf.sqrt(1 / f_in / 10), tf.float32))
    else:
        return tf.random_normal(shape=shape, stddev=tf.cast(tf.sqrt(1 / f_in / 10), tf.float32))


def weight_klambauer_elu(shape, seed=None):
    """ Preferred variable initialization method for the non-scaled ELU activation function.

    Parameters
    -------
    shape : array
        the shape of the weight matrix. First dimension contains the width of the layer below.
    seed : integer
        seed for the initialization
    """
    
    klambauer_constat = 1.5505188080679277
    initial = tf.contrib.layers.variance_scaling_initializer(factor=klambauer_constat, mode='FAN_IN',
                                                             uniform=False, seed=seed, dtype=tf.float32)
    return initial(shape)


def gaussian(x, mu, std):
    return (1. / (np.sqrt(2 * np.pi) * std)) * (np.exp(- (x - mu) * (x - mu) / (2 * std * std)))


def weight_xavier(shape, uniform=False, seed=None):
    initial = tf.contrib.layers.xavier_initializer(uniform=uniform, seed=seed, dtype=tf.float32)
    return initial(shape)


def weight_xavier_conv2d(shape, uniform=False, seed=None):
    initial = tf.contrib.layers.xavier_initializer_conv2d(uniform=uniform, seed=seed, dtype=tf.float32)
    return initial(shape)


def weight_he(shape, seed=None):
    initial = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=seed,
                                                             dtype=tf.float32)
    return initial(shape)


weight_he_conv2d = weight_he


def weight_truncated_normal(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return initial


def weight_gauss_conv2d(shape, dtype=np.float32):
    # Use gauss PDF with center at middle of axes to initialize weights
    x = np.arange(np.ceil(shape[0] / 2))
    if (shape[0] % 2) == 0:
        x = np.append(x, x[::-1])
    else:
        x = np.append(x, x[-2::-1])
    
    p_x = gaussian(x, mu=np.ceil(shape[0] / 2), std=np.ceil(shape[0] / 2) / 4)
    
    y = np.arange(np.ceil(shape[1] / 2))
    if (shape[1] % 2) == 0:
        y = np.append(y, y[::-1])
    else:
        y = np.append(y, y[-2::-1])
    
    p_y = gaussian(y, mu=np.ceil(shape[1] / 2), std=np.ceil(shape[1] / 2) / 4)
    
    p_xy = np.outer(p_x, p_y)
    
    W = np.zeros(shape, dtype=dtype)
    W[:, :, :, :] = p_xy[:, :, None, None]
    
    return constant(init=W)
