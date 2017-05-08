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


def gaussian(x, mu, std):
    return (1. / (np.sqrt(2 * np.pi) * std)) * (np.exp(- (x - mu) * (x - mu) / (2 * std * std)))


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
