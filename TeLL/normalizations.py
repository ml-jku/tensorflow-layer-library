# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017
Functions for normalization

"""

import tensorflow as tf


def max_norm(tensor):
    """Simple normalization by maximum"""
    maximum = tf.reduce_max(tf.abs(tensor))
    tensor /= maximum
    return tensor


def max_norm_all_tensors(tensor_list, clip: bool = True, max_val=tf.constant(1.0)):
    """Normalization of list of tensors by maximum of tensors"""
    maxima = [tf.reduce_max(tf.abs(tensor)) for tensor in tensor_list]
    maxima = tf.stack(maxima)
    if clip:
        maximum = tf.reduce_max(maxima) + 1e-16
    else:
        maximum = tf.reduce_max(maxima)
    return [tf.divide(tensor, maximum) * max_val for tensor in tensor_list]


def euclid_norm(tensor):
    """Normalization by euclidean distance"""
    summation = tf.reduce_sum(tf.square(tensor))
    tensor /= tf.sqrt(summation)
    return tensor
