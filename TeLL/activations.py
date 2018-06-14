# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Functions for regularization and convenience wrappers for tensorflow regularization functions

"""
import tensorflow as tf

'''
Tensorflow Implementation of the Scaled ELU function and Dropout
'''


def selu(x):
    """ When using SELUs you have to keep the following in mind:
    # (1) scale inputs to zero mean and unit variance
    # (2) use SELUs
    # (3) initialize weights with stddev sqrt(1/n)
    # (4) use SELU dropout
    """
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def selu_offset(offset=None):
    """selu with constant offset, offset defaults to alpha*scale, i.e. only positive activations"""
    with tf.name_scope('selu_offset'):
        alpha = 1.6732632423543772848170429916717+1e-7
        scale = 1.0507009873554804934193349852946
        if offset is None:
            offset = tf.constant(alpha*scale, dtype=tf.float32)
        return lambda *args, **kwargs: selu(*args, **kwargs) + offset
