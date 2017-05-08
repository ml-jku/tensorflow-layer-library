# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Functions for regularization and convenience wrappers for tensorflow regularization functions

"""

import tensorflow as tf

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