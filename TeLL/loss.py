# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017
Functions for computations of losses

"""

import tensorflow as tf
import numpy as np
from collections import OrderedDict

from TeLL.initializations import weight_gauss_conv2d
from TeLL.dataprocessing import gaussian_blur


def image_crossentropy(pred, target, class_weights=None, loss_slack=0, reduce_by='debug', calc_statistics=False,
                       incl_histograms=False, pixel_weights=None):
    """
    Calculate the pixelwise cross-entropy loss.

    Parameters
    -----
    pred : tensorflow tensor
        Predicted classes per pixel as tensor of shape(samples, x, y, features) or
        (samples, sequence_positions, x, y, features); features are the output neurons/features of e.g. a conv. layer
    target : tensorflow tensor shape=(samples, x, y)
        Target classes per pixel as tensor of shape(samples, x, y) or (samples, sequence_positions, x, y);
    class_weights : numpy array shape=(n_classes) or None
        Factors for weighting the classes
    loss_slack : float in range [0, 1]
        Ignore error measured w.r.t. the softmax output (range [0, 1]) and the true labels (range [0, 1]), i.e.
        ce_loss *= ce_loss > tf.log(1-loss_slack); loss_slack should be in range [0, 1];
    reduce_by : string ("mean", "sum", "weighted_sum")
        Reduce losses for each pixel to scalar by either mean (sum/n_pixels), sum, or weighted sum (sum/weighted_pixels)
    calc_statistics : bool
        If False: return tuple with (loss, {})
        If True: return tuple with (loss, loss_statistics), where loss_statistics is a OrderedDict with various tensors
        for loss evaluation
    incl_histograms : bool
        If True: also include histograms in statistics
    pixel_weights : tensor or None
        Float32 tensor holding weights for the individual pixels; Has same shape as target;
    
    Returns
    -----
    Tuple with cross-entropy loss as tensorflow scalar and optional statistics (see calc_statistics)

    """
    
    with tf.variable_scope('crossentropy_loss') as scope:
        scope.reuse_variables()
        loss_statistics = OrderedDict()
        
        # Select predictions where true label != 0
        # flatten label matrix
        target_classes = tf.reshape(target, shape=(-1,))

        if pixel_weights is not None:
            target_weights = tf.reshape(pixel_weights, shape=(-1,))
        else:
            target_weights = tf.ones(target_classes.shape)
        
        n_pixels = target_classes.get_shape()[0]
        n_classes = pred.get_shape()[-1]
        # generate element-wise indices for selection from flattened matrix
        inds_mult = tf.range(start=0, limit=n_pixels * n_classes, delta=n_classes)
        target_inds = target_classes + inds_mult
        
        #
        # Calculate numerically stable log-softmax (=cross-entropy softmax)
        #
        
        # gather relevant predictions (i.e. predictions at target index)
        target_pred = tf.gather(tf.reshape(pred, shape=(-1,)), target_inds)
        
        # Reshape prediction matrix
        pred_reshaped = tf.reshape(pred, shape=(tf.reduce_prod(pred.get_shape()[:-1]), -1))
        
        # Calc log-softmax
        if loss_slack > 0:
            # Calculate log-softmax at once (more stable + faster)
            log_softmax = target_pred - tf.reduce_logsumexp(pred_reshaped, axis=[1])
            # Apply slack
            log_softmax *= tf.cast(log_softmax < tf.log(1.0 - loss_slack), dtype=tf.float32)
        else:
            # Calculate log-softmax at once (more stable + faster)
            log_softmax = target_pred - tf.reduce_logsumexp(pred_reshaped, axis=[1])
        
        #
        # Apply pixel-wise weighting
        #
        if pixel_weights is not None:
            log_softmax *= target_weights
        
        # Calculate cross-entropy loss
        if class_weights is None:
            ce_loss = log_softmax
        else:
            # turn class_weights into tensorflow tensor
            class_weights = tf.constant(class_weights, dtype=tf.float32)
            # multiply cross-entropies elementwise with corresponding class weigth
            class_weights_pixel = tf.gather(class_weights, target_classes)
            ce_loss = log_softmax * class_weights_pixel
            
            if calc_statistics:
                for cl in range(n_classes):
                    class_mask = tf.equal(target_classes, tf.constant(cl))
                    pixel_ce_per_class = -tf.boolean_mask(log_softmax, class_mask)
                    if incl_histograms:
                        loss_statistics['pixel ce class {}'.format(cl)] = pixel_ce_per_class
                    loss_statistics['mean ce class {}'.format(cl)] = (tf.reduce_sum(pixel_ce_per_class) /
                                                                      tf.reduce_sum(target_weights))
            
            if incl_histograms:
                loss_statistics['ce per pixel'] = -ce_loss
            loss_statistics['weighted sum ce'] = -tf.reduce_sum(ce_loss) / tf.reduce_sum(class_weights_pixel *
                                                                                         target_weights)
        
        loss_statistics['mean ce'] = -tf.reduce_sum(ce_loss) / tf.reduce_sum(target_weights)
        
        # Return cross-entropy loss
        if calc_statistics:
            if reduce_by == 'mean':
                return loss_statistics['mean ce'], loss_statistics
            elif reduce_by == 'sum':
                return tf.reduce_sum(-ce_loss), loss_statistics
            elif reduce_by == 'weighted_sum':
                return loss_statistics['weighted sum ce'], loss_statistics
        else:
            if reduce_by == 'mean':
                return loss_statistics['mean ce'], {}
            elif reduce_by == 'sum':
                return tf.reduce_sum(-ce_loss), {}
            elif reduce_by == 'weighted_sum':
                return loss_statistics['weighted sum ce'], {}


def crossentropy_without_softmax(output, target):
    """Crossentropy between an output tensor and a target tensor.
    In contrast to tf.nn.softmax_crossentropy_with_logits expects probabilities (i.e. does not perform softmax).
    
    WARNING: Numerically not stable, use at your own risk!
     
    # Arguments
        output: A tensor resulting from a softmax.
        target: A tensor of the same shape as `output`.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, here we expect "probabilities".
    return - tf.reduce_sum(target * tf.log(output), reduction_indices=len(output.get_shape()) - 1)


def softmax(target, axis, name=None):
    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210

    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    with tf.name_scope(name, 'softmax', values=[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax

def blurred_cross_entropy(output, target, filter_size=11, sampling_range=3.5, pixel_weights=None):
    """
    Apply a Gaussian smoothing filter to the target probabilities (i.e. the one-hot 
    representation of target) and compute the cross entropy loss between softmax(output) 
    and the blurred target probabilities. 
    
    :param output: A rank-4 or rank-5 tensor with shape=(samples, [sequence_position,] x, y, num_classes) 
        representing the network input of the output layer (not activated)
    :param target: A rank-3 or rank-4 tensor with shape=(samples, [sequence_position,] x, y) representing 
        the target labels. It must contain int values in 0..num_classes-1. 
    :param filter_size: A length-2 list of int specifying the size of the Gaussian filter that will be 
        applied to the target probabilities. 
    :param pixel_weights: A rank-3 or rank-4 tensor with shape=(samples, [sequence_position,] x, y) 
        representing factors, that will be applied to the loss of the corresponding pixel. This can be 
        e.g. used to void certain pixels by weighting them to 0, i.e. suppress their error induction. 
    :return: A scalar operation representing the blurred cross entropy loss. 
    """
    # convert target to one-hot
    output_shape = output.shape.as_list()
    one_hot = tf.one_hot(target, output_shape[-1], dtype=tf.float32)

    if (len(output_shape) > 4):
        one_hot = tf.reshape(one_hot, [np.prod(output_shape[:-3])] + output_shape[-3:])

    # blur target probabilities
    #gauss_filter = weight_gauss_conv2d(filter_size + [output_shape[-1], 1])
    #blurred_target = tf.nn.depthwise_conv2d(one_hot, gauss_filter, [1, 1, 1, 1], 'SAME')
    blurred_target = gaussian_blur(one_hot, filter_size, sampling_range)

    if (len(output_shape) > 4):
        blurred_target = tf.reshape(blurred_target, output_shape)

    # compute log softmax predictions and cross entropy
    log_pred = output - tf.reduce_logsumexp(output, axis=[len(output_shape) - 1], keep_dims=True)

    # Apply pixel-wise weighting
    if pixel_weights is not None:
        log_pred *= pixel_weights

    cross_entropy = -tf.reduce_sum(blurred_target * log_pred, axis=[len(output_shape)-1])

    if pixel_weights is not None:
        loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(pixel_weights)
    else:
        loss = tf.reduce_mean(cross_entropy)

    return loss

def one_hot_patch(x, depth):
    # workaround by name-name
    sparse_labels = tf.reshape(x, [-1, 1])
    derived_size = tf.shape(sparse_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    outshape = tf.concat(axis=0, values=[tf.reshape(derived_size, [1]), tf.reshape(depth, [1])])
    return tf.sparse_to_dense(concated, outshape, 1.0, 0.0)


def iou_loss(pred, target, calc_statistics=False, pixel_weights=None):
    """
    Calculate the approximate IoU loss according to
    "Rahman, A., & Wang, Y. (2016). Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation.
    International Symposium on Visual Computing."
    {http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf}

    Parameters
    -----
    pred : tensorflow tensor
        Predicted classes per pixel in a tensor of shape(samples, x, y, features) or
        (samples, sequence_positions, x, y, features); Features are the output neurons/features of e.g. a conv. layer
        to be used for softmax calculation (i.e. linear output units);
    target : tensorflow tensor
        Target classes per pixel in tensor of shape(samples, x, y) or (samples, sequence_positions, x, y);
    calc_statistics : bool
        If False: return tuple with (loss, {})
        If True: return tuple with (loss, loss_statistics), where loss_statistics is a OrderedDict with various tensors
        for loss evaluation
    pixel_weights : tensor or None
        Float32 tensor holding weights for the individual pixels; Has same shape as target;

    Returns
    -----
    Tuple with IoU loss as tensorflow scalar and optional statistics (see calc_statistics)
    """
    
    # if pixel_weights is not None:
    #     # TODO: pixel_weights for iou loss
    #     raise ValueError("pixel_weights not implemented for iou loss yet; please create a feature request; sorry!")
    
    with tf.variable_scope('iou_loss') as scope:
        scope.reuse_variables()
        loss_statistics = OrderedDict()
        
        #
        # Make sure pred and target have correct shape
        #
        pred_shape = pred.get_shape().as_list()
        n_classes = pred_shape[-1]
        if len(pred_shape) == 5:
            pred = tf.reshape(pred, [-1] + pred_shape[2:])
        elif len(pred_shape) == 4:
            pass
        else:
            raise AttributeError("pred must have 4 or 5 dimensions but has shape {}".format(pred_shape))
        
        target_shape = target.get_shape().as_list()
        if len(target_shape) == 4:
            target = tf.reshape(target, [-1] + target_shape[2:])
            pixel_weights = tf.reshape(pixel_weights, [-1] + target_shape[2:])
        elif len(target_shape) == 3:
            pass
        else:
            raise AttributeError("target must have 3 or 4 dimensions but has shape {}".format(target_shape))
        
        # Calculate softmax of predictions
        pred_softmax = softmax(pred, 3, name="iou_loss_softmax")
        
        # Convert targets to one-hot encoding resulting in same shape as prediction (batch, x, y, features)
        # target_onehot = tf.one_hot(target, pred.get_shape()[-1], on_value=1.0, off_value=0.0, dtype=tf.float32)
        target_onehot = tf.reshape(one_hot_patch(tf.reshape(target, shape=(-1,)),
                                                 pred.get_shape()[-1]),
                                   shape=(tuple(target.get_shape().as_list()) + (pred.get_shape().as_list()[-1],)))
        
        # approximate intersection and union per class (Eq. 3 and 4)
        if pixel_weights is not None:
            intersection = tf.reduce_sum(pred_softmax * target_onehot * tf.expand_dims(pixel_weights, axis=-1),
                                         axis=[-3, -2])
            union = tf.reduce_sum(((pred_softmax + target_onehot) - (pred_softmax * target_onehot))
                                  * tf.expand_dims(pixel_weights, axis=-1), axis=[-3, -2])
        else:
            intersection = tf.reduce_sum(pred_softmax * target_onehot, axis=[-3, -2])
            union = tf.reduce_sum((pred_softmax + target_onehot) - (pred_softmax * target_onehot), axis=[-3, -2])
        
        # iou_per_sample_per_class has shape (samples, classes)
        iou_per_sample_per_class = tf.div(intersection, union)
        
        iou_per_sample = list()
        for i in range(0, target_shape[0]):
            nonzero = tf.is_finite(iou_per_sample_per_class[i])
            iou_per_sample.append(tf.reduce_mean(tf.boolean_mask(iou_per_sample_per_class[i], nonzero)))
        
        iou = tf.reduce_mean(tf.convert_to_tensor(iou_per_sample, dtype=tf.float32))
        
        # IoU Loss (Eq. 5)
        loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), iou)
        
        if calc_statistics:
            # Get IoU per class for evaluation purposes
            iou_per_class = tf.reduce_mean(iou_per_sample_per_class, axis=0)
            for cl in range(n_classes):
                loss_statistics['IoU class {}'.format(cl)] = iou_per_class[tf.constant(cl)]
            
            loss_statistics['IoU mean'] = iou
            return loss, loss_statistics
        else:
            return loss, {}
