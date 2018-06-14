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
                if class_weights is None:
                    raise ValueError("'weighted sum ce' not possible as reduce_by option if class_weights are None")
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


def blurred_cross_entropy(output, target, filter_size=11, sampling_range=3.5, pixel_weights=None, calc_statistics=False):
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

    if len(output_shape) > 4:
        one_hot = tf.reshape(one_hot, [np.prod(output_shape[:-3])] + output_shape[-3:])

    # blur target probabilities
    # gauss_filter = weight_gauss_conv2d(filter_size + [output_shape[-1], 1])
    # blurred_target = tf.nn.depthwise_conv2d(one_hot, gauss_filter, [1, 1, 1, 1], 'SAME')
    blurred_target = gaussian_blur(one_hot, filter_size, sampling_range)
    # although the blurred targets should theoretically be distributions (i.e. sum to 1), there
    # might be numerical inaccuracies due to fft convolution, so we ensure here, that our targets
    # sum to 1.
    blurred_target /= tf.reduce_sum(blurred_target, 3, keep_dims=True)

    if len(output_shape) > 4:
        blurred_target = tf.reshape(blurred_target, output_shape)

    if calc_statistics:
        # TODO: return!!!
        clipped_blurred_target = tf.clip_by_value(blurred_target, 1e-15, 1)
        entropy = tf.reduce_mean(-tf.reduce_sum(clipped_blurred_target * tf.log(clipped_blurred_target),
                                                axis=[len(output_shape)-1]))

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
        if len(target_shape) == 5:
            target = tf.reshape(target, [-1] + target_shape[2:-1])
            pixel_weights = tf.reshape(pixel_weights, [-1] + target_shape[2:-1])
        elif len(target_shape) == 4:
            target = tf.reshape(target, [-1] + target_shape[2:])
            pixel_weights = tf.reshape(pixel_weights, [-1] + target_shape[2:])
        elif len(target_shape) == 3:
            pass
        else:
            raise AttributeError("target must have 3 or 4 or 5 dimensions but has shape {}".format(target_shape))
        
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


# def lrp(r_j, z_j, z_i, w_ij, n=None, epsilon=0.001, delta=None, b_j=None):
#     """Layer-wise Relevance Propagation as proposed in
#     https: // arxiv.org / abs / 1706.07979, https://arxiv.org/abs/1706.07206, and
#     https://github.com/ArrasL/LRP_for_LSTM/blob/master/code/LSTM/LRP_linear_layer.py"""
#
#     if n is None:
#         n = w_ij.shape.as_list()[-1]
#
#     stabilizer = epsilon * tf.sign(z_j)
#     if delta is not None:
#         r_ij = (z_i * w_ij + (stabilizer + delta * b_j) / n) / (z_j + stabilizer)
#     else:
#         r_ij = (z_i * w_ij + stabilizer / n) / (z_j + stabilizer)
#
#     return r_ij * r_j


# def lrp_lstm(w_ci, h, c, incoming, r):
#     """lrp applied to TeLL LSTMLayer"""
#
#     r_h = lrp(r_j=r, z_j=h, z_i=c, w_ij=tf.constant(1), n=c.shape.as_list()[-1])
#     # sum over units
#     r_h = tf.reduce_sum(r_h, axis=-1)
#
#     r_c = lrp(r_j=r_h, z_j=c, z_i=incoming, w_ij=w_ci, n=incoming.shape.as_list()[-1])
#     r_c = tf.reduce_sum(r_c, axis=-1)
#
#     return r_h, r_c

class NonNegSpace(object):
    def __init__(self):
        pass

    @staticmethod
    def scalar_to_tuple(x):
        return tf.stack([tf.where(x > 0, x, tf.zeros_like(x)), tf.where(x < 0, -x, tf.zeros_like(x))], axis=0)

    @staticmethod
    def tuple_to_scalar(x):
        return x[0] - x[1]

    @staticmethod
    def abs(x):
        return tf.reduce_sum(x, axis=0)

    @staticmethod
    def sum(x):
        return tf.stack([tf.reduce_sum(x[0]), tf.reduce_sum(x[1])], axis=0)

    @staticmethod
    def additive_multiplication_rule(z, *args, **kwargs):
        return z * 0.5

    @staticmethod
    def z_rule(w, x):
        return NonNegSpace.scalar_to_tuple(w * x)

    @staticmethod
    def r_from_k(r_k, w_from_k, x):
        """
    
        Parameters
        -------
        r_k : tensor (2, 1)
            Relevance coming from upper layer neuron k
        w_from_k : tensor (2, n_lower_layer_units)
            Weights from lower layer to upper layer neuron k
        x : tensor (2, n_lower_layer_units)
            Activations coming from lower layer
        """
        z_from_k = NonNegSpace.z_rule(w_from_k, x)
        
        z = z_from_k / NonNegSpace.abs(NonNegSpace.sum(z_from_k))
        z = tf.where(tf.is_nan(z), tf.zeros_like(z), z)
        
        return NonNegSpace.abs(r_k) * z

    @staticmethod
    def lrp(r, w, x):
        """
    
        Parameters
        -------
        r : tensor (2, n_upper_layer_units)
            Relevance coming from upper layer
        w : tensor (n_lower_layer_units, n_upper_layer_units)
            Weights from lower layer to upper layer
        x : tensor (n_lower_layer_units)
            Activations coming from lower layer
        """
        n_upper_units = r.get_shape().as_list()[-1]
        r_from_upper_layer = []
        
        for k in range(n_upper_units):
            r_from_upper_layer.append(NonNegSpace.r_from_k(r_k=r[:, k], w_from_k=w[:, k], x=x))
        r_from_upper_layer = tf.stack(r_from_upper_layer, axis=-1)
        
        return tf.reduce_sum(r_from_upper_layer, axis=-1)

    @staticmethod
    def lrp_lstm_c(r_out, o, c, i, z, w_o, w_i, act_h, n_timesteps, multiplication_rule):
        """lrp applied to TeLL LSTMLayer
    
        Parameters
        -------
        r_out : tensor (batchsize, timesteps, units)
        o, c, i, z  : tensor (batchsize, timesteps, units)
        w_o, w_i : tensor (incoming, outgoing)
        act_h activation function after cell
        multiplication_rule : int
        0...50/50 rule; 1...proportional rule; 3...no multiplication rule, no relevance through recurrent gate connections
        """
        if multiplication_rule == 0:
            mul_rule = additive_multiplication_rule
        elif multiplication_rule == 1:
            mul_rule = proportional_multiplication_rule
        elif multiplication_rule == 3:
            mul_rule = None
        else:
            raise AttributeError("Only multiplication_rule 0 and 1 are implemented")
        
        # Initialize input gate and output gate relevance with 0
        r_from_o = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_o redistributed to the individual units in t-1
        r_from_i = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_i redistributed to the individual units in t-1
        r_cc = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_ct<-ct+1
        
        r_y = []
        r_cy = []  # r_ct<-yt
        r_o = []
        r_c = []
        r_zi = []
        r_z = []
        r_i = []
        ttt = []
        
        rev_timesteps = np.arange(n_timesteps)[::-1]
        for t in rev_timesteps:
            #
            # for time t
            #
            ttt.append(r_out[:, t, :])
            if mul_rule is None:
                r_y.append(r_out[:, t, :])
                r_cy.append(r_y[-1])
                r_o.append(tf.zeros_like(r_y[-1]))
                
                r_c.append(r_cy[-1] + r_cc[-1])
                
                r_zi.append(r_c[-1] * (i[:, t, :] * z[:, t, :] / c[:, t, :]))
                r_zi[-1] = tf.where(tf.is_nan(r_zi[-1]), tf.zeros_like(r_zi[-1]), r_zi[
                    -1])  # TODO: This only holds for all-positive case! Otherwise we will need to consider r_zi[-2] to assign either full R or 0
                
                r_z.append(r_zi[-1])
                r_i.append(tf.zeros_like(r_zi[-1]))
            
            else:
                r_y.append(r_out[:, t, :] + r_from_o[-1] + r_from_i[-1])
                r_cy.append(mul_rule(act_h(c[:, t, :]), o[:, t, :], r_y[-1], c_min, c_max, o_min, o_max))
                r_o.append(mul_rule(o[:, t, :], act_h(c[:, t, :]), r_y[-1], o_min, o_max, c_min, c_max))
                
                r_c.append(r_cy[-1] + r_cc[-1])
                
                r_zi.append(r_c[-1] * (i[:, t, :] * z[:, t, :] / c[:, t, :]))
                r_zi[-1] = tf.where(tf.is_nan(r_zi[-1]), tf.zeros_like(r_zi[-1]), r_zi[
                    -1])  # TODO: This only holds for all-positive case! Otherwise we will need to consider r_zi[-2] to assign either full R or 0
                
                r_z.append(mul_rule(z[:, t, :], i[:, t, :], r_zi[-1], z_min, z_max, i_min, i_max))
                r_i.append(mul_rule(i[:, t, :], z[:, t, :], r_zi[-1], i_min, i_max, z_min, z_max))
            
            if t > 0:
                #
                # distribute R to units through recurrent connections
                #
                r_from_o_t = lrp(r=r_o[-1], w=w_o, x=o[:, t - 1, :], x_min=o_min, alpha=alpha, beta=beta)
                r_from_o.append(r_from_o_t)
                
                r_from_i_t = lrp(r=r_i[-1], w=w_i, x=i[:, t - 1, :], x_min=i_min, alpha=alpha, beta=beta)
                r_from_i.append(r_from_i_t)
                
                #
                # for time t-1
                #
                r_cc.append(c[:, t - 1, :] / c[:, t, :] * r_c[-1])
                r_cc[-1] = tf.where(tf.is_nan(r_cc[-1]), tf.zeros_like(r_cc[-1]),
                                    r_cc[-1])  # TODO: This only holds for all-positive case!
        
        r_collection = dict(r_from_o=tf.stack(r_from_o, axis=1), r_from_i=tf.stack(r_from_i, axis=1),
                            r_cc=tf.stack(r_cc, axis=1), r_y=tf.stack(r_y, axis=1), r_cy=tf.stack(r_cy, axis=1),
                            r_o=tf.stack(r_o, axis=1), r_c=tf.stack(r_c, axis=1), r_zi=tf.stack(r_zi, axis=1),
                            r_z=tf.stack(r_z, axis=1), r_i=tf.stack(r_i, axis=1), ttt=tf.stack(ttt, axis=1))
        
        # Relevance is stored with reversed time dimension - correct it
        r_collection = dict((k, v[:, ::-1, :]) for k, v in r_collection.items())
        
        return r_collection['r_z'], r_collection


def proportional_multiplication_rule(x, y, z, x_min, x_max, y_min, y_max):
    
    r = z * ((x-x_min) / (x_max - x_min)) / (((x-x_min) / (x_max - x_min)) + ((y-y_min) / (y_max - y_min)))
    return tf.where(tf.is_nan(r), tf.zeros_like(r), r)


def additive_multiplication_rule(x, y, z, x_min, x_max, y_min, y_max):
    return z * 0.5


def z_rule(w, x, x_min):
    return w * (x - x_min)


def r_from_k(r_k, w_from_k, x, x_min, alpha, beta):
    """
    
    Parameters
    -------
    r_k : tensor (1)
        Relevance coming from upper layer neuron k
    w_from_k : tensor (n_lower_layer_units)
        Weights from lower layer to upper layer neuron k
    x : tensor (n_lower_layer_units)
        Activations coming from lower layer
    """
    z_from_k = z_rule(w_from_k, x, x_min)
    
    positive_z = tf.nn.relu(z_from_k) / tf.reduce_sum(tf.nn.relu(z_from_k))
    positive_z = tf.where(tf.is_nan(positive_z), tf.zeros_like(positive_z), positive_z)
    
    negative_z = (tf.clip_by_value(z_from_k, clip_value_min=-np.infty, clip_value_max=0) /
                  tf.reduce_sum(tf.clip_by_value(z_from_k, clip_value_min=-np.infty, clip_value_max=0)))
    negative_z = tf.where(tf.is_nan(negative_z), tf.zeros_like(negative_z), negative_z)
    
    return r_k * (alpha*positive_z + beta*negative_z)


def lrp(r, w, x, x_min, alpha=1, beta=0):
    """
    
    Parameters
    -------
    r : tensor (batchsize, n_upper_layer_units)
        Relevance coming from upper layer
    w : tensor (n_lower_layer_units, n_upper_layer_units)
        Weights from lower layer to upper layer
    x : tensor (batchsize, n_lower_layer_units)
        Activations coming from lower layer
    x_min : tensor (batchsize, n_lower_layer_units)
        Minimum over all timesteps of activations coming from lower layer per neuron
    """
    n_upper_units = r.get_shape().as_list()[-1]
    r_from_upper_layer = []
    
    for k in range(n_upper_units):
        r_from_upper_layer.append(r_from_k(r_k=r[:, k], w_from_k=w[:, k], x=x, x_min=x_min, alpha=alpha, beta=beta))
    r_from_upper_layer = tf.stack(r_from_upper_layer, axis=-1)
    
    return tf.reduce_sum(r_from_upper_layer, axis=-1)


def lrp_broadcasted(r, w, x, x_min, alpha=1, beta=0, parallel_iterations=10):
    """
    
    Parameters
    -------
    r : tensor (batchsize, timesteps, n_upper_layer_units)
        Relevance coming from upper layer
    w : tensor (n_lower_layer_units, n_upper_layer_units)
        Weights from lower layer to upper layer
    x : tensor (batchsize, timesteps, n_lower_layer_units)
        Activations coming from lower layer
    x_min : tensor (batchsize, n_lower_layer_units)
        Minimum over all timesteps of activations coming from lower layer per neuron
    """
    n_timesteps = tf.shape(r)[1]
    
    init_lrp = tf.expand_dims(lrp(r=r[:, 0, :], w=w, x=x[:, 0, :], x_min=x_min, alpha=alpha, beta=beta), axis=1)
    
    with tf.name_scope("LRPRNNLoop"):
        # Create initial tensors
        init_tensors = OrderedDict([('t', tf.constant(1, dtype=tf.int32))])
        init_tensors.update(dict(lrp_t=init_lrp))
        
        # Get initial tensor shapes in tf format
        init_shapes = OrderedDict([('t', init_tensors['t'].get_shape())])
        init_shapes.update(dict(lrp_t=tf.TensorShape(init_lrp.get_shape().as_list()[:1] + [None] +
                                                     init_lrp.get_shape().as_list()[2:])))
    
    def cond(t, *args):
        return tf.less(t, n_timesteps)
    
    def body(t, lrp_t):
        new_lrp = lrp(r=r[:, t, :], w=w, x=x[:, t, :], x_min=x_min, alpha=alpha, beta=beta)
        lrp_t = tf.concat([lrp_t, tf.expand_dims(new_lrp, axis=1)], axis=1)
        t += 1
        return t, lrp_t

    wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=tuple(init_tensors.values()),
                           shape_invariants=tuple(init_shapes.values()), parallel_iterations=parallel_iterations,
                           back_prop=True, swap_memory=True)
    
    return wl_ret[1]
    


def lrp_lstm(r_out, o, c, i, z, w_o, w_i, act_h, multiplication_rule, alpha=1, beta=0, o_min=None, i_min=None,
             c_min=None, z_min=None, o_max=None, i_max=None, c_max=None, z_max=None):
    """lrp applied to TeLL LSTMLayer
    
    Parameters
    -------
    r_out : tensor (batchsize, timesteps, units)
    o, c, i, z  : tensor (batchsize, timesteps, units)
    w_o, w_i : tensor (incoming, outgoing)
    act_h activation function after cell
    multiplication_rule : int
    0...50/50 rule; 1...proportional rule; 3...no multiplication rule, no relevance through recurrent gate connections
    """
    n_timesteps = tf.shape(r_out)[1]
    
    if multiplication_rule == 0:
        mul_rule = additive_multiplication_rule
    elif multiplication_rule == 1:
        mul_rule = proportional_multiplication_rule
    elif multiplication_rule == 3:
        mul_rule = None
    else:
        raise AttributeError("Only multiplication_rule 0 and 1 are implemented")
    
    if beta == 0:
        o_min = tf.reduce_min(o, axis=1)
        i_min = tf.reduce_min(i, axis=1)
        c_min = tf.reduce_min(c, axis=1)
        z_min = tf.reduce_min(z, axis=1)
        
        o_max = tf.reduce_max(o, axis=1)
        i_max = tf.reduce_max(i, axis=1)
        c_max = tf.reduce_max(c, axis=1)
        z_max = tf.reduce_max(z, axis=1)
    else:
        o_min = tf.reduce_mean(o, axis=1)
        i_min = tf.reduce_mean(i, axis=1)
        c_min = tf.reduce_mean(c, axis=1)
        z_min = tf.reduce_mean(z, axis=1)
        
        o_max = 1
        i_max = 1
        c_max = 1
        z_max = 1

    # Create an set initializations for dict with LRP variables
    lrp_keys = ['r_from_o', 'r_from_i', 'r_cc', 'r_y', 'r_cy', 'r_o', 'r_c', 'r_zi', 'r_z', 'r_i']
    zero = tf.constant(0, dtype=tf.int32)
    zero_init = tf.zeros_like(r_out[:, 0:1, :], tf.float32)
    lrp_dict = OrderedDict([(k, zero_init) for k in lrp_keys])
    
    with tf.name_scope("LRPRNNLoop"):
        # Create initial tensors
        init_tensors = OrderedDict([('t', n_timesteps-1)])
        init_tensors.update(lrp_dict)
        
        # Get initial tensor shapes in tf format
        init_shapes = OrderedDict([('t', init_tensors['t'].get_shape())])
        lrp_shapes = OrderedDict((k, tf.TensorShape(lrp_dict[k].get_shape().as_list()[:1] + [None] +
                                                    lrp_dict[k].get_shape().as_list()[2:])) for k in lrp_dict.keys())
        init_shapes.update(lrp_shapes)
    
    def cond(t, *args):
        return tf.greater(t, zero)

    def body(t, r_from_o, r_from_i, r_cc, r_y, r_cy, r_o, r_c, r_zi, r_z, r_i):
        #
        # for time t
        #
        if mul_rule is None:
            r_y = tf.concat([r_y, tf.expand_dims(r_out[:, t, :], axis=1)], axis=1)
            r_cy = tf.concat([r_cy, tf.expand_dims(r_y[:, -1, :], axis=1)], axis=1)
            r_o = tf.concat([r_o, zero_init], axis=1)

            r_c = tf.concat([r_c, tf.expand_dims(r_cy[:, -1, :] + r_cc[:, -1, :], axis=1)], axis=1)
            
            r_zi_new = tf.expand_dims(r_c[:, -1, :] * (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
            r_zi = tf.concat([r_zi, tf.where(tf.is_nan(r_zi_new), zero_init, r_zi_new)], axis=1)

            r_z = tf.concat([r_z, tf.expand_dims(r_zi[:, -1, :], axis=1)], axis=1)
            r_i = tf.concat([r_i, zero_init], axis=1)
            
        else:
            r_y = tf.concat([r_y, tf.expand_dims(r_out[:, t, :] + r_from_o[:, -1, :] + r_from_i[:, -1, :], axis=1)],
                            axis=1)
            r_cy = tf.concat([r_cy, tf.expand_dims(mul_rule(act_h(c[:, t, :]), o[:, t, :], r_y[:, -1, :],
                                                            c_min, c_max, o_min, o_max), axis=1)], axis=1)
            r_o = tf.concat([r_o, tf.expand_dims(mul_rule(o[:, t, :], act_h(c[:, t, :]), r_y[:, -1, :],
                                                          o_min, o_max, c_min, c_max), axis=1)], axis=1)

            r_c = tf.concat([r_c, tf.expand_dims(r_cy[:, -1, :] + r_cc[:, -1, :], axis=1)], axis=1)
            
            r_zi_new = tf.expand_dims(r_c[:, -1, :] * (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
            r_zi = tf.concat([r_zi, tf.where(tf.is_nan(r_zi_new), zero_init, r_zi_new)], axis=1)

            r_z = tf.concat([r_z, tf.expand_dims(mul_rule(z[:, t, :], i[:, t, :], r_zi[:, -1, :],
                                                          z_min, z_max, i_min, i_max), axis=1)], axis=1)
            r_i = tf.concat([r_i, tf.expand_dims(mul_rule(i[:, t, :], z[:, t, :], r_zi[:, -1, :],
                                                          i_min, i_max, z_min, z_max), axis=1)], axis=1)
        
        #
        # distribute R to units through recurrent connections
        #
        r_from_o_t = lrp(r=r_o[:, -1, :], w=w_o, x=o[:, t-1, :], x_min=o_min, alpha=alpha, beta=beta)
        r_from_o = tf.concat([r_from_o, tf.expand_dims(r_from_o_t, axis=1)], axis=1)
        
        r_from_i_t = lrp(r=r_i[:, -1, :], w=w_i, x=i[:, t-1, :], x_min=i_min, alpha=alpha, beta=beta)
        r_from_i = tf.concat([r_from_i, tf.expand_dims(r_from_i_t, axis=1)], axis=1)
        
        #
        # for time t-1
        #
        r_cc_new = tf.expand_dims(c[:, t-1, :] / c[:, t, :] * r_c[:, -1, :], axis=1)
        r_cc = tf.concat([r_cc, tf.where(tf.is_nan(r_cc_new), zero_init, r_cc_new)], axis=1)
        
        t -= 1
        
        return [t, r_from_o, r_from_i, r_cc, r_y, r_cy, r_o, r_c, r_zi, r_z, r_i]
    
    wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=tuple(init_tensors.values()),
                           shape_invariants=tuple(init_shapes.values()), parallel_iterations=10,
                           back_prop=True, swap_memory=True)

    # Re-Associate returned tensors with keys
    r_collection = OrderedDict(zip(init_tensors.keys(), wl_ret))
    _ = r_collection.pop('t')
    
    # Remove artificial timestep at end of sequences (sequences are in reversed temporal order)
    for k in r_collection.keys():
        if k not in ['r_from_o', 'r_from_i', 'r_cc']:
            r_collection[k] = r_collection[k][:, 1:, :]
    
    #
    # for time t=0
    #
    t = 0
    if mul_rule is None:
        r_collection['r_y'] = tf.concat([r_collection['r_y'], tf.expand_dims(r_out[:, t, :], axis=1)], axis=1)
        r_collection['r_cy'] = tf.concat([r_collection['r_cy'], tf.expand_dims(r_collection['r_y'][:, -1, :], axis=1)], axis=1)
        r_collection['r_o'] = tf.concat([r_collection['r_o'], zero_init], axis=1)

        r_collection['r_c'] = tf.concat([r_collection['r_c'],
                                         tf.expand_dims(r_collection['r_cy'][:, -1, :] +
                                                        r_collection['r_cc'][:, -1, :], axis=1)], axis=1)

        r_collection['r_zi_new'] = tf.expand_dims(r_collection['r_c'][:, -1, :] *
                                                  (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
        r_collection['r_zi'] = tf.concat([r_collection['r_zi'],
                                          tf.where(tf.is_nan(r_collection['r_zi_new']), zero_init,
                                                   r_collection['r_zi_new'])], axis=1)

        r_collection['r_z'] = tf.concat([r_collection['r_z'], tf.expand_dims(r_collection['r_zi'][:, -1, :], axis=1)],
                                        axis=1)
        r_collection['r_i'] = tf.concat([r_collection['r_i'], zero_init], axis=1)
        
    else:
        r_collection['r_y'] = tf.concat([r_collection['r_y'],
                                         tf.expand_dims(r_out[:, t, :] + r_collection['r_from_o'][:, -1, :] +
                                                        r_collection['r_from_i'][:, -1, :], axis=1)], axis=1)
        r_collection['r_cy'] = tf.concat([r_collection['r_cy'],
                                          tf.expand_dims(mul_rule(act_h(c[:, t, :]), o[:, t, :],
                                                                  r_collection['r_y'][:, -1, :],
                                                                  c_min, c_max, o_min, o_max), axis=1)], axis=1)
        r_collection['r_o'] = tf.concat([r_collection['r_o'],
                                         tf.expand_dims(mul_rule(o[:, t, :], act_h(c[:, t, :]),
                                                                 r_collection['r_y'][:, -1, :],
                                                                 o_min, o_max, c_min, c_max), axis=1)], axis=1)

        r_collection['r_c'] = tf.concat([r_collection['r_c'],
                                         tf.expand_dims(r_collection['r_cy'][:, -1, :] +
                                                        r_collection['r_cc'][:, -1, :], axis=1)], axis=1)

        r_zi_new = tf.expand_dims(r_collection['r_c'][:, -1, :] * (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
        r_collection['r_zi'] = tf.concat([r_collection['r_zi'], tf.where(tf.is_nan(r_zi_new), zero_init, r_zi_new)],
                                         axis=1)

        r_collection['r_z'] = tf.concat([r_collection['r_z'],
                                         tf.expand_dims(mul_rule(z[:, t, :], i[:, t, :], r_collection['r_zi'][:, -1, :],
                                                                 z_min, z_max, i_min, i_max), axis=1)], axis=1)
        r_collection['r_i'] = tf.concat([r_collection['r_i'],
                                         tf.expand_dims(mul_rule(i[:, t, :], z[:, t, :], r_collection['r_zi'][:, -1, :],
                                                                 i_min, i_max, z_min, z_max), axis=1)], axis=1)
    
    
    # # Initialize input gate and output gate relevance with 0
    # r_from_o = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_o redistributed to the individual units in t-1
    # r_from_i = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_i redistributed to the individual units in t-1
    # r_cc = [tf.zeros_like(r_out[:, 0, :], tf.float32)]  # r_ct<-ct+1
    #
    # r_y = []
    # r_cy = []  # r_ct<-yt
    # r_o = []
    # r_c = []
    # r_zi = []
    # r_z = []
    # r_i = []
    # for t in rev_timesteps:
    #     #
    #     # for time t
    #     #
    #     if mul_rule is None:
    #         r_y.append(r_out[:, t, :])
    #         r_cy.append(r_y[-1])
    #         r_o.append(tf.zeros_like(r_y[-1]))
    #
    #         r_c.append(r_cy[-1] + r_cc[-1])
    #
    #         r_zi.append(r_c[-1] * (i[:, t, :] * z[:, t, :] / c[:, t, :]))
    #         r_zi[-1] = tf.where(tf.is_nan(r_zi[-1]), tf.zeros_like(r_zi[-1]), r_zi[
    #             -1])  # TODO: This only holds for all-positive case! Otherwise we will need to consider r_zi[-2] to assign either full R or 0
    #
    #         r_z.append(r_zi[-1])
    #         r_i.append(tf.zeros_like(r_zi[-1]))
    #
    #     else:
    #         r_y.append(r_out[:, t, :] + r_from_o[-1] + r_from_i[-1])
    #         r_cy.append(mul_rule(act_h(c[:, t, :]), o[:, t, :], r_y[-1], c_min, c_max, o_min, o_max))
    #         r_o.append(mul_rule(o[:, t, :], act_h(c[:, t, :]), r_y[-1], o_min, o_max, c_min, c_max))
    #
    #         r_c.append(r_cy[-1] + r_cc[-1])
    #
    #         r_zi.append(r_c[-1] * (i[:, t, :] * z[:, t, :] / c[:, t, :]))
    #         r_zi[-1] = tf.where(tf.is_nan(r_zi[-1]), tf.zeros_like(r_zi[-1]), r_zi[-1])  # TODO: This only holds for all-positive case! Otherwise we will need to consider r_zi[-2] to assign either full R or 0
    #
    #         r_z.append(mul_rule(z[:, t, :], i[:, t, :], r_zi[-1], z_min, z_max, i_min, i_max))
    #         r_i.append(mul_rule(i[:, t, :], z[:, t, :], r_zi[-1], i_min, i_max, z_min, z_max))
    #
    #     if t > 0:
    #         #
    #         # distribute R to units through recurrent connections
    #         #
    #         r_from_o_t = lrp(r=r_o[-1], w=w_o, x=o[:, t-1, :], x_min=o_min, alpha=alpha, beta=beta)
    #         r_from_o.append(r_from_o_t)
    #
    #         r_from_i_t = lrp(r=r_i[-1], w=w_i, x=i[:, t-1, :], x_min=i_min, alpha=alpha, beta=beta)
    #         r_from_i.append(r_from_i_t)
    #
    #         #
    #         # for time t-1
    #         #
    #         r_cc.append(c[:, t-1, :] / c[:, t, :] * r_c[-1])
    #         r_cc[-1] = tf.where(tf.is_nan(r_cc[-1]), tf.zeros_like(r_cc[-1]), r_cc[-1])  # TODO: This only holds for all-positive case!
    #
    # r_collection = dict(r_from_o=tf.stack(r_from_o, axis=1), r_from_i=tf.stack(r_from_i, axis=1),
    #                     r_cc=tf.stack(r_cc, axis=1), r_y=tf.stack(r_y, axis=1), r_cy=tf.stack(r_cy, axis=1),
    #                     r_o=tf.stack(r_o, axis=1), r_c=tf.stack(r_c, axis=1), r_zi=tf.stack(r_zi, axis=1),
    #                     r_z=tf.stack(r_z, axis=1), r_i=tf.stack(r_i, axis=1))
    
    # Relevance is stored with reversed time dimension - correct it
    r_collection = OrderedDict((k, v[:, ::-1, :]) for k, v in r_collection.items())
    
    return r_collection['r_z'], r_collection


class LRPLSTM(object):
    def __init__(self, o, c, i, z, w_o, w_i, act_h, multiplication_rule, alpha=1, beta=0, o_min=None,
                 i_min=None, c_min=None, z_min=None, o_max=None, i_max=None, c_max=None, z_max=None):
        """lrp applied to TeLL LSTMLayer
        
        Parameters
        -------
        o, c, i, z  : tensor (batchsize, timesteps, units)
            output gate, cell state, input gate, and cell input activations for all timesteps
        w_o, w_i : tensor (incoming, outgoing)
        act_h activation function after cell
        multiplication_rule : int
        0...50/50 rule; 1...proportional rule; 3...no multiplication rule, no relevance propagated through gates
        """
        
        n_timesteps = tf.shape(c)[1]
        
        if multiplication_rule == 0:
            mul_rule = additive_multiplication_rule
        elif multiplication_rule == 1:
            mul_rule = proportional_multiplication_rule
        elif multiplication_rule == 3:
            mul_rule = None
        else:
            raise AttributeError("Only multiplication_rule 0 and 1 are implemented")

        def init_minmax(tensor, default):
            if tensor is not None:
                return tensor
            else:
                return default
        
        if beta == 0:
            o_min = init_minmax(o_min, tf.reduce_min(o, axis=1))
            i_min = init_minmax(i_min, tf.reduce_min(i, axis=1))
            c_min = init_minmax(c_min, tf.reduce_min(c, axis=1))
            z_min = init_minmax(z_min, tf.reduce_min(z, axis=1))
            
            o_max = init_minmax(o_max, tf.reduce_max(o, axis=1))
            i_max = init_minmax(i_max, tf.reduce_max(i, axis=1))
            c_max = init_minmax(c_max, tf.reduce_max(c, axis=1))
            z_max = init_minmax(z_max, tf.reduce_max(z, axis=1))
        else:
            o_min = init_minmax(o_min, tf.reduce_mean(o, axis=1))
            i_min = init_minmax(i_min, tf.reduce_mean(i, axis=1))
            c_min = init_minmax(c_min, tf.reduce_mean(c, axis=1))
            z_min = init_minmax(z_min, tf.reduce_mean(z, axis=1))
            
            o_max = init_minmax(o_min, tf.constant(1, dtype=tf.float32))
            i_max = init_minmax(i_max, tf.constant(1, dtype=tf.float32))
            c_max = init_minmax(c_max, tf.constant(1, dtype=tf.float32))
            z_max = init_minmax(z_max, tf.constant(1, dtype=tf.float32))
            
        # Create an set initializations for dict with LRP variables
        lrp_keys = ['r_z', 'r_from_o', 'r_from_i', 'r_cc', 'r_y', 'r_cy', 'r_o', 'r_c', 'r_zi', 'r_i']
        zero = tf.constant(0, dtype=tf.int32)
        zero_init = tf.zeros_like(c[:, 0:1, :], tf.float32)
        lrp_dict = OrderedDict([(k, zero_init) for k in lrp_keys])
        
        self._n_timesteps_ = n_timesteps
        self._lrp_keys_ = lrp_keys
        self._lrp_dict_ = lrp_dict
        
        self._alpha_, self._beta_ = alpha, beta
        self._mul_rule_ = mul_rule
        
        self._act_h_ = act_h
        self._w_o_, self._w_i_ = w_o, w_i
        self._o_, self._c_, self._i_, self._z_ = o, c, i, z
        
        self._o_min_, self._i_min_, self._c_min_, self._z_min_ = o_min, i_min, c_min, z_min
        self._o_max_, self._i_max_, self._c_max_, self._z_max_ = o_max, i_max, c_max, z_max
        
        self.__zero__ = zero
        self.__zero_init__ = zero_init
        
    def get_loop_tensors(self):
        return list(self._lrp_dict_.values())
    
    def set_loop_tensors(self, tensors):
        self._lrp_dict_ = OrderedDict(((k, v) for k, v in zip(self._lrp_keys_, tensors)))
    
    def get_relevance(self):
        
        # Relevance is stored with reversed time dimension - correct it and remove artificial timestep at sequence end
        lrp_dict = OrderedDict((k, v[:, ::-1, :]) if k in ['r_from_o', 'r_from_i', 'r_cc']
                               else (k, v[:, -2::-1, :]) for k, v in self._lrp_dict_.items())
        
        return lrp_dict['r_z'], lrp_dict
        
    def lrp_one_timestep(self, r_incoming, t):
        """lrp applied to TeLL LSTMLayer for 1 timestep
        
        Parameters
        -------
        r_incoming : tensor (batchsize, 1, units)
            relevance coming in (flowing in from upper layer/future timestep)
        t : tensor
            int tensor with current timestep (as to be used to index o, c, i, z)
        """
        zero = self.__zero__
        zero_init = self.__zero_init__
        
        alpha, beta = self._alpha_, self._beta_
        mul_rule = self._mul_rule_
        
        act_h = self._act_h_
        w_o, w_i = self._w_o_, self._w_i_
        o_min, i_min, c_min, z_min = self._o_min_, self._i_min_, self._c_min_, self._z_min_
        o_max, i_max, c_max, z_max = self._o_max_, self._i_max_, self._c_max_, self._z_max_
        o, c, i, z = self._o_, self._c_, self._i_, self._z_
        
        lrp_keys = self._lrp_keys_
        lrp_dict = self._lrp_dict_
        r_z, r_from_o, r_from_i, r_cc, r_y, r_cy, r_o, r_c, r_zi, r_i = [lrp_dict[k] for k in lrp_keys]
        
        #
        # for time t
        #
        if mul_rule is None:
            r_y = tf.concat([r_y, tf.expand_dims(r_incoming[:, -1, :], axis=1)], axis=1)
            r_cy = tf.concat([r_cy, tf.expand_dims(r_y[:, -1, :], axis=1)], axis=1)
            r_o = tf.concat([r_o, zero_init], axis=1)
    
            r_c = tf.concat([r_c, tf.expand_dims(r_cy[:, -1, :] + r_cc[:, -1, :], axis=1)], axis=1)
    
            r_zi_new = tf.expand_dims(r_c[:, -1, :] * (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
            r_zi = tf.concat([r_zi, tf.where(tf.is_nan(r_zi_new), zero_init, r_zi_new)], axis=1)
    
            r_z = tf.concat([r_z, tf.expand_dims(r_zi[:, -1, :], axis=1)], axis=1)
            r_i = tf.concat([r_i, zero_init], axis=1)

        else:
            r_y = tf.concat([r_y, tf.expand_dims(r_incoming[:, -1, :] + r_from_o[:, -1, :] + r_from_i[:, -1, :],
                                                 axis=1)], axis=1)
            r_cy = tf.concat([r_cy, tf.expand_dims(mul_rule(act_h(c[:, t, :]), o[:, t, :], r_y[:, -1, :],
                                                            c_min, c_max, o_min, o_max), axis=1)], axis=1)
            r_o = tf.concat([r_o, tf.expand_dims(mul_rule(o[:, t, :], act_h(c[:, t, :]), r_y[:, -1, :],
                                                          o_min, o_max, c_min, c_max), axis=1)], axis=1)
    
            r_c = tf.concat([r_c, tf.expand_dims(r_cy[:, -1, :] + r_cc[:, -1, :], axis=1)], axis=1)
    
            r_zi_new = tf.expand_dims(r_c[:, -1, :] * (i[:, t, :] * z[:, t, :] / c[:, t, :]), axis=1)
            r_zi = tf.concat([r_zi, tf.where(tf.is_nan(r_zi_new), zero_init, r_zi_new)], axis=1)
    
            r_z = tf.concat([r_z, tf.expand_dims(mul_rule(z[:, t, :], i[:, t, :], r_zi[:, -1, :],
                                                          z_min, z_max, i_min, i_max), axis=1)], axis=1)
            r_i = tf.concat([r_i, tf.expand_dims(mul_rule(i[:, t, :], z[:, t, :], r_zi[:, -1, :],
                                                          i_min, i_max, z_min, z_max), axis=1)], axis=1)
        
        #
        # distribute R to units through recurrent connections
        #
        t_greater_0 = tf.greater(t, zero)
        r_from_o_t = lrp(r=r_o[:, -1, :], w=w_o, x=o[:, t - 1, :], x_min=o_min, alpha=alpha, beta=beta)
        r_from_o = tf.cond(t_greater_0,
                           lambda: tf.concat([r_from_o, tf.expand_dims(r_from_o_t, axis=1)], axis=1),
                           lambda: r_from_o)

        r_from_i_t = lrp(r=r_i[:, -1, :], w=w_i, x=i[:, t - 1, :], x_min=i_min, alpha=alpha, beta=beta)
        r_from_i = tf.cond(t_greater_0,
                           lambda: tf.concat([r_from_i, tf.expand_dims(r_from_i_t, axis=1)], axis=1),
                           lambda: r_from_i)
        
        #
        # for time t-1
        #
        r_cc_new = tf.expand_dims(c[:, t - 1, :] / c[:, t, :] * r_c[:, -1, :], axis=1)
        r_cc = tf.cond(t_greater_0,
                       lambda: tf.concat([r_cc, tf.where(tf.is_nan(r_cc_new), zero_init, r_cc_new)], axis=1),
                       lambda: r_cc)
        
        self._lrp_dict_ = OrderedDict(((k, v) for k, v in
                                       zip(lrp_keys, [r_z, r_from_o, r_from_i, r_cc, r_y, r_cy, r_o, r_c, r_zi, r_i])))
