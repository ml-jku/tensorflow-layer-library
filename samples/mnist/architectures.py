"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import tensorflow as tf
from TeLL.layers import DenseLayer, DropoutLayer, ConvLayer, MaxPoolingLayer
from TeLL.initializations import weight_truncated_normal, weight_gauss_conv2d
from TeLL.config import Config


class DenseNet(object):
    def __init__(self, config: Config):
        # Network Parameters
        n_input = 784  # MNIST data input (img shape: 28*28)
        n_classes = 10  # MNIST total classes (0-9 digits)
        n_hidden_1 = config.n_hidden_1
        n_hidden_2 = config.n_hidden_2
        
        # tf Graph input
        X = tf.placeholder(tf.float32, [None, n_input], name="Features")
        y_ = tf.placeholder(tf.float32, [None, n_classes], name="Labels")
        d = tf.placeholder(tf.float32)

        #X_2d = tf.reshape(X, [-1, 28, 28, 1])

        #hidden1 = ConvLayer(X_2d, W=weight_gauss_conv2d([3, 3, 1, n_hidden_1]), name="ConvLayer1")
        #maxpool1 = MaxPoolingLayer(hidden1, name="MaxPoolingLayer1")
        #hidden2 = ConvLayer(maxpool1, W=weight_gauss_conv2d([3, 3, n_hidden_1, n_hidden_2]), name="ConvLayer2")
        #maxpool2 = MaxPoolingLayer(hidden2, name="MaxPoolingLayer2")
        #flat = tf.contrib.layers.flatten(maxpool2.get_output())
        
        # Hidden 1
        hidden1 = DenseLayer(X, n_hidden_1, name="Hidden_Layer_1",
                             a=tf.nn.sigmoid, W=weight_truncated_normal, b=tf.zeros)
        # Hidden 2
        hidden2 = DenseLayer(hidden1, n_hidden_2, name="Hidden_Layer_2",
                             a=tf.nn.sigmoid, W=weight_truncated_normal, b=tf.zeros)

        # Output
        out = DenseLayer(hidden2, n_classes, name="Output_Layer",
                         a=tf.identity, W=weight_truncated_normal, b=tf.zeros)
        
        self.X = X
        self.y_ = y_
        self.dropout = d
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output = out.get_output()
