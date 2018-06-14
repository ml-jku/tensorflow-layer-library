# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Some example architectures to be used with corresponding config files in folder configs/examples.

Overview:
---------
ArchitectureDense: Simple dense layer network (use with main_lstm.py)
ArchitectureLSTM: Most simple usage example of LSTM (use with main_lstm.py)
ArchitectureLSTM... : More advanced (optimized/flexible) usage examples of LSTM (use with main_lstm.py)
ArchitectureConvLSTM: Example for ConvLSTM and plotting (use with main_convlstm.py)

"""

import tensorflow as tf

from TeLL.config import Config
from TeLL.initializations import constant, weight_xavier_conv2d
from TeLL.layers import ConcatLayer, ConvLSTMLayer, ConvLayer, DenseLayer, LSTMLayer, RNNInputLayer, MaxPoolingLayer, \
    ScalingLayer
from TeLL.utility.misc import get_rec_attr


class ArchitectureDense(object):
    def __init__(self, config: Config, dataset):
        """Simple network with dense layer and dense output layer;
        
        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_dense.json
        
        Example input shapes: [n_samples, n_features]
        Example output shapes: [n_samples, n_features]
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        
        #
        # Create placeholders for input data (shape: [n_samples, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Dense Layer
        #  Input for the dense layer shall be X (TeLL layers take tensors or TeLL Layer instances as input)
        #
        print("\tDense layer...")
        
        dense_layer = DenseLayer(incoming=X, n_units=config.n_dense, name='DenseLayer', W=w_init, b=tf.zeros,
                                 a=tf.nn.elu)
        layers.append(dense_layer)
        
        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=dense_layer, n_units=n_output_units, name='DenseLayerOut', W=w_init,
                                  b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)
        
        #
        # Calculate output
        #  This will calculate the output of output_layer, including all dependencies
        #
        output = output_layer.get_output()
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers
    
    def get_layers(self):
        return self.__layers


class ArchitectureLSTM(object):
    def __init__(self, config: Config, dataset):
        """Simple network with LSTM layer and dense output layer; All sequence positions are fed to the LSTM layer at
        once, this is the most convenient but least flexible design; see ArchitectureLSTM_optimized for a faster
        version;
        
        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json
        
        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        
        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # LSTM Layer
        #  We want to create an output sequence with the LSTM instead of only returning the ouput at the last sequence
        #  position -> return_states=True
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=X, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)
        
        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)
        
        #
        # Calculate output
        #
        output = output_layer.get_output(tickersteps=config.tickersteps)
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers
    
    def get_layers(self):
        return self.__layers


class ArchitectureLSTMFlexible(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by dense output layer; Inputs are fed to LSTM layer sequence position
        by sequence position in a for-loop; this is the most flexible design, as showed e.g. in ArchitectureLSTM3;
        
        Command-line usage:
        Change entry
        "architecture": "sample_architectures.ArchitectureLSTM"
        to
        "architecture": "sample_architectures.ArchitectureLSTMFlexible" in samples/config_lstm.json. Then run
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json
        
        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        
        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.float32, shape=dataset.y_shape)
        n_output_units = dataset.y_shape[-1]  # nr of output features is number of classes
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, features]
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)
        
        #
        # LSTM Layer
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)
        
        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)
        
        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                
                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output()
        
        #
        # Loop through tickersteps
        #
        # Use zero input during ticker steps
        tickerstep_input = tf.zeros(dataset.X_shape[:1] + (1,) + dataset.X_shape[2:], dtype=tf.float32,
                                    name="tickerstep_input")
        
        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                
                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output(tickerstep_nodes=True)
        
        #
        # Calculate output but consider that the lstm_layer is already computed (i.e. do not modify cell states any
        # further)
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers
    
    def get_layers(self):
        return self.__layers


# TODO: Implement continuous prediction
class ArchitectureLSTMOptimized(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by dense output layer; Inputs are fed to LSTM layer sequence position
        by sequence position in tensorflow tf.while_loop();
        This is not as flexible as using a for-loop and more difficult to use but can be faster and optimized
        differently; LSTM return_states is not possible here unless manually implemented into the tf.while_loop (that is
        why we are only using the prediction at the last sequence position in this example);
        This is an advanced example, see ArchitectureLSTM to get started;
        
        Command-line usage:
        Change entry
        "architecture": "sample_architectures.ArchitectureLSTM"
        to
        "architecture": "sample_architectures.ArchitectureLSTMOptimized" in samples/config_lstm.json. Then run
        >>> python3 samples/main_lstm.py --config=samples/config_lstm.json
        
        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, 1, n_features]
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        
        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.float32, shape=dataset.y_shape)
        n_output_units = dataset.y_shape[-1]  # nr of output features is number of classes
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, features]
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)
        
        #
        # LSTM Layer
        #
        print("\tLSTM...")
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=w_init, W_ig=w_init, W_og=w_init, W_fg=w_init,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True)
        layers.append(lstm_layer)
        
        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=lstm_layer, n_units=n_output_units, name='DenseLayerOut',
                                  W=w_init, b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)
        
        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Loop through sequence positions
        #
        if n_seq_pos:
            def cond(seq_pos, *args):
                return seq_pos < n_seq_pos
            
            def body(seq_pos, lstm_h, lstm_c):
                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])
                
                # Update lstm states
                lstm_layer.h[-1], lstm_layer.c[-1] = lstm_h, lstm_c
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output()
                
                seq_pos = tf.add(seq_pos, 1)
                
                return seq_pos, lstm_layer.h[-1], lstm_layer.c[-1]
            
            with tf.name_scope("Sequence_pos"):
                print("\t  seq. pos. ...")
                wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=(tf.constant(0),
                                                                        lstm_layer.h[-1], lstm_layer.c[-1]),
                                       parallel_iterations=10, back_prop=True, swap_memory=True)
                lstm_layer.h[-1], lstm_layer.c[-1] = wl_ret[-2], wl_ret[-1]
        
        #
        # Loop through tickersteps
        #
        if config.tickersteps:
            def cond(seq_pos, *args):
                return seq_pos < config.tickersteps
            
            def body(seq_pos, lstm_h, lstm_c):
                # Set rnn input layer to input at current sequence position
                rnn_input_layer.update(X[:, -1:, :])
                
                # Update lstm states
                lstm_layer.h[-1], lstm_layer.c[-1] = lstm_h, lstm_c
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                _ = lstm_layer.get_output(tickerstep_nodes=True)
                
                seq_pos = tf.add(seq_pos, 1)
                
                return seq_pos, lstm_layer.h[-1], lstm_layer.c[-1]
            
            with tf.name_scope("Tickersteps"):
                print("\t  tickersteps ...")
                wl_ret = tf.while_loop(cond=cond, body=body, loop_vars=(tf.constant(0),
                                                                        lstm_layer.h[-1], lstm_layer.c[-1]),
                                       parallel_iterations=10, back_prop=True, swap_memory=True)
                lstm_layer.h[-1], lstm_layer.c[-1] = wl_ret[-2], wl_ret[-1]
        
        #
        # Calculate output but consider that the lstm_layer is already computed
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers
    
    def get_layers(self):
        return self.__layers


class ArchitectureLSTM3(object):
    def __init__(self, config: Config, dataset):
        """Architecture with LSTM layer followed by 2 dense layers and a dense output layer; The outputs of the 2 dense
        layers are used as additional recurrent connections for the LSTM; Inputs are fed to LSTM layer sequence
        position by sequence position in RNN loop; This is an advanced example, see ArchitectureLSTM to get started;
        
        Command-line usage:
        >>> python3 samples/main_lstm.py --config=samples/config_lstm3.json
        
        Example input shapes: [n_samples, n_sequence_positions, n_features]
        Example output shapes: [n_samples, n_sequence_positions, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # Prepare xavier initialization for weights
        w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        
        #
        # Create placeholders for input data (shape: [n_samples, n_sequence_positions, n_features])
        #
        input_shapes = dataset.get_input_shapes()
        X = tf.placeholder(tf.float32, shape=input_shapes['X'].shape)
        y_ = tf.placeholder(tf.float32, shape=input_shapes['y'].shape)
        n_output_units = dataset.datareader.get_num_classes()  # nr of output features is number of classes
        
        n_seq_pos = input_shapes['X'].shape[1]  # dataset.X_shape is [sample, seq_pos, features]
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Input Layer
        #  RNNInputLayer will hold the input to network at each sequence position. We will initalize it with zeros-
        #  tensor of shape [sample, 1, features]
        #
        input_shape = input_shapes['X'].shape[:1] + (1,) + input_shapes['X'].shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)
        
        #
        # LSTM Layer
        #
        print("\tLSTM...")
        # We want to modify the number of recurrent connections -> we have to specify the shape of the recurrent weights
        rec_w_shape = (sum(config.n_dense_units) + config.n_lstm, config.n_lstm)
        # The forward weights can be initialized automatically, for the recurrent ones we will use our rec_w_shape
        lstm_w = [w_init, w_init(rec_w_shape)]
        lstm_layer = LSTMLayer(incoming=rnn_input_layer, n_units=config.n_lstm, name='LSTM',
                               W_ci=lstm_w, W_ig=lstm_w, W_og=lstm_w, W_fg=lstm_w,
                               b_ci=tf.zeros, b_ig=tf.zeros, b_og=tf.zeros, b_fg=tf.zeros,
                               a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid, a_out=tf.nn.elu,
                               c_init=tf.zeros, h_init=tf.zeros, forgetgate=True, precomp_fwds=True, return_states=True)
        layers.append(lstm_layer)
        
        #
        # Dense Layers
        #
        print("\tDense layers...")
        dense_layers = list()
        for n_units in config.n_dense_units:
            dense_layers.append(DenseLayer(incoming=layers[-1], n_units=n_units, name='DenseLayer', W=w_init,
                                           b=tf.zeros, a=tf.nn.elu))
            layers.append(layers[-1])
        
        #
        # Use dense layers as additional recurrent input to LSTM
        #
        full_lstm_input = ConcatLayer([lstm_layer] + dense_layers, name='LSTMRecurrence')
        lstm_layer.add_external_recurrence(full_lstm_input)
        
        #
        # Output Layer
        #
        print("\tOutput layer...")
        output_layer = DenseLayer(incoming=dense_layers[-1], n_units=n_output_units, name='DenseLayerOut', W=w_init,
                                  b=tf.zeros, a=tf.sigmoid)
        layers.append(output_layer)
        
        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to input at current sequence position
                layers[0].update(X[:, seq_pos:seq_pos + 1, :])
                
                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer.get_output()
        
        #
        # Loop through tickersteps
        #
        # Use zeros as input during ticker steps
        tickerstep_input = tf.zeros(input_shape, dtype=tf.float32,
                                    name="tickerstep_input")
        
        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                # Set rnn input layer to input at current sequence position
                layers[0].update(tickerstep_input)
                
                # Calculate new lstm state (this automatically computes all dependencies, including rec. connections)
                _ = lstm_layer.get_output(tickerstep_nodes=True)
        
        #
        # Calculate output but consider that the lstm_layer is already computed
        #
        output = output_layer.get_output(prev_layers=[lstm_layer])
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # Store layers in list for regularization in main file
        self.__layers = layers
    
    def get_layers(self):
        return self.__layers


class ArchitectureConvLSTM(object):
    def __init__(self, config: Config, dataset):
        """Example for convolutional network with convLSTM and convolutional output layer; Plots cell states, hidden
        states, X, y_, and a argmax over the convLSTM units outputs;
        
        Command-line usage:
        >>> python3 samples/main_convlstm.py --config=samples/config_convlstm.json
        
        Example input shapes: [n_samples, n_sequence_positions, x_dim, y_dim, n_features]
        Example output shapes: [n_samples, 1, x_dim, y_dim, n_features] (with return_states=True),
        [n_samples, 1, n_features] (with return_states=False)
        """
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        # We will use xavier initialization later
        conv_W_initializer = weight_xavier_conv2d
        
        #
        # Create placeholders for feeding an input frame and a label at the each sequence position
        #
        X_shape = dataset.mb_info['X'][0]
        y_shape = dataset.mb_info['y'][0]
        n_classes = y_shape[-1]
        n_seq_pos = X_shape[1]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        X = tf.placeholder(tf.float32, shape=X_shape)
        y_ = tf.placeholder(dataset.mb_info['y'][1], shape=y_shape)  # dataset.y_shape is [sample, seq_pos, features)
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Initialize input to network of shape [sample, 1, x, y, features] with zero tensor of size of a frame
        #
        input_shape = X_shape[:1] + (1,) + X_shape[2:]
        rnn_input_layer = RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)
        
        #
        # ConvLSTM Layer
        #
        n_lstm = config.n_lstm  # number of output feature channels
        lstm_x_fwd = config.kernel_lstm_fwd  # x/y size of kernel for forward connections
        lstm_y_fwd = config.kernel_lstm_fwd  # x/y size of kernel for forward connections
        lstm_x_bwd = config.kernel_lstm_bwd  # x/y size of kernel for recurrent connections
        lstm_y_bwd = config.kernel_lstm_bwd  # x/y size of kernel for recurrent connections
        lstm_input_channels_fwd = rnn_input_layer.get_output_shape()[-1]  # number of input channels
        if config.reduced_rec_lstm:
            lstm_input_channels_bwd = config.reduced_rec_lstm  # number of recurrent connections (after squashing)
        else:
            lstm_input_channels_bwd = n_lstm  # number of recurrent connections
        
        # Here we create our kernels and biases for the convLSTM; See ConvLSTMLayer() documentation for more info;
        lstm_init = dict(W_ci=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_ig=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_og=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         W_fg=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                               conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                         b_ci=constant([n_lstm]),
                         b_ig=constant([n_lstm]),
                         b_og=constant([n_lstm]),
                         b_fg=constant([n_lstm], 1))
        
        print("\tConvLSTM...")
        conv_lstm = ConvLSTMLayer(incoming=rnn_input_layer, n_units=n_lstm, **lstm_init,
                                  a_out=get_rec_attr(tf, config.lstm_act),
                                  forgetgate=getattr(config, 'forgetgate', True), store_states=config.store_states,
                                  return_states=False, precomp_fwds=False, tickerstep_biases=tf.zeros)
        layers.append(conv_lstm)
        
        #
        # Optional feature squashing of recurrent convLSTM connections
        #  We can use an additional convolutional layer to squash the number of LSTM output features and use its output
        #  to replace the convLSTM recurrent connections
        #
        if config.reduced_rec_lstm:
            print("\tFeatureSquashing...")
            # Define a weight kernel and create the convolutional layer for squashing
            kernel = conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                         conv_lstm.get_output_shape()[-1], config.reduced_rec_lstm])
            squashed_recurrences = ConvLayer(incoming=conv_lstm, W=kernel, padding='SAME',
                                             name='ConvLayerFeatureSquashing', a=tf.nn.elu)
            layers.append(squashed_recurrences)
            
            print("\tConvLSTMRecurrence...")
            # Overwrite the existing ConvLSTM recurrences with the output of the feature squashing layer
            conv_lstm.add_external_recurrence(squashed_recurrences)
        
        #
        # Conventional ConvLayer after convLSTM as output layer (tf.identitiy output function because of cross-entropy)
        #
        print("\tConvLayer...")
        output_layer = ConvLayer(incoming=conv_lstm,
                                 W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                       conv_lstm.get_output_shape()[-1], n_classes]),
                                 padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity)
        layers.append(output_layer)
        
        # ----------------------------------------------------------------------------------------------------------
        #  Create graph through sequence positions and ticker steps
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to current frame
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                output = output_layer.get_output()
        
        #
        # Loop through tickersteps
        #
        # Use last frame as input during ticker steps
        tickerstep_input = X[:, -1:, :]
        
        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                
                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)
                
                # Calculate new network state at new frame and activate tickerstep biases
                output = output_layer.get_output(tickerstep_nodes=True)
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = output
        # We will use this list of layers for regularization in the main file
        self.__layers = layers
        # We will plot some parts of the lstm, so we make it accessible as attribute
        self.lstm_layer = conv_lstm
    
    def get_layers(self):
        return self.__layers


class ArchitectureConvLSTMMNIST(object):
    def __init__(self, config: Config, dataset):
        """Example for convolutional network with convLSTM and convolutional output layer; Plots cell states, hidden
        states, X, y_, and a argmax over the convLSTM units outputs;
        
        Command-line usage:
        >>> python3 samples/main_convlstm_mnist.py --config=samples/config_convlstm_mnist.json
        """
        import TeLL
        
        #
        # Some convenience objects
        #
        # We will use a list to store all layers for regularization etc. (this is optional)
        layers = []
        
        depth = config.get_value("enc_dec_depth", 2)
        basenr_convs = config.get_value("enc_dec_conv_maps_base", 16)
        init_name = config.get_value("conv_W_initializer", "weight_xavier_conv2d")
        conv_W_initializer = getattr(TeLL.initializations, init_name)
        
        #
        # Create placeholders for feeding an input frame and a label at each sequence position
        #
        X_shape = dataset.mb_info['X'][0]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        y_shape = dataset.mb_info['y'][0]
        frame_input_shape = X_shape[:1] + (1,) + X_shape[2:]
        n_classes = 11
        frame_output_shape = y_shape[:1] + (1,) + y_shape[2:] + (n_classes,)
        n_seq_pos = X_shape[1]
        X = tf.placeholder(tf.float32, shape=X_shape)
        y_ = tf.placeholder(tf.int32, shape=y_shape)  # dataset.y_shape is [sample, seq_pos, features)
        
        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        
        #
        # Initialize input to network of shape [sample, 1, x, y, features] with zero tensor of size of a frame
        #
        rnn_input_layer = RNNInputLayer(tf.zeros(frame_input_shape, dtype=tf.float32))
        layers.append(rnn_input_layer)
        
        #
        # Encoder and maxpooling layers
        #
        encoders = list()
        for d in range(1, depth + 1):
            print("\tConvLayerEncoder{}...".format(d))
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1], basenr_convs * (2 ** d)]),
                                    padding='SAME', name='ConvLayerEncoder{}'.format(d), a=tf.nn.elu))
            encoders.append(layers[-1])
            print("\tMaxpoolingLayer{}...".format(d))
            layers.append(MaxPoolingLayer(incoming=layers[-1], ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME',
                                          name='MaxpoolingLayer{}'.format(d)))
        
        #
        # ConvLSTM Layer
        #
        if config.n_lstm:
            n_lstm = config.n_lstm
            lstm_x_fwd = config.kernel_lstm_fwd
            lstm_y_fwd = config.kernel_lstm_fwd
            lstm_x_bwd = config.kernel_lstm_bwd
            lstm_y_bwd = config.kernel_lstm_bwd
            
            lstm_input_channels_fwd = layers[-1].get_output_shape()[-1]
            if config.reduced_rec_lstm:
                lstm_input_channels_bwd = config.reduced_rec_lstm
            else:
                lstm_input_channels_bwd = n_lstm
            
            lstm_init = dict(W_ci=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_ig=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_og=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             W_fg=[conv_W_initializer([lstm_x_fwd, lstm_y_fwd, lstm_input_channels_fwd, n_lstm]),
                                   conv_W_initializer([lstm_x_bwd, lstm_y_bwd, lstm_input_channels_bwd, n_lstm])],
                             b_ci=constant([n_lstm]),
                             b_ig=constant([n_lstm]),
                             b_og=constant([n_lstm]),
                             b_fg=constant([n_lstm], 1))
            
            print("\tConvLSTM...")
            layers.append(ConvLSTMLayer(incoming=layers[-1], n_units=n_lstm, **lstm_init,
                                        a_out=get_rec_attr(tf, config.lstm_act), forgetgate=config.forgetgate,
                                        store_states=config.store_states, tickerstep_biases=tf.zeros,
                                        output_dropout=config.lstm_output_dropout, precomp_fwds=False))
            lstm_layer = layers[-1]
            
            #
            # Optional feature squashing
            #
            ext_lstm_recurrence = None
            
            if config.reduced_rec_lstm:
                print("\tFeatureSquashing...")
                layers.append(ConvLayer(incoming=layers[-1],
                                        W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                              layers[-1].get_output_shape()[-1],
                                                              config.reduced_rec_lstm]),
                                        padding='SAME', name='ConvLayerFeatureSquashing', a=tf.nn.elu))
                print("\tConvLSTMRecurrence...")
                ext_lstm_recurrence = layers[-1]
            
            if ext_lstm_recurrence is not None:
                lstm_layer.add_external_recurrence(ext_lstm_recurrence)
        else:
            print("\tSubstituteConvLayer...")
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1],
                                                          int(basenr_convs * (2 ** depth) * 4.5)]),
                                    padding='SAME', name='SubstituteConvLayer', a=tf.nn.elu))
            lstm_layer = layers[-1]
        
        #
        # ConvLayer for semantic segmentation
        #
        print("\tConvLayerSemanticSegmentation...")
        layers.append(ConvLayer(incoming=layers[-1],
                                W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                      layers[-1].get_output_shape()[-1], n_classes]),
                                padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity))
        
        #
        # Upscaling layer
        #
        print("\tUpscalingLayer...")
        layers[-1] = ScalingLayer(incoming=layers[-1], size=frame_output_shape[-3:-1], name='UpscalingLayergLayer')
        
        output_layer = layers[-1]
        
        # ----------------------------------------------------------------------------------------------------------
        #  Create graph through sequence positions and ticker steps
        # ----------------------------------------------------------------------------------------------------------
        outputs_all_timesteps = []
        
        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set rnn input layer to current frame
                rnn_input_layer.update(X[:, seq_pos:seq_pos + 1, :])
                
                # Calculate new network state at new frame (this updates the network's hidden activations, cell states,
                # and dependencies automatically)
                output = output_layer.get_output()
                outputs_all_timesteps.append(output)
        
        #
        # Loop through tickersteps
        #
        # Use last frame as input during ticker steps
        tickerstep_input = X[:, -1:, :]
        
        for tickerstep in range(config.tickersteps):
            with tf.name_scope("Tickerstep_{}".format(tickerstep)):
                print("\t  tickerstep {}...".format(tickerstep))
                
                # Set rnn input layer to tickerstep input
                rnn_input_layer.update(tickerstep_input)
                
                # Calculate new network state at new frame and activate tickerstep biases
                output = output_layer.get_output(tickerstep_nodes=True)
                outputs_all_timesteps.append(output)
        
        print("\tDone!")
        
        #
        # Publish
        #
        self.X = X
        self.y_ = y_
        self.output = tf.concat(outputs_all_timesteps, axis=1, name='outputs_all_timesteps')
        pixel_weights = tf.ones_like(y_, dtype=tf.float32)
        # pixel_weights -= tf.cast(y_ == 0, dtype=tf.float32) * tf.constant(1.-0.2)
        self.pixel_weights = pixel_weights
        # We will use this list of layers for regularization in the main file
        self.__layers = layers
        # We will plot some parts of the lstm, so we make it accessible as attribute
        self.lstm_layer = lstm_layer
    
    def get_layers(self):
        return self.__layers
