"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import tensorflow as tf
import TeLL
from TeLL.layers import DenseLayer, DropoutLayer, ConvLayer, RNNInputLayer, MaxPoolingLayer, DeConvLayer, ConvLSTMLayer, ScalingLayer, ConcatLayer
from TeLL.initializations import weight_truncated_normal, constant
from TeLL.config import Config
from TeLL.utility.misc import get_rec_attr

from collections import OrderedDict


class ConvLSTMSemsegEfor(object):
    def __init__(self, config: Config, dataset):
        """Architecture for semantic segmentation as described in presentation using standard for loop."""
        depth = config.get_value("enc_dec_depth", 2)
        basenr_convs = config.get_value("enc_dec_conv_maps_base", 16)
        include_org_label = config.get_value("include_org_label", False)
        init_name = config.get_value("conv_W_initializer", "weight_xavier_conv2d")
        conv_W_initializer = getattr(TeLL.initializations, init_name)

        #
        # Layer list
        #
        layers = list()

        #
        # Create placeholders for feeding an input frame and a label at the first timestep
        #
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.int32, shape=dataset.y_shape)

        if include_org_label:
            y_org = tf.placeholder(tf.int32, shape=dataset.y_org_shape)

        # ----------------------------------------------------------------------------------------------------------
        # Define network architecture
        # ----------------------------------------------------------------------------------------------------------
        # initializer for weight values of kernels
        # conv_W_initializer = weight_xavier_conv2d

        #
        # Initialize input to network of shape [sample, 1, x, y, features] with zero tensor of size of a frame
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        layers.append(RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32)))
        rnn_input_layer = layers[-1]

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
                                        comb=config.lstm_comb, store_states=config.store_states,
                                        tickerstep_biases=tf.zeros, output_dropout=config.lstm_output_dropout,
                                        precomp_fwds=False))
            lstm_layer = layers[-1]

            #
            # Optional maxpooling and upscaling of rec LSTM connections combined with/or optional feature squashing
            #
            ext_lstm_recurrence = None
            if config.lstm_rec_maxpooling:
                print("\tMaxpoolingDeconv...")
                layers.append(
                    MaxPoolingLayer(incoming=layers[-1], ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME',
                                    name='MaxPoolingLayer'))
                layers.append(DeConvLayer(incoming=layers[-1], a=tf.nn.elu,
                                          W=conv_W_initializer([3, 3, layers[-1].get_output_shape()[-1],
                                                                layers[-1].get_output_shape()[-1]]),
                                          strides=(1, 2, 2, 1),
                                          padding='SAME', name='DeConvLayer'))
                print("\tConvLSTMRecurrence...")
                ext_lstm_recurrence = layers[-1]

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
            n_lstm = basenr_convs * (2 ** depth) * 4
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1],
                                                          int(basenr_convs * (2 ** depth) * 4.5)]),
                                    padding='SAME', name='SubstituteConvLayer', a=tf.nn.elu))
            lstm_layer = layers[-1]

        #
        # Decoder and upscaling layers
        #
        for d in list(range(1, depth + 1))[::-1]:
            print("\tUpscalingLayer{}...".format(d))
            layers[-1] = ScalingLayer(incoming=layers[-1], size=encoders[d - 1].get_output_shape()[-3:-1],
                                      name='UpscalingLayergLayer{}'.format(d))

            print("\tConcatLayer{}...".format(d))
            layers.append(ConcatLayer([encoders[d - 1], layers[-1]], name='ConcatLayer{}'.format(d)))

            print("\tConvLayerDecoder{}...".format(d))
            layers.append(ConvLayer(incoming=layers[-1],
                                    W=conv_W_initializer([config.kernel_conv, config.kernel_conv,
                                                          layers[-1].get_output_shape()[-1], basenr_convs * (2 ** d)]),
                                    padding='SAME', name='ConvLayerDecoder{}'.format(d), a=tf.nn.elu))

        #
        # ConvLayer for semantic segmentation
        #
        print("\tConvLayerSemanticSegmentation...")
        layers.append(ConvLayer(incoming=layers[-1],
                                W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                      layers[-1].get_output_shape()[-1], 11]),
                                padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity))
        sem_seg_layer = layers[-1]

        # ----------------------------------------------------------------------------------------------------------
        # Loop through sequence positions and create graph
        # ----------------------------------------------------------------------------------------------------------

        #
        # Loop through sequence positions
        #
        print("\tRNN Loop...")
        sem_seg_out = list()
        for seq_pos in range(n_seq_pos):
            with tf.name_scope("Sequence_pos_{}".format(seq_pos)):
                print("\t  seq. pos. {}...".format(seq_pos))
                # Set input layer to X at frame (t) and outputs of upper layers at (t-1)
                layers[0].update(X[:, seq_pos:seq_pos + 1, :])

                # Calculate new network output at (t), including new hidden states
                _ = lstm_layer.get_output()
                sem_seg_out.append(sem_seg_layer.get_output(prev_layers=encoders + [lstm_layer]))

        #
        # Loop through tickersteps
        #
        # # Use empty frame as X during ticker steps (did not work so good)
        # tickerstep_input = tf.zeros(dataset.X_shape[:1] + (1,) + dataset.X_shape[2:], dtype=tf.float32,
        #                             name="tickerframe")

        # Use last frame as X during ticker steps
        #tickerstep_input = X[:, -1:, :]
        #layers[0].update(tickerstep_input)

        #for tickerstep in range(config.tickersteps):
        #    with tf.name_scope("Tickerstep_{}".format(tickerstep)):
        #        print("\t  tickerstep {}...".format(tickerstep))
        #
        #        # Calculate new network output at (t), including new hidden states
        #        _ = lstm_layer.get_output(tickerstep_nodes=True)


        #sem_seg_out = sem_seg_layer.get_output(prev_layers=encoders + [lstm_layer])

        print("\tDone!")

        #
        # Publish
        #
        self.X = X
        self.y_feed = y_
        self.y_ = y_[:, 10:]
        self.output = tf.concat(sem_seg_out[10:], 1)
        self.__layers = layers
        self.__n_lstm = n_lstm
        self.__lstm_layer = lstm_layer
        self.lstm_layer = lstm_layer
        self.__plot_dict, self.__plot_range_dict, self.__plotsink = self.__setup_plotting(config)
        if include_org_label:
            self.y_org = y_org

    def get_layers(self):
        return self.__layers

    def get_plotsink(self):
        return self.__plotsink

    def get_plot_dict(self):
        return self.__plot_dict

    def get_plot_range_dict(self):
        return self.__plot_range_dict

    def __setup_plotting(self, config):
        # Prepare for plotting
        # Create a list of lists with keys to plot in a subplot
        plotsink = []
        plot_dict = OrderedDict()
        plot_range_dict = OrderedDict()
        plot_sample = 0
        # only plot first, middle, and end frame when plotting LSTM outputs
        plot_frames = slice(-1, None)  # slice(0, sample_len, int(sample_len/2))

        #
        # init plot sink
        #
        plot_dict['X'] = self.X[plot_sample, plot_frames, :, :, :]
        plot_range_dict['X'] = [0, 1]
        plotsink.append(['X'])

        plot_dict['y_'] = self.y_[plot_sample, None, :, :]
        plot_range_dict['y_'] = [0, 20]
        plotsink[0].append('y_')

        #
        # Plot LSTM layer
        #

        # Output for first sample in mb
        if config.n_lstm:
            try:
                plot_dict_new, plotsink_new, plot_range_dict_new = self.__lstm_layer.get_plots_out(sample=plot_sample)

                # add LSTM output plot to first subfigure
                plotsink[0] += plotsink_new[0]
                plot_dict.update(plot_dict_new)
                plot_range_dict.update(plot_range_dict_new)

                # Weights from first LSTM unit
                plot_dict_new, plotsink_new, _ = self.__lstm_layer.get_plots_w(max_num_inp=6, max_num_out=1)

                plotsink += plotsink_new
                plot_dict.update(plot_dict_new)

                # States, if possible
                plot_dict_new, plotsink_new, _ = self.__lstm_layer.get_plots_state()
                plotsink += plotsink_new
                plot_dict.update(plot_dict_new)

                #
                # Plot activations in outputs from first 4 units
                #
                for unit in range(4):
                    plot_dict['ConvLSTM_f{}'.format(unit)] = self.__lstm_layer.out[plot_sample, plot_frames, :, :, unit]
                    plotsink.append(['ConvLSTM_f{}'.format(unit)])

            except AttributeError:
                pass
        #
        # Plot outputs
        #
        plot_dict['out'] = tf.arg_max(self.output[plot_sample, plot_frames, :, :, :], 3)
        plot_range_dict['out'] = [0, 20]
        plotsink[0].append('out')

        #
        # Publish sink
        #
        return [plot_dict, plot_range_dict, plotsink]


class Scaler(object):
    def __init__(self, config: Config, dataset):
        """
             
        """

        depth = config.get_value("enc_dec_depth", 2)
        basenr_convs = config.get_value("enc_dec_conv_maps_base", 32)
        include_org_label = config.get_value("include_org_label", False)
        init_name = config.get_value("conv_W_initializer", "weight_xavier_conv2d")
        conv_W_initializer = getattr(TeLL.initializations, init_name)
        shared = False

        #
        # Layer list
        #
        layers = list()

        #
        # Create placeholders for feeding an input frame and a label at the first timestep
        #
        n_seq_pos = dataset.X_shape[1]  # dataset.X_shape is [sample, seq_pos, x, y, features)
        X = tf.placeholder(tf.float32, shape=dataset.X_shape)
        y_ = tf.placeholder(tf.int32, shape=dataset.y_shape)

        if include_org_label:
            y_org = tf.placeholder(tf.int32, shape=dataset.y_org_shape)

        #
        # Input Layer
        #
        input_shape = dataset.X_shape[:1] + (1,) + dataset.X_shape[2:]
        layers.append(RNNInputLayer(tf.zeros(input_shape, dtype=tf.float32)))

        #
        # Scaler Structure
        #
        conv_weights_shape = [config.kernel_conv,
                              config.kernel_conv,
                              layers[-1].get_output_shape()[-1],
                              basenr_convs]

        if shared:
            shared_input_conv_weights = conv_W_initializer(conv_weights_shape)
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[11, 11]))
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[9, 9]))
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[7, 7]))
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[5, 5]))
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[3, 3]))
            layers.append(ConvLayer(incoming=layers[0], W=shared_input_conv_weights,
                                    a=tf.nn.elu, dilation_rate=[1, 1]))
        else:
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[11, 11]))
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[9, 9]))
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[7, 7]))
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[5, 5]))
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[3, 3]))
            layers.append(ConvLayer(incoming=layers[0], W=conv_W_initializer(conv_weights_shape),
                                    a=tf.nn.elu, dilation_rate=[1, 1]))

        # concat feature maps of all scale levels and reduce the number of features with a 1x1 conv
        layers.append(ConcatLayer(incomings=layers[1:]))
        conv_weights_shape = [1, 1, layers[-1].get_output_shape()[-1], basenr_convs]
        layers.append(ConvLayer(incoming=layers[-1], W=conv_W_initializer(conv_weights_shape)))

        # add 3 more conv layers to have some depth
        conv_weights_shape = [config.kernel_conv,
                              config.kernel_conv,
                              layers[-1].get_output_shape()[-1],
                              basenr_convs]
        layers.append(ConvLayer(incoming=layers[-1], W=conv_W_initializer(conv_weights_shape)))
        layers.append(ConvLayer(incoming=layers[-1], W=conv_W_initializer(conv_weights_shape)))
        layers.append(ConvLayer(incoming=layers[-1], W=conv_W_initializer(conv_weights_shape)))

        #
        # Output Layer
        #
        layers.append(ConvLayer(incoming=layers[-1],
                                W=conv_W_initializer([config.kernel_conv_out, config.kernel_conv_out,
                                                      layers[-1].get_output_shape()[-1], 11]),
                                padding='SAME', name='ConvLayerSemanticSegmentation', a=tf.identity))
        sem_seg_layer = layers[-1]

        self.X = X
        self.y_ = y_
        self.output = sem_seg_layer.get_output()
