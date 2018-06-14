# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Main file for simple convLSTM example

Main file for simple convLSTM example to be used with convLSTM architecture in sample_architectures and config file
config_convlstm.json; Plots input, output, weights, ConvLSTM output (of 2 ConvLSTM units), and cell states in
working_dir;

"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

#
# Imports before spawning workers (do NOT import tensorflow or matplotlib here)
#
import os
import sys
from collections import OrderedDict

import numpy as np
import progressbar

from TeLL.config import Config
from TeLL.dataprocessing import Normalize
from TeLL.datareaders import initialize_datareaders, DataLoader
from TeLL.utility.misc import AbortRun, check_kill_file
from TeLL.utility.plotting import launch_plotting_daemon, save_subplots
from TeLL.utility.timer import Timer
from TeLL.utility.workingdir import Workspace

if __name__ == "__main__":
    #
    # Start subprocess for plotting workers
    #  Due to a garbage-collector bug with matplotlib/GPU, launch_plotting_daemon needs so be called before tensorflow
    #  import
    #
    launch_plotting_daemon(num_workers=3)
    
    from TeLL.session import TeLLSession
    
    import tensorflow as tf
    from TeLL.regularization import regularize
    from TeLL.loss import image_crossentropy


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def update_step(loss, config, clip_gradient=1., scope='optimizer'):
    """Computation of gradients and application of weight updates
    
    Optimizer can be supplied via config file, e.g. as
    "optimizer_params": {"learning_rate": 1e-3},
    "optimizer": "'AdamOptimizer'"
    
    Parameters
    -------
    loss : tensor
        Tensor representing the
    config : config file
        Configuration file
    clip_gradient : (positive) float or False
        Clip gradient at +/- clip_gradient or don't clip gradient if clip_gradient=False
    
    Returns
    -------
    : tensor
        Application of gradients via optimizer
    """
    # Define weight update
    with tf.variable_scope(scope):
        trainables = tf.trainable_variables()
        # Set optimizer (one learning rate for all layers)
        optimizer = getattr(tf.train, config.optimizer)(**config.optimizer_params)
        
        # Calculate gradients
        gradients = tf.gradients(loss, trainables)
        # Clip all gradients
        if clip_gradient:
            gradients = [tf.clip_by_value(grad, -clip_gradient, clip_gradient) for grad in gradients]
        # Set and return weight update
        return optimizer.apply_gradients(zip(gradients, trainables))


def evaluate_on_validation_set(validationset, step: int, session, model, summary_writer, validation_summary,
                               val_loss, workspace: Workspace):
    """Convenience function for evaluating network on a validation set
    
    Parameters
    -------
    validationset : dataset reader
        Dataset reader for the validation set
    step : int
        Current step in training
    session : tf.session
        Tensorflow session to use
    model : network model
        Network model
    val_loss : tensor
        Tensor representing the validation loss computation
    
    Returns
    -------
    : float
        Loss averaged over validation set
    """
    loss = 0
    
    _pbw = ['Evaluating on validation set: ', progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=_pbw, maxval=validationset.n_mbs - 1, redirect_stdout=True).start()
    
    mb_validation = validationset.batch_loader()
    
    with Timer(verbose=True, name="Evaluate on Validation Set"):
        for mb_i, mb in enumerate(mb_validation):
            # Abort if indicated by file
            check_kill_file(workspace)
            
            val_summary, cur_loss = session.run([validation_summary, val_loss],
                                                feed_dict={model.X: mb['X'], model.y_: mb['y']})
            
            loss += cur_loss
            progress.update(mb_i)
            
            mb.clear()
            del mb
    
    progress.finish()
    
    avg_loss = loss / validationset.n_mbs
    
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Validation Loss", simple_value=avg_loss)]),
                               step)
    
    print("Validation scores:\n\tstep {} validation loss {}".format(step, avg_loss))
    sys.stdout.flush()
    
    return avg_loss


def main(_):
    # ------------------------------------------------------------------------------------------------------------------
    # Setup training
    # ------------------------------------------------------------------------------------------------------------------
    
    # Initialize config, parses command line and reads specified file; also supports overriding of values from cmd
    config = Config()
    
    #
    # Prepare input data
    #
    
    # Make sure datareader is reproducible
    random_seed = config.get_value('random_seed', 12345)
    np.random.seed(random_seed)  # not threadsafe, use rnd_gen object where possible
    rnd_gen = np.random.RandomState(seed=random_seed)
    
    # Set datareaders
    n_timesteps = config.get_value('mnist_n_timesteps', 20)
    
    # Load datasets for trainingset
    with Timer(name="Loading Data"):
        readers = initialize_datareaders(config, required=("train", "val"))
    
    # Set Preprocessing
    trainingset = Normalize(readers["train"], apply_to=['X', 'y'])
    validationset = Normalize(readers["val"], apply_to=['X', 'y'])
    
    # Set minibatch loaders
    trainingset = DataLoader(trainingset, batchsize=2, batchsize_method='zeropad', verbose=False)
    validationset = DataLoader(validationset, batchsize=2, batchsize_method='zeropad', verbose=False)
    
    #
    # Initialize TeLL session
    #
    tell = TeLLSession(config=config, summaries=["train", "validation"], model_params={"dataset": trainingset})
    
    # Get some members from the session for easier usage
    sess = tell.tf_session
    summary_writer_train, summary_writer_validation = tell.tf_summaries["train"], tell.tf_summaries["validation"]
    model = tell.model
    workspace, config = tell.workspace, tell.config
    
    #
    # Define loss functions and update steps
    #
    print("Initializing loss calculation...")
    loss, _ = image_crossentropy(target=model.y_[:, 10:, :, :], pred=model.output[:, 10:, :, :, :],
                                 pixel_weights=model.pixel_weights[:, 10:, :, :], reduce_by='mean')
    train_summary = tf.summary.scalar("Training Loss", loss)  # create summary to add to tensorboard
    
    # Loss function for validationset
    val_loss = loss
    val_loss_summary = tf.summary.scalar("Validation Loss", val_loss)  # create summary to add to tensorboard
    
    # Regularization
    reg_penalty = regularize(layers=model.get_layers(), l1=config.l1, l2=config.l2,
                             regularize_weights=True, regularize_biases=True)
    regpen_summary = tf.summary.scalar("Regularization Penalty", reg_penalty)  # create summary to add to tensorboard
    
    # Update step for weights
    update = update_step(loss + reg_penalty, config)
    
    #
    # Initialize tensorflow variables (either initializes them from scratch or restores from checkpoint)
    #
    global_step = tell.initialize_tf_variables().global_step
    
    #
    # Set up plotting
    #  (store tensors we want to plot in a dictionary for easier tensor-evaluation)
    #
    # We want to plot input, output and target for the 1st sample, 1st frame, and 1st channel in subplot 1
    tensors_subplot1 = OrderedDict()
    tensors_subplot2 = OrderedDict()
    tensors_subplot3 = OrderedDict()
    for frame in range(n_timesteps):
        tensors_subplot1['input_{}'.format(frame)] = model.X[0, frame, :, :]
        tensors_subplot2['target_{}'.format(frame)] = model.y_[0, frame, :, :] - 1
        tensors_subplot3['network_output_{}'.format(frame)] = tf.argmax(model.output[0, frame, :, :, :], axis=-1) - 1
    # We also want to plot the cell states and hidden states for each frame (again of the 1st sample and 1st lstm unit)
    # in subplot 2 and 3
    tensors_subplot4 = OrderedDict()
    tensors_subplot5 = OrderedDict()
    for frame in range(len(model.lstm_layer.c)):
        tensors_subplot4['hiddenstate_{}'.format(frame)] = model.lstm_layer.h[frame][0, :, :, 0]
        tensors_subplot5['cellstate_{}'.format(frame)] = model.lstm_layer.c[frame][0, :, :, 0]
    # Create a list to store all symbolic tensors for plotting
    plotting_tensors = list(tensors_subplot1.values()) + list(tensors_subplot2.values()) + \
                       list(tensors_subplot3.values()) + list(tensors_subplot4.values()) + \
                       list(tensors_subplot5.values())
    
    #
    # Finalize graph
    #  This makes our tensorflow graph read-only and prevents further additions to the graph
    #
    sess.graph.finalize()
    if sess.graph.finalized:
        print("Graph is finalized!")
    else:
        raise ValueError("Could not finalize graph!")
    
    sys.stdout.flush()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------------------------------------------------------
    
    try:
        epoch = int(global_step / trainingset.n_mbs)
        epochs = range(epoch, config.n_epochs)
        
        # Loop through epochs
        print("Starting training")
        
        for ep in epochs:
            epoch = ep
            print("Starting training epoch: {}".format(ep))
            # Initialize variables for over-all loss per epoch
            train_loss = 0
            
            # Load one minibatch at a time and perform a training step
            t_mb = Timer(verbose=True, name="Load Minibatch")
            mb_training = trainingset.batch_loader(rnd_gen=rnd_gen)
            
            #
            # Loop through minibatches
            #
            for mb_i, mb in enumerate(mb_training):
                sys.stdout.flush()
                # Print minibatch load time
                t_mb.print()
                
                # Abort if indicated by file
                check_kill_file(workspace)
                
                #
                # Calculate scores on validation set
                #
                if global_step % config.score_at == 0:
                    print("Starting scoring on validation set...")
                    evaluate_on_validation_set(validationset, global_step, sess, model, summary_writer_validation,
                                               val_loss_summary, val_loss, workspace)
                
                #
                # Perform weight updates and do plotting
                #
                if (mb_i % config.plot_at) == 0 and os.path.isfile(workspace.get_plot_file()):
                    # Perform weight update, return summary values and values for plotting
                    with Timer(verbose=True, name="Weight Update"):
                        plotting_values = []
                        train_summ, regpen_summ, _, cur_loss, *plotting_values = sess.run(
                            [train_summary, regpen_summary, update, loss, *plotting_tensors],
                            feed_dict={model.X: mb['X'], model.y_: mb['y']})
                    
                    # Add current summary values to tensorboard
                    summary_writer_train.add_summary(train_summ, global_step=global_step)
                    summary_writer_train.add_summary(regpen_summ, global_step=global_step)
                    
                    # Create and save subplot 1 (input)
                    save_subplots(images=plotting_values[:len(tensors_subplot1)],
                                  subfigtitles=list(tensors_subplot1.keys()),
                                  subplotranges=[(0, 1)] * n_timesteps, colorbar=True, automatic_positioning=True,
                                  tight_layout=True,
                                  filename=os.path.join(workspace.get_result_dir(),
                                                        "input_ep{}_mb{}.png".format(ep, mb_i)))
                    del plotting_values[:len(tensors_subplot1)]
                    
                    # Create and save subplot 2 (target)
                    save_subplots(images=plotting_values[:len(tensors_subplot2)],
                                  subfigtitles=list(tensors_subplot2.keys()),
                                  subplotranges=[(0, 10) * n_timesteps], colorbar=True, automatic_positioning=True,
                                  tight_layout=True,
                                  filename=os.path.join(workspace.get_result_dir(),
                                                        "target_ep{}_mb{}.png".format(ep, mb_i)))
                    del plotting_values[:len(tensors_subplot2)]
                    
                    # Create and save subplot 3 (output)
                    save_subplots(images=plotting_values[:len(tensors_subplot3)],
                                  subfigtitles=list(tensors_subplot3.keys()),
                                  # subplotranges=[(0, 10)] * n_timesteps,
                                  colorbar=True, automatic_positioning=True,
                                  tight_layout=True,
                                  filename=os.path.join(workspace.get_result_dir(),
                                                        "output_ep{}_mb{}.png".format(ep, mb_i)))
                    del plotting_values[:len(tensors_subplot3)]
                    
                    # Create and save subplot 2 (hidden states, i.e. ConvLSTM outputs)
                    save_subplots(images=plotting_values[:len(tensors_subplot4)],
                                  subfigtitles=list(tensors_subplot4.keys()),
                                  title='ConvLSTM hidden states (outputs)', colorbar=True, automatic_positioning=True,
                                  tight_layout=True,
                                  filename=os.path.join(workspace.get_result_dir(),
                                                        "hidden_ep{}_mb{}.png".format(ep, mb_i)))
                    del plotting_values[:len(tensors_subplot4)]
                    
                    # Create and save subplot 3 (cell states)
                    save_subplots(images=plotting_values[:len(tensors_subplot5)],
                                  subfigtitles=list(tensors_subplot5.keys()),
                                  title='ConvLSTM cell states', colorbar=True, automatic_positioning=True,
                                  tight_layout=True,
                                  filename=os.path.join(workspace.get_result_dir(),
                                                        "cell_ep{}_mb{}.png".format(ep, mb_i)))
                    del plotting_values[:len(tensors_subplot5)]
                
                else:
                    #
                    # Perform weight update without plotting
                    #
                    with Timer(verbose=True, name="Weight Update"):
                        train_summ, regpen_summ, _, cur_loss = sess.run([
                            train_summary, regpen_summary, update, loss],
                            feed_dict={model.X: mb['X'], model.y_: mb['y']})
                    
                    # Add current summary values to tensorboard
                    summary_writer_train.add_summary(train_summ, global_step=global_step)
                    summary_writer_train.add_summary(regpen_summ, global_step=global_step)
                
                # Add current loss to running average loss
                train_loss += cur_loss
                
                # Print some status info
                print("ep {} mb {} loss {} (avg. loss {})".format(ep, mb_i, cur_loss, train_loss / (mb_i + 1)))
                
                # Reset timer
                t_mb = Timer(name="Load Minibatch")
                
                # Free the memory allocated for the minibatch data
                mb.clear()
                del mb
                
                global_step += 1
            
            #
            # Calculate scores on validation set
            #
            
            # Perform scoring on validation set
            print("Starting scoring on validation set...")
            evaluate_on_validation_set(validationset, global_step, sess, model, summary_writer_validation,
                                       val_loss_summary, val_loss, workspace)
            
            # Save the model
            tell.save_checkpoint(global_step=global_step)
            
            # Abort if indicated by file
            check_kill_file(workspace)
    
    except AbortRun:
        print("Detected kill file, aborting...")
    
    finally:
        #
        # If the program executed correctly or an error was raised, close the data readers and save the model and exit
        #
        trainingset.close()
        validationset.close()
        tell.close(save_checkpoint=True, global_step=global_step)


if __name__ == "__main__":
    tf.app.run()
