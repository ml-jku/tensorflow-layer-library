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

from TeLL.datasets import MovingDotDataset
from TeLL.config import Config
from TeLL.utility.plotting import save_subplots
from TeLL.utility.timer import Timer
from TeLL.utility.workingdir import Workspace
from TeLL.utility.plotting import Plotter
from TeLL.utility.misc import AbortRun, check_kill_file

if __name__ == "__main__":
    # Due to a garbage-collector bug with matplotlib/GPU, plotters need to be created before tensorflow is imported
    plotter = Plotter(num_workers=5, plot_function=save_subplots)
    
    from TeLL.session import TeLLSession
    import tensorflow as tf
    from TeLL.regularization import regularize


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
    # Load datasets for training and validation
    #
    with Timer(name="Loading Data", verbose=True):
        # Make sure datareader is reproducible
        random_seed = config.get_value('random_seed', 12345)
        np.random.seed(random_seed)  # not threadsafe, use rnd_gen object where possible
        rnd_gen = np.random.RandomState(seed=random_seed)
        
        print("Loading training data...")
        trainingset = MovingDotDataset(n_timesteps=5, n_samples=50, batchsize=config.batchsize, rnd_gen=rnd_gen)
        print("Loading validation data...")
        validationset = MovingDotDataset(n_timesteps=5, n_samples=25, batchsize=config.batchsize, rnd_gen=rnd_gen)
    
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
    pos_target_weight = np.prod(trainingset.y_shape[2:])-1  # only 1 pixel per sample is of positive class -> up-weight!
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=model.y_, logits=model.output,
                                                                   pos_weight=pos_target_weight))
    # loss = tf.reduce_mean(-tf.reduce_sum((model.y_ * tf.log(model.output)) *
    #                                      -tf.reduce_sum(model.y_ - 1) / tf.reduce_sum(model.y_),
    #                                      axis=[1, 2, 3, 4]))
    train_summary = tf.summary.scalar("Training Loss", loss)  # create summary to add to tensorboard
    
    # Loss function for validationset
    val_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=model.y_, logits=model.output,
                                                                       pos_weight=pos_target_weight))
    # val_loss = tf.reduce_mean(-tf.reduce_sum(model.y_ * tf.log(model.output) *
    #                                          -tf.reduce_sum(model.y_ - 1) / tf.reduce_sum(model.y_),
    #                                          axis=[1, 2, 3, 4]))
    val_loss_summary = tf.summary.scalar("Validation Loss", val_loss)  # create summary to add to tensorboard
    
    # Regularization
    reg_penalty = regularize(layers=model.get_layers(), l1=config.l1, l2=config.l2,
                             regularize_weights=True, regularize_biases=True)
    regpen_summary = tf.summary.scalar("Regularization Penalty", reg_penalty)  # create summary to add to tensorboard
    
    # Update step for weights
    update = update_step(loss + reg_penalty, config)

    #
    # Prepare plotting
    #
    plot_elements_sym = list(model.get_plot_dict().values())
    plot_elements = list()
    plot_ranges = model.get_plot_range_dict()
    
    #
    # Initialize tensorflow variables (either initializes them from scratch or restores from checkpoint)
    #
    global_step = tell.initialize_tf_variables().global_step

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
                    # Perform weight update, return summary_str and values for plotting
                    with Timer(verbose=True, name="Weight Update"):
                        train_summ, regpen_summ, _, cur_loss, cur_output, *plot_elements = sess.run(
                            [train_summary, regpen_summary, update, loss, model.output, *plot_elements_sym],
                            feed_dict={model.X: mb['X'], model.y_: mb['y']})
                    
                    # Add current summary values to tensorboard
                    summary_writer_train.add_summary(train_summ, global_step=global_step)
                    summary_writer_train.add_summary(regpen_summ, global_step=global_step)
                    
                    # Re-associate returned tensorflow values to plotting keys
                    plot_dict = OrderedDict(zip(list(model.get_plot_dict().keys()), plot_elements))
                    
                    #
                    # Plot subplots in plot_dict
                    # Loop through each element in plotlist and pass it to the save_subplots function for plotting
                    # (adapt this to your needs for plotting)
                    #
                    with Timer(verbose=True, name="Plotting", precision="msec"):
                        for plotlist_i, plotlist in enumerate(model.get_plotsink()):
                            for frame in range(len(plot_dict[plotlist[0]])):
                                subplotlist = []
                                subfigtitles = []
                                subplotranges = []
                                n_cols = int(np.ceil(np.sqrt(len(plotlist))))
                                
                                for col_i, col_i in enumerate(range(n_cols)):
                                    subfigtitles.append(plotlist[n_cols * col_i:n_cols * col_i + n_cols])
                                    subplotlist.append([plot_dict[p][frame * (frame < len(plot_dict[p])), :] for p in
                                                        plotlist[n_cols * col_i:n_cols * col_i + n_cols]])
                                    subplotranges.append([plot_ranges.get(p, False) for p in
                                                          plotlist[n_cols * col_i:n_cols * col_i + n_cols]])
                                
                                # remove rows/columns without images
                                subplotlist = [p for p in subplotlist if p != []]
                                
                                plot_args = dict(images=subplotlist,
                                                 filename=os.path.join(workspace.get_result_dir(),
                                                                       "plot{}_ep{}_mb{}_fr{}.png".format(plotlist_i,
                                                                                                          ep,
                                                                                                          mb_i,
                                                                                                          frame)),
                                                 subfigtitles=subfigtitles, subplotranges=subplotranges)
                                plotter.set_plot_kwargs(plot_args)
                                plotter.plot()
                    
                    # Plot outputs and cell states over frames if specified
                    if config.store_states and 'ConvLSTMLayer_h' in plot_dict:
                        convh = plot_dict['ConvLSTMLayer_h']
                        convrh = [c[0, :, :, 0] for c in convh]
                        convrh = [convrh[:6], convrh[6:12], convrh[12:18], convrh[18:24], convrh[24:]]
                        plot_args = dict(images=convrh,
                                         filename=os.path.join(workspace.get_result_dir(),
                                                               "plot{}_ep{}_mb{}_h.png".format(plotlist_i, ep,
                                                                                               mb_i)))
                        plotter.set_plot_kwargs(plot_args)
                        plotter.plot()
                    
                    if config.store_states and 'ConvLSTMLayer_c' in plot_dict:
                        convc = plot_dict['ConvLSTMLayer_c']
                        convrc = [c[0, :, :, 0] for c in convc]
                        convrc = [convrc[:6], convrc[6:12], convrc[12:18], convrc[18:24], convrc[24:]]
                        plot_args = dict(images=convrc,
                                         filename=os.path.join(workspace.get_result_dir(),
                                                               "plot{}_ep{}_mb{}_c.png".format(plotlist_i, ep,
                                                                                               mb_i)))
                        plotter.set_plot_kwargs(plot_args)
                        plotter.plot()
                
                else:
                    #
                    # Perform weight update without plotting
                    #
                    with Timer(verbose=True, name="Weight Update"):
                        train_summ, regpen_summ, _, cur_loss = sess.run([train_summary, regpen_summary, update, loss],
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
        plotter.close()


if __name__ == "__main__":
    tf.app.run()
