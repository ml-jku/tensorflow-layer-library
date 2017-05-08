# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Main file for LSTM and dense layer example

Main file for LSTM example to be used with LSTM architecture in sample_architectures and config file config_lstm.json;
Also to be used with other LSTM example architectures and a dense layer architecture (see sample_architectures.py for
the different examples and descriptions);

"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

#
# Imports before spawning workers (do NOT import tensorflow or matplotlib here)
#
import sys
import numpy as np
import progressbar

# Import TeLL
from TeLL.config import Config
from TeLL.utility.workingdir import Workspace
from TeLL.datasets import ShortLongDataset
from TeLL.utility.timer import Timer
from TeLL.utility.misc import AbortRun, check_kill_file

if __name__ == "__main__":
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
    
    # Load datasets for trainingset
    with Timer(name="Loading Training Data"):
        # Make sure datareader is reproducible
        random_seed = config.get_value('random_seed', 12345)
        np.random.seed(random_seed)  # not threadsafe, use rnd_gen object where possible
        rnd_gen = np.random.RandomState(seed=random_seed)
        
        print("Loading training data...")
        trainingset = ShortLongDataset(n_timesteps=250, n_samples=3000, batchsize=config.batchsize, rnd_gen=rnd_gen)
        
        # Load datasets for validationset
        validationset = ShortLongDataset(n_timesteps=250, n_samples=300, batchsize=config.batchsize, rnd_gen=rnd_gen)
    
    # Initialize TeLL session
    tell = TeLLSession(config=config, summaries=["train"], model_params={"dataset": trainingset})
    
    # Get some members from the session for easier usage
    session = tell.tf_session
    summary_writer = tell.tf_summaries["train"]
    model = tell.model
    workspace, config = tell.workspace, tell.config
    
    # Loss function for trainingset
    print("Initializing loss calculation...")
    loss = tf.reduce_mean(tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(model.y_, model.output,
                                                                                  -tf.reduce_sum(
                                                                                      model.y_ - 1) / tf.reduce_sum(
                                                                                      model.y_)), axis=[1]))
    train_summary = tf.summary.scalar("Training Loss", loss)  # add loss to tensorboard
    
    # Loss function for validationset
    val_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(model.y_, model.output,
                                                                                      -tf.reduce_sum(
                                                                                          model.y_ - 1) / tf.reduce_sum(
                                                                                          model.y_)), axis=[1]))
    val_loss_summary = tf.summary.scalar("Validation Loss", val_loss)  # add val_loss to tensorboard
    
    # Regularization
    reg_penalty = regularize(layers=model.get_layers(), l1=config.l1, l2=config.l2,
                             regularize_weights=True, regularize_biases=True)
    regpen_summary = tf.summary.scalar("Regularization Penalty", reg_penalty)  # add reg_penalty to tensorboard
    
    # Update step for weights
    update = update_step(loss + reg_penalty, config)
    
    # Initialize Tensorflow variables
    global_step = tell.initialize_tf_variables().global_step
    
    sys.stdout.flush()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------------------------------------------------------
    
    try:
        epoch = int(global_step / trainingset.n_mbs)
        epochs = range(epoch, config.n_epochs)
        
        #
        # Loop through epochs
        #
        print("Starting training")
        
        for ep in epochs:
            print("Starting training epoch: {}".format(ep))
            # Initialize variables for over-all loss per epoch
            train_loss = 0
            
            # Load one minibatch at a time and perform a training step
            t_mb = Timer(name="Load Minibatch")
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
                    evaluate_on_validation_set(validationset, global_step, session, model, summary_writer,
                                               val_loss_summary, val_loss, workspace)
                
                #
                # Perform weight update
                #
                with Timer(name="Weight Update"):
                    train_summ, regpen_summ, _, cur_loss = session.run(
                        [train_summary, regpen_summary, update, loss],
                        feed_dict={model.X: mb['X'], model.y_: mb['y']})
                
                # Add current summary values to tensorboard
                summary_writer.add_summary(train_summ, global_step=global_step)
                summary_writer.add_summary(regpen_summ, global_step=global_step)
                
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
            # Calculate scores on validation set after training is done
            #
            
            # Perform scoring on validation set
            print("Starting scoring on validation set...")
            evaluate_on_validation_set(validationset, global_step, session, model, summary_writer, val_loss_summary,
                                       val_loss, workspace)
            
            tell.save_checkpoint(global_step=global_step)
            
            # Abort if indicated by file
            check_kill_file(workspace)
    
    except AbortRun:
        print("Detected kill file, aborting...")
    
    finally:
        tell.close(save_checkpoint=True, global_step=global_step)


if __name__ == "__main__":
    tf.app.run()
