"""
© Michael Widrich, Markus Hofmarcher, 2017

"""

# Import TeLL
from TeLL.config import Config
from TeLL.session import TeLLSession
from TeLL.datareaders import MNISTReader, DataLoader
from TeLL.dataprocessing import DataProcessing, Normalize, Zoom
from TeLL.utility.misc import AbortRun, check_kill_file
from TeLL.regularization import decor_penalty

# Import Tensorflow
if __name__ == "__main__":
    import tensorflow as tf


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def main(_):
    config = Config()
    # Create new TeLL session with two summary writers
    tell = TeLLSession(config=config, summaries=["train", "validation"])
    
    # Get some members from the session for easier usage
    session = tell.tf_session
    summary_writer_train, summary_writer_validation = tell.tf_summaries["train"], tell.tf_summaries["validation"]
    model = tell.model
    workspace, config = tell.workspace, tell.config
    
    # Parameters
    learning_rate = config.get_value("learning_rate", 1e-3)
    iterations = config.get_value("iterations", 1000)
    batchsize = config.get_value("batchsize", 250)
    display_step = config.get_value("display_step", 10)
    dropout = config.get_value("dropout_prob", 0.25)
    
    #
    # Prepare input data
    #
    
    # Set datareaders
    training_reader = MNISTReader(dset='train')
    validation_reader = MNISTReader(dset='validation')
    test_reader = MNISTReader(dset='test')
    
    # Set Preprocessing
    training_data_preprocessed = DataProcessing(training_reader, apply_to='X')
    training_data_preprocessed = Normalize(training_data_preprocessed, apply_to='X')
    training_data_preprocessed = Normalize(training_data_preprocessed, apply_to=['X', 'Y'])
    
    # Set minibatch loaders
    training_loader = DataLoader(training_data_preprocessed, batchsize=50, batchsize_method='zeropad')
    validation_loader = DataLoader(validation_reader, batchsize=50, batchsize_method='zeropad')
    test_loader = DataLoader(test_reader, batchsize=50, batchsize_method='zeropad')
    
    #
    # Define loss and optimizer
    #
    with tf.name_scope("Cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.output, labels=model.y_))
        decor1 = decor_penalty(model.hidden1, model.y_, 10, [1], 0.)
        decor2 = decor_penalty(model.hidden2, model.y_, 10, [1], 6e-5)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost + decor1 + decor2)
        tf.summary.scalar("Loss", cost)
        tf.summary.scalar("Decor", decor1 + decor2)
    
    # Evaluate model
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(model.output, 1), tf.argmax(model.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("Accuracy", accuracy)
    
    merged_summaries = tf.summary.merge_all()
    
    # Initialize tensorflow variables (either initializes them from scratch or restores from checkpoint)
    step = tell.initialize_tf_variables().global_step
    
    # -------------------------------------------------------------------------
    # Start training
    # -------------------------------------------------------------------------
    acc_train = 0.
    try:
        while step < iterations:
            # Loop through training set
            for mb_i, mb in enumerate(training_loader.batch_loader(num_cached=5, num_threads=3)):
                check_kill_file(workspace=workspace)
                
                # Perform weight update
                summary, acc_train, _ = session.run([merged_summaries, accuracy, optimizer],
                                                    feed_dict={model.X: mb['X'], model.y_: mb['y'],
                                                               model.dropout: dropout})
                summary_writer_train.add_summary(summary, mb_i + step * batchsize)
                
                if step % display_step == 0:
                    # Loop through validation set
                    cos_sum, acc_sum, cor_sum = (0, 0, 0)
                    for vmb_i, vmb in enumerate(validation_loader.batch_loader(num_cached=5, num_threads=3)):
                        cos, acc, cor = session.run([cost, accuracy, correct_pred],
                                                    feed_dict={model.X: vmb['X'], model.y_: vmb['y'],
                                                               model.dropout: 0})
                        cos_sum += cos
                        acc_sum += acc
                        cor_sum += cor
                    print('step {}: train acc {}, valid acc {}'.format(mb_i + step * batchsize, cos_sum/vmb_i,
                                                                       acc_sum/vmb_i, cor_sum/vmb_i))
                
                step += 1
                if step >= iterations:
                    break
        
        print("Training Finished!")
        
        # Final Eval
        for tmb_i, tmb in enumerate(test_loader.batch_loader(num_cached=len(test_reader.get_sample_keys()),
                                                             num_threads=1)):
            print("Test Accuracy:",
                  session.run(accuracy, feed_dict={model.X: tmb['X'],
                                                   model.y_: tmb['y'],
                                                   model.dropout: 0}))
    except AbortRun:
        print("Aborting...")
    finally:
        tell.close(global_step=step)


if __name__ == "__main__":
    tf.app.run()
