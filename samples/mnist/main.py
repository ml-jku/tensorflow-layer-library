"""
© Michael Widrich, Markus Hofmarcher, 2017

Example for mnist predictions via dense network

Command-line usage:
>>> python3 samples/mnist/main_convlstm.py --config=samples/mnist/config.json
"""
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Import TeLL
from TeLL.config import Config
from TeLL.session import TeLLSession
from TeLL.utility.timer import Timer
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
    # Load Data
    #
    with Timer(name="Load data"):
        mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    
    # Define loss and optimizer
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
            check_kill_file(workspace=workspace)
            batch_x, batch_y = mnist.train.next_batch(batchsize)
            
            i = step * batchsize
            if step % display_step == 0:
                summary, acc = session.run([merged_summaries, accuracy],
                                           feed_dict={model.X: mnist.validation.images[:2048],
                                                      model.y_: mnist.validation.labels[:2048],
                                                      model.dropout: 0})
                summary_writer_validation.add_summary(summary, i)
                print('step {}: train acc {}, valid acc {}'.format(i, acc_train, acc))
            else:
                summary, acc_train, _ = session.run([merged_summaries, accuracy, optimizer],
                                              feed_dict={model.X: batch_x, model.y_: batch_y, model.dropout: dropout})
                summary_writer_train.add_summary(summary, i)
            
            step += 1
        
        print("Training Finished!")
        
        # Final Eval
        print("Test Accuracy:",
              session.run(accuracy, feed_dict={model.X: mnist.test.images[:2048],
                                               model.y_: mnist.test.labels[:2048],
                                               model.dropout: 0}))
    except AbortRun:
        print("Aborting...")
    finally:
        tell.close(global_step=step)


if __name__ == "__main__":
    tf.app.run()
