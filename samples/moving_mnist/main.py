"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from dataset import BouncingMNISTDataHandler
import numpy as np

# Import TeLL
from TeLL.config import Config
from TeLL.session import TeLLSession
from TeLL.utility.timer import Timer
from TeLL.utility.misc import AbortRun, check_kill_file
from TeLL.loss import image_crossentropy, iou_loss, blurred_cross_entropy
from TeLL.evaluation import Evaluation
from TeLL.utility.plotting import Plotter, save_subplots
from TeLL.utility.plotting_daemons import start_plotting_daemon, stop_plotting_daemon
from collections import OrderedDict
import os

# Import Tensorflow
if __name__ == "__main__":
    plotter = Plotter(num_workers=5, plot_function=save_subplots)

    import tensorflow as tf

from scipy.misc import imsave

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

class DataSet(object):
    def __init__(self, x_shape, y_shape):
        self.X_shape = x_shape
        self.y_shape = y_shape

def to_color(labels):
    image = np.zeros(labels.shape + (3,))
    print(labels.shape)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                for l in range(labels.shape[3]):
                    image[i, j, k, l] = {
                         0: [  0,   0,   0],
                         1: [255,   0,   0],
                         2: [  0, 255,   0],
                         3: [  0,   0, 255],
                         4: [255, 255,   0],
                         5: [  0, 255, 255],
                         6: [255,   0, 255],
                         7: [255, 255, 255],
                         8: [128, 255,   0],
                         9: [  0, 128, 255],
                        10: [255,   0, 128],
                        11: [255, 128,   0]
                    }[labels[i, j, k, l]]

    return image


def to_image(pred, true):
    # in  shape is (20, batch_size, 64, 64, 3)
    # out shape is (batch_size, 64 * 10, 64 * 4, 3)

    assert(pred.shape == true.shape)
    shape = pred.shape
    out = np.zeros((shape[1], 256, 640, 3))

    for i in range(shape[0]):
        for t in range(shape[1]):
            h_from = (t % 10) * 64
            h_to = h_from + 64
            v_from = (0 if t < 10 else 2) * 64
            v_to = v_from + 64
            out[i, v_from:v_to, h_from:h_to] = true[i, t]
            out[i, v_from+64:v_to+64, h_from:h_to] = pred[i, t]

    return out


def main(_):
    np.random.seed(0)
    rng = np.random.RandomState(seed=0)

    config = Config()

    #
    # Load Data
    #
    with Timer(name="Load data"):
        training_data = BouncingMNISTDataHandler(
            config, config.mnist_train_images, config.mnist_train_labels, rng)
        test_data = BouncingMNISTDataHandler(
            config, config.mnist_test_images, config.mnist_test_labels, rng)

    dataset = DataSet((config.batch_size, config.num_frames, config.image_size, config.image_size, 1),
                      (config.batch_size, config.num_frames, config.image_size, config.image_size))

    # Create new TeLL session with two summary writers
    tell = TeLLSession(config=config, summaries=["train", "validation"], model_params={"dataset": dataset})
    
    # Get some members from the session for easier usage
    session = tell.tf_session
    summary_writer_train, summary_writer_validation = tell.tf_summaries["train"], tell.tf_summaries["validation"]
    model = tell.model
    workspace, config = tell.workspace, tell.config
    
    # Parameters
    learning_rate = config.get_value("learning_rate", 1e-3)
    iterations = config.get_value("iterations", 1000)
    batch_size = config.get_value("batch_size", 256)
    display_step = config.get_value("display_step", 10)
    calc_statistics = config.get_value("calc_statistics", False)
    blur_filter_size = config.get_value("blur_filter_size", None)

    training_summary_tensors = OrderedDict()

    # Define loss and optimizer
    #with tf.name_scope("Cost"):
    #    sem_seg_loss, _ = image_crossentropy(pred=model.output, target=model.y_,
    #                                         calc_statistics=calc_statistics, reduce_by="sum")
    #    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sem_seg_loss)
    #    tf.summary.scalar("Loss", sem_seg_loss)

    # Evaluate model
    validation_summary_tensors = OrderedDict()

    # validationset always uses class weights for loss calculation
    with tf.name_scope('Cost'):
        blur_sampling_range = tf.placeholder(tf.float32)

        if blur_filter_size is not None:
            sem_seg_loss = blurred_cross_entropy(output=model.output, target=model.y_,
                                                 filter_size=blur_filter_size,
                                                 sampling_range=blur_sampling_range)
        else:
            sem_seg_loss, _ = image_crossentropy(pred=model.output, target=model.y_,
                                                 reduce_by="mean", calc_statistics=calc_statistics)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sem_seg_loss)
        iou, iou_op = tf.contrib.metrics.streaming_mean_iou(
            predictions=tf.squeeze(tf.arg_max(model.output, 4)),
            labels=tf.squeeze(model.y_),
            num_classes=model.output.get_shape()[-1])
        loss_prot = tf.summary.scalar("Loss", sem_seg_loss)
        iou_prot = tf.summary.scalar("IoU", iou)

    train_summaries = tf.summary.merge([loss_prot])
    valid_summaries = tf.summary.merge([loss_prot, iou_prot])
    
    # Initialize tensorflow variables (either initializes them from scratch or restores from checkpoint)
    step = tell.initialize_tf_variables().global_step
    
    # -------------------------------------------------------------------------
    # Start training
    # -------------------------------------------------------------------------

    plot_elements_sym = list(model.get_plot_dict().values())
    plot_elements = list()
    plot_ranges = model.get_plot_range_dict()

    try:
        while step < iterations:
            check_kill_file(workspace=workspace)
            batch_x, batch_y = training_data.GetBatch()
            
            i = step * batch_size
            if step % display_step == 0:
                mean_loss = 0.
                for j in range(10):
                    test_x, test_y = test_data.GetBatch()

                    summary, loss, _, *plot_elements = session.run([valid_summaries, sem_seg_loss, iou_op, *plot_elements_sym],
                                               feed_dict={model.X: test_x,
                                                          model.y_feed: test_y,
                                                          blur_sampling_range: 3.5})

                    summary_writer_validation.add_summary(summary, i)
                    mean_loss += loss

                    # Re-associate returned tensorflow values to plotting keys
                    plot_dict = OrderedDict(zip(list(model.get_plot_dict().keys()), plot_elements))

                    # Plot outputs and cell states over frames if specified
                    if config.store_states and 'ConvLSTMLayer_h' in plot_dict and step % config.plot_at == 0:
                        convh = plot_dict['ConvLSTMLayer_h']
                        convrh = [c[0, :, :, 0] for c in convh]
                        convrh = [convrh[:6], convrh[6:12], convrh[12:18], convrh[18:24], convrh[24:]]
                        plot_args = dict(images=convrh,
                                         filename=os.path.join(workspace.get_result_dir(),
                                                               "step{}_h.png".format(step)))
                        plotter.set_plot_kwargs(plot_args)
                        plotter.plot()

                    if config.store_states and 'ConvLSTMLayer_c' in plot_dict and step % config.plot_at == 0:
                        convc = plot_dict['ConvLSTMLayer_c']
                        convrc = [c[0, :, :, 0] for c in convc]
                        convrc = [convrc[:6], convrc[6:12], convrc[12:18], convrc[18:24], convrc[24:]]
                        plot_args = dict(images=convrc,
                                         filename=os.path.join(workspace.get_result_dir(),
                                                               "step{}_c.png".format(step)))
                        plotter.set_plot_kwargs(plot_args)
                        plotter.plot()
                print('Validation Loss at step {}: {}'.format(i, mean_loss / 10))

            summary, loss, _ = session.run([train_summaries, sem_seg_loss, optimizer],
                                          feed_dict={model.X: batch_x,
                                                     model.y_feed: batch_y,
                                                     blur_sampling_range: 3.5})
            summary_writer_train.add_summary(summary, i)
            
            step += 1
        
        print("Training Finished!")

        # Final Eval
        mean_loss = 0.

        for j in range(100):
            test_x, test_y = test_data.GetBatch()
            summary, loss, _ = session.run([valid_summaries, sem_seg_loss, iou_op],
                                        feed_dict={model.X: test_x,
                                                   model.y_feed: test_y,
                                                   blur_sampling_range: 3.5})
            mean_loss += loss

        test_x, test_y = test_data.GetBatch()
        pred = session.run(tf.argmax(model.output, 4), feed_dict={model.X: test_x})

        pred = to_color(pred)
        true = to_color(test_y)
        out = to_image(pred, true)

        for i in range(pred.shape[0]):
            imsave(tell.workspace.get_result_dir() + '/sample_{:02d}.png'.format(i), out[i,])

        print("Validation Loss {}".format(mean_loss / 100))
    except AbortRun:
        print("Aborting...")
    finally:
        tell.close(global_step=step)
        plotter.close()


if __name__ == "__main__":
    tf.app.run()
