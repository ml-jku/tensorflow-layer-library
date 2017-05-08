# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Functions for dropout masks

"""

import tensorflow as tf
import numpy as np
import os
import argparse
from PIL import Image
from TeLL.utility.timer import Timer
import logging


def make_ising_mask(shape, keep_prob, num_steps, beta, beta_step=1.01):
    """
    Create x ising patterns, return 4 x by flipping the patterns
    """

    # start off with random multinomial samples in range {-1,1}
    samples = 2 * tf.multinomial(tf.log([[1 - keep_prob, keep_prob]]), np.prod(shape)) - 1
    samples = tf.to_float(tf.reshape(samples, shape))
    ising_filter = tf.constant([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=tf.float32)
    ising_filter = tf.reshape(ising_filter, [3, 3, 1, 1])

    i = tf.constant(0)
    beta = tf.constant(beta)
    beta_step = tf.constant(beta_step)

    condition = lambda i_, beta_, samples_: tf.less(i_, num_steps)

    def body(i_, beta_, samples_):
        # only consider a small subset of bits to flip

        # We've got to find the average of the 4 nearest pixels
        # Ideas on how to do this:
        #    - hand written convolutional kernel
        #    - shift the matrix
        conv = tf.nn.conv2d(samples_, ising_filter, [1, 1, 1, 1], 'SAME')

        # Calculate the energy difference if we flip this pixel
        # energy = -1 * mul(conv, samples)
        # ed = -1 * mul(conv, -1 * samples) + mul(conv, samples) = 2 * mul(conv, samples)
        flip_bit = 2 * tf.multiply(conv, samples_)

        # ok, so here if everything is the same [low energy] this guy is really small.
        flip_bit = tf.exp(-beta_ * flip_bit)
        flip_bit = tf.to_float(tf.greater(flip_bit, tf.random_uniform(shape)))
        bits_to_flip = tf.multinomial(tf.log([[9., 1.]]), np.prod(shape))
        bits_to_flip = tf.to_float(tf.reshape(bits_to_flip, shape))
        flip_bit = tf.to_float(-2 * tf.multiply(flip_bit, bits_to_flip) + 1)
        samples_ = tf.multiply(flip_bit, samples_)
        if 0:
            beta_ = tf.Print(beta_, [tf.reduce_sum(samples_[0, :, :, 0])], message="This is the sum: ")
        i_ = tf.add(i_, 1)
        beta_ = tf.multiply(beta_, beta_step)
        return [i_, beta_, samples_]

    _, _, samples_out = tf.while_loop(condition, body, [i, beta, samples])
    samples_out = (samples_out + 1) / 2
    return samples_out


def boosted_ising_mask(shape, keep_prob, num_steps, beta, beta_step=1.01):
    """
    Create x ising patterns, return 4 x by flipping the patterns
    """

    assert len(shape) == 4
    assert shape[0] % 4 == 0
    c = shape[3]
    shape[3] = 1
    if keep_prob == 1:
        return tf.constant(1, dtype=tf.float32, shape=shape)

    shape[0] = int(shape[0] / 4)
    img_shape = shape[1:]
    samples = make_ising_mask(shape, keep_prob, num_steps, beta, beta_step=beta_step)
    my_image_list = []

    for s in tf.split(axis=0, num_or_size_splits=shape[0], value=samples):
        s = tf.reshape(s, img_shape)
        my_image_list.append(s)
        my_image_list.append(tf.image.flip_up_down(s))
        my_image_list.append(tf.image.flip_left_right(s))
        my_image_list.append(tf.image.flip_up_down(my_image_list[-1]))

    for i in range(len(my_image_list)):
        my_image_list[i] = tf.reshape(my_image_list[i], [1] + img_shape)
        # Create multiple color channels, each with the same values
        my_image_list[i] = tf.concat(axis=3, values=c * [my_image_list[i]])

    return tf.concat(axis=0, values=my_image_list)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--path", type=str, default="./", help="Path to output the images")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images (must be dividable by 4)")
    parser.add_argument("--keep_prob", type=float, default=.5, help="Probability for pattern")
    parser.add_argument("--num_steps", type=int, default=400, help="Iterations")
    parser.add_argument("--width", type=int, default=2000, help="Width of the picture to create")
    parser.add_argument("--height", type=int, default=1000, help="Height of the picture to create")
    parser.add_argument("--random_num", type=int, default=12345, help="Random number initialization")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES string. Used for resumed runs.")
    args = parser.parse_args()
    # --
    logging.info("Creating images in {}; keep_prob {}; num_steps {}; gpu {}".format(
        args.path, args.keep_prob, args.num_steps, args.gpu))
    return args


def main():
    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()

    out_path = args.path
    os.makedirs(out_path, exist_ok=True)

    if not args.num_images % 4 == 0:
        raise ValueError("Number of images must be a multiple of 4 due to data augmentation")

    if args.num_images >= 200:
        print("Many images are going to be created, an overflow on the GPU could occur")

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    tf.set_random_seed(args.random_num)

    samples = boosted_ising_mask(shape=[args.num_images, 400, int(400 * (args.width / args.height)), 3],
                                 keep_prob=args.keep_prob, num_steps=args.num_steps, beta=.5, beta_step=1.1)
    # samples = tf.image.resize_images(samples, (args.height, args.width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    logging.info('Samples' + str(samples))

    with Timer(verbose=True, name="Ising") as t:
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        samples_np = sess.run(samples)


    # Append parameters to log file
    params = ""
    for arg in vars(args):
        if arg not in ["path", "gpu"]:
            params += arg + ":" + str(getattr(args, arg)) + "-"

    params = params[:-1]

    logfile = os.path.join(out_path, "log.txt")
    if os.path.isfile(logfile):
        with open(logfile, "r") as f:
            x = f.read().split("\n")
        offset = int(x[-1].split("\t")[1])
    else:
        offset = 0
    with open(logfile, "a") as myfile:
        myfile.write("\n" + params + "\t" + str(offset + getattr(args, "num_images")))

    for i in range(samples_np.shape[0]):
        out = Image.fromarray(samples_np[i, :, :, 0] * 255)
        out = out.convert("1")
        out.save(out_path + '/image' + '_' + str(offset + i + 1) + '.png')

    sess.close()


if __name__ == "__main__":
    main()
