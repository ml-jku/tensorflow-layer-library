"""
Essence downloaded from http://www.cs.toronto.edu/~nitish/unsupervised_video/
"""

import numpy as np
import gzip
import sys

from TeLL.config import Config


def load_mnist(file, labels=False):
    print('open ' + file)
    with gzip.open(file, 'rb') as f:
        # skip header
        f.read(8 if labels else 16)
        a = np.array([float(i) for i in f.read()])

        if not labels:
            a = np.reshape(a, [-1, 28, 28])

        return a


def load_mnist_images(file):
    return load_mnist(file, labels=False)


def load_mnist_labels(file):
    return load_mnist(file, labels=True)


class BouncingMNISTDataHandler(object):
    def __init__(self, config: Config, images_file, labels_file, rng):
        self.rng = rng
        self.seq_length_ = config.num_frames
        self.batch_size_ = config.batch_size
        self.image_size_ = config.image_size
        self.num_digits_ = config.num_digits
        self.step_length_ = config.step_length
        self.label_threshold_ = config.label_threshold
        self.higher_num_on_top_ = config.higher_num_on_top
        self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2
        self.data_ = load_mnist_images(images_file)
        self.labels_ = load_mnist_labels(labels_file)
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        self.rng.shuffle(self.indices_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        self.row_ = 0

    def GetRandomTrajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = self.rng.rand(batch_size)
        x = self.rng.rand(batch_size)

        # Choose a random velocity.
        theta = self.rng.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        # return b

    def GetBatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_

        start_y, start_x = self.GetRandomTrajectory(batch_size * self.num_digits_)
        data = np.zeros((self.seq_length_, batch_size, self.image_size_, self.image_size_), dtype=np.float32)
        labels = np.zeros((self.seq_length_, batch_size, self.image_size_, self.image_size_, self.num_digits_),
                          dtype=np.float32)
        for j in range(batch_size):
            for n in range(self.num_digits_):
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    self.rng.shuffle(self.indices_)

                digit_image = self.data_[ind, :, :]
                digit_label = self.labels_[ind]

                for i in range(self.seq_length_):
                    # draw digit into data
                    top = start_y[i, j * self.num_digits_ + n]
                    left = start_x[i, j * self.num_digits_ + n]
                    bottom = top + self.digit_size_
                    right = left + self.digit_size_
                    data[i, j, top:bottom, left:right] = self.Overlap(
                        data[i, j, top:bottom, left:right], digit_image)

                    # draw digit into labels
                    labels[i, j, top:bottom, left:right, n] = \
                        np.asarray(digit_image > self.label_threshold_, np.int) * (digit_label + 1)

        if self.higher_num_on_top_:
            labels = np.amax(labels, 4)

        return np.swapaxes(np.reshape(data, list(data.shape) + [1]), 0, 1), \
               np.swapaxes(labels, 0, 1)


#def main():
#    data_pb = ReadDataProto(sys.argv[1])
#
#    print(data_pb.data_file)
#    print(data_pb.labels_file)
#
#    dh = BouncingMNISTDataHandler(data_pb)
#    data, labels = dh.GetBatch()
#    np.save(data_pb.dataset_name + '.npy', data)
#    np.save(data_pb.dataset_name + '_labels.npy', labels)


#if __name__ == '__main__':
#    main()
