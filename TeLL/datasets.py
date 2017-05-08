# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Template and parent classes for creating reader/loader classes for datasets

"""

import glob
import time
import numpy as np
import multiprocessing
import pandas as pd
from os import path
from PIL import Image
from TeLL.utility.misc import load_files_in_dir
from abc import ABCMeta, abstractmethod


class DataReader:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def batch_loader(self): pass
    
    @abstractmethod
    def close(self): pass


def load_image(image_path, image_scaling_factor, resampling_method):
    """Load an image from image_path, resize according to self.image_scaling_factor with resampling_method
    and apply optional function 'preprocessing'
    """
    # Load image
    with Image.open(image_path) as im:
        image = im.copy()
    
    # Resize if necessary
    if image_scaling_factor != 1:
        width, height = image.size
        resolution = (int(width / image_scaling_factor), int(height / image_scaling_factor))
        image = image.resize(resolution, resample=resampling_method)
    
    return image


def add_color_jittering(image, jitter):
    image = np.asarray(image, dtype=np.float32)
    
    r = np.clip((image[:, :, 0] + jitter[0])[:, :, None], 0, 255)
    g = np.clip((image[:, :, 1] + jitter[1])[:, :, None], 0, 255)
    b = np.clip((image[:, :, 2] + jitter[2])[:, :, None], 0, 255)
    
    image = np.concatenate((r, g, b), axis=2)
    
    return image


def add_luminance(image):
    """Calculate luminance and add it as channel to the input image"""
    image = np.asarray(image, dtype=np.float32)
    image = np.concatenate((image, (0.2126 * image[:, :, 0]
                                    + 0.7152 * image[:, :, 1]
                                    + 0.0722 * image[:, :, 2])[:, :, None]), axis=2)
    return image


def stretch_values(image):
    """Stretch values in an image/array to be within [0,1]"""
    image = np.asarray(image, dtype=np.float32)
    # Stretch pixel values from 0-1 (and make sure division is not by 0)
    image -= np.min(image)
    image /= (np.max(image) or 1)
    return image


def zoom_into_image(image: Image, zoom_factor: float, left_lower_corner: tuple = (0, 0), resample: int = Image.NEAREST):
    """Zoom into area of image (i.e. crop to area at position left_lower_corner and rescale area to original image size)
    
    Parameters
    -------
    image : PIL.Image
        PIL image
    
    zoom_factor : float
        Zoom into image with a factor zoom_factor >= 1.
    
    left_lower_corner: tuple
        Tuple with position of left lower corner of area to be zoomed into as (horizontal_pos, vertical_pos)
    
    resample: int
        Resampling filter to be used by PIL resize
    """
    if zoom_factor < 1.:
        raise ValueError("zoom_factor has to be >= 1. but is {}".format(zoom_factor))
    elif zoom_factor == 1.:
        return image
    
    full_size = image.size
    zoom_area_shape = [np.round(size / zoom_factor).astype(np.int) for size in full_size]
    crop_box = (left_lower_corner[0], left_lower_corner[1],
                left_lower_corner[0] + zoom_area_shape[0], left_lower_corner[1] + zoom_area_shape[1])
    zoom_area = image.crop(crop_box)  # Error in PIL documentation: crop_box is actually (left, lower, right, upper)!!!
    zoom_area = zoom_area.resize(full_size, resample=resample)
    
    return zoom_area


def ising_dropout(shape, generate: bool = False, directory: str = None, keep_prob: float = 0.5,
                  beta: float = 0.5, beta_step: float = 1.1, num_steps: int = 400, **kwargs):
    """Apply Ising dropout

    On the fly generation is not yet supported

    Parameters
    -------
    shape : vector
        Contains width and height of the image

    generate : bool
        If false, pictures from path are going to be sampled

    directory: string
        The directory where precomputed Ising images can be found
    """
    import os
    if generate:
        from TeLL.scripts.dropoutmask import make_ising_mask
        from PIL import Image
        import tensorflow as tf

        shape = np.asarray(shape[0:2])
        shape = np.insert(shape, 0, 1)
        shape = np.append(shape, 1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        ising_img_tf = make_ising_mask(shape=shape, keep_prob=keep_prob, num_steps=num_steps, beta=beta,
                                       beta_step=beta_step)

        sess.run(tf.global_variables_initializer())
        samples_np = sess.run(ising_img_tf)
        ising_img = Image.fromarray(samples_np[0, :, :, 0])
        sess.close()
    else:
        import random
        from PIL import Image

        random.seed(12345)
        ising_files = glob.glob(os.path.join(directory, '*.png'))
        ising_file = ising_files[random.randint(0, len(ising_files) - 1)]

        with Image.open(ising_file) as im:
            ising_img = im.copy()

        ising_img = ising_img.resize((shape[1], shape[0]), Image.ANTIALIAS)

    return ising_img


def ising_image_overlay_input(input_img, ising_img):
    input_img = np.asarray(input_img)
    ising_img = np.asarray(ising_img)

    # Sample and label image have different shapes (label has dim of 2)
    if input_img.shape[-1] is 3:
        return input_img * np.repeat(ising_img, 3).reshape((ising_img.shape[0], ising_img.shape[1], 3))
    else:
        return input_img * ising_img


def ising_image_overlay_label(input_img, ising_img, void_class=255):
    input_img = np.asarray(input_img)
    ising_img = np.asarray(ising_img)
    image_out = np.where(ising_img == 1, input_img, void_class)
    return image_out


class DatareaderSimpleFiles(object):
    def __init__(self, sample_directory: str, label_directory: str, batchsize: int, sample_suffix: str = '',
                 label_suffix: str = '', verbose: bool = True):
        """Dataset reader template
        
        Template for a dataset reader with background workers for loading the samples in sample_directory and the
        labels in label_directory into minibatches of size batchsize; Parts that need to be adapted to the specific
        tasks are indicated with "TODO"; Derive from this class to create new datareaders and overwrite __init__ and
        other functions as required for your task;
        
        Parameters
        -------
        sample_directory : str
            Path to input files
        label_directory : str
            Path to label files
        batchsize : int
            Batchsize
        sample_suffix : str
            Optional suffix to filter sample files by (e.g. '.png')
        label_suffix : str
            Optional suffix to filter label files by (e.g. '.png')
        """
        
        #
        # Search for and store the filenames
        #
        self.log("Collecting sample data...")
        samples = load_files_in_dir(sample_directory, sample_suffix)
        self.log("Found {} sample files".format(len(samples)))
        
        self.log("Collecting label data...")
        labels = load_files_in_dir(label_directory, label_suffix)
        self.log("Found {} label files".format(len(labels)))
        
        #
        # Calculate number of minibatches
        #
        n_mbs = np.int(np.ceil(len(labels) / batchsize))
        
        #
        # Determine shape of samples and labels
        #
        # TODO: Since this is a template, the shape determination has to modified to fit the specific task
        X_shape = (batchsize, 5)
        y_shape = (batchsize, 1)
        
        #
        # Set attributes of reader (these are required for the class to work properly)
        #
        self.verbose = verbose
        self.processes = list()
        self.samples = samples
        self.n_samples = len(samples)
        self.labels = labels
        self.n_labels = len(labels)
        self.batchsize = batchsize
        self.n_mbs = n_mbs
        self.X_shape = X_shape
        self.y_shape = y_shape
    
    def create_minibatch(self, mb_indices, mb_id):
        """This function shall load the data at label index mb_indices into one minibatch and has to be adapted to the
        specific task; It should return an object that will then automatically be returned for each iteration of
        batch_loader;
        
        Parameters
        -------
        mb_indices : int
            (Shuffled) index for self.labels to use in this minibatch
        mb_id : int
            Current minibatch number
            
        Returns
        ------
        : object
            Yields a new minibatch for each iteration over batch_loader
        """
        self.log("Fetching batch {}...".format(mb_id))
        
        # TODO: In this function you specify how you want to load or preprocess your data for the label-index mb_indices
        
        def some_loader_function(something_to_load):
            """Load and preprocess your data"""
            some_input, some_target, some_id = np.zeros((50, 7))
            return some_input, some_target, some_id
        
        # Reset minibatch values
        X, y, ID = some_loader_function(self.labels[mb_indices])
        
        return dict(X=X, y=y, ID=ID)
    
    def batch_loader(self, num_cached: int = 5, num_threads: int = 3, rnd_gen=None, shuffle=True):
        """Function to use for loading minibatches
        
        This function will start num_threads background workers to load the minibatches via the create_minibatch
        function; At most num_cached minibatches will be hold in memory at once per thread; The returned object is
        the minibatch that is yielded (i.e. this function can be iterated over);
        
        Parameters
        -------
        num_cached : int
            Maximum number of minibatches to be stored in memory at once per thread
        num_threads : int
            Number of background workers to use for loading the minibatches
        rnd_gen : numpy random generator or None
            Random generator to use for shuffling of samples; If None, a new numpy random generator will be created and
            used;
        shuffle : bool
            True: Shuffle the samples
            
        Yields
        ------
        : object
            Yields a new minibatch for each iteration over batch_loader
        
        Example
        ------
        >>> trainingset = DatareaderSimpleFiles(...)
        >>> mb_loader = trainingset.batch_loader(...):
        >>> for mb_i, mb in enumerate(mb_loader):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        #
        # Create queues and workers
        #
        mb_ind_queues = [multiprocessing.Queue(0) for _ in range(num_threads)]
        mb_queues = [multiprocessing.Queue(num_cached) for _ in range(num_threads)]
        self.log("Starting background loaders...", end=" ")
        
        for thread in range(num_threads):
            proc = multiprocessing.Process(target=self.__load_mb__, args=(mb_ind_queues[thread], mb_queues[thread]))
            proc.daemon = False
            proc.start()
            self.processes.append(proc)
        
        self.log("DONE")
        
        #
        # Get indices of valid samples to load
        #
        indices = np.arange(self.n_labels)
        
        # shuffle batches across minibatches
        self.log("  Shuffling samples...", end=" ")
        
        if shuffle:
            if rnd_gen is None:
                np.random.shuffle(indices)
            else:
                rnd_gen.shuffle(indices)
        
        self.log("DONE")
        
        minibatch_slices = [slice(i * self.batchsize, (i + 1) * self.batchsize) for i in np.arange(self.n_mbs)]
        
        # Put indices to be processed into queue
        self.log("  Filling input queue...", end=" ")
        
        thread = 0
        for mb_sl_i, mb_sl in enumerate(minibatch_slices):
            mb_ind_queues[thread].put([indices[mb_sl], mb_sl_i])
            thread += 1
            if thread >= num_threads:
                thread = 0
        
        # Put None at end of queue to signal end
        for thread in range(num_threads):
            mb_ind_queues[thread].put(None)
        
        self.log("DONE")
        
        # Get results from background workers, loop through different worker queues to keep order
        thread = 0
        for _ in minibatch_slices:
            # each subprocess returns a minibatch and its index in the procs list
            mb = mb_queues[thread].get()
            yield mb
            
            thread += 1
            if thread >= num_threads:
                thread = 0
        
        # Check if each worker has reached its end
        for thread in range(num_threads):
            if mb_queues[thread].get() is not None:
                raise ValueError("Error in queues!")
        
        # Close queue and processes
        for thread in range(num_threads):
            mb_ind_queues[thread].close()
            mb_queues[thread].close()
        self.close()
    
    def __load_mb__(self, in_queue, out_queue):
        """
        Load sample ids from in_queue and write loaded samples into out queue
        :param in_queue:
        :param out_queue:
        :return:
        """
        while True:
            input = in_queue.get()
            if input is None:
                self.log("Putting sentinel", end=" ")
                out_queue.put(None)
                self.log("Terminated")
                return 0
            
            mb_indices, mb_id = input
            
            minibatch = self.create_minibatch(mb_indices, mb_id)
            
            out_queue.put(minibatch)
            self.log("Fetched batch {}!".format(mb_id))
    
    def log(self, message, end="\n"):
        if self.verbose:
            print(message, end=end)
    
    def close(self):
        timeout = 10
        for proc in self.processes:
            try:
                start = time.time()
                proc.join(timeout)
                if time.time() - start >= timeout:
                    proc.terminate()
            except:
                self.log("Error when closing background worker")
            
            del proc
        self.processes.clear()


class DatareaderAdvancedImageFiles(object):
    def __init__(self,
                 sample_directory: str,
                 label_directory: str,
                 sample_suffix: str = None,
                 label_suffix: str = None,
                 batchsize: int = 128,
                 subset: bool = False,
                 preceding_frames: int = 0,
                 frame_steps: int = 1,
                 stretch_values: bool = True,
                 image_scaling_factor: float = 1,
                 add_luminance: bool = False,
                 add_color_jittering: bool = False,
                 add_zoom: bool = False,
                 add_flip: bool = False,
                 apply_ising: bool = False,
                 ising_params: str = None,
                 void_class=None,
                 id2label=None,
                 num_classes: int = None,
                 verbose: bool = True):
        """Dataset reader template, works with cityscapes Img8bit_sequence data

        Each sample contains of sample_len frames, with the last frame being labeled

        Parameters
        -------
        sample_directory : str
            Path to PNG input files
        label_directory : str
            Path to PNG label files
        label_suffix: str (optional)
            Suffix of the label file names(e.g. "_label.png")
        batchsize : int
            Batchsize
        num_classes : int
            Number of classes, not used if id2label dict is passed
        id2label : dict
            Dictionary mapping label ids to training ids, if passed number of samples will be set to length of this dict
        void_class : int
            Training id of void class (no error signal for pixels of this class); this class will NOT count towards the
             number of classes (i.e. reduces num_classes by one)
        subset:
            False or fraction of dataset to load
        add_zoom : None, float, int, or list
            If not False: Add zoomed input augmentation with factor add_zoom or loop through factors in add_zoom, if
            add_zoom is an array
        add_flip : bool

            If True: Add flipped input augmentation; add_zoom also applies to flipped images
        apply_ising : bool
            If True: An Ising filter is going to be deployed
        ising_params: 
            Additional parameters for Ising
        void_class: int or None
            If int: Use class void_class as void class, i.e. pixels of this class do not contribute to the loss
        """
        self.label_interpolation = Image.NEAREST
        self.input_interpolation = Image.BICUBIC

        self.preceding_frames = preceding_frames
        self.frame_steps = frame_steps
        self.resolution = None
        self.stretch_values = stretch_values
        self.image_scaling_factor = image_scaling_factor
        self.add_luminance = add_luminance
        if add_zoom and not isinstance(add_zoom, list):
            add_zoom = [add_zoom]
        self.add_zoom = add_zoom
        self.add_flip = add_flip
        self.add_color_jittering = add_color_jittering
        self.apply_ising = apply_ising
        self.ising_params = ising_params
        self.id2label = id2label

        #
        # Load list of labels and samples, store everything in dictionaries and prepare some stuff
        #

        # prepare list for background worker
        self.verbose = verbose
        self.batchsize = batchsize
        self.processes = list()

        if id2label is not None:
            self.num_classes = len(set(id2label.values()))
            if void_class is not None:
                self.num_classes -= 1
        elif num_classes is not None:
            self.num_classes = num_classes

        # Load label filenames
        self.log("Collecting label data...")
        self.labels, self.n_labels = self.load_file_dataframe(label_directory, label_suffix, subset=subset)
        if self.add_zoom or self.add_flip or self.add_color_jittering:
            self.labels, self.n_labels = self.prepare_augmented_dataframe(self.labels, flip=self.add_flip,
                                                                          jittering=self.add_color_jittering,
                                                                          zoom=self.add_zoom)

        # Load a single label to determine shape
        label_image = self.load_label_image(path=self.labels[0][0])
        self.label_image_shape = label_image.shape

        # Get coordinates for potential zoom
        if self.add_zoom:
            left_zoom_border = np.round(self.label_image_shape[-1] / 10).astype(np.int)
            right_zoom_border = np.round(self.label_image_shape[-1] / max(self.add_zoom) -
                                         left_zoom_border).astype(np.int)
            self.label_zoom_area = (np.linspace(left_zoom_border, right_zoom_border, num=5, dtype=np.int),
                                    [np.round(self.label_image_shape[-2] / 2).astype(np.int)])

        self.log("Found {} label images with shape {} (including augmentations)".format(self.n_labels,
                                                                                        self.label_image_shape))

        # Calculate number of minibatches
        self.n_mbs = np.int(np.ceil(self.n_labels / batchsize))
        if (self.n_labels % batchsize) != 0:
            raise AttributeError("Number of samples not dividable by minibatch-size! Please tell Michael to fiiiinally "
                                 "allow for this... ;)")

        # Load input image filenames
        self.log("Collecting input data...")
        self.samples, self.n_samples = self.load_file_dataframe(sample_directory, sample_suffix)

        # Load a single input image to determine shape
        input_image = self.load_input_image(path=self.samples[0][0])
        self.input_image_shape = input_image.shape

        # Get coordinates for potential zoom
        if self.add_zoom:
            left_zoom_border = np.round(self.input_image_shape[-2] / 10).astype(np.int)
            right_zoom_border = np.round(self.input_image_shape[-2] / max(self.add_zoom) -
                                         left_zoom_border).astype(np.int)
            self.input_zoom_area = (np.linspace(left_zoom_border, right_zoom_border, num=5, dtype=np.int),
                                    [np.round(self.input_image_shape[-3] / 2).astype(np.int)])

        if self.add_color_jittering:
            lower_bound = -20
            upper_bound = 20
            self.jitters = (np.roll(np.linspace(lower_bound, upper_bound, num=6, dtype=np.int), 2),
                            np.roll(np.linspace(upper_bound, lower_bound, num=6, dtype=np.int), 3),
                            np.concatenate((np.linspace(lower_bound, upper_bound / 2, num=3, dtype=np.int),
                                            np.linspace(upper_bound / 2, lower_bound, num=3, dtype=np.int)), axis=0))

        self.log("Found {} input images with shape {}".format(self.n_samples, self.input_image_shape))

        # structure of inputs will be (samples, x_axis, y_axis, channels)
        self.X_shape = (batchsize, preceding_frames + 1) + self.input_image_shape
        # structure of targets will be (samples, x_axis, y_axis, channels)
        self.y_shape = (batchsize,) + self.label_image_shape
        # structure of optional pixel-weights would be same as y_shape
        self.pixel_weights_shape = self.y_shape
        self.void_class = void_class

    def load_file_dataframe(self, directory: str, suffix: str = "", subset=False):
        """Load all filenames and file paths into pandas dataframe"""
        pattern = "**/*{}".format(suffix)

        # Collect files in path, sort them by name, and store them into dictionary
        file_paths = glob.glob(path.join(directory, pattern))
        file_paths.sort()

        # If subset is specified load only fraction of file_paths
        if subset:
            file_paths = file_paths[:int(len(file_paths) * subset)]

        # Extract keys that correspond to file base name without path and suffix
        keys = [path.basename(file)[:-len(suffix)] for file in file_paths]

        # Store in data frame for easy indexing and key->value resolution
        file_dataframe = pd.DataFrame(index=keys, data=file_paths)

        return file_dataframe, len(file_dataframe)

    def load_input_image(self, path, flip=False, zoom_factor=1, left_lower_corner=(0, 0), jitter=None):
        """Load a single input image"""
        image = load_image(image_path=path, image_scaling_factor=self.image_scaling_factor,
                           resampling_method=self.input_interpolation)

        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if zoom_factor != 1:
            image = zoom_into_image(image=image, zoom_factor=zoom_factor, left_lower_corner=left_lower_corner,
                                    resample=self.input_interpolation)

        if jitter:
            image = add_color_jittering(image, jitter)

        if self.add_luminance:
            image = add_luminance(image)

        if self.stretch_values:
            image = stretch_values(image)

        return np.asarray(image, dtype=np.float32)

    def load_label_image(self, path, flip=False, zoom_factor=1, left_lower_corner=(0, 0)):
        """Load a single label image"""
        image = load_image(image_path=path, image_scaling_factor=self.image_scaling_factor,
                           resampling_method=self.label_interpolation)

        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if zoom_factor != 1:
            image = zoom_into_image(image=image, zoom_factor=zoom_factor, left_lower_corner=left_lower_corner,
                                    resample=self.label_interpolation)

        label = np.asarray(image, dtype=np.uint8)
        
        if self.id2label is not None:
            temp = np.array(image)
            for k, v in self.id2label.items():
                temp[label == k] = v
            label = temp
        
        return label

    def load_sample(self, input_image_index: int, augmentation_params=None):
        """Load multiple input images as frames for a sample"""
        if augmentation_params is None:
            augmentation_params = dict()
        # Create slice for found input image and include prec_frames, suc_frames, and the frame step size
        sample_slice = slice(input_image_index - self.preceding_frames * self.frame_steps,
                             input_image_index + 1, self.frame_steps)
        # Use slice to get array of input image names, corresponding to the desired subsequence
        frames = self.samples.iloc[sample_slice, 0].values

        sample = np.empty((self.preceding_frames + 1,) + self.input_image_shape, dtype=np.float32)
        # Loop through subsequence filenames and load images into matrix X
        for frame_idx, frame_path in enumerate(frames):
            # Load image in read-only mode
            self.log("Loading Frame {}".format(frame_path))
            frame = self.load_input_image(path=frame_path, **augmentation_params)
            sample[frame_idx, :] = frame

        return sample

    def prepare_augmented_dataframe(self, dataframe, flip=True, zoom=True, jittering=True):
        """Duplicate dataframe and add columns for augmentations"""

        if flip:
            dataframe['flip'] = False
            flipped = dataframe.copy()
            flipped['flip'] = True
            dataframe = dataframe.append(flipped)

        if zoom:
            dataframe['zoom'] = False
            zoomed = dataframe.copy()
            zoomed['flip'] = True
            dataframe = dataframe.append(zoomed)

        if jittering:
            dataframe['jittering'] = False
            jittered = dataframe.copy()
            jittered['jittering'] = True
            dataframe = dataframe.append(jittered)

        return dataframe, len(dataframe)

    def get_class_occurrences(self, dset='train'):
        """ Return occurrences of classes in training, validation, and test set"""
        if dset == 'train':
            class_occ = np.array([718432047, 2036049361, 336031674, 1259776091, 36211223,
                                  48487162, 67768934, 11509943, 30521298, 878734354,
                                  63965201, 221459496, 67202363, 7444910, 386502819,
                                  14775009, 12995807, 12863940, 5445904, 22849664])
        elif dset == 'val':
            class_occ = np.array([131939476, 345222010, 49559045, 200895273, 6720678,
                                  7527026, 13564731, 1813749, 6110454, 158682893,
                                  7625936, 30708074, 11890229, 1970543, 59759319,
                                  2760469, 3564221, 1032100, 728922, 6500852])
        elif dset == 'test':
            # occurrences for the testset are unknown
            class_occ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        else:
            raise AttributeError("Occurrences of classes in dataset {} unknown!".format(dset))

        return class_occ

    def get_class_weights(self, dset='train', max_w=1.):
        """ Return factors for normalizing weights

        Example: loss_normed = loss_per_class * get_class_weights()
        """
        # get occurrences of classes
        class_occ = self.get_class_occurrences(dset)
        # get ratio of classes wrt largest class
        ratios = class_occ / class_occ.sum()
        # turn ratios into factors for normalization to a max factor of max_w
        return max_w * ratios.min() / ratios

    def log(self, message, end="\n"):
        if self.verbose:
            print(message, end=end)

    def __load_mb__(self, in_queue, out_queue):
        """
        Load sample ids from in_queue and write loaded samples into out queue
        :param in_queue:
        :param out_queue:
        :return:
        """
        while True:
            input = in_queue.get()
            if input is None:
                self.log("Putting sentinel", end=" ")
                out_queue.put(None)
                self.log("Terminated")
                return 0

            mb_samples, mb_nr = input

            # X will be [batchsize, frames, cameras, x, y, channel]
            batchsize = mb_samples.size
            X_shape = self.X_shape
            y_shape = self.y_shape

            self.log("Fetching batch {}...".format(mb_nr))

            # Reset minibatch values
            X = np.empty(X_shape, np.float32)
            y = np.empty(y_shape, np.uint8)
            ID = np.empty((batchsize,), np.object_)

            # Loop through all label indices in the current minibatch
            for i, label_ind in enumerate(mb_samples):

                # Find index of input image in self.samples dataframe corresponding to label key
                label_key = self.labels.index[label_ind]
                input_image_index = self.samples.index.get_loc(label_key)

                # Get augmentation specifications from dataframe
                sample_dataframe = self.labels.iloc[label_ind]
                label_path = sample_dataframe[0]
                flip = sample_dataframe.get('flip', False)
                zoom = sample_dataframe.get('zoom', False)
                jittering = sample_dataframe.get('jittering', False)

                input_augmentation_dict = dict(flip=flip)
                label_augmentation_dict = input_augmentation_dict.copy()
                if zoom:
                    input_augmentation_dict['zoom_factor'] = self.add_zoom[mb_nr % len(self.add_zoom)]
                    label_augmentation_dict['zoom_factor'] = self.add_zoom[mb_nr % len(self.add_zoom)]

                    hori_pos = self.label_zoom_area[0][mb_nr % len(self.label_zoom_area[0])]
                    vert_pos = self.label_zoom_area[1][mb_nr % len(self.label_zoom_area[1])]
                    vert_pos -= np.round(vert_pos / label_augmentation_dict['zoom_factor']).astype(np.int)
                    label_augmentation_dict['left_lower_corner'] = (hori_pos, vert_pos)

                    hori_pos = self.input_zoom_area[0][mb_nr % len(self.input_zoom_area[0])]
                    vert_pos = self.input_zoom_area[1][mb_nr % len(self.input_zoom_area[1])]
                    vert_pos -= np.round(vert_pos / input_augmentation_dict['zoom_factor']).astype(np.int)
                    input_augmentation_dict['left_lower_corner'] = (hori_pos, vert_pos)

                if jittering:
                    jitter_r = self.jitters[0][label_ind % len(self.jitters[0])]
                    jitter_g = self.jitters[1][label_ind % len(self.jitters[1])]
                    jitter_b = self.jitters[2][label_ind % len(self.jitters[2])]
                    input_augmentation_dict['jitter'] = (jitter_r, jitter_g, jitter_b)
                
                if self.apply_ising:
                    # Load/generate Ising image
                    ising_img = ising_dropout(shape=self.input_image_shape, **self.ising_params)

                    # Load sample into minibatch
                    sample_img = self.load_sample(input_image_index=input_image_index,
                                                  augmentation_params=input_augmentation_dict)
                    X[i, :] = ising_image_overlay_input(sample_img, ising_img)

                    # Load labeled image into correct sample position at y
                    if self.ising_params.get('apply_on_label', False):
                        label_img = self.load_label_image(path=label_path, **label_augmentation_dict)
                        y[i, :] = ising_image_overlay_label(label_img, ising_img, self.void_class)
                    else:
                        y[i, :] = self.load_label_image(path=label_path, **label_augmentation_dict)
                else:
                    # Load sample into minibatch
                    X[i, :] = self.load_sample(input_image_index=input_image_index,
                                               augmentation_params=input_augmentation_dict)

                    # Load labeled image into correct sample position at y
                    y[i, :] = self.load_label_image(path=label_path, **label_augmentation_dict)

                self.log("Key: {} Label: {} Sample: {}".format(label_key, label_path,
                                                               self.samples.iloc[input_image_index][0]))

                # Store label key into sample position at ID
                ID[i] = label_key

            if self.void_class is not None:
                pixel_weights = np.array(y != self.void_class, dtype=np.float32)
                y[y == self.void_class] = 0
                minibatch = dict(X=X, y=y, pixel_weights=pixel_weights, ID=ID)
            else:
                minibatch = dict(X=X, y=y, ID=ID)
            
            out_queue.put(minibatch)
            self.log("Fetched batch {}!".format(mb_nr))

    def batch_loader(self, num_cached=3, num_threads=3, rnd_gen=None, shuffle=True):
        """Function to use for loading minibatches
        
        This function will start num_threads background workers to load the minibatches; At most num_cached minibatches
        will be hold in memory at once per thread; The returned object is a dictionary with the minibatch that is
        yielded (i.e. this function can be iterated over);
        
        Parameters
        -------
        num_cached : int
            Maximum number of minibatches to be stored in memory at once
        num_threads : int
            Number of background workers to use for loading the minibatches
        rnd_gen : numpy random generator or None
            Random generator to use for shuffling of samples; If None, a new numpy random generator will be created and
            used;
        shuffle : bool
            True: Shuffle the samples
            
        Yields
        ------
        : object
            Yields a new minibatch for each iteration over batch_loader
        
        Example
        ------
        >>> trainingset = DatareaderSimpleFiles(...)
        >>> mb_loader = trainingset.batch_loader(...):
        >>> for mb_i, mb in enumerate(mb_loader):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        #
        # Create queues and workers
        #
        mb_ind_queues = [multiprocessing.Queue(0) for _ in range(num_threads)]
        mb_queues = [multiprocessing.Queue(num_cached) for _ in range(num_threads)]
        self.log("Starting background loaders...", end=" ")

        for thread in range(num_threads):
            proc = multiprocessing.Process(target=self.__load_mb__, args=(mb_ind_queues[thread], mb_queues[thread]))
            proc.daemon = False
            proc.start()
            self.processes.append(proc)

        self.log("DONE")

        #
        # Put indices in input queue
        #
        label_inds = np.arange(self.n_labels)

        # shuffle batches across minibatches
        self.log("  Shuffling samples...", end=" ")

        if shuffle:
            if rnd_gen is None:
                np.random.shuffle(label_inds)
            else:
                rnd_gen.shuffle(label_inds)

        self.log("DONE")

        minibatch_slices = [slice(i * self.batchsize, (i + 1) * self.batchsize) for i in np.arange(self.n_mbs)]

        # Put indices to be processed into queues, distribute them among workers to keep order
        self.log("  Filling input queue...", end=" ")

        thread = 0
        for mb_sl_i, mb_sl in enumerate(minibatch_slices):
            mb_ind_queues[thread].put([label_inds[mb_sl], mb_sl_i])
            thread += 1
            if thread >= num_threads:
                thread = 0

        # Put None at end of queue to signal end
        for thread in range(num_threads):
            mb_ind_queues[thread].put(None)

        self.log("DONE")

        # Get results from background workers, loop through different worker queues to keep order
        thread = 0
        for _ in minibatch_slices:
            # each subprocess returns a minibatch and its index in the procs list
            mb = mb_queues[thread].get()
            yield mb

            thread += 1
            if thread >= num_threads:
                thread = 0

        # Check if each worker has reached its end
        for thread in range(num_threads):
            if mb_queues[thread].get() is not None:
                raise ValueError("Error in queues!")

        # Close queue and processes
        for thread in range(num_threads):
            mb_ind_queues[thread].close()
            mb_queues[thread].close()
        self.close()

    def close(self):
        timeout = 10
        for proc in self.processes:
            try:
                start = time.time()
                proc.join(timeout)
                if time.time() - start >= timeout:
                    proc.terminate()
            except:
                self.log("Error when closing background worker")

            del proc
        self.processes.clear()


class MovingDotDataset(object):
    def __init__(self, dtype=np.float32, rnd_gen=np.random, batchsize=20, n_timesteps=20, n_samples=500, edge_size=35):
        """Example class containing dataset with video sequences in which two single pixels move on the x dimension
        in opposite directions;
        Frame resolution is edge_size x edge_size; Target is the next frame after the last frame;
        x_shape=(n_samples, n_timesteps, edge_size, edge_size, 1); y_shape=(n_samples, 1, edge_size, edge_size, 1);
        """
        
        # Two feature dimensions for the 2 moving directions, which will be merged into 1 dimension
        x = np.zeros((n_samples, n_timesteps + 1, edge_size, edge_size, 2), dtype=np.bool)
        
        #
        # Set an initial position for the two moving pixels
        #
        positions_pot = np.array(range(0, edge_size), dtype=np.int)
        x_positions = rnd_gen.choice(positions_pot, size=(n_samples, 2), replace=True)
        y_positions = rnd_gen.choice(positions_pot, size=(n_samples, 2), replace=True)
        
        #
        # Tile positions for all frames and move the x_positions
        #
        for s in range(n_samples):
            x_pos = np.arange(n_timesteps + 1, dtype=np.int) * 2 + x_positions[s, 0]
            x_pos %= edge_size
            y_pos = y_positions[s, 0]
            x[s, np.arange(n_timesteps + 1), x_pos, y_pos, 0] = 1
            
            x_pos[:] = np.arange(n_timesteps + 1, dtype=np.int) * 2 + x_positions[s, 1]
            x_pos = x_pos[::-1]
            x_pos %= edge_size
            y_pos = y_positions[s, 1]
            x[s, np.arange(n_timesteps + 1), x_pos, y_pos, 1] = 1
        
        #
        # Join the two feature dimension into one
        #
        x = np.array(np.logical_or(x[:, :, :, :, 0], x[:, :, :, :, 1]), dtype=np.float32)
        x = x[:, :, :, :, None]  # add empty feature dimension
        
        #
        # y is the next frame
        #
        y = x[:, -1:, :]
        x = x[:, :-1, :]
        
        self.x = x
        self.y = y
        self.ids = np.arange(n_samples)
        self.n_samples = n_samples
        self.batchsize = batchsize
        self.n_mbs = np.int(np.ceil(self.n_samples / batchsize))
        self.X_shape = (batchsize,) + self.x.shape[1:]
        self.y_shape = (batchsize,) + self.y.shape[1:]
    
    def batch_loader(self, rnd_gen=np.random, shuffle=True):
        """load_mbs yields a new minibatch at each iteration"""
        batchsize = self.batchsize
        inds = np.arange(self.n_samples)
        if shuffle:
            rnd_gen.shuffle(inds)
        n_mbs = np.int(np.ceil(self.n_samples / batchsize))
        
        x = np.zeros(self.X_shape, np.float32)
        y = np.zeros(self.y_shape, np.float32)
        ids = np.empty((batchsize,), np.object_)
        
        for m in range(n_mbs):
            start = m * batchsize
            end = (m + 1) * batchsize
            if end > self.n_samples:
                end = self.n_samples
            mb_slice = slice(start, end)
            
            x[:end - start, :] = self.x[inds[mb_slice], :]
            y[:end - start, :] = self.y[inds[mb_slice], :]
            ids[:end - start] = self.ids[inds[mb_slice]]
            
            yield dict(X=x, y=y, ID=ids)
    
    def close(self):
        return 0


class ShortLongDataset(object):
    def __init__(self, dtype=np.float32, rnd_gen=np.random, batchsize=20, n_timesteps=1000, n_samples=50000):
        """Class containing dataset with sequences where information over short and long subsequences need to be
        recognized
        
        sl = periodic signal with value 1 and frequency fl
        sh = random dirac peaks with peak distances pd [0, n_timesteps/3]
        pl = sum(sl)/wl/2 for half a wavelength wl
        ph = 1 if 0.5 < sl < 0.7
        
        x = abs(sl) + abs(sh)
        y = 0.5 < (pl + ph) / 2 < 0.75
        
        fl = n_timesteps / {5,6,7,8,9,10}
        fh = {1,2,3,4,5,6,7,8,9,10}
        """
        
        x = np.zeros((n_samples, n_timesteps, 2), dtype=dtype)
        y = np.zeros((n_samples, n_timesteps, 1), dtype=dtype)
        
        #
        # Create signal sl
        #
        # Pick random numbers for lengths of sinus waves
        fl_pot = np.array(range(5, 50))
        fl_picks = rnd_gen.choice(fl_pot, size=n_samples, replace=True)
        # Pick random offsets from starting position of sequence
        roll_picks = rnd_gen.choice(np.arange(0, np.max(fl_pot) * 2), size=n_samples, replace=True)
        
        for s_i in range(n_samples):
            fl = fl_picks[s_i]
            wave = np.array(np.sin(np.linspace(0, (np.pi - 1e-6) * 2, num=n_timesteps / fl)) > 0,
                            dtype=dtype)
            wave_bc = np.broadcast_to(wave, (fl, int(n_timesteps / fl)))
            x[s_i, :np.prod(wave_bc.shape), 0] = wave_bc.flatten()
            x[s_i, :, 0] = np.roll(x[s_i, :, 0], shift=roll_picks[s_i], axis=0)
            
            true_wave = np.array(wave, dtype=bool)
            wave[true_wave] = np.cumsum(wave[true_wave])
            wave /= np.max(wave)
            wave_cs = np.broadcast_to(wave, (fl, int(n_timesteps / fl)))
            y[s_i, :np.prod(wave_bc.shape), 0] = wave_cs.flatten()
            y[s_i, :, 0] = np.roll(y[s_i, :, 0], shift=roll_picks[s_i], axis=0)
        
        #
        # Create signal sh
        #
        pd_pot = np.array(range(5, int(n_timesteps / 10)))
        for s_i in range(n_samples):
            pd_pick = 0
            while True:
                pd_pick += rnd_gen.choice(pd_pot, size=1, replace=True)
                if pd_pick >= n_timesteps:
                    break
                x[s_i, pd_pick, 1] += 1
                y[s_i, pd_pick, 0] += 1
        
        y /= 2.
        y[:] = (y > 0.5) & (y < 0.75)
        
        self.x = x
        self.y = y
        self.ids = np.arange(n_samples)
        self.n_samples = n_samples
        self.batchsize = batchsize
        self.n_mbs = np.int(np.ceil(self.n_samples / batchsize))
        self.X_shape = (batchsize,) + self.x.shape[1:]
        self.y_shape = (batchsize,) + self.y.shape[1:]
    
    def batch_loader(self, rnd_gen=np.random, shuffle=True):
        """load_mbs yields a new minibatch at each iteration"""
        batchsize = self.batchsize
        inds = np.arange(self.n_samples)
        if shuffle:
            rnd_gen.shuffle(inds)
        n_mbs = np.int(np.ceil(self.n_samples / batchsize))
        
        x = np.zeros(self.X_shape, np.float32)
        y = np.zeros(self.y_shape, np.float32)
        ids = np.empty((batchsize,), np.object_)
        
        for m in range(n_mbs):
            start = m * batchsize
            end = (m + 1) * batchsize
            if end > self.n_samples:
                end = self.n_samples
            mb_slice = slice(start, end)
            
            x[:end - start, :] = self.x[inds[mb_slice], :]
            y[:end - start, :] = self.y[inds[mb_slice], :]
            ids[:end - start] = self.ids[inds[mb_slice]]
            
            yield dict(X=x, y=y, ID=ids)
    
    def close(self):
        return 0
