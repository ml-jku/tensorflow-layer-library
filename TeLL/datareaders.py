# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Template and parent classes for creating reader/loader classes for datasets

"""

import inspect
import threading
import time
from collections import OrderedDict, namedtuple
from os import path
from typing import Union

import numpy as np
from PIL import Image
from multiprocess import Process, Queue

from TeLL.config import Config
from TeLL.dataprocessing import DataProcessing
from TeLL.utility.misc import import_object
from TeLL.utility.timer import Timer


# ------------------------------------------------------------------------------------------------------------------
#  Initializer Function
# ------------------------------------------------------------------------------------------------------------------

def initialize_datareaders(config: Config, required: Union[str, list, tuple] = None):
    """
    Initializes datasets from config. Imports specified data reader and instantiates it with parameters from config.
    :param config: config
    :param required: string, list or tuple specifying which datasets have to be loaded (e.g. ["train", "val"])
    :return: initialized data readers
    """
    # Initialize Data Reader if specified
    readers = {}
    if config.has_value("dataset"):
        def to_list(value):
            if value is None:
                result = []
            elif isinstance(value, str):
                result = list([value])
            else:
                result = list(value)
            return result
        
        dataset = config.dataset
        required = to_list(required)
        
        try:
            reader_class = import_object(dataset["reader"])
            reader_args = inspect.signature(reader_class).parameters.keys()
            datasets = [key for key in dataset.keys() if key not in reader_args and key != "reader"]
            global_args = [key for key in dataset.keys() if key not in datasets and key != "reader"]
            
            # check for required datasets before loading anything
            if required is not None:
                required = to_list(required)
                missing = [d for d in required if d not in datasets]
                if len(missing) > 0:
                    raise Exception("Missing required dataset(s) {}".format(missing))
            
            # read "global" parameters
            global_pars = {}
            for key in global_args:
                value = dataset[key]
                global_pars[key] = value
                if isinstance(value, str) and "import::" in value:
                    global_pars[key] = import_object(value[len("import::"):])
            # read dataset specific parameters
            for dset in datasets:
                # inspect parameters and resolve if necessary
                for key, value in dataset[dset].items():
                    if isinstance(value, str) and "import::" in value:
                        dataset[dset][key] = import_object(value[len("import::"):])
                print("Loading dataset '{}'...".format(dset))
                readers[dset] = reader_class(**{**dataset[dset], **global_pars})
        except (AttributeError, TypeError) as e:
            print("Unable to import '{}'".format(e))
            raise e
    return readers


# ------------------------------------------------------------------------------------------------------------------
#  Classes
# ------------------------------------------------------------------------------------------------------------------

class DataReader:
    def __init__(self):
        """Base class for data readers; has to provide a read_sample(key) and get_sample_keys() class;

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader
        >>> # Define your DataReader class or use an existing one
        >>> reader = CityScapesReader(...)
        >>> # NOTE: CityScapesReader reads samples in a dictionary {'X': input image, 'y': label image, ...}
        >>>
        >>> # Stack some preprocessings for input images only
        >>> normalized = Normalize(reader, apply_to=['X'])
        >>>
        >>> # Stack some preprocessings for input and label images
        >>> zoomed = Zoom(normalized, apply_to=['X', 'y'])
        >>>
        >>> # Create a DataLoader instance
        >>> trainingset = DataLoader(data=zoomed, batchsize=5)
        >>>
        >>> # trainingset.batch_loader() will load your minibatches in background workers and yield them
        >>> for mb_i, mb in enumerate(trainingset.batch_loader(num_cached=5, num_threads=3)):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        pass
    
    def read_sample(self, key):
        """Read a single sample associated with 'key' from disk into dictionary;
        Dictionary keys can be associated with a preprocessing pipeline (see example below);
        For preprocessing, images or sequences of images should be numpy arrays of shape [frames, x, y, channels] or
        [x, y, channels] and pixel values should be in range [0, 1];"""
        pass
    
    def get_sample_keys(self):
        """Return a list of keys, where each key identifies a sample"""
        pass
    
    def get_num_classes(self):
        pass
    
    def log(self, message, end="\n"):
        if self.verbose:
            print(message, end=end)


class MovingMNIST(DataReader):
    def __init__(self, dset='train', scaling_factor=(0.5, 5.), scaling_velocity=(0., 5. / 20),
                 velocity=(28. / 5, 28. / 2), rotation_angle=(0., 360.), rotation_velocity=(0., 5.), canvas_shape=None,
                 n_timesteps=5, n_samples=5, resample=Image.BICUBIC, n_objects=2, object_shape=(28, 28),
                 random_seed: int = None):
        super(MovingMNIST).__init__()
        
        # Use tensorflow MNIST reader
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("./samples/MNIST_data", one_hot=False)
        
        if dset == 'train':
            samples = mnist.train
        elif dset == 'validation':
            samples = mnist.validation
        elif dset == 'test':
            samples = mnist.test
        else:
            raise AttributeError("mnist has not dset {}".format(dset))
        
        if random_seed is None:
            random_seed = np.random.randint(0, np.iinfo().max)
        rnd_gen = np.random.RandomState(random_seed)
        keys = rnd_gen.randint(low=0, high=samples.num_examples, size=(n_samples, n_objects), dtype=np.int)
        
        self.n_classes = 11
        
        self._object_shape = np.asarray(object_shape, dtype=np.int)
        self._keys = keys
        self._samples = samples
        self._n_samples = n_samples
        self._n_objects = n_objects
        self._rnd_gen = rnd_gen
        self._n_timesteps = n_timesteps
        if isinstance(scaling_factor, (tuple, list)):
            self._scaling_factor = scaling_factor
        else:
            self._scaling_factor = [scaling_factor]
        if isinstance(scaling_velocity, (tuple, list)):
            self._scaling_velocity = scaling_velocity
        else:
            self._scaling_velocity = [scaling_velocity]
        if isinstance(velocity, (tuple, list)):
            self._velocity = velocity
        else:
            self._velocity = [velocity]
        if isinstance(rotation_velocity, (tuple, list)):
            self._rotation_velocity = rotation_velocity
        else:
            self._rotation_velocity = [rotation_velocity]
        self._canvas_shape = canvas_shape
        self._resample = resample
        self._rotation_angle = rotation_angle
    
    def _rand_uniform(self, range_or_scalar, rnd_gen):
        """Return random """
        try:
            return rnd_gen.uniform(low=range_or_scalar[0], high=range_or_scalar[1])
        except (TypeError, IndexError):
            try:
                return range_or_scalar[0]
            except (TypeError, IndexError):
                return range_or_scalar
    
    def _rand_int(self, range_or_scalar, rnd_gen):
        """Return random """
        try:
            return rnd_gen.randint(low=range_or_scalar[0], high=range_or_scalar[1])
        except TypeError:
            return range_or_scalar
    
    def get_sample_keys(self):
        """Return a list of keys, where each key identifies a sample"""
        return self._keys
    
    def read_sample(self, key):
        # Get settings
        rnd_gen = self._rnd_gen
        n_timesteps = self._n_timesteps
        scaling_factor = self._scaling_factor
        scaling_velocity = self._scaling_velocity
        velocity = self._velocity
        rotation_velocity = self._rotation_velocity
        canvas_shape = self._canvas_shape
        resample = self._resample
        object_shape = self._object_shape
        # Get object class label
        target_numbers = [self._samples.labels[k] for k in key]
        # Maximum space to be used by rotated, scaled object
        object_space_max_len = np.int(np.ceil(np.max(object_shape) * 2. ** (1. / 2.)) * self._scaling_factor[-1])
        
        if canvas_shape is None:
            canvas_shape = np.array([object_space_max_len, object_space_max_len], dtype=np.int) * 2
        else:
            canvas_shape = np.array(canvas_shape, dtype=np.int)
        
        #
        # Pre-allocs
        #
        # Object as read from dataset (i.e. MNIST number)
        np_obj = np.empty(object_shape)
        # Canvas for painting current object in a frame
        cur_canvas = np.empty(tuple(canvas_shape), dtype=np.float32)
        # Inputs (added-up canvases for all timesteps)
        inputs = np.zeros((n_timesteps,) + tuple(canvas_shape), dtype=np.float32)
        # Inputs (overlayed canvases for all timesteps, larger numbers over lower numbers, -1)
        labels = np.zeros((n_timesteps,) + tuple(canvas_shape), dtype=np.int32)
        # Actions
        actions = np.zeros((n_timesteps, 2), dtype=np.float32)
        
        # --------------------------------------------------------------------------------------------------------------
        # Put object into canvas, starting with lower numbers
        # --------------------------------------------------------------------------------------------------------------
        target_numbers = list(zip(target_numbers, key))
        target_numbers.sort()
        for target_number, k in target_numbers:
            np_obj[:] = self._samples.images[k].reshape(tuple(object_shape))
            obj_orig = Image.fromarray(np_obj)
            
            #
            # Initial Randomization
            #
            # Rotation of the object
            cur_rot_angle = self._rand_uniform(self._rotation_angle, rnd_gen)
            cur_rotation_velocity = self._rand_uniform(rotation_velocity, rnd_gen)
            
            # Scaling of the object (pseudo movement on z axis)
            cur_scale = self._rand_uniform(scaling_factor, rnd_gen)
            cur_scaling_velocity = self._rand_uniform(scaling_velocity, rnd_gen)
            
            # Position in 2D plane (of object center)
            cur_position = rnd_gen.randint(low=int(np.ceil(object_space_max_len / 2)),
                                           high=int(canvas_shape[0] - object_space_max_len / 2), size=(2,),
                                           dtype=np.int)
            
            # Velocity (movement on 2D plane)
            cur_velocity = self._rand_uniform(velocity, rnd_gen)
            
            # Angle for movement on 2D plane
            cur_movement_angle = rnd_gen.uniform(low=0., high=1.) * 2 * np.pi
            
            # Bounce state (1 for standard direction, -1 for inverted direction, after bounce-off)
            cur_bounce = np.array([1, 1])
            
            for ts in range(n_timesteps):
                cur_canvas.fill(0)
                #
                # Update Movements
                #
                # Rotation of the object
                cur_rot_angle += cur_rotation_velocity  # apply velocity
                cur_rot_angle %= 360.  # keep range [0, 360] degrees, i.e. fully rotate, no "bounce rotation"
                # Scaling of the object (pseudo movement on z axis)
                cur_scale += cur_scaling_velocity  # apply velocity
                # Bounce of scaling_factor range edges
                try:
                    if cur_scale > scaling_factor[1]:
                        cur_scale = scaling_factor[1] * 2 - cur_scale
                        cur_scaling_velocity *= -1
                    elif cur_scale < scaling_factor[0]:
                        cur_scale = scaling_factor[0] * 2 - cur_scale
                        cur_scaling_velocity *= -1
                except IndexError:
                    pass
            
                if len(scaling_factor) == 2:
                    if cur_scale > scaling_factor[1]:
                        cur_scale = scaling_factor[1] * 2 - cur_scale
                        cur_scaling_velocity *= -1
                    elif cur_scale < scaling_factor[0]:
                        cur_scale = scaling_factor[0] * 2 - cur_scale
                        cur_scaling_velocity *= -1
                
                #
                # Rotation
                #
                # TODO: wtf is this value interpolation, from 0-255 to e.g. -37-306!!!
                obj = obj_orig.rotate(cur_rot_angle, resample=resample, expand=True)
                
                #
                # Scaling
                #
                obj = obj.resize(size=(int(obj.size[0] * cur_scale), int(obj.size[1] * cur_scale)), resample=resample)
                
                #
                # Positioning
                #
                
                # Get position change
                pos_delta = np.array(np.rint([np.sin(cur_movement_angle) * cur_velocity,
                                              np.cos(cur_movement_angle) * cur_velocity]),
                                     dtype=np.int)
                cur_position += cur_bounce * pos_delta
                # Position for upper left object corner
                cur_uc_position = np.asarray(np.rint(cur_position - (np.asarray(obj.size) / 2.)), dtype=np.int)
                
                # Bounce off canvas edges
                uc_canvas_shape = canvas_shape - obj.size  # valid canvas shape for upper left object corner
                
                cur_bounce[cur_uc_position > uc_canvas_shape] = -cur_bounce[cur_uc_position > uc_canvas_shape]
                cur_uc_position[cur_uc_position > uc_canvas_shape] = \
                    (uc_canvas_shape * 2 - cur_uc_position)[cur_uc_position > uc_canvas_shape]
                
                cur_bounce[cur_uc_position < 0] = -cur_bounce[cur_uc_position < 0]
                cur_uc_position[cur_uc_position < 0] = -cur_uc_position[cur_uc_position < 0]
                
                cur_position[:] = np.rint(cur_uc_position + (np.asarray(obj.size) / 2.))
                
                # Place object into canvas
                ends = cur_uc_position + obj.size
                cur_canvas[cur_uc_position[0]:ends[0], cur_uc_position[1]:ends[1]] = np.asarray(obj, dtype=np.float32)
                cur_canvas[cur_canvas < 0] = 0
                cur_canvas[cur_canvas > 1] = 1
                
                # Add canvas to inputs (inputs are the overlays of the objects with value)
                inputs[ts, :] += cur_canvas
                
                # Collaps all object label dimensions into 1 dimension, with the larger numbers "on top"
                labels[ts, cur_canvas > 0] = target_number + 1  # +1 because 0 is background

                # Store activation at t
                actions[ts, :] = cur_bounce * pos_delta
                labels[ts, cur_canvas > 0.5] = target_number + 1  # +1 because 0 is background
        
        inputs[inputs > 1.] = 1.
        
        return dict(X=np.expand_dims(inputs, axis=-1), y=labels, ID='_'.join([str(s) for s in key]),
                    actions=actions*(1./np.max(velocity)))


class CityScapesReader(DataReader):
    def __init__(self, input_directory: str, label_directory: str, disparity_directory: str = '',
                 input_suffix: str = '', label_suffix: str = '', disparity_suffix: str = '',
                 subset: float = 1., preceding_frames: int = 0, frame_steps: int = 1, void_class=None, id2label=None,
                 num_classes: int = None, verbose: bool = False):
        """Dataset reader for CityScapes

        Reads samples in as dictionaries {'X': input_sequence, 'y': label, ['pixel_weights': pixel_weights,]
        'ID': sample_ID}.

        input_sequence: [n_frames, x, y, channels]
            Input image or frames
        label: [x, y, channels]
            Label image
        pixel_weights: [x, y, channels]
            Optional weights for loss weighting (and ignoring void class)
        ID: []
            ID/Key of current sample

        Parameters
        -------
        input_directory : str
            Path to input files
        label_directory : str
            Path to label files
        input_suffix : str
            Optional suffix to filter sample files by (e.g. '.png')
        label_suffix : str
            Optional suffix to filter label files by (e.g. '.png')
        subset : float
            Fraction of labels to use; Has to be in range [0, 1];
        preceding_frames : int
            Additional input frames before labeled frame
        frame_steps : int
            Allows to skip frames; defaults to 1 (=take every frame);
        image_scaling_factor
            Factor for image scaling
        void_class : int or None
            If int: Use class void_class as void class, i.e. pixels of this class should not contribute to the loss;
            This will result in another sample entry with key "pixel_weights" and the same dimensions as the label y,
            where "pixel_weights" is 0 if y==void_class and 1 otherwise; This class will NOT count towards the number
            of classes (i.e. reduces num_classes by one);
        id2label : dict
            Dictionary mapping label ids to training ids; if passed, number of samples will be set to length of this
            id2label;
        num_classes : int or None
            Number of classes in dataset
        """
        super(CityScapesReader).__init__()
        
        self.id2label = id2label
        self.void_class = void_class
        self.verbose = verbose
        
        self.preceding_frames = preceding_frames
        self.frame_steps = frame_steps
        
        if id2label is not None:
            self.num_classes = len(set(id2label.values()))
            if void_class is not None:
                self.num_classes -= 1
        elif num_classes is not None:
            self.num_classes = num_classes
        
        #
        # Search for and store the filenames
        #
        # Load sample filenames
        self.log("Collecting sample data...")
        inputs = self.load_file_dataframe(input_directory, input_suffix)
        n_inputs = len(inputs)
        input_keys = list(inputs.keys())
        self.log("Found {} sample files".format(n_inputs))
        
        if n_inputs == 0:
            raise Exception("Empty dataset!")
        
        # Load a single input image to determine shape
        input_image = self.load_image(filepath=inputs[input_keys[0]])
        self.input_image_shape = input_image.shape
        
        self.inputs = inputs
        self.n_inputs = n_inputs
        self.input_keys = input_keys.copy()
        self.inputkey_to_index = OrderedDict(zip(input_keys, range(len(input_keys))))
        
        # Load label filenames
        self.log("Collecting label data...")
        labels = self.load_file_dataframe(label_directory, label_suffix, subset=subset)
        n_labels = len(labels)
        label_keys = list(labels.keys())
        self.log("Found {} label files within a subset of {}".format(n_labels, subset))
        
        # Load a single label image to determine shape
        label_image = self.load_image(filepath=labels[label_keys[0]])
        self.label_image_shape = label_image.shape
        
        self.labels = labels
        self.n_labels = n_labels
        self.label_keys = label_keys.copy()
        self.labelkey_to_index = OrderedDict(zip(label_keys, range(len(label_keys))))
        
        if disparity_directory is not None and len(disparity_directory) > 0:
            # Load disparity filenames
            self.log("Collecting disparity data...")
            disparities = self.load_file_dataframe(disparity_directory, disparity_suffix, subset=subset)
            n_disparities = len(disparities)
            disparity_keys = list(disparities.keys())
            self.log("Found {} disparity files within a subset of {}".format(n_disparities, subset))
            
            # Load a single disparity image to determine shape
            disparity_image = self.load_image(filepath=disparities[disparity_keys[0]])
            self.disparity_image_shape = disparity_image.shape + (1,)
            
            self.disparities = disparities
            self.n_disparities = n_disparities
            self.disparity_keys = disparity_keys.copy()
            self.disparitykey_to_index = OrderedDict(zip(disparity_keys, range(len(disparity_keys))))
    
    def read_sample(self, key):
        input_sequence = self.load_input_sequence(key=key)
        
        if hasattr(self, "disparity_keys"):
            disparity_sequence = self.load_input_sequence(key=key, key_list=self.disparity_keys,
                                                          file_names=self.disparities,
                                                          key_to_index=self.disparitykey_to_index,
                                                          image_shape=self.disparity_image_shape)
        
        label = self.load_image(filepath=self.labels[key])
        label = np.expand_dims(np.expand_dims(label, axis=0), axis=-1)
        
        if self.id2label is not None:
            temp = np.array(label)
            for k, v in self.id2label.items():
                temp[label == k] = v
            label = temp
        
        if self.void_class is not None:
            pixel_weights = np.array(label != self.void_class, dtype=np.float32)
            label[label == self.void_class] = np.unique(label)[np.where(np.unique(label) != self.void_class)][0]
            sample = dict(X=input_sequence, y=label, pixel_weights=pixel_weights, ID=key)
        else:
            sample = dict(X=input_sequence, y=label, ID=key)
        
        if hasattr(self, "disparity_keys"):
            sample['disparity'] = disparity_sequence
        
        return sample
    
    def get_sample_keys(self):
        return self.label_keys.copy()
    
    def get_num_classes(self):
        return self.num_classes
    
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
    
    def load_file_dataframe(self, directory: str, suffix: str = "", subset: float = 1.):
        """Load all filenames and file paths into OrderedDict"""
        
        from glob import glob
        
        sample_pattern = "**/*{}".format(suffix)
        file_paths = glob(path.join(directory, sample_pattern))
        file_paths.sort()
        
        if subset != 1.:
            file_paths = file_paths[:int(len(file_paths) * subset)]
        
        # Extract keys that correspond to file base name without path and suffix
        if len(suffix) is 0:
            keys = [path.basename(file) for file in file_paths]
        else:
            keys = [path.basename(file)[:-len(suffix)] for file in file_paths]
        
        # Store in OrderedDict
        file_dict = OrderedDict(zip(keys, file_paths))
        
        # TODO: Check if sorting is correct
        return file_dict
    
    def load_image(self, filepath):
        """Load a single input image from full filepath+name 'filepath'"""
        # Load image
        with Image.open(filepath) as im:
            image = im.copy()
        
        if image.mode in ('I', 'F'):
            arr = np.asarray(image, dtype=np.float32)
            arr_min = np.min(arr)
            arr = (arr - arr_min) / ((np.max(arr) - arr_min) / np.iinfo(np.uint8).max)
            arr = np.round(arr).astype(np.uint8)
        else:
            arr = np.asarray(image, dtype=np.uint8)
        
        return arr
    
    def load_input_sequence(self, key, key_list=None, file_names=None, key_to_index=None, image_shape=None):
        """Load multiple input images as frames ending at position 'input_index' in self.label_keys"""
        
        if key_list is None:
            key_list = self.input_keys
        if file_names is None:
            file_names = self.inputs
        if key_to_index is None:
            key_to_index = self.inputkey_to_index
        if image_shape is None:
            image_shape = self.input_image_shape
        
        # Get position of key in key list
        index = key_to_index[key]
        
        # Create slice for input image and include prec_frames, suc_frames, and the frame step size
        frames_slice = slice(index - self.preceding_frames * self.frame_steps,
                             index + 1, self.frame_steps)
        
        # Use slice to get list of input image names, corresponding to the desired subsequence
        frames = key_list[frames_slice]
        
        input_sequence = np.empty((self.preceding_frames + 1,) + image_shape, dtype=np.uint8)
        
        # Loop through subsequence filenames and load images into input_sequence
        for frame_idx, frame_path in enumerate(frames):
            # Load image in read-only mode
            self.log("Loading Frame {}".format(frame_path))
            frame = self.load_image(filepath=file_names[frame_path])
            input_sequence[frame_idx, :] = np.reshape(frame, image_shape)
        
        return input_sequence


class DataLoader(object):
    def __init__(self, data: DataReader or DataProcessing, batchsize: int, batchsize_method='zeropad',
                 verbose: bool = False):
        """Dataset loader for loading minibatches and applying preprocessing in background

        Parameters
        -------
        data : TeLL.datareaders.DataReader object or TeLL.dataprocessing.DataProcessing object
            Datareader or preprocessing pipeline to use for reading the files from the disk (preprocessing will be done
            on CPU in backgroundworkers).
        batchsize : int
            Number of samples in minibatch
        batchsize_method : str
            How to deal with only partially filled minibatches;
            'drop': Ignore incomplete minibatches, reduce number of samples after shuffling;
            'zeropad': Keep minibatch size static, pad missing samples with zeros, and set weights for these samples
            to '0';

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader
        >>> # Define your DataReader class or use an existing one
        >>> reader = CityScapesReader(...)
        >>> # NOTE: CityScapesReader reads samples in a dictionary {'X': input image, 'y': label image, ...}
        >>>
        >>> # Stack some preprocessings for input images only
        >>> normalized = Normalize(reader, apply_to=['X'])
        >>>
        >>> # Stack some preprocessings for input and label images
        >>> zoomed = Zoom(normalized, apply_to=['X', 'y'])
        >>>
        >>> # Create a DataLoader instance
        >>> trainingset = DataLoader(data=zoomed, batchsize=5)
        >>>
        >>> # trainingset.batch_loader() will load your minibatches in background workers and yield them
        >>> for mb_i, mb in enumerate(trainingset.batch_loader(num_cached=5, num_threads=3)):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        self.verbose = verbose
        
        #
        # Get datareader
        #
        if isinstance(data, DataReader):
            data = DataProcessing(incoming=data)
        datareader = data.datareader
        
        #
        # Get processing pipeline (list of lists with columns [sample keys, [processing_fct, kwargs]]) for each key
        # in sample-dict
        #
        example_processing_list = data.get_processing_list()
        example_sample_entry = example_processing_list[0]
        example_sample = datareader.read_sample(example_sample_entry[0])
        n_samples = len(example_processing_list)
        processing_list = data.get_processing_list()
        #
        # Calculate number of minibatches
        #
        if batchsize_method == 'drop':
            # Round towards 0
            n_mbs = np.int(n_samples / batchsize)
        elif batchsize_method == 'zeropad':
            # Round towards inf
            n_mbs = np.int(np.ceil(n_samples / batchsize))
        else:
            raise ValueError("Unknown option {} as batchsize_method for DataLoader".format(batchsize_method))
        
        #
        # Determine shape of elements in sample
        #  (Take sample 0, load, and process its elements to get their dimensions)
        #
        mb_info = dict()
        temp_id, temp_proc_fct = processing_list[0]
        input_info = namedtuple("InputInfo", "shape dtype")
        for sk in example_sample.keys():
            temp_processed = temp_proc_fct(example_sample[sk], apply_key=sk)
            shape = (batchsize,) + tuple(np.shape(temp_processed))
            try:
                dtype = temp_processed.dtype
            except AttributeError:
                dtype = type(temp_processed)
                if dtype is str:
                    dtype = np.object
            info = input_info(shape=shape, dtype=dtype)
            # Store into as mb_info=dict(mb_key=[shape, dtype])
            mb_info[sk] = info
        
        self.log("Minibatch-Info: {} (determined from sample with ID {})".format(mb_info, temp_id))
        
        #
        # Set attributes of reader (these are required for the class to work properly)
        #
        self.processes = list()
        self.batchsize_method = batchsize_method
        
        self.batchsize = batchsize
        self.n_mbs = n_mbs
        self.mb_info = mb_info
        
        self.sample_keys = [p[0] for p in example_processing_list]
        self.n_samples = n_samples
        self.num_classes = datareader.get_num_classes()
        
        self.processing_list = processing_list
        self.datareader = datareader
    
    def get_input_shapes(self):
        return self.mb_info
    
    def create_minibatch(self, mb_indices, mb_id, rnd_gen):
        """This function shall load the data at label index mb_indices into one minibatch and has to be adapted to the
        specific task; It should return an object that will then automatically be returned for each iteration of
        batch_loader;

        Parameters
        -------
        mb_indices : iterable
            (Shuffled) indices for sample to use in this minibatch
        mb_id : int
            Current minibatch number

        Returns
        ------
        : object
            Yields a new minibatch for each iteration over batch_loader
        """
        self.log("Fetching batch {}...".format(mb_id))
        minibatch = self.mb_info.copy()
        
        for key in minibatch.keys():
            minibatch[key] = np.empty(shape=minibatch[key][0], dtype=minibatch[key][1])
        
        for mb_ind, sample_ind in enumerate(mb_indices):
            if rnd_gen is None:
                rnd_gen = np.random
            rnd_key = rnd_gen.randint(0, np.iinfo(np.int32).max)  # random seed for all particles of a sample
            # compile processing list
            sample_key, processing = self.processing_list[sample_ind]
            # load sample
            sample = self.datareader.read_sample(sample_key)
            # push through processing graph
            for mb_key in minibatch.keys():
                minibatch[mb_key][mb_ind] = processing(sample[mb_key], apply_key=mb_key, rnd_key=rnd_key)
        
        if len(mb_indices) < self.batchsize:
            for key in minibatch.keys():
                minibatch[key][len(mb_indices):] = 0
        
        return minibatch
    
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
        >>> trainingset = DataLoader(...)
        >>> mb_loader = trainingset.batch_loader(...):
        >>> for mb_i, mb in enumerate(mb_loader):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        #
        # Create queues and workers
        #
        mb_ind_queues = [Queue(0) for _ in range(num_threads)]
        mb_queues = [Queue(num_cached) for _ in range(num_threads)]
        self.log("Starting background loaders...", end=" ")
        if rnd_gen is None:
            rnd_gen = np.random
        
        local_rng = np.random.RandomState(rnd_gen.randint(np.iinfo(np.int32).max))
        for thread in range(num_threads):
            thread_rng = np.random.RandomState(local_rng.randint(np.iinfo(np.int32).max))
            proc = Process(target=self.__load_mb__, args=(mb_ind_queues[thread], mb_queues[thread], thread_rng))
            proc.daemon = False
            proc.start()
            self.processes.append(proc)
        
        self.log("DONE")
        
        #
        # Get indices of valid samples to load
        #
        indices = np.arange(self.n_samples)
        
        # Shuffle batches across minibatches
        self.log("  Shuffling samples...", end=" ")
        
        if shuffle:
            rnd_gen.shuffle(indices)
        
        self.log("DONE")
        
        # Drop indices if batchsize_method = 'drop'
        if self.batchsize_method == 'drop':
            self.log("  Dropping {} samples after shuffling, since batchsize_method == 'drop'...".format(
                self.n_samples - self.n_mbs * self.batchsize), end=" ")
            indices = indices[:self.n_mbs * self.batchsize]
        
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
    
    def __load_mb__(self, in_queue, out_queue, rnd_gen):
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
            
            minibatch = self.create_minibatch(mb_indices, mb_id, rnd_gen=rnd_gen)
            
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


class TFQueueLoader(object):
    # import tensorflow only if this loader is used
    tf = __import__('tensorflow')
    
    def __init__(self, data: DataReader or DataProcessing, batchsize: int, queue_capacity: int = 10,
                 batchsize_method='zeropad', verbose: bool = True):
        """Dataset loader for loading minibatches and applying preprocessing in background

        Parameters
        -------
        data : TeLL.datareaders.DataReader object or TeLL.dataprocessing.DataProcessing object
            Datareader or preprocessing pipeline to use for reading the files from the disk (preprocessing will be done
            on CPU in backgroundworkers).

        batchsize : int
            Number of samples in minibatch
        batchsize_method : str
            How to deal with only partially filled minibatches;
            'drop': Ignore incomplete minibatches, reduce number of samples after shuffling;
            'zeropad': Keep minibatch size static, pad missing samples with zeros, and set weights for these samples
            to '0';

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader
        >>> # Define your DataReader class or use an existing one
        >>> reader = CityScapesReader(...)
        >>> # NOTE: CityScapesReader reads samples in a dictionary {'X': input image, 'y': label image, ...}
        >>>
        >>> # Stack some preprocessings for input images only
        >>> normalized = Normalize(reader, apply_to=['X'])
        >>>
        >>> # Stack some preprocessings for input and label images
        >>> zoomed = Zoom(normalized, apply_to=['X', 'y'])
        >>>
        >>> # Create a DataLoader instance
        >>> trainingset = DataLoader(data=zoomed, batchsize=5)
        >>>
        >>> # trainingset.batch_loader() will load your minibatches in background workers and yield them
        >>> for mb_i, mb in enumerate(trainingset.batch_loader(num_cached=5, num_threads=3)):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        self.verbose = verbose
        
        #
        # Get datareader
        #
        if isinstance(data, DataReader):
            data = DataProcessing(incoming=data)
        datareader = data.datareader
        
        #
        # Get processing pipeline (list of lists with columns [sample keys, [processing_fct, kwargs]]) for each key
        # in sample-dict
        #
        example_processing_list = data.get_processing_list()
        example_sample_entry = example_processing_list[0]
        example_sample = datareader.read_sample(example_sample_entry[0])
        n_samples = len(example_processing_list)
        processing_list = data.get_processing_list()
        #
        # Calculate number of minibatches
        #
        if batchsize_method == 'drop':
            # Round towards 0
            n_mbs = np.int(n_samples / batchsize)
        elif batchsize_method == 'zeropad':
            # Round towards inf
            n_mbs = np.int(np.ceil(n_samples / batchsize))
        else:
            raise ValueError("Unknown option {} as batchsize_method for DataLoader".format(batchsize_method))
        
        #
        # Determine shape of elements in sample
        #  (Take sample 0, load, and process its elements to get their dimensions)
        #
        mb_info = dict()
        input_info = namedtuple("InputInfo", "shape dtype")
        input_placeholder = OrderedDict()
        temp_id, temp_proc_fct = processing_list[0]
        for sk in example_sample.keys():
            temp_processed = temp_proc_fct(example_sample[sk], apply_key=sk)
            shape = (batchsize,) + tuple(np.shape(temp_processed))
            try:
                if temp_processed.dtype == np.uint8:
                    dtype = self.tf.int32
                else:
                    dtype = self.tf.as_dtype(temp_processed.dtype)
            except AttributeError:
                if type(temp_processed) == str:
                    dtype = self.tf.string
                else:
                    raise TypeError("Unsupported type: {}".format(type(temp_processed)))
            # Store into as mb_info=dict(mb_key=[shape, dtype])
            mb_info[sk] = input_info(shape=shape, dtype=dtype)
            input_placeholder[sk] = self.tf.placeholder(dtype, shape=shape, name=sk)
        
        self.log("Minibatch-Info: {} (determined from sample with ID {})".format(mb_info, temp_id))
        
        # create tensorflow queue
        self.tf_queue = self.tf.PaddingFIFOQueue(capacity=batchsize * queue_capacity,
                                                 dtypes=[t.dtype for t in input_placeholder.values()],
                                                 shapes=[t.shape for t in input_placeholder.values()],
                                                 names=[t for t in input_placeholder.keys()])
        self.enqueue = self.tf_queue.enqueue(input_placeholder)
        self.dequeue = self.tf_queue.dequeue()
        
        #
        # Set attributes of reader (these are required for the class to work properly)
        #
        self.processes = list()
        self.batchsize_method = batchsize_method
        
        self.batchsize = batchsize
        self.n_mbs = n_mbs
        self.input_placeholder = input_placeholder
        self.input_description = mb_info
        
        self.sample_keys = [p[0] for p in example_processing_list]
        self.n_samples = n_samples
        self.num_classes = datareader.get_num_classes()
        
        self.processing_list = processing_list
        self.datareader = datareader
    
    def get_input_shapes(self):
        return self.input_description
    
    def get_input_placeholders(self):
        return self.input_placeholder
    
    def create_minibatch(self, mb_indices, mb_id, rnd_gen):
        """This function shall load the data at label index mb_indices into one minibatch and has to be adapted to the
        specific task; It should return an object that will then automatically be returned for each iteration of
        batch_loader;

        Parameters
        -------
        mb_indices : iterable
            (Shuffled) indices for sample to use in this minibatch
        mb_id : int
            Current minibatch number

        Returns
        ------
        : object
            Yields a new minibatch for each iteration over batch_loader
        """
        self.log("Fetching batch {}...".format(mb_id))
        minibatch = self.input_placeholder.copy()
        
        for key in minibatch.keys():
            if minibatch[key].dtype == self.tf.string:
                dtype = np.object
            else:
                dtype = minibatch[key].dtype.as_numpy_dtype()
            minibatch[key] = np.empty(shape=minibatch[key].get_shape().as_list(), dtype=dtype)
        
        with Timer(verbose=True, name="Processing Batch", precision="msec"):
            for mb_ind, sample_ind in enumerate(mb_indices):
                if rnd_gen is None:
                    rnd_gen = np.random
                rnd_key = rnd_gen.randint(0, np.iinfo(np.int32).max)  # random seed for all particles of a sample
                # compile processing list
                sample_key, processing = self.processing_list[sample_ind]
                # load sample
                sample = self.datareader.read_sample(sample_key)
                # push through processing graph
                for mb_key in minibatch.keys():
                    minibatch[mb_key][mb_ind] = processing(sample[mb_key], apply_key=mb_key, rnd_key=rnd_key)
        
        if len(mb_indices) < self.batchsize:
            for key in minibatch.keys():
                if minibatch[key].dtype == np.object:
                    minibatch[key][len(mb_indices):] = ''
                else:
                    minibatch[key][len(mb_indices):] = 0
        
        return minibatch
    
    def batch_loader(self, session, epochs: int = 1, num_cached: int = 5, num_threads: int = 3, rnd_gen=None,
                     shuffle=True):
        """Function to use for loading minibatches

        This function will start num_threads background workers to load the minibatches via the create_minibatch
        function; At most num_cached minibatches will be hold in memory at once per thread; The returned object is
        the minibatch that is yielded (i.e. this function can be iterated over);

        Parameters
        -------
        tf_session : Tensorflow Session
            Current Tensorflow session
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
        >>> trainingset = DataLoader(...)
        >>> mb_loader = trainingset.batch_loader(...):
        >>> for mb_i, mb in enumerate(mb_loader):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        #
        # Create queues and workers
        #
        mb_ind_queues = [Queue(0) for _ in range(num_threads)]
        mb_queues = [Queue(num_cached) for _ in range(num_threads)]
        
        if rnd_gen is None:
            rnd_gen = np.random
        # start background preprocessors
        thread_rng = np.random.RandomState(rnd_gen.randint(np.iinfo(np.int32).max))
        self.__start_bg_workers__(num_threads, mb_ind_queues, mb_queues, thread_rng)
        
        # prepare minibatch slices for all epochs
        self.log("  Filling input queue...", end=" ")
        minibatch_slices = []
        # fill queue with minibatch slices for all epochs
        for e in range(0, epochs):
            indices = np.arange(self.n_samples)
            
            # Shuffle batches across minibatches
            if shuffle:
                rnd_gen.shuffle(indices)
            
            # Drop indices if batchsize_method = 'drop'
            if self.batchsize_method == 'drop':
                self.log("  Dropping {} samples after shuffling, since batchsize_method == 'drop'...".format(
                    self.n_samples - self.n_mbs * self.batchsize), end=" ")
                indices = indices[:self.n_mbs * self.batchsize]
            
            minibatch_slices.extend(
                [slice(i * self.batchsize, (i + 1) * self.batchsize) for i in np.arange(self.n_mbs)])
        
        # Consume minibatches from background workers and put them into tensorflow queue
        self.log("  Starting tensorflow queue pipe...", end=" ")
        tf_queue_transport = threading.Thread(
            target=self.__fill_queues__,
            args=(session, minibatch_slices, num_threads, mb_ind_queues, mb_queues))
        tf_queue_transport.daemon = False
        tf_queue_transport.start()
        self.log("DONE")
        
        # Put indices to be processed into queues (round robin)
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
        
        return len(minibatch_slices)
    
    def __start_bg_workers__(self, num_threads, mb_ind_queues, mb_queues, rnd_gen):
        self.log("Starting background loaders...", end=" ")
        
        for thread in range(num_threads):
            proc = threading.Thread(target=self.__load_mb__, args=(
                mb_ind_queues[thread], mb_queues[thread],
                np.random.RandomState(rnd_gen.randint(np.iinfo(np.int32).max))))
            # proc = Process(target=self.__load_mb__, args=(mb_ind_queues[thread], mb_queues[thread], rnd_gen))
            proc.daemon = False
            proc.start()
            self.processes.append(proc)
        
        self.log("DONE")
    
    def __fill_queues__(self, session, minibatch_slices, num_threads, mb_ind_queues, mb_queues):
        # Get results from background workers, loop through different worker queues to keep order
        thread = 0
        for _ in minibatch_slices:
            # each subprocess returns a minibatch and its index in the procs list
            mb = mb_queues[thread].get()
            if mb is not None:
                feed_dict = {}
                for k, v in mb.items():
                    feed_dict[self.input_placeholder[k]] = v
                
                session.run(self.enqueue, feed_dict=feed_dict)
            
            thread += 1
            if thread >= num_threads:
                thread = 0
        
        # Check if each worker has reached its end
        for thread in range(num_threads):
            if mb_queues[thread].get() is not None:
                raise ValueError("Error in queues!")
        
        # Close queue and processes
        self.log("Data loader finished")
        for thread in range(num_threads):
            mb_ind_queues[thread].close()
            mb_queues[thread].close()
        self.close()
    
    def __load_mb__(self, in_queue, out_queue, rnd_gen):
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
            
            minibatch = self.create_minibatch(mb_indices, mb_id, rnd_gen=rnd_gen)
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
                if not isinstance(proc, threading.Thread) and time.time() - start >= timeout:
                    proc.terminate()
            except:
                self.log("Error when closing background worker")
            
            del proc
        self.processes.clear()


class MNISTReader(DataReader):
    def __init__(self, dset='train'):
        """Wrapper class for tensorflow MNIST reader in TeLL DataReader format"""
        super(MNISTReader, self).__init__()
        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        
        #
        # Load Data
        #
        with Timer(name="Load data"):
            self.mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
        self.dset = dset
    
    def read_sample(self, key):
        """Read a single sample associated with 'key' from disk into dictionary;
        Dictionary keys can be associated with a preprocessing pipeline (see example below);
        For preprocessing, images or sequences of images should be numpy arrays of shape [frames, x, y, channels] or
        [x, y, channels] and pixel values should be in range [0, 1];"""
        if self.dset == 'train':
            sample = self.mnist.train
        elif self.dset == 'validation':
            sample = self.mnist.validation
        elif self.dset == 'test':
            sample = self.mnist.test
        else:
            raise AttributeError("mnist has no dataset {}".format(self.dset))
        return dict(X=sample.images[key], y=sample.labels[key], ID=key)
    
    def get_sample_keys(self):
        """Return a list of keys, where each key identifies a sample"""
        if self.dset == 'train':
            samples = self.mnist.train
        elif self.dset == 'validation':
            samples = self.mnist.validation
        elif self.dset == 'test':
            samples = self.mnist.test
        else:
            raise AttributeError("mnist has no dataset {}".format(self.dset))
        return np.arange(samples.num_examples, dtype=np.int)


class MovingDots(DataReader):
    def __init__(self, random_seed: int = None, n_timesteps=20, n_samples=500, image_resolution=35, dtype=np.float32):
        """Example class containing dataset with video sequences in which two single pixels move on the x dimension
        in opposite directions; target y is the next frame after the whole sequence x;
        Frame resolution is image_resolution x image_resolution; Target is the next frame after the last frame;
        x_shape=(n_samples, n_timesteps, edge_size, edge_size, 1); y_shape=(n_samples, 1, edge_size, edge_size, 1);
        """
        super(MovingDots, self).__init__()
        #
        # Generate Data
        #
        with Timer(name="Generating data"):
            # Two feature dimensions for the 2 moving directions, which will be merged into 1 dimension
            x = np.zeros((n_samples, n_timesteps + 1, image_resolution, image_resolution, 2), dtype=np.bool)
            
            #
            # Set an initial position for the two moving pixels
            #
            if random_seed is None:
                random_seed = np.random.randint(0, np.iinfo().max)
            rnd_gen = np.random.RandomState(random_seed)
            positions_pot = np.array(range(0, image_resolution), dtype=np.int)
            x_positions = rnd_gen.choice(positions_pot, size=(n_samples, 2), replace=True)
            y_positions = rnd_gen.choice(positions_pot, size=(n_samples, 2), replace=True)
            
            #
            # Tile positions for all frames and move the x_positions
            #
            for s in range(n_samples):
                x_pos = np.arange(n_timesteps + 1, dtype=np.int) * 2 + x_positions[s, 0]
                x_pos %= image_resolution
                y_pos = y_positions[s, 0]
                x[s, np.arange(n_timesteps + 1), x_pos, y_pos, 0] = 1
                
                x_pos[:] = np.arange(n_timesteps + 1, dtype=np.int) * 2 + x_positions[s, 1]
                x_pos = x_pos[::-1]
                x_pos %= image_resolution
                y_pos = y_positions[s, 1]
                x[s, np.arange(n_timesteps + 1), x_pos, y_pos, 1] = 1
            
            #
            # Join the two feature dimension into one
            #
            x = np.array(np.logical_or(x[:, :, :, :, 0], x[:, :, :, :, 1]), dtype=dtype)
            x = x[:, :, :, :, None]  # add empty feature dimension
            
            #
            # y is the next frame
            #
            y = x[:, -1:, :]
            x = x[:, :-1, :]
        
        self.x = x
        self.y = y
        self.ids = np.arange(n_samples, dtype=np.int)
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps
    
    def read_sample(self, key):
        """Read a single sample associated with 'key' into dictionary;
        Dictionary keys can be associated with a preprocessing pipeline (see example below);
        For preprocessing, images or sequences of images should be numpy arrays of shape [frames, x, y, channels] or
        [x, y, channels] and pixel values should be in range [0, 1];"""
        return dict(X=self.x[key], y=self.y[key], ID=self.ids[key])
    
    def get_sample_keys(self):
        """Return a list of keys, where each key identifies a sample"""
        return self.ids


class ShortLongDataset(DataReader):
    def __init__(self, n_timesteps=1000, n_samples=50000, random_seed=0, dtype=np.float32):
        """Class containing dataset with 2 sequences where information over short and long subsequences need to be
        recognized.
        
        Input sequence design:
        Long subsequences: periodic low-frequency rectangular signal
            sl = periodic signal with value 1 and wavelength wl
        Short subsequences: short random fixed-width rectangular signals
            sh = random signal with value 1 and wavelength wh
        Target sequence design:
        Long subsequences: low-frequency rectangular signal
            pl = integral(sl) / wl/2 per wavelength wl
        Short subsequences: high-frequency rectangular signal
            ph = integral(sh) / wh/2 per wavelength wh
        
        x = sl + sh
        y = 0.75 < (pl + ph) / 2
        
        wl = n_timesteps / 5
        wh = 5
        """
        super(ShortLongDataset, self).__init__()
        fl = 3
        wl = int(n_timesteps / fl)
        wh = int(5)
            
        self.n_samples = n_samples
        self.dtype = dtype
        self.fl = fl
        self.wl = wl
        self.wh = wh
        self.ids = np.arange(n_samples, dtype=np.int)
        self.n_timesteps = n_timesteps
        self.random_seed = random_seed
    
    def read_sample(self, key):
        """Read a single sample associated with 'key' into dictionary;
        Dictionary keys can be associated with a preprocessing pipeline (see example below);
        For preprocessing, images or sequences of images should be numpy arrays of shape [frames, x, y, channels] or
        [x, y, channels] and pixel values should be in range [0, 1];"""
        n_samples = self.n_samples
        dtype = self.dtype
        fl = self.fl
        wl = self.wl
        wh = self.wh
        n_timesteps = self.n_timesteps
        key = self.ids[key]
        random_seed = self.random_seed
        
        rnd_gen = np.random.RandomState(int(key+random_seed))
        
        # Preallocate arrays
        x = np.zeros((n_timesteps, 2), dtype=dtype)
        y = np.zeros((n_timesteps, 1), dtype=dtype)
        
        #
        # Create signal sl
        #
        # Longer signal input
        wave = np.array(np.sin(np.linspace(0, (np.pi - 1e-6) * 2, num=wl)) > 0, dtype=dtype)
        wave_bc = np.broadcast_to(wave, (fl, wl))
        x[:np.prod(wave_bc.shape), 0] = wave_bc.flatten()
        
        # Pick random offsets from starting position of sequence for long signal
        rnd_offset = rnd_gen.choice(np.arange(0, wl), size=1, replace=True)
        x[:, 0] = np.roll(x[:, 0], shift=rnd_offset, axis=0)
        
        # Longer signal target
        true_wave = np.array(wave, dtype=bool)
        wave[true_wave] = np.cumsum(wave[true_wave])
        wave /= np.max(wave)
        wave_cs = np.broadcast_to(wave, (fl, int(n_timesteps / fl)))
        y[:np.prod(wave_bc.shape), 0] = wave_cs.flatten()
        y[:, 0] = np.roll(y[:, 0], shift=rnd_offset, axis=0)

        #
        # Create signal sh
        #
        pd_pot = np.array(range(10, int(n_timesteps / 10)))
        pd_pick = 0
        while True:
            pd_pick += rnd_gen.choice(pd_pot, size=1, replace=True)[0]
            if pd_pick+wh >= n_timesteps:
                break
            x[pd_pick:pd_pick+wh, 1] += 1
            y[pd_pick+wh-1, 0] += 1
        
        y[:] = (y > 1.5)
        
        return dict(X=x, y=y, ID=key)
    
    def get_sample_keys(self):
        """Return a list of keys, where each key identifies a sample"""
        return self.ids
    
    def get_num_classes(self):
        return 1
