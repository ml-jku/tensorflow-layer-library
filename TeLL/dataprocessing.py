# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Template and parent classes for creating reader/loader classes for datasets

"""

from typing import Union

import numpy as np
from PIL import Image


# from skimage import color


def get_input(incoming):
    """Get input from DataProcessing class or list of sample keys

    Check if input is available via get_processing_list() function or convert list of sample keys

    Returns
    -------
    """
    datareader_ret = None
    
    if isinstance(incoming, DataProcessing):
        # Get processing list from other DataProcessing instance
        datareader_ret = incoming.datareader
        incoming = incoming.get_processing_list()
    else:
        # Get list of sample keys from DataReader instance or assume incoming is a list of keys
        if hasattr(incoming, 'get_sample_keys'):
            datareader_ret = incoming
            incoming = incoming.get_sample_keys()
        
        # Create a processing list with a dummy-function as processing
        incoming = [[i, lambda inp, **kwargs: inp] for i in incoming]
    
    return incoming, datareader_ret


def add_luminance(image):
    """Calculate luminance and add it as channel to the input image"""
    image = np.asarray(image, dtype=np.float32)
    image = np.concatenate((image, (0.2126 * image[:, :, 0]
                                    + 0.7152 * image[:, :, 1]
                                    + 0.0722 * image[:, :, 2])[:, :, None]), axis=2)
    return image


def gaussian_blur(input, filter_size, filter_sampling_range=3.5, strides=[1, 1, 1, 1], padding='SAME'):
    """
    Blur input with a 2D Gaussian filter of size filter_size x filter_size. The filter's values are 
    sampled from an evenly spaced grid on the 2D standard normal distribution in the range 
    [-filter_sampling_range, filter_sampling_range] in both dimensions. 

    :param input: A rank-4 tensor with shape=(samples, x, y, n_channels). The same Gaussian filter 
        will be applied to all n_channels feature maps of input. 
    :param filter_size: The size of one edge of the square-shaped Gaussian filter. 
    :param filter_sampling_range: The range in which to sample from the standard normal distribution in 
        both dimensions, i.e. a sampling range of 1 corresponds to sampling in a square grid that bounds 
        the standard deviation circle.
    :param strides: Param strides as passed to tf.nn.depthwise_conv2d.
    :param padding: Param padding as passed to tf.nn.depthwise_conv2d.
    :return: The result of the Gaussian blur as a rank-4 tensor with the same shape as input.
    """
    
    # import tensorflow locally
    tf = __import__('tensorflow')
    
    # make 2D distribution
    mu = np.repeat(np.float32(0.), 2)
    sig = np.repeat(np.float32(1.), 2)
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu, sig)
    
    # sample from distribution on a grid
    sampling_range = tf.cast(filter_sampling_range, tf.float32)
    x_1D = tf.linspace(-sampling_range, sampling_range, filter_size)
    x = tf.stack(tf.meshgrid(x_1D, x_1D), 2)
    kern = dist.pdf(x)
    kern /= tf.reduce_sum(kern)
    kern = tf.reshape(kern, kern.shape.as_list() + [1, 1])
    kern = tf.tile(kern, [1, 1, input.shape.as_list()[-1], 1])
    
    return tf.nn.depthwise_conv2d(input, kern, strides, padding)


def stretch_values(inp, low, high):
    """Stetch values in an array to be within [low, high]"""
    inp = np.asarray(inp, dtype=np.float32)
    # Stretch pixel values from 0-1 (and make sure division is not by 0)
    inp -= np.min(inp)
    inp /= (np.max(inp) or 1)
    if (high - low) != 1.:
        inp *= (high - low)
    if low != 0.:
        inp += low
    return inp


def zoom_into_image(image: Image, zoom_factor: float, left_lower_corner: tuple = (0, 0),
                    resample: int = Image.NEAREST):
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


def gaussian_blur(input, filter_size, filter_sampling_range=3.5, strides=(1, 1, 1, 1), padding='SAME'):
    """
    Blur input with a Gaussian filter of the given size, which is sampled from the 2D standard
    normal distribution in the range [-filter_sampling_range, filter_sampling_range] in both
    dimensions.

    :param input: A rank-4 tensor with shape=(samples, x, y, n_channels) to which to apply the filter.
    :param filter_size: The size of one edge of the quadratic kernel.
    :param filter_sampling_range: The range in which to sample from the standard normal distribution in
        both dimensions, i.e. a sampling range of 1 corresponds to sampling in a square that bounds the
        standard deviation circle.
    :param strides: Param strides as passed to tf.nn.depthwise_conv2d.
    :param padding: Param padding as passed to tf.nn.depthwise_conv2d.
    :return: The result of the Gaussian blur as a rank-4 tensor with the same shape as input.
    """
    
    import tensorflow as tf
    
    # make 2D distribution
    mu = np.repeat(np.float32(0.), 2)
    sig = np.repeat(np.float32(1.), 2)
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu, sig)
    
    # sample from distribution on a grid
    sampling_range = tf.cast(filter_sampling_range, tf.float32)
    x_1D = tf.linspace(-sampling_range, sampling_range, filter_size)
    x = tf.stack(tf.meshgrid(x_1D, x_1D), 2)
    kern = dist.pdf(x)
    kern /= tf.reduce_sum(kern)
    kern = tf.reshape(kern, kern.shape.as_list() + [1, 1])
    kern = tf.tile(kern, [1, 1, input.shape.as_list()[-1], 1])
    
    return tf.nn.depthwise_conv2d(input, kern, strides, padding)


def init_processing_layer(incoming, apply_to, duplicate, seed, name):
    """Set required attributes of processing-layer (call this in each __init__())"""
    # Get incoming DataProcessing instance or DataReader
    incoming, datareader = get_input(incoming)
    
    if not isinstance(apply_to, list):
        apply_to = [apply_to]
    
    if seed is None:
        personal_seed = np.random.randint(0, np.iinfo(np.int32).max)
    else:
        personal_seed = np.random.RandomState(seed).randint(0, np.iinfo(np.int32).max)
    
    return incoming, datareader, apply_to, duplicate, name, personal_seed


def cast_with_round(inp, dtype):
    """Cast inp to datatype dtype and round if dtype is integer"""
    if inp.dtype != dtype:
        if str(dtype).find('int') != -1:
            inp = np.round(inp)
        inp = np.array(inp, dtype=dtype)
    return inp


# ------------------------------------------------------------------------------------------------------------------
#  Base Class
# ------------------------------------------------------------------------------------------------------------------
class DataProcessing(object):
    def __init__(self, incoming, apply_to=None, duplicate=False, seed: int = None, name="DataProcessing"):
        """Template class for all data processing classes
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        seed : int
            Seed for internal random generator
        name : str
            Name of DataProcessing instance
        
        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"
        
        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        # Set required attributes of processing-layer (call this in each __init__()!)
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
    
    def _apply_selectively_(self, fct):
        def apply_depending_on_key(inp, apply_key=None, **kwargs):
            inp = fct(inp, apply_key=apply_key, **kwargs)
            if apply_key is None or apply_key in self.apply_to:
                inp = self.process(inp, particle=apply_key, **kwargs)
            return inp
        
        return apply_depending_on_key
    
    def get_input_size(self, inp):
        # determine image size
        if len(inp.shape) in (2, 3):
            size = (inp.shape[1], inp.shape[0])
        elif len(inp.shape) == 4:
            size = (inp.shape[2], inp.shape[1])
        else:
            raise Exception("Invalid input shape {}".format(inp.shape))
        return size
    
    def get_random(self, kwargs):
        if "rnd_key" in kwargs and kwargs['rnd_key'] is not None:
            rnd_gen = np.random.RandomState(np.mod(self.seed + kwargs['rnd_key'], np.iinfo(np.int32).max))
        else:
            rnd_gen = np.random.RandomState(self.seed)
        return rnd_gen
    
    def get_processing_list(self):
        """Prepare and return list with columns [sample keys, processing]"""
        processing_list = [[i[0], self._apply_selectively_(i[1])] for i in self.incoming]
        
        if self.duplicate:
            processing_list += self.incoming
        
        return processing_list
    
    def process(self, inp, **kwargs):
        """Processes and returns a numpy array "inp"
        
        Parameters
        -------
        inp : numpy array
            Input array of shape [frames, x, y, features] or [x, y, features]
        kwargs : keyword arguments
            E.g. pseudo random integer
        
        Returns
        -------
         : numpy array
            Processed version of input array
         : keyword arguments
            E.g. pseudo random integer
        """
        return inp


# ------------------------------------------------------------------------------------------------------------------
#  DataProcessing Implementations
# ------------------------------------------------------------------------------------------------------------------

class Add(DataProcessing):
    def __init__(self, incoming, operand, apply_to=None, duplicate=False, seed: int = None, name="Add"):
        """Add operand to data

        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        operand : The operand to add to data; A numpy array that has a shape which can be broadcasted on data
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        seed : int
            Seed for internal random generator
        name : str
            Name of DataProcessing instance

        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(Add, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.operand = operand
    
    def process(self, inp, **kwargs):
        return inp + self.operand


class Multiply(DataProcessing):
    def __init__(self, incoming, operand, apply_to=None, duplicate=False, seed: int = None, name="Multiply"):
        """Multiply operand with data

        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        operand : The operand to multiply with data; A numpy array that has a shape which can be broadcasted on data
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        seed : int
            Seed for internal random generator
        name : str
            Name of DataProcessing instance

        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(Multiply, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.operand = operand
    
    def process(self, inp, **kwargs):
        return inp * self.operand


class Normalize(DataProcessing):
    def __init__(self, incoming, mean=None, stddev=None, apply_to=None, duplicate=False, seed: int = None,
                 keep_dtype=False,
                 name="Normalize"):
        """Normalize values by mean and standard variance
        
        Normalize data by global or sample mean and standard deviation (substract mean, divide by standard deviation)
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        mean : float or None
            None: Subtract mean of current array; Float: Subtract fixed specified mean;
        stddev : float or None
            None: Divide by standard deviation of current array; Float: Divide by fixed specified standard deviation;
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        name : str
            Name of DataProcessing instance
        
        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"
        
        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(Normalize, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.mean = mean
        self.stddev = stddev
        self.keep_dtype = keep_dtype
    
    def process(self, inp, **kwargs):
        if self.keep_dtype:
            dtype = inp.dtype
        if self.mean is None:
            mean = np.mean(inp)
        else:
            mean = self.mean
        
        if self.stddev is None:
            stddev = np.std(inp)
        else:
            stddev = self.stddev
        
        outp = (inp - mean) / stddev
        
        if self.keep_dtype:
            outp = cast_with_round(outp, dtype)
        else:
            outp = np.array(outp, dtype=np.float32)
        return outp


class StretchValues(DataProcessing):
    def __init__(self, incoming, low=0., high=1., apply_to=None, duplicate=False, seed: int = None, keep_dtype=False,
                 name="StretchValues"):
        """Stretch values to given limits
        
        Stretch values in every sample, so that the lowest value is equal to the lower limit 'low' and  the highest
        value is equal to the upper limit 'high'
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        low : float
            Lower limit for values
        high : float
            Upper limit for values
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        keep_dtype : bool
            Preserve original datatype or cast to float32?
        name : str
            Name of DataProcessing instance
        
        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"
        
        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(StretchValues, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.low = low
        self.high = high
        self.keep_dtype = keep_dtype
    
    def process(self, inp, **kwargs):
        if self.keep_dtype:
            dtype = inp.dtype
        outp = stretch_values(inp, self.low, self.high)
        if self.keep_dtype:
            outp = cast_with_round(outp, dtype)
        return outp


class Flip(DataProcessing):
    def __init__(self, incoming, axis=0, apply_to=None, duplicate=False, seed: int = None, name="Flip"):
        """Flip array at specified axis
        
        Flip axis 'axis' of an array, i.e. reverse entries along this axis via np.flip();
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        axis : int
            Axis to flip; (If an array has shape [n_frames, width, height], a horizontal flip will be on axis 1)
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        name : str
            Name of DataProcessing instance
        
        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"
        
        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(Flip, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.axis = axis
    
    def process(self, inp, **kwargs):
        return np.flip(inp, axis=self.axis)


class Zoom(DataProcessing):
    # TODO: test this
    def __init__(self, incoming, center: Union[str, list] = 'random', factor=2., border=(0, 0),
                 resample: Union[str, dict] = "nearest", name="Zoom", duplicate=False, seed: int = None, apply_to=None):
        """Zoom into images
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        center : Either a list of two real numbers from the interval [0, 1] defining the
            x and y coordinates of the zoom center relative to the possible ranges, or
            'random' for uniformly sampling the center coordinates per image.
        factor : Either a float defining a fixed zoom factor, or a list of two floats
            bounding the range in which the zoom factor should be sampled uniformly per image.
        border : float or iterable
            Percentage of the image on sides which will not be included when random zoom center is generated. If float,
            border will be the same for all sides, otherwise it will be interpreted as borders for [x, y]
        resample : str or dict
            Str: global resampling method;
            Dict: dictionary mapping particles of sample (e.g. X, y) to resampling methods;
            Both: resampling method can be one of "nearest", "bicubic", "bilinear", "lanczos"
        apply_to : list or object
           Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
           processing to all elements;
        duplicate : bool
           If False, all samples are processed; If True, processed and original samples are stored;
        seed : int
           Seed for internal random generator
        name : str
           Name of DataProcessing instance
        
        Methods
        -------
        get_processing_list() : function
           Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
           Processes and returns a numpy array "inp"
        
        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
        >>> # Define your DataReader class or use an existing one
        >>> reader = CityScapesReader(...)
        >>> # NOTE: CityScapesReader reads samples in a dictionary {'X': input image, 'y': label image, ...}
        >>>
        >>> # Stack some preprocessings for input images only
        >>> normalized = Normalize(reader, apply_to=['X'])
        >>>
        >>> # Stack some preprocessings for input and label images
        >>> zoomed = Zoom(normalized, center='random', resample={'X': 'bicubic', 'y': 'nearest'}, apply_to=['X', 'y'])
        >>>
        >>> # Create a DataLoader instance
        >>> trainingset = DataLoader(data=zoomed, batchsize=5)
        >>>
        >>> # trainingset.batch_loader() will load your minibatches in background workers and yield them
        >>> for mb_i, mb in enumerate(trainingset.batch_loader(num_cached=5, num_threads=3)):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        super(Zoom, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.resample_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS}
        
        self.center = center
        self.factor = factor
        self.border = border
        self.resample = resample
    
    def process(self, inp, **kwargs):
        # get random generator
        rnd_gen = self.get_random(kwargs)
        
        # determine image size
        size = self.get_input_size(inp)
        
        # determine zoom factor
        if (isinstance(self.factor, list) or isinstance(self.factor, tuple)) and len(self.factor) == 2:
            factor = rnd_gen.uniform(self.factor[0], self.factor[1])
        else:
            factor = self.factor
        
        # determine center
        if self.center == "random":
            try:
                center_constraint_x = max(1. / (2 * factor), self.border[0])
                center_constraint_y = max(1. / (2 * factor), self.border[1])
                center = (rnd_gen.uniform(center_constraint_x, 1 - center_constraint_x),
                          rnd_gen.uniform(center_constraint_y, 1 - center_constraint_y))
            except (IndexError, TypeError, AttributeError):
                center_constraint = max(1. / (2 * factor), self.border)
                center = (rnd_gen.uniform(center_constraint, 1 - center_constraint),
                          rnd_gen.uniform(center_constraint, 1 - center_constraint))
        else:
            center = self.center
        
        # determine lower left corner of zoom area
        lower_left = [int((p - 1. / (2 * factor)) * (s / factor)) for p, s in zip(center, size)]
        
        # determine resampling method
        if isinstance(self.resample, dict):
            resample = self.resample_methods[self.resample[kwargs["particle"]]]
        else:
            resample = self.resample_methods[self.resample]
        
        # apply zoom
        if len(inp.shape) in (2, 3):
            zoomed = self.__zoom(inp, factor, lower_left, resample)
        elif len(inp.shape) == 4:
            zoomed = np.empty(inp.shape, dtype=inp.dtype)
            for i, frame in enumerate(inp):
                zoomed[i, :, :, :] = self.__zoom(frame, factor, lower_left, resample)
        return zoomed
    
    def __zoom(self, frame, factor, lower_left, resample):
        if frame.shape[-1] is 1:
            frame = np.squeeze(frame)
            image = Image.fromarray(frame)
            image = zoom_into_image(image, factor, tuple(lower_left), resample=resample)
            image = np.asarray(image, dtype=frame.dtype)
            image = image.reshape(image.shape + (1,))
        else:
            image = Image.fromarray(frame)
            image = zoom_into_image(image, factor, tuple(lower_left), resample=resample)
            image = np.asarray(image, dtype=frame.dtype)
        
        return image


class Resize(DataProcessing):
    def __init__(self, incoming, scaling_factor: int = None, size: tuple = None, resample: Union[str, dict] = "nearest",
                 apply_to=None, duplicate=False, seed: int = None, name="Resize"):
        """Resize layer

        Resize input by a scaling factor (divide dimensions by given factor) or to specific size.

        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        scaling_factor : int or None
            Int: factor by which width and height are divided to arrive at new size (ignored if size is specified);
            None: if none size has to be specified
        size : tuple or None
            None: scaling_factor has to be specified;
            Tuple: new width and height in the format (width, height)
        resample : str or dict
            Str: global resampling method;
            Dict: dictionary mapping particles of sample (e.g. X, y) to resampling methods;
            Both: resampling method can be one of "nearest", "bicubic", "bilinear", "lanczos"
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        name : str
            Name of DataProcessing instance

        Methods
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, Zoom
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
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
        super(Resize, self).__init__(incoming=incoming, name=name)
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
        
        self.resample_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS}
        
        self.scaling_factor = scaling_factor
        self.size = size
        self.resample = resample
    
    def process(self, inp, **kwargs):
        # determine input size
        size_old = self.get_input_size(inp)
        
        # calculate target size
        if self.scaling_factor is not None and self.size is None:
            size = (int(size_old[0] / self.scaling_factor), int(size_old[1] / self.scaling_factor))
        elif self.size is not None:
            size = self.size
        else:
            raise Exception("Require at least some hints...")
        
        # determine resampling method
        if isinstance(self.resample, dict):
            resample = self.resample_methods[self.resample[kwargs["particle"]]]
        else:
            resample = self.resample_methods[self.resample]
        
        if len(inp.shape) in (2, 3):
            scaled = self.__scale(inp, size, resample)
        elif len(inp.shape) == 4:
            scaled = np.empty((inp.shape[0], size[1], size[0], inp.shape[3]), dtype=inp.dtype)
            for i, frame in enumerate(inp):
                scaled[i, :, :, :] = self.__scale(frame, size, resample)
        return scaled
    
    def __scale(self, frame, size, resample):
        if frame.shape[-1] is 1:
            frame = np.squeeze(frame)
            image = Image.fromarray(frame)
            image = image.resize(size, resample=resample)
            image = np.asarray(image, dtype=frame.dtype)
            image = image.reshape(image.shape + (1,))
        else:
            image = Image.fromarray(frame)
            image = image.resize(size, resample=resample)
            image = np.asarray(image, dtype=frame.dtype)
        return image


class ColorJittering(DataProcessing):
    def __init__(self, incoming, apply_to=None, duplicate=False, seed: int = None, name="ColorJittering"):
        """ColorJittering layer

        This layer takes an input image and add jitters it's color channels.
        Therefore, the RGB input image gets converted to a HSV image.
        After the augmentation the image is converted back to an RGB image.
        
        The jittering is implemented as defined in arxiv:1406.6909.
        
        Definition:
        v(i,j) = h(i,j)**(2**{-2,2}) * 2**{-0.5,0.5}) + {-0.1,0.1}
        s(i,j) = s(i,j)**(2**{-2,2}) * 2**{-0.5,0.5}) + {-0.1,0.1}
        h(i,j) = v(i,j) + {-0.1,0.1}
        
        Where
        h(i,j) is the Hue value of the pixel at position (i,j)
        and
        {x,y} is a uniform random number in range from x to y.
        All random numbers are constant over the whole image.

        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        apply_to : list or object
            Optional: Key(s) of elements in sample dictionary to which processing shall be applied; Defaults to applying
            processing to all elements;
        duplicate : bool
            If False, all samples are processed; If True, processed and original samples are stored;
        seed : int
            Seed for internal random generator
        name : str
            Name of DataProcessing instance

        Methods
        -------
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"

        Example
        ------
        >>> from TeLL.dataprocessing import Normalize, ColorJittering
        >>> from TeLL.datareaders import CityScapesReader, DataLoader
        >>> # Define your DataReader class or use an existing one
        >>> reader = CityScapesReader(...)
        >>> # NOTE: CityScapesReader reads samples in a dictionary {'X': input image, 'y': label image, ...}
        >>>
        >>> # Stack some preprocessings for input images only
        >>> normalized = Normalize(reader, apply_to=['X'])
        >>>
        >>> # Stack some preprocessings for input and label images
        >>> jittered = ColorJittering(normalized, apply_to=['X'])
        >>>
        >>> # Create a DataLoader instance
        >>> trainingset = DataLoader(data=jittered, batchsize=5)
        >>>
        >>> # trainingset.batch_loader() will load your minibatches in background workers and yield them
        >>> for mb_i, mb in enumerate(trainingset.batch_loader(num_cached=5, num_threads=3)):
        >>>     print("Minibatch number {} has the contents {}".format(mb_i, mb))
        """
        super(ColorJittering, self).__init__(incoming=incoming, name=name)
        self.skimage = __import__('skimage')
        
        # Set required attributes of processing-layer
        self.incoming, self.datareader, self.apply_to, self.duplicate, self.name, self.seed = \
            init_processing_layer(incoming, apply_to, duplicate, seed, name)
    
    def process(self, inp, **kwargs):
        # get random generator
        rnd_gen = self.get_random(kwargs)
        
        # convert image into HSV color space
        image = self.skimage.color.rgb2hsv(inp[0])
        
        v_power = 2 ** (np.ones(image.shape[0:2]) * rnd_gen.uniform(-2, 2, 1))
        v_power = v_power[..., np.newaxis]
        v_mul = 2 ** (np.ones(image.shape[0:2]) * rnd_gen.uniform(-0.5, 0.5, 1))
        v_mul = v_mul[..., np.newaxis]
        v_add = np.ones(image.shape[0:2]) * rnd_gen.uniform(-0.1, 0.1, 1)
        v_add = v_add[..., np.newaxis]
        
        s_power = 2 ** (np.ones(image.shape[0:2]) * rnd_gen.uniform(-2, 2, 1))
        s_power = s_power[..., np.newaxis]
        s_mul = 2 ** (np.ones(image.shape[0:2]) * rnd_gen.uniform(-0.5, 0.5, 1))
        s_mul = s_mul[..., np.newaxis]
        s_add = np.ones(image.shape[0:2]) * rnd_gen.uniform(-0.1, 0.1, 1)
        s_add = s_add[..., np.newaxis]
        
        h_add = rnd_gen.uniform(-0.1, 0.1, image.shape[0:2])
        h_add = h_add[..., np.newaxis]
        
        image **= np.concatenate((np.ones(image.shape[0:2])[..., np.newaxis], s_power, v_power), axis=2)
        image *= np.concatenate((np.ones(image.shape[0:2])[..., np.newaxis], s_mul, v_mul), axis=2)
        image += np.concatenate((h_add, s_add, v_add), axis=2)
        
        image = np.clip(image, 0, 1)
        
        return np.asarray([self.skimage.color.hsv2rgb(image) * 255], dtype=inp.dtype)
