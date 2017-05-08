# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Template and parent classes for creating reader/loader classes for datasets

"""

import tensorflow as tf
import numpy as np
from PIL import Image


def get_input(incoming):
    """Get input from DataProcessing class or list of sample keys

    Check if input is available via get_processing_list() function or convert list of sample keys

    Returns
    -------
    """
    if isinstance(incoming, DataProcessing):
        return incoming.get_processing_list()
    else:
        def return_sample(inp, **kwargs):
            return inp, kwargs
        incoming = [[i, return_sample] for i in incoming]
        return incoming

def apply_to_channels(nparray, function, **kwargs):
    # FIXME: untested
    if len(nparray.shape) == 3:
        d = np.reshape(nparray, [1] + list(nparray.shape))
    elif len(nparray.shape) == 4:
        d = nparray
    else:
        raise ValueError("input must have 3 or 4 axes, %d given" % len(nparray.shape))

    for i in d.shape[0]:
        for j in d.shape[3]:
            function(nparray[i,:,:,j], kwargs)



# ------------------------------------------------------------------------------------------------------------------
#  Classes
# ------------------------------------------------------------------------------------------------------------------
class DataProcessing(object):
    def __init__(self, incoming, duplicate=True, name="DataProcessing"):
        """Template class for all data processing classes
        
        Parameters
        -------
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        name : str
            Name of DataProcessing instance
        
        Attributes
        -------
        get_processing_list() : function
            Returns list with columns [sample keys, processing] with processing being a DataProcessing instance
        process(inp, **kwargs) : function
            Processes and returns a numpy array "inp"
        name : str
            Name of DataProcessing instance
        """
        self.duplicate = duplicate
        self.incoming = get_input(incoming)
        self.name = name
        
    def get_processing_list(self):
        """Prepare and return list with columns [sample keys, processing]"""
        if self.duplicate:
            processing_list = [[i[0], lambda inp, **kwargs: (self.process(i[1](inp, kwargs)), kwargs.copy())]
                               for i in self.incoming]
            processing_list += self.incoming
        else:
            processing_list = [[i[0], lambda inp, **kwargs: (self.process(i[1](inp, kwargs)), kwargs.copy())]
                               for i in self.incoming]
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

class Normalize(DataProcessing):
    # FIXME: untested
    def __init__(self, incoming, mean=0., stddev=1., name="Normalize"):
        super(Normalize, self).__init__(incoming=incoming, name=name)
        self.mean = mean
        self.stddev = stddev

    def process(self, inp, **kwargs):
        return np.divide(np.subtract(inp, self.mean), self.stddev)

class Zoom(DataProcessing):
    # FIXME: untested
    def __init__(self, incoming, position='random', factor=2., resample="nearest", name="Zoom"):
        """Zoom into images
        
        incoming : DataProcessing class instance or list
            Instance of DataProcessing class or list of sample keys
        center : Either a list of two real numbers from the interval [0, 1] defining the  
            x and y coordinates of the zoom center relative to the possible ranges, or 
            'random' for uniformly sampling the center coordinates per image.
        factor : Either a float defining a fixed zoom factor, or a list of two floats 
            bounding the range in which the zoom factor should be sampled uniformly 
            per image. 
        name : str
            Name of DataProcessing instance
        """

        super(Zoom, self).__init__(incoming=incoming, name=name)
        self.position = position
        self.factor = float(factor)
        self.resample = resample

    def process(self, inp, **kwargs):
        apply_to_channels(inp, self.per_image)

    def per_image(self, img):
        resample = {
            'nearest' : Image.NEAREST,
            'bilinear' : Image.BILINEAR,
            'bicubic' : Image.BICUBIC,
            'lanczos' : Image.LANCZOS}[self.resample]

        lower_left = [p * (s / self.factor) for p, s in zip(self.position, img.size)]
        return zoom_into_image(img, self.factor, tuple(lower_left), resample=resample)



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
    """Stetch values in an image/array to be within [0,1]"""
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
