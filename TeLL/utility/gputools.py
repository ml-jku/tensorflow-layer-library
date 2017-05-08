# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
from tensorflow.python.client import device_lib
import itertools as it


class ResourceManager(object):
    def __init__(self):
        self.devices = device_lib.list_local_devices()
        self.cpus = [x.name for x in self.devices if x.device_type == 'CPU']
        self.gpus = [x.name for x in self.devices if x.device_type == 'GPU']
        self.iterate_cpus = it.cycle(self.cpus)
        self.iterate_gpus = it.cycle(self.gpus)
    
    def next_device(self):
        """
        Returns the id of the next available device. If GPUs are present will
        cycle through GPUs, otherwise returns CPU ids.

        :return: next available device id
        """
        try:
            return self.get_next_gpu()
        except StopIteration:
            return self.get_next_cpu()
    
    def get_available_cpus(self):
        """
        :return: indices of available CPUs
        """
        return self.cpus
    
    def get_available_gpus(self):
        """
        :return: indices of available GPUs
        """
        return self.gpus
    
    def get_next_gpu(self):
        """
        :return: next available gpu id
        """
        return self.iterate_gpus.__next__()
    
    def get_next_cpu(self):
        """
        :return: next available cpu id
        """
        return self.iterate_cpus.__next__()
