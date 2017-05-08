# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import time


class Timer(object):
    def __init__(self, name="", verbose=True, precision='msec'):
        self.verbose = verbose
        self.name = name
        self.precision = precision
        self.start = time.time()
        self.end = self.start
        self.secs = 0
        self.msecs = 0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        if self.verbose:
            self.print()
    
    def print(self):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.precision == 'msec':
            print('Timer ({0}): {1:.2f} ms'.format(self.name, self.msecs))
        else:
            print('Timer ({0}): {1:.3f} s'.format(self.name, self.secs))
