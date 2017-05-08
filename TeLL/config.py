# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Default configuration settings

"""
import os
import json
import importlib
from TeLL.utility.misc import get_rec_attr, extract_named_args, try_to_number, parse_args


class Config(object):
    def __init__(self, filename: str = None):
        """Create config object from json file.
        
        filename : optional;
            If passed read config from specified file, otherwise parse command line for config parameter and optionally
            override arguments.
        """
        if filename is None:
            args, override_args = parse_args()
            config_file = args.config
        else:
            args = None
            override_args = None
            config_file = filename
        
        # Read config and override with args if passed
        if os.path.exists(config_file):
            with open(config_file) as f:
                self.initialize_from_json(json.loads(f.read()).items())
            # set restore path if passed
            if args is not None and args.restore is not None:
                self.restore = args.restore
            # override if necessary
            if override_args is not None:
                self.override_from_commandline(override_args)
        else:
            raise Exception("Configuration file does not exist!")
    
    def override(self, name, value):
        if value is not None:
            setattr(self, name, value)
            print("CONFIG: {}={}".format(name, getattr(self, name)))
    
    def get_value(self, name, default=None):
        return getattr(self, name, default)
    
    def import_architecture(self):
        if hasattr(self, "architecture"):
            architecture = importlib.import_module(self.architecture.split('.', maxsplit=1)[0])
            return get_rec_attr(architecture, self.architecture.split('.', maxsplit=1)[-1])
        else:
            return None
    
    def initialize_from_json(self, nv_pairs=None):
        if nv_pairs:
            for i, (name, value) in enumerate(nv_pairs):
                self.override(name, value)
    
    def override_from_commandline(self, override_args=None):
        if override_args is not None:
            override = extract_named_args(override_args)
            for k, v in override.items():
                name = k[2:]
                value = v if v.startswith('"') or v.startswith("'") else try_to_number(v)
                self.override(name, value)
    
    #
    # Default settings
    #
    specs = 'default'
    
    # logging and plotting
    plot_at = 100  # plot at each xth weight update
    score_at = 100  # calculate score on validation set at each xth weight update
    
    # GPU and CPU usage
    cuda_gpu = "0"
    inter_op_parallelism_threads = 1
    intra_op_parallelism_threads = 1
    log_device_placement = False
    
    # Default paths
    restore = None
    working_dir = "working_dir"
