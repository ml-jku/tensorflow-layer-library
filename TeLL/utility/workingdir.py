# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import os
import sys
import datetime as dt
import glob
import tempfile
import TeLL
from natsort import natsorted
from TeLL.utility.misc import make_sure_path_exists, touch, zipdir, chmod, copydir, rmdir
import __main__


class Workspace(object):
    def __init__(self, workspace: str, specs: str, restore: str = None):
        """

        :param workspace: str
            path to general workspace directory
        :param specs: str
            short description of specs for a specific test run;
            will be part of the run specific working directory so don't use spaces or special chars
        """
        self.workspace = os.path.realpath(workspace)
        self.specs = specs
        
        if restore is None:
            self.timestamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.working_dir, self.result_dir, self.tensorboard_dir, self.kill_file, self.plot_file, self.checkpoint = self.__setup_working_dir__()
        else:
            self.working_dir, self.result_dir, self.tensorboard_dir, self.kill_file, self.plot_file, self.checkpoint = self.__resume_from_dir__(
                restore)
    
    def get_result_dir(self):
        return self.result_dir
    
    def get_tensorboard_dir(self):
        return self.tensorboard_dir
    
    def get_kill_file(self):
        return self.kill_file
    
    def get_plot_file(self):
        return self.plot_file
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_checkpoint(self):
        return self.checkpoint
    
    def __setup_working_dir__(self):
        # fix permissions of workspace root
        make_sure_path_exists(self.workspace)
        try:
            chmod(self.workspace, 0o775)
        except PermissionError:
            print("PermissionError when trying to change permissions of workspace to 775")
            
        # setup working directory
        specs_dir = os.path.realpath("{}/{}".format(self.workspace, self.specs))
        working_dir = os.path.realpath("{}/{}".format(specs_dir, self.timestamp))
        # Set up result folder structure
        results_path = "{}/results".format(working_dir, self.timestamp)
        make_sure_path_exists(results_path)
        
        # Set up tensorboard directory
        tensorboard = "{}/tensorboard".format(working_dir, self.timestamp)
        make_sure_path_exists(tensorboard)
        
        # set path to kill file (if this file exists abort run)
        kill_file_name = "ABORT_RUN"
        kill_file = os.path.join(working_dir, kill_file_name)
        
        # create plot file to plot by default
        plot_file_name = "PLOT_ON"
        plot_file = os.path.join(working_dir, plot_file_name)
        touch(plot_file)
        
        # remove kill file before starting the run (should not exist anyway)
        if os.path.isfile(kill_file):
            os.remove(kill_file)
        
        # fix permissions to grant group write access (to allow kill_file creation and plot control)
        try:
            chmod(self.workspace, 0o775, recursive=False)
            chmod(specs_dir, 0o775, recursive=False)
            chmod(working_dir, 0o775, recursive=True)
            chmod(plot_file, 0o664)
        except PermissionError:
            print("PermissionError when trying to change permissions of workspace to 775")

        # compress and copy current script and dependencies to results dir
        command = " ".join(sys.argv)
        # copy current code to temp dir
        script_dir = os.path.dirname(os.path.realpath(__main__.__file__))
        tempdir = tempfile.mkdtemp("tell")
        copydir(script_dir, tempdir,
                exclude=[self.workspace, os.path.join(script_dir, ".git"), os.path.join(script_dir, ".idea"),
                         os.path.join(script_dir, "__pycache__")])
        # also copy currently used TeLL library so it can be used for resuming runs
        copydir(TeLL.__path__[0], os.path.join(tempdir, os.path.basename(TeLL.__path__[0])))
        rmdir(os.path.join(os.path.join(tempdir, os.path.basename(TeLL.__path__[0])), "__pycache__"))
        zipdir(dir=tempdir, zip=os.path.join(working_dir, '00-script.zip'), info=command,
               exclude=[self.workspace, '.git'])
        rmdir(tempdir)
        return [working_dir, results_path, tensorboard, kill_file, plot_file, None]
    
    def __resume_from_dir__(self, dir):
        # setup working directory
        working_dir = os.path.realpath(dir)
        self.timestamp = os.path.basename(working_dir)
        
        # Set up result folder structure
        results_path = "{}/results".format(working_dir, self.timestamp)
        
        # Set up tensorboard directory
        tensorboard = "{}/tensorboard".format(working_dir, self.timestamp)
        
        # set path to kill file (if this file exists abort run)
        kill_file_name = "ABORT_RUN"
        kill_file = os.path.join(working_dir, kill_file_name)
        if os.path.exists(kill_file):
            os.remove(kill_file)
        
        # create plot file to plot by default
        plot_file_name = "PLOT_ON"
        plot_file = os.path.join(working_dir, plot_file_name)
        
        if not (os.path.exists(results_path) or not os.path.exists(tensorboard)):
            raise Exception("can not resume from given directory")
        
        checkpoints = glob.glob1(results_path, "*.ckpt*")
        checkpoints = [checkpoint for checkpoint in checkpoints if
                       not (".meta" in checkpoint or ".index" in checkpoint)]
        checkpoints = natsorted(checkpoints)
        checkpoint = os.path.join(results_path, checkpoints[-1])
        
        if not os.path.exists(checkpoint):
            raise Exception("could not find checkpoint in given directory")
        
        if not checkpoint.endswith("ckpt"):
            checkpoint = checkpoint[:checkpoint.index(".ckpt.") + 5]
        
        return [working_dir, results_path, tensorboard, kill_file, plot_file, checkpoint]
