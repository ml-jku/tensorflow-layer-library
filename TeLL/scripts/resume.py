# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Script for resuming from saved checkpoint

"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import sys
import signal
import subprocess
import argparse
import shlex

from TeLL.config import Config
from TeLL.utility.misc import extract_to_tmp, rmdir, extract_named_args, extract_unnamed_args

# ----------------------------------------------------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------------------------------------------------
process_handle = None
working_dir = None
kill_retry_max = 10
kill_retry_count = 0


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def read_config(config: Config, epochs: int = None, gpu: str = None):
    """Get config either from file or use default config"""
    if os.path.isfile(config):
        config = Config.from_file(config)
    else:
        config = Config()
    
    config.override("n_epochs", epochs)
    config.override("cuda_gpu", gpu)
    
    if epochs is not None:
        config.n_epochs = epochs
    
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None,
                        help="Path to previous working directory of run that should be resumed")
    args, unknown = parser.parse_known_args()
    if args.path is None:
        parser.print_help()
        sys.exit(1)
    # --
    print("Resuming {}".format(args.path))
    return args, unknown


def parse_and_merge_args(command, override_args):
    parts = shlex.split(command)
    result = [parts[0]]
    orig_args = extract_named_args(parts[1:])
    override_args = extract_named_args(override_args)
    merged = {**orig_args, **override_args}
    for k, v in merged.items():
        if v is None:
            result.append(k)
        else:
            result.extend([k, v])
    result.extend(extract_unnamed_args(parts[1:]))
    return " ".join(result)


def resume(directory: str, unknown_args: str = None):
    if os.path.isdir(directory):
        results = os.path.join(directory, "results")
        archive = os.path.join(directory, "00-script.zip")
        if os.path.exists(archive) and os.path.exists(results):
            global working_dir
            working_dir = extract_to_tmp(archive)
            # parse used config
            with open(os.path.join(working_dir, "00-INFO")) as f:
                command = f.readline().strip()
                command = parse_and_merge_args(command, unknown_args)
                # start
                cmd_sep = " &&" if sys.platform == "win32" else "; "
                cmd = ["cd \"{}\"{}".format(working_dir, cmd_sep),
                       '"{}"'.format(sys.executable),
                       command,
                       "--restore \"{}\"".format(directory)]
                cmd = " ".join(cmd)
            print("Resuming with command '{}' in directory '{}'".format(cmd, working_dir))
            initial_working_dir = os.getcwd()
            os.chdir(working_dir)
            global process_handle
            process_handle = subprocess.Popen(cmd, cwd=working_dir, shell=True, start_new_session=True)
            process_handle.wait()
            # clean up
            print("Cleaning up temp directory...")
            os.chdir(initial_working_dir)
            rmdir(working_dir)
            print("Done!")
    else:
        print("Can't resume from {}".format(directory))


def sigint_handler(sig, frame):
    print("Killing sub-process...")
    if process_handle is not None:
        global kill_retry_count
        while process_handle.returncode is None and kill_retry_count < kill_retry_max:
            kill_retry_count += 1
            print("Killing sub-process ({})...".format(kill_retry_count))
            try:
                os.killpg(os.getpgid(process_handle.pid), signal.SIGTERM)
                os.waitpid(process_handle.pid, os.WNOHANG)
            except ProcessLookupError:
                break
            
            try:
                process_handle.wait(1)
            except subprocess.TimeoutExpired:
                pass
    
    if working_dir is not None:
        rmdir(working_dir)
    
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    args, unknown_args = parse_args()
    
    # If resume option specified resume from snapshot and exit here
    if args.path is not None:
        resume(args.path, unknown_args)


if __name__ == "__main__":
    main()
