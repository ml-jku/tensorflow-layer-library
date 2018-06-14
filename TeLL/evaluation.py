"""
© Michael Widrich, Markus Hofmarcher, 2017
"""

import sys
import progressbar
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from TeLL.utility.misc import custom_tensorflow_histogram, check_kill_file
from TeLL.utility.timer import Timer
from TeLL.utility.workingdir import Workspace


class Evaluation(object):
    def __init__(self, dataset, session, model, workspace: Workspace, summary_tensor_dict=None, scope=None):
        """Evaluate model on dataset"""
        
        #
        # Create tensors and tf operations
        #
        if summary_tensor_dict is None:
            summary_tensor_dict = {}
        
        summary_tensors = [tens[0] if isinstance(tens, tuple) else tens for tens in summary_tensor_dict.values()]
        summary_tensor_is_op = [True if isinstance(tens, tuple) else False for tens in summary_tensor_dict.values()]
        summary_ops = [tens[1] for tens in summary_tensor_dict.values() if isinstance(tens, tuple)]
        
        self.dataset = dataset
        self.session = session
        self.model = model
        self.workspace = workspace
        self.scope = scope
        
        self.summary_tensor_dict = summary_tensor_dict
        self.summary_tensors = summary_tensors
        self.summary_tensor_is_op = summary_tensor_is_op
        self.summary_ops = summary_ops
        
        resetable_tensors = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self.scope.name)
        self.variables_initializer = tf.variables_initializer(resetable_tensors)
        self.reset_tensors()
    
    def reset_tensors(self):
        if self.scope is not None:
            self.session.run([self.variables_initializer])
    
    def evaluate(self, step: int, summary_writer, prefix='validation ', num_cached=5, num_threads=3, rnd_gen=None,
                 plotter=None, model=None):
        # Reset streaming measures
        self.reset_tensors()
        
        # Get tensors to evaluate for plotting
        if plotter is not None:
            plot_tensors = plotter.get_tensors()
        
        # Set up progress bar
        _pbw = ['Evaluating on {}set:'.format(prefix), progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=_pbw, maxval=self.dataset.n_mbs - 1).start()
        
        #
        # Iterate over dataset minibatches
        #
        mb_validation = self.dataset.batch_loader(num_cached=num_cached, num_threads=num_threads, rnd_gen=rnd_gen)
        with Timer(verbose=True, name="Evaluate on {}set".format(prefix)):
            summary_values_filled = None
            
            for mb_i, mb in enumerate(mb_validation):
                
                # Abort if indicated by file
                check_kill_file(self.workspace)
                
                if mb.get('pixel_weights', None) is None:
                    feed_dict = {self.model.X: mb['X'], self.model.y_: mb['y']}
                else:
                    feed_dict = {self.model.X: mb['X'], self.model.y_: mb['y'],
                                 self.model.pixel_weights: mb['pixel_weights']}
                
                if plotter is not None:
                    evaluated_tensors = self.session.run([*self.summary_ops, *self.summary_tensors, *plot_tensors],
                                                         feed_dict=feed_dict)
                else:
                    evaluated_tensors = self.session.run([*self.summary_ops, *self.summary_tensors],
                                                         feed_dict=feed_dict)
                
                # Discard return values from summary_ops (=update operations)
                evaluated_tensors = evaluated_tensors[len(self.summary_ops):]
                summary_values = evaluated_tensors[:len(self.summary_tensors)]
                
                # Perform plotting
                if plotter is not None:
                    plotter.set_tensor_values(evaluated_tensors[len(self.summary_tensors):len(self.plot_tensors) +
                                                                                          len(plot_tensors)])
                    plotter.plot(evaluate_tensors=False)
                
                # Re-associate returned tensorflow values to keys and incorporate new minibatch values
                if summary_values_filled is None:
                    # Fill summary_values_filled for the first time
                    summary_values_filled = OrderedDict(zip(list(self.summary_tensor_dict.keys()), summary_values))
                    for key_i, key in enumerate(summary_values_filled.keys()):
                        if not self.summary_tensor_is_op[key_i]:
                            if isinstance(summary_values_filled[key], np.ndarray):
                                summary_values_filled[key] = [summary_values_filled[key]]
                            elif np.isfinite(summary_values_filled[key]):
                                summary_values_filled[key] = [summary_values_filled[key]]
                            else:
                                summary_values_filled[key] = []
                else:
                    for key_i, key in enumerate(summary_values_filled.keys()):
                        if not self.summary_tensor_is_op[key_i]:
                            if isinstance(summary_values[key_i], np.ndarray):
                                summary_values_filled[key].append(summary_values[key_i])
                            elif np.isfinite(summary_values[key_i]):
                                summary_values_filled[key].append(summary_values[key_i])
                        else:
                            summary_values_filled[key] = summary_values[key_i]
                
                # Update progress bar and clear minibatch
                progress.update(mb_i)
                mb.clear()
                del mb
        
        progress.finish()
        
        #
        # Divide sums by number of samples for tensors that do not have an update function
        #
        if len(summary_values_filled):
            for key_i, key in enumerate(summary_values_filled.keys()):
                if not self.summary_tensor_is_op[key_i]:
                    if len(summary_values_filled[key]):
                        if not isinstance(summary_values_filled[key][0], np.ndarray):
                            summary_values_filled[key] = np.mean(summary_values_filled[key])
                        else:
                            summary_values_filled[key] = np.concatenate(summary_values_filled[key])
                    else:
                        summary_values_filled[key] = np.nan
        
        #
        # Go through values to use as summaries, create histograms if values are not scalars
        #
        values_to_print = OrderedDict()
        if len(summary_values_filled):
            for key_i, key in enumerate(summary_values_filled.keys()):
                if not isinstance(summary_values_filled[key], np.ndarray):
                    values_to_print.update({key: summary_values_filled[key]})
                    summary = tf.Summary(value=[tf.Summary.Value(tag=prefix + key,
                                                                 simple_value=float(summary_values_filled[key]))])
                else:
                    hist = custom_tensorflow_histogram(summary_values_filled[key], bins=100)
                    summary = tf.Summary(value=[tf.Summary.Value(tag=prefix + key, histo=hist)])
                
                summary_writer.add_summary(summary, step)
        
        print("{}scores:\n\tstep {}, {}".format(prefix, step, values_to_print))
        summary_writer.flush()
        sys.stdout.flush()
