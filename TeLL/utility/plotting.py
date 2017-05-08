# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Example functions for plotting of results - use on own risk

"""

import os
import numpy as np
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

from TeLL.utility.timer import Timer
from TeLL.utility.misc import make_sure_path_exists
from TeLL.utility.plotting_daemons import start_plotting_daemon, stop_plotting_daemon


def save_subplots(kwargs):
    """Plotting list of images as subplots
    
    Parameters
    -------
    kwargs : dict
        images : list of lists of numpy arrays or list of numpy arrays or numpy array of shape (x,y,3) or (x,y)
            List of lists of images to plot; Images with shape (x,y,3) or (x,y);
            List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
            number of columns);
        filename : str
            Filename to save subplot as
        title : str
            Main title of plot
        subfigtitles : list of str
            Strings to be used as titles for the subfigures
        colorbar : bool
            True: plot colorbar
        
    Returns
    -------
    Saves figure to filename
    """
    images = kwargs.get('images')
    filename = kwargs.get('filename')
    title = kwargs.get('title', None)
    subfigtitles = kwargs.get('subfigtitles', [])
    subplotranges = kwargs.get('subplotranges', [])
    colorbar = kwargs.get('colorbar', True)
    
    # Check if filepath is valid
    if len(os.path.dirname(filename)):
        make_sure_path_exists(os.path.dirname(filename))
    
    # Check if image is valid, nest list to 2D if necessary
    if not isinstance(images, list):
        images = [images]
    if not isinstance(images[0], list):
        images = [images]
    
    # Create empty subplot
    n_cols = len(images)
    n_rows = max([len(l) for l in images])
    if n_cols <= 1:
        n_cols = 2
    
    if n_rows <= 1:
        n_rows = 2
    
    f, ax = pl.subplots(n_rows, n_cols)
    
    if title is not None:
        f.suptitle(title)
    
    for col in ax:
        for row in col:
            row.get_xaxis().set_visible(False)
            row.get_yaxis().set_visible(False)
    
    for col_i, col in enumerate(images):
        for row_i, row in enumerate(col):
            
            # Check dimensions of image, squeeze if necessary
            if len(row.shape) == 3:
                if row.shape[-1] == 1:
                    # If an image has shape (x, y, 1) flatten it to (x, y) to plot it as grayscale
                    row = row.reshape(row.shape[:-1])
            
            # Plot image at position in subfigure with optional color-range
            try:
                if subplotranges[col_i][row_i]:
                    im = ax[row_i][col_i].imshow(np.array(row, dtype=np.float), interpolation='nearest',
                                                 vmin=subplotranges[col_i][row_i][0], vmax=subplotranges[col_i][row_i][1])
                else:
                    im = ax[row_i][col_i].imshow(np.array(row, dtype=np.float), interpolation='nearest')
            except IndexError:
                im = ax[row_i][col_i].imshow(np.array(row, dtype=np.float), interpolation='nearest')

            # Try to add title
            try:
                ax[row_i][col_i].set_title(subfigtitles[col_i][row_i], fontdict=dict(fontsize=6))
            except IndexError:
                pass
            
            # Add colorbar to subplot
            if colorbar:
                # Create divider for existing axes instance
                divider = make_axes_locatable(ax[row_i][col_i])
                # Append axes to the right of ax, with 20% width of ax
                cax = divider.append_axes("right", size="20%", pad=0.05)
                # Create colorbar in the appended axes
                # Tick locations can be set with the kwarg `ticks`
                # and the format of the ticklabels with kwarg `format`
                cbar = pl.colorbar(im, cax=cax)#, ticks=MultipleLocator(0.2), format="%.2f")
                cbar.ax.tick_params(labelsize=5)
            
            ax[row_i][col_i].get_xaxis().set_visible(False)
            ax[row_i][col_i].get_yaxis().set_visible(False)
            ax[row_i][col_i].set_aspect('auto')
    
    # for row_i in ax:
    #     for col_i in ax:
    #         col_i.get_xaxis().set_visible(False)
    #         col_i.get_xaxis().set_visible(False)
    
    #pl.tight_layout()
    f.savefig(filename)
    pl.close()

def make_movie(kwargs):
    """ vid, filename, title=None, subfigtitles=[], colorbar=True, interval=1, dpi=100"""
    vid = kwargs.get('vid')
    filename = kwargs.get('filename')
    title = kwargs.get('title', None)
    subfigtitles = kwargs.get('subfigtitles', [])
    colorbar = kwargs.get('colorbar', True)
    interval = kwargs.get('interval', 100)
    dpi = kwargs.get('dpi', 100)
    
    
    if len(os.path.dirname(filename)):
        make_sure_path_exists(os.path.dirname(filename))
    
    # create empty subplot
    n_cols = len(vid[0])
    n_rows = max([len(l) for l in vid[0]])
    f, ax = pl.subplots(n_rows, n_cols)
    im = []
    
    # Initialize subplot
    for col_i, col in enumerate(vid[0]):
        for row_i, row in enumerate(col):
            # plot image at position in subfigure
            im.append(ax[row_i][col_i].imshow(np.array(row, dtype=np.float), interpolation='nearest'))
            # add title to subplot, if existent
            try:
                ax[row_i][col_i].set_title(subfigtitles[0][col_i][row_i])
            except IndexError:
                pass
            if colorbar:
                # Create divider for existing axes instance
                divider = make_axes_locatable(ax[row_i][col_i])
                # Append axes to the right of ax, with 20% width of ax
                cax = divider.append_axes("right", size="20%", pad=0.05)
                # Create colorbar in the appended axes
                # Tick locations can be set with the kwarg `ticks`
                # and the format of the ticklabels with kwarg `format`
                pl.colorbar(im[row_i+col_i*len(col)], cax=cax)#, ticks=MultipleLocator(0.2), format="%.2f")
            ax[row_i][col_i].get_xaxis().set_visible(False)
            ax[row_i][col_i].get_yaxis().set_visible(False)
            ax[row_i][col_i].set_aspect('auto')
            
    #pl.tight_layout()
    
    def make_frame(vid):
        for f_i, frame in enumerate(len(vid)):
            images=vid[frame]
            for col_i, col in enumerate(images):
                for row_i, row in enumerate(col):
                    # plot image at position in subfigure
                    im[row_i+col_i*len(col)].set_data(np.array(row, dtype=np.float))
                    # add title to subplot, if existent
                    try:
                        ax[row_i][col_i].set_title(subfigtitles[f_i][col_i][row_i])
                    except IndexError:
                        pass
                    if colorbar:
                        # Create divider for existing axes instance
                        divider = make_axes_locatable(ax[row_i][col_i])
                        # Append axes to the right of ax, with 20% width of ax
                        cax = divider.append_axes("right", size="20%", pad=0.05)
                        # Create colorbar in the appended axes
                        # Tick locations can be set with the kwarg `ticks`
                        # and the format of the ticklabels with kwarg `format`
                        pl.colorbar(im[row_i+col_i*len(col)], cax=cax)#, ticks=MultipleLocator(0.2), format="%.2f")
                    ax[row_i][col_i].get_xaxis().set_visible(False)
                    ax[row_i][col_i].get_yaxis().set_visible(False)
                    
            #pl.tight_layout()
        yield im
    
    
    animated = animation.FuncAnimation(f, make_frame, vid, blit=False, interval=interval, repeat=False)
    animated.save(filename, dpi=dpi)
    pl.close()
    

def save_subplots_tell(images, filename, title=None, colorbar=True):
    """Plotting list of images as subplots

    Parameters
    -------
    kwargs : dict
        images : list of lists of numpy arrays of shape (x,y,3) or (x,y)
            List of lists of images to plot; Images with shape (x,y,3) or (x,y);
            List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
            number of columns);
        filename : str
            Filename to save subplot as
        title : str
            Main title of plot
        subfigtitles : list of str
            Strings to be used as titles for the subfigures
        colorbar : bool
            True: plot colorbar

    Returns
    -------
    Saves figure to filename
    """
    kwargs = dict(images=images, filename=filename, title=title, colorbar=colorbar)
    save_subplots(kwargs)


class Plotter(object):
    def __init__(self, num_workers: int = 0, plot_function=None, plot_kwargs=None, session=None, verbose: bool = False):
        """Base class for plotting
        
        This class manages sets up a set of background workers (if num_workers > 0) at initialization. The function
        to be used to create and save the plots is defined with Plotter.set_plotting_function().
        Plotter.close() handles the correct termination of the background workers.
        
        Parameters
        -------
        num_workers : int
            Number of background workers to spawn (0 corresponds to no background workers being used); Has to be >= 0;
        plot_function : function
            Function used for creating/saving the plots. Should take kwargs arguments. Can be specified later via
            Plotter.set_plotting_function().
        plot_kwargs : dict or None
            This dictionary consists of {key: value} to be passed to the plotting function. If value has a .eval()
            function, it is treated as tf.Tensor and evaluated before passing it on to the plotting function.
            New kwargs can be added later via Plotter.update_plot_kwargs().
            Kwargs can be reset via Plotter.reset_plot_kwargs().
        verbose : bool
            Verbose behaviour
        """
        self.verbose = verbose
        
        #
        # Handle background workers
        #
        self.plotting_queue = None
        self.plotting_proc = None
        if num_workers > 0:
            self.log("Spawning plotting background workers...")
            (self.plotting_queue, self.plotting_proc) = start_plotting_daemon(multicore=num_workers)
        
        #
        # Initialize plotting function
        #
        self.plotting_function = None
        if plot_function is not None:
            self.set_plotting_function(plot_function)
        
        #
        # Initialize kwargs for plotting
        #
        self.plot_kwargs = OrderedDict()
        if plot_kwargs is not None:
            self.plot_kwargs = plot_kwargs
        
        # TODO
        self.session = session
        self.tensor_vals = []
    
    def get_tensor_kwargs(self):
        """Return ordered dictionary {key: value} containing only the tensorflow tensors in kwargs"""
        tensor_kwargs = OrderedDict()
        for item in self.plot_kwargs.items():
            if hasattr(item[1], 'eval'):
                # If item is tensor, add it to tensor_kwargs
                tensor_kwargs[item[0]] = item[1]
        return tensor_kwargs
        
    def get_tensors(self):
        """Return ordered dictionary {key: value} containing only the tensorflow tensors in kwargs"""
        return list(self.get_tensor_kwargs().values())
        
    def get_tensor_keys(self):
        """Return ordered dictionary {key: value} containing only the tensorflow tensors in kwargs"""
        return list(self.get_tensor_kwargs().keys())
    
    def raw_plot(self, plot_kwargs=None):
        """Execute the plotting function defined via Plotter.set_plotting_function() (in background workers if
        existent).
        """
        if plot_kwargs is None:
            plot_kwargs = self.plot_kwargs
        if self.plotting_queue is not None:
            self.plotting_queue.put([save_subplots, [plot_kwargs]])
        elif self.plotting_function is not None:
            self.plotting_function(plot_kwargs)
        else:
            raise AttributeError("Plotting function undefined. Please use Plotter.set_plotting_function() before "
                                 "calling Plotter.plot().")
    
    def plot(self, session=None, evaluate_tensors=True):
        """Execute the plotting function defined via Plotter.set_plotting_function() (in background workers if
        existent). Tensors will be evaluated before the plotting function is called.
        
        Parameters
        --------
        session : tensorflow session or None
            Tensorflow session to use for evaluation of tensors; If None the session defined in __init__() or
            set_session() will be used;
        evaluate_tensors : bool
            If True, the tensors will be evaluated; If False, the previously determined results will be used for the
            tensors (see access_tensor_values()):
        """
        evaluated_params = self.plot_kwargs.copy()
        
        tensor_dict_items = list(self.get_tensor_kwargs().items())
        tensor_keys = [i[0] for i in tensor_dict_items]
        tensor_vals = [i[1] for i in tensor_dict_items]
        
        # Evaluate tensors in plot_kwargs if necessary
        if len(self.get_tensor_kwargs()):
            if evaluate_tensors:
                if session is None:
                    session = self.session
                
                tensor_vals = session.run(tensor_vals)
                self.tensor_vals = tensor_vals
            else:
                tensor_vals = self.tensor_vals

        tensor_evals = OrderedDict(zip(tensor_keys, tensor_vals))
        evaluated_params.update(tensor_evals)
        
        self.raw_plot(evaluated_params)
    
    def set_plotting_function(self, plotting_function):
        """Set the plotting function to use"""
        self.plotting_function = plotting_function
        
    def set_plot_kwargs(self, plot_kwargs):
        """Set keyword arguments to pass to the plotting function in form ofn a dictionary
        
        Parameters
        -------
        plot_kwargs : dict
            Set keyword arguments to be passed to the plotting function defined via set_plotting_function().
        """
        self.plot_kwargs = plot_kwargs
    
    def set_session(self, session):
        """Set the tensorflow session to use"""
        self.session = session
    
    def set_tensor_values(self, tensor_vals):
        """Can be used to pre-evaluate tensors before calling plot()
        
        Example
        ------
        >>> plotter = Plotter(...)
        >>> ..., plotting_tensors = session.run(..., *plotter.get_tensors())
        >>> plotting_tensors = plotter.set_tensor_values(plotting_tensors)
        >>> plotter.plot(evaluate_tensors=False)
        """
        self.tensor_vals = tensor_vals
        
    def update_plot_kwargs(self, plot_kwargs):
        """Add keyword arguments to the plotting function
        
        Add new key-value pairs in form of a dictionary to pass to the plotting function defined via
        set_plotting_function().
        If value has a .eval() function, it is treated as tf.Tensor and evaluated before passing it on to the plotting
        function.
        This will overwrite existing kwargs which share the same keys.
        
        Parameters
        -------
        plot_kwargs : dict
            Add keyword arguments to be passed to the plotting function defined via Plotter.set_plotting_function().
        """
        self.plot_kwargs.update(plot_kwargs)
    
    def log(self, message, end="\n"):
        if self.verbose:
            print(message, end=end)
    
    def close(self):
        """Terminate background workers, if existent"""
        if self.plotting_queue is not None:
            stop_plotting_daemon(self.plotting_queue, self.plotting_proc)