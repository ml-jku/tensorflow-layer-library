# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Example functions for plotting of results - use on own risk

"""

import os
import sys
import numpy as np
from collections import OrderedDict

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

from TeLL.utility.misc import make_sure_path_exists
from TeLL.utility.plotting_daemons import start_plotting_daemon, stop_plotting_daemon

PLOTTING_QUEUE = None
PLOTTING_PROC = None


def launch_plotting_daemon(num_workers: int = 3):
    """Start backgoundworkers/threads to be used for plotting; Should be called before importing tensorflow due to
    possible memory leaks with matplotlib/GPU;

    Parameters
    ------
    num_workers : int
        The number of backgroundworkers/threads to use for plotting
    """
    global PLOTTING_QUEUE, PLOTTING_PROC
    if num_workers > 0:
        print("Spawning plotting daemon...")
        # Check if tensorflow has already been imported and print warning if so
        if 'tensorflow' in sys.modules:
            print("WARNING: Tensorflow has been imported at start of plotting-subprocess! This can lead to memory "
                  "leaks, please instantiate the TeLL session before importing tensorflow!")
        PLOTTING_QUEUE, PLOTTING_PROC = start_plotting_daemon(multicore=num_workers)
    else:
        PLOTTING_QUEUE, PLOTTING_PROC = (None, None)


def terminate_plotting_daemon():
    """Terminate plotting daemon and close pipes correctly (automatically called by TeLLSession.close())"""
    if PLOTTING_QUEUE is not None or PLOTTING_PROC is not None:
        print("Terminating plotting background workers...")
        stop_plotting_daemon(PLOTTING_QUEUE, PLOTTING_PROC)


########################################################################################################################
# Helper Functions
########################################################################################################################
def start_function(fct, kwargs):
    #
    # Handle background workers
    #
    if PLOTTING_QUEUE is not None and PLOTTING_PROC is not None:
        PLOTTING_QUEUE.put([fct, ((), kwargs)])
    else:
        fct(**kwargs)
    return 0


def quadratic_list_layout(l):
    """Take 1D or 2D list and resort elements into a quadratic 2D layout"""
    if isinstance(l[0], list):
        # Flatten list
        l = [val for sublist in l for val in sublist]
    # Create quatratic 2D layout
    lenght = len(l)
    shape = int(np.ceil(np.sqrt(lenght)))
    l += [None] * (shape ** 2 - len(l))
    square_l = [[l[r * shape + c] for c in range(shape)] for r in range(shape)]
    return square_l


def _make_2d_list(l):
    if not isinstance(l, list):
        l = [l]
    if not isinstance(l[0], list):
        l = [l]
    return l


########################################################################################################################
# Plotting Functions
########################################################################################################################
def save_subplots(images, filename, title=None, subfigtitles=(), subplotranges=(), colorbar=True,
                           automatic_positioning=False, tight_layout=False, interpolation='nearest', aspect='auto',
                           resolution=None, fontdict=None):
    """Plotting list of images as subplots

    Parameters
    -------
    images : list of lists of numpy arrays or list of numpy arrays or numpy array of shape (x,y,3) or (x,y)
        List of lists of images to plot; Images with shape (x,y,3) or (x,y);
        List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
        number of columns);
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subfigtitles : list or tuple of str
        Strings to be used as titles for the subfigures
    colorbar : bool
        True: plot colorbar
    automatic_positioning : bool
        True: Automatically position plots in subplot
    tight_layout : bool
        True: Create figure in tight_layout mode
    fontdict : dict
        fontdict of subfig .set_title function

    Returns
    -------
    Saves figure to filename
    """
    kwargs = locals().copy()
    start_function(_save_subplots, kwargs)


def save_movie(images, filename, title=None, subplotranges=(), tight_layout=False,
               interpolation='nearest', aspect='auto', resolution=None, fontdict=None, fps=30, interval=200):
    """Plotting list of images as subplots

    Parameters
    -------
    images : list of 2D or 3D numpy array of shape (x,y,3) or (x,y)
        list of 2D or 3D numpy arrays of shape (x,y,3) or (x,y)
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subplotranges : tuple(vmin, vmax) or None
        Tuples defining the maximum and minimum values to plot in the figure
    tight_layout : bool
        True: Create figure in tight_layout mode
    resolution : int
        resolution in dpi
    fontdict : dict
        fontdict of subfig .set_title function
    fps : int
        framerate of writer
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    Saves figure to filename
    """
    kwargs = locals().copy()
    start_function(_save_movie, kwargs)


def _save_subplots(images, filename, title=None, subfigtitles=(), subplotranges=(), colorbar=True,
                            automatic_positioning=False, tight_layout=False, interpolation='nearest', aspect='auto',
                            resolution=None, fontdict=None):
    """Plotting list of images as subplots

    Parameters
    -------
    images : list of lists of numpy arrays or list of numpy arrays or numpy array of shape (x,y,3) or (x,y)
        List of lists of images to plot; Images with shape (x,y,3) or (x,y);
        List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
        number of columns);
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subfigtitles : list of lists of strings or list of strings or string
        Strings to be used as titles for the subfigures
    subplotranges : list of lists of tuples or list of tuples or tuple
        Tuples defining the maximum and minimum values to plot in the figure
    colorbar : bool
        True: plot colorbar
    automatic_positioning : bool
        True: Automatically position plots in subplot
    tight_layout : bool
        True: Create figure in tight_layout mode
    fontdict : dict
        fontdict of subfig .set_title function

    Returns
    -------
    Saves figure to filename
    """
    # Check if filepath is valid
    if len(os.path.dirname(filename)):
        make_sure_path_exists(os.path.dirname(filename))
    
    # Check if images are valid, nest list to 2D if necessary
    images = _make_2d_list(images)
    if automatic_positioning:
        images = quadratic_list_layout(images)
    
    # Check if subfigtitles are valid, nest list to 2D if necessary
    subfigtitles = _make_2d_list(subfigtitles)
    if automatic_positioning:
        subfigtitles = quadratic_list_layout(subfigtitles)
    
    if fontdict is None:
        fontdict = dict(fontsize=6)
    
    # Check if plotranges are valid, nest list to 2D if necessary
    subplotranges = _make_2d_list(subplotranges)
    if automatic_positioning:
        subplotranges = quadratic_list_layout(subplotranges)
    
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
            if row is None:
                continue
            # Check dimensions of image, squeeze if necessary
            if len(row.shape) == 3:
                if row.shape[-1] == 1:
                    # If an image has shape (x, y, 1) flatten it to (x, y) to plot it as grayscale
                    row = row.reshape(row.shape[:-1])
            
            data = np.array(row, dtype=np.float)
            if len(data.shape) > 1:
                # Plot image at position in subfigure with optional color-range
                try:
                    if len(data.shape) > 1:
                        # For image data
                        if subplotranges[col_i][row_i]:
                            im = ax[row_i][col_i].imshow(data, interpolation=interpolation,
                                                         vmin=subplotranges[col_i][row_i][0],
                                                         vmax=subplotranges[col_i][row_i][1])
                        else:
                            im = ax[row_i][col_i].imshow(data, interpolation=interpolation)
                except IndexError:
                    im = ax[row_i][col_i].imshow(np.array(row, dtype=np.float), interpolation=interpolation)
                
                # Try to add title
                try:
                    ax[row_i][col_i].set_title(subfigtitles[col_i][row_i], fontdict=fontdict)
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
                    cbar = pl.colorbar(im, cax=cax)  # , ticks=MultipleLocator(0.2), format="%.2f")
                    cbar.ax.tick_params(labelsize=5)
                
                ax[row_i][col_i].get_xaxis().set_visible(False)
                ax[row_i][col_i].get_yaxis().set_visible(False)
                ax[row_i][col_i].set_aspect(aspect)
            else:
                # For 1D data
                im = ax[row_i][col_i].plot(data)
                
                # Try to add title
                try:
                    ax[row_i][col_i].set_title(subfigtitles[col_i][row_i], fontdict=dict(fontsize=6))
                except IndexError:
                    pass
                
                ax[row_i][col_i].get_xaxis().set_visible(True)
                ax[row_i][col_i].get_yaxis().set_visible(True)
                ax[row_i][col_i].set_aspect(aspect)
    
    if tight_layout:
        pl.tight_layout()
    
    if resolution is None:
        f.savefig(filename)
    else:
        f.savefig(filename, dpi=resolution)
    
    pl.close()


def _save_movie(images, filename, title=None, subplotranges=(), tight_layout=False,
                interpolation='nearest', aspect='auto', resolution=None, fontdict=None, fps=30, interval=200):
    """Plotting list of images as subplots

    Parameters
    -------
    images : list of 2D or 3D numpy array of shape (x,y,3) or (x,y)
        list of 2D or 3D numpy arrays of shape (x,y,3) or (x,y)
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subplotranges : tuple(vmin, vmax) or None
        Tuples defining the maximum and minimum values to plot in the figure
    tight_layout : bool
        True: Create figure in tight_layout mode
    resolution : int
        resolution in dpi
    fontdict : dict
        fontdict of subfig .set_title function
    fps : int
        framerate of writer
    interval : int
        Delay between frames in milliseconds

    Returns
    -------
    Saves figure to filename
    """
    # Check if filepath is valid
    if len(os.path.dirname(filename)):
        make_sure_path_exists(os.path.dirname(filename))
    
    # Check if images are valid, i.d. list of 2D arrays
    images = [np.array(im, dtype=np.float) if len(im.shape) == 2 or len(im.shape) == 3 else None for im in images]
    if np.any(np.isnan(images)):
        raise ValueError("List of images must only contain 2D or 3D numpy arrays!")
    
    if fontdict is None:
        fontdict = dict(fontsize=6)
    
    # Create empty subplot
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect(aspect)
    
    try:
        im = ax.imshow(images[0], interpolation=interpolation, vmin=subplotranges[0], vmax=subplotranges[1])
    except IndexError:
        im = ax.imshow(images[0], interpolation=interpolation)
    
    if title is not None:
        fig.suptitle(title, fontdict=fontdict)
    
    if tight_layout:
        pl.tight_layout()
    
    def update_img(n):
        try:
            im.set_data(images[n])
        except IndexError:
            im.set_data(images[n])
        return im
    
    ani = animation.FuncAnimation(fig, update_img, frames=len(images), interval=interval)
    writer = animation.writers['ffmpeg'](fps=fps)
    ani.save(filename, writer=writer, dpi=resolution)
    del ani
    pl.close()


########################################################################################################################
# Under Construction, use at own risk
########################################################################################################################

def save_subplots_line_plots(images, filename, title=None, subfigtitles=(), subplotranges=(),
                             automatic_positioning=False, tight_layout=False, aspect='auto',
                             resolution=None, fontdict=None):
    """Plotting list of line plots as subplots

    Parameters
    -------
    images : list of lists of numpy arrays or list of numpy arrays or numpy array of shape (x,y,3) or (x,y)
        List of lists of images to plot; Images with shape (x,y,3) or (x,y);
        List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
        number of columns);
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subfigtitles : list or tuple of str
        Strings to be used as titles for the subfigures
    automatic_positioning : bool
        True: Automatically position plots in subplot
    tight_layout : bool
        True: Create figure in tight_layout mode
    fontdict : dict
        fontdict of subfig .set_title function

    Returns
    -------
    Saves figure to filename
    """
    kwargs = locals().copy()
    start_function(_save_subplots_line_plots, kwargs)


def _save_subplots_line_plots(images, filename, title=None, subfigtitles=(), subplotranges=(),
                              automatic_positioning=False, tight_layout=False, aspect='auto',
                              resolution=None, fontdict=None, x_size=15, y_size=15):
    """Plotting list of line plots as subplots

    Parameters
    -------
    images : list of lists of numpy arrays or list of numpy arrays or numpy array of shape (x,y,3) or (x,y)
        List of lists of images to plot; Images with shape (x,y,3) or (x,y);
        List with shape [[image_11, image_21], [image_12, image_22]] for placement in subfigure (len(images) will be the
        number of columns);
    filename : str
        Filename to save subplot as
    title : str
        Main title of plot
    subfigtitles : list of lists of strings or list of strings or string
        Strings to be used as titles for the subfigures
    subplotranges : list of lists of tuples or list of tuples or tuple
        Tuples defining the maximum and minimum values to plot in the figure
    automatic_positioning : bool
        True: Automatically position plots in subplot
    tight_layout : bool
        True: Create figure in tight_layout mode
    fontdict : dict
        fontdict of subfig .set_title function

    Returns
    -------
    Saves figure to filename
    """
    # Check if filepath is valid
    if len(os.path.dirname(filename)):
        make_sure_path_exists(os.path.dirname(filename))
    
    # Check if images are valid, nest list to 2D if necessary
    images = _make_2d_list(images)
    if automatic_positioning:
        images = quadratic_list_layout(images)
    
    # Check if subfigtitles are valid, nest list to 2D if necessary
    subfigtitles = _make_2d_list(subfigtitles)
    if automatic_positioning:
        subfigtitles = quadratic_list_layout(subfigtitles)
    
    if fontdict is None:
        fontdict = dict(fontsize=6)
    
    # Create empty subplot
    n_cols = len(images)
    n_rows = max([len(l) for l in images])
    if n_cols <= 1:
        n_cols = 2
    
    if n_rows <= 1:
        n_rows = 2
    
    f, ax = pl.subplots(n_rows, n_cols, figsize=(x_size, y_size))
    
    if title is not None:
        f.suptitle(title)
    
    for col in ax:
        for row in col:
            row.get_xaxis().set_visible(False)
            row.get_yaxis().set_visible(False)
    
    for col_i, col in enumerate(images):
        for row_i, row in enumerate(col):
            if row is None:
                continue
            # Check dimensions of image, squeeze if necessary
            if len(row.shape) == 3:
                if row.shape[-1] == 1:
                    # If an image has shape (x, y, 1) flatten it to (x, y) to plot it as grayscale
                    row = row.reshape(row.shape[:-1])
            
            data = np.array(row, dtype=np.float)
            if len(data.shape) > 1:
                
                for i in range(0, data.shape[1]):
                    im = ax[row_i][col_i].plot(data[:, i])
                
                # Try to add title
                try:
                    ax[row_i][col_i].set_title(subfigtitles[col_i][row_i], fontdict=dict(fontsize=6))
                except IndexError:
                    pass
                
                ax[row_i][col_i].get_xaxis().set_visible(True)
                ax[row_i][col_i].get_yaxis().set_visible(True)
                ax[row_i][col_i].set_aspect(aspect)
            else:
                # For 1D data
                im = ax[row_i][col_i].plot(data)
                
                # Try to add title
                try:
                    ax[row_i][col_i].set_title(subfigtitles[col_i][row_i], fontdict=dict(fontsize=6))
                except IndexError:
                    pass
                
                ax[row_i][col_i].get_xaxis().set_visible(True)
                ax[row_i][col_i].get_yaxis().set_visible(True)
                ax[row_i][col_i].set_aspect(aspect)
    
    if tight_layout:
        pl.tight_layout()
    
    if resolution is None:
        f.savefig(filename)
    else:
        f.savefig(filename, dpi=resolution)
    
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
                pl.colorbar(im[row_i + col_i * len(col)], cax=cax)  # , ticks=MultipleLocator(0.2), format="%.2f")
            ax[row_i][col_i].get_xaxis().set_visible(False)
            ax[row_i][col_i].get_yaxis().set_visible(False)
            ax[row_i][col_i].set_aspect('auto')
    
    # pl.tight_layout()
    
    def make_frame(vid):
        for f_i, frame in enumerate(len(vid)):
            images = vid[frame]
            for col_i, col in enumerate(images):
                for row_i, row in enumerate(col):
                    # plot image at position in subfigure
                    im[row_i + col_i * len(col)].set_data(np.array(row, dtype=np.float))
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
                        pl.colorbar(im[row_i + col_i * len(col)],
                                    cax=cax)  # , ticks=MultipleLocator(0.2), format="%.2f")
                    ax[row_i][col_i].get_xaxis().set_visible(False)
                    ax[row_i][col_i].get_yaxis().set_visible(False)
                    
                    # pl.tight_layout()
        yield im
    
    animated = animation.FuncAnimation(f, make_frame, vid, blit=False, interval=interval, repeat=False)
    animated.save(filename, dpi=dpi)
    pl.close()


class Plotter(object):
    def __init__(self, plot_function=None, plot_kwargs=None, session=None, verbose: bool = False):
        """Base class for plotting

        Manages a plotting function plot_function and its arguments plot_kwargs which can be tensors. Call
        Plotter.plot() to perform plotting.
        Call TeLL.utility.plotting.launch_plotting_daemon() before creating Plotter-instances to perform plotting in
        background threads. Call TeLL.utility.plotting.terminate_plotting_daemon() to correctly terminate the threads.

        Parameters
        -------
        plot_function : function
            Function used for creating/saving the plots. Should take kwargs arguments.
        plot_kwargs : dict or None
            This dictionary consists of {key: value} to be passed to the plotting function. If value has a .eval()
            function, it is treated as tf.Tensor and evaluated before passing it on to the plotting function. Lists of
            tf.Tensor are also allowed.
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
        if PLOTTING_QUEUE is not None and PLOTTING_PROC is not None:
            (self.plotting_queue, self.plotting_proc) = (PLOTTING_QUEUE, PLOTTING_PROC)
        
        #
        # Initialize plotting function
        #
        self.plotting_function = plot_function
        
        #
        # Initialize kwargs for plotting
        #
        self.plot_kwargs = OrderedDict()
        if plot_kwargs is not None:
            self.plot_kwargs = plot_kwargs
        
        self.session = session
        self.tensor_vals = []
    
    def get_tensor_kwargs(self):
        """Return ordered dictionary {key: value} containing only the tensorflow tensors in kwargs"""
        tensor_kwargs = OrderedDict()
        for item in self.plot_kwargs.items():
            if hasattr(item[1], 'eval'):
                # If item is tensor, add it to tensor_kwargs
                tensor_kwargs[item[0]] = item[1]
            elif isinstance(item[1], list):
                # If item is a list of tensors, add it to tensor_kwargs
                if hasattr(item[1][0], 'eval'):
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
    
    def plot(self, session=None, feed_dict=None, evaluate_tensors=True):
        """Execute the plotting function defined via Plotter.set_plotting_function() (in background workers if
        existent). Tensors will be evaluated before the plotting function is called.

        Parameters
        --------
        session : tensorflow session or None
            Tensorflow session to use for evaluation of tensors; If None the session defined in __init__() or
            set_session() will be used;
        feed_dict : dict or None
            Optional feed_dict for tensorflow session
        evaluate_tensors : bool
            If True, the tensors will be evaluated; If False, the previously determined results will be used for the
            tensors (see access_tensor_values()):
        """
        evaluated_params = self.plot_kwargs.copy()
        
        tensor_dict_items = list(self.get_tensor_kwargs().items())
        tensor_keys = [i[0] for i in tensor_dict_items]
        tensor_vals = [i[1] for i in tensor_dict_items]
        tensor_vals_lens = [len(tv) if isinstance(tv, list) else 1 for tv in tensor_vals]
        tensor_vals = [tv for tvs in tensor_vals for tv in tvs]  # flatten list in case of list of tensors
        
        # Evaluate tensors in plot_kwargs if necessary
        if len(self.get_tensor_kwargs()):
            if evaluate_tensors:
                if session is None:
                    session = self.session
                if feed_dict is None:
                    feed_dict = {}
                
                tensor_vals_eval = session.run(tensor_vals, feed_dict=feed_dict)
                tensor_vals = []
                
                # Handle lists of tensors
                for tvl in tensor_vals_lens:
                    if tvl > 1:
                        tensor_vals += tensor_vals_eval[:tvl]
                    else:
                        tensor_vals += [tensor_vals_eval]
                    tensor_vals_eval = tensor_vals_eval[tvl:]
                
                self.tensor_vals = tensor_vals
            else:
                tensor_vals = self.tensor_vals
        
        tensor_evals = OrderedDict(zip(tensor_keys, tensor_vals))
        evaluated_params.update(tensor_evals)
        
        self.raw_plot(evaluated_params)
    
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
