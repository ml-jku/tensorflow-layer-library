Tensorflow Layer Library (TeLL)
===============================

Provides a variety of tensorflow-based network layers, flexible
(recurrent) network designs, convenience routines for saving and
resuming networks, and more!

Copyright (c) 2016-2017 Michael Widrich and Markus Hofmarcher, Institute
of Bioinformatics, Johannes Kepler University Linz, Austria

Setup
-----

You can either use TeLL as a git-submodule in your git project or as a
static Python package.

If you intend on using multiple versions of TeLL in different projects,
we recommend to use the `git-submodule <#tell-as-git-submodule>`__
approach. If you use the same TeLL version with all of your projects, a
`static Python package <#tell-as-static-python-package>`__ is
sufficient.

TeLL will run with tensorflow version 1.0.

TeLL as Pip Package
~~~~~~~~~~~~~~~~~~~

Download the TeLL package and install it via

::

    pip install <path-to-package-file>

If you want to install the tensorflow dependencies as well specify
"tensorflow" or "tensorflow-gpu" in brackets after the package, e.g.

::

    pip install <path-to-package-file>[tensorflow-gpu]

TeLL as Static Python Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download TeLL from GitLab or clone it to your disk. Continue with
section `Usage <#usage>`__.

TeLL as Git-Submodule
~~~~~~~~~~~~~~~~~~~~~

If you want to keep TeLL as a subfolder in your git-project, with
different git-projects having different TeLL versions, it may be best to
add TeLL as a git-submodule to your project. This will create a
subfolder "tensorflow-layer-library" in your project folder, which can
be separately updated to the last version of TeLL. Let us assume that
your project folder has the following structure:

.. code:: ruby

    myproject/
    | 
    +-- my_main_file.
    |
    +-- my_other_file.py

As described in
`this <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__ guide,
you will have to move into your project directory and add the submodule:

::

    cd myproject/
    git submodule add https://the-git-repository-address.git

This will add the submodule and a .gitmodules file to your directory,
resulting in the following structure:

.. code:: ruby

    myproject/
    | 
    +-- my_main_file.py
    |
    +-- tensorflow-layer-library/
    |   |
    |   +-- TeLL/
    |
    +-- .gitmodules

Now you have to change the path in the .gitmodules file to a relative
path, if your project is hosted on the same server as the submodule:

::

    [submodule "tensorflow-layer-library"]
      path = tensorflow-layer-library
      url = ../../TeLL/tensorflow-layer-library.git

Sources: https://git-scm.com/book/en/v2/Git-Tools-Submodules,
https://docs.gitlab.com/ce/ci/git_submodules.html

Run Example Main-File
~~~~~~~~~~~~~~~~~~~~~

Try to run one of the examples
`main\_lstm\_example.py <samples/main_lstm_example.py>`__
or
`main\_convlstm\_example.py <samples/main_convlstm_example.py>`__
provided in the tensorflow-layer-library folder. The following should
start the computations and create a working\_dir folder in the
tensorflow-layer-library folder:

::

    cd tensorflow-layer-library/
    python3 main_lstm_example.py --config TeLL/configs/examples/lstm_example.json

Usage:
------

A focus of this project is to provide easy and fast usability while
keeping the design flexible. There are three basic steps to perform to
create and run your architecture:

Design Dataloader
~~~~~~~~~~~~~~~~~

In order to access/create your dataset, a reader/loader class should be
used. This class has to contain a batch\_loader() function to yield the
minibatches. Examples for creator-classes are ShortLongDataset and
MovingDotDataset in
`TeLL/network\_modules/datasets.py <TeLL/network_modules/datasets.py>`__,
which can be adapted for your needs. For reading data,
`TeLL/network\_modules/datasets.py <TeLL/network_modules/datareader.py>`__
provides the classes DatareaderSimpleFiles and
DatareaderAdvancedImageFiles, from which readers can be derived from.
DatareaderSimpleFiles and DatareaderAdvancedImageFiles provide support
for automatic loading of data in background processes, search for
datafiles, etc..

Design Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is advised to create a new class for each network architecture, e.g.
in a file my\_architectures.py in your project folder. In general, the
layers can simply be stacked as follows:

.. code:: python

    # define some placeholder for the input and target
    X = tf.placeholder(tf.float32, shape=input_shape)
    y_ = tf.placeholder(tf.float32, shape=target_shape)

    # stack some layers
    layer1 = Layer(incoming=X, ...)
    layer2 = Layer(incoming=layer1, ...)
    outputlayer = Layer(incoming=layer2, ...)

    # calculate the output of the last layer
    output = outputlayer.get_output()

A collection of forward- and recurrent network sample architectures can
be found in
`TeLL/architectures/sample\_architectures.py <TeLL/architectures/sample_architectures.py>`__.

Adapt Main-File
~~~~~~~~~~~~~~~

To adapt the main-file to your needs, copy the example file
`TeLL/architectures/main\_lstm\_example.py <main_lstm_example.py>`__
or
`architectures/main\_lstm\_example.py <TeLL/architectures/main_convlstm_example.py>`__
and modify the loss calculations, starting at line 246, and the
dataloader.

You will probably also have to adapt the path in variable
tell\_library\_path (first code line in the main files) to your path
'/somepath/tensorflow-layer-library/'. Alternatively, you may also add
the path to the system's PYTHONPATH.

Finally, you will need to create your configuration file (examples can
be found in
`TeLL/configs/examples <TeLL/configs/examples>`__)
and you are good to go!

Utility Features
~~~~~~~~~~~~~~~~

Storage/Resumption
^^^^^^^^^^^^^^^^^^

By default, TeLL will create checkpoints for each run in the
working\_dir folder. These checkpoints contain a .zip of the directory
the main file is located in, so that the code base is at the correct
version when the run is resumed.

To resume an experiment run the following command:

::

    tell-resume --epochs <number of total epochs> --gpu <tensorflow gpu string> --path <path to working dir containing results and 00-script.zip>

Plotting
^^^^^^^^

Directory Structure
-------------------

The project directory is structured as follows:

.. code:: ruby

    tensorflow-layer-library/
    | '''the TeLL project, including example scripts'''
    +-- TeLL/
    |   | '''the TeLL package'''
    |   +-- architectures/
    |   |   +-- sample_architectures.py
    |   |     '''some example network architectures'''
    |   +-- configs/
    |   |   +-- examples/
    |   |   | '''example configuration files for usage with sample_architectures.py'''
    |   |   +-- config.py
    |   |     '''default configuration settings'''
    |   +-- network_modules/
    |   | '''holds modules for network'''
    |   |   +-- datareader.py
    |   |   | '''base class for dataset readers'''
    |   |   +-- datasets.py
    |   |   | '''classes for dataset loaders and creators'''
    |   |   +-- initializations.py
    |   |   | '''initializers for variables'''
    |   |   +-- layers.py 
    |   |   | '''network layer classes'''
    |   |   +-- loss.py 
    |   |   | '''loss functions'''
    |   |   +-- regularization.py
    |   |     '''regularization functions'''
    |   +-- utility/
    |     '''holds convenience functions'''
    |       +-- misc.py 
    |       | '''unclassified convenience functions'''
    |       +-- plotting.py 
    |       | '''functions for plotting and saving images/videos'''
    |       +-- plotting_daemons.py
    |         '''functions for creating and starting (sub)processes for plotting'''
    +-- README.md
    |     '''this file'''
    +-- main_lstm_example.py
    |     '''example main file for LSTM architectures'''
    +-- main_convlstm_example.py
    |     '''example main file for convLSTM architectures'''
    +-- main_convlstm_advanced_example.py
    |     '''example main file for advanced convLSTM architectures'''
    +-- todo.py
          '''todo-list: indicate on what you are working and strikethrough when you are done'''