# Tensorflow Layer Library (TeLL)
Provides a variety of tensorflow-based network layers, flexible (recurrent) network designs, convenience routines for saving and resuming networks, and more!

Copyright (c) Michael Widrich and Markus Hofmarcher, Institute of Bioinformatics, Johannes Kepler University Linz, Austria.

If you use TeLL or parts of the code in your work, please cite accordingly.

## Setup
You can either install TeLL via pip, use TeLL as a git-submodule in your git project, or download it as a static Python package.

If you intend on using multiple versions of TeLL in different projects, we recommend to use the [git-submodule](#tell-as-git-submodule) approach.
If you use the same TeLL version with all of your projects, a [pip installation](#tell-as-pip-package) or [static Python package](#tell-as-static-python-package) is sufficient.

TeLL has been testet with tensorflow version 1.0.

### TeLL as Pip Package
Download the TeLL package and install it via

```
pip install yourpath/tensorflow-layer-library
```

If you want to install the tensorflow dependencies as well, specify "tensorflow" for CPU only or "tensorflow-gpu" for GPU support in brackets after the package, e.g.

```
pip install yourpath/tensorflow-layer-library[tensorflow-gpu]
```

Continue with section [Usage](#usage).

### TeLL as Static Python Package
Download TeLL from GitLab or clone it to your disk. Continue with section [Usage](#usage).

### TeLL as Git-Submodule
If you want to keep TeLL as a subfolder in your git-project, with different git-projects having different TeLL versions, it may be best to add TeLL as a git-submodule to your project.
This will create a subfolder "tensorflow-layer-library" in your project folder, which can be separately updated to the last version of TeLL.
Let us assume that your project folder has the following structure:
``` ruby
myproject/
| 
+-- my_main_file.
|
+-- my_other_file.py
```

As described in [this](https://git-scm.com/book/en/v2/Git-Tools-Submodules) guide, you will have to move into your project directory and add the submodule:

```
cd myproject/
git submodule add https://the-git-repository-address.git
```

This will add the submodule and a .gitmodules file to your directory, resulting in the following structure:

``` ruby
myproject/
| 
+-- my_main_file.py
|
+-- tensorflow-layer-library/
|   |
|   +-- TeLL/
|
+-- .gitmodules
```

Now you have to change the path in the .gitmodules file to a relative path, if your project is hosted on the same server as the submodule:

```
[submodule "tensorflow-layer-library"]
  path = tensorflow-layer-library
  url = ../../TeLL/tensorflow-layer-library.git
```

Sources: [https://git-scm.com/book/en/v2/Git-Tools-Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules),
[https://docs.gitlab.com/ce/ci/git_submodules.html](https://docs.gitlab.com/ce/ci/git_submodules.html)

### Run Example Main-File
Try to run one of the examples [samples/main_lstm.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/samples/main_lstm.py) or [samples/main_convlstm.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/samples/main_convlstm.py) provided in the tensorflow-layer-library folder.
The following should start the computations and create a working_dir folder in the tensorflow-layer-library/samples folder:

```
cd tensorflow-layer-library/samples/
python3 main_lstm_example.py --config lstm_example.json
```

## Usage:
A focus of this project is to provide easy and fast usability while keeping the design flexible.
There are three basic steps to perform to create and run your architecture:

### Design Dataloader
In order to access/create your dataset, a reader/loader class should be used. This class has to contain a batch_loader() function to yield the minibatches.
Examples for creator-classes are ShortLongDataset and MovingDotDataset in [TeLL/datasets.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/TeLL/datasets.py), which can be adapted for your needs.
For reading data, [TeLL/datasets.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/TeLL/datasets.py) provides the classes DatareaderSimpleFiles and DatareaderAdvancedImageFiles, from which readers can be derived from.
DatareaderSimpleFiles and DatareaderAdvancedImageFiles provide support for automatic loading of data in background processes, search for datafiles, etc..

### Design Network Architecture
It is advised to create a new class for each network architecture, e.g. in a file my_architectures.py in your project folder.
In general, the layers can simply be stacked as follows:

```python
# define some placeholder for the input and target
X = tf.placeholder(tf.float32, shape=input_shape)
y_ = tf.placeholder(tf.float32, shape=target_shape)

# stack some layers
layer1 = Layer(incoming=X, ...)
layer2 = Layer(incoming=layer1, ...)
outputlayer = Layer(incoming=layer2, ...)

# calculate the output of the last layer
output = outputlayer.get_output()
```

A collection of forward- and recurrent network sample architectures can be found in [TeLL/samples/sample_architectures.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/samples/sample_architectures.py).

### Adapt Main-File
To adapt the main-file to your needs, copy the example file [samples/main_lstm.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/samples/main_lstm.py) or [samples/main_convlstm.py](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/samples/main_convlstm.py) and modify the loss calculations, starting at line 246, and the dataloader.

Finally, you will need to create your configuration file (examples can be found in [samples/](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/tree/master/samples)) and you are good to go!

### Utility Features

#### Storage/Resumption
By default, TeLL will create checkpoints for each run in the working_dir folder.
These checkpoints contain a .zip of the directory the main file is located in, so that the code base is at the correct version when the run is resumed.

To resume an experiment execute tell-resume and pass the path to the working directory of the experiment as well as parameters you want to overwrite (e.g. epochs, data directories, learning rate, ...)

```
tell-resume <path to working dir containing results and 00-script.zip> --epochs <number of total epochs> --gpu <tensorflow gpu string> 
```

#### Plotting
tba

## Contributing

If you want to contribute to TeLL, please read the [guidelines](https://git.bioinf.jku.at/TeLL/tensorflow-layer-library/blob/master/CONTRIBUTING.md), create a branch or fork, and send merge-requests.
For contribution to this project, your have to assign the copyright of the contribution to the TeLL project.
Please include the statement "I hereby assign copyright in this code to the TeLL project, to be licensed under the same terms as the rest of the code." in your merge requests.