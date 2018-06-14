# Coding Conventions (Python)

In general, stick to the [PEP standards](https://www.python.org/dev/peps/pep-0008) as good as possible.

When adding new functions/modules, use the existing directory structure or include/describe the newly added folders or files in the README.md.

Keep consistent with existing file, function, and class docstrings. Use and update comments, so that others can quickly see what your code is about.

Thank you for contributing! :)

## Comments for File Structure

Top-level comment:
```python
code


# ----------------------------------------------------------------------------------------------------------------------
# Comment
# ----------------------------------------------------------------------------------------------------------------------

code
```

or

```python
code


# ----------------------------------------------------------------------------------------------------------------------
# Comment
# Description of following section
# ----------------------------------------------------------------------------------------------------------------------

code
```

Mid-level comment:
```python
code

#
# Comment
#
code
```

or

```python
code

#
# Comment
# Description of following section
#
code
```

Low-level  comment:

```python
code

# Comment
code
```


## PyCharm Configuration

If working with PyCharm, please use the provided [configuration file](https://gitlab.markushofmarcher.at/markus.hofmarcher/tools/blob/b2556cd8a097377eb5bf55beedc21b4992a91724/misc/codestyle_cs.xml). Import via

File->Settings->Editor->Code Style->Manage->Import...

File->Settings->Editor->File and Code Templates->Python

## Docstring conventions

[Numpy style](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) is interpretable by sphinx and recommended:

## File Template

```python
# -*- coding: utf-8 -*-
"""
Short discription of contents and purpose of file

"""
```