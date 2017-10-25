"""
.. _tut_intro_python:

Introduction to Python
======================

`Python <https://www.python.org/>`_ is a modern general-purpose object-oriented
high-level programming language. First make sure you have a working Python
environment and dependencies (see :ref:`install_python_and_mne_python`). If you
are completely new to Python, don't worry, it's just like any other programming
language, only easier. Here are a few great resources to get you started:

* `SciPy lectures <http://scipy-lectures.github.io>`_
* `Learn X in Y minutes: Python <https://learnxinyminutes.com/docs/python/>`_
* `NumPy for MATLAB users <https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_

We highly recommend watching the SciPy videos and reading through these sites
to get a sense of how scientific computing is done in Python.
"""    # noqa
###############################################################################
# Here are few important points to familiarize yourself with Python. First,
# everything is dynamically typed. There is no need to declare and initialize
# data structures or variables separately.

a = 3
print(type(a))
b = [1, 2.5, 'This is a string']
print(type(b))
c = 'Hello world!'
print(type(c))

###############################################################################
# Second, if you have a MATLAB background remember that indexing in Python
# starts from zero (and is done with square brackets):
a = [1, 2, 3, 4]
print('This is the zeroth value in the list: {}'.format(a[0]))

###############################################################################
# Finally, often there is no need to reinvent the wheel. SciPy and NumPy are
# battle-hardened libraries that offer a vast variety of functions for most
# needs. Consult the documentation and remember that you can always ask the
# IPython interpreter for help with a question mark at the beginning or end of
# a function:
#
#      >>> import numpy as np
#      >>> np.arange?
