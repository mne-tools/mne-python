"""
.. _tut_intro_pyton:

Introduction to Python
======================
"""

###############################################################################
# Python is a modern, general-purpose, object-oriented, high-level programming
# language. First make sure you have a working python environment and
# dependencies (see :ref:`install_python_and_mne_python`). If you are
# completely new to python, don't worry, it's just like any other programming
# language, only easier. Here are a few great resources to get you started:
#
# * `SciPy lectures <http://scipy-lectures.github.io>`_
# * `Learn X in Y minutes: Python <https://learnxinyminutes.com/docs/python/>`_
# * `NumPy for MATLAB users <https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_  # noqa
#
# We highly recommend watching the Scipy videos and reading through these
# sites to get a sense of how scientific computing is done in Python.
#
# Here are few bulletin points to familiarise yourself with python:
#
# Everything is dynamically typed. No need to declare simple data
# structures or variables separately.
a = 3
print(type(a))
b = [1, 2.5, 'This is a string']
print(type(b))
c = 'Hello world!'
print(type(c))

###############################################################################
# If you come from a background of matlab, remember that indexing in python
# starts from zero:
a = [1, 2, 3, 4]
print('This is the zeroth value in the list: {}'.format(a[0]))

###############################################################################
# No need to reinvent the wheel. Scipy and Numpy are battle field tested
# libraries that have a vast variety of functions for your needs. Consult the
# documentation and remember, you can always ask the interpreter for help with
# a question mark at the end of a function::
#
#    >>> import numpy as np
#    >>> np.arange?
