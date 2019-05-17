# -*- coding: utf-8 -*-
"""

Configuring MNE-Python
======================

This tutorial covers how to configure MNE-Python to suit your local system and
your analysis preferences.

.. contents:: Page contents
   :local:
   :depth: 1

We begin by importing the necessary Python modules:
"""

import os
import warnings
import mne

###############################################################################
# Where configurations are stored
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python stores configuration variables in a `JSON`_ file. By default, this
# file is located in :file:`{%USERPROFILE%}\.mne\mne-python.json` on Windows
# and :file:`{$HOME}/.mne/mne-python.json` on Linux or macOS. You can get the
# full path to the config file with :func:`mne.get_config_path`.

mne.get_config_path()

###############################################################################
# However it is not a good idea to directly edit files in the :file:`.mne`
# directory; use the getting and setting functions described in :ref:`the next
# section <config-get-set>`.
#
# If for some reason you want to load the configuration from a different
# location, you can pass the ``home_dir`` parameter to
# :func:`~mne.get_config_path`, specifying the parent directory of the
# :file:`.mne` directory where the configuration file you wish to load is
# stored.
#
#
# .. _config-get-set:
#
# Getting and setting configuration variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Configuration variables are read and written using the functions
# :func:`mne.get_config` and :func:`mne.set_config`. To read a specific
# configuration variable, pass its name to :func:`~mne.get_config` as the
# ``key`` parameter (``key`` is the first parameter so you can pass it unnamed
# if you want):

mne.get_config('MNE_USE_CUDA')

###############################################################################
# Notice that the string values read from the JSON file are not parsed in any
# way, so :func:`~mne.get_config` returns a string rather than a
# :ref:`Python boolean value <bltin-boolean-values>`. Similarly,
# :func:`~mne.set_config` will only set string values:

try:
    mne.set_config('MNE_USE_CUDA', True)
except TypeError as err:
    print(err)

###############################################################################
# If you're unsure whether a config variable has been set, there is a
# convenient way to check it and provide a fallback in case it doesn't exist:
# :func:`~mne.get_config` has a ``default`` parameter.

mne.get_config('foo', default='bar')

###############################################################################
# :func:`~mne.get_config` also has two convenience modes. The first will return
# all config variables that have been set on your system; this is done by
# passing ``key=None`` (which is the default):

mne.get_config()  # same as mne.get_config(key=None)

###############################################################################
# The second convenience mode will return a :class:`tuple` of all the keys that
# MNE-Python recognizes and uses, regardless of whether they've been set on
# your system. This is done by passing an empty string ``''`` as the ``key``:

mne.get_config(key='')

###############################################################################
# It is possible to add config variables that are not part of the recognized
# list, by passing any arbitrary key to :func:`~mne.set_config`. This will
# yield a warning, however, so we'll catch and print it so our tutorial will
# still run properly:

with warnings.catch_warnings(record=True) as w:
    mne.set_config('foo', 'bar')
    print(w[0].message)

###############################################################################
# Let's delete that config variable ``foo`` we just created. To unset a config
# variable, use :func:`~mne.set_config` with ``value=None``. Since we're still
# dealing with an unrecognized key (as far as MNE-Python is concerned) we'll
# use the same :func:`~warnings.catch_warnings` trick as above, but this time
# we won't bother printing the message again:

with warnings.catch_warnings(record=True):
    mne.set_config('foo', None)

###############################################################################
# Using environment variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For compatibility with :doc:`MNE-C <install_mne_c>`, MNE-Python also reads
# and writes `environment variables`_ to specify configuration. This is done
# with the same functions that read and write the JSON configuration, and is
# controlled with the parameters ``use_env`` and ``set_env``. By default,
# :func:`~mne.get_config` will check :attr:`os.environ` before checking the
# MNE-Python JSON file; to check *only* the JSON file use ``use_env=False``.
# To demonstrate, here's an environment variable that is
# not specific to MNE-Python (and thus is not in the JSON config file):

# make sure it's not in the JSON file:
assert mne.get_config('CONDA_DEFAULT_ENV', use_env=False) is None
# but it is in the environment:
mne.get_config('CONDA_DEFAULT_ENV')

###############################################################################
# Also by default, :func:`~mne.set_env` will set values in both the JSON file
# and in :attr:`os.environ`; to set a config variable *only* in the JSON file
# use ``set_env=False``. Here we'll use the Python :ref:`assert` statement to
# show that an environment variable is being created and deleted:

with warnings.catch_warnings(record=True):
    mne.set_config('foo', 'bar', set_env=False)
    assert 'foo' not in os.environ.keys()
    mne.set_config('foo', 'bar')
    assert 'foo' in os.environ.keys()
    mne.set_config('foo', None)  # deleting keys also deletes from environment
    assert 'foo' not in os.environ.keys()

###############################################################################
# .. _tut_logging:
#
# Logging
# ^^^^^^^
#
# One important configuration variable is ``MNE_LOGGING_LEVEL``. Throughout the
# module, messages are generated describing the actions MNE-Python is taking
# behind-the-scenes. How you set ``MNE_LOGGING_LEVEL`` determines how many of
# those messages you see. The default logging level on a fresh install of
# MNE-Python is `'info'`:

mne.get_config('MNE_LOGGING_LEVEL')

###############################################################################
# The logging levels that can be set as config variables are ``debug``,
# ``info``, ``warning``, ``error``, and ``critical``. Around 90% of the log
# messages in MNE-Python are ``info`` messages, so for most users the choice is
# between ``info`` (tell me what is happening) and ``warning`` (tell me only if
# something worrisome happens). The ``debug`` logging level is intended for
# MNE-Python developers.
#
#
# :func:`mne.set_config` is used to change the logging level for the current
# Python session and all future sessions. To change the logging level only for
# the current Python session, you can use :func:`mne.set_log_level` instead.
# :func:`~mne.set_log_level` takes the same five string options that are used
# for the ``MNE_LOGGING_LEVEL`` config variable; additionally, it can accept
# :class:`int` or :class:`bool` values that are equivalent to the strings. The
# equivalencies are given in this table:
#
# .. _log-levels-table:
#
# +----------+---------+---------+
# | String   | Integer | Boolean |
# +==========+=========+=========+
# | DEBUG    | 10      |         |
# +----------+---------+---------+
# | INFO     | 20      | True    |
# +----------+---------+---------+
# | WARNING  | 30      | False   |
# +----------+---------+---------+
# | ERROR    | 40      |         |
# +----------+---------+---------+
# | CRITICAL | 50      |         |
# +----------+---------+---------+
#
# Finally, with many functions it is possible to change the logging level
# temporarily for just that function call, by using the ``verbose`` parameter.
# To illustrate this, we'll load some sample data with different logging levels
# set. First, with log level ``warning``:


kit_data_path = os.path.join(os.path.abspath(os.path.dirname(mne.__file__)),
                             'io', 'kit', 'tests', 'data', 'test.sqd')
raw = mne.io.read_raw_kit(kit_data_path, verbose='warning')

###############################################################################
# No messages were generated, because none of the messages were of severity
# "warning" or worse. Next, we'll load the same file with log level ``info``
# (the default level):

raw = mne.io.read_raw_kit(kit_data_path, verbose='info')

###############################################################################
# This time, we got a few messages about extracting information from the file,
# converting that information into the MNE-Python :class:`~mne.Info` format,
# etc. Finally, if we request ``debug``-level information, we get even more
# detail:

raw = mne.io.read_raw_kit(kit_data_path, verbose='debug')

###############################################################################
# We've been passing string values to the ``verbose`` parameter, but we can see
# from the table that ``verbose=True`` will give us the ``info`` messages and
# ``verbose=False`` will suppress them; this is a useful shorthand to use in
# scripts, so you don't have to remember the specific names of the different
# logging levels. One final note: ``verbose=None`` (which is the default for
# functions that have a ``verbose`` parameter) will fall back on whatever
# logging level was most recently set by :func:`mne.set_log_level`, or if that
# hasn't been called during the current Python session, it will fall back to
# the value of ``mne.get_config('MNE_LOGGING_LEVEL')``.
#
#
# .. LINKS
#
# .. _json: https://en.wikipedia.org/wiki/JSON
# .. _`environment variables`: https://en.wikipedia.org/wiki/Environment_variable`
