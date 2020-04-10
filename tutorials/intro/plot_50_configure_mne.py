# -*- coding: utf-8 -*-
"""
.. _tut-configure-mne:

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
import mne

###############################################################################
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

print(mne.get_config('MNE_USE_CUDA'))
print(type(mne.get_config('MNE_USE_CUDA')))

###############################################################################
# Note that the string values read from the JSON file are not parsed in any
# way, so :func:`~mne.get_config` returns a string even for true/false config
# values, rather than a Python :ref:`boolean <bltin-boolean-values>`.
# Similarly, :func:`~mne.set_config` will only set string values (or ``None``
# values, to unset a variable):

try:
    mne.set_config('MNE_USE_CUDA', True)
except TypeError as err:
    print(err)

###############################################################################
# If you're unsure whether a config variable has been set, there is a
# convenient way to check it and provide a fallback in case it doesn't exist:
# :func:`~mne.get_config` has a ``default`` parameter.

print(mne.get_config('missing_config_key', default='fallback value'))

###############################################################################
# There are also two convenience modes of :func:`~mne.get_config`. The first
# will return a :class:`dict` containing all config variables (and their
# values) that have been set on your system; this is done by passing
# ``key=None`` (which is the default, so it can be omitted):

print(mne.get_config())  # same as mne.get_config(key=None)

###############################################################################
# The second convenience mode will return a :class:`tuple` of all the keys that
# MNE-Python recognizes and uses, regardless of whether they've been set on
# your system. This is done by passing an empty string ``''`` as the ``key``:

print(mne.get_config(key=''))

###############################################################################
# It is possible to add config variables that are not part of the recognized
# list, by passing any arbitrary key to :func:`~mne.set_config`. This will
# yield a warning, however, which is a nice check in cases where you meant to
# set a valid key but simply misspelled it:

mne.set_config('MNEE_USE_CUUDAA', 'false')

###############################################################################
# Let's delete that config variable we just created. To unset a config
# variable, use :func:`~mne.set_config` with ``value=None``. Since we're still
# dealing with an unrecognized key (as far as MNE-Python is concerned) we'll
# still get a warning, but the key will be unset:

mne.set_config('MNEE_USE_CUUDAA', None)
assert 'MNEE_USE_CUUDAA' not in mne.get_config('')

###############################################################################
# Where configurations are stored
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python stores configuration variables in a `JSON`_ file. By default, this
# file is located in :file:`{%USERPROFILE%}\\.mne\\mne-python.json` on Windows
# and :file:`{$HOME}/.mne/mne-python.json` on Linux or macOS. You can get the
# full path to the config file with :func:`mne.get_config_path`.

print(mne.get_config_path())

###############################################################################
# However it is not a good idea to directly edit files in the :file:`.mne`
# directory; use the getting and setting functions described in :ref:`the
# previous section <config-get-set>`.
#
# If for some reason you want to load the configuration from a different
# location, you can pass the ``home_dir`` parameter to
# :func:`~mne.get_config_path`, specifying the parent directory of the
# :file:`.mne` directory where the configuration file you wish to load is
# stored.
#
#
# Using environment variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For compatibility with :doc:`MNE-C <../../install/mne_c>`, MNE-Python
# also reads and writes `environment variables`_ to specify configuration. This
# is done with the same functions that read and write the JSON configuration,
# and is controlled with the parameters ``use_env`` and ``set_env``. By
# default, :func:`~mne.get_config` will check :data:`os.environ` before
# checking the MNE-Python JSON file; to check *only* the JSON file use
# ``use_env=False``. To demonstrate, here's an environment variable that is not
# specific to MNE-Python (and thus is not in the JSON config file):

# make sure it's not in the JSON file (no error means our assertion held):
assert mne.get_config('PATH', use_env=False) is None
# but it *is* in the environment:
print(mne.get_config('PATH'))

###############################################################################
# Also by default, :func:`~mne.set_config` will set values in both the JSON
# file and in :data:`os.environ`; to set a config variable *only* in the JSON
# file use ``set_env=False``. Here we'll use :func:`print` statement to confirm
# that an environment variable is being created and deleted (we could have used
# the Python :ref:`assert statement <assert>` instead, but it doesn't print any
# output when it succeeds so it's a little less obvious):

mne.set_config('foo', 'bar', set_env=False)
print('foo' in os.environ.keys())
mne.set_config('foo', 'bar')
print('foo' in os.environ.keys())
mne.set_config('foo', None)  # unsetting a key deletes var from environment
print('foo' in os.environ.keys())

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
# MNE-Python is ``info``:

print(mne.get_config('MNE_LOGGING_LEVEL'))

###############################################################################
# The logging levels that can be set as config variables are ``debug``,
# ``info``, ``warning``, ``error``, and ``critical``. Around 90% of the log
# messages in MNE-Python are ``info`` messages, so for most users the choice is
# between ``info`` (tell me what is happening) and ``warning`` (tell me only if
# something worrisome happens). The ``debug`` logging level is intended for
# MNE-Python developers.
#
#
# In :ref:`an earlier section <config-get-set>` we saw how
# :func:`mne.set_config` is used to change the logging level for the current
# Python session and all future sessions. To change the logging level only for
# the current Python session, you can use :func:`mne.set_log_level` instead.
# The :func:`~mne.set_log_level` function takes the same five string options
# that are used for the ``MNE_LOGGING_LEVEL`` config variable; additionally, it
# can accept :class:`int` or :class:`bool` values that are equivalent to those
# strings. The equivalencies are given in this table:
#
# .. _table-log-levels:
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
# With many MNE-Python functions it is possible to change the logging level
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
# from :ref:`the table above <table-log-levels>` that ``verbose=True`` will
# give us the ``info`` messages and ``verbose=False`` will suppress them; this
# is a useful shorthand to use in scripts, so you don't have to remember the
# specific names of the different logging levels. One final note:
# ``verbose=None`` (which is the default for functions that have a ``verbose``
# parameter) will fall back on whatever logging level was most recently set by
# :func:`mne.set_log_level`, or if that hasn't been called during the current
# Python session, it will fall back to the value of
# ``mne.get_config('MNE_LOGGING_LEVEL')``.
#
#
# .. LINKS
#
# .. _json: https://en.wikipedia.org/wiki/JSON
# .. _`environment variables`: https://wikipedia.org/wiki/Environment_variable
