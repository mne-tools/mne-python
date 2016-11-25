# -*- coding: utf-8 -*-
"""
========================================
Background information on configurations
========================================

This tutorial gives a short introduction to MNE configurations.
"""
import os.path as op

import mne
from mne.datasets.sample import data_path

fname = op.join(data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(fname).crop(0, 10)
original_level = mne.get_config('MNE_LOGGING_LEVEL', 'INFO')

###############################################################################
# MNE-python stores configurations to a folder called `.mne` in the user's
# home directory, or to AppData directory on Windows. The path to the config
# file can be found out by calling :func:`mne.get_config_path`.
print(mne.get_config_path())

###############################################################################
# These configurations include information like sample data paths and plotter
# window sizes. Files inside this folder should never be modified manually.
# Let's see what the configurations contain.
print(mne.get_config())

###############################################################################
# We see fields like "MNE_DATASETS_SAMPLE_PATH". As the name suggests, this is
# the path the sample data is downloaded to. All the fields in the
# configuration file can be modified by calling :func:`mne.set_config`.

###############################################################################
#
# .. _tut_logging:
#
# Logging
# =======
# Configurations also include the default logging level for the functions. This
# field is called "MNE_LOGGING_LEVEL".
mne.set_config('MNE_LOGGING_LEVEL', 'INFO')
print(mne.get_config(key='MNE_LOGGING_LEVEL'))

###############################################################################
# The default value is now set to INFO. This level will now be used by default
# every time we call a function in MNE. We can set the global logging level for
# only this session by calling :func:`mne.set_log_level` function.
mne.set_log_level('WARNING')
print(mne.get_config(key='MNE_LOGGING_LEVEL'))

###############################################################################
# Notice how the value in the config file was not changed. Logging level of
# WARNING only applies for this session. Let's see what logging level of
# WARNING prints for :func:`mne.compute_raw_covariance`.
cov = mne.compute_raw_covariance(raw)

###############################################################################
# Nothing. This means that no warnings were emitted during the computation. If
# you look at the documentation of :func:`mne.compute_raw_covariance`, you
# notice the ``verbose`` keyword. Setting this parameter does not touch the
# configurations, but sets the logging level for just this one function call.
# Let's see what happens with logging level of INFO.
cov = mne.compute_raw_covariance(raw, verbose=True)

###############################################################################
# As you see there is some info about what the function is doing. The logging
# level can be set to 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'. It can
# also be set to an integer or a boolean value. The correspondance to string
# values can be seen in the table below. ``verbose=None`` uses the default
# value from the configuration file.
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
mne.set_config('MNE_LOGGING_LEVEL', original_level)
