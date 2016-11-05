# -*- coding: utf-8 -*-
"""
.. _tut_raw_objects:

The :class:`Raw <mne.io.Raw>` data structure: continuous data
=============================================================
"""

from __future__ import print_function

import mne
import os.path as op
from matplotlib import pyplot as plt

###############################################################################
# Continuous data is stored in objects of type :class:`Raw <mne.io.Raw>`.
# The core data structure is simply a 2D numpy array (channels × samples,
# stored in a private attribute called `._data`) combined with an
# :class:`Info <mne.Info>` object (`.info` attribute)
# (see :ref:`tut_info_objects`).
#
# The most common way to load continuous data is from a .fif file. For more
# information on :ref:`loading data from other formats <ch_convert>`, or
# creating it :ref:`from scratch <tut_creating_data_structures>`.


###############################################################################
# Loading continuous data
# -----------------------

# Load an example dataset, the preload flag loads the data into memory now
data_path = op.join(mne.datasets.sample.data_path(), 'MEG',
                    'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(data_path, preload=True)
raw.set_eeg_reference()  # set EEG average reference

# Give the sample rate
print('sample rate:', raw.info['sfreq'], 'Hz')
# Give the size of the data matrix
print('channels x samples:', raw._data.shape)

###############################################################################
# .. note:: Accessing the `._data` attribute is done here for educational
#           purposes. However this is a private attribute as its name starts
#           with an `_`. This suggests that you should **not** access this
#           variable directly but rely on indexing syntax detailed just below.

###############################################################################
# Information about the channels contained in the :class:`Raw <mne.io.Raw>`
# object is contained in the :class:`Info <mne.Info>` attribute.
# This is essentially a dictionary with a number of relevant fields (see
# :ref:`tut_info_objects`).


###############################################################################
# Indexing data
# -------------
#
# To access the data stored within :class:`Raw <mne.io.Raw>` objects,
# it is possible to index the :class:`Raw <mne.io.Raw>` object.
#
# Indexing a :class:`Raw <mne.io.Raw>` object will return two arrays: an array
# of times, as well as the data representing those timepoints. This works
# even if the data is not preloaded, in which case the data will be read from
# disk when indexing. The syntax is as follows:

# Extract data from the first 5 channels, from 1 s to 3 s.
sfreq = raw.info['sfreq']
data, times = raw[:5, int(sfreq * 1):int(sfreq * 3)]
_ = plt.plot(times, data.T)
_ = plt.title('Sample channels')

###############################################################################
# -----------------------------------------
# Selecting subsets of channels and samples
# -----------------------------------------
#
# It is possible to use more intelligent indexing to extract data, using
# channel names, types or time ranges.

# Pull all MEG gradiometer channels:
# Make sure to use .copy() or it will overwrite the data
meg_only = raw.copy().pick_types(meg=True)
eeg_only = raw.copy().pick_types(meg=False, eeg=True)

# The MEG flag in particular lets you specify a string for more specificity
grad_only = raw.copy().pick_types(meg='grad')

# Or you can use custom channel names
pick_chans = ['MEG 0112', 'MEG 0111', 'MEG 0122', 'MEG 0123']
specific_chans = raw.copy().pick_channels(pick_chans)
print(meg_only, eeg_only, grad_only, specific_chans, sep='\n')

###############################################################################
# Notice the different scalings of these types

f, (a1, a2) = plt.subplots(2, 1)
eeg, times = eeg_only[0, :int(sfreq * 2)]
meg, times = meg_only[0, :int(sfreq * 2)]
a1.plot(times, meg[0])
a2.plot(times, eeg[0])
del eeg, meg, meg_only, grad_only, eeg_only, data, specific_chans

###############################################################################
# You can restrict the data to a specific time range
raw = raw.crop(0, 50)  # in seconds
print('New time range from', raw.times.min(), 's to', raw.times.max(), 's')

###############################################################################
# And drop channels by name
nchan = raw.info['nchan']
raw = raw.drop_channels(['MEG 0241', 'EEG 001'])
print('Number of channels reduced from', nchan, 'to', raw.info['nchan'])

###############################################################################
# --------------------------------------------------
# Concatenating :class:`Raw <mne.io.Raw>` objects
# --------------------------------------------------
#
# :class:`Raw <mne.io.Raw>` objects can be concatenated in time by using the
# :func:`append <mne.io.Raw.append>` function. For this to work, they must
# have the same number of channels and their :class:`Info
# <mne.Info>` structures should be compatible.

# Create multiple :class:`Raw <mne.io.RawFIF>` objects
raw1 = raw.copy().crop(0, 10)
raw2 = raw.copy().crop(10, 20)
raw3 = raw.copy().crop(20, 40)

# Concatenate in time (also works without preloading)
raw1.append([raw2, raw3])
print('Time extends from', raw1.times.min(), 's to', raw1.times.max(), 's')
