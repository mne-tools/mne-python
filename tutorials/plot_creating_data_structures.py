"""
.. _tut_creating_data_structures:

Creating MNE-Python's data structures from scratch
==================================================
"""

from __future__ import print_function

import mne
import numpy as np


###############################################################################
# ------------------------------------------------------
# Creating :class:`Info <mne.Info>` objects
# ------------------------------------------------------
#
# .. note:: for full documentation on the `Info` object, see
#           :ref:`tut_info_objects`. See also
#           :ref:`sphx_glr_auto_examples_io_plot_objects_from_arrays.py`.
#
# Normally, :class:`mne.Info` objects are created by the various
# :ref:`data import functions <ch_convert>`.
# However, if you wish to create one from scratch, you can use the
# :func:`mne.create_info` function to initialize the minimally required
# fields. Further fields can be assigned later as one would with a regular
# dictionary.
#
# The following creates the absolute minimum info structure:

# Create some dummy metadata
n_channels = 32
sampling_rate = 200
info = mne.create_info(n_channels, sampling_rate)
print(info)

###############################################################################
# You can also supply more extensive metadata:

# Names for each channel
channel_names = ['MEG1', 'MEG2', 'Cz', 'Pz', 'EOG']

# The type (mag, grad, eeg, eog, misc, ...) of each channel
channel_types = ['grad', 'grad', 'eeg', 'eeg', 'eog']

# The sampling rate of the recording
sfreq = 1000  # in Hertz

# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
# will be added for them
montage = 'standard_1005'

# Initialize required fields
info = mne.create_info(channel_names, sfreq, channel_types, montage)

# Add some more information
info['description'] = 'My custom dataset'
info['bads'] = ['Pz']  # Names of bad channels

print(info)

###############################################################################
# .. note:: When assigning new values to the fields of an
#           :class:`mne.Info` object, it is important that the
#           fields are consistent:
#
#           - The length of the channel information field `chs` must be
#             `nchan`.
#           - The length of the `ch_names` field must be `nchan`.
#           - The `ch_names` field should be consistent with the `name` field
#             of the channel information contained in `chs`.
#
# ---------------------------------------------
# Creating :class:`Raw <mne.io.Raw>` objects
# ---------------------------------------------
#
# To create a :class:`mne.io.Raw` object from scratch, you can use the
# :class:`mne.io.RawArray` class, which implements raw data that is backed by a
# numpy array. The correct units for the data are:
#
# - V: eeg, eog, seeg, emg, ecg, bio, ecog
# - T: mag
# - T/m: grad
# - M: hbo, hbr
# - Am: dipole
# - AU: misc
#
# The :class:`mne.io.RawArray` constructor simply takes the data matrix and
# :class:`mne.Info` object:

# Generate some random data
data = np.random.randn(5, 1000)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)

custom_raw = mne.io.RawArray(data, info)
print(custom_raw)

###############################################################################
# ---------------------------------------------
# Creating :class:`Epochs <mne.Epochs>` objects
# ---------------------------------------------
#
# To create an :class:`mne.Epochs` object from scratch, you can use the
# :class:`mne.EpochsArray` class, which uses a numpy array directly without
# wrapping a raw object. The array must be of `shape(n_epochs, n_chans,
# n_times)`. The proper units of measure are listed above.

# Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
sfreq = 100
data = np.random.randn(10, 5, sfreq * 2)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=sfreq
)

###############################################################################
# It is necessary to supply an "events" array in order to create an Epochs
# object. This is of `shape(n_events, 3)` where the first column is the sample
# number (time) of the event, the second column indicates the value from which
# the transition is made from (only used when the new value is bigger than the
# old one), and the third column is the new event value.

# Create an event matrix: 10 events with alternating event codes
events = np.array([
    [0, 0, 1],
    [1, 0, 2],
    [2, 0, 1],
    [3, 0, 2],
    [4, 0, 1],
    [5, 0, 2],
    [6, 0, 1],
    [7, 0, 2],
    [8, 0, 1],
    [9, 0, 2],
])

###############################################################################
# More information about the event codes: subject was either smiling or
# frowning
event_id = dict(smiling=1, frowning=2)

###############################################################################
# Finally, we must specify the beginning of an epoch (the end will be inferred
# from the sampling frequency and n_samples)

# Trials were cut from -0.1 to 1.0 seconds
tmin = -0.1

###############################################################################
# Now we can create the :class:`mne.EpochsArray` object
custom_epochs = mne.EpochsArray(data, info, events, tmin, event_id)

print(custom_epochs)

# We can treat the epochs object as we would any other
_ = custom_epochs['smiling'].average().plot()

###############################################################################
# ---------------------------------------------
# Creating :class:`Evoked <mne.Evoked>` Objects
# ---------------------------------------------
# If you already have data that is collapsed across trials, you may also
# directly create an evoked array.  Its constructor accepts an array of
# `shape(n_chans, n_times)` in addition to some bookkeeping parameters.
# The proper units of measure for the data are listed above.

# The averaged data
data_evoked = data.mean(0)

# The number of epochs that were averaged
nave = data.shape[0]

# A comment to describe to evoked (usually the condition name)
comment = "Smiley faces"

# Create the Evoked object
evoked_array = mne.EvokedArray(data_evoked, info, tmin,
                               comment=comment, nave=nave)
print(evoked_array)
_ = evoked_array.plot()
