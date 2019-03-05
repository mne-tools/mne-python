# -*- coding: utf-8 -*-
"""
.. _subselecting-raw-tutorial:

Subselecting data from :class:`~mne.io.Raw` objects
===================================================

This tutorial covers how to select portions of data the :class:`~mne.io.Raw`
(both channels and time spans), how to reorder and rename channels, and also
illustrates some basic plotting of :class:`mne.io.Raw` data using matplotlib.
As before, we'll start by importing the Python modules we need, and repeating
the commands to load the raw data:
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)


###############################################################################
# Indexing :class:`~mne.io.Raw` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To select portions of the data, :class:`~mne.io.Raw` objects can be indexed
# using square brackets. However, indexing :class:`~mne.io.Raw` works
# differently than indexing a NumPy array in two ways:
#
# 1. Along with the requested sample value(s) MNE-Python also returns an array
#    of times (in seconds) corresponding to the requested samples. The
#    data array and the times array are returned together as elements of a
#    tuple.
#
# 2. The data array will always be 2-dimensional even if you request only a
#    single time sample or a single channel.
#
# To illustrate this, let's select just a few seconds of data from the first
# channel:

sampling_frequency = raw.info['sfreq']
starting_sample = int(10 * sampling_frequency)
ending_sample = int(13 * sampling_frequency)
raw_selection = raw[0, starting_sample:ending_sample]
print(raw_selection)

###############################################################################
# You can see that it contains 2 arrays. This combination of data and times
# makes it easy to plot selections of raw data (although note that we're
# transposing the data array so that each channel is a column instead of a row,
# to match what matplotlib expects when plotting 2-dimensional ``y`` against
# 1-dimensional ``x``):

x = raw_selection[1]
y = raw_selection[0].T
_ = plt.plot(x, y)

###############################################################################
# The :meth:`~mne.io.Raw.get_data` method
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you only want the data (not the corresponding array of times),
# :class:`~mne.io.Raw` objects have a :meth:`mne.io.Raw.get_data` method. Used
# with no parameters specified, it will return all data from all channels, in a
# (n_channels, n_timepoints) array:

data = raw.get_data()
print(data.shape)

###############################################################################
# If you want the array of times, :meth:`mne.io.Raw.get_data` has an optional
# ``return_times`` parameter:

data, times = raw.get_data(return_times=True)
print(data.shape)
print(times.shape)

###############################################################################
#
# Selecting channels by name
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`~mne.io.Raw` object can also be indexed with the names of
# channels instead of their index numbers. You can pass a single string to get
# just one channel, or a list of strings to select multiple channels. As with
# integer indexing, this will return a tuple of ``(data_array, times_array)``
# that can be easily plotted. Since we're plotting 2 channels this time, we'll
# add a little offset to the second channel so it's not plotted right on top of
# the first one:

two_meg_chans = raw[['MEG 0712', 'MEG 1022'], starting_sample:ending_sample]
y_offset = np.array([0, 5e-11])
x = two_meg_chans[1]
y = two_meg_chans[0].T + y_offset
_ = plt.plot(x, y)

###############################################################################
# The :meth:`~mne.io.Raw.get_data` method can also be used in this way, via its
# ``picks``, ``start``, and ``stop`` parameters:

two_meg_chans = raw.get_data(picks=['MEG 0712', 'MEG 1022'],
                             start=starting_sample, stop=ending_sample)

###############################################################################
# In addition to indexing and the :meth:`~mne.io.Raw.get_data` method (which
# both leave the original :class:`~mne.io.Raw` object unchanged), there are
# also methods :meth:`~mne.io.Raw.pick_channels` and
# :meth:`~mne.io.Raw.drop_channels` that will modify the :class:`~mne.io.Raw`
# object in place. For that reason, during interactive, exploratory analysis it
# is common to make a copy of the :class:`~mne.io.Raw` object first when using
# these methods, to avoid having to re-load the data from disk if you decide
# you don't like the results of a ``pick`` or ``drop`` operation.

raw2 = raw.copy()
print('Number of channels in raw:')
print(len(raw2.ch_names), end=' → drop two → ')
raw2.drop_channels(['EEG 037', 'EEG 059'])
print(len(raw2.ch_names), end=' → pick three → ')
raw2.pick_channels(['MEG 1811', 'EEG 017', 'EOG 061'])
print(len(raw2.ch_names))

###############################################################################
# If you want the selected channels to occur in a specific order, you can also
# use :meth:`~mne.io.Raw.reorder_channels`, which takes a list of channel names
# and returns a :class:`~mne.io.Raw` object with the selected channels in the
# given order. This can be useful for plotting (e.g., if you want a plot of
# just the EOG and a few EEG channel(s), with EOG at the top):

chans = ['EOG 061', 'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004']
eog_and_eeg = raw.copy().reorder_channels(chans)
start, stop = eog_and_eeg.time_as_index([42, 45])
y, x = eog_and_eeg[:, start:stop]
y_offset = np.linspace(0, -1e-3, 5)
_ = plt.plot(x, y.T + y_offset)

###############################################################################
# .. note::
#
#     :meth:`~mne.io.Raw.get_data` also preserves the requested channel order
#     given as its ``picks`` parameter.
#
# Notice that we saved some keystrokes by *chaining* the
# :meth:`~mne.io.Raw.copy` and the :meth:`~mne.io.Raw.pick_types` methods on
# one line; MNE-Python is designed to make chaining easy, which can help
# conserve memory by not storing lots of intermediate copies as you step
# through an analysis pipeline.
#
#
# Selecting channels by type
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To conveniently select all channels of a single type, you can use the
# :meth:`~mne.io.Raw.pick_types` method (there is no corresponding
# ``drop_types()`` method). :meth:`~mne.io.Raw.pick_types` takes boolean or
# string parameters for many different channel types (``eeg``, ``eeg``,
# ``eog``, ``fnirs``, ``chpi``, etc.); you select which channel type(s) to keep
# by passing ``True`` for that parameter. The ``meg`` parameter defaults to
# ``True``, and all others default to ``False``, so to get just the EEG and EOG
# channels, we pass ``True`` for each of those parameters and ``False`` for the
# MEG parameter:

eeg_and_eog = raw.copy().pick_types(meg=False, eeg=True, eog=True)
print(len(eeg_and_eog.ch_names))

###############################################################################
# Some of the parameters of :meth:`~mne.io.Raw.pick_types` accept string
# arguments as well as booleans. For example, the ``meg`` parameter can take
# values ``'mag'``, ``'grad'``, ``'planar1'``, or ``'planar2'`` to select only
# magnetometers, all gradiometers, or a specific type of gradiometer. See the
# docstring of :meth:`~mne.io.Raw.pick_types` for full details.
#
# Again, :meth:`~mne.io.Raw.get_data` provides similar functionality without
# modifying the underlying :class:`~mne.io.Raw` object, returning a NumPy
# array:

eeg_and_eog_data = raw.get_data(picks=['eeg', 'eog'])
print(eeg_and_eog_data.shape)

###############################################################################
# Manipulating channel names
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You may have noticed that the EEG channel names in the sample data are
# numbered rather than labelled according to a standard nomenclature such as
# the `10-20 <ten_twenty>`_ or `10-05 <ten_oh_five>`_ systems, or perhaps it
# bothers you that the channel names contain spaces. It is possible to rename
# channels using the :meth:`~mne.io.Raw.rename_channels` method, which takes a
# Python dictionary to map old names to new names. You need not rename all
# channels at once; provide only the dictionary entries for the channels you
# want to rename. Here's a frivolous example:

raw.rename_channels({'EOG 061': 'blink detector'})

###############################################################################
# .. note::
#
#     Due to limitations in the ``.fif`` file format (which MNE-Python uses to
#     save :class:`~mne.io.Raw` objects), channel names are limited to a
#     maximum of 15 characters.
#
# This next example replaces spaces in the channel names with underscores,
# using a Python `dict comprehension`_:

print(raw.ch_names[-3:])
chan_renaming_dict = {name: name.replace(' ', '_') for name in raw.ch_names}
raw.rename_channels(chan_renaming_dict)
print(raw.ch_names[-3:])

###############################################################################
# .. include:: ../../tutorial_links.inc
