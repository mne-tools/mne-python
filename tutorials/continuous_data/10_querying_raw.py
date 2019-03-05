# -*- coding: utf-8 -*-
"""
.. _querying-raw-tutorial:

Querying the :class:`~mne.io.Raw` object
========================================

This tutorial covers how to query the :class:`mne.io.Raw` object once it's
loaded into memory. As before, we'll start by importing the Python modules we
need:
"""

import os
import numpy as np
import mne

###############################################################################
# We'll start by repeating the loading steps from the last tutorial; if you
# still have the Python interpreter open after the last tutorial you can skip
# this block:

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# As shown in the tutorial on `loading raw data <loading-raw-tutorial>`_,
# printing the :class:`~mne.io.Raw` object displays some basic information like
# the total number of channels, the number of time points at which the data
# were sampled, total duration, and the approximate size in memory. Much more
# information is available through the various *attributes* and *methods* of
# the :class:`~mne.io.Raw` class.
#
# .. note::
#
#     ***Attributes*** are usually static properties of Python objects — things
#     that are pre-computed and stored as part of the object's representation
#     in memory. Attributes are accessed with the ``.`` operator and do not
#     require parentheses after the attribute name (example: ``raw.ch_names``).
#
#     ***Methods*** are like specialized functions attached to the object.
#     Usually they require additional information from the user and/or require
#     some computation to yield an answer. Methods always have parentheses
#     after their names; additional arguments (if any) go inside those
#     parentheses (examples: ``raw.estimate_rank()``,
#     ``raw.drop_channels(['EEG 030', 'MEG 2242'])``).
#
# Some useful attributes of :class:`~mne.io.Raw` objects include a list of the
# channel names (:attr:`~mne.io.Raw.ch_names`), an array of the sample times in
# seconds (:attr:`~mne.io.Raw.times`, in seconds), and the total number of
# samples (:attr:`~mne.io.Raw.n_times`).

n_time_samps = raw.n_times
time_secs = raw.times
chan_names = raw.ch_names
n_chan = len(chan_names)  # note: there is no raw.n_channels attribute
print(f'the sample data file has {n_time_samps} time samples and {n_chan} '
      f'channels.\nThe last time sample is at {time_secs[-1]} seconds.\nThe '
      f'first few channel names are {", ".join(chan_names[:3])}.')

###############################################################################
# .. note::
#
#     Although the :class:`~mne.io.Raw` object underlyingly stores data samples
#     in a NumPy array of shape (n_channels, n_timepoints), the
#     :class:`~mne.io.Raw` object itself behaves differently from NumPy arrays
#     with respect to the ``len`` function.  ``len(raw)`` will return the
#     number of timepoints (length along data axis 1), not the number of
#     channels (length along data axis 0).
#
# There is also quite a lot of information stored in the ``raw.info``
# attribute, which stores an :class:`~mne.Info` object that is similar to a
# Python dictionary (in that it has fields accessed via named keys):

print(raw.info['bads'])         # channels marked "bad" during acquisition
print(raw.info['sfreq'])        # sampling frequency
print(raw.info['description'])  # miscellaneous acquisition info

###############################################################################
# Like Python dictionaries, ``raw.info`` has a ``.keys()`` method that shows
# all the available field names; unlike Python dictionaries, printing
# ``raw.info`` will print a nicely-formatted glimpse of each field's data:

print(raw.info)

###############################################################################
# .. note::
#
#     Most of the fields of ``raw.info`` reflect metadata recorded at
#     acquisition time, and should not be changed by the user. There are a few
#     exceptions (such as ``raw.info['bads']`` and ``raw.info['projs']``), but
#     in most cases there are dedicated MNE-Python functions or methods to
#     update the :class:`~mne.Info` object safely (such as
#     :meth:`~mne.io.Raw.add_proj` to update ``raw.info['projs']``).
#
# .. _`time-as-index`:
#
# Time, sample number, and sample index
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One method of :class:`~mne.io.Raw` objects that is frequently useful is
# :meth:`~mne.io.Raw.time_as_index`, which converts a time (in seconds) into
# the integer index of the sample occurring closest to that time. The method
# can also take a list or array of times, and will return an array of indices:

print(raw.time_as_index(20))
print(raw.time_as_index([20, 30, 40]))

###############################################################################
# It is important to remember that there may not be a data sample at *exactly*
# the time requested, so the number of samples between ``time = 1`` second and
# ``time = 2`` seconds may be different than the number of samples between
# ``time = 2`` and ``time = 3``:

print(np.diff(raw.time_as_index([1, 2, 3])))

###############################################################################
# It is also important to distinguish *sample number* from *sample index*. Some
# MEG systems (in particular, Vectorview) start counting samples when the
# acquisition system is initiated, not when the data begin to be written to
# disk. In such cases, *sample number* (according to the acquisition hardware)
# and *sample index* (the index of where that sample is stored in the
# :class:`~mne.io.Raw` object) will not match. MNE-Python handles this with an
# attribute :attr:`~mne.io.Raw.first_samp` that gives the acquisition system's
# sample number of the sample stored at index ``0`` of the :class:`~mne.io.Raw`
# object. Here, we see that the first data sample written to disk was acquired
# about 43 seconds after the acquisition system was initialized:

print(raw.first_samp / raw.info['sfreq'])

###############################################################################
# For these and other reasons, it is important to carefully read the
# documentation for MNE-Python functions or methods that have parameters such
# as ``start``, ``stop``, ``tmin`` or ``tmax`` so you can be sure to provide a
# time, sample number, or index as appropriate. In most cases MNE-Python will
# automatically check and adjust for ``first_samp`` when computing times from
# sample numbers, but it is prudent to use care when working with sample
# numbers nonetheless.
