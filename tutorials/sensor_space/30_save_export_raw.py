# -*- coding: utf-8 -*-
"""
.. _save-export-raw-tutorial:

Exporting and saving data from :class:`~mne.io.Raw` objects
===========================================================

This tutorial covers how to save :class:`~mne.io.Raw` objects, and how to
extract data from :class:`~mne.io.Raw` objects and save it as a NumPy array or
Pandas DataFrame. As always we'll start by importing the modules we need, and
loading some example data:
"""

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

###############################################################################
# :class:`~mne.io.Raw` objects have a built-in :meth:`~mne.io.Raw.save` method,
# which can be used to write partially processed :class:`~mne.io.Raw` objects
# to disk as a ``.fif`` file, such that it can be re-loaded later with its
# various attributes intact (but see :ref:`precision` for an important note
# about numerical precision when saving). There are a few other ways to export
# just the sensor data from a :class:`~mne.io.Raw` object. One is to use
# indexing or the :meth:`~mne.io.Raw.get_data` method to extract the data, and
# use NumPy to save the data array:

data = raw.get_data()  # or data, times = raw[:]
np.save(file='my_data.npy', arr=data)

###############################################################################
# It is also possible to export the data to a Pandas DataFrame object, and use
# the saving methods that Pandas affords. :meth:`~mne.io.Raw.to_data_frame` is
# similar to :meth:`~mne.io.Raw.get_data` in that it has a ``picks`` parameter
# for restricting which channels are exported, and ``start`` and ``stop``
# parameters for restricting the time domain. Note that, by default, times will
# be converted to milliseconds, rounded to the nearest millisecond, and used as
# the DataFrame index; see the ``scaling_time`` parameter in the documentation
# of :meth:`~mne.io.Raw.to_data_frame` for more details.

sampling_frequency = raw.info['sfreq']
starting_sample = int(10 * sampling_frequency)
ending_sample = int(13 * sampling_frequency)
df = raw.to_data_frame(picks=['eeg'], start=starting_sample,
                       stop=ending_sample)
# then save using df.to_csv(...), df.to_hdf(...), etc
print(df.head())

###############################################################################
# .. note::
#     When exporting data as a NumPy array or Pandas DataFrame, be sure to
#     properly account for the :ref:`unit of representation <units>` in your
#     subsequent analyses.
