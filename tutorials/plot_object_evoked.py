"""
.. _tut_evoked_objects:

The :class:`Evoked <mne.Evoked>` data structure: evoked/averaged data
=====================================================================
"""
import os.path as op

import mne

###############################################################################
# The :class:`Evoked <mne.Evoked>` data structure is mainly used for storing
# averaged data over trials. In MNE the evoked objects are created by averaging
# epochs data with :func:`mne.Epochs.average`. Here we read the evoked dataset
# from a file.
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
evokeds = mne.read_evokeds(fname, baseline=(None, 0), proj=True)
print(evokeds)

###############################################################################
# Notice that the reader function returned a list of evoked instances. This is
# because you can store multiple categories into a single file. Here we have
# categories of
# ``['Left Auditory', 'Right Auditory', 'Left Visual', 'Right Visual']``.
# We can also use ``condition`` parameter to read in only one category.
evoked = mne.read_evokeds(fname, condition='Left Auditory')
evoked.apply_baseline((None, 0)).apply_proj()
print(evoked)

###############################################################################
# If you're gone through the tutorials of raw and epochs datasets, you're
# probably already familiar with the :class:`Info <mne.Info>` attribute.
# There is nothing new or special with the ``evoked.info``. All the relevant
# info is still there.
print(evoked.info)
print(evoked.times)

###############################################################################
# The evoked data structure also contains some new attributes easily
# accessible:
print(evoked.nave)  # Number of averaged epochs.
print(evoked.first)  # First time sample.
print(evoked.last)  # Last time sample.
print(evoked.comment)  # Comment on dataset. Usually the condition.
print(evoked.kind)  # Type of data, either average or standard_error.

###############################################################################
# The data is also easily accessible. Since the evoked data arrays are usually
# much smaller than raw or epochs datasets, they are preloaded into the memory
# when the evoked object is constructed. You can access the data as a numpy
# array.
data = evoked.data
print(data.shape)

###############################################################################
# The data is arranged in an array of shape `(n_channels, n_times)`. Notice
# that unlike epochs, evoked object does not support indexing. This means that
# to access the data of a specific channel you must use the data array
# directly.
print('Data from channel {0}:'.format(evoked.ch_names[10]))
print(data[10])

###############################################################################
# If you want to import evoked data from some other system and you have it in a
# numpy array you can use :class:`mne.EvokedArray` for that. All you need is
# the data and some info about the evoked data. For more information, see
# :ref:`tut_creating_data_structures`.
evoked = mne.EvokedArray(data, evoked.info, tmin=evoked.times[0])
evoked.plot()

###############################################################################
# To write an evoked dataset to a file, use the :meth:`mne.Evoked.save` method.
# To save multiple categories to a single file, see :func:`mne.write_evokeds`.
