"""
.. _tut_epochs_objects:

The :class:`Epochs <mne.Epochs>` data structure: epoched data
=============================================================
"""

from __future__ import print_function

import mne
import os.path as op
import numpy as np
from matplotlib import pyplot as plt

###############################################################################
# :class:`Epochs <mne.Epochs>` objects are a way of representing continuous
# data as a collection of time-locked trials, stored in an array of
# `shape(n_events, n_channels, n_times)`. They are useful for many statistical
# methods in neuroscience, and make it easy to quickly overview what occurs
# during a trial.
#
# :class:`Epochs <mne.Epochs>` objects can be created in three ways:
#  1. From a :class:`Raw <mne.io.RawFIF>` object, along with event times
#  2. From an :class:`Epochs <mne.Epochs>` object that has been saved as a
#     `.fif` file
#  3. From scratch using :class:`EpochsArray <mne.EpochsArray>`. See
#     :ref:`tut_creating_data_structures`

data_path = mne.datasets.sample.data_path()
# Load a dataset that contains events
raw = mne.io.read_raw_fif(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'))

# If your raw object has a stim channel, you can construct an event array
# easily
events = mne.find_events(raw, stim_channel='STI 014')

# Show the number of events (number of rows)
print('Number of events:', len(events))

# Show all unique event codes (3rd column)
print('Unique event codes:', np.unique(events[:, 2]))

# Specify event codes of interest with descriptive labels.
# This dataset also has visual left (3) and right (4) events, but
# to save time and memory we'll just look at the auditory conditions
# for now.
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}

###############################################################################
# Now, we can create an :class:`mne.Epochs` object with the events we've
# extracted. Note that epochs constructed in this manner will not have their
# data available until explicitly read into memory, which you can do with
# :func:`get_data <mne.Epochs.get_data>`. Alternatively, you can use
# `preload=True`.
#
# Expose the raw data as epochs, cut from -0.1 s to 1.0 s relative to the event
# onsets
epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True)
print(epochs)

###############################################################################
# Epochs behave similarly to :class:`mne.io.Raw` objects. They have an
# :class:`info <mne.Info>` attribute that has all of the same
# information, as well as a number of attributes unique to the events contained
# within the object.

print(epochs.events[:3], epochs.event_id, sep='\n\n')

###############################################################################
# You can select subsets of epochs by indexing the :class:`Epochs <mne.Epochs>`
# object directly. Alternatively, if you have epoch names specified in
# `event_id` then you may index with strings instead.

print(epochs[1:5])
print(epochs['Auditory/Right'])

###############################################################################
# It is also possible to iterate through :class:`Epochs <mne.Epochs>` objects
# in this way. Note that behavior is different if you iterate on `Epochs`
# directly rather than indexing:

# These will be epochs objects
for i in range(3):
    print(epochs[i])

# These will be arrays
for ep in epochs[:2]:
    print(ep)

###############################################################################
# You can manually remove epochs from the Epochs object by using
# :func:`epochs.drop(idx) <mne.Epochs.drop>`, or by using rejection or flat
# thresholds with :func:`epochs.drop_bad(reject, flat) <mne.Epochs.drop_bad>`.
# You can also inspect the reason why epochs were dropped by looking at the
# list stored in ``epochs.drop_log`` or plot them with
# :func:`epochs.plot_drop_log() <mne.Epochs.plot_drop_log>`. The indices
# from the original set of events are stored in ``epochs.selection``.

epochs.drop([0], reason='User reason')
epochs.drop_bad(reject=dict(grad=2500e-13, mag=4e-12, eog=200e-6), flat=None)
print(epochs.drop_log)
epochs.plot_drop_log()
print('Selection from original events:\n%s' % epochs.selection)
print('Removed events (from numpy setdiff1d):\n%s'
      % (np.setdiff1d(np.arange(len(events)), epochs.selection).tolist(),))
print('Removed events (from list comprehension -- should match!):\n%s'
      % ([li for li, log in enumerate(epochs.drop_log) if len(log) > 0]))

###############################################################################
# If you wish to save the epochs as a file, you can do it with
# :func:`mne.Epochs.save`. To conform to MNE naming conventions, the
# epochs file names should end with '-epo.fif'.
epochs_fname = op.join(data_path, 'MEG', 'sample', 'sample-epo.fif')
epochs.save(epochs_fname)

###############################################################################
# Later on you can read the epochs with :func:`mne.read_epochs`. For reading
# EEGLAB epochs files see :func:`mne.read_epochs_eeglab`. We can also use
# ``preload=False`` to save memory, loading the epochs from disk on demand.
epochs = mne.read_epochs(epochs_fname, preload=False)

###############################################################################
# If you wish to look at the average across trial types, then you may do so,
# creating an :class:`Evoked <mne.Evoked>` object in the process. Instances
# of `Evoked` are usually created by calling :func:`mne.Epochs.average`. For
# creating `Evoked` from other data structures see :class:`mne.EvokedArray` and
# :ref:`tut_creating_data_structures`.

ev_left = epochs['Auditory/Left'].average()
ev_right = epochs['Auditory/Right'].average()

f, axs = plt.subplots(3, 2, figsize=(10, 5))
_ = f.suptitle('Left / Right auditory', fontsize=20)
_ = ev_left.plot(axes=axs[:, 0], show=False)
_ = ev_right.plot(axes=axs[:, 1], show=False)
plt.tight_layout()

###############################################################################
# To export and manipulate Epochs using Pandas see :ref:`tut_io_export_pandas`.
