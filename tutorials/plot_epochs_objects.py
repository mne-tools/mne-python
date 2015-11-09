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
#  3. From scratch using :class:`EpochsArray <mne.EpochsArray>`

# Load a dataset that contains events
raw = mne.io.RawFIF(
    op.join(mne.datasets.sample.data_path(), 'MEG', 'sample',
            'sample_audvis_raw.fif'))

# If your raw object has a stim channel, you can construct an event array
# easily
events = mne.find_events(raw, stim_channel='STI 014')

# Show the number of events (number of rows)
print('Number of events:', len(events))

# Show all unique event codes (3rd column)
print('Unique event codes:', np.unique(events[:, 2]))

# Specify event codes of interest with descriptive labels
event_id = dict(left=1, right=2)

###############################################################################
# Now, we can create an :class:`mne.Epochs` object with the events we've
# extracted. Note that epochs constructed in this manner will not have their
# data available until explicitly read into memory, which you can do with
# :func:`get_data <mne.Epochs.get_data>`. Alternatively, you can use
# `preload=True`.
#
# Note that there are many options available when loading an
# :class:`mne.Epochs` object.  For more detailed information, see (**LINK TO
# EPOCHS LOADING TUTORIAL**)

# Expose the raw data as epochs, cut from -0.1 s to 1.0 s relative to the event
# onsets
epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True)
print(epochs)

###############################################################################
# Epochs behave similarly to :class:`mne.io.Raw` objects. They have an
# :class:`info <mne.io.meas_info.Info>` attribute that has all of the same
# information, as well as a number of attributes unique to the events contained
# within the object.

print(epochs.events[:3], epochs.event_id, sep='\n\n')

###############################################################################
# You can select subsets of epochs by indexing the :class:`Epochs <mne.Epochs>`
# object directly. Alternatively, if you have epoch names specified in
# `event_id` then you may index with strings instead.

print(epochs[1:5])
print(epochs['right'])

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
# If you wish to look at the average across trial types, then you may do so,
# creating an `Evoked` object in the process.

ev_left = epochs['left'].average()
ev_right = epochs['right'].average()

f, axs = plt.subplots(3, 2, figsize=(10, 5))
_ = f.suptitle('Left / Right', fontsize=20)
_ = ev_left.plot(axes=axs[:, 0], show=False)
_ = ev_right.plot(axes=axs[:, 1], show=False)
plt.tight_layout()
