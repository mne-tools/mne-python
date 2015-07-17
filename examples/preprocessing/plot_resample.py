"""
===============
Resampling data
===============

Resampling can save memory and computation time. This example shows some
approaches to resample data from 600 Hz to 100 Hz.
"""
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)
#
from __future__ import print_function

import mne
from mne.io import Raw
from mne.datasets import sample

print(__doc__)


###############################################################################
# The first approach is to resample the raw, continous data.

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
raw = Raw(raw_fname, preload=True)
print('Original sampling rate:', raw.info['sfreq'], 'Hz')

# Resample to 300 Hz 
raw_resampled = raw.resample(300, copy=True)
print('New sampling rate:', raw.info['sfreq'], 'Hz')

###############################################################################
# Because resampling also affects the stim channels, some trigger onsets might
# be lost. While MNE attempts to downsample the stim channels in an intelligent
# manner to avoid this, the recommended approach is to find events on the
# original data before downsampling.
print('Number of events before resampling:', len(mne.find_events(raw)))

# Resample to 100 Hz (generates warning)
raw_resampled = raw.resample(100, copy=True)
print('Number of events after resampling:',
      len(mne.find_events(raw_resampled)))

# To avoid losing events, jointly resample the data and event matrix
events = mne.find_events(raw)
raw_resampled, events_resampled = raw.resample(100, events=events, copy=True)
print('Number of events after resampling:', len(events_resampled))

###############################################################################
# Epoched data can be resampled as well.
epochs = mne.Epochs(raw, events, [1, 2, 3, 4], tmin=-0.1, tmax=0.8,
                    preload=True)
epochs_resampled = epochs.resample(100, copy=True)
