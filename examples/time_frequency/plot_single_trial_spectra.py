"""
======================================
Investigate Single Trial Power Spectra
======================================

In this example we will look at single trial spectra and then
compute average spectra to identify channels and
frequencies of interest for subsequent TFR analyses.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.time_frequency import compute_epochs_psd

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

tmin, tmax, event_id = -1., 1., 1
include = []
raw.info['bads'] += ['MEG 2443']  # bads

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=(None, 0),
                    reject=dict(grad=4000e-13, eog=150e-6))

n_fft = 256  # the FFT size. Ideally a power of 2

# Let's first check out all channel types by averaging across epochs.
epochs.plot_psds(plot_kind=1, fmin=2, fmax=200, n_fft=n_fft, n_jobs=2)

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, include=include, exclude='bads')

# A second way to plot the psds and look at the psds.
epochs.plot_psds(plot_kind=2, fmin=2, fmax=200, n_fft=n_fft, picks=picks, n_jobs=2)

# In the second image we clearly observe certain channel groups exposing
# stronger power than others. Second, in comparison to the single
# trial image we can see the frequency extent slightly growing for these
# channels which might indicate oscillatory responses.
# The ``plot_time_frequency.py`` example investigates one of the channels
# around index 140.
# Finally, also note the power line artifacts across all channels.
