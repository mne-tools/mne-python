"""
=============================================
Compute the power spectral density of epochs
=============================================

This script shows how to compute the power spectral density (PSD)
of measurements on epochs. It also show how to plot its spatial
distribution.
"""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.time_frequency import compute_epochs_psd
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

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
					proj=True, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

n_fft = 256  # the FFT size. Ideally a power of 2

# Let's first check out all channel types by averaging across epochs.
epochs.plot_psds(fmin=2, fmax=200, n_fft=n_fft, n_jobs=2)

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, include=include, exclude='bads')

# Now let's take a look at the spatial distributions of the psd.
epochs.plot_psds_topomap(picks=picks, n_fft=n_fft, n_jobs=2)
