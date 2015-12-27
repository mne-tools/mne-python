"""
=============================================
Compute the power spectral density of epochs
=============================================

This script shows how to compute the power spectral density (PSD)
of measurements on epochs. It also shows how to plot its spatial
distribution.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne import io
from mne.datasets import sample
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch, psd_multitaper

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

plt.ion()

# Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

tmin, tmax, event_id = -1., 1., 1
raw.info['bads'] += ['MEG 2443']  # bads

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    proj=True, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

# Let's first check out all channel types by averaging across epochs.
epochs.plot_psd(fmin=2, fmax=200)

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Now let's take a look at the spatial distributions of the psd.
epochs.plot_psd_topomap(ch_type='grad', normalize=True)

# Alternatively, you may also create PSDs from Raw objects with psd_XXX
psds_mt, freqs_mt = psd_multitaper(epochs[:5], low_bias=True, tmin=tmin,
                                   tmax=tmax, picks=[1], n_jobs=1)
f, ax = plt.subplots()
ax.plot(freqs_mt, 10*np.log10(psds_mt[:, 0, :].T))
ax.set(title='Multitaper PSD',  xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
mne.viz.tight_layout()
