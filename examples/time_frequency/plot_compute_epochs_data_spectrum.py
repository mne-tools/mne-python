"""
==================================================
Compute the power spectral density of epochs data
==================================================

This script shows how to compute the power spectral density (PSD)
of measurements on a epochs dataset. It also show the effect of applying SSP
to the data to reduce ECG and EOG artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io, read_selection
from mne.time_frequency.psd import compute_epochs_psd
# from mne.datasets import somato

###############################################################################
# Load real somatosensory sample data.
# data_path = somato.data_path()
data_path = '/home/ybekhti/work/src/mne-python/examples/MNE-somato-data'
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.

# Setup for reading the raw data
raw = io.Raw(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                    reject=dict(grad=4000e-13), preload=True)

# Let's first check out all channel types
epochs.plot_psds(area_mode='range')

# Now let's focus on a smaller subset:
# Pick MEG magnetometers in the Left-temporal region
selection = read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads', selection=selection)

# Let's just look at the first few channels for demonstration purposes
picks = picks[:4]

plt.figure()
ax = plt.axes()
epochs.plot_psds(fmin=fmin, fmax=fmax, n_fft=n_fft, ax=ax,
                 n_jobs=1, color=(0, 0, 1),  picks=picks)

ax.set_title('Four left-temporal magnetometers')
plt.show()
