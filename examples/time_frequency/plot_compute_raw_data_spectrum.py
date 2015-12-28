"""
==================================================
Compute the power spectral density of raw data
==================================================

This script shows how to compute the power spectral density (PSD)
of measurements on a raw dataset. It also show the effect of applying SSP
to the data to reduce ECG and EOG artifacts.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io, read_proj, read_selection
from mne.datasets import sample
from mne.time_frequency import psd_multitaper

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
proj_fname = data_path + '/MEG/sample/sample_audvis_eog_proj.fif'

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# Add SSP projection vectors to reduce EOG and ECG artifacts
projs = read_proj(proj_fname)
raw.add_proj(projs, remove_existing=True)

tmin, tmax = 0, 60  # use the first 60s of data
fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

# Let's first check out all channel types
raw.plot_psd(area_mode='range', tmax=10.0)

# Now let's focus on a smaller subset:
# Pick MEG magnetometers in the Left-temporal region
selection = read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads', selection=selection)

# Let's just look at the first few channels for demonstration purposes
picks = picks[:4]

plt.figure()
ax = plt.axes()
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=False, ax=ax, color=(0, 0, 1),  picks=picks)

# And now do the same with SSP applied
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=True, ax=ax, color=(0, 1, 0), picks=picks)

# And now do the same with SSP + notch filtering
raw.notch_filter(np.arange(60, 241, 60), picks=picks, n_jobs=1)
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=True, ax=ax, color=(1, 0, 0), picks=picks)

ax.set_title('Four left-temporal magnetometers')
plt.legend(['Without SSP', 'With SSP', 'SSP + Notch'])

# Alternatively, you may also create PSDs from Raw objects with psd_XXX
f, ax = plt.subplots()
psds_mt, freqs_mt = psd_multitaper(raw, low_bias=True, tmin=tmin, tmax=tmax,
                                   fmin=fmin, fmax=fmax, picks=picks, n_jobs=1)
ax.plot(freqs_mt, 10 * np.log10(psds_mt).T)
ax.set(title='Multitaper PSD', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
mne.viz.tight_layout()
