"""
==================================================
Compute the spectrum of raw data using multi-taper
==================================================

This script shows how to compute the power spectral density (PSD)
of measurements on a raw dataset.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

from mne import fiff
from mne.time_frequency import compute_raw_psd
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                                                stim=False, exclude=exclude)

tmin, tmax = 0, 60  # use the first 60s of data
fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz
NFFT = 2048 # the FFT size (NFFT). Ideally a power of 2
psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                              fmin=fmin, fmax=fmax, NFFT=NFFT, n_jobs=1)

###############################################################################
# Compute mean and standard deviation accross channels and then plot
psd_mean = np.mean(psds, axis=0)
psd_std = np.std(psds, axis=0)

hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)

import pylab as pl
pl.close('all')
pl.plot(freqs, psd_mean)
pl.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1], color=(1, 0, 0, .3),
                alpha=0.5)
pl.xlabel('Freq (Hz)')
pl.ylabel('PSD')
pl.show()
