"""
================================
Temporal whitening with AR model
================================

This script shows how to fit an AR model to data and use it
to temporally whiten the signals.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
from scipy import signal
import pylab as pl

import mne
from mne.time_frequency import ar_raw
from mne.datasets import sample
data_path = sample.data_path('..')

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
proj_fname = data_path + '/MEG/sample/sample_audvis_ecg_proj.fif'

raw = mne.fiff.Raw(raw_fname)
proj = mne.read_proj(proj_fname)
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

# Set up pick list: Gradiometers - bad channels
picks = mne.fiff.pick_types(raw.info, meg='grad', exclude=raw.info['bads'])

order = 5  # define model order
picks = picks[:5]

# Estimate AR models on raw data
coefs = ar_raw(raw, order=order, picks=picks, tmin=60, tmax=180)
mean_coefs = np.mean(coefs, axis=0)  # mean model accross channels

filt = np.r_[1, -mean_coefs]  # filter coefficient
d, times = raw[0, 1e4:2e4]  # look at one channel from now on
d = d.ravel()  # make flat vector
innovation = signal.convolve(d, filt, 'valid')
d_ = signal.lfilter([1], filt, innovation)  # regenerate the signal
d_ = np.r_[d_[0] * np.ones(order), d_]  # dummy samples to keep signal length

###############################################################################
# Plot the different time series and PSDs
pl.close('all')
pl.figure()
pl.plot(d[:100], label='signal')
pl.plot(d_[:100], label='regenerated signal')
pl.legend()

pl.figure()
pl.psd(d, Fs=raw.info['sfreq'], NFFT=2048)
pl.psd(innovation, Fs=raw.info['sfreq'], NFFT=2048)
pl.psd(d_, Fs=raw.info['sfreq'], NFFT=2048, linestyle='--')
pl.legend(('Signal', 'Innovation', 'Regenerated signal'))
pl.show()
