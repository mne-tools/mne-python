"""
==================================================
Compute the power spectral density of raw data
==================================================

This script shows how to compute the power spectral density (PSD)
of measurements on a raw dataset. It also show the effect of applying SSP
to the data to reduce ECG and EOG artifacts.
"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
# License: BSD (3-clause)

print __doc__

import numpy as np

from mne import fiff, read_proj, read_selection
from mne.time_frequency import compute_raw_psd
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
proj_fname = data_path + '/MEG/sample/sample_audvis_eog_proj.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# Add SSP projection vectors to reduce EOG and ECG artifacts
projs = read_proj(proj_fname)
raw.add_proj(projs, remove_existing=True)

# Pick MEG magnetometers in the Left-temporal region
selection = read_selection('Left-temporal')
picks = fiff.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                        stim=False, exclude=exclude, selection=selection)

tmin, tmax = 0, 60  # use the first 60s of data
fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
NFFT = 2048  # the FFT size (NFFT). Ideally a power of 2
psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                              fmin=fmin, fmax=fmax, NFFT=NFFT, n_jobs=1,
                              plot=False, proj=False)

# And now do the same with SSP applied
psds_ssp, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                  fmin=fmin, fmax=fmax, NFFT=NFFT, n_jobs=1,
                                  plot=False, proj=True)

# Convert PSDs to dB
psds = 10 * np.log10(psds)
psds_ssp = 10 * np.log10(psds_ssp)

###############################################################################
# Compute mean and standard deviation accross channels and then plot
def plot_psds(freqs, psds, fill_color):
    psd_mean = np.mean(psds, axis=0)
    psd_std = np.std(psds, axis=0)
    hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)

    pl.plot(freqs, psd_mean)
    pl.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1], color=fill_color,
                    alpha=0.5)

import pylab as pl
pl.figure()
plot_psds(freqs, psds, (0, 0, 1, .3))
plot_psds(freqs, psds_ssp, (0, 1, 0, .3))
pl.xlabel('Freq (Hz)')
pl.ylabel('Power Spectral Density (dB/Hz)')
pl.legend(['Without SSP', 'With SSP'])
pl.show()

