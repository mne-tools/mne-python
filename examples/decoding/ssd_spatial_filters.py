# -*- coding: utf-8 -*-
"""
.. _ex-ssd-spatial-filters:

===========================================================
Compute Spectro-Spatial Decomposition (SSD) spatial filters
===========================================================

In this example, we will compute spatial filters for retaining
oscillatory brain activity and down-weighting 1/f background signals
as proposed by :footcite:`NikulinEtAl2011`.
The idea is to learn spatial filters that separate oscillatory dynamics
from surrounding non-oscillatory noise based on the covariance in the
frequency band of interest and the noise covariance based on surrounding
frequencies.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD-3-Clause

# %%


import matplotlib.pyplot as plt
import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne.decoding import SSD

# %%
# Define parameters
fname = data_path() / 'SubjectCMC.ds'

# Prepare data
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 110.).load_data()  # crop for memory purposes
raw.resample(sfreq=250)

raw.pick_types(meg=True, eeg=False, ref_meg=False)

freqs_sig = 9, 12
freqs_noise = 8, 13


ssd = SSD(info=raw.info,
          reg='oas',
          sort_by_spectral_ratio=False,  # False for purpose of example.
          filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                  l_trans_bandwidth=1, h_trans_bandwidth=1),
          filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                 l_trans_bandwidth=1, h_trans_bandwidth=1))
ssd.fit(X=raw.get_data())


# %%
# Let's investigate spatial filter with max power ratio.
# We will first inspect the topographies.
# According to Nikulin et al. 2011 this is done by either inverting the filters
# (W^{-1}) or by multiplying the noise cov with the filters Eq. (22) (C_n W)^t.
# We rely on the inversion approach here.

pattern = mne.EvokedArray(data=ssd.patterns_[:4].T,
                          info=ssd.info)
pattern.plot_topomap(units=dict(mag='A.U.'), time_format='')

# The topographies suggest that we picked up a parietal alpha generator.

# Transform
ssd_sources = ssd.transform(X=raw.get_data())

# Get psd of SSD-filtered signals.
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)

# Get spec_ratio information (already sorted).
# Note that this is not necessary if sort_by_spectral_ratio=True (default).
spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

# Plot spectral ratio (see Eq. 24 in Nikulin 2011).
fig, ax = plt.subplots(1)
ax.plot(spec_ratio, color='black')
ax.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
ax.legend()
ax.axhline(1, linestyle='--')

# We can see that the initial sorting based on the eigenvalues
# was already quite good. However, when using few components only
# the sorting might make a difference.

# %%
# Let's also look at the power spectrum of that source and compare it to
# to the power spectrum of the source with lowest SNR.

below50 = freqs < 50
# for highlighting the freq. band of interest
bandfilt = (freqs_sig[0] <= freqs) & (freqs <= freqs_sig[1])
fig, ax = plt.subplots(1)
ax.loglog(freqs[below50], psd[0, below50], label='max SNR')
ax.loglog(freqs[below50], psd[-1, below50], label='min SNR')
ax.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
ax.fill_between(freqs[bandfilt], 0, 10000, color='green', alpha=0.15)
ax.set_xlabel('log(frequency)')
ax.set_ylabel('log(power)')
ax.legend()

# We can clearly see that the selected component enjoys an SNR that is
# way above the average power spectrum.

# %%
# Epoched data
# ------------
# Although we suggest to use this method before epoching, there might be some
# situations in which data can only be treated by chunks.

# Build epochs as sliding windows over the continuous raw file.
events = mne.make_fixed_length_events(raw, id=1, duration=5.0, overlap=0.0)

# Epoch length is 5 seconds.
epochs = Epochs(raw, events, tmin=0., tmax=5,
                baseline=None, preload=True)

ssd_epochs = SSD(info=epochs.info,
                 reg='oas',
                 filt_params_signal=dict(l_freq=freqs_sig[0],
                                         h_freq=freqs_sig[1],
                                         l_trans_bandwidth=1,
                                         h_trans_bandwidth=1),
                 filt_params_noise=dict(l_freq=freqs_noise[0],
                                        h_freq=freqs_noise[1],
                                        l_trans_bandwidth=1,
                                        h_trans_bandwidth=1))
ssd_epochs.fit(X=epochs.get_data())

# Plot topographies.
pattern_epochs = mne.EvokedArray(data=ssd_epochs.patterns_[:4].T,
                                 info=ssd_epochs.info)
pattern_epochs.plot_topomap(units=dict(mag='A.U.'), time_format='')
# %%
# References
# ----------
#
# .. footbibliography::
