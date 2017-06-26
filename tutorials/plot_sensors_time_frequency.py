"""
.. _tut_sensors_time_frequency:

=============================================
Frequency and time-frequency sensors analysis
=============================================

The objective is to show you how to explore the spectral content
of your data (frequency and time-frequency). Here we'll work on Epochs.

We will use the somatosensory dataset that contains so
called event related synchronizations (ERS) / desynchronizations (ERD) in
the beta band.
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato

###############################################################################
# Set parameters
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

# Construct Epochs
event_id, tmin, tmax = 1, -1., 3.
baseline = (None, 0)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
                    preload=True)

epochs.resample(150., npad='auto')  # resample to reduce computation time

###############################################################################
# Frequency analysis
# ------------------
#
# We start by exploring the frequency content of our epochs.
# Let's first check out average spectrum by taking the mean across epochs:
epochs.plot_psd(fmin=2., fmax=40.)

###############################################################################
# Now let's take a look at the spatial distributions of the PSD in various
# frequency bands:
epochs.plot_psd_topomap(ch_type='grad', normalize=True)

###############################################################################
# Alternatively, you can also create PSDs from Raw or Epochs objects with
# functions that start with ``psd_`` such as:
# :func:`mne.time_frequency.psd_multitaper` and
# :func:`mne.time_frequency.psd_welch`.
# These functions return arrays of shape n_channels x n_frequencies for Raw and
# n_epochs x n_channels x n_frequencies for Epochs. The second output is always
# a vector of frequency bins.

f, ax = plt.subplots()
psds, freqs = psd_multitaper(epochs, fmin=2, fmax=40, n_jobs=1)
psds = 10 * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
psds_std = psds.mean(0).std(0)

ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=.15, lw=0)
ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()

###############################################################################
# To see the difference between :func:`psd_multitaper` and :func:`psd_welch`
# - we are going to compare PSDs obtained with each function.
ch_index = epochs.ch_names.index('MEG 2333')

psds_m, freqs_m = psd_multitaper(epochs, picks=[ch_index],
                                 fmin=2, fmax=17, n_jobs=1)
psds_w, freqs_w = psd_welch(epochs, n_fft=epochs.info['sfreq'] * 2,
                            picks=[ch_index], fmin=2, fmax=17, n_jobs=1)

# drop channel dimension and average epochs
psd_m = psds_m.squeeze(axis=1).mean(axis=0)
psd_w = psds_w.squeeze(axis=1).mean(axis=0)

# normalize multitaper and welch to put them on the same scale
psd_m /= psd_m.sum() / len(freqs_m)
psd_w /= psd_w.sum() / len(freqs_w)

# plot
f, ax = plt.subplots()
ax.plot(freqs_m, psd_m, label='multitaper', lw=2)
ax.plot(freqs_w, psd_w, label='welch', lw=2)
ax.set(title='Multitaper vs Welch PSD (gradiometers)', xlabel='Frequency',
       ylabel='Power Spectral Density')
ax.legend()
plt.show()

###############################################################################
# We can observe that multitaper estimation yields smoother spectrum - the
# amount of smoothing can be controlled with ``bandwidth`` argument:
bandwidths = np.arange(1, 7)
colors = plt.cm.viridis(bandwidths / 7)

fig, ax = plt.subplots()

for bnd_idx, bnd in enumerate(bandwidths):
    psd, freqs = psd_multitaper(epochs, picks=[ch_index],
                                fmin=2, fmax=17, bandwidth=bnd)
    ax.plot(freqs, psd[:, 0].mean(axis=0), label='bandwidth={}'.format(bnd),
            color=colors[bnd_idx], lw=2)

ax.set(title='Multitaper with different bandwidth', xlabel='Frequency',
       ylabel='Power Spectral Density')
ax.legend(loc='best')
plt.show()

###############################################################################
# While in multitaper the averaging is done across independent realizations of
# the signal (using slepian tapers), welch method averages across time
# segments of the signal (often overlapping). Instead of averaging the windows
# you can choose a different reduction by specifying ``combine``:
welch_args = dict(picks=[ch_index], n_fft=epochs.info['sfreq'],
                  n_overlap=int(epochs.info['sfreq'] / 2), fmin=2, fmax=25)
psds_w, freqs = psd_welch(epochs, combine='mean', **welch_args)
psds_w_trimmed, _ = psd_welch(epochs, combine=0.15, **welch_args)
psds_w_median, _ = psd_welch(epochs, combine='median', **welch_args)

psd_w = psds_w[:, 0].mean(axis=0)
psd_w_trim = psds_w_trimmed[:, 0].mean(axis=0)
psd_w_med = psds_w_median[:, 0].mean(axis=0)

# plot
f, ax = plt.subplots()
ax.plot(freqs, psd_w, color='cornflowerblue', label='welch, mean')
ax.plot(freqs, psd_w_trim, color='seagreen', label='welch, trimmed mean')
ax.plot(freqs, psd_w_med, color='crimson', label='welch, median')

ax.set(title='Welch with mean and median combine', xlabel='Frequency',
       ylabel='Power Spectral Density')
ax.legend()
plt.show()

###############################################################################
# The reduction in power that can be seen in the figure above is due to the
# fact that values for power spectral density follow a positive skewed
# gamma-like distribution. Lets take a look at this distribution. First we will
# use ``combine=None`` to get all the welch windows without averaging. Notice
# the dimensions of the output.
psds_windows, freqs = psd_welch(epochs, combine=None, **welch_args)
n_epochs, n_windows, n_channels, n_freqs = psds_windows.shape
print('dimensions of returned PSDs: n_epochs, n_windows, n_channels, n_freqs')
print('PSDs shape in each dimension: ', end='')
print(n_epochs, n_windows, n_channels, n_freqs, sep=', ')

###############################################################################
# Now we'll pick power only at 10 Hz and unroll epochs and welch windows into
# one vector whose histogram we plot.
alpha = np.where(freqs == 10)[0][0]
counts, bins, patches = plt.hist(
    psds_windows[..., alpha].reshape(n_epochs * n_windows), bins=50)
plt.show()

###############################################################################
# Time-frequency analysis: power and intertrial coherence
# -------------------------------------------------------
#
# We now compute time-frequency representations (TFRs) from our Epochs.
# We'll look at power and intertrial coherence (ITC).
#
# To this we'll use the function :func:`mne.time_frequency.tfr_morlet`
# but you can also use :func:`mne.time_frequency.tfr_multitaper`
# or :func:`mne.time_frequency.tfr_stockwell`.

# define frequencies of interest (log-spaced)
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

###############################################################################
# Inspect power
# -------------
#
# .. note::
#     The generated figures are interactive. In the topo you can click
#     on an image to visualize the data for one censor.
#     You can also select a portion in the time-frequency plane to
#     obtain a topomap for a certain time-frequency region.
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
power.plot([82], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[82])

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0],
                   title='Alpha', vmax=0.45, show=False)
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1],
                   title='Beta', vmax=0.45, show=False)
mne.viz.tight_layout()
plt.show()

###############################################################################
# Inspect ITC
# -----------
itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')

###############################################################################
# .. note::
#     Baseline correction can be applied to power or done in plots
#     To illustrate the baseline correction in plots the next line is
#     commented:

# power.apply_baseline(baseline=(-0.5, 0), mode='logratio')

###############################################################################
# Exercise
# --------
#
#    - Visualize the intertrial coherence values as topomaps as done with
#      power.
