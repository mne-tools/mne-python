"""
.. _tut-sensors-time-freq:

=============================================
Frequency and time-frequency sensors analysis
=============================================

The objective is to show you how to explore the spectral content
of your data (frequency and time-frequency). Here we'll work on Epochs.

We will use this dataset: :ref:`somato-dataset`. It contains so-called event
related synchronizations (ERS) / desynchronizations (ERD) in the beta band.
"""  # noqa: E501
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper
from mne.datasets import somato

###############################################################################
# Set parameters
data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))

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
# We start by exploring the frequence content of our epochs.


###############################################################################
# Let's first check out all channel types by averaging across epochs.
epochs.plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)

###############################################################################
# Now let's take a look at the spatial distributions of the PSD.
epochs.plot_psd_topomap(ch_type='grad', normalize=True)

###############################################################################
# Alternatively, you can also create PSDs from Epochs objects with functions
# that start with ``psd_`` such as
# :func:`mne.time_frequency.psd_multitaper` and
# :func:`mne.time_frequency.psd_welch`.

f, ax = plt.subplots()
psds, freqs = psd_multitaper(epochs, fmin=2, fmax=40, n_jobs=1)
psds = 10. * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
psds_std = psds.mean(0).std(0)

ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=.5)
ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()

###############################################################################
# .. _inter-trial-coherence:
#
# Time-frequency analysis: power and inter-trial coherence
# --------------------------------------------------------
#
# We now compute time-frequency representations (TFRs) from our Epochs.
# We'll look at power and inter-trial coherence (ITC).
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
#     on an image to visualize the data for one sensor.
#     You can also select a portion in the time-frequency plane to
#     obtain a topomap for a certain time-frequency region.
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
power.plot([82], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[82])

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0],
                   title='Alpha', show=False)
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1],
                   title='Beta', show=False)
mne.viz.tight_layout()
plt.show()

###############################################################################
# Joint Plot
# ----------
# You can also create a joint plot showing both the aggregated TFR
# across channels and topomaps at specific times and frequencies to obtain
# a quick overview regarding oscillatory effects across time and space.

power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
                 timefreqs=[(.5, 10), (1.3, 8)])

###############################################################################
# Inspect ITC
# -----------
itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')

###############################################################################
# .. note::
#     Baseline correction can be applied to power or done in plots.
#     To illustrate the baseline correction in plots, the next line is
#     commented power.apply_baseline(baseline=(-0.5, 0), mode='logratio')

###############################################################################
# Exercise
# --------
#
#    - Visualize the inter-trial coherence values as topomaps as done with
#      power.
