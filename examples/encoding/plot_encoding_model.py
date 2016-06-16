"""
========================================
Regression on continuous and event-based data)
========================================
Neat description. Oooo so cool.

rERPs are described in:

  Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
  waveforms: II. Non-linear effects, overlap correction, and practical
  considerations. Psychophysiology, 52(2), 169-189.

Estimation of receptive fields and modeling with continuous inputs
is described in:

  Theunissen, F. E. et al. Estimating spatio-temporal receptive
  fields of auditory and visual neurons from their responses to
  natural stimuli. Network 12, 289-316 (2001).

  Willmore, B. & Smyth, D. Methods for first-order kernel
  estimation: simple-cell receptive fields from responses to
  natural scenes. Network 14, 553-77 (2003).
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.de>
#
#
# License: BSD (3-clause)

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stt
from sklearn.linear_model import Ridge
from mne.encoding import EncodingModel, delay_timeseries
from mne.encoding import DataSubsetter, EventsBinarizer, DataDelayer
from sklearn.pipeline import Pipeline


# Helper function to create data
def modulate_noise(signal, sfreq, delays, weights, snr=1):
    sig_delayed = delay_timeseries(signal[np.newaxis, :], sfreq, delays)
    output = np.dot(weights, sig_delayed)
    noise = (np.percentile(output, 95) / snr) * np.random.randn(len(output))
    return output + noise

# -- Creating stimuli --
# Define our continuous signal
sfreq = 1000.
n_sec = 10.
freq = 3.
amp = 1
time = np.linspace(0, n_sec, sfreq * n_sec)
stim_cont = amp * np.sin(2 * np.pi * freq * time)

# Define events
n_events = 20
events = np.linspace(0, stim_cont.shape[0] - 1, n_events).astype(int)
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)]).T

# Define weights we'll use to modulate the signal
delays_sig = np.arange(0, -.4, -.01)
weights = stt.norm.pdf(delays_sig, -.2, .05)

# Now create our stimulus-modulated signal
snr = 3
sig_cont = modulate_noise(stim_cont, sfreq, delays_sig,
                          weights=weights, snr=snr)

# And a signal that responds to event onsets
binarizer = EventsBinarizer(stim_cont.shape[-1], sfreq=1)
stim_events = binarizer.fit_transform(events[:, 0], event_ids=events[:, -1])
sig_event = modulate_noise(stim_events[0], sfreq, delays_sig,
                           weights=weights, snr=snr)
info = mne.create_info(['ch'], sfreq, ch_types=['eeg'])
sig_cont = mne.io.RawArray(sig_cont[np.newaxis, :], info)
sig_event = mne.io.RawArray(sig_event[np.newaxis, :], info)


# -- Preparing model features --
# Iterate through our events-based and continuous datasets
data_iterator = [(stim_events, sig_event, 'Events'),
                 (stim_cont[np.newaxis, :], sig_cont, 'Continuous')]

# Define a training / test set of indices
ixs = np.arange(sig_cont.n_times)
ix_split = int(.8 * ixs.shape[0])
tr = ixs[:ix_split]
tt = ixs[ix_split:]


# -- Define preprocessing pipelines for X/y --
# For creating time lags
delays_model = delays_sig
delayer = DataDelayer(delays=delays_model, sfreq=sig_cont.info['sfreq'])

# To subset training data
sub_tr = DataSubsetter(tr)
sub_tt = DataSubsetter(tt)

# Now put them together into pipelines
pipe_x = Pipeline([('delays', delayer), ('subset', sub_tr)])
pipe_y = Pipeline([('subset', sub_tr)])
preproc_x_pred = Pipeline([('delays', delayer), ('subset', sub_tt)])


# -- Fit the model, iterating through alphas --
alphas = np.logspace(0, 2, 4)
f, axs_coef = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8),
                           sharex=True, sharey=True)
f, axs_pred = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8),
                           sharex=True, sharey=False)
for alpha, axcol_coef, axcol_pred in zip(alphas, axs_coef.T, axs_pred.T):
    axcol_coef[0].set_title('Alpha: %s' % np.round(np.log10(alpha), 2))
    axcol_pred[0].set_title('Alpha: %s' % np.round(np.log10(alpha), 2))

    # Define our estimator and create the model
    clf = Ridge(alpha=alpha)
    mod = EncodingModel(est=clf, preproc_x=pipe_x, preproc_y=pipe_y)

    # Fit the model and plot coefficients
    iter_fit = zip(data_iterator, axcol_coef, axcol_pred)
    for (X, y, title), ax_coef, ax_pred in iter_fit:
        mod.fit(X, y._data)

        # Plot coefficients
        ax = ax_coef
        ax.plot(delays_model, mod.coef_.squeeze())
        ax.plot(delays_sig, weights)
        ax.set_xlabel('Lag (s)')

        # Plot predictions (on same data, so kinda cheating)
        predicted = mod.predict(X, preproc_x=preproc_x_pred)
        ax_pred.plot(y._data[:, tt].squeeze(), color='k', alpha=.3)
        ax_pred.plot(predicted.squeeze(), color='r')

# Cleaning up plots
axs_coef[0, -1].legend(['Model coefficients', 'True weights'],
                       loc=(.95, .9), fontsize='small')
axs_pred[0, -1].legend(['Actual signal', 'Model prediction'],
                       loc=(.95, .9), fontsize='small')
for axgrp in [axs_coef, axs_pred]:
    axgrp[0, 0].set_ylabel('Continuous input')
    axgrp[1, 0].set_ylabel('Events-based input (rERP)')
plt.show()
