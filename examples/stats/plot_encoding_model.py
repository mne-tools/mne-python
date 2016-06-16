"""
========================================
Regression on continuous and event-based data)
========================================

Neat description. Oooo so cool.

rERPs are described in:
Smith, N. J., & Kutas, M. (2015). Regression-based estimation of ERP
waveforms: II. Non-linear effects, overlap correction, and practical
considerations. Psychophysiology, 52(2), 169-189.
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.de>
#
#
# License: BSD (3-clause)
import mne
import numpy as np
from scipy import stats as stt
from sklearn.linear_model import Ridge
from mne.stats.regression import EncodingModel, delay_timeseries
import matplotlib.pyplot as plt


def modulate_noise(signal, sfreq, delays, weights, snr=1):
    sig_delayed = delay_timeseries(signal[np.newaxis, :], sfreq, delays)
    output = np.dot(weights, sig_delayed)
    noise = (np.percentile(output, 95) / snr) * np.random.randn(len(output))
    return output + noise


# Define our continuous signal
sfreq = 1000.
n_sec = 10.
freq = 3.
amp = 1
time = np.linspace(0, n_sec, sfreq * n_sec)
stim_cont = amp * np.sin(2 * np.pi * freq * time)

# Define event-based signal
n_events = 20
events = np.linspace(0, stim_cont.shape[0] - 1, n_events).astype(int)
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)]).T
stim_events = np.zeros_like(stim_cont)
stim_events[events[:, 0]] = 1

# Define weights we'll use to modulate the signal
delays_sig = np.arange(0, -.4, -.01)
weights = stt.norm.pdf(delays_sig, -.2, .05)

# Now create our stimulus-modulated signal
snr = 3
sig_cont = modulate_noise(stim_cont, sfreq, delays_sig,
                          weights=weights, snr=snr)
sig_event = modulate_noise(stim_events, sfreq, delays_sig,
                           weights=weights, snr=snr)
info = mne.create_info(['ch'], sfreq, ch_types=['eeg'])
sig_cont = mne.io.RawArray(sig_cont[np.newaxis, :], info)
sig_event = mne.io.RawArray(sig_event[np.newaxis, :], info)

# Now fit models with a cross-validation object
data_iterator = [(sig_cont, stim_cont,
                  dict(continuous=stim_cont), 'Continuous'),
                 (sig_event, stim_events,
                  dict(events=events), 'Events')]
ixs = np.arange(sig_cont.n_times)
ix_split = int(.8 * ixs.shape[0])
tr = ixs[:ix_split]
tt = ixs[ix_split:]

alphas = np.logspace(-.5, 1, 4)
f, axs_coef = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8),
                           sharex=True, sharey=True)
f, axs_pred = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8),
                           sharex=True, sharey=False)

for alpha, axcol_coef, axcol_pred in zip(alphas, axs_coef.T, axs_pred.T):
    axcol_coef[0].set_title('Alpha: %s' % np.round(np.log10(alpha), 2))
    axcol_pred[0].set_title('Alpha: %s' % np.round(np.log10(alpha), 2))

    clf = Ridge(alpha=alpha)
    delays_model = delays_sig
    mod = EncodingModel(delays=delays_model, est=clf)
    iter_fit = zip(data_iterator, axcol_coef, axcol_pred)
    for (sig, stim, sig_dict, title), ax_coef, ax_pred in iter_fit:
        mod.fit(sig, fit_ixs=tr, **sig_dict)
        # Plot coefficients
        ax = ax_coef
        ax.plot(delays_model, mod.coef_.squeeze())
        ax.plot(delays_sig, weights)
        ax.set_xlabel('Lag (s)')

        # Plot predictions (on same data, so kinda cheating)
        predicted = mod.predict(stim[np.newaxis, tt])
        ax_pred.plot(sig._data[0, tt].squeeze())
        ax_pred.plot(predicted.squeeze())
axs_coef[0, -1].legend(['Model coefficients', 'True weights'],
                       loc=(.95, .9), fontsize='small')
axs_pred[0, -1].legend(['Model prediction', 'Actual signal'],
                       loc=(.95, .9), fontsize='small')
for axgrp in [axs_coef, axs_pred]:
    axgrp[0, 0].set_ylabel('Continuous input')
    axgrp[1, 0].set_ylabel('Events-based input (rERP)')
plt.show()
