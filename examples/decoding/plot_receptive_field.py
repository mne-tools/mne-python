"""
=============================================
Receptive field estimation on continuous data
=============================================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, we simulate the model behind a spectro-temporal receptive
field (or STRF). First, we create a linear filter that maps patterns in
spectro-temporal space onto an output, representing neural activity. We fit
a receptive field model that attempts to recovre the original linear filter
that was used to create this data.

Estimation of spectro-temporal and spatio-temporal receptive fields using
modeling with continuous inputs is described in:

    Theunissen, F. E. et al. Estimating spatio-temporal receptive
    fields of auditory and visual neurons from their responses to
    natural stimuli. Network 12, 289-316 (2001).

    Willmore, B. & Smyth, D. Methods for first-order kernel
    estimation: simple-cell receptive fields from responses to
    natural scenes. Network 14, 553-77 (2003).
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stt
from mne.decoding.receptive_field import ReceptiveField, delay_time_series
from sklearn.preprocessing import scale
np.random.seed(1337)

# Path to audio stimuli
audio_path = 'audio-epo.fif'

# --- Create a receptive field ---
# Define the time/freq space for our receptive field
n_freqs = 32
n_lags = 20
lags = np.linspace(0, .4, n_lags)
freqs = np.logspace(2, np.log10(6000), n_freqs)
grid = np.array(np.meshgrid(lags, freqs))
grid = grid.swapaxes(0, -1).swapaxes(0, 1)

# Receptive field parameters
means_hi = [.1, 2000]
means_lo = [.15, 3000]
cov = [[.0005, 0], [0, 200000]]
gauss_hi = stt.multivariate_normal.pdf(grid, means_hi, cov)
gauss_lo = -1 * stt.multivariate_normal.pdf(grid, means_lo, cov)
weights = gauss_hi + gauss_lo  # Combine to create the "true" STRF

# --- Use the receptive field to simulate a response to sound ---
# Load the auditory input and turn it into a spectrogram
# XXX only pulling a few trials to speed up
audio = mne.read_epochs(audio_path)
n_ep = len(audio)

n_decim = 8
spec = mne.time_frequency.tfr_morlet(audio, freqs, 10, return_itc=False,
                                     average=False, picks=[0], decim=n_decim)
sfreq_new = audio.info['sfreq'] / n_decim
X = spec.data[:6, 0, ...]
n_times = X.shape[-1]

# Delay the spectrogram according to lags so it can be combined w/ the STRF
X_del = delay_time_series(X, lags, sfreq_new)
X_del = X_del.swapaxes(0, 1).reshape([n_ep, -1, n_times])
weights_sim = weights.reshape([-1])

# Simulate a neural response to the sound, given this STRF
y = np.zeros([X_del.shape[0], X_del.shape[-1]])
# import IPython; IPython.embed()
for ii, iep in enumerate(X_del):
    # Simulate this epoch and add random noise
    y[ii] = np.dot(weights_sim, iep) + 1e5 * np.random.randn(y.shape[-1])

# Plot the audio and the simulated electrode activity
X_plt = scale(np.hstack(X[:3]).T).T
y_plt = scale(np.hstack(y[:3]))
time = np.arange(X_plt.shape[-1]) / sfreq_new
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='viridis')
axs[0].set_title('Input auditory features')
axs[1].plot(time, y_plt)
_ = plt.setp(axs[1], xlim=[time.min(), time.max()],
             title='Simulated response')


# --- Fit a model to recover this receptive field using simulated data ---
# Create training and testing data
X_train = X[:5]
X_test = X[5]
y_train = y[:5]
y_test = y[5]

# Model the simulated data as a function of the spectrogram input
mod = ReceptiveField(freqs, lags, sfreq=sfreq_new)
mod.fit(X_train, y_train)

# Now make predictions about the model output, given input stimuli.
y_pred = mod.predict(X_test)

# Plot the actual response and the predicted response on a held out stimulus
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
axs[0].pcolormesh(lags, freqs, weights, cmap='coolwarm')
mod.plot_coefs(axs[1])
axs[0].set_title('Original STRF')
axs[1].set_title('Reconstructed STRF')

# Plot the original STRF, and the one that we recovered with modeling.
fig, ax = plt.subplots()
ax.plot(y_test[~mod.msk_remove], color='k', alpha=.2, lw=4)
ax.plot(y_pred, color='r', lw=1)
ax.set_title('Original and predicted activity')
ax.legend(['Original', 'Predicted'])
plt.show()
