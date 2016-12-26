"""
=============================================
Receptive field estimation on continuous data
=============================================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, we simulate the model behind a spectro-temporal receptive
field (or STRF). First, we create a linear filter that maps patterns in
spectro-temporal space onto an output, representing neural activity. We fit
a receptive field model that attempts to recover the original linear filter
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
from scipy.stats import multivariate_normal
from mne.decoding import ReceptiveField, delay_time_series
from sklearn.preprocessing import scale
np.random.seed(1337)  # To make this example reproducible


###############################################################################
# Create a receptive field
# ------------------------
#
# We'll simulate a linear receptive field for a theoretical neural signal. This
# defines how the signal will respond to power in this receptive field space.
n_freqs = 32
n_lags = 20
lags = np.linspace(0, .4, n_lags)
freqs = np.logspace(2, np.log10(5000), n_freqs)
grid = np.array(np.meshgrid(lags, freqs))

# We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
grid = grid.swapaxes(0, -1).swapaxes(0, 1)

# Simulate a temporal receptive field with a Gabor filter
means_high = [.1, 2000]
means_low = [.15, 3000]
cov = [[.0005, 0], [0, 200000]]
gauss_high = multivariate_normal.pdf(grid, means_high, cov)
gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
weights = gauss_high + gauss_low  # Combine to create the "true" STRF
fig, ax = plt.subplots()
ax.pcolormesh(lags, freqs, weights, cmap='coolwarm')
ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
plt.autoscale(tight=True)


###############################################################################
# Simulate a neural response
# --------------------------
#
# Using this receptive field, we'll create an artificial neural response to
# a stimulus.

# Read in audio that's been recorded in epochs. We'll use the first 6.
path_audio = mne.datasets.misc.data_path()
audio = mne.read_epochs(path_audio + '/audio/audio-epo.fif')
audio = audio[:6]
n_epochs = len(audio)
n_decim = 8
spec = mne.time_frequency.tfr_morlet(audio, freqs, 10, return_itc=False,
                                     average=False, picks=[0], decim=n_decim)
sfreq_new = audio.info['sfreq'] / n_decim
X = spec.data.squeeze()
n_times = X.shape[-1]

# Delay the spectrogram according to lags so it can be combined w/ the STRF
# Lags will now be in the 1st dimension, then we reshape to vectorize
X_del = delay_time_series(X, lags, sfreq_new, newaxis=1)
X_del = X_del.reshape([n_epochs, -1, n_times])
n_epochs, n_features, n_delays = X_del.shape
weights_sim = weights.ravel()

# Simulate a neural response to the sound, given this STRF
y = np.zeros([n_features, n_delays])
# import IPython; IPython.embed()
for ii, iep in enumerate(X_del):
    # Simulate this epoch and add random noise
    noise_amp = 1e5
    y[ii] = np.dot(weights_sim, iep) + noise_amp * np.random.randn(n_delays)

# Plot the first 3 trials of audio and the simulated electrode activity
X_plt = scale(np.hstack(X[:3]).T).T
y_plt = scale(np.hstack(y[:3]))
time = np.arange(X_plt.shape[-1]) / sfreq_new
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='viridis')
axs[0].set_title('Input auditory features')
axs[0].set(ylim=[freqs.min(), freqs.max()])
axs[1].plot(time, y_plt)
axs[1].set(xlim=[time.min(), time.max()], title='Simulated response',
           xlabel='Time (s)')

###############################################################################
# Fit a model to recover this receptive field
# -----------------
#
# Finally, we'll use the `ReceptiveField` class to recover the linear receptive
# field of this signal.

# Create training and testing data
train, test = range(5), 5
X_train = X[train]
X_test = X[test]
y_train = y[train]
y_test = y[test]

# Model the simulated data as a function of the spectrogram input
mod = ReceptiveField(lags, freqs, sfreq=sfreq_new)
mod.fit(X_train, y_train)

# Now make predictions about the model output, given input stimuli.
y_pred = mod.predict(X_test).squeeze()

# Plot the actual response and the predicted response on a held out stimulus
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
axs[0].pcolormesh(lags, freqs, weights, cmap='coolwarm')
axs[1].pcolormesh(mod.lags, mod.feature_names, mod.coef_, cmap='coolwarm')
plt.autoscale(tight=True)
axs[0].set_title('Original STRF')
axs[1].set_title('Reconstructed STRF')
plt.setp([iax.get_xticklabels() for iax in axs], rotation=45)

# Plot the original STRF, and the one that we recovered with modeling.
time_pred = np.arange(y_pred.shape[0]) / sfreq_new
fig, ax = plt.subplots()
ax.plot(time_pred, y_test[~mod.msk_remove], color='k', alpha=.2, lw=4)
ax.plot(time_pred, y_pred, color='r', lw=1)
ax.set(title='Original and predicted activity', xlabel='Time (s)')
ax.legend(['Original', 'Predicted'])
plt.show()
