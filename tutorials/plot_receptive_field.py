"""
=====================================================================
Spectro-temporal receptive field (STRF) estimation on continuous data
=====================================================================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, we simulate the model behind a spectro-temporal receptive
field (or STRF). First, we create a linear filter that maps patterns in
spectro-temporal space onto an output, representing neural activity. We fit
a receptive field model that attempts to recover the original linear filter
that was used to create this data.

References
----------
Estimation of spectro-temporal and spatio-temporal receptive fields using
modeling with continuous inputs is described in:

.. [1] Theunissen, F. E. et al. Estimating spatio-temporal receptive
       fields of auditory and visual neurons from their responses to
       natural stimuli. Network 12, 289-316 (2001).

.. [2] Willmore, B. & Smyth, D. Methods for first-order kernel
       estimation: simple-cell receptive fields from responses to
       natural scenes. Network 14, 553-77 (2003).

.. [3] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
       The Multivariate Temporal Response Function (mTRF) Toolbox:
       A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
       Frontiers in Human Neuroscience 10, 604.
       doi:10.3389/fnhum.2016.00604

.. [4] Holdgraf, C. R. et al. Rapid tuning shifts in human auditory cortex
       enhance speech intelligibility. Nature Communications, 7, 13654 (2016).
       doi:10.1038/ncomms13654
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import ReceptiveField, delay_time_series

from scipy.stats import multivariate_normal
from scipy.io import loadmat
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge
rng = np.random.RandomState(1337)  # To make this example reproducible

###############################################################################
# Load audio data
# ---------------
#
# We'll read in the audio data from [3]_ in order to simulate a response.

# Read in audio that's been recorded in epochs.
path_audio = mne.datasets.mtrf.data_path()
data = loadmat(path_audio + '/speech_data.mat')
audio = data['spectrogram'].T
sfreq = float(data['Fs'][0, 0])
n_decim = 2
audio = mne.filter.resample(audio, down=n_decim, npad='auto')
sfreq /= n_decim

###############################################################################
# Create a receptive field
# ------------------------
#
# We'll simulate a linear receptive field for a theoretical neural signal. This
# defines how the signal will respond to power in this receptive field space.
n_freqs = 16
tmin, tmax = -.4, 0.

# To simulate the data we'll create explicit delays here
delays_samp = np.arange(np.round(tmin * sfreq),
                        np.round(tmax * sfreq) + 1).astype(int)
delays_sec = delays_samp / sfreq
freqs = np.logspace(2, np.log10(5000), n_freqs)
grid = np.array(np.meshgrid(delays_sec, freqs))

# We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
grid = grid.swapaxes(0, -1).swapaxes(0, 1)

# Simulate a temporal receptive field with a Gabor filter
means_high = [-.1, 1000]
means_low = [-.15, 2000]
cov = [[.0005, 0], [0, 200000]]
gauss_high = multivariate_normal.pdf(grid, means_high, cov)
gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
weights = gauss_high + gauss_low  # Combine to create the "true" STRF
kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
              cmap='RdBu_r')

fig, ax = plt.subplots()
ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.autoscale(tight=True)
mne.viz.tight_layout()


###############################################################################
# Simulate a neural response
# --------------------------
#
# Using this receptive field, we'll create an artificial neural response to
# a stimulus.

# Reshape audio to split into epochs, then make epochs the first dimension.
n_epochs, n_seconds = 30, 3
audio = audio[:, :int(n_seconds * sfreq * n_epochs)]
X = audio.reshape([n_freqs, n_epochs, -1]).swapaxes(0, 1)
n_times = X.shape[-1]

# Delay the spectrogram according to delays so it can be combined w/ the STRF
# Lags will now be in axis 1, then we reshape to vectorize
X_del = delay_time_series(X, tmin, tmax, sfreq, newaxis=1, axis=-1)
X_del = X_del.reshape([n_epochs, -1, n_times])
n_features = X_del.shape[1]
weights_sim = weights.ravel()

# Simulate a neural response to the sound, given this STRF
y = np.zeros([n_epochs, n_times])
for ii, iep in enumerate(X_del):
    # Simulate this epoch and add random noise
    noise_amp = .0005
    y[ii] = np.dot(weights_sim, iep) + noise_amp * rng.randn(n_times)

# Plot the first 2 trials of audio and the simulated electrode activity
X_plt = scale(np.hstack(X[:2]).T).T
y_plt = scale(np.hstack(y[:2]))
time = np.arange(X_plt.shape[-1]) / sfreq
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='viridis')
axs[0].set_title('Input auditory features')
axs[0].set(ylim=[freqs.min(), freqs.max()], ylabel='Frequency (Hz)')
axs[1].plot(time, y_plt)
axs[1].set(xlim=[time.min(), time.max()], title='Simulated response',
           xlabel='Time (s)', ylabel='Activity (a.u.)')
mne.viz.tight_layout()


###############################################################################
# Fit a model to recover this receptive field
# -------------------------------------------
#
# Finally, we'll use the :class:`mne.decoding.ReceptiveField` class to recover
# the linear receptive field of this signal. Note that properties of the
# receptive field (e.g. smoothness) will depend on the autocorrelation in the
# inputs and outputs.

# Create training and testing data
train, test = np.arange(20), 21
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
                                    (X_train, X_test, y_train, y_test)]
# Model the simulated data as a function of the spectrogram input
alphas = np.logspace(-4, 0, 10)
scores = np.zeros_like(alphas)
models = []
for ii, alpha in enumerate(alphas):
    rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=Ridge(alpha))
    rf.fit(X_train, y_train)

    # Now make predictions about the model output, given input stimuli.
    scores[ii] = rf.score(X_test, y_test)
    models.append(rf)

# Choose the model that performed best on the held out data
ix_best_alpha = np.argmax(scores)
best_mod = models[ix_best_alpha]
coefs = best_mod.coef_
best_pred = best_mod.predict(X_test).squeeze()

# Plot the original STRF, and the one that we recovered with modeling.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
ax1.pcolormesh(delays_sec, freqs, weights, **kwargs)
ax2.pcolormesh(rf.times, rf.feature_names, coefs, **kwargs)
ax1.set_title('Original STRF')
ax2.set_title('Best Reconstructed STRF')
plt.setp([iax.get_xticklabels() for iax in [ax1, ax2]], rotation=45)
plt.autoscale(tight=True)
mne.viz.tight_layout()

# Plot the actual response and the predicted response on a held out stimulus
time_pred = np.arange(best_pred.shape[0]) / sfreq
fig, ax = plt.subplots()
ax.plot(time_pred, y_test, color='k', alpha=.2, lw=4)
ax.plot(time_pred, best_pred, color='r', lw=1)
ax.set(title='Original and predicted activity', xlabel='Time (s)')
ax.legend(['Original', 'Predicted'])
plt.autoscale(tight=True)
mne.viz.tight_layout()


###############################################################################
# Visualize the effects of regularization
# ---------------------------------------
#
# Above we fit a :class:`mne.decoding.ReceptiveField` model for one of many
# values for the "ridge" parameter. Here we will plot the model score as well
# as the model coefficients for each value, in order to visualize how
# coefficients change with different levels of regularization. These issues
# as well as the STRF pipeline are described in detail in [1]_ and [2]_

# Plot model score for each ridge parameter
fig = plt.figure(figsize=(20, 4))
ax = plt.subplot2grid([2, 10], [1, 0], 1, 10)
ax.plot(np.arange(len(alphas)), scores, marker='o', color='r')
ax.annotate('Best parameter', (ix_best_alpha, scores[ix_best_alpha]),
            (ix_best_alpha - 1, scores[ix_best_alpha] - .02),
            arrowprops={'arrowstyle': '->'})
plt.xticks(np.arange(len(alphas)), ["%.0e" % ii for ii in alphas])
ax.set(xlabel="Ridge regularization value", ylabel="Score ($R^2$)",
       ylim=[.9, .96], xlim=[-.4, len(alphas) - .6])
mne.viz.tight_layout()

# Plot the STRF of each ridge parameter
for ii, (rf, i_alpha) in enumerate(zip(models, alphas)):
    ax = plt.subplot2grid([2, 10], [0, ii], 1, 1)
    ax.pcolormesh(rf.times, rf.feature_names, rf.coef_, **kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.autoscale(tight=True)
fig.suptitle('Model coefficients / scores for many ridge parameters', y=1)
mne.viz.tight_layout()

plt.show()
