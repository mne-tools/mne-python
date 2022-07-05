# -*- coding: utf-8 -*-
"""
.. _tut-strf:

=====================================================================
Spectro-temporal receptive field (STRF) estimation on continuous data
=====================================================================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, we simulate the model behind a spectro-temporal receptive
field (or STRF). First, we create a linear filter that maps patterns in
spectro-temporal space onto an output, representing neural activity. We fit
a receptive field model that attempts to recover the original linear filter
that was used to create this data.
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

# sphinx_gallery_thumbnail_number = 7

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge

from scipy.stats import multivariate_normal
from scipy.io import loadmat
from sklearn.preprocessing import scale
rng = np.random.RandomState(1337)  # To make this example reproducible

# %%
# Load audio data
# ---------------
#
# We'll read in the audio data from :footcite:`CrosseEtAl2016` in order to
# simulate a response.
#
# In addition, we'll downsample the data along the time dimension in order to
# speed up computation. Note that depending on the input values, this may
# not be desired. For example if your input stimulus varies more quickly than
# 1/2 the sampling rate to which we are downsampling.

# Read in audio that's been recorded in epochs.
path_audio = mne.datasets.mtrf.data_path()
data = loadmat(str(path_audio / 'speech_data.mat'))
audio = data['spectrogram'].T
sfreq = float(data['Fs'][0, 0])
n_decim = 2
audio = mne.filter.resample(audio, down=n_decim, npad='auto')
sfreq /= n_decim

# %%
# Create a receptive field
# ------------------------
#
# We'll simulate a linear receptive field for a theoretical neural signal. This
# defines how the signal will respond to power in this receptive field space.
n_freqs = 20
tmin, tmax = -0.1, 0.4

# To simulate the data we'll create explicit delays here
delays_samp = np.arange(np.round(tmin * sfreq),
                        np.round(tmax * sfreq) + 1).astype(int)
delays_sec = delays_samp / sfreq
freqs = np.linspace(50, 5000, n_freqs)
grid = np.array(np.meshgrid(delays_sec, freqs))

# We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
grid = grid.swapaxes(0, -1).swapaxes(0, 1)

# Simulate a temporal receptive field with a Gabor filter
means_high = [.1, 500]
means_low = [.2, 2500]
cov = [[.001, 0], [0, 500000]]
gauss_high = multivariate_normal.pdf(grid, means_high, cov)
gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
weights = gauss_high + gauss_low  # Combine to create the "true" STRF
kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
              cmap='RdBu_r', shading='gouraud')

fig, ax = plt.subplots()
ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.autoscale(tight=True)
mne.viz.tight_layout()


# %%
# Simulate a neural response
# --------------------------
#
# Using this receptive field, we'll create an artificial neural response to
# a stimulus.
#
# To do this, we'll create a time-delayed version of the receptive field, and
# then calculate the dot product between this and the stimulus. Note that this
# is effectively doing a convolution between the stimulus and the receptive
# field. See `here <https://en.wikipedia.org/wiki/Convolution>`_ for more
# information.

# Reshape audio to split into epochs, then make epochs the first dimension.
n_epochs, n_seconds = 16, 5
audio = audio[:, :int(n_seconds * sfreq * n_epochs)]
X = audio.reshape([n_freqs, n_epochs, -1]).swapaxes(0, 1)
n_times = X.shape[-1]

# Delay the spectrogram according to delays so it can be combined w/ the STRF
# Lags will now be in axis 1, then we reshape to vectorize
delays = np.arange(np.round(tmin * sfreq),
                   np.round(tmax * sfreq) + 1).astype(int)

# Iterate through indices and append
X_del = np.zeros((len(delays),) + X.shape)
for ii, ix_delay in enumerate(delays):
    # These arrays will take/put particular indices in the data
    take = [slice(None)] * X.ndim
    put = [slice(None)] * X.ndim
    if ix_delay > 0:
        take[-1] = slice(None, -ix_delay)
        put[-1] = slice(ix_delay, None)
    elif ix_delay < 0:
        take[-1] = slice(-ix_delay, None)
        put[-1] = slice(None, ix_delay)
    X_del[ii][tuple(put)] = X[tuple(take)]

# Now set the delayed axis to the 2nd dimension
X_del = np.rollaxis(X_del, 0, 3)
X_del = X_del.reshape([n_epochs, -1, n_times])
n_features = X_del.shape[1]
weights_sim = weights.ravel()

# Simulate a neural response to the sound, given this STRF
y = np.zeros((n_epochs, n_times))
for ii, iep in enumerate(X_del):
    # Simulate this epoch and add random noise
    noise_amp = .002
    y[ii] = np.dot(weights_sim, iep) + noise_amp * rng.randn(n_times)

# Plot the first 2 trials of audio and the simulated electrode activity
X_plt = scale(np.hstack(X[:2]).T).T
y_plt = scale(np.hstack(y[:2]))
time = np.arange(X_plt.shape[-1]) / sfreq
_, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax1.pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='Reds',
               shading='gouraud')
ax1.set_title('Input auditory features')
ax1.set(ylim=[freqs.min(), freqs.max()], ylabel='Frequency (Hz)')
ax2.plot(time, y_plt)
ax2.set(xlim=[time.min(), time.max()], title='Simulated response',
        xlabel='Time (s)', ylabel='Activity (a.u.)')
mne.viz.tight_layout()


# %%
# Fit a model to recover this receptive field
# -------------------------------------------
#
# Finally, we'll use the :class:`mne.decoding.ReceptiveField` class to recover
# the linear receptive field of this signal. Note that properties of the
# receptive field (e.g. smoothness) will depend on the autocorrelation in the
# inputs and outputs.

# Create training and testing data
train, test = np.arange(n_epochs - 1), n_epochs - 1
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
                                    (X_train, X_test, y_train, y_test)]
# Model the simulated data as a function of the spectrogram input
alphas = np.logspace(-3, 3, 7)
scores = np.zeros_like(alphas)
models = []
for ii, alpha in enumerate(alphas):
    rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=alpha)
    rf.fit(X_train, y_train)

    # Now make predictions about the model output, given input stimuli.
    scores[ii] = rf.score(X_test, y_test)
    models.append(rf)

times = rf.delays_ / float(rf.sfreq)

# Choose the model that performed best on the held out data
ix_best_alpha = np.argmax(scores)
best_mod = models[ix_best_alpha]
coefs = best_mod.coef_[0]
best_pred = best_mod.predict(X_test)[:, 0]

# Plot the original STRF, and the one that we recovered with modeling.
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
ax1.pcolormesh(delays_sec, freqs, weights, **kwargs)
ax2.pcolormesh(times, rf.feature_names, coefs, **kwargs)
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


# %%
# Visualize the effects of regularization
# ---------------------------------------
#
# Above we fit a :class:`mne.decoding.ReceptiveField` model for one of many
# values for the ridge regularization parameter. Here we will plot the model
# score as well as the model coefficients for each value, in order to
# visualize how coefficients change with different levels of regularization.
# These issues as well as the STRF pipeline are described in detail
# in :footcite:`TheunissenEtAl2001,WillmoreSmyth2003,HoldgrafEtAl2016`.

# Plot model score for each ridge parameter
fig = plt.figure(figsize=(10, 4))
ax = plt.subplot2grid([2, len(alphas)], [1, 0], 1, len(alphas))
ax.plot(np.arange(len(alphas)), scores, marker='o', color='r')
ax.annotate('Best parameter', (ix_best_alpha, scores[ix_best_alpha]),
            (ix_best_alpha, scores[ix_best_alpha] - .1),
            arrowprops={'arrowstyle': '->'})
plt.xticks(np.arange(len(alphas)), ["%.0e" % ii for ii in alphas])
ax.set(xlabel="Ridge regularization value", ylabel="Score ($R^2$)",
       xlim=[-.4, len(alphas) - .6])
mne.viz.tight_layout()

# Plot the STRF of each ridge parameter
for ii, (rf, i_alpha) in enumerate(zip(models, alphas)):
    ax = plt.subplot2grid([2, len(alphas)], [0, ii], 1, 1)
    ax.pcolormesh(times, rf.feature_names, rf.coef_[0], **kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.autoscale(tight=True)
fig.suptitle('Model coefficients / scores for many ridge parameters', y=1)
mne.viz.tight_layout()

# %%
# Using different regularization types
# ------------------------------------
# In addition to the standard ridge regularization, the
# :class:`mne.decoding.TimeDelayingRidge` class also exposes
# `Laplacian <https://en.wikipedia.org/wiki/Laplacian_matrix>`_ regularization
# term as:
#
# .. math::
#    \left[\begin{matrix}
#         1 & -1 &   &   & & \\
#        -1 &  2 & -1 &   & & \\
#           & -1 & 2 & -1 & & \\
#           & & \ddots & \ddots & \ddots & \\
#           & & & -1 & 2 & -1 \\
#           & & &    & -1 & 1\end{matrix}\right]
#
# This imposes a smoothness constraint of nearby time samples and/or features.
# Quoting :footcite:`CrosseEtAl2016` :
#
#    Tikhonov [identity] regularization (Equation 5) reduces overfitting by
#    smoothing the TRF estimate in a way that is insensitive to
#    the amplitude of the signal of interest. However, the Laplacian
#    approach (Equation 6) reduces off-sample error whilst preserving
#    signal amplitude (Lalor et al., 2006). As a result, this approach
#    usually leads to an improved estimate of the system’s response (as
#    indexed by MSE) compared to Tikhonov regularization.
#

scores_lap = np.zeros_like(alphas)
models_lap = []
for ii, alpha in enumerate(alphas):
    estimator = TimeDelayingRidge(tmin, tmax, sfreq, reg_type='laplacian',
                                  alpha=alpha)
    rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=estimator)
    rf.fit(X_train, y_train)

    # Now make predictions about the model output, given input stimuli.
    scores_lap[ii] = rf.score(X_test, y_test)
    models_lap.append(rf)

ix_best_alpha_lap = np.argmax(scores_lap)

# %%
# Compare model performance
# -------------------------
# Below we visualize the model performance of each regularization method
# (ridge vs. Laplacian) for different levels of alpha. As you can see, the
# Laplacian method performs better in general, because it imposes a smoothness
# constraint along the time and feature dimensions of the coefficients.
# This matches the "true" receptive field structure and results in a better
# model fit.

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot2grid([3, len(alphas)], [2, 0], 1, len(alphas))
ax.plot(np.arange(len(alphas)), scores_lap, marker='o', color='r')
ax.plot(np.arange(len(alphas)), scores, marker='o', color='0.5', ls=':')
ax.annotate('Best Laplacian', (ix_best_alpha_lap,
                               scores_lap[ix_best_alpha_lap]),
            (ix_best_alpha_lap, scores_lap[ix_best_alpha_lap] - .1),
            arrowprops={'arrowstyle': '->'})
ax.annotate('Best Ridge', (ix_best_alpha, scores[ix_best_alpha]),
            (ix_best_alpha, scores[ix_best_alpha] - .1),
            arrowprops={'arrowstyle': '->'})
plt.xticks(np.arange(len(alphas)), ["%.0e" % ii for ii in alphas])
ax.set(xlabel="Laplacian regularization value", ylabel="Score ($R^2$)",
       xlim=[-.4, len(alphas) - .6])
mne.viz.tight_layout()

# Plot the STRF of each ridge parameter
xlim = times[[0, -1]]
for ii, (rf_lap, rf, i_alpha) in enumerate(zip(models_lap, models, alphas)):
    ax = plt.subplot2grid([3, len(alphas)], [0, ii], 1, 1)
    ax.pcolormesh(times, rf_lap.feature_names, rf_lap.coef_[0], **kwargs)
    ax.set(xticks=[], yticks=[], xlim=xlim)
    if ii == 0:
        ax.set(ylabel='Laplacian')
    ax = plt.subplot2grid([3, len(alphas)], [1, ii], 1, 1)
    ax.pcolormesh(times, rf.feature_names, rf.coef_[0], **kwargs)
    ax.set(xticks=[], yticks=[], xlim=xlim)
    if ii == 0:
        ax.set(ylabel='Ridge')
fig.suptitle('Model coefficients / scores for laplacian regularization', y=1)
mne.viz.tight_layout()

# %%
# Plot the original STRF, and the one that we recovered with modeling.
rf = models[ix_best_alpha]
rf_lap = models_lap[ix_best_alpha_lap]
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3),
                                  sharey=True, sharex=True)
ax1.pcolormesh(delays_sec, freqs, weights, **kwargs)
ax2.pcolormesh(times, rf.feature_names, rf.coef_[0], **kwargs)
ax3.pcolormesh(times, rf_lap.feature_names, rf_lap.coef_[0], **kwargs)
ax1.set_title('Original STRF')
ax2.set_title('Best Ridge STRF')
ax3.set_title('Best Laplacian STRF')
plt.setp([iax.get_xticklabels() for iax in [ax1, ax2, ax3]], rotation=45)
plt.autoscale(tight=True)
mne.viz.tight_layout()

# %%
# References
# ==========
# .. footbibliography::
