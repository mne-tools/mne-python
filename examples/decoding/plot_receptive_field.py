"""
=========================================
Receptive Field Estimation and Prediction
=========================================

This example reproduces figures from Lalor et al's mTRF toolbox in
matlab [1]_. We will show how the :class:`mne.decoding.ReceptiveField` class
can perform a similar function along with scikit-learn. We will fit a
linear encoding model using the continuously-varying speech envelope to
predict activity of a 128 channel EEG system.

References
----------
.. [1] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
       The Multivariate Temporal Response Function (mTRF) Toolbox:
       A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
       Frontiers in Human Neuroscience 10, 604. doi:10.3389/fnhum.2016.00604

.. _figure 1: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F1
.. _figure 2: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F2
"""  # noqa: E501

# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join

import mne
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale


###############################################################################
# Load the data from the publication
# ----------------------------------
#
# First we will load the data collected in [1]_. In this experiment subjects
# listened to natural speech. Raw EEG and the speech stimulus was collected.
# We will load these below, downsampling the data in order to speed up
# computation since we know that our features are primarily low-frequency in
# nature. Then we'll visualize both the EEG and speech envelope.

path = mne.datasets.mtrf.data_path()
decim = 2
data = loadmat(join(path, 'speech_data.mat'))
raw = data['EEG'].T
speech = data['envelope'].T
sfreq = float(data['Fs'])
sfreq /= decim
speech = mne.filter.resample(speech, down=decim, npad='auto')
raw = mne.filter.resample(raw, down=decim, npad='auto')


# Read in channel positions and create our MNE objects from the raw data
montage = mne.channels.read_montage('biosemi128')
montage.selection = montage.selection[:128]
info = mne.create_info(montage.ch_names[:128], sfreq, 'eeg', montage=montage)
raw = mne.io.RawArray(raw, info)
n_channels = len(raw.ch_names)

# Plot a sample of brain and stimulus activity
fig, ax = plt.subplots()
lns = ax.plot(scale(raw[:, :800][0].T), color='k', alpha=.1)
ln1 = ax.plot(scale(speech[0, :800]), color='r', lw=2)
ax.legend([lns[0], ln1[0]], ['EEG', 'Speech Envelope'], frameon=False)
ax.set(title="Sample activity", xlabel="Time (s)")
mne.viz.tight_layout()

###############################################################################
# Create and fit a receptive field model
# --------------------------------------
#
# We will construct a model to find the linear relationship between the EEG
# signal and a time-delayed version of the speech envelope. This allows
# us to make predictions about the response to new stimuli.

# Define the delays that we will use in the receptive field
tmin, tmax = -.4, .2

# Initialize the model
rf = ReceptiveField(tmin, tmax, sfreq, feature_names=['envelope'],
                    estimator=1., scoring='corrcoef')
# We'll have (tmax - tmin) * sfreq delays
# and an extra 2 delays since we are inclusive on the beginning / end index
n_delays = int((tmax - tmin) * sfreq) + 2

n_splits = 3
cv = KFold(n_splits)

# Prepare model data (make time the first dimension)
speech = speech.T
Y, _ = raw[:]  # Outputs for the model
Y = Y.T

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
scores = np.zeros((n_splits, n_channels))
for ii, (train, test) in enumerate(cv.split(speech)):
    print('split %s / %s' % (ii, n_splits))
    rf.fit(speech[train], Y[train])
    scores[ii] = rf.score(speech[test], Y[test])
    # coef_ is shape (n_outputs, n_features, n_delays). we only have 1 feature
    coefs[ii] = rf.coef_[:, 0, :]
times = rf.delays_ / float(rf.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_scores = scores.mean(axis=0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots()
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mean_scores)
ax.axhline(0, ls='--', color='r')
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
mne.viz.tight_layout()

###############################################################################
# Investigate model coefficients
# ------------------------------
#
# Finally, we will look at how the linear coefficients (sometimes
# referred to as beta values) are distributed across time delays as well as
# across the scalp. We will recreate `figure 1`_ and `figure 2`_ from [1]_.

# Print mean coefficients across all time delays / channels (see Fig 1 in [1])
time_plot = -.180  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8))
max_coef = mean_coefs.max()
ax.pcolormesh(times, ix_chs, mean_coefs, cmap='RdBu_r',
              vmin=-max_coef, vmax=max_coef)
ax.axvline(time_plot, ls='--', color='k', lw=2)
ax.set(xlabel='Delay (s)', ylabel='Channel', title="Mean Model\nCoefficients",
       xlim=times[[0, -1]], ylim=[len(ix_chs) - 1, 0],
       xticks=np.arange(tmin, tmax + .2, .2))
plt.setp(ax.get_xticklabels(), rotation=45)
mne.viz.tight_layout()

# Make a topographic map of coefficients for a given delay (see Fig 2C in [1])
ix_plot = np.argmin(np.abs(time_plot - times))
fig, ax = plt.subplots()
mne.viz.plot_topomap(mean_coefs[:, ix_plot], pos=info, axes=ax, show=False,
                     vmin=-max_coef, vmax=max_coef)
ax.set(title="Topomap of model coefficients\nfor delay %s" % time_plot)
mne.viz.tight_layout()

plt.show()
