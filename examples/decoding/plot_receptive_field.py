"""
=========================================
Receptive Field Estimation and Prediction
=========================================

This example re-creates figures from a paper from the Lalor group which
demos the mTRF toolbox in Matlab _[1]. We will show how the
:class:`mne.decoding.ReceptiveField` class can perform a similar function
along with :mod:`sklearn`. We'll fit a linear encoding model using the
continuously-varying speech envelope to predict activity of a 128 channel
EEG system.

References
----------
.. [1] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
       The Multivariate Temporal Response Function (mTRF) Toolbox:
       A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
       Frontiers in Human Neuroscience 10, 604. doi:10.3389/fnhum.2016.00604
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)


from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import mne


###############################################################################
# Load the data from the publication
path = mne.datasets.mtrf.data_path()
data = loadmat(path + '/speech_data.mat')
raw = data['EEG'].T
speech = data['envelope'].T
sfreq = float(data['Fs'].squeeze())

# Read in channel positions and create our MNE objects from the raw data
mon = mne.channels.read_montage('biosemi128')
mon.selection = mon.selection[:128]
layout = mne.channels.read_layout('Vectorview-all')
info = mne.create_info(mon.ch_names[:128], sfreq, 'eeg', montage=mon)
layout = mne.channels.make_eeg_layout(info)
raw = mne.io.RawArray(raw, info)

###############################################################################
# Plot a sample of brain and stimulus activity
fig, ax = plt.subplots()
ax.plot(scale(raw[:][0][..., :800].T), color='k', alpha=.1)
ax.plot(scale(speech[0, :800]), color='r', lw=2)
ax.set(title="Sample brain and stimulus activity", xlabel="Time (s)")
mne.viz.tight_layout()

###############################################################################
# Create and fit a receptive field model

# Define the lags that we'll use in the receptive field
step_lag = .02
lags_time = np.arange(.2, -.4, -step_lag)
lags_sample = (lags_time * sfreq).astype(int)
n_lags = len(lags_sample)

# Initialize the model
spec_names = ['freq_%s' % ii for ii in range(speech.shape[0])]
mod = Ridge()
rf = ReceptiveField(lags_sample, spec_names)

n_cv = 3
kf = KFold(n_cv)

# Iterate through folds, fit the model, and predict/test on held-out data
coefs = np.zeros([n_cv, len(raw.ch_names), len(lags_sample)])
scores = np.zeros([n_cv, len(raw.ch_names)])
for ii, (tr, tt) in enumerate(kf.split(speech.T)):
    print('CV iteration %s' % ii)
    for jj, i_ch in enumerate(raw[:][0]):
        rf.fit(speech[..., tr], i_ch[tr])
        preds = rf.predict(speech[..., tt]).squeeze()
        scores[ii, jj] = np.corrcoef(preds, i_ch[tt][rf.mask_pred])[1, 0]
        coefs[ii, jj] = rf.coef_

mn_coefs = coefs.mean(0)
mn_scores = scores.mean(0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots()
ix_chs = range(len(raw.ch_names))
ax.plot(ix_chs, mn_scores)
ax.axhline(0, ls='--', color='r')
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
mne.viz.tight_layout()

###############################################################################
# Investigate model coefficients
#
# Here we'll look at how coefficients are distributed across time lag as well
# as across the scalp. We'll recreate figures 1 and 2C in [1]_

# Print mean coefficients across all time lags / channels (see Fig 1 in [1])
time_plot = -.18  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8))
ax.pcolormesh(lags_time, ix_chs, mn_coefs, cmap='RdBu_r')
ax.axvline(time_plot, ls='--', color='k', lw=2)
ax.set(xlabel='Lag (s)', ylabel='Channel', title="Mean Model\nCoefficients",
       xlim=[lags_time.min(), lags_time.max()], ylim=[0, len(ix_chs)])
plt.setp(ax.get_xticklabels(), rotation=45)
mne.viz.tight_layout()

# Make a topographic map of coefficients for a given lag (see Fig 2C in [1])
ix_plot = np.argmin(np.abs(time_plot - lags_time))
fig, ax = plt.subplots()
mne.viz.plot_topomap(mn_coefs[:, ix_plot], pos=layout.pos, axes=ax, show=False)
ax.set(title="Topomap of model coefficicients\nfor lag %s" % time_plot)
mne.viz.tight_layout()

plt.show()
