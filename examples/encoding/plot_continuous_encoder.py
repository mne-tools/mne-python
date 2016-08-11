"""
=============================
Regression on continuous data
=============================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, a encoding model is fit for one EEG channel, using the
values of all other channels as inputs. The coefficients in this case are
interpreted as a measure of functional connectivity. These inputs could also
be stimulus features, such as spectral features of sound. In this case, model
coefficients would be measures of receptive field properties for the channel.

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

import matplotlib.pyplot as plt

import mne
import numpy as np
from mne.datasets import sample
from mne.encoding import SubsetEstimator, FeatureDelayer, get_coefs
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

###############################################################################
# Load and preprocess data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True).pick_types(
    meg=False, stim=False, eeg=True).filter(1, None, method='iir')
seed_channel = 'EEG 011'  # Choosing a frontal electrode
ix_seed = mne.pick_channels(raw.ch_names, [seed_channel])[0]
ix_targets = np.setdiff1d(range(len(raw.ch_names)), [ix_seed])

# Set up encoding model variables
# Multiple continuous electrode values as input
X = raw[:][0][ix_targets].T  # estimators expect shape (n_samples, n_features)
# Single continuous electrode values as output
y = raw[:][0][ix_seed]
n_times, n_targets = X.shape

###############################################################################
# Fit the model / Make predictions
# Define model preprocessing and parameters
ix_train = np.arange(n_times // 2)
ix_test = np.arange(n_times // 2, n_times)

# This creates time-lagged versions of the data (in seconds)
delay_step = .01  # Time between delays (in seconds)
delayer = FeatureDelayer(delays=np.arange(-.1, .1, delay_step),
                         sfreq=raw.info['sfreq'])
info_new = raw.copy().drop_channels([seed_channel]).info.copy()
info_new['sfreq'] = 1. / delay_step

# This is the final estimator that will be called after masking data
estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.))
# This allows us to use a subset of indices for training / prediction
masker = SubsetEstimator(estimator, samples_train=ix_train,
                         samples_pred=ix_test)
# Finally, place a delayer first so that samples are delayed before being split
pipeline = make_pipeline(delayer, Vectorizer(), masker)

# Fit / predict with the model
pipeline.fit(X, y)
y_pred = pipeline.predict(X)

###############################################################################
# Plot raw data
fig, (ax_raw, ax_pred) = plt.subplots(2, 1, figsize=(8, 8))
plt_times = raw.times[:200]
plt_data = [(X, .2, 1, 'k'), (y, .8, 3, 'r')]
for idata, alpha, lw, color in plt_data:
    ax_raw.plot(plt_times, idata[:200], color=color, alpha=alpha, lw=lw)
ax_raw.set_title('Continuous training data')
# ax_raw.set_xlim([0, plt_times[-1]])
ax_raw.legend([ax_raw.lines[0], ax_raw.lines[-1]], ['Inputs', 'Seed'])
ax_raw.set_xlabel('Time (s)')
ax_raw.set_ylabel('Amplitude (mV)')

# Plot the predictions on the test set
n_pred = int(4 * raw.info['sfreq'])
time_test = np.arange(n_pred) / float(raw.info['sfreq'])
ln_pred = ax_pred.plot(time_test, y_pred[:n_pred], color='r', alpha=.6, lw=2)
# Plot true output signal
ln_true = ax_pred.plot(time_test, y[ix_test][:n_pred],
                       color='k', alpha=.3, lw=3)
ax_pred.legend([ln_pred[0], ln_true[0]], ['Predicted', 'Actual'])
ax_pred.set_title('Model predictions')
ax_pred.set_xlabel('Time (s)')
ax_pred.set_ylabel('Signal amplitude')
ax_pred.set_xlim([0, time_test[-1]])
plt.tight_layout()

###############################################################################
# Pull and reshape the coefficients for plotting
coefs = get_coefs(pipeline)
vectorizer = pipeline.named_steps['vectorizer']
coefs_delayed = vectorizer.inverse_transform(coefs[np.newaxis, :])[0]
coefs_evoked = mne.EvokedArray(coefs_delayed, info_new, tmin=-.1)
fig = coefs_evoked.plot_joint(title='Coefficients over time delays')
