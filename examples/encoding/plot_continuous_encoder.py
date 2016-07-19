"""
========================================
Regression on continuous data
========================================

This demonstrates how encoding models can be fit with multiple continuous
inputs. In this case, a encoding model is fit for one electrode, using the
values of all other electrodes as inputs. These inputs could also be stimulus
features, such as spectral features of sound.

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
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
import numpy as np
from mne.datasets import sample
from mne.encoding import SampleMasker
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, scale

# Load and preprocess data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True).pick_types(
    meg='grad', stim=False, eeg=False).filter(1, None, method='iir')
ix_enc = 10
ixs_X = np.setdiff1d(range(len(raw.ch_names)), [ix_enc])

# Encoding Model
X = raw._data[ixs_X]
y = raw._data[ix_enc]

f, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ix_plt = 200
axs[0].plot(raw.times[:ix_plt], X[:, :ix_plt].T, color='k', alpha=.2)
axs[0].set_title('Continuous Input Data')
axs[0].set_xlim([0, raw.times[:ix_plt][-1]])
axs[1].plot(raw.times[:ix_plt], y[:ix_plt], color='k')
axs[1].set_title('Continuous Output Data')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude (mV)')

# Define model preprocessing and parameters
ix_split = int(X.shape[-1] * .998)  # Using 99.8% of the data to train
tr = np.arange(X.shape[-1])[:ix_split]
tt = np.arange(X.shape[-1])[ix_split:]
splitter_tr = SampleMasker(ixs=tr)
splitter_tt = SampleMasker(ixs=tt)

pipe_est = make_pipeline(StandardScaler(), Ridge(alpha=0.))
pipe_full = SampleMasker(pipe_est, ixs=splitter_tr)

# Fit / predict w/ the model
pipe_full.fit(X.T, y)
y_pred = pipe_full.predict(X.T)

# Now plot results
cmap = plt.cm.rainbow
f, axs = plt.subplots(1, 2, figsize=(15, 8))

# Plot the first 40 coefficients (one coefficient per channel)
ax = axs[0]
coefs = pipe_full._final_estimator.coef_[0]
ax.plot(coefs)
ax.set_xlabel('Channel index')
ax.set_ylabel('Channel weight')
ax.set_title('Encoding model coefficients')
ax.set_xlim([0, coefs.shape[-1]])


# Now plot the predictions on the test set
ax = axs[1]
time_test = np.arange(y_pred.shape[0]) / float(raw.info['sfreq'])
ax.plot(time_test, y_pred, color='r', alpha=.6)

# Plot true output signal
ln = ax.plot(time_test, scale(y[tt]), color='k', alpha=.6, lw=3,
             label='True Signal')
ax.set_title('Predicted (colors) and actual (black) activity')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Signal amplitude (z-scored)')
ax.set_xlim([0, time_test[-1]])

plt.tight_layout()
plt.show()
