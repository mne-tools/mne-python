"""
========================================
Regression on continuous data
========================================

This demonstrates how encoding models can be fit with multiple continuous
inputs. In this case, a encoding model is fit for one electrode, using the
values of all other electrodes as inputs.
"""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
import numpy as np
from mne.datasets import sample
from mne.encoding.model import EncodingModel
from mne.encoding import DataSubsetter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from copy import deepcopy

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
alphas = np.logspace(1, 6, 10)
ix_split = int(X.shape[-1] * .998)  # Using 99.8% of the data to train
tr = np.arange(X.shape[-1])[:ix_split]
tt = np.arange(X.shape[-1])[ix_split:]
splitter_tr = DataSubsetter(tr)
splitter_tt = DataSubsetter(tt)
preproc = Pipeline([('split', splitter_tr), ('scaler', StandardScaler())])
# For predictions, make the splitter pull testing points.
preproc_pred = deepcopy(preproc)
preproc_pred.steps[0] = ('split', splitter_tt)

# Fit / predict w/ the model
mod = Ridge(alpha=0.)
enc = EncodingModel(est=mod, preproc_x=preproc, preproc_y=preproc)
enc.fit(X.T, y)
y_pred = enc.predict(X.T, preproc_x=preproc_pred)

# Now plot results
cmap = plt.cm.rainbow
f, axs = plt.subplots(1, 2, figsize=(15, 8))

# Plot the first 40 coefficients (one coefficient per channel)
ax = axs[0]
coefs = enc.est._final_estimator.coef_[0]
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
