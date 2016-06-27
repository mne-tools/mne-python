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

f, axs = plt.subplots(2, 1, figsize=(10, 10))
ix_plt = 200
axs[0].plot(raw.times[:ix_plt], X[:, :ix_plt].T, color='k', alpha=.2)
axs[0].set_title('Continuous Input Data')
axs[1].plot(raw.times[:ix_plt], y[:ix_plt], color='k')
axs[1].set_title('Continuous Output Data')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude (mV)')

# Define model preprocessing and parameters
alphas = np.logspace(1, 6, 10)
ix_split = int(X.shape[-1] * .998)  # Using 99.8% of the data to train
tr = np.arange(X.shape[-1])[:ix_split]
tt = np.arange(X.shape[-1])[ix_split:]
splitter = DataSubsetter(tr)
preproc = Pipeline([('cv', splitter), ('scaler', StandardScaler())])


# Iterate through regularization parameters, fit model, and plot coefficients
f, axs = plt.subplots(1, 2, figsize=(15, 8))
cmap = plt.cm.rainbow
for ii, ialpha in enumerate(alphas):
    mod = Ridge(alpha=ialpha)
    enc = EncodingModel(est=mod, preproc_x=preproc, preproc_y=preproc)
    enc.fit(X.T, y)
    preproc_pred = deepcopy(preproc)
    preproc_pred.steps[0] = ('cv', DataSubsetter(tt))
    y_pred = enc.predict(X.T, preproc_x=preproc_pred)

    # Plot the first 40 coefficients (one coefficient per channel)
    ax = axs[0]
    coefs = enc.est._final_estimator.coef_[0, :50]
    ax.plot(coefs, color=cmap(float(ii) / len(alphas)),
            label=np.round(np.log10(ialpha), 2))
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Channel weight')

    # Now plot the predictions on the test set
    time_test = np.arange(y_pred.shape[0]) / float(raw.info['sfreq'])
    ax = axs[1]
    ax.plot(time_test, y_pred, color=cmap(float(ii) / len(alphas)))

# Plot true output signal
ln = axs[1].plot(time_test, scale(y[tt]), color='k', alpha=.6, lw=3,
                 label='True Signal')

# Formatting
axs[0].legend(title='Log Alpha', ncol=2)
axs[0].set_title('Encoding model coefficients, many levels of regularization')

axs[1].set_title('Predicted (colors) and actual (black) activity')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal amplitude (z-scored)')
plt.tight_layout()
plt.show()
