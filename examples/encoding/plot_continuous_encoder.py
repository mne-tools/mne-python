"""
=============================
Regression on continuous data
=============================

This demonstrates how an encoding model can be fit with multiple continuous
inputs. In this case, a encoding model is fit for one EEG channel, using the
values of all other electrodes as inputs. The coefficients in this case are
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
from mne.encoding import SampleMasker, get_coefs, get_final_est
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

###############################################################################
# Load and preprocess data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True).pick_types(
    meg=False, stim=False, eeg=True).filter(1, None, method='iir')
seed_elec = 'EEG 011'
ix_seed = mne.pick_channels(raw.ch_names, [seed_elec])[0]
ix_targets = np.setdiff1d(range(len(raw.ch_names)), [ix_seed])

# Set up encoding model variables
X = raw[:][0][ix_targets]
y = raw[:][0][ix_seed]
n_targets, n_times = X.shape

###############################################################################
# Plot raw data
plt_times = raw.times[:200]
fig, ax = plt.subplots(figsize=(10, 5))
for idata, alpha, lw, color in zip([X, y], [.2, .8], [1, 3], ['k', 'r']):
    ax.plot(plt_times, idata[..., :200].T, color=color, alpha=alpha, lw=lw)
ax.set_title('Continuous Data')
ax.set_xlim([0, plt_times[-1]])
ax.legend([ax.lines[0], ax.lines[-1]], ['Inputs', 'Seed'])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')

###############################################################################
# Fit the model / Make predictions
# Define model preprocessing and parameters
ix_split = int(n_times * .998)  # Using subset of data so it fits in a plot
ix_tr = np.arange(n_times)[:ix_split]
ix_tt = np.arange(n_times)[ix_split:]

# This is the final estimator that will be called after masking data
pipe_est = make_pipeline(StandardScaler(), Ridge(alpha=0.))
# This allows us to use a subset of indices for training / prediction
pipe_full = SampleMasker(pipe_est, ixs=ix_tr, ixs_pred=ix_tt)

# Fit / predict w/ the model
pipe_full.fit(X.T, y)
y_pred = pipe_full.predict(X.T)


###############################################################################
# Plot raw coefficients
cmap = plt.get_cmap('rainbow')
fig, (ax_input, ax_output) = plt.subplots(1, 2, figsize=(15, 8))

# Plot the first 40 coefficients (one coefficient per channel)
coefs = get_coefs(get_final_est(pipe_full))[0]
ax_input.plot(coefs)
ax_input.set_xlabel('Channel index')
ax_input.set_ylabel('Channel weight')
ax_input.set_title('Encoding model coefficients')
ax_input.set_xlim([0, coefs.shape[-1]])

###############################################################################
# Now plot the predictions on the test set
time_test = np.arange(y_pred.shape[0]) / float(raw.info['sfreq'])
ax_output.plot(time_test, y_pred, color='r', alpha=.6)

# Plot true output signal
ln = ax_output.plot(time_test, y[ix_tt], color='k', alpha=.3, lw=3,
                    label='True Signal')
ax_output.set_title('Predicted (red) and actual (black) activity')
ax_output.set_xlabel('Time (s)')
ax_output.set_ylabel('Signal amplitude (z-scored)')
ax_output.set_xlim([0, time_test[-1]])

###############################################################################
# Plot the coefficients as a topomap
fig = raw.plot_sensors()
scat = fig.axes[0].collections[0]
coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min())
colors = plt.cm.coolwarm(coefs)
sizes = coefs * 1000
# Insert values for the seed electrode
colors = np.insert(colors, ix_seed, [1, 1, 1, 1], axis=0)
sizes = np.insert(sizes, ix_seed, 500)
scat.set_facecolors(colors)
scat.set_sizes(sizes)
fig.suptitle('Scaled model coefficients (white = seed)')

plt.tight_layout()
plt.show()
