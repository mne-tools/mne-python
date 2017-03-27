"""
==========================
Decoding sensor space data
==========================

Decoding, a.k.a MVPA or supervised machine learning applied to MEG
data in sensor space. Here the classifier is applied to every time
point.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


import mne
from mne.datasets import sample
from mne.decoding import (_SearchLight, _GeneralizationLight,
                          cross_val_multiscore)

data_path = sample.data_path()

plt.close('all')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(2, None)  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

epochs_list = [epochs[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochs_list)

###############################################################################
# Temporal decoding
# -----------------
#
# We'll use a Logistic Regression for a binary classification as machine
# learning model.

# We will train the classifier on all left visual vs auditory trials on MEG
data_picks = mne.pick_types(epochs.info, meg=True, exclude='bads')
X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
X = X[:, data_picks, :]  # take only MEG channels
y = epochs.events[:, 2]  # target: Audio left or right

clf = make_pipeline(StandardScaler(), LogisticRegression())

sl = _SearchLight(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(sl, X, y, cv=4, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('ROC AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

###############################################################################
# Generalization Across Time
# --------------------------
#
# This runs the analysis used in [1]_ and further detailed in [2]_
#
# The idea is to fit the models on each time instant and see how it
# generalizes to any other time point.

# define the Generalization Across Time (GAT) object
gl = _GeneralizationLight(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(gl, X, y, cv=4, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('ROC AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

# Plot the full matrix
fig, ax = plt.subplots(1, 1)
times = epochs.times
tlim = [times[0], times[-1], times[0], times[-1]]
vmin, vmax = 0., 1.
im = ax.imshow(scores, interpolation='nearest', origin='lower',
               extent=tlim, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time (GAT)')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
ax.set_xlim(tlim[:2])
ax.set_ylim(tlim[2:])
plt.colorbar(im, ax=ax)
plt.show()

###############################################################################
# Exercise
# --------
#  - Can you improve the performance using full epochs and a common spatial
#    pattern (CSP) used by most BCI systems?
#  - Explore other datasets from MNE (e.g. Face dataset from SPM to predict
#    Face vs. Scrambled)
#
# Have a look at the example
# :ref:`sphx_glr_auto_examples_decoding_plot_decoding_csp_space.py`
#
# References
# ==========
#
# .. [1] Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
#        and Stanislas Dehaene, "Two distinct dynamic modes subtend the
#        detection of unexpected sounds", PLOS ONE, 2013,
#        http://www.ncbi.nlm.nih.gov/pubmed/24475052
#
# .. [2] King & Dehaene (2014) 'Characterizing the dynamics of mental
#        representations: the temporal generalization method', Trends In
#        Cognitive Sciences, 18(4), 203-210.
#        http://www.ncbi.nlm.nih.gov/pubmed/24593982
