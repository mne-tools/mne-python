"""
=================================
Decoding sensor space data (MVPA)
=================================

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
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

data_path = sample.data_path()

plt.close('all')

# sphinx_gallery_thumbnail_number = 4

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
tmin, tmax = -0.200, 0.500
event_id = dict(audio_left=1, visual_left=3)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the examlpe to run faster.
raw.filter(None, 20., fir_design='firwin')
events = mne.find_events(raw, 'STI 014')

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0.), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6), decim=10)
epochs.pick_types(meg=True, exclude='bads')

###############################################################################
# Temporal decoding
# -----------------
#
# We'll use a Logistic Regression for a binary classification as machine
# learning model.

# We will train the classifier on all left visual vs auditory trials on MEG

X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
y = epochs.events[:, 2]  # target: Audio left or right

clf = make_pipeline(StandardScaler(), LogisticRegression())

time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel
clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
time_decod.fit(X, y)

coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
evoked = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
evoked.plot_joint(times=np.arange(0., .500, .100), title='patterns')

###############################################################################
# Temporal Generalization
# -----------------------
#
# This runs the analysis used in [1]_ and further detailed in [2]_
#
# The idea is to fit the models on each time instant and see how it
# generalizes to any other time point.

# define the Temporal Generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')
plt.show()

# Plot the full matrix
fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal Generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
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
