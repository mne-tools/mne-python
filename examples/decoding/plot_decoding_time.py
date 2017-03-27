"""
==========================
Decoding sensor space data
==========================

"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding.search_light import _SearchLight
from mne.decoding import cross_val_multiscore


print(__doc__)

# Preprocess data
data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1, 30, method='fft')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2}
decim = 2  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim, verbose=False)

# We will train the classifier on all left visual vs auditory trials
X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
y = epochs.events[:, 2]  # target: Audio left or right

# We will apply a logistic regression classifier on each time sample:
clf = make_pipeline(StandardScaler(), LogisticRegression())
sl = _SearchLight(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(sl, X, y, cv=4, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots(1)
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
plt.show()
