"""
===============================================================
Decoding sensor space data using a linear classifier 
Patterns and filters visualization of the linear classifier
===============================================================

Decoding, a.k.a MVPA or supervised machine learning applied to MEG
data in sensor space. Here the classifier is applied to a single time point.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)

# crop the epochs to a single time point
epochs.crop(tmin=.095, tmax=.105)
epochs_list = [epochs[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochs_list)

###############################################################################
# Decoding in sensor space using a linear classifier

# Make arrays X and y such that :
# X is 2d with X.shape[0] is the total number of epochs to classify
# y is filled with integers coding for the class to predict
# We must have X.shape[0] equal to y.shape[0]
X = [e.get_data()[:,:,0] for e in epochs_list]
y = [k * np.ones(len(this_X)) for k, this_X in enumerate(X)]
X = np.concatenate(X)
y = np.concatenate(y)

# import a linear classifier from mne.decoding
from mne.decoding import LinearClassifier
from sklearn.preprocessing import StandardScaler

clf = LinearClassifier()
sc = StandardScaler()

X = sc.fit_transform(X)
clf.fit(X, y)

# create patterns as an evoked array and plot it
patterns = mne.EvokedArray(clf.patterns_, epochs.info, tmin=epochs.times[0])
patterns.plot_topomap(times=epochs.times[0])
# create filters as an evoked array and plot it
filters = mne.EvokedArray(clf.filters_.T, epochs.info, tmin=epochs.times[0])
filters.plot_topomap(times=epochs.times[0])

from sklearn.cross_validation import cross_val_score, ShuffleSplit
cv = ShuffleSplit(len(y), 10, test_size=0.2)
# computes some cross validated scores
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
print np.mean(scores)
