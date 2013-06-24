"""
==========================
Decoding sensor space data
==========================

Decoding, a.k.a MVPA or supervised machine learning applied to MEG
data in sensor space. Here the classifier is applied to every time
point.
"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__
import pylab as pl
import numpy as np

import mne
from mne import fiff
from mne.datasets import sample
from mne.realtime.classifier import Scaler, ConcatenateChannels

data_path = sample.data_path()

pl.close('all')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

epochs_list = [epochs[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochs_list)

###############################################################################
# Decoding in sensor space using a linear SVM
n_times = len(epochs.times)
# Take only the data channels (here the gradiometers)
data_picks = fiff.pick_types(epochs.info, meg=True, exclude='bads')
# Make arrays X and y such that :
# X is 3d with X.shape[0] is the total number of epochs to classify
# y is filled with integers coding for the class to predict
# We must have X.shape[0] equal to y.shape[0]
X = [e.get_data()[:, data_picks, :] for e in epochs_list]
y = [k * np.ones(len(this_X)) for k, this_X in enumerate(X)]
X = np.concatenate(X)
y = np.concatenate(y)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit

cv = ShuffleSplit(len(y), 10, test_size=0.2)

for train_idx, test_idx in cv:
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = Scaler(epochs.info)
    concatenator = ConcatenateChannels()
    clf = SVC(C=1, kernel='linear')

    scaled_classifier = Pipeline([('scl', scaler), ('concat', concatenator),
                                  ('svm', clf)])

    scaled_classifier = scaled_classifier.fit(epochs[train_idx], y_train)
    score = scaled_classifier.score(epochs[test_idx], y_test)*100

    print score
