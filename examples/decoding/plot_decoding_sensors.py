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
from sklearn.cross_validation import cross_val_score, ShuffleSplit

clf = SVC(C=1, kernel='linear')
# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(X), 10, test_size=0.2)

scores = np.empty(n_times)
std_scores = np.empty(n_times)

for t in xrange(n_times):
    Xt = X[:, :, t]
    # Standardize features
    Xt -= Xt.mean(axis=0)
    Xt /= Xt.std(axis=0)
    # Run cross-validation
    # Note : for sklearn the Xt matrix should be 2d (n_samples x n_features)
    scores_t = cross_val_score(clf, Xt, y, cv=cv, n_jobs=1)
    scores[t] = scores_t.mean()
    std_scores[t] = scores_t.std()

times = 1e3 * epochs.times
scores *= 100  # make it percentage
std_scores *= 100
pl.plot(times, scores, label="Classif. score")
pl.axhline(50, color='k', linestyle='--', label="Chance level")
pl.axvline(0, color='r', label='stim onset')
pl.legend()
hyp_limits = (scores - std_scores, scores + std_scores)
pl.fill_between(times, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
pl.xlabel('Times (ms)')
pl.ylabel('CV classification score (% correct)')
pl.ylim([30, 100])
pl.title('Sensor space decoding')
pl.show()
