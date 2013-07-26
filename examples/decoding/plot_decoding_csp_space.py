"""
====================================================================
Decoding in sensor space data using the Common Spatial Pattern (CSP)
====================================================================

Decoding applied to MEG data in sensor space decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

print __doc__
import numpy as np

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1]
evoked = epochs.average()

###############################################################################
# Decoding in sensor space using a linear SVM

from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from mne.decoding import CSP

n_components = 3  # pick some components
svc = SVC(C=1, kernel='linear')
csp = CSP(n_components=n_components)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs.get_data()

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print "Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance)

# Or use much more convenient scikit-learn cross_val_score function using
# a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
print scores.mean()  # should match results above

# And using reuglarized csp with Ledoit-Wolf estimator
csp = CSP(n_components=n_components, reg='lws')
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
print scores.mean()  # should get better results than above

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
evoked.data = csp.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])
evoked.plot_topomap(times=[0, 1, 201, 202], ch_type='grad',
                    colorbar=False, size=1.5)
