"""
====================================================================
Decoding in sensor space data using Riemannian Geometry and XDAWN
====================================================================

Decoding applied to MEG data in sensor space decomposed using Xdawn
And Riemannian Geometry.
After spatial filtering, covariances matrices are estimated and
classified by the MDM algorithm (Nearest centroid).

4 Xdawn spatial patterns (2 for each class) are displayed, as per the
two mean-covariance matrices used by the classification algorithm.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <romain.trachel@inria.fr>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from pylab import plt
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM

import mne
from mne import io
from mne.datasets import sample

from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import ShuffleSplit

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1]
evoked = epochs.average()

###############################################################################
# Decoding in sensor space using a MDM


n_components = 3  # pick some components

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs.get_data()


clf = Pipeline([('COV', XdawnCovariances(n_components)), ('MDM', MDM())])

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf.fit(epochs_data[train_idx], y_train)
    scores.append(clf.score(epochs_data[test_idx], y_test))

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# spatial patterns
xd = XdawnCovariances(n_components)
Cov = xd.fit_transform(epochs_data, labels)

evoked.data = xd.Xd._patterns.T
evoked.times = np.arange(evoked.data.shape[0])
evoked.plot_topomap(times=[0, 1, n_components, n_components + 1],
                    ch_type='grad', colorbar=False, size=1.5)

# prototyped covariance matrices
mdm = MDM()
mdm.fit(Cov, labels)
fig, axe = plt.subplots(1, 2)
axe[0].matshow(mdm.covmeans[0])
axe[0].set_title('Class 1 covariance matrix')
axe[1].matshow(mdm.covmeans[1])
axe[1].set_title('Class 2 covariance matrix')
plt.show()
