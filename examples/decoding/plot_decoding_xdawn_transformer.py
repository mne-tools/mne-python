"""
====================================================================
Decoding in sensor space data using the Xdawn Transformer
====================================================================

Decoding applied to MEG data in sensor space decomposed using Xdawn.
Here the classifier is applied to features extracted on Xdawn filtered signals.

"""
# Authors: Asish Panda <asishrocks95@gmail.com>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)
import numpy as np

from sklearn.cross_validation import cross_val_score, ShuffleSplit  # noqa
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC  # noqa
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from mne import io, pick_types, read_events, Epochs
from mne.viz import plot_topomap
from mne.datasets import sample
from mne.decoding import Vectorizer
from mne.preprocessing import XdawnTransformer

import matplotlib.pyplot as plt

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(2, None, method='iir')
events = read_events(event_fname)

picks = pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                picks=picks, baseline=None, preload=True, verbose=False)

X = epochs.get_data()
y = label_binarize(epochs.events[:, 2], classes=[1, 3]).ravel()

clf = make_pipeline(XdawnTransformer(n_components=2),
                    Vectorizer(),
                    StandardScaler(),
                    SVC(C=1, kernel='linear'))

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(y), 10, test_size=0.2, random_state=42)

scores = cross_val_score(clf, X, y, cv=cv)

class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

###############################################################################
# plot Xdawn patterns estimated on full data for visualization

xdawn = XdawnTransformer(n_components=2)
xdawn.fit(X, y)
data = xdawn.patterns_
fig, axes = plt.subplots(1, 4)
for idx in range(4):
    plot_topomap(data[idx], epochs.info, axes=axes[idx], show=False)
fig.suptitle('Xdawn patterns')
fig.tight_layout()
plt.show()
