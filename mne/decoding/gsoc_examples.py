"""
============
Applies PCA
============
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.decoding.gsoc import (UnsupervisedSpatialFilter, XdawnTransformer,
                               Vectorizer)
from mne.decoding import EpochsVectorizer

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold

print(__doc__)

# Preprocess data
data_path = sample.data_path()

# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 20, method='iir')
events = mne.read_events(event_fname)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                picks=picks, baseline=None, preload=True,
                add_eeg_ref=False, verbose=False)

# Example of a PCA spatial filter
spatial_filter = UnsupervisedSpatialFilter(PCA(10))
X = spatial_filter.fit_transform(epochs)
print(X.shape)  # X has components and not channels

# pipeline for XdawnTransformer
X, y = EpochsVectorizer().fit_transform(epochs)
cv = StratifiedKFold(y=y, n_folds=10, shuffle=True, random_state=42)
clf = make_pipeline(XdawnTransformer(n_chan=epochs.info['nchan'],
                                     n_components=1),
                    Vectorizer(), LogisticRegression())

preds = np.empty(len(y))
for train, test in cv:
    clf.fit(X[train], y[train])
    preds[test] = clf.predict(X[test])

# pipeline for UnsupervisedTransformer
clf = make_pipeline(UnsupervisedSpatialFilter(PCA(5), 
                                              n_chan=epochs.info['nchan']),
                    Vectorizer(), LogisticRegression())
preds = np.empty(len(y))
for train, test in cv:
    clf.fit(X[train], y[train])
    preds[test] = clf.predict(X[test])
