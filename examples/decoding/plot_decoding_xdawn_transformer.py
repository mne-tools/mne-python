"""
====================================================================
Minimal example for using scikit-learn pipeline
====================================================================

This example creates a scikit-learn pipeline with mne transformers.

"""
# Authors: Asish Panda <asishrocks95@gmail.com>
#
# License: BSD (3-clause)

from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

from mne import io, pick_types, read_events, Epochs
from mne.datasets import sample
from mne.decoding.xdawn import XdawnTransformer

data_path = sample.data_path()


raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 20, method='iir')
events = read_events(event_fname)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                picks=picks, baseline=None, preload=True,
                add_eeg_ref=False, verbose=False)

X = epochs.get_data()
y = label_binarize(epochs.events[:, 2], classes=[1, 3]).ravel()

clf = make_pipeline(XdawnTransformer(n_components=3),
                    Vectorizer(),
                    LogisticRegression())
score = cross_val_score(clf, X, y, cv=5)
print(score)
