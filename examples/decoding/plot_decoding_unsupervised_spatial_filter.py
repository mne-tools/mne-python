"""
==================================================================
Analysis of evoked response using ICA and PCA reduction techniques
==================================================================

This example computes PCA and ICA of epochs data. Then the evoked response
for any single event is taken for visual comparison between the two.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Asish Panda <asishrocks95@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA, FastICA

print(__doc__)
spatial_filter = make_pipeline(UnsupervisedSpatialFilter(PCA(10)),

# Preprocess data
data_path = sample.data_path()

# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 20)
events = mne.read_events(event_fname)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)

X = epochs.get_data()

pca = UnsupervisedSpatialFilter(PCA(10))
pca_data = pca.fit_transform(X)
ev = mne.EvokedArray(np.mean(X, axis=0), epochs.info, tmin=tmin)
ev.data = np.average(pca_data, axis=0)
ev.plot(show=False, window_title='PCA')

ica = UnsupervisedSpatialFilter(FastICA(10))
ica_data = ica.fit_transform(X)
ev1 = mne.EvokedArray(np.mean(X, axis=0), epochs.info, tmin=tmin)
ev1.data = ica_data
ev1.plot(show=False, window_title='ICA')

plt.show()
