"""
============
Applies PCA
============
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.decoding.gsoc import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA

print(__doc__)

# Preprocess data
data_path = sample.data_path()

# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, decim=2)

# Example of a PCA spatial filter
spatial_filter = UnsupervisedSpatialFilter(PCA(10))
X = spatial_filter.fit_transform(epochs)
print(X.shape)  # X has components and not channels
