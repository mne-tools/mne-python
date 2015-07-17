"""
====================================================================
Decoding in sensor space data using the linear models
====================================================================

Decoding applied to MEG data in sensor space using a linear classifier
Here the model coefficient are interpreted using [1].

[1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D.,
Blankertz, B., & Bießmann, F. (2014). On the interpretation of
weight vectors of linear models in multivariate neuroimaging.
NeuroImage, 87, 96–110. doi:10.1016/j.neuroimage.2013.10.067
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, aud_r=3)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)
# get class labels
labels = epochs.events[:, -1]

import sklearn.linear_model as lm
from mne.decoding import compute_patterns

# computes patterns estimated with a ridge classifier
ridge = lm.RidgeClassifier()
ridge_patterns = compute_patterns(epochs, ridge)
ridge_patterns.plot_topomap(title='classifier patterns for aud_l vs aud_r')

# compare patterns topography with a simple t-test
import scipy.stats as scistats
l_label = epochs.events[:, -1] == event_id['aud_l']
r_label = epochs.events[:, -1] == event_id['aud_r']
# computes ttest
t_val, _ = scistats.ttest_ind(epochs.get_data()[l_label],
                              epochs.get_data()[r_label], axis=0)

evoked = mne.EvokedArray(t_val, epochs.info, tmin=epochs.tmin)
evoked.plot_topomap(title='t-values for aud_l vs aud_r')

# now start decoding using a cross validation
from sklearn.cross_validation import ShuffleSplit, cross_val_score
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
epochs_data = epochs.get_data().reshape(len(labels), -1)
scores = cross_val_score(ridge, epochs_data, labels, cv=cv, n_jobs=1)
print(scores.mean())  # should match results above
