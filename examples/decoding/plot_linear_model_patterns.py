"""
===============================================================
Linear classifier on sensor data with plot patterns and filters
===============================================================

Decoding, a.k.a MVPA or supervised machine learning applied to MEG and EEG
data in sensor space. Fit a linear classifier with the LinearModel object
providing topographical patterns which are more neurophysiologically
interpretable [1] than the classifier filters (weight vectors).
The patterns explain how the MEG and EEG data were generated from the
discriminant neural sources which are extracted by the filters.
Note patterns/filters in MEG data are more similar than EEG data
because the noise is less spatially correlated in MEG than EEG.

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    decim=4, baseline=None, preload=True)

labels = epochs.events[:, -1]

# get MEG and EEG data
meg_epochs = epochs.copy().pick_types(meg=True, eeg=False)
meg_data = meg_epochs.get_data().reshape(len(labels), -1)
eeg_epochs = epochs.copy().pick_types(meg=False, eeg=True)
eeg_data = eeg_epochs.get_data().reshape(len(labels), -1)

###############################################################################
# Decoding in sensor space using a LogisticRegression classifier

clf = LogisticRegression()
sc = StandardScaler()

# create a linear model with LogisticRegression
model = LinearModel(clf)

# fit the classifier on MEG data
X = sc.fit_transform(meg_data)
model.fit(X, labels)
# plot patterns and filters
model.plot_patterns(meg_epochs.info, title='MEG Patterns')
model.plot_filters(meg_epochs.info, title='MEG Filters')

# fit the classifier on EEG data
X = sc.fit_transform(eeg_data)
model.fit(X, labels)
# plot patterns and filters
model.plot_patterns(eeg_epochs.info, title='EEG Patterns')
model.plot_filters(eeg_epochs.info, title='EEG Filters')
