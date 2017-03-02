# -*- coding: utf-8 -*-
"""
===============================================================
Linear classifier on sensor data with plot patterns and filters
===============================================================

Decoding, a.k.a MVPA or supervised machine learning applied to MEG and EEG
data in sensor space. Fit a linear classifier with the LinearModel object
providing topographical patterns which are more neurophysiologically
interpretable [1]_ than the classifier filters (weight vectors).
The patterns explain how the MEG and EEG data were generated from the
discriminant neural sources which are extracted by the filters.
Note patterns/filters in MEG data are more similar than EEG data
because the noise is less spatially correlated in MEG than EEG.

References
----------

.. [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D.,
       Blankertz, B., & Bießmann, F. (2014). On the interpretation of
       weight vectors of linear models in multivariate neuroimaging.
       NeuroImage, 87, 96–110. doi:10.1016/j.neuroimage.2013.10.067
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <trachelr@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import Vectorizer, get_coef

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.4
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(.5, 25)
events = mne.read_events(event_fname)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    decim=4, baseline=None, preload=True)

labels = epochs.events[:, -1]

# get MEG and EEG data
meg_epochs = epochs.copy().pick_types(meg=True, eeg=False)
meg_data = meg_epochs.get_data().reshape(len(labels), -1)

###############################################################################
# Decoding in sensor space using a LogisticRegression classifier

clf = LogisticRegression()
scaler = StandardScaler()

# create a linear model with LogisticRegression
model = LinearModel(clf)

# fit the classifier on MEG data
X = scaler.fit_transform(meg_data)
model.fit(X, labels)

# Extract and plot spatial filters and spatial patterns
for name, coef in (('patterns', model.patterns_), ('filters', model.filters_)):
    # We fitted the linear model onto Z-scored data. To make the filters
    # interpretable, we must reverse this normalization step
    coef = scaler.inverse_transform([coef])[0]

    # The data was vectorized to fit a single model across all time points and
    # all channels. We thus reshape it:
    coef = coef.reshape(len(meg_epochs.ch_names), -1)

    # Plot
    evoked = EvokedArray(coef, meg_epochs.info, tmin=epochs.tmin)
    evoked.plot_topomap(title='MEG %s' % name)

###############################################################################
# Let's do the same on EEG data using a scikit-learn pipeline

X = epochs.pick_types(meg=False, eeg=True)
y = epochs.events[:, 2]

# Define a unique pipeline to sequentially:
clf = make_pipeline(
    Vectorizer(),                       # 1) vectorize across time and channels
    StandardScaler(),                   # 2) normalize features across trials
    LinearModel(LogisticRegression()))  # 3) fits a logistic regression
clf.fit(X, y)

# Extract and plot patterns and filters
for name in ('patterns_', 'filters_'):
    # The `inverse_transform` parameter will call this method on any estimator
    # contained in the pipeline, in reverse order.
    coef = get_coef(clf, name, inverse_transform=True)
    evoked = EvokedArray(coef, epochs.info, tmin=epochs.tmin)
    evoked.plot_topomap(title='EEG %s' % name[:-1])
