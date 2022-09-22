# -*- coding: utf-8 -*-
"""
.. _ex-linear-patterns:

===============================================================
Linear classifier on sensor data with plot patterns and filters
===============================================================

Here decoding, a.k.a MVPA or supervised machine learning, is applied to M/EEG
data in sensor space. Fit a linear classifier with the LinearModel object
providing topographical patterns which are more neurophysiologically
interpretable :footcite:`HaufeEtAl2014` than the classifier filters (weight
vectors). The patterns explain how the MEG and EEG data were generated from
the discriminant neural sources which are extracted by the filters.
Note patterns/filters in MEG data are more similar than EEG data
because the noise is less spatially correlated in MEG than EEG.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Romain Trachel <trachelr@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD-3-Clause

# %%

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
sample_path = data_path / 'MEG' / 'sample'

# %%
# Set parameters
raw_fname = sample_path / 'sample_audvis_filt-0-40_raw.fif'
event_fname = sample_path / 'sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.4
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(.5, 25, fir_design='firwin')
events = mne.read_events(event_fname)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    decim=2, baseline=None, preload=True)
del raw

labels = epochs.events[:, -1]

# get MEG and EEG data
meg_epochs = epochs.copy().pick_types(meg=True, eeg=False)
meg_data = meg_epochs.get_data().reshape(len(labels), -1)

# %%
# Decoding in sensor space using a LogisticRegression classifier
# --------------------------------------------------------------

clf = LogisticRegression(solver='liblinear')  # liblinear is faster than lbfgs
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
    fig = evoked.plot_topomap()
    fig.suptitle(f'MEG {name}')

# %%
# Let's do the same on EEG data using a scikit-learn pipeline

X = epochs.pick_types(meg=False, eeg=True)
y = epochs.events[:, 2]

# Define a unique pipeline to sequentially:
clf = make_pipeline(
    Vectorizer(),                       # 1) vectorize across time and channels
    StandardScaler(),                   # 2) normalize features across trials
    LinearModel(                        # 3) fits a logistic regression
        LogisticRegression(solver='liblinear')
    )
)
clf.fit(X, y)

# Extract and plot patterns and filters
for name in ('patterns_', 'filters_'):
    # The `inverse_transform` parameter will call this method on any estimator
    # contained in the pipeline, in reverse order.
    coef = get_coef(clf, name, inverse_transform=True)
    evoked = EvokedArray(coef, epochs.info, tmin=epochs.tmin)
    fig = evoked.plot_topomap()
    fig.suptitle(f'EEG {name[:-1]}')

# %%
# References
# ----------
# .. footbibliography::
