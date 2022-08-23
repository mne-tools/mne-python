# -*- coding: utf-8 -*-
"""
.. _ex-linear-sensor-decoding:

=========================================================================
Decoding sensor space data with generalization across time and conditions
=========================================================================

This example runs the analysis described in :footcite:`KingDehaene2014`. It
illustrates how one can
fit a linear classifier to identify a discriminatory topography at a given time
instant and subsequently assess whether this linear model can accurately
predict all of the time samples of a second set of conditions.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import GeneralizingEstimator

print(__doc__)

# Preprocess data
data_path = sample.data_path()
# Load and filter data, set up epochs
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
events_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1., 30., fir_design='firwin')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}
tmin = -0.050
tmax = 0.400
# decimate to make the example faster to run, but then use verbose='error' in
# the Epochs constructor to suppress warning about decimation causing aliasing
decim = 2
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    proj=True, picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim, verbose='error')

# %%
# We will train the classifier on all left visual vs auditory trials
# and test on all right visual vs auditory trials.
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear')  # liblinear is faster than lbfgs
)
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=None,
                                 verbose=True)

# Fit classifiers on the epochs where the stimulus was presented to the left.
# Note that the experimental condition y indicates auditory or visual
time_gen.fit(X=epochs['Left'].get_data(),
             y=epochs['Left'].events[:, 2] > 2)

# %%
# Score on the epochs where the stimulus was presented to the right.
scores = time_gen.score(X=epochs['Right'].get_data(),
                        y=epochs['Right'].events[:, 2] > 2)

# %%
# Plot
fig, ax = plt.subplots(1)
im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',
                extent=epochs.times[[0, -1, 0, -1]])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time and condition')
plt.colorbar(im, ax=ax)
plt.show()

##############################################################################
# References
# ----------
# .. footbibliography::
