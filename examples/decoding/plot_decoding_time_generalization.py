"""
==========================================================
Decoding sensor space data with generalization across time
==========================================================

This example runs the analysis computed in:

Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
unexpected sounds", PLOS ONE, 2013

The idea is to learn at one time instant and assess if the decoder
can predict accurately over time.
"""

# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
import mne
from mne.datasets import sample
from mne.decoding import GeneralizationAcrossTime

print(__doc__)

# --------------------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------------------
data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.Raw(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1, 30, method='iir')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
decim = 3  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim)
# Define events of interest
y_vis_audio = epochs.events[:, 2] <= 2
y_left_right = np.mod(epochs.events[:, 2], 2)


# ----------------------------------------------------------------------------
# GENERALIZATION ACROSS TIME (GAT)
# ----------------------------------------------------------------------------
# The function implements the method used in:
# King, Gramfort, Schurger, Naccache & Dehaene, "Two distinct dynamic modes
# subtend the detection of unexpected sounds", PLOS ONE, 2013
gat = GeneralizationAcrossTime()
gat.fit(epochs, y=y_vis_audio)
gat.score(epochs, y=y_vis_audio)
gat.plot_diagonal()  # plot decoding across time (correspond to GAT diagonal)
gat.plot()  # plot full GAT matrix


# ----------------------------------------------------------------------------
# GENERALIZATION ACROSS TIME AND ACROSS CONDITIONS
# ----------------------------------------------------------------------------
# As proposed in King & Dehaene (2014) 'Characterizing the dynamics of mental
# representations: the temporal generalization method', Trends In Cognitive
# Sciences, 18(4), 203-210.
gat = GeneralizationAcrossTime()
# Train on visual versus audio: left stimuli only.
gat.fit(epochs[y_left_right==1], y=y_vis_audio[y_left_right==1])
# Test on visual versus audio: right stimuli only.
# In this case, because the test data is independent, we test the
# classifier of each folds and average their respective prediction:
gat.score(epochs[y_left_right==0], y=y_vis_audio[y_left_right==0], independent=True)
gat.plot()

