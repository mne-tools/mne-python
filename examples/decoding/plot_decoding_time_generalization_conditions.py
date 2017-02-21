"""
=========================================================================
Decoding sensor space data with generalization across time and conditions
=========================================================================

This example runs the analysis described in [1]_. It illustrates how one can
fit a linear classifier to identify a discriminatory topography at a given time
instant and subsequently assess whether this linear model can accurately
predict all of the time samples of a second set of conditions.

References
----------

.. [1] King & Dehaene (2014) 'Characterizing the dynamics of mental
       representations: the temporal generalization method', Trends In
       Cognitive Sciences, 18(4), 203-210. doi: 10.1016/j.tics.2014.01.002.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.decoding import GeneralizationAcrossTime

print(__doc__)

# Preprocess data
data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1, 30, method='fft')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
decim = 2  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim, verbose=False)

# We will train the classifier on all left visual vs auditory trials
# and test on all right visual vs auditory trials.

# In this case, because the test data is independent from the train data,
# we test the classifier of each fold and average the respective predictions.

# Define events of interest
triggers = epochs.events[:, 2]
viz_vs_auditory = np.in1d(triggers, (1, 2)).astype(int)

gat = GeneralizationAcrossTime(predict_mode='mean-prediction', n_jobs=1)

# For our left events, which ones are visual?
viz_vs_auditory_l = (triggers[np.in1d(triggers, (1, 3))] == 3).astype(int)
# To make scikit-learn happy, we converted the bool array to integers
# in the same line. This results in an array of zeros and ones:
print("The unique classes' labels are: %s" % np.unique(viz_vs_auditory_l))

gat.fit(epochs[('AudL', 'VisL')], y=viz_vs_auditory_l)

# For our right events, which ones are visual?
viz_vs_auditory_r = (triggers[np.in1d(triggers, (2, 4))] == 4).astype(int)

gat.score(epochs[('AudR', 'VisR')], y=viz_vs_auditory_r)
gat.plot(title="Temporal Generalization (visual vs auditory): left to right")
