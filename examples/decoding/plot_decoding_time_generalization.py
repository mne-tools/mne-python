"""
==========================================================
Decoding sensor space data with generalization across time
==========================================================

This example runs the analysis computed in:

Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
unexpected sounds", PLOS ONE, 2013,
http://www.ncbi.nlm.nih.gov/pubmed/24475052

The idea is to learn at one time instant and assess if the decoder
can predict accurately over time. In the second part of example, the decoder
will be trained on one set of conditions and evaluated on a second set of
conditions.
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

###############################################################################
# Preprocess data

data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.Raw(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
raw.filter(1, 30, method='fft')  # Band pass filtering signals
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
decim = 2  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=5e-12), decim=decim, verbose=False)

###############################################################################
# Generalization across time (GAT)

# Define events of interest
triggers = epochs.events[:, 2]
viz_vs_auditory = np.in1d(triggers, (1, 2)).astype(int)

gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1)

# fit and score
gat.fit(epochs, y=viz_vs_auditory)
gat.score(epochs, y=viz_vs_auditory)
gat.plot_diagonal()  # plot decoding across time (correspond to GAT diagonal)
gat.plot()  # plot full GAT matrix


###############################################################################
# Generalization across time and across conditions

# We will train the classifier on all left visual vs auditory trials
# and test on all right visual vs auditory trials

# In this case, because the test data is independent from the train data,
# we test the classifier of each fold and average the respective prediction:
gat.predict_mode = 'mean-prediction'

# For our left events, which ones are visual?
viz_vs_auditory_l = (triggers[np.in1d(triggers, (1, 3))] == 3).astype(int)
# To make scikit-learn happy, we converted the bool array to integers
# in the same line. This results in an array of zeros and ones:
print("The unique classes' labels are: %s" % np.unique(viz_vs_auditory_l))

gat.fit(epochs[('AudL', 'VisL')], y=viz_vs_auditory_l)

# For our right events, which ones are visual?
viz_vs_auditory_r = (triggers[np.in1d(triggers, (2, 4))] == 4).astype(int)

gat.score(epochs[('AudR', 'VisR')], y=viz_vs_auditory_r)
gat.plot()
