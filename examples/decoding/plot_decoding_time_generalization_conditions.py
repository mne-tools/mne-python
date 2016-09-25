"""
=========================================================================
Decoding sensor space data with generalization across time and conditions
=========================================================================

This example runs the analysis computed in:

Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
unexpected sounds", PLOS ONE, 2013,
http://www.ncbi.nlm.nih.gov/pubmed/24475052

King & Dehaene (2014) 'Characterizing the dynamics of mental
representations: the temporal generalization method', Trends In Cognitive
Sciences, 18(4), 203-210.
http://www.ncbi.nlm.nih.gov/pubmed/24593982

The idea is to learn at one time instant and assess if the decoder
can predict accurately over time and on a second set of conditions.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.decoding.search_light import _GeneralizationLight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print(__doc__)

# Preprocess data
data_path = sample.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag')  # Pick magnetometers only
events = mne.read_events(events_fname)
event_id = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
decim = 2  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, -0.050, 0.400, proj=True,
                    picks=picks, baseline=None, preload=True,
                    decim=decim, verbose=False)

# We will train the classifier on all left visual vs auditory trials
# and test on all right visual vs auditory trials.

# In this case, because the test data is independent from the train data,
# we do not need a cross validation.

# Define events of interest
triggers = epochs.events[:, 2]


# Each estimator fitted at each time point is an independent Scikit-Learn
# pipeline with a ``fit``, and a ``score`` method.
gat = _GeneralizationLight(
    make_pipeline(StandardScaler(), LogisticRegression()),
    n_jobs=1)

# Fit: for our left events, which ones are visual?
X = epochs[('AudL', 'VisL')].get_data()
y = triggers[np.in1d(triggers, (1, 3))] == 3
gat.fit(X, y)

# Generalize: for our right events, which ones are visual?
X = epochs[('AudR', 'VisR')].get_data()
y = triggers[np.in1d(triggers, (2, 4))] == 4
score = gat.score(X, y)

# Plot temporal generalization accuracies.
extent = epochs.times[[0, -1, 0, -1]]
fig, ax = plt.subplots(1)
im = ax.matshow(score, origin='lower', cmap='RdBu_r', vmin=0., vmax=1.,
                extent=extent)
ticks = np.arange(0., .401, .100)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks)
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im)
plt.show()
