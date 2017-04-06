"""
=============================================================================
Decoding sensor space data with generalization across time in alpha frequency
=============================================================================

This example runs a variant of the temporal generalization analysis described
in [1]_ in the power domain (alpha band). The motor Imagery dataset from the
BCI competition IV-2a is used [2]_. Data are filtered in the alpha frequency
band, and log-power is estimated using a sliding window befor being fed to the
GeneralizationAcrossTime object.

We observe a constant generalization in the patter during the duration of the
movement, suggesting a maintained induced activity during the task.

References
----------

.. [1] King & Dehaene (2014) 'Characterizing the dynamics of mental
       representations: the temporal generalization method', Trends In
       Cognitive Sciences, 18(4), 203-210. doi: 10.1016/j.tics.2014.01.002.
.. [2] Tangermann et al (2012). 'Review of the BCI competition IV.' Frontiers
       in neuroscience, vol.6 , p55. doi: 10.3389/fnins.2012.00055
"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import bnci
from mne.decoding import GeneralizationAcrossTime

from sklearn.metrics import roc_auc_score
print(__doc__)

# we use the dataset 001-2014, a 4 class motor imagery dataset. this dataset
# contains 2 sessions, we use the first for training and the second
# for evaluation
raws, event_id = bnci.load_data(subject=1, dataset='001-2014', verbose=False)

event_id = {'right hand': 2, 'feet': 3}

tmin = 0
tmax = 7.5

for raw in raws:
    raw.filter(8, 15)

# training data
# find events
events = mne.find_events(raws[0])

picks = mne.pick_types(raws[0].info, eeg=True)

epochs_tr = mne.Epochs(raws[0], events, event_id, tmin=tmin, tmax=tmax,
                       add_eeg_ref=False, proj=False,
                       picks=picks, baseline=None, preload=True,
                       reject=None, verbose=False)

# Estimates log power using a sliding window
window = 0.5
step = 0.1
sfreq = epochs_tr.info['sfreq']
X = epochs_tr.get_data()
times = np.arange(tmin, tmax - window,  step)
powers = np.zeros((X.shape[0], X.shape[1], len(times)))
for ii, tstart in enumerate(times):
    tend = tstart + window
    sl = slice(int(tstart * sfreq), int(tend * sfreq))
    powers[..., ii] = np.log(X[..., sl].var(2))


powers_tr = mne.EpochsArray(powers, epochs_tr.info, events=epochs_tr.events)
powers_tr.times = times
y_tr = powers_tr.events[:, -1]

# test data
events = mne.find_events(raws[1])

epochs_te = mne.Epochs(raws[1], events, event_id, tmin=tmin, tmax=tmax,
                       add_eeg_ref=False, proj=False,
                       picks=picks, baseline=None, preload=True,
                       reject=None, verbose=False)

X = epochs_te.get_data()
powers = np.zeros((X.shape[0], X.shape[1], len(times)))
for ii, tstart in enumerate(times):
    tend = tstart + window
    sl = slice(int(tstart * sfreq), int(tend * sfreq))
    powers[..., ii] = np.log(X[..., sl].var(2))

powers_te = mne.EpochsArray(powers, epochs_te.info, events=epochs_te.events)
powers_te.times = times
y_te = powers_te.events[:, -1]


def scorer(true_labels, probas):
    """roc auc scorer"""
    return roc_auc_score(true_labels == 3, probas[:, 1])

gat = GeneralizationAcrossTime(predict_mode='mean-prediction',
                               predict_method='predict_proba',
                               scorer=scorer, n_jobs=1)

gat.fit(powers_tr, y=y_tr)
gat.score(powers_te, y=y_te)

gat.plot(title="Time Generalization in alpha band", vmin=0, vmax=1)
