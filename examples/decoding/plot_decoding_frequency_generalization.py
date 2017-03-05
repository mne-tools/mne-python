"""
=========================================================================
Decoding sensor space data with generalization across frequency
=========================================================================

This example runs a variant of the temporal generalization analysis described
in [1]_ in the frequency domain. A motor Imagery dataset is used. Epochs are
transformed in the frequency domain with a PSD estimation and the
generalization across frequency is studied.

References
----------

.. [1] King & Dehaene (2014) 'Characterizing the dynamics of mental
       representations: the temporal generalization method', Trends In
       Cognitive Sciences, 18(4), 203-210. doi: 10.1016/j.tics.2014.01.002.
"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import bnci
from mne.decoding import GeneralizationAcrossTime, PSDEstimator

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
print(__doc__)

# we use the dataset 001-2014, a 4 class motor imagery dataset. this dataset
# contains 2 sessions, we use the first for training and the second
# for evaluation
raws, event_id = bnci.load_data(subject=1, dataset='001-2014', verbose=False)

event_id = {'right hand': 2, 'feet': 3}
fmin = 1
fmax = 40

# training data
# find events
events = mne.find_events(raws[0])

picks = mne.pick_types(raws[0].info, eeg=True)

epochs_tr = mne.Epochs(raws[0], events, event_id, 3.5, 5.5, proj=False,
                       add_eeg_ref=False,
                       picks=picks, baseline=None, preload=True,
                       reject=None, verbose=False)

# PSD are estimated and injected in an epoch object to be compatible with
# the GeneralizationAcrossTime object. Instead of time, frequency will be used.
psd = PSDEstimator(sfreq=epochs_tr.info['sfreq'], fmin=fmin, fmax=fmax)
powers = psd.transform(epochs_tr.get_data())

epochs_tr._data = powers
epochs_tr.times = np.linspace(fmin, fmax, powers.shape[-1])
y_tr = epochs_tr.events[:, -1]

# test data
events = mne.find_events(raws[1])

epochs_te = mne.Epochs(raws[1], events, event_id, 3.5, 5.5, proj=False,
                       add_eeg_ref=False,
                       picks=picks, baseline=None, preload=True,
                       reject=None, verbose=False)

powers = psd.transform(epochs_te.get_data())
epochs_te._data = powers
epochs_te.times = np.linspace(fmin, fmax, powers.shape[-1])
y_te = epochs_te.events[:, -1]


def scorer(true_labels, probas):
    """roc auc scorer"""
    return roc_auc_score(true_labels == 3, probas[:, 1])

gat = GeneralizationAcrossTime(predict_mode='mean-prediction',
                               predict_method='predict_proba',
                               scorer=scorer, n_jobs=1)

gat.fit(epochs_tr, y=y_tr)
gat.score(epochs_te, y=y_te)

gat.plot(title="Frequency Generalization", show=False)
plt.xlabel('Training Frequency (Hz)')
plt.ylabel('Test Frequency (Hz)')
plt.show()
