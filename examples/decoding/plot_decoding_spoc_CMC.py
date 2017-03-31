"""
====================================
Continuous Target Decoding with SPoC
====================================

Source Power Comodulation (SPoC) [1] allows to identify the composition of
orthogonal spatial filters that maximally correlate with a continuous target.

SPoC can be seen as an extension of the CSP for continuous variables.

Here, SPoC is applied to decode the (continuous) fluctuation of an
electromyogram from MEG beta activity [2].

References
----------

.. [1] Dahne, S., et al (2014). SPoC: a novel framework for relating the
       amplitude of neuronal oscillations to behaviorally relevant parameters.
       NeuroImage, 86, 111-122.

.. [2] http://www.fieldtriptoolbox.org/tutorial/coherence

"""

# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import SPoC

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

# define parameters
fname = '/home/jrking/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname, preload=True)
raw.crop(50., 250.)  # crop for memory purposes

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGlft'])
emg.filter(20., None)

# Filter MEG data to focus on alpha band
picks = mne.pick_types(raw.info, meg=True, ref_meg=True, eeg=False)
raw.pick_channels(picks)
raw.filter(15., 30., picks=picks, method='iir')

# Build epochs as sliding windows over the continuous raw file
onsets = raw.first_samp + np.arange(0., raw.n_times, .400 * raw.info['sfreq'])
events = np.c_[onsets, np.zeros((len(onsets), 2))].astype(int)

meg_epochs = mne.Epochs(raw, events, tmin=0., tmax=.500, baseline=None)
emg_epochs = mne.Epochs(emg, events, tmin=0., tmax=.500, baseline=None)

# Prepare classification
X = meg_epochs.get_data()
y = np.log(emg_epochs.get_data().var(axis=2)[:, 0])  # target is EMG log power

X -= X.mean(axis=2, keepdims=True)  # XXX fix filter error

clf = make_pipeline(SPoC(n_components=4, log=True), Ridge())
cv = KFold(n_splits=2)

y_preds = cross_val_predict(clf, X, y, cv=cv)

fig, ax = plt.subplots(1, 1)
times = np.linspace(onsets[0], onsets[-1], len(onsets))[:len(y)]
ax.plot(times, y_preds, color='b', label='Predicted EMG')
ax.plot(times, y, color='r', label='True EMG')
ax.set_xlabel('Time (s)')
ax.set_ylabel('fT')
ax.set_title('SPoC MEG Predictions')
plt.legend()
plt.show()
