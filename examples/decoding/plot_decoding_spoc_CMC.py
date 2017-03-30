"""
====================================
Continuous Target Decoding with SPoC
====================================



"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import SPoC

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

# define parameters
# ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/tutorial/SubjectCMC.zip
fname = '/home/kirch/Documents/Data/fieldtrip/CMC/SubjectCMC.ds'
fname = '/home/jrking/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname, preload=True)
raw.crop(50., 250.)  # crop for memory purposes

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGlft'])
emg.filter(20., None)

# Filter MEG data to focus on alpha band
raw.pick_types(meg=True, eeg=False)
raw.filter(15., 30., method='iir')

# Build epochs as sliding windows over the continuous raw file
onsets = raw.first_samp + np.arange(0., raw.n_times, .400 * raw.info['sfreq'])
events = np.c_[onsets, np.zeros((len(onsets), 2))].astype(int)

meg_epochs = mne.Epochs(raw, events, tmin=0., tmax=.500, baseline=None)
emg_epochs = mne.Epochs(emg, events, tmin=0., tmax=.500, baseline=None)

# Prepare classification
X = meg_epochs.get_data()
y = np.log(emg_epochs.get_data().var(axis=2)[:, 0])  # log power of EMG*

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
