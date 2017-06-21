"""
==================
Find ECG artifacts
==================

Locate QRS component of ECG.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)

event_id = 999
ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, event_id,
                                                     ch_name='MEG 1531')

# Read epochs
picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False,
                       include=['MEG 1531'], exclude='bads')
tmin, tmax = -0.1, 0.1
epochs = mne.Epochs(raw, ecg_events, event_id, tmin, tmax, picks=picks,
                    proj=False)
data = epochs.get_data()

print("Number of detected ECG artifacts : %d" % len(data))

###############################################################################
# Plot ECG artifacts
plt.plot(1e3 * epochs.times, np.squeeze(data).T)
plt.xlabel('Times (ms)')
plt.ylabel('ECG')
plt.show()
