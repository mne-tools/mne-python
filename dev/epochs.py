
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
raw = raw.pick_types(eog=True, eeg=True, meg=False, stim=True)

raw.plot()

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}

epochs = mne.Epochs(raw, events, picks = ['eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
epochs.plot(picks= ['eeg', 'eog'], n_epochs=10)


print('Regressing out eye movements')
epochs = mne.preprocessing.regress_eog(epochs)
epochs.plot(picks= ['eeg', 'eog'], n_epochs=10)




