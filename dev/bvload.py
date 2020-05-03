import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
import os

exp = 'P3'
data_dir = os.path.join('dev')
sub = '002'
session = 'PassiveWet'
scalings = dict(eeg=20e-6, eog=20e-3)
filename = sub + '_' + exp + '_' + session + '.vhdr'

fname = os.path.join(data_dir, filename)
#load .vhdr files from brain vision recorder
raw = io.read_raw_brainvision(fname,
          eog=('HEOG', 'VEOG'),
          preload=True)

sfreq = raw.info['sfreq']

#Epoching
events, event_id = mne.events_from_annotations(raw)
print(events)
print(event_id)

#epoch timing
epoch_time=(-.2,1)
tmin=epoch_time[0]
tmax=epoch_time[1]
baseline=(-.2,0)
#artifact rejection
rej_thresh_uV=200
rej_thresh = rej_thresh_uV*1e-6

#Construct events - Main function from MNE
epochs = mne.Epochs(raw, events=events, event_id=event_id,
              tmin=tmin, tmax=tmax, baseline=baseline,
              preload=True,reject={'eeg':rej_thresh},
              verbose=False)

epochs.plot(picks = ['eeg', 'eog'], n_epochs=10, scalings=scalings)

epochs = mne.preprocessing.regress_eog(epochs)

epochs.plot(picks = ['eeg', 'eog'], n_epochs=10, scalings=scalings)





