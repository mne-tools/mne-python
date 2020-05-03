import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample

data_dir = '/Users/kylemathewson/data/'
exp = 'P3'
subs = ['001','002','004','005']
sub = subs[1];
sessions = ['PassiveWet']
session = sessions[0]
event_id = {'Target': 1, 'Standard': 2}

fname = data_dir + exp + '/' + sub + '_' + exp + '_' + session + '.vhdr'
#load .vhdr files from brain vision recorder
raw = io.read_raw_brainvision(fname,
          eog=('HEOG', 'VEOG'),
          preload=True)
raw.plot(scalings=dict(eeg=20e-6, eog=20e-3))


sfreq = raw.info['sfreq']

# # #Filtering
# # print('Rerefering to average mastoid')
# # raw = mastoidReref(raw)

# #Epoching
# events, event_id = mne.events_from_annotations(raw)
# print(events)
# print(event_id)

# color = {1: 'red', 2: 'black'}

# #artifact rejection
# rej_thresh_uV=200
# rej_thresh = rej_thresh_uV*1e-6

# epoch_time=(-.2,1)
# tmin=epoch_time[0]
# tmax=epoch_time[1]
# baseline=(-.2,0)

# #Construct events - Main function from MNE
# epochs = mne.Epochs(raw, events=events, event_id=event_id,
#               tmin=tmin, tmax=tmax, baseline=baseline,
#               preload=True,reject={'eeg':rej_thresh},
#               verbose=False)

# print('Remaining Trials: ' + str(len(epochs)))

# #Gratton eye movement correction procedure on epochs
# print('Epochs Eye Movement Correct')
# epochs = mne.preprocessing.regress_eog(epochs)

# ## plot ERP at each electrode
# evoked_dict = {event_names[0]:epochs[event_names[0]].average(),
#                           event_names[1]:epochs[event_names[1]].average()}





