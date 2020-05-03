
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






# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_raw.fif')
# raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=60)





       
# event_id = 998
# eog_events = mne.preprocessing.find_eog_events(raw, event_id)

# # Read epochs
# picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True,
#                        exclude='bads')
# tmin, tmax = -0.2, 0.2
# epochs = mne.Epochs(raw, eog_events, event_id, tmin, tmax, picks=picks)
# data = epochs.get_data()

# print("Number of detected EOG artifacts : %d" % len(data))

# plt.plot(1e3 * epochs.times, np.squeeze(data).T)
# plt.xlabel('Times (ms)')
# plt.ylabel('EOG (µV)')
# plt.show()