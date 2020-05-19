#%% imports
import mne
import os

#%% set path
folder = '/home/th/temp'
fname = os.path.join(mne.datasets.testing.data_path(),
                     'fieldtrip',
                     'old_version.mat')
raw_file = 'SubjectCMC.ds'

#%% read data
data_no_info = mne.read_epochs_fieldtrip(os.path.join(folder, fname), None)

#%% read data with info...
raw = mne.io.read_raw_ctf(os.path.join(folder, raw_file))
info = raw.info

data_info = mne.read_epochs_fieldtrip(os.path.join(folder, fname), info)