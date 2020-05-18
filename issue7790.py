#%% imports
import mne
import os

#%% set path
folder = '/home/th/temp'
data_file = 'data.mat'
raw_file = 'SubjectCMC.ds'

#%% read data
#data_no_info = mne.read_epochs_fieldtrip(os.path.join(folder, data_file), None)

#%% read data with info...
raw = mne.io.read_raw_ctf(os.path.join(folder, raw_file))
info = raw.info

data_info = mne.read_epochs_fieldtrip(os.path.join(folder, data_file), info)