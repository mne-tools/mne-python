import pylab as pl
import numpy as np

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path()

pl.close('all')

raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=True, eog=True,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

epochs_list = [epochs[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochs_list)

###############################################################################
# Decoding in sensor space using a linear SVM
n_times = len(epochs.times)
# Take only the data channels (here the gradiometers)
data_picks = fiff.pick_types(epochs.info, meg='grad', exclude='bads')

X = [e.get_data()[:, data_picks, :] for e in epochs_list]
X = np.concatenate(X)

# X_reshaped = X.transpose((0,1,2)).reshape(X.shape[0], X.shape[1]*X.shape[2])
X_reshaped = X.reshape(X.shape[0], X.shape[2]*X.shape[1])

pl.plot(X[0, 0, 0:106])
pl.plot(X_reshaped[0, 0:106])
