import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP
from mne.channels import read_layout

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1., 4.
# frequency band for features extraction
fmin, fmax = 7., 30.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
layout = read_layout('EEG1005')

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names
raw.info['ch_names'] = [chn.strip('.') for chn in raw.info['ch_names']]

# Apply band-pass filter
raw.filter(fmin, fmax, method='iir')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False)
epochs_train = epochs.crop(tmin=1., tmax=2., copy=True)
labels = epochs.events[:, -1] - 2

# import a few transformer objects from mne.decoding
from mne.decoding import Scaler, PSDEstimator, ConcatenateChannels
# import a linear classifier from mne.decoding
from mne.decoding import LinearClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit

cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)

info = epochs.info
psd = PSDEstimator(sfreq=info['sfreq'], 
                   fmin=7, fmax=30, 
                   bandwidth=6, adaptive=True)
sc  = Scaler(info)
cat = ConcatenateChannels()
clf = LinearClassifier(LogisticRegression())
psd = PSDEstimator(sfreq=info['sfreq'], 
                   fmin=fmin, fmax=fmax, 
                   bandwidth=3)

pipeline = Pipeline((('psd', psd), ('scaler', sc),
                     ('cat', cat), ('linear', clf)))

pipeline.fit(epochs_train.get_data(), labels)
# get the patterns
patterns = pipeline.steps[-1][1].patterns_

# sampling of psd estimates at 1 Hz (1 sec epochs)
info['sfreq'] = 1
patterns = EvokedArray(patterns.reshape(info['nchan'], -1), info, tmin=fmin)
patterns.plot_topomap(layout=layout, scale_time=1, time_format='%01d Hz')

# computes some cross validated scores
scores = cross_val_score(pipeline, epochs_train.get_data(), 
                         labels, cv=cv, n_jobs=1)

print np.mean(scores)