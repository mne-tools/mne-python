"""
==================================================
Whiten evoked data using a noise covariance matrix
==================================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
# raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

###############################################################################
# Set epochs parameters
event_id = 1
tmin = -0.2
tmax = 0.5

###############################################################################
# Create evoked data

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.find_events(raw)

# pick EEG channels - bad channels (modify to your needs)
exclude = raw.info['bads'] + ['EEG 053'] # bads + 1 more
picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                        exclude=exclude)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=40e-6, eog=150e-6))
evoked = epochs.average() # average epochs and get an Evoked dataset.

cov = mne.Covariance(cov_fname)

# Whiten data
W, ch_names = cov.whitener(evoked.info, pca=False) # get whitening matrix
sel = mne.fiff.pick_channels(evoked.ch_names, include=ch_names) # channels id
whitened_data = np.dot(W, evoked.data[sel]) # apply whitening

###############################################################################
# Show result
times = 1e3 * epochs.times # in ms
import pylab as pl
pl.clf()
pl.plot(times, whitened_data.T)
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('data (NA)')
pl.title('Whitened EEG data')
pl.show()
