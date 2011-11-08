"""
==================
Find ECG artifacts
==================

Locate QRS component of ECG.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import pylab as pl
import mne
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('..')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)

event_id = 999
ecg_events, _, _ = mne.artifacts.find_ecg_events(raw, event_id,
                                                 ch_name='MEG 1531')

# Read epochs
picks = fiff.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False,
                        include=['MEG 1531'])
tmin, tmax = -0.1, 0.1
epochs = mne.Epochs(raw, ecg_events, event_id, tmin, tmax, picks=picks,
                    proj=False)
data = epochs.get_data()

print "Number of detected EOG artifacts : %d" % len(data)

###############################################################################
# Plot EOG artifacts
pl.plot(1e3 * epochs.times, np.squeeze(data).T)
pl.xlabel('Times (ms)')
pl.ylabel('ECG')
pl.show()
