"""
===============================================
Estimate covariance matrix from Epochs baseline
===============================================

We first define a set of Epochs from events and a raw file.
Then we estimate the noise covariance of prestimulus data,
a.k.a. baseline.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

raw = io.read_raw_fif(fname)

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

#   Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
# Read epochs, with proj off by default so we can plot either way later
reject = dict(grad=4000e-13, mag=4e-12, eeg=80e-6, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, proj=False)

# Compute the covariance on baseline
cov = mne.compute_covariance(epochs, tmin=None, tmax=0)
print(cov)

###############################################################################
# Show covariance
mne.viz.plot_cov(cov, raw.info, colorbar=True, proj=True)
# try setting proj to False to see the effect
