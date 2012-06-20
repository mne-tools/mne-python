"""
======================================
Compute LCMV beamformer on evoked data
======================================

Compute LCMV beamformer solution on evoked dataset
and stores the solution in stc files for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import numpy as np

import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.beamformer import lcmv

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

###############################################################################
# Get epochs
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
left_temporal_channels = mne.read_selection('Left-temporal')
picks = pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                   exclude=raw.info['bads'], selection=left_temporal_channels)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

forward = mne.read_forward_solution(fname_fwd)

noise_cov = mne.read_cov(fname_cov)
noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               mag=0.05, grad=0.05, eeg=0.1, proj=True)

data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01)

# Save result in stc files
stc.save('lcmv')

###############################################################################
# View activation time-series
data, times, _ = mne.label_time_courses(fname_label, "lcmv-lh.stc")
pl.close('all')
pl.plot(1e3 * times, np.mean(data, axis=0))
pl.xlabel('time (ms)')
pl.ylabel('LCMV value')
pl.title('LCMV in %s' % label_name)
pl.show()
