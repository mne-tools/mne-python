"""
===================================================================
Compute LCMV inverse solution on evoked data in volume source space
===================================================================

Compute LCMV inverse solution on an auditory evoked dataset in a volume source
space. It stores the solution in a nifti file for visualisation e.g. with
Freeview.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.beamformer import lcmv


data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'

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
stc.save('lcmv-vol')

stc.crop(0.0, 0.2)

# Save result in a 4D nifti file
img = mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc,
        forward['src'], mri_resolution=True)  # True for full MRI resolution
