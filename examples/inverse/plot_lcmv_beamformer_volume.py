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

import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.beamformer import lcmv


data_path = sample.data_path()
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
                   exclude='bads', selection=left_temporal_channels)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

noise_cov = mne.read_cov(fname_cov)
noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               mag=0.05, grad=0.05, eeg=0.1, proj=True)

data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01)
stc_normal = lcmv(evoked, forward, noise_cov, data_cov, pick_ori='normal',
                  reg=0.01)
stc_optimal = lcmv(evoked, forward, noise_cov, data_cov, pick_ori='optimal',
                   reg=0.01)

# Save result in stc files
stc.save('lcmv-vol')
stc_normal.save('lcmv-normal-vol')
stc_optimal.save('lcmv-optimal-vol')

stc.crop(0.0, 0.2)
stc_normal.crop(0.0, 0.2)
stc_optimal.crop(0.0, 0.2)

# Save result in a 4D nifti file
img = mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc,
        forward['src'], mri_resolution=False)  # True for full MRI resolution
img_normal = mne.save_stc_as_volume('lcmv_normal_inverse.nii.gz', stc_normal,
                                    forward['src'], mri_resolution=False)
img_optimal = mne.save_stc_as_volume('lcmv_optimal_inverse.nii.gz',
                                     stc_optimal, forward['src'],
                                     mri_resolution=False)

# plot result (one slice)
pl.close('all')
data = img.get_data()
data_normal = img_normal.get_data()
data_optimal = img_optimal.get_data()
coronal_slice = data[:, 10, :, 60]
coronal_slice_normal = data_normal[:, 10, :, 60]
coronal_slice_optimal = data_optimal[:, 10, :, 60]
c_limits = (-2, 2.3)
pl.figure(figsize=(20, 5))
pl.subplot(1, 3, 1)
#pl.imshow(np.ma.masked_less(coronal_slice, 1), cmap=pl.cm.Reds,
#          interpolation='nearest')
pl.imshow(coronal_slice, cmap=pl.cm.Spectral_r,
          interpolation='nearest')
pl.clim(c_limits)
pl.colorbar()
pl.contour(coronal_slice != 0, 1, colors=['black'])
pl.xticks([])
pl.yticks([])

pl.subplot(1, 3, 2)
pl.imshow(coronal_slice_normal, cmap=pl.cm.Spectral_r,
          interpolation='nearest')
pl.clim(c_limits)
pl.colorbar()
pl.contour(coronal_slice != 0, 1, colors=['black'])
pl.xticks([])
pl.yticks([])

pl.subplot(1, 3, 3)
pl.imshow(coronal_slice_optimal, cmap=pl.cm.Spectral_r,
          interpolation='nearest')
pl.clim(c_limits)
pl.colorbar()
pl.contour(coronal_slice != 0, 1, colors=['black'])
pl.xticks([])
pl.yticks([])


# plot source time courses with the maximum peak amplitudes
pl.figure(figsize=(20, 5))
y_limits = (-2, 2.3)
pl.subplot(1, 3, 1)
pl.plot(stc.times, stc.data[np.argsort(np.max(stc.data, axis=1))[-40:]].T)
pl.ylim(y_limits)
pl.subplot(1, 3, 2)
pl.plot(stc_normal.times, stc_normal.data[np.argsort(np.max(stc.data,
                                          axis=1))[-40:]].T)
pl.ylim(y_limits)
pl.subplot(1, 3, 3)
pl.plot(stc_optimal.times, stc_optimal.data[np.argsort(np.max(stc.data,
                                            axis=1))[-40:]].T)
pl.ylim(y_limits)
pl.xlabel('Time (ms)')
pl.ylabel('LCMV value')
pl.show()
