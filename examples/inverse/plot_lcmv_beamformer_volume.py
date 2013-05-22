"""
===================================================================
Compute LCMV inverse solution on evoked data in volume source space
===================================================================

Compute LCMV inverse solution on an auditory evoked dataset in a volume source
space with three different settings for picking source orientation. It stores
the solution in a nifti file for visualisation e.g. with Freeview.

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

pl.close('all')
fig_1, axes_1 = pl.subplots(nrows=1, ncols=3, figsize=(17, 5))
fig_2, axes_2 = pl.subplots(nrows=1, ncols=3, figsize=(18, 5))

cutoff_point = 0.8
lcmv_limits = (-2, 2.3)

pick_oris = [None, 'normal', 'optimal']
names = ['free', 'normal', 'optimal']
descriptions = ['Free orientation', 'Normal orientation', 'Optimal '
                'orientation']

for pick_ori, name, desc, ax_1, ax_2 in zip(pick_oris, names, descriptions,
                                            axes_1, axes_2):
    stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01,
               pick_ori=pick_ori)

    # Save result in stc files
    stc.save('lcmv-' + name + '-vol')

    stc.crop(0.0, 0.2)

    # Save result in a 4D nifti file
    # (for full MRI resolution use mri_resolution=True)
    img = mne.save_stc_as_volume('lcmv_inverse_ ' + name + '.nii.gz', stc,
                                 forward['src'], mri_resolution=False)

    # Plot result (one slice)
    data = img.get_data()
    coronal_slice = data[:, 10, :, 60]

    boo = ax_1.imshow(np.ma.masked_inside(coronal_slice, -1 * cutoff_point,
                      cutoff_point), cmap=pl.cm.Spectral_r,
                      interpolation='nearest')
    boo.set_clim(lcmv_limits)
    ax_1.contour(coronal_slice != 0, 1, colors=['black'])
    ax_1.set_title(desc)
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    ax_1.set_frame_on(False)

    ax_2.plot(stc.times, stc.data[np.argsort(np.max(stc.data,
              axis=1))[-40:]].T)
    ax_2.set_title(desc)
    ax_2.set_ylim(lcmv_limits)

fig_1.subplots_adjust(right=0.97)
cbar_ax = fig_1.add_axes([0.98, 0.2, 0.015, 0.6])
cbar = fig_1.colorbar(cax=cbar_ax, mappable=axes_1[0].get_images()[0])
cbar.set_label('LCMV value', rotation=270)

axes_2[0].set_xlabel('Time (ms)')
axes_2[0].set_ylabel('LCMV value')

pl.show()
