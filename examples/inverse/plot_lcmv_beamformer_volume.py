"""
===================================================================
Compute LCMV inverse solution on evoked data in volume source space
===================================================================

Compute LCMV inverse solution on an auditory evoked dataset in a volume source
space. It stores the solution in a nifti file for visualisation, e.g. with
Freeview.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, apply_lcmv

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

print(__doc__)

# sphinx_gallery_thumbnail_number = 3


###############################################################################
# Data preprocessing:

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'

# Get epochs
event_id, tmin, tmax = [1, 2], -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.read_events(event_fname)

# Set up pick list: gradiometers and magnetometers, excluding bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                       exclude='bads')

# Pick the channels of interest
raw.pick_channels([raw.ch_names[pick] for pick in picks])

# Re-normalize our empty-room projectors, so they are fine after subselection
raw.info.normalize_proj()

# Read epochs
proj = False  # already applied
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=proj,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

# Visualize sensor space data
evoked.plot_joint(ts_args=dict(time_unit='s'),
                  topomap_args=dict(time_unit='s'))

###############################################################################
# Compute covariance matrices, fit and apply  spatial filter.

# Read regularized noise covariance and compute regularized data covariance
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk')
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk')

# Read forward model
forward = mne.read_forward_solution(fname_fwd)

# Compute weights of free orientation (vector) beamformer with weight
# normalization (neural activity index, NAI). Providing a noise covariance
# matrix enables whitening of the data and forward solution. Source orientation
# is optimized by setting pick_ori to 'max-power'.
# weight_norm can also be set to 'unit-noise-gain'. Source orientation can also
# be 'normal' (but only when using a surface-based source space) or None,
# which computes a vector beamfomer. Note, however, that not all combinations
# of orientation selection and weight normalization are implemented yet.
filters = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='nai')

# Apply this spatial filter to the evoked data.
stc = apply_lcmv(evoked, filters, max_ori_out='signed')

###############################################################################
# Plot source space activity:

# take absolute values for plotting
stc.data[:, :] = np.abs(stc.data)

# Save result in stc files
stc.save('lcmv-vol')

stc.crop(0.0, 0.2)

# Save result in a 4D nifti file
img = mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc,
                             forward['src'], mri_resolution=False)

t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

# Plotting with nilearn ######################################################
# Based on the visualization of the sensor space data (gradiometers), plot
# activity at 88 ms
idx = stc.time_as_index(0.088)
plot_stat_map(index_img(img, idx), t1_fname, threshold=0.45,
              title='LCMV (t=%.3f s.)' % stc.times[idx])

# plot source time courses with the maximum peak amplitudes at 88 ms
plt.figure()
plt.plot(stc.times, stc.data[np.argsort(np.max(stc.data[:, idx],
                                               axis=1))[-40:]].T)
plt.xlabel('Time (ms)')
plt.ylabel('LCMV value')
plt.show()
