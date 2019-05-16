"""
====================================================
Compute LCMV inverse solution in volume source space
====================================================

Compute LCMV beamformer on an auditory evoked dataset in a volume source space.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 3

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, apply_lcmv

print(__doc__)

###############################################################################
# Data preprocessing:

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'

# Get epochs
event_id, tmin, tmax = [1, 2], -0.2, 0.5

# Read forward model
forward = mne.read_forward_solution(fname_fwd)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.read_events(event_fname)

# Pick the channels of interest
raw.pick(['meg', 'eog'])

# Read epochs
proj = False  # already applied
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=proj,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()

# Visualize sensor space data
evoked.plot_joint()

###############################################################################
# Compute covariance matrices, fit and apply  spatial filter.

# Read regularized noise covariance and compute regularized data covariance
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method='shrunk',
                                   rank=None)
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk', rank=None)

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
                    weight_norm='nai', rank=None)
print(filters)

# You can save these with:
# filters.save('filters-lcmv.h5')

# Apply this spatial filter to the evoked data.
stc = apply_lcmv(evoked, filters, max_ori_out='signed')

###############################################################################
# Plot source space activity:

# You can save result in stc files with:
# stc.save('lcmv-vol')

clim = dict(kind='value', pos_lims=[0.3, 0.6, 0.9])
stc.plot(src=forward['src'], subject='sample', subjects_dir=subjects_dir,
         clim=clim)

###############################################################################
# We can also visualize the activity on a "glass brain" (shown here with
# absolute values):

clim = dict(kind='value', lims=[0.3, 0.6, 0.9])
abs(stc).plot(src=forward['src'], subject='sample', subjects_dir=subjects_dir,
              mode='glass_brain', clim=clim)
