"""
====================================================
Compute LCMV inverse solution in volume source space
====================================================

Compute LCMV beamformer on an auditory evoked dataset in a volume source space,
and show activation on ``fsaverage``.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample, fetch_fsaverage
from mne.beamformer import make_lcmv, apply_lcmv

print(__doc__)

###############################################################################
# Data preprocessing:

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
fetch_fsaverage(subjects_dir)  # ensure fsaverage src exists
fname_fs_src = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'

# Get epochs
event_id, tmin, tmax = [1, 2], -0.2, 0.5

# Read forward model
forward = mne.read_forward_solution(fname_fwd)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
events = mne.find_events(raw)

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
# Compute covariance matrices
# ---------------------------
#
# These matrices need to be inverted at some point, but since they are rank
# deficient, some regularization needs to be done for them to be invertable.
# Regularization can be added either by the :func:`mne.compute_covariance`
# function or later by the :func:`mne.beamformer.make_lcmv` function. In this
# example, we'll go with the latter option, so we specify ``method='empirical``
# here.

# Read regularized noise covariance and compute regularized data covariance
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0,
                                   method='empirical')
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='empirical')

###############################################################################
# Compute beamformer filters
# --------------------------
#
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
# Plot source space activity
# --------------------------

# You can save result in stc files with:
# stc.save('lcmv-vol')
lims = [0.3, 0.6, 0.9]
stc.plot(
    src=forward['src'], subject='sample', subjects_dir=subjects_dir,
    clim=dict(kind='value', pos_lims=lims), mode='stat_map',
    initial_time=0.1, verbose=True)

###############################################################################
# Now let's plot this on a glass brain, which will automatically transform the
# data to MNI Talairach space:

# sphinx_gallery_thumbnail_number = 4

stc.plot(
    src=forward['src'], subject='sample', subjects_dir=subjects_dir,
    mode='glass_brain', clim=dict(kind='value', lims=lims),
    initial_time=0.1, verbose=True)

###############################################################################
# Finally let's get another view, this time plotting again a ``'stat_map'``
# style but using volumetric morphing to get data to fsaverage space,
# which we can get by passing a :class:`mne.SourceMorph` as the ``src``
# argument to `mne.VolSourceEstimate.plot`. To save a bit of speed when
# applying the morph, we will crop the STC:

src_fs = mne.read_source_spaces(fname_fs_src)
morph = mne.compute_source_morph(
    forward['src'], subject_from='sample', src_to=src_fs,
    subjects_dir=subjects_dir,
    niter_sdr=[10, 10, 5], niter_affine=[10, 10, 5],  # just for speed
    verbose=True)
stc_fs = morph.apply(stc.copy().crop(0.05, 0.18))
stc_fs.plot(
    src=src_fs, mode='stat_map', initial_time=0.1, subjects_dir=subjects_dir,
    clim=dict(kind='value', pos_lims=lims), verbose=True)
