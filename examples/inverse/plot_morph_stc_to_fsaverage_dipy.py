"""
==================================================================
Demonstrate morphing subject source volume to fsaverage using dipy
==================================================================

This example demonstrates how to morph an individual subject source space to a
common reference space. For this purpose dipy (http://nipy.org/dipy/) will be
used in order to perform the necessary transforms to a fsaverage serving as
reference space. The example uses parts of the MNE example lcmv beamformer
pipeline (examples/inverse/plot_lcmv_beamformer_volume.py).
The respective result will be morphed based on an affine transformation and a
nonlinear morph, estimated based on respective transformation from the
subject's anatomical T1 (brain) to fsaverage T1 (brain). Afterwards the
transformation will be applied to the beamformer result. Affine transformations
are computed based on the mutual information and nonlinear transformations
will be performed as Symmetric Diffeomorphic Registration using the
cross-correlation metric [1]_

.. warning:: Please do not copy the patterns presented here for your own
             analysis, this is example is purely illustrative.

.. note:: This example does quite a bit of processing, so even on a
          fast machine it can take a couple of minutes to complete.

References
----------
.. [1] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
        Symmetric Diffeomorphic Image Registration with Cross- Correlation:
        Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
        12(1), 26-41.

"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, apply_lcmv

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
import nibabel as nib

# uncomment the line(s) below if you encounter problems with ssl certification
# import ssl
# if hasattr(ssl, '_create_unverified_context'):
#     ssl._create_default_https_context = ssl._create_unverified_context

print(__doc__)

##############################################################################
# from examples/inverse/plot_lcmv_beamformer_volume.py

# Setup paths
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

# take absolute values for plotting
stc.data[:, :] = np.abs(stc.data)

# Save result in stc files
stc.save('lcmv-vol')

# stc.crop(0.0, 0.2)
# Save result in a 4D nifti file
mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc,
                       forward['src'], mri_resolution=True)  # set to true

###############################################################################
# Load nifti data

# load lcmv inverse
stc = nib.load('lcmv_inverse.nii.gz')

# select first time volume and output ndarray
stc = stc.dataobj[:, :, :, 1]

# normalize values
stc = stc / stc.max()

# load subject brain (Moving)
t1_fname = data_path + '/subjects/sample/mri/brain.mgz'
t1_m = nib.load(t1_fname)

# get Moving to world transform
t1_m_grid2world = t1_m.affine

# output Moving as ndarray
t1_m = t1_m.dataobj[:, :, :]

# normalize values
t1_m = t1_m.astype('float') / t1_m.max()

# load fsaverage brain (Static)
t1_fname = data_path + '/subjects/fsaverage/mri/brain.mgz'
t1_s = nib.load(t1_fname)

# get Static to world transform
t1_s_grid2world = t1_s.affine

# output Static as ndarray
t1_s = t1_s.dataobj[:, :, :]

# normalize values
t1_s = t1_s.astype('float') / t1_s.max()

##############################################################################
# Compute affine transformation using mutual information metric

# compute center of mass
c_of_mass = transform_centers_of_mass(t1_s, t1_s_grid2world,
                                      t1_m, t1_m_grid2world)

nbins = 32

# prepare affine registration
affreg = AffineRegistration(metric=MutualInformationMetric(nbins, None),
                            level_iters=[10000, 1000, 100],
                            sigmas=[3.0, 1.0, 0.0],
                            factors=[4, 2, 1])

# translation
translation = affreg.optimize(t1_s, t1_m, TranslationTransform3D(), None,
                              t1_s_grid2world, t1_m_grid2world,
                              starting_affine=c_of_mass.affine)

# rigid body transform (translation + rotation)
rigid = affreg.optimize(t1_s, t1_m, RigidTransform3D(), None,
                        t1_s_grid2world, t1_m_grid2world,
                        starting_affine=translation.affine)

# affine transform (translation + rotation + scaling)
affine = affreg.optimize(t1_s, t1_m, AffineTransform3D(), None,
                         t1_s_grid2world, t1_m_grid2world,
                         starting_affine=rigid.affine)

# apply affine transformation
t1_m_affine = affine.transform(t1_m)

##############################################################################
# Compute Symmetric Diffeomorphic Registration

# set up Symmetric Diffeomorphic Registration (metric, iterations per level)
sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), [10, 10, 5])

# compute mapping
mapping = sdr.optimize(t1_s, t1_m_affine)

##############################################################################
# Apply transformations and plot

# morph stc data
stc_sdr_affine = mapping.transform(affine.transform(stc))

# plot result
slice_sel = 100

fig = plt.figure()
plt.subplot(131)
plt.imshow(t1_m[slice_sel, :, :] + stc[slice_sel, :, :])
plt.title('subject brain')
plt.set_cmap('jet')
plt.axis('off')
plt.subplot(132)
plt.imshow(t1_s[slice_sel, :, :] + stc[slice_sel, :, :])
plt.title('fsaverage brain\nprior transformation')
plt.axis('off')
plt.subplot(133)
plt.title('fsaverage brain\nafter transformation')
plt.imshow(t1_s[slice_sel, :, :] + stc_sdr_affine[slice_sel, :, :])
plt.axis('off')
plt.suptitle('moving individual source data to fsaverage')
plt.show()
