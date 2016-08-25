"""
=============================================
Compute MxNE with time-frequency sparse prior
=============================================

The TF-MxNE solver is a distributed inverse method (like dSPM or sLORETA)
that promotes focal (sparse) sources (such as dipole fitting techniques).
The benefit of this approach is that:

  - it is spatio-temporal without assuming stationarity (sources properties
    can vary over time)
  - activations are localized in space, time and frequency in one step.
  - with a built-in filtering process based on a short time Fourier
    transform (STFT), data does not need to be low passed (just high pass
    to make the signals zero mean).
  - the solver solves a convex optimization problem, hence cannot be
    trapped in local minima.

References:

A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
non-stationary source activations
Neuroimage, Volume 70, 15 April 2013, Pages 410-422, ISSN 1053-8119,
DOI: 10.1016/j.neuroimage.2012.12.051.

A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
Functional Brain Imaging with M/EEG Using Structured Sparsity in
Time-Frequency Dictionaries
Proceedings Information Processing in Medical Imaging
Lecture Notes in Computer Science, 2011, Volume 6801/2011,
600-611, DOI: 10.1007/978-3-642-22092-0_49
http://dx.doi.org/10.1007/978-3-642-22092-0_49
"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.image import index_img

import mne
from mne.datasets import sample
from mne.inverse_sparse import tf_mixed_norm
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

# Handling average file
condition = 'Left visual'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)
# evoked = mne.pick_channels_evoked(evoked)

# We make the window slightly larger than what you'll eventually be interested
# in ([0.0, 0.25]) to avoid edge effects.
evoked.crop(tmin=-0.05, tmax=0.3)

ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.plot(ylim=ylim, proj=True)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run solver

# alpha_space regularization parameter is between 0 and 100 (100 is high)
alpha_space = 30.  # spatial regularization parameter
# alpha_time parameter promotes temporal smoothness
# (0 means no temporal regularization)
alpha_time = 5.  # temporal regularization parameter

loose, depth = 1.0, 1.0  # loose orientation & depth weighting

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=loose, depth=depth)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute TF-MxNE inverse solution
stc, residual = tf_mixed_norm(evoked, forward, cov, alpha_space, alpha_time,
                              loose=loose, depth=depth, maxit=200, tol=1e-4,
                              weights=stc_dspm, weights_min=8., debias=True,
                              wsize=32, tstep=4, window=0.05,
                              return_residual=True)

# Crop to remove edges
stc.crop(tmin=0.0, tmax=0.25)
evoked.crop(tmin=0.0, tmax=0.25)
residual.crop(tmin=0.0, tmax=0.25)

# Show residual
residual.plot(ylim=ylim, proj=True)

###############################################################################
# View source estimate as an overlay on the structural image
stc.save('tfmxne_vis-vol')

t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

img = mne.save_stc_as_volume('tfmxne_inverse_vis.nii.gz', stc,
                             forward['src'], mri_resolution=False)

tidx = np.argmin(np.abs(stc.times - 0.088))
_img = index_img(img, tidx)
plot_glass_brain(_img, colorbar=True, threshold=1e-10,
                 title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])
plot_stat_map(_img, t1_fname, threshold=1e-10,
              title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])

tidx = np.argmin(np.abs(stc.times - 0.135))
_img = index_img(img, tidx)
plot_glass_brain(_img, colorbar=True, threshold=1e-10,
                 title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])
plot_stat_map(_img, t1_fname, threshold=1e-10,
              title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])

tidx = np.argmin(np.abs(stc.times - 0.170))
_img = index_img(img, tidx)
plot_glass_brain(_img, colorbar=True, threshold=1e-10,
                 title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])
plot_stat_map(_img, t1_fname, threshold=1e-10,
              title='TF-MxNE (t=%.3f s.)' % stc.times[tidx])

# plot source time courses
plt.figure()
plt.plot(stc.times, stc.data.T * 1e9)
plt.xlabel('Time (ms)')
plt.ylabel('Ampltidue (nAm)')
plt.show()
