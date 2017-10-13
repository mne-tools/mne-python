"""
=============================================
Compute MxNE with time-frequency sparse prior
=============================================

The TF-MxNE solver is a distributed inverse method (like dSPM or sLORETA)
that promotes focal (sparse) sources (such as dipole fitting techniques)
[1]_ [2]_. The benefit of this approach is that:

  - it is spatio-temporal without assuming stationarity (sources properties
    can vary over time)
  - activations are localized in space, time and frequency in one step.
  - with a built-in filtering process based on a short time Fourier
    transform (STFT), data does not need to be low passed (just high pass
    to make the signals zero mean).
  - the solver solves a convex optimization problem, hence cannot be
    trapped in local minima.

References
----------
.. [1] A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
   "Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
   non-stationary source activations",
   Neuroimage, Volume 70, pp. 410-422, 15 April 2013.
   DOI: 10.1016/j.neuroimage.2012.12.051

.. [2] A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
   "Functional Brain Imaging with M/EEG Using Structured Sparsity in
   Time-Frequency Dictionaries",
   Proceedings Information Processing in Medical Imaging
   Lecture Notes in Computer Science, Volume 6801/2011, pp. 600-611, 2011.
   DOI: 10.1007/978-3-642-22092-0_49
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.inverse_sparse import tf_mixed_norm, make_stc_from_dipoles
from mne.viz import (plot_sparse_source_estimates,
                     plot_dipole_locations, plot_dipole_amplitudes)

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

# Handling average file
condition = 'Left visual'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked = mne.pick_channels_evoked(evoked)
# We make the window slightly larger than what you'll eventually be interested
# in ([-0.05, 0.3]) to avoid edge effects.
evoked.crop(tmin=-0.1, tmax=0.4)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run solver

# alpha_space regularization parameter is between 0 and 100 (100 is high)
alpha_space = 30.  # spatial regularization parameter
# alpha_time parameter promotes temporal smoothness
# (0 means no temporal regularization)
alpha_time = 1.  # temporal regularization parameter

loose, depth = 0.2, 0.9  # loose orientation & depth weighting

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=loose, depth=depth)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute TF-MxNE inverse solution with dipole output
dipoles, residual = tf_mixed_norm(
    evoked, forward, cov, alpha_space, alpha_time, loose=loose, depth=depth,
    maxit=200, tol=1e-6, weights=stc_dspm, weights_min=8., debias=True,
    wsize=16, tstep=4, window=0.05, return_as_dipoles=True,
    return_residual=True)

# Crop to remove edges
for dip in dipoles:
    dip.crop(tmin=-0.05, tmax=0.3)
evoked.crop(tmin=-0.05, tmax=0.3)
residual.crop(tmin=-0.05, tmax=0.3)

###############################################################################
# Plot dipole activations
plot_dipole_amplitudes(dipoles)

# Plot dipole location of the strongest dipole with MRI slices
idx = np.argmax([np.max(np.abs(dip.amplitude)) for dip in dipoles])
plot_dipole_locations(dipoles[idx], forward['mri_head_t'], 'sample',
                      subjects_dir=subjects_dir, mode='orthoview',
                      idx='amplitude')

# # Plot dipole locations of all dipoles with MRI slices
# for dip in dipoles:
#     plot_dipole_locations(dip, forward['mri_head_t'], 'sample',
#                           subjects_dir=subjects_dir, mode='orthoview',
#                           idx='amplitude')

###############################################################################
# Show the evoked response and the residual for gradiometers
ylim = dict(grad=[-120, 120])
evoked.pick_types(meg='grad', exclude='bads')
evoked.plot(titles=dict(grad='Evoked Response: Gradiometers'), ylim=ylim,
            proj=True)

residual.pick_types(meg='grad', exclude='bads')
residual.plot(titles=dict(grad='Residuals: Gradiometers'), ylim=ylim,
              proj=True)

###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="TF-MxNE (cond %s)"
                             % condition, modes=['sphere'], scale_factors=[1.])

time_label = 'TF-MxNE time=%0.2f ms'
clim = dict(kind='value', lims=[10e-9, 15e-9, 20e-9])
brain = stc.plot('sample', 'inflated', 'rh', views='medial',
                 clim=clim, time_label=time_label, smoothing_steps=5,
                 subjects_dir=subjects_dir, initial_time=150, time_unit='ms')
brain.add_label("V1", color="yellow", scalar_thresh=.5, borders=True)
brain.add_label("V2", color="red", scalar_thresh=.5, borders=True)
