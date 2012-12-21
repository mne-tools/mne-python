"""
=============================================
Compute MxNE with time-frequency sparse prior
=============================================

References:

A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
non-stationary source activations
Neuroimage, (to appear)

A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
Functional Brain Imaging with M/EEG Using Structured Sparsity in
Time-Frequency Dictionaries
Proceedings Information Processing in Medical Imaging
Lecture Notes in Computer Science, 2011, Volume 6801/2011,
600-611, DOI: 10.1007/978-3-642-22092-0_49
http://dx.doi.org/10.1007/978-3-642-22092-0_49
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.mixed_norm import tf_mixed_norm
from mne.viz import plot_sparse_source_estimates, plot_evoked

data_path = sample.data_path('..')
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

# Handling average file
setno = 2
evoked = fiff.read_evoked(ave_fname, setno=setno, baseline=(None, 0))
evoked = fiff.pick.pick_channels_evoked(evoked, exclude=evoked.info['bads'])
evoked.crop(tmin=-0.1, tmax=0.4)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname, force_fixed=False,
                                    surf_ori=True)

cov = mne.cov.regularize(cov, evoked.info)

###############################################################################
# Run solver

# alpha_space regularization parameter is between 0 and 100 (100 is high)
alpha_space = 30.  # spatial regularization parameter
alpha_time = 1.  # temporal regularization parameter

loose, depth = 0.2, 0.9  # loose orientation & depth weighting

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=loose, depth=depth)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute TF-MxNE inverse solution
stc, residual = tf_mixed_norm(evoked, forward, cov, alpha_space, alpha_time,
                    loose=loose, depth=depth, maxit=200, tol=1e-4,
                    weights=stc_dspm, weights_min=8., debias=True,
                    wsize=16, tstep=4, window=0.05, return_residual=True)

evoked = mne.fiff.pick_types_evoked(evoked, meg='grad')
residual = mne.fiff.pick_types_evoked(residual, meg='grad')

# Crop to remove edges
stc.crop(tmin=-0.05, tmax=0.3)
evoked.crop(tmin=-0.05, tmax=0.3)
residual.crop(tmin=-0.05, tmax=0.3)

import pylab as pl
pl.figure(-1)
ylim = dict(eeg=[-10, 10], grad=[-200, 250], mag=[-600, 600])
plot_evoked(evoked, ylim=ylim, proj=True)

pl.figure(-2)
plot_evoked(residual, ylim=ylim, proj=True)

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="TF-MxNE (cond %s)" % setno,
                             fig_number=setno, modes=['sphere'],
                             scale_factors=[2.])

time_label = 'TF-MxNE time=%0.2f ms'
brain = stc.plot('sample', 'inflated', 'rh', fmin=10e-9, fmid=15e-9,
                 fmax=20e-9, time_label=time_label, smoothing_steps=5,
                 subjects_dir=data_path + '/subjects')
brain.show_view('medial')
brain.set_data_time_index(120)
brain.add_label("V1", color="yellow", scalar_thresh=.5, borders=True)
brain.add_label("V2", color="red", scalar_thresh=.5, borders=True)
