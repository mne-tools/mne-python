"""
====================================================================
Compute sparse inverse solution based on L0.5/L2 mixed norm (irMxNE)
====================================================================

Compared to MxNE (L1/L2 mixed norm), irMxNE allows for sparser
source estimates with less amplitude bias due to the non-convexity
of the L0.5/L2 mixed norm penalty.

See
Strohmeier D., Haueisen J., and Gramfort A.:
Improved MEG/EEG source localization with reweighted mixed-norms,
4th International Workshop on Pattern Recognition in Neuroimaging,
Tuebingen, 2014
DOI: 10.1109/PRNI.2014.6858545
"""
# Author: Daniel Strohmeier <daniel.strohmeier@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paritech.fr>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
from mne.inverse_sparse import mixed_norm
from mne.viz import plot_sparse_source_estimates

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'

# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0, tmax=0.3)
ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.plot(ylim=ylim, proj=True)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True)

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)

###############################################################################
# Run solver
alpha = 60  # regularization parameter between 0 and 100 (100 is high)
loose, depth = 0.2, 0.9  # loose orientation & depth weighting

# Compute MxNE inverse solution
stc_mxne, residual_mxne = mixed_norm(evoked, forward, cov, alpha, loose=loose,
                                     depth=depth, maxit=1000, tol=1e-4,
                                     active_set_size=10, debias=True,
                                     weights=None, weights_min=None,
                                     n_mxne_iter=1, return_residual=True)
residual_mxne.plot(ylim=ylim, proj=True)

stc_irmxne, residual_irmxne = mixed_norm(evoked, forward, cov, alpha,
                                         loose=loose, depth=depth, maxit=1000,
                                         tol=1e-4, active_set_size=10,
                                         debias=True, weights=None,
                                         weights_min=None, n_mxne_iter=50,
                                         return_residual=True)
residual_irmxne.plot(ylim=ylim, proj=True)

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(forward['src'], [stc_irmxne, stc_mxne],
                             bgcolor=(1, 1, 1), opacity=0.1,
                             modes=['sphere', 'cone'],
                             scale_factors=[0.6, 1.0],
                             fig_name="irMxNE and MxNE (cond %s)" % condition,
                             high_resolution=True)

# and on the fsaverage brain after morphing
stc_fsaverage = stc_irmxne.morph(subject_from='sample', subject_to='fsaverage',
                                 grade=None, sparse=True,
                                 subjects_dir=subjects_dir)
src_fsaverage_fname = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'
src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

plot_sparse_source_estimates(src_fsaverage, stc_fsaverage, bgcolor=(1, 1, 1),
                             opacity=0.1)
