""".

test

"""

import mne
from mne.datasets import sample  # , somato
from mne.inverse_sparse import mixed_norm
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_sparse_source_estimates

# import numpy as np

data = 'somato'
if data == 'sample':
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
    subjects_dir = data_path + '/subjects'
    condition = 'Left Auditory'
elif data == 'somato':
    # data_path = somato.data_path()
    # fwd_fname = data_path + '/MEG/somato/sef_oct-6_fwd.fif'
    # ave_fname = data_path + '/MEG/somato/sef-ave.fif'
    # cov_fname = data_path + '/MEG/somato/sef-cov.fif'
    # subjects_dir = data_path + '/subjects'
    # condition = 'Unknown'
    data_path = 'SEF_data/'

    ave_fname = data_path + 'mind006_051209_median01_raw_daniel_long-ave.fif'
    cov_fname = data_path + 'mind006_051209_median01_raw_daniel_long-cov.fif'
    fwd_fname = data_path + 'mind006_051209_median01_raw-oct-6-fwd.fif'
    condition = 'Unknown'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
if data == 'sample':
    evoked.crop(tmin=0.04, tmax=0.18)
else:
    evoked.crop(tmin=0.008, tmax=0.25)

evoked = evoked.pick_types(eeg=False, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True)

# ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
# evoked.plot(ylim=ylim, proj=True)

###############################################################################
# Run solver
loose, depth = 0.2, 0.9  # loose orientation & depth weighting
update_alpha = False
if update_alpha:
    alpha = 80.  # * np.ones((forward['sol']['data'].shape[1],))
    n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solve
    # if n_mxne_iter > 1 dSPM weighting can be avoided.
else:
    alpha = 20.
    n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solve

# Compute dSPM solution to be used as weights in MxNE
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         loose=None, depth=depth, fixed=True)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')

# Compute (ir)MxNE inverse solution
out = mixed_norm(evoked, forward, cov, alpha, loose=loose, depth=depth,
                 maxit=3000, tol=1e-4, active_set_size=50, debias=True,
                 weights=stc_dspm, weights_min=8., n_mxne_iter=n_mxne_iter,
                 return_residual=True, update_alpha=update_alpha,
                 time_pca=False, verbose=True)
# residual.plot(ylim=ylim, proj=True)
if update_alpha:
    (stc, residual), alphas = out
else:
    stc, residual = out

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
hp = '- hp estimated' if update_alpha else '- no estimation'
if condition == 'Unknown':
    condition = 'mind'
if n_mxne_iter == 1:
    solver = "MxNE"
else:
    solver = "irMxNE"
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             fig_name="%s (cond %s) %s" % (solver, condition,
                                                           hp),
                             opacity=0.1)

# # and on the fsaverage brain after morphing
# stc_fsaverage = stc.morph(subject_from='sample', subject_to='fsaverage',
#                           grade=None, sparse=True, subjects_dir=subjects_dir)
# src_fsaverage_fname = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'
# src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

# plot_sparse_source_estimates(src_fsaverage, stc_fsaverage, bgcolor=(1, 1, 1),
#                              opacity=0.1)
