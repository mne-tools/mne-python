# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: Simplified BSD

import os.path as op
import copy
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_true

from mne.datasets import sample
from mne.label import read_label
from mne import read_cov, read_forward_solution, read_evokeds
from mne.inverse_sparse import mixed_norm, tf_mixed_norm
from mne.minimum_norm import apply_inverse, make_inverse_operator


data_path = sample.data_path(download=False)
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
label = 'Aud-rh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)


@sample.requires_sample_data
def test_mxne_inverse():
    """Test (TF-)MxNE inverse computation"""
    # Read noise covariance matrix
    cov = read_cov(fname_cov)

    # Handling average file
    loose = None
    depth = 0.9

    evoked = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked.crop(tmin=-0.03, tmax=0.19)

    evoked_l21 = copy.deepcopy(evoked)
    evoked_l21.crop(tmin=0.07, tmax=0.09)

    label = read_label(fname_label)
    weights_min = 0.5
    forward = read_forward_solution(fname_fwd, force_fixed=False,
                                    surf_ori=True)

    # Reduce source space to make test computation faster
    inverse_operator = make_inverse_operator(evoked_l21.info, forward, cov,
                                             loose=loose, depth=depth,
                                             fixed=True)
    stc_dspm = apply_inverse(evoked_l21, inverse_operator, lambda2=1. / 9.,
                             method='dSPM')
    stc_dspm.data[np.abs(stc_dspm.data) < 12] = 0.0
    stc_dspm.data[np.abs(stc_dspm.data) >= 12] = 1.

    # MxNE tests
    alpha = 60  # spatial regularization parameter

    stc_cd, _ = mixed_norm(evoked_l21, forward, cov, alpha, loose=None,
                        depth=0.9, maxit=1000, tol=1e-8,
                        active_set_size=10, return_residual=True,
                        solver='cd')
    assert_array_almost_equal(stc_cd.times, evoked_l21.times, 5)
    assert_true(stc_cd.vertno[1][0] in label.vertices)

    stc_prox = mixed_norm(evoked_l21, forward, cov, alpha, loose=None,
                          depth=0.9, maxit=1000, tol=1e-8, active_set_size=10,
                          weights=stc_dspm, weights_min=weights_min,
                          solver='prox')
    stc_cd = mixed_norm(evoked_l21, forward, cov, alpha, loose=None,
                        depth=0.9, maxit=1000, tol=1e-8, active_set_size=10,
                        weights=stc_dspm, weights_min=weights_min,
                        solver='cd')
    stc_bcd = mixed_norm(evoked_l21, forward, cov, alpha, loose=None,
                         depth=0.9, maxit=1000, tol=1e-8, active_set_size=10,
                         weights=stc_dspm, weights_min=weights_min,
                         solver='bcd')
    stc_bcd_sloreta = mixed_norm(evoked_l21, forward, cov, alpha, loose=None,
                         depth='sLORETA', maxit=1000, tol=1e-8,
                         active_set_size=10, weights=stc_dspm,
                         weights_min=weights_min, solver='cd')
    assert_array_almost_equal(stc_prox.times, evoked_l21.times, 5)
    assert_array_almost_equal(stc_cd.times, evoked_l21.times, 5)
    assert_array_almost_equal(stc_bcd.times, evoked_l21.times, 5)
    assert_array_almost_equal(stc_bcd_sloreta.times, evoked_l21.times, 5)
    assert_allclose(stc_prox.data, stc_cd.data, rtol=1e-3, atol=0.0)
    assert_allclose(stc_prox.data, stc_bcd.data, rtol=1e-3, atol=0.0)
    assert_allclose(stc_cd.data, stc_bcd.data, rtol=1e-3, atol=0.0)
    assert_true(stc_prox.vertno[1][0] in label.vertices)
    assert_true(stc_cd.vertno[1][0] in label.vertices)
    assert_true(stc_bcd.vertno[1][0] in label.vertices)

    # irMxNE tests
    stc_cd = mixed_norm(evoked_l21, forward, cov, alpha,
                        n_mxne_iter=5, loose=None, depth=0.9,
                        maxit=1000, tol=1e-8, active_set_size=10,
                        solver='cd')
    assert_array_almost_equal(stc_cd.times, evoked_l21.times, 5)
    assert_true(stc_cd.vertno[1][0] in label.vertices)

    # Do with TF-MxNE for test memory savings
    alpha_space = 60.  # spatial regularization parameter
    alpha_time = 1.  # temporal regularization parameter

    stc_prox, _ = tf_mixed_norm(evoked, forward, cov, alpha_space, alpha_time,
                           loose=loose, depth=depth, maxit=100, tol=1e-4,
                           tstep=4, wsize=16, window=0.1, weights=stc_dspm,
                           weights_min=weights_min, return_residual=True)
    assert_array_almost_equal(stc_prox.times, evoked.times, 5)
    assert_true(stc_prox.vertno[1][0] in label.vertices)
