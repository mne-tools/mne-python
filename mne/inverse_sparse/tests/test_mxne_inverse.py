# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: Simplified BSD

import os.path as op
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_true, assert_equal, assert_raises
import pytest

import mne
from mne.datasets import testing
from mne.label import read_label
from mne import (read_cov, read_forward_solution, read_evokeds,
                 convert_forward_solution)
from mne.inverse_sparse import mixed_norm, tf_mixed_norm
from mne.inverse_sparse.mxne_inverse import make_stc_from_dipoles
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.utils import run_tests_if_main
from mne.dipole import Dipole
from mne.source_estimate import VolSourceEstimate


data_path = testing.data_path(download=False)
# NOTE: These use the ave and cov from sample dataset (no _trunc)
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
label = 'Aud-rh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)


def _check_stcs(stc1, stc2):
    """Helper to check correctness"""
    assert_allclose(stc1.times, stc2.times)
    assert_allclose(stc1.data, stc2.data)
    assert_allclose(stc1.vertices[0], stc2.vertices[0])
    assert_allclose(stc1.vertices[1], stc2.vertices[1])
    assert_allclose(stc1.tmin, stc2.tmin)
    assert_allclose(stc1.tstep, stc2.tstep)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_mxne_inverse():
    """Test (TF-)MxNE inverse computation"""
    # Read noise covariance matrix
    cov = read_cov(fname_cov)

    # Handling average file
    loose = 0.0
    depth = 0.9

    evoked = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked.crop(tmin=-0.05, tmax=0.2)

    evoked_l21 = evoked.copy()
    evoked_l21.crop(tmin=0.081, tmax=0.1)
    label = read_label(fname_label)

    forward = read_forward_solution(fname_fwd)
    forward = convert_forward_solution(forward, surf_ori=True)

    # Reduce source space to make test computation faster
    inverse_operator = make_inverse_operator(evoked_l21.info, forward, cov,
                                             loose=loose, depth=depth,
                                             fixed=True, use_cps=True)
    stc_dspm = apply_inverse(evoked_l21, inverse_operator, lambda2=1. / 9.,
                             method='dSPM')
    stc_dspm.data[np.abs(stc_dspm.data) < 12] = 0.0
    stc_dspm.data[np.abs(stc_dspm.data) >= 12] = 1.
    weights_min = 0.5

    # MxNE tests
    alpha = 70  # spatial regularization parameter

    stc_prox = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                          depth=depth, maxit=300, tol=1e-8,
                          active_set_size=10, weights=stc_dspm,
                          weights_min=weights_min, solver='prox')
    stc_cd = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                        depth=depth, maxit=300, tol=1e-8, active_set_size=10,
                        weights=stc_dspm, weights_min=weights_min,
                        solver='cd')
    stc_bcd = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                         depth=depth, maxit=300, tol=1e-8, active_set_size=10,
                         weights=stc_dspm, weights_min=weights_min,
                         solver='bcd')
    assert_array_almost_equal(stc_prox.times, evoked_l21.times, 5)
    assert_array_almost_equal(stc_cd.times, evoked_l21.times, 5)
    assert_array_almost_equal(stc_bcd.times, evoked_l21.times, 5)
    assert_allclose(stc_prox.data, stc_cd.data, rtol=1e-3, atol=0.0)
    assert_allclose(stc_prox.data, stc_bcd.data, rtol=1e-3, atol=0.0)
    assert_allclose(stc_cd.data, stc_bcd.data, rtol=1e-3, atol=0.0)
    assert_true(stc_prox.vertices[1][0] in label.vertices)
    assert_true(stc_cd.vertices[1][0] in label.vertices)
    assert_true(stc_bcd.vertices[1][0] in label.vertices)

    dips = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                      depth=depth, maxit=300, tol=1e-8, active_set_size=10,
                      weights=stc_dspm, weights_min=weights_min,
                      solver='cd', return_as_dipoles=True)
    stc_dip = make_stc_from_dipoles(dips, forward['src'])
    assert_true(isinstance(dips[0], Dipole))
    _check_stcs(stc_cd, stc_dip)

    stc, _ = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                        depth=depth, maxit=300, tol=1e-8,
                        active_set_size=10, return_residual=True,
                        solver='cd')
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)
    assert_true(stc.vertices[1][0] in label.vertices)

    # irMxNE tests
    stc = mixed_norm(evoked_l21, forward, cov, alpha,
                     n_mxne_iter=5, loose=loose, depth=depth,
                     maxit=300, tol=1e-8, active_set_size=10,
                     solver='cd')
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)
    assert_true(stc.vertices[1][0] in label.vertices)
    assert_equal(stc.vertices, [[63152], [79017]])

    # Do with TF-MxNE for test memory savings
    alpha = 60.  # overall regularization parameter
    l1_ratio = 0.01  # temporal regularization proportion

    stc, _ = tf_mixed_norm(evoked, forward, cov, None, None,
                           loose=loose, depth=depth, maxit=100, tol=1e-4,
                           tstep=4, wsize=16, window=0.1, weights=stc_dspm,
                           weights_min=weights_min, return_residual=True,
                           alpha=alpha, l1_ratio=l1_ratio)
    assert_array_almost_equal(stc.times, evoked.times, 5)
    assert_true(stc.vertices[1][0] in label.vertices)

    with warnings.catch_warnings(record=True) as w:
        assert_raises(ValueError, tf_mixed_norm, evoked, forward, cov,
                      101., 3.)
        assert_raises(ValueError, tf_mixed_norm, evoked, forward, cov,
                      50, 101.)
        assert_true(len(w) == 2)

    assert_raises(ValueError, tf_mixed_norm, evoked, forward, cov, None, None,
                  alpha=101, l1_ratio=0.03)
    assert_raises(ValueError, tf_mixed_norm, evoked, forward, cov, None, None,
                  alpha=50., l1_ratio=1.01)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_mxne_vol_sphere():
    """(TF-)MxNE with a sphere forward and volumic source space"""
    evoked = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked.crop(tmin=-0.05, tmax=0.2)
    cov = read_cov(fname_cov)

    evoked_l21 = evoked.copy()
    evoked_l21.crop(tmin=0.081, tmax=0.1)

    info = evoked.info
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(subject=None, pos=15., mri=None,
                                        sphere=(0.0, 0.0, 0.0, 80.0),
                                        bem=None, mindist=5.0,
                                        exclude=2.0)
    fwd = mne.make_forward_solution(info, trans=None, src=src,
                                    bem=sphere, eeg=False, meg=True)

    alpha = 80.
    assert_raises(ValueError, mixed_norm, evoked, fwd, cov, alpha,
                  loose=0.0, return_residual=False,
                  maxit=3, tol=1e-8, active_set_size=10)

    assert_raises(ValueError, mixed_norm, evoked, fwd, cov, alpha,
                  loose=0.2, return_residual=False,
                  maxit=3, tol=1e-8, active_set_size=10)

    # irMxNE tests
    stc = mixed_norm(evoked_l21, fwd, cov, alpha,
                     n_mxne_iter=1, maxit=30, tol=1e-8,
                     active_set_size=10)
    assert_true(isinstance(stc, VolSourceEstimate))
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)

    # Compare orientation obtained using fit_dipole and gamma_map
    # for a simulated evoked containing a single dipole
    stc = mne.VolSourceEstimate(50e-9 * np.random.RandomState(42).randn(1, 4),
                                vertices=stc.vertices[:1],
                                tmin=stc.tmin,
                                tstep=stc.tstep)
    evoked_dip = mne.simulation.simulate_evoked(fwd, stc, info, cov, nave=1e9,
                                                use_cps=True)

    dip_mxne = mixed_norm(evoked_dip, fwd, cov, alpha=80,
                          n_mxne_iter=1, maxit=30, tol=1e-8,
                          active_set_size=10, return_as_dipoles=True)

    amp_max = [np.max(d.amplitude) for d in dip_mxne]
    dip_mxne = dip_mxne[np.argmax(amp_max)]
    assert_true(dip_mxne.pos[0] in src[0]['rr'][stc.vertices])

    dip_fit = mne.fit_dipole(evoked_dip, cov, sphere)[0]
    assert_true(np.abs(np.dot(dip_fit.ori[0], dip_mxne.ori[0])) > 0.99)

    # Do with TF-MxNE for test memory savings
    alpha = 60.  # overall regularization parameter
    l1_ratio = 0.01  # temporal regularization proportion

    stc, _ = tf_mixed_norm(evoked, fwd, cov, maxit=3, tol=1e-4,
                           tstep=16, wsize=32, window=0.1, alpha=alpha,
                           l1_ratio=l1_ratio, return_residual=True)
    assert_true(isinstance(stc, VolSourceEstimate))
    assert_array_almost_equal(stc.times, evoked.times, 5)

run_tests_if_main()
