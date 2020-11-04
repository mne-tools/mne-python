# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: Simplified BSD

import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_less)
import pytest

import mne
from mne.datasets import testing
from mne.label import read_label
from mne import (read_cov, read_forward_solution, read_evokeds,
                 convert_forward_solution)
from mne.inverse_sparse import mixed_norm, tf_mixed_norm
from mne.inverse_sparse.mxne_inverse import make_stc_from_dipoles, _split_gof
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.minimum_norm.tests.test_inverse import \
    assert_var_exp_log, assert_stc_res
from mne.utils import assert_stcs_equal, run_tests_if_main, catch_logging
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


@pytest.fixture(scope='module', params=[testing._pytest_param])
def forward():
    """Get a forward solution."""
    # module scope it for speed (but don't overwrite in use!)
    return read_forward_solution(fname_fwd)


@testing.requires_testing_data
@pytest.mark.timeout(150)  # ~30 sec on Travis Linux
@pytest.mark.slowtest
def test_mxne_inverse_standard(forward):
    """Test (TF-)MxNE inverse computation."""
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
    assert label.hemi == 'rh'

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
    with pytest.warns(None):  # CD
        stc_cd = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                            depth=depth, maxit=300, tol=1e-8,
                            active_set_size=10, weights=stc_dspm,
                            weights_min=weights_min, solver='cd')
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
    assert stc_prox.vertices[1][0] in label.vertices
    assert stc_cd.vertices[1][0] in label.vertices
    assert stc_bcd.vertices[1][0] in label.vertices

    # vector
    with pytest.warns(None):  # no convergence
        stc = mixed_norm(evoked_l21, forward, cov, alpha, loose=1, maxit=2)
    with pytest.warns(None):  # no convergence
        stc_vec = mixed_norm(evoked_l21, forward, cov, alpha, loose=1, maxit=2,
                             pick_ori='vector')
    assert_stcs_equal(stc_vec.magnitude(), stc)
    with pytest.warns(None), pytest.raises(ValueError, match='pick_ori='):
        mixed_norm(evoked_l21, forward, cov, alpha, loose=0, maxit=2,
                   pick_ori='vector')

    with pytest.warns(None), catch_logging() as log:  # CD
        dips = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                          depth=depth, maxit=300, tol=1e-8, active_set_size=10,
                          weights=stc_dspm, weights_min=weights_min,
                          solver='cd', return_as_dipoles=True, verbose=True)
    stc_dip = make_stc_from_dipoles(dips, forward['src'])
    assert isinstance(dips[0], Dipole)
    assert stc_dip.subject == "sample"
    assert_stcs_equal(stc_cd, stc_dip)
    assert_var_exp_log(log.getvalue(), 51, 53)  # 51.8

    # Single time point things should match
    with pytest.warns(None), catch_logging() as log:
        dips = mixed_norm(evoked_l21.copy().crop(0.081, 0.081),
                          forward, cov, alpha, loose=loose,
                          depth=depth, maxit=300, tol=1e-8, active_set_size=10,
                          weights=stc_dspm, weights_min=weights_min,
                          solver='cd', return_as_dipoles=True, verbose=True)
    assert_var_exp_log(log.getvalue(), 37.8, 38.0)  # 37.9
    gof = sum(dip.gof[0] for dip in dips)  # these are now partial exp vars
    assert_allclose(gof, 37.9, atol=0.1)

    with pytest.warns(None), catch_logging() as log:
        stc, res = mixed_norm(evoked_l21, forward, cov, alpha, loose=loose,
                              depth=depth, maxit=300, tol=1e-8,
                              weights=stc_dspm,  # gh-6382
                              active_set_size=10, return_residual=True,
                              solver='cd', verbose=True)
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)
    assert stc.vertices[1][0] in label.vertices
    assert_var_exp_log(log.getvalue(), 51, 53)  # 51.8
    assert stc.data.min() < -1e-9  # signed
    assert_stc_res(evoked_l21, stc, forward, res)

    # irMxNE tests
    with pytest.warns(None), catch_logging() as log:  # CD
        stc, residual = mixed_norm(
            evoked_l21, forward, cov, alpha, n_mxne_iter=5, loose=0.0001,
            depth=depth, maxit=300, tol=1e-8, active_set_size=10,
            solver='cd', return_residual=True, pick_ori='vector', verbose=True)
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)
    assert stc.vertices[1][0] in label.vertices
    assert stc.vertices == [[63152], [79017]]
    assert_var_exp_log(log.getvalue(), 51, 53)  # 51.8
    assert_stc_res(evoked_l21, stc, forward, residual)

    # Do with TF-MxNE for test memory savings
    alpha = 60.  # overall regularization parameter
    l1_ratio = 0.01  # temporal regularization proportion

    stc, _ = tf_mixed_norm(evoked, forward, cov,
                           loose=loose, depth=depth, maxit=100, tol=1e-4,
                           tstep=4, wsize=16, window=0.1, weights=stc_dspm,
                           weights_min=weights_min, return_residual=True,
                           alpha=alpha, l1_ratio=l1_ratio)
    assert_array_almost_equal(stc.times, evoked.times, 5)
    assert stc.vertices[1][0] in label.vertices

    # vector
    stc_nrm = tf_mixed_norm(
        evoked, forward, cov, loose=1, depth=depth, maxit=2, tol=1e-4,
        tstep=4, wsize=16, window=0.1, weights=stc_dspm,
        weights_min=weights_min, alpha=alpha, l1_ratio=l1_ratio)
    stc_vec, residual = tf_mixed_norm(
        evoked, forward, cov, loose=1, depth=depth, maxit=2, tol=1e-4,
        tstep=4, wsize=16, window=0.1, weights=stc_dspm,
        weights_min=weights_min, alpha=alpha, l1_ratio=l1_ratio,
        pick_ori='vector', return_residual=True)
    assert_stcs_equal(stc_vec.magnitude(), stc_nrm)

    pytest.raises(ValueError, tf_mixed_norm, evoked, forward, cov,
                  alpha=101, l1_ratio=0.03)
    pytest.raises(ValueError, tf_mixed_norm, evoked, forward, cov,
                  alpha=50., l1_ratio=1.01)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_mxne_vol_sphere():
    """Test (TF-)MxNE with a sphere forward and volumic source space."""
    evoked = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked.crop(tmin=-0.05, tmax=0.2)
    cov = read_cov(fname_cov)

    evoked_l21 = evoked.copy()
    evoked_l21.crop(tmin=0.081, tmax=0.1)

    info = evoked.info
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(subject=None, pos=15., mri=None,
                                        sphere=(0.0, 0.0, 0.0, 0.08),
                                        bem=None, mindist=5.0,
                                        exclude=2.0, sphere_units='m')
    fwd = mne.make_forward_solution(info, trans=None, src=src,
                                    bem=sphere, eeg=False, meg=True)

    alpha = 80.
    pytest.raises(ValueError, mixed_norm, evoked, fwd, cov, alpha,
                  loose=0.0, return_residual=False,
                  maxit=3, tol=1e-8, active_set_size=10)

    pytest.raises(ValueError, mixed_norm, evoked, fwd, cov, alpha,
                  loose=0.2, return_residual=False,
                  maxit=3, tol=1e-8, active_set_size=10)

    # irMxNE tests
    with catch_logging() as log:
        stc = mixed_norm(evoked_l21, fwd, cov, alpha,
                         n_mxne_iter=1, maxit=30, tol=1e-8,
                         active_set_size=10, verbose=True)
    assert isinstance(stc, VolSourceEstimate)
    assert_array_almost_equal(stc.times, evoked_l21.times, 5)
    assert_var_exp_log(log.getvalue(), 9, 11)  # 10.2

    # Compare orientation obtained using fit_dipole and gamma_map
    # for a simulated evoked containing a single dipole
    stc = mne.VolSourceEstimate(50e-9 * np.random.RandomState(42).randn(1, 4),
                                vertices=[stc.vertices[0][:1]],
                                tmin=stc.tmin,
                                tstep=stc.tstep)
    evoked_dip = mne.simulation.simulate_evoked(fwd, stc, info, cov, nave=1e9,
                                                use_cps=True)

    dip_mxne = mixed_norm(evoked_dip, fwd, cov, alpha=80,
                          n_mxne_iter=1, maxit=30, tol=1e-8,
                          active_set_size=10, return_as_dipoles=True)

    amp_max = [np.max(d.amplitude) for d in dip_mxne]
    dip_mxne = dip_mxne[np.argmax(amp_max)]
    assert dip_mxne.pos[0] in src[0]['rr'][stc.vertices[0]]

    dip_fit = mne.fit_dipole(evoked_dip, cov, sphere)[0]
    assert np.abs(np.dot(dip_fit.ori[0], dip_mxne.ori[0])) > 0.99
    dist = 1000 * np.linalg.norm(dip_fit.pos[0] - dip_mxne.pos[0])
    assert dist < 4.  # within 4 mm

    # Do with TF-MxNE for test memory savings
    alpha = 60.  # overall regularization parameter
    l1_ratio = 0.01  # temporal regularization proportion

    stc, _ = tf_mixed_norm(evoked, fwd, cov, maxit=3, tol=1e-4,
                           tstep=16, wsize=32, window=0.1, alpha=alpha,
                           l1_ratio=l1_ratio, return_residual=True)
    assert isinstance(stc, VolSourceEstimate)
    assert_array_almost_equal(stc.times, evoked.times, 5)


@pytest.mark.parametrize('mod', (
    None, 'mult', 'augment', 'sign', 'zero', 'less'))
def test_split_gof_basic(mod):
    """Test splitting the goodness of fit."""
    # first a trivial case
    gain = np.array([[0., 1., 1.], [1., 1., 0.]]).T
    M = np.ones((3, 1))
    X = np.ones((2, 1))
    M_est = gain @ X
    assert_allclose(M_est, np.array([[1., 2., 1.]]).T)  # a reasonable estimate
    if mod == 'mult':
        gain *= [1., -0.5]
        X[1] *= -2
    elif mod == 'augment':
        gain = np.concatenate((gain, np.zeros((3, 1))), axis=1)
        X = np.concatenate((X, [[1.]]))
    elif mod == 'sign':
        gain[1] *= -1
        M[1] *= -1
        M_est[1] *= -1
    elif mod in ('zero', 'less'):
        gain = np.array([[1, 1., 1.], [1., 1., 1.]]).T
        if mod == 'zero':
            X[:, 0] = [1., 0.]
        else:
            X[:, 0] = [1., 0.5]
        M_est = gain @ X
    else:
        assert mod is None
    res = M - M_est
    gof = 100 * (1. - (res * res).sum() / (M * M).sum())
    gof_split = _split_gof(M, X, gain)
    assert_allclose(gof_split.sum(), gof)
    want = gof_split[[0, 0]]
    if mod == 'augment':
        want = np.concatenate((want, [[0]]))
    if mod in ('mult', 'less'):
        assert_array_less(gof_split[1], gof_split[0])
    elif mod == 'zero':
        assert_allclose(gof_split[0], gof_split.sum(0))
        assert_allclose(gof_split[1], 0., atol=1e-6)
    else:
        assert_allclose(gof_split, want, atol=1e-12)


@testing.requires_testing_data
@pytest.mark.parametrize('idx, weights', [
    # empirically determined approximately orthogonal columns: 0, 15157, 19448
    ([0], [1]),
    ([0, 15157], [1, 1]),
    ([0, 15157], [1, 3]),
    ([0, 15157], [5, -1]),
    ([0, 15157, 19448], [1, 1, 1]),
    ([0, 15157, 19448], [1e-2, 1, 5]),
])
def test_split_gof_meg(forward, idx, weights):
    """Test GOF splitting on MEG data."""
    gain = forward['sol']['data'][:, idx]
    # close to orthogonal
    norms = np.linalg.norm(gain, axis=0)
    triu = np.triu_indices(len(idx), 1)
    prods = np.abs(np.dot(gain.T, gain) / np.outer(norms, norms))[triu]
    assert_array_less(prods, 5e-3)  # approximately orthogonal
    # first, split across time (one dipole per time point)
    M = gain * weights
    gof_split = _split_gof(M, np.diag(weights), gain)
    assert_allclose(gof_split.sum(0), 100., atol=1e-5)  # all sum to 100
    assert_allclose(gof_split, 100 * np.eye(len(weights)), atol=1)  # loc
    # next, summed to a single time point (all dipoles active at one time pt)
    weights = np.array(weights)[:, np.newaxis]
    x = gain @ weights
    assert x.shape == (gain.shape[0], 1)
    gof_split = _split_gof(x, weights, gain)
    want = (norms * weights.T).T ** 2
    want = 100 * want / want.sum()
    assert_allclose(gof_split, want, atol=1e-3, rtol=1e-2)
    assert_allclose(gof_split.sum(), 100, rtol=1e-5)


run_tests_if_main()
