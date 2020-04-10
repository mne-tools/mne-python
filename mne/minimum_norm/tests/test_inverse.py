import os.path as op
from pathlib import Path
import re

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_allclose, assert_array_equal,
                           assert_array_less)
from scipy import sparse

import pytest
import copy

import mne
from mne.datasets import testing
from mne.label import read_label, label_sign_flip
from mne.event import read_events
from mne.epochs import Epochs
from mne.forward import restrict_forward_to_stc, apply_forward, is_fixed_orient
from mne.source_estimate import read_source_estimate, VolSourceEstimate
from mne.source_space import _get_src_nn
from mne.surface import _normal_orth
from mne import (read_cov, read_forward_solution, read_evokeds, pick_types,
                 pick_types_forward, make_forward_solution, EvokedArray,
                 convert_forward_solution, Covariance, combine_evoked,
                 SourceEstimate, make_sphere_model, make_ad_hoc_cov,
                 pick_channels_forward, compute_raw_covariance)
from mne.io import read_raw_fif
from mne.io.proj import make_projector
from mne.minimum_norm import (apply_inverse, read_inverse_operator,
                              apply_inverse_raw, apply_inverse_epochs,
                              make_inverse_operator, apply_inverse_cov,
                              write_inverse_operator, prepare_inverse_operator,
                              compute_rank_inverse, INVERSE_METHODS)
from mne.utils import _TempDir, run_tests_if_main, catch_logging

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
# Four inverses:
fname_full = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_inv = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
fname_inv_fixed_nodepth = op.join(s_path,
                                  'sample_audvis_trunc-meg-eeg-oct-4-meg'
                                  '-nodepth-fixed-inv.fif')
fname_inv_fixed_depth = op.join(s_path,
                                'sample_audvis_trunc-meg-eeg-oct-4-meg'
                                '-fixed-inv.fif')
fname_inv_meeg_diag = op.join(s_path,
                              'sample_audvis_trunc-'
                              'meg-eeg-oct-4-meg-eeg-diagnoise-inv.fif')

fname_data = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_raw = op.join(s_path, 'sample_audvis_trunc_raw.fif')
fname_raw_ctf = op.join(test_path, 'CTF', 'somMDYO-18av.ds')
fname_event = op.join(s_path, 'sample_audvis_trunc_raw-eve.fif')
fname_label = op.join(s_path, 'labels', '%s.label')
fname_vol_inv = op.join(s_path,
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
# trans and bem needed for channel reordering tests incl. forward computation
fname_trans = op.join(s_path, 'sample_audvis_trunc-trans.fif')
s_path_bem = op.join(test_path, 'subjects', 'sample', 'bem')
fname_bem = op.join(s_path_bem, 'sample-320-320-320-bem-sol.fif')
src_fname = op.join(s_path_bem, 'sample-oct-4-src.fif')

snr = 3.0
lambda2 = 1.0 / snr ** 2

last_keys = [None] * 10


def read_forward_solution_meg(fname, **kwargs):
    """Read MEG forward."""
    fwd = convert_forward_solution(read_forward_solution(fname), copy=False,
                                   **kwargs)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    return fwd


def read_forward_solution_eeg(fname, **kwargs):
    """Read EEG forward."""
    fwd = convert_forward_solution(read_forward_solution(fname), copy=False,
                                   **kwargs)
    fwd = pick_types_forward(fwd, meg=False, eeg=True)
    return fwd


def _compare(a, b):
    """Compare two python objects."""
    global last_keys
    skip_types = ['whitener', 'proj', 'reginv', 'noisenorm', 'nchan',
                  'command_line', 'working_dir', 'mri_file', 'mri_id',
                  'scanno']
    try:
        if isinstance(a, dict):
            assert isinstance(b, dict)
            for k, v in a.items():
                if k not in b and k not in skip_types:
                    raise ValueError('First one had one second one didn\'t:\n'
                                     '%s not in %s' % (k, b.keys()))
                if k not in skip_types:
                    last_keys.pop()
                    last_keys = [k] + last_keys
                    _compare(v, b[k])
            for k in b.keys():
                if k not in a and k not in skip_types:
                    raise ValueError('Second one had one first one didn\'t:\n'
                                     '%s not in %s' % (k, sorted(a.keys())))
        elif isinstance(a, list):
            assert (len(a) == len(b))
            for i, j in zip(a, b):
                _compare(i, j)
        elif isinstance(a, sparse.csr.csr_matrix):
            assert_array_almost_equal(a.data, b.data)
            assert_equal(a.indices, b.indices)
            assert_equal(a.indptr, b.indptr)
        elif isinstance(a, np.ndarray):
            assert_array_almost_equal(a, b)
        else:
            assert a == b
    except Exception:
        print(last_keys)
        raise


def _compare_inverses_approx(inv_1, inv_2, evoked, rtol, atol,
                             depth_atol=1e-6, ctol=0.999999,
                             check_nn=True, check_K=True):
    """Compare inverses."""
    # depth prior
    if inv_1['depth_prior'] is not None:
        assert_allclose(inv_1['depth_prior']['data'],
                        inv_2['depth_prior']['data'], atol=depth_atol)
    else:
        assert (inv_2['depth_prior'] is None)
    # orient prior
    if inv_1['orient_prior'] is not None:
        assert_allclose(inv_1['orient_prior']['data'],
                        inv_2['orient_prior']['data'], atol=1e-7)
    else:
        assert (inv_2['orient_prior'] is None)
    # source cov
    assert_allclose(inv_1['source_cov']['data'], inv_2['source_cov']['data'],
                    atol=1e-7)
    for key in ('units', 'eigen_leads_weighted', 'nsource', 'coord_frame'):
        assert_equal(inv_1[key], inv_2[key], err_msg=key)
    assert_equal(inv_1['eigen_leads']['ncol'], inv_2['eigen_leads']['ncol'])
    K_1 = np.dot(inv_1['eigen_leads']['data'] * inv_1['sing'].astype(float),
                 inv_1['eigen_fields']['data'])
    K_2 = np.dot(inv_2['eigen_leads']['data'] * inv_2['sing'].astype(float),
                 inv_2['eigen_fields']['data'])
    # for free + surf ori, we only care about the ::2
    # (the other two dimensions have arbitrary direction)
    if inv_1['nsource'] * 3 == inv_1['source_nn'].shape[0]:
        # Technically this undersamples the free-orientation, non-surf-ori
        # inverse, but it's probably okay
        sl = slice(2, None, 3)
    else:
        sl = slice(None)
    if check_nn:
        assert_allclose(inv_1['source_nn'][sl], inv_2['source_nn'][sl],
                        atol=1e-4)
    if check_K:
        assert_allclose(np.abs(K_1[sl]), np.abs(K_2[sl]), rtol=rtol, atol=atol)

    # Now let's do some practical tests, too
    evoked = EvokedArray(np.eye(len(evoked.ch_names)), evoked.info)
    for method in ('MNE', 'dSPM'):
        stc_1 = apply_inverse(evoked, inv_1, lambda2, method)
        stc_2 = apply_inverse(evoked, inv_2, lambda2, method)
        assert_equal(stc_1.subject, stc_2.subject)
        assert_equal(stc_1.times, stc_2.times)
        stc_1 = stc_1.data
        stc_2 = stc_2.data
        norms = np.max(stc_1, axis=-1, keepdims=True)
        stc_1 /= norms
        stc_2 /= norms
        corr = np.corrcoef(stc_1.ravel(), stc_2.ravel())[0, 1]
        assert corr > ctol
        assert_allclose(stc_1, stc_2, rtol=rtol, atol=atol,
                        err_msg='%s: %s' % (method, corr))


def _compare_io(inv_op, out_file_ext='.fif'):
    """Compare inverse IO."""
    tempdir = _TempDir()
    if out_file_ext == '.fif':
        out_file = op.join(tempdir, 'test-inv.fif')
    elif out_file_ext == '.gz':
        out_file = op.join(tempdir, 'test-inv.fif.gz')
    else:
        raise ValueError('IO test could not complete')
    out_file = Path(out_file)
    # Test io operations
    inv_init = copy.deepcopy(inv_op)
    write_inverse_operator(out_file, inv_op)
    read_inv_op = read_inverse_operator(out_file)
    _compare(inv_init, read_inv_op)
    _compare(inv_init, inv_op)


def test_warn_inverse_operator(evoked, noise_cov):
    """Test MNE inverse warning without average EEG projection."""
    bad_info = evoked.info
    bad_info['projs'] = list()
    fwd_op = convert_forward_solution(read_forward_solution(fname_fwd),
                                      surf_ori=True, copy=False)
    noise_cov['projs'].pop(-1)  # get rid of avg EEG ref proj
    with pytest.warns(RuntimeWarning, match='reference'):
        make_inverse_operator(bad_info, fwd_op, noise_cov)


@pytest.mark.slowtest
def test_make_inverse_operator_loose(evoked):
    """Test MNE inverse computation (precomputed and non-precomputed)."""
    # Test old version of inverse computation starting from forward operator
    noise_cov = read_cov(fname_cov)
    inverse_operator = read_inverse_operator(fname_inv)
    fwd_op = convert_forward_solution(read_forward_solution_meg(fname_fwd),
                                      surf_ori=True, copy=False)
    with catch_logging() as log:
        my_inv_op = make_inverse_operator(
            evoked.info, fwd_op, noise_cov, loose=0.2,
            depth=dict(exp=0.8, limit_depth_chs=False), verbose=True)
    log = log.getvalue()
    assert 'MEG: rank 302 computed' in log
    assert 'limit = 1/%d' % fwd_op['nsource'] in log
    _compare_io(my_inv_op)
    assert_equal(inverse_operator['units'], 'Am')
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked,
                             rtol=1e-2, atol=1e-5, depth_atol=1e-3)
    # Test MNE inverse computation starting from forward operator
    with catch_logging() as log:
        my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                          loose='auto', depth=0.8,
                                          fixed=False, verbose=True)
    log = log.getvalue()
    assert 'MEG: rank 302 computed from 305' in log
    _compare_io(my_inv_op)
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked,
                             rtol=1e-3, atol=1e-5)
    assert ('dev_head_t' in my_inv_op['info'])
    assert ('mri_head_t' in my_inv_op)


@pytest.mark.slowtest
def test_inverse_operator_channel_ordering(evoked, noise_cov):
    """Test MNE inverse computation is immune to channel reorderings."""
    # These are with original ordering
    evoked_orig = evoked.copy()
    fwd_orig = make_forward_solution(evoked.info, fname_trans, src_fname,
                                     fname_bem, eeg=True, mindist=5.0)
    fwd_orig = convert_forward_solution(fwd_orig, surf_ori=True)
    depth = dict(exp=0.8, limit_depth_chs=False)
    with catch_logging() as log:
        inv_orig = make_inverse_operator(evoked.info, fwd_orig, noise_cov,
                                         loose=0.2, depth=depth, verbose=True)
    log = log.getvalue()
    assert 'limit = 1/%s' % fwd_orig['nsource'] in log
    stc_1 = apply_inverse(evoked, inv_orig, lambda2, "dSPM")

    # Assume that a raw reordering applies to both evoked and noise_cov,
    # so we don't need to create those from scratch. Just reorder them,
    # then try to apply the original inverse operator
    new_order = np.arange(len(evoked.info['ch_names']))
    randomiser = np.random.RandomState(42)
    randomiser.shuffle(new_order)
    evoked.data = evoked.data[new_order]
    evoked.info['chs'] = [evoked.info['chs'][n] for n in new_order]
    evoked.info._update_redundant()
    evoked.info._check_consistency()

    cov_ch_reorder = [c for c in evoked.info['ch_names']
                      if (c in noise_cov.ch_names)]

    new_order_cov = [noise_cov.ch_names.index(name) for name in cov_ch_reorder]
    noise_cov['data'] = noise_cov.data[np.ix_(new_order_cov, new_order_cov)]
    noise_cov['names'] = [noise_cov['names'][idx] for idx in new_order_cov]

    fwd_reorder = make_forward_solution(evoked.info, fname_trans, src_fname,
                                        fname_bem, eeg=True, mindist=5.0)
    fwd_reorder = convert_forward_solution(fwd_reorder, surf_ori=True)
    inv_reorder = make_inverse_operator(evoked.info, fwd_reorder, noise_cov,
                                        loose=0.2, depth=depth)

    stc_2 = apply_inverse(evoked, inv_reorder, lambda2, "dSPM")

    assert_equal(stc_1.subject, stc_2.subject)
    assert_array_equal(stc_1.times, stc_2.times)
    assert_allclose(stc_1.data, stc_2.data, rtol=1e-5, atol=1e-5)
    assert (inv_orig['units'] == inv_reorder['units'])

    # Reload with original ordering & apply reordered inverse
    evoked = evoked_orig
    noise_cov = read_cov(fname_cov)

    stc_3 = apply_inverse(evoked, inv_reorder, lambda2, "dSPM")
    assert_allclose(stc_1.data, stc_3.data, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('method, lower, upper, depth', [
    ('MNE', 54, 57, dict(limit=None, combine_xyz=False, exp=1.)),  # DICS def
    ('MNE', 75, 80, dict(limit_depth_chs=False)),  # ancient MNE default
    ('MNE', 83, 87, 0.8),  # MNE default
    ('MNE', 89, 92, dict(limit_depth_chs='whiten')),  # sparse default
    ('dSPM', 96, 98, 0.8),
    ('sLORETA', 100, 100, 0.8),
    ('eLORETA', 100, 100, None),
    ('eLORETA', 100, 100, 0.8),
])
def test_localization_bias_fixed(bias_params_fixed, method, lower, upper,
                                 depth):
    """Test inverse localization bias for fixed minimum-norm solvers."""
    evoked, fwd, noise_cov, _, want = bias_params_fixed
    fwd_use = convert_forward_solution(fwd, force_fixed=False)
    inv_fixed = make_inverse_operator(evoked.info, fwd_use, noise_cov,
                                      loose=0., depth=depth)
    loc = np.abs(apply_inverse(evoked, inv_fixed, lambda2, method,
                               verbose='debug').data)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper, method


@pytest.mark.parametrize('method, lower, upper, depth, loose', [
    ('MNE', 32, 36, dict(limit=None, combine_xyz=False, exp=1.), 0.2),  # DICS
    ('MNE', 78, 81, 0.8, 0.2),  # MNE default
    ('MNE', 89, 92, dict(limit_depth_chs='whiten'), 0.2),  # sparse default
    ('dSPM', 85, 87, 0.8, 0.2),
    ('sLORETA', 100, 100, 0.8, 0.2),
    ('eLORETA', 99, 100, None, 0.2),
    ('eLORETA', 99, 100, 0.8, 0.2),
    ('eLORETA', 99, 100, 0.8, 0.001),
])
def test_localization_bias_loose(bias_params_fixed, method, lower, upper,
                                 depth, loose):
    """Test inverse localization bias for loose minimum-norm solvers."""
    evoked, fwd, noise_cov, _, want = bias_params_fixed
    fwd = convert_forward_solution(fwd, surf_ori=False, force_fixed=False)
    assert not is_fixed_orient(fwd)
    inv_loose = make_inverse_operator(evoked.info, fwd, noise_cov, loose=loose,
                                      depth=depth)
    loc = apply_inverse(evoked, inv_loose, lambda2, method).data
    assert (loc >= 0).all()
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper, method


@pytest.mark.parametrize('method, lower, upper, kwargs, depth, loose', [
    ('MNE', 21, 24, {}, dict(limit=None, combine_xyz=False, exp=1.), 1),
    ('MNE', 35, 40, {}, dict(limit_depth_chs=False), 1),  # ancient default
    ('MNE', 45, 55, {}, 0.8, 1),  # MNE default
    ('MNE', 65, 70, {}, dict(limit_depth_chs='whiten'), 1),  # sparse default
    ('dSPM', 40, 45, {}, 0.8, 1),
    ('sLORETA', 90, 95, {}, 0.8, 1),
    ('eLORETA', 93, 100, dict(method_params=dict(force_equal=True)), None, 1),
    ('eLORETA', 100, 100, {}, None, 1.0),
    ('eLORETA', 100, 100, {}, 0.8, 1.0),
    ('eLORETA', 100, 100, {}, 0.8, 0.999),
])
def test_localization_bias_free(bias_params_free, method, lower, upper,
                                kwargs, depth, loose):
    """Test inverse localization bias for free minimum-norm solvers."""
    evoked, fwd, noise_cov, _, want = bias_params_free
    inv_free = make_inverse_operator(evoked.info, fwd, noise_cov, loose=1.,
                                     depth=depth)
    loc = apply_inverse(evoked, inv_free, lambda2, method,
                        pick_ori='vector', verbose='debug', **kwargs).data
    loc = np.linalg.norm(loc, axis=1)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper, method


def test_apply_inverse_sphere(evoked):
    """Test applying an inverse with a sphere model (rank-deficient)."""
    evoked.pick_channels(evoked.ch_names[:306:8])
    evoked.info['projs'] = []
    cov = make_ad_hoc_cov(evoked.info)
    sphere = make_sphere_model('auto', 'auto', evoked.info)
    fwd = read_forward_solution(fname_fwd)
    vertices = [fwd['src'][0]['vertno'][::5],
                fwd['src'][1]['vertno'][::5]]
    stc = SourceEstimate(np.zeros((sum(len(v) for v in vertices), 1)),
                         vertices, 0., 1.)
    fwd = restrict_forward_to_stc(fwd, stc)
    fwd = make_forward_solution(evoked.info, fwd['mri_head_t'], fwd['src'],
                                sphere, mindist=5.)
    evoked = EvokedArray(fwd['sol']['data'].copy(), evoked.info)
    assert fwd['sol']['nrow'] == 39
    assert fwd['nsource'] == 101
    assert fwd['sol']['ncol'] == 303
    tempdir = _TempDir()
    temp_fname = op.join(tempdir, 'temp-inv.fif')
    inv = make_inverse_operator(evoked.info, fwd, cov, loose=1.)
    # This forces everything to be float32
    write_inverse_operator(temp_fname, inv)
    inv = read_inverse_operator(temp_fname)
    stc = apply_inverse(evoked, inv, method='eLORETA',
                        method_params=dict(eps=1e-2))
    # assert zero localization bias
    assert_array_equal(np.argmax(stc.data, axis=0),
                       np.repeat(np.arange(101), 3))


@pytest.mark.parametrize('loose', [0., 0.2, 1.])
@pytest.mark.parametrize('lambda2', [1. / 9., 0.])
def test_apply_inverse_eLORETA_MNE_equiv(bias_params_free, loose, lambda2):
    """Test that eLORETA with no iterations is the same as MNE."""
    method_params = dict(max_iter=0, force_equal=False)
    pick_ori = None if loose == 0 else 'vector'
    evoked, fwd, noise_cov, _, _ = bias_params_free
    inv = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=loose, depth=None,
        verbose='debug')
    stc_mne = apply_inverse(evoked, inv, lambda2, 'MNE', pick_ori=pick_ori,
                            verbose='debug')
    with pytest.warns(RuntimeWarning, match='converge'):
        stc_e = apply_inverse(evoked, inv, lambda2, 'eLORETA',
                              method_params=method_params, pick_ori=pick_ori,
                              verbose='debug')
    atol = np.mean(np.abs(stc_mne.data)) * 1e-6
    assert 3e-9 < atol < 3e-6  # nothing has blown up
    assert_allclose(stc_mne.data, stc_e.data, atol=atol, rtol=1e-4)


@pytest.mark.slowtest
@pytest.mark.parametrize('inv, min_, max_', [
    (fname_inv, 0, 13e-9),
    (fname_inv_fixed_depth, -25e-9, 25e-9),
])
def test_apply_inverse_operator(evoked, inv, min_, max_):
    """Test MNE inverse application."""
    # use fname_inv as it will be faster than fname_full (fewer verts and chs)
    inverse_operator = read_inverse_operator(inv)

    # Inverse has 306 channels - 4 proj = 302
    assert (compute_rank_inverse(inverse_operator) == 302)

    # Inverse has 306 channels - 4 proj = 302
    assert (compute_rank_inverse(inverse_operator) == 302)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "MNE")
    assert stc.subject == 'sample'
    assert stc.data.min() > min_
    assert stc.data.max() < max_
    assert abs(stc).data.mean() > 1e-11

    # test if using prepared and not prepared inverse operator give the same
    # result
    inv_op = prepare_inverse_operator(inverse_operator, nave=evoked.nave,
                                      lambda2=lambda2, method="MNE")
    stc2 = apply_inverse(evoked, inv_op, lambda2, "MNE")
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)

    # This is little more than a smoke test...
    stc = apply_inverse(evoked, inverse_operator, lambda2, "sLORETA")
    assert stc.subject == 'sample'
    assert abs(stc).data.min() > 0
    assert 2 < stc.data.max() < 7
    assert abs(stc).data.mean() > 0.1

    stc = apply_inverse(evoked, inverse_operator, lambda2, "eLORETA")
    assert stc.subject == 'sample'
    assert abs(stc).data.min() > min_
    assert stc.data.max() < max_ * 2
    assert abs(stc).data.mean() > 1e-11

    stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM")
    assert stc.subject == 'sample'
    assert abs(stc).data.min() > 0
    assert 7.5 < stc.data.max() < 15
    assert abs(stc).data.mean() > 0.1

    # test without using a label (so delayed computation is used)
    label = read_label(fname_label % 'Aud-lh')
    for method in INVERSE_METHODS:
        stc = apply_inverse(evoked, inv_op, lambda2, method)
        stc_label = apply_inverse(evoked, inv_op, lambda2, method,
                                  label=label)
        assert_equal(stc_label.subject, 'sample')
        label_stc = stc.in_label(label)
        assert label_stc.subject == 'sample'
        assert_allclose(stc_label.data, label_stc.data)

    # Test that no errors are raised with loose inverse ops and picking normals
    noise_cov = read_cov(fname_cov)
    fwd = read_forward_solution_meg(fname_fwd)
    inv_op_meg = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=1,
        fixed='auto', depth=None)
    apply_inverse(evoked, inv_op_meg, 1 / 9., method='MNE', pick_ori='normal')

    # Test we get errors when using custom ref or no average proj is present
    evoked.info['custom_ref_applied'] = True
    pytest.raises(ValueError, apply_inverse, evoked, inv_op, lambda2, "MNE")
    evoked.info['custom_ref_applied'] = False
    evoked.info['projs'] = []  # remove EEG proj
    pytest.raises(ValueError, apply_inverse, evoked, inv_op, lambda2, "MNE")

    # But test that we do not get EEG-related errors on MEG-only inv (gh-4650)
    apply_inverse(evoked, inv_op_meg, 1. / 9.)


@pytest.mark.parametrize('method', INVERSE_METHODS)
@pytest.mark.parametrize('looses, vmin, vmax, nmin, nmax', [
    ((1., 0.8), 0.87, 0.94, 0.9, 1.1),  # almost the same as free
    ((0., 0.2), 0.3, 0.6, 2, 4),  # similar to fixed
])
def test_orientation_prior(bias_params_free, method, looses, vmin, vmax,
                           nmin, nmax):
    """Test that orientation priors are handled properly."""
    evoked, fwd, noise_cov, _, _ = bias_params_free
    stcs = list()
    vec_stc = None
    for loose in looses:
        inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=loose)
        if looses[0] == 0.:
            pick_ori = None if loose == 0 else 'normal'
        else:
            pick_ori = 'vector'
        stcs.append(apply_inverse(
            evoked, inv, method=method, pick_ori=pick_ori))
        if loose in (1., 0.2):
            assert vec_stc is None
            vec_stc = apply_inverse(
                evoked, inv, method=method, pick_ori='vector')
    assert vec_stc is not None
    rot = _normal_orth(np.concatenate(
        [_get_src_nn(s) for s in inv['src']]))
    vec_stc_surf = np.matmul(rot, vec_stc.data)
    if 0. in looses:
        vec_stc_normal = vec_stc.normal(inv['src'])
        assert_allclose(stcs[1].data, vec_stc_normal.data)
        del vec_stc
        assert_allclose(vec_stc_normal.data, vec_stc_surf[:, 2])
        assert_allclose(vec_stc_normal.data, stcs[1].data)
    # Ensure that our relative strengths are reasonable
    # (normal should be much larger than tangential)
    normal = np.linalg.norm(vec_stc_surf[:, 2].ravel())
    for ii in range(2):
        tangential = np.linalg.norm(vec_stc_surf[:, ii].ravel())
        ratio = normal / tangential
        assert nmin < ratio < nmax
    assert stcs[0].data.shape == stcs[1].data.shape
    R2 = 1. - (
        np.linalg.norm(stcs[0].data.ravel() - stcs[1].data.ravel()) /
        np.linalg.norm(stcs[0].data.ravel()))
    assert vmin < R2 < vmax


@pytest.mark.parametrize('method', INVERSE_METHODS)
def test_inverse_residual(evoked, method):
    """Test MNE inverse application."""
    # use fname_inv as it will be faster than fname_full (fewer verts and chs)
    evoked = evoked.pick_types()
    inv = read_inverse_operator(fname_inv_fixed_depth)
    fwd = read_forward_solution(fname_fwd)
    pick_channels_forward(fwd, evoked.ch_names, copy=False)
    fwd = convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
    matcher = re.compile(r'.* ([0-9]?[0-9]?[0-9]?\.[0-9])% variance.*')

    with catch_logging() as log:
        stc, residual = apply_inverse(
            evoked, inv, method=method, return_residual=True, verbose=True)
    log = log.getvalue()
    match = matcher.match(log.replace('\n', ' '))
    assert match is not None
    match = float(match.group(1))
    assert 45 < match < 50
    if method not in ('dSPM', 'sLORETA'):
        recon = apply_forward(fwd, stc, evoked.info)
        proj_op = make_projector(evoked.info['projs'], evoked.ch_names)[0]
        recon.data[:] = np.dot(proj_op, recon.data)
        residual_fwd = evoked.copy()
        residual_fwd.data -= recon.data
        corr = np.corrcoef(residual_fwd.data.ravel(),
                           residual.data.ravel())[0, 1]
        assert corr > 0.999

    if method != 'sLORETA':  # XXX divide by zero error
        with catch_logging() as log:
            _, residual = apply_inverse(
                evoked, inv, 0., method, return_residual=True, verbose=True)
        log = log.getvalue()
        match = matcher.match(log.replace('\n', ' '))
        assert match is not None
        match = float(match.group(1))
        assert match == 100.
        assert_array_less(np.abs(residual.data), 1e-15)


def test_make_inverse_operator_fixed(evoked, noise_cov):
    """Test MNE inverse computation (fixed orientation)."""
    fwd = read_forward_solution_meg(fname_fwd)

    # can't make fixed inv with depth weighting without free ori fwd
    fwd_fixed = convert_forward_solution(fwd, force_fixed=True,
                                         use_cps=True)
    pytest.raises(ValueError, make_inverse_operator, evoked.info, fwd_fixed,
                  noise_cov, depth=0.8, fixed=True)

    # now compare to C solution
    # note that the forward solution must not be surface-oriented
    # to get equivalence (surf_ori=True changes the normals)
    with catch_logging() as log:
        inv_op = make_inverse_operator(  # test depth=0. alias for depth=None
            evoked.info, fwd, noise_cov, depth=0., fixed=True,
            use_cps=False, verbose=True)
    log = log.getvalue()
    assert 'MEG: rank 302 computed from 305' in log
    assert 'EEG channels: 0' in repr(inv_op)
    assert 'MEG channels: 305' in repr(inv_op)
    del fwd_fixed
    inverse_operator_nodepth = read_inverse_operator(fname_inv_fixed_nodepth)
    # XXX We should have this but we don't (MNE-C doesn't restrict info):
    # assert 'EEG channels: 0' in repr(inverse_operator_nodepth)
    assert 'MEG channels: 305' in repr(inverse_operator_nodepth)
    _compare_inverses_approx(inverse_operator_nodepth, inv_op, evoked,
                             rtol=1e-5, atol=1e-4)
    # Inverse has 306 channels - 6 proj = 302
    assert (compute_rank_inverse(inverse_operator_nodepth) == 302)
    # Now with depth
    fwd_surf = convert_forward_solution(fwd, surf_ori=True)  # not fixed
    for kwargs, use_fwd in zip([dict(fixed=True), dict(loose=0.)],
                               [fwd, fwd_surf]):  # Should be equiv.
        inv_op_depth = make_inverse_operator(
            evoked.info, use_fwd, noise_cov, depth=0.8, use_cps=True,
            **kwargs)
        inverse_operator_depth = read_inverse_operator(fname_inv_fixed_depth)
        # Normals should be the adjusted ones
        assert_allclose(inverse_operator_depth['source_nn'],
                        fwd_surf['source_nn'][2::3], atol=1e-5)
        _compare_inverses_approx(inverse_operator_depth, inv_op_depth, evoked,
                                 rtol=1e-3, atol=1e-4)


def test_make_inverse_operator_free(evoked, noise_cov):
    """Test MNE inverse computation (free orientation)."""
    fwd = read_forward_solution_meg(fname_fwd)
    fwd_surf = convert_forward_solution(fwd, surf_ori=True)
    fwd_fixed = convert_forward_solution(fwd, force_fixed=True,
                                         use_cps=True)

    # can't make free inv with fixed fwd
    with pytest.raises(ValueError, match='can only be used'):
        make_inverse_operator(evoked.info, fwd_fixed, noise_cov, depth=None)

    # for depth=None (or depth=0.8), surf_ori of the fwd should not matter
    inv_surf = make_inverse_operator(evoked.info, fwd_surf, noise_cov,
                                     depth=None, loose=1.)
    inv = make_inverse_operator(evoked.info, fwd, noise_cov,
                                depth=None, loose=1.)
    _compare_inverses_approx(inv, inv_surf, evoked, rtol=1e-5, atol=1e-8,
                             check_nn=False, check_K=False)
    for pick_ori in (None, 'vector', 'normal'):
        stc = apply_inverse(evoked, inv, pick_ori=pick_ori)
        stc_surf = apply_inverse(evoked, inv_surf, pick_ori=pick_ori)
        assert_allclose(stc_surf.data, stc.data, atol=1e-2)


def test_make_inverse_operator_vector(evoked, noise_cov):
    """Test MNE inverse computation (vector result)."""
    fwd_surf = read_forward_solution_meg(fname_fwd, surf_ori=True)
    fwd = read_forward_solution_meg(fname_fwd, surf_ori=False)

    # Make different version of the inverse operator
    inv_1 = make_inverse_operator(evoked.info, fwd, noise_cov, loose=1)
    inv_2 = make_inverse_operator(evoked.info, fwd_surf, noise_cov, depth=None,
                                  use_cps=True)
    inv_3 = make_inverse_operator(evoked.info, fwd_surf, noise_cov, fixed=True,
                                  use_cps=True)
    inv_4 = make_inverse_operator(evoked.info, fwd, noise_cov,
                                  loose=.2, depth=None)

    # Apply the inverse operators and check the result
    for ii, inv in enumerate((inv_1, inv_2, inv_4)):
        # Don't do eLORETA here as it will be quite slow
        methods = ['MNE', 'dSPM', 'sLORETA'] if ii < 2 else ['MNE']
        for method in methods:
            stc = apply_inverse(evoked, inv, method=method)
            stc_vec = apply_inverse(evoked, inv, pick_ori='vector',
                                    method=method)
            assert_allclose(stc.data, stc_vec.magnitude().data)

    # Vector estimates don't work when using fixed orientations
    with pytest.raises(RuntimeError, match='fixed orientation'):
        apply_inverse(evoked, inv_3, pick_ori='vector')

    # When computing with vector fields, computing the difference between two
    # evokeds and then performing the inverse should yield the same result as
    # computing the difference between the inverses.
    evoked0 = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked0.crop(0, 0.2)
    evoked1 = read_evokeds(fname_data, condition=1, baseline=(None, 0))
    evoked1.crop(0, 0.2)
    diff = combine_evoked((evoked0, evoked1), [1, -1])
    stc_diff = apply_inverse(diff, inv_1, method='MNE')
    stc_diff_vec = apply_inverse(diff, inv_1, method='MNE', pick_ori='vector')
    stc_vec0 = apply_inverse(evoked0, inv_1, method='MNE', pick_ori='vector')
    stc_vec1 = apply_inverse(evoked1, inv_1, method='MNE', pick_ori='vector')
    assert_allclose(stc_diff_vec.data, (stc_vec0 - stc_vec1).data,
                    atol=1e-20)
    assert_allclose(stc_diff.data, (stc_vec0 - stc_vec1).magnitude().data,
                    atol=1e-20)


def test_make_inverse_operator_diag(evoked, noise_cov):
    """Test MNE inverse computation with diagonal noise cov."""
    noise_cov = noise_cov.as_diag()
    fwd_op = convert_forward_solution(read_forward_solution(fname_fwd),
                                      surf_ori=True)
    inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                   loose=0.2, depth=0.8)
    _compare_io(inv_op)
    inverse_operator_diag = read_inverse_operator(fname_inv_meeg_diag)
    # This one is pretty bad
    _compare_inverses_approx(inverse_operator_diag, inv_op, evoked,
                             rtol=1e-1, atol=1e-1, ctol=0.99, check_K=False)
    # Inverse has 366 channels - 6 proj = 360
    assert (compute_rank_inverse(inverse_operator_diag) == 360)


def test_inverse_operator_noise_cov_rank(evoked, noise_cov):
    """Test MNE inverse operator with a specified noise cov rank."""
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    inv = make_inverse_operator(
        evoked.info, fwd_op, noise_cov, rank=dict(meg=64))
    assert (compute_rank_inverse(inv) == 64)
    inv = make_inverse_operator(
        evoked.info, fwd_op, noise_cov, rank=dict(meg=64))
    assert (compute_rank_inverse(inv) == 64)

    fwd_op = read_forward_solution_eeg(fname_fwd, surf_ori=True)
    inv = make_inverse_operator(
        evoked.info, fwd_op, noise_cov, rank=dict(eeg=20))
    assert (compute_rank_inverse(inv) == 20)


def test_inverse_operator_volume(evoked):
    """Test MNE inverse computation on volume source space."""
    tempdir = _TempDir()
    inv_vol = read_inverse_operator(fname_vol_inv)
    assert (repr(inv_vol))
    stc = apply_inverse(evoked, inv_vol, lambda2, 'dSPM')
    assert (isinstance(stc, VolSourceEstimate))
    # volume inverses don't have associated subject IDs
    assert (stc.subject is None)
    stc.save(op.join(tempdir, 'tmp-vl.stc'))
    stc2 = read_source_estimate(op.join(tempdir, 'tmp-vl.stc'))
    assert (np.all(stc.data > 0))
    assert (np.all(stc.data < 35))
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)
    # vector source estimate
    stc_vec = apply_inverse(evoked, inv_vol, lambda2, 'dSPM', 'vector')
    assert (repr(stc_vec))
    assert_allclose(np.linalg.norm(stc_vec.data, axis=1), stc.data)
    with pytest.raises(RuntimeError, match='surface or discrete'):
        apply_inverse(evoked, inv_vol, pick_ori='normal')


@pytest.mark.slowtest
@testing.requires_testing_data
def test_io_inverse_operator():
    """Test IO of inverse_operator."""
    tempdir = _TempDir()
    inverse_operator = read_inverse_operator(fname_inv)
    x = repr(inverse_operator)
    assert (x)
    assert (isinstance(inverse_operator['noise_cov'], Covariance))
    # just do one example for .gz, as it should generalize
    _compare_io(inverse_operator, '.gz')

    # test warnings on bad filenames
    inv_badname = op.join(tempdir, 'test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='-inv.fif'):
        write_inverse_operator(inv_badname, inverse_operator)
    with pytest.warns(RuntimeWarning, match='-inv.fif'):
        read_inverse_operator(inv_badname)

    # make sure we can write and read
    inv_fname = op.join(tempdir, 'test-inv.fif')
    args = (10, 1. / 9., 'dSPM')
    inv_prep = prepare_inverse_operator(inverse_operator, *args)
    write_inverse_operator(inv_fname, inv_prep)
    inv_read = read_inverse_operator(inv_fname)
    _compare(inverse_operator, inv_read)
    inv_read_prep = prepare_inverse_operator(inv_read, *args)
    _compare(inv_prep, inv_read_prep)
    inv_prep_prep = prepare_inverse_operator(inv_prep, *args)
    _compare(inv_prep, inv_prep_prep)


# eLORETA is slow and we can trust that it will work because we just route
# through apply_inverse
_fast_methods = list(INVERSE_METHODS)
_fast_methods.pop(_fast_methods.index('eLORETA'))


@testing.requires_testing_data
@pytest.mark.parametrize('method', _fast_methods)
@pytest.mark.parametrize('pick_ori', ['normal', None])
def test_apply_inverse_cov(method, pick_ori):
    """Test MNE with precomputed inverse operator on cov."""
    raw = read_raw_fif(fname_raw, preload=True)
    # use 10 sec of data
    raw.crop(0, 10)

    raw.filter(1, None)
    label_lh = read_label(fname_label % 'Aud-lh')

    # test with a free ori inverse
    inverse_operator = read_inverse_operator(fname_inv)

    data_cov = compute_raw_covariance(raw, tstep=None)

    with pytest.raises(ValueError, match='has not been prepared'):
        apply_inverse_cov(data_cov, raw.info, inverse_operator,
                          lambda2=lambda2, prepared=True)

    this_inv_op = prepare_inverse_operator(inverse_operator, nave=1,
                                           lambda2=lambda2, method=method)

    raw_ori = 'normal' if pick_ori == 'normal' else 'vector'
    stc_raw = apply_inverse_raw(
        raw, this_inv_op, lambda2, method, label=label_lh, nave=1,
        pick_ori=raw_ori, prepared=True)
    stc_cov = apply_inverse_cov(
        data_cov, raw.info, this_inv_op, method=method, pick_ori=pick_ori,
        label=label_lh, prepared=True, lambda2=lambda2)
    n_sources = np.prod(stc_cov.data.shape[:-1])
    raw_data = stc_raw.data.reshape(n_sources, -1)
    exp_res = np.diag(np.cov(raw_data, ddof=1)).copy()
    exp_res *= 1 if raw_ori == pick_ori else 3.
    # There seems to be some precision penalty when combining orientations,
    # but it's probably acceptable
    rtol = 5e-4 if pick_ori is None else 1e-12
    assert_allclose(exp_res, stc_cov.data.ravel(), rtol=rtol)

    with pytest.raises(ValueError, match='Invalid value'):
        apply_inverse_cov(
            data_cov, raw.info, this_inv_op, method=method, pick_ori='vector')


@testing.requires_testing_data
def test_apply_mne_inverse_raw():
    """Test MNE with precomputed inverse operator on Raw."""
    start = 3
    stop = 10
    raw = read_raw_fif(fname_raw)
    label_lh = read_label(fname_label % 'Aud-lh')
    _, times = raw[0, start:stop]
    inverse_operator = read_inverse_operator(fname_full)
    with pytest.raises(ValueError, match='has not been prepared'):
        apply_inverse_raw(raw, inverse_operator, lambda2, prepared=True)
    inverse_operator = prepare_inverse_operator(inverse_operator, nave=1,
                                                lambda2=lambda2, method="dSPM")
    for pick_ori in [None, "normal", "vector"]:
        stc = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                label=label_lh, start=start, stop=stop, nave=1,
                                pick_ori=pick_ori, buffer_size=None,
                                prepared=True)

        stc2 = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                 label=label_lh, start=start, stop=stop,
                                 nave=1, pick_ori=pick_ori,
                                 buffer_size=3, prepared=True)

        if pick_ori is None:
            assert (np.all(stc.data > 0))
            assert (np.all(stc2.data > 0))

        assert (stc.subject == 'sample')
        assert (stc2.subject == 'sample')
        assert_array_almost_equal(stc.times, times)
        assert_array_almost_equal(stc2.times, times)
        assert_array_almost_equal(stc.data, stc2.data)


@testing.requires_testing_data
def test_apply_mne_inverse_fixed_raw():
    """Test MNE with fixed-orientation inverse operator on Raw."""
    raw = read_raw_fif(fname_raw)
    start = 3
    stop = 10
    _, times = raw[0, start:stop]
    label_lh = read_label(fname_label % 'Aud-lh')

    # create a fixed-orientation inverse operator
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=False,
                                    surf_ori=True)
    noise_cov = read_cov(fname_cov)
    pytest.raises(ValueError, make_inverse_operator,
                  raw.info, fwd, noise_cov, loose=1., fixed=True)
    inv_op = make_inverse_operator(raw.info, fwd, noise_cov,
                                   fixed=True, use_cps=True)

    inv_op2 = prepare_inverse_operator(inv_op, nave=1,
                                       lambda2=lambda2, method="dSPM")
    stc = apply_inverse_raw(raw, inv_op2, lambda2, "dSPM",
                            label=label_lh, start=start, stop=stop, nave=1,
                            pick_ori=None, buffer_size=None, prepared=True)

    stc2 = apply_inverse_raw(raw, inv_op2, lambda2, "dSPM",
                             label=label_lh, start=start, stop=stop, nave=1,
                             pick_ori=None, buffer_size=3, prepared=True)

    stc3 = apply_inverse_raw(raw, inv_op, lambda2, "dSPM",
                             label=label_lh, start=start, stop=stop, nave=1,
                             pick_ori=None, buffer_size=None)

    assert (stc.subject == 'sample')
    assert (stc2.subject == 'sample')
    assert_array_almost_equal(stc.times, times)
    assert_array_almost_equal(stc2.times, times)
    assert_array_almost_equal(stc3.times, times)
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.data, stc3.data)


@testing.requires_testing_data
def test_apply_mne_inverse_epochs():
    """Test MNE with precomputed inverse operator on Epochs."""
    inverse_operator = read_inverse_operator(fname_full)
    label_lh = read_label(fname_label % 'Aud-lh')
    label_rh = read_label(fname_label % 'Aud-rh')
    event_id, tmin, tmax = 1, -0.2, 0.5
    raw = read_raw_fif(fname_raw)

    picks = pick_types(raw.info, meg=True, eeg=False, stim=True, ecg=True,
                       eog=True, include=['STI 014'], exclude='bads')
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    flat = dict(grad=1e-15, mag=1e-15)

    events = read_events(fname_event)[:15]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat)

    inverse_operator = prepare_inverse_operator(inverse_operator, nave=1,
                                                lambda2=lambda2,
                                                method="dSPM")
    for pick_ori in [None, "normal", "vector"]:
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                    label=label_lh, pick_ori=pick_ori)
        stcs2 = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                     label=label_lh, pick_ori=pick_ori,
                                     prepared=True)
        # test if using prepared and not prepared inverse operator give the
        # same result
        assert_array_almost_equal(stcs[0].data, stcs2[0].data)
        assert_array_almost_equal(stcs[0].times, stcs2[0].times)

        assert (len(stcs) == 2)
        assert (3 < stcs[0].data.max() < 10)
        assert (stcs[0].subject == 'sample')
    inverse_operator = read_inverse_operator(fname_full)

    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                label=label_lh, pick_ori='normal')
    data = sum(stc.data for stc in stcs) / len(stcs)
    flip = label_sign_flip(label_lh, inverse_operator['src'])

    label_mean = np.mean(data, axis=0)
    label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)

    assert (label_mean.max() < label_mean_flip.max())

    # test extracting a BiHemiLabel
    inverse_operator = prepare_inverse_operator(inverse_operator, nave=1,
                                                lambda2=lambda2,
                                                method="dSPM")
    stcs_rh = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                   label=label_rh, pick_ori="normal",
                                   prepared=True)
    stcs_bh = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                   label=label_lh + label_rh,
                                   pick_ori="normal",
                                   prepared=True)

    n_lh = len(stcs[0].data)
    assert_array_almost_equal(stcs[0].data, stcs_bh[0].data[:n_lh])
    assert_array_almost_equal(stcs_rh[0].data, stcs_bh[0].data[n_lh:])

    # test without using a label (so delayed computation is used)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                pick_ori="normal", prepared=True)
    assert (stcs[0].subject == 'sample')
    label_stc = stcs[0].in_label(label_rh)
    assert (label_stc.subject == 'sample')
    assert_array_almost_equal(stcs_rh[0].data, label_stc.data)


def test_make_inverse_operator_bads(evoked, noise_cov):
    """Test MNE inverse computation given a mismatch of bad channels."""
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    assert evoked.info['bads'] == noise_cov['bads']
    assert evoked.info['bads'] == fwd_op['info']['bads'] + ['EEG 053']

    # one fewer bad in evoked than cov
    bad = evoked.info['bads'].pop()
    inv_ = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=1.)
    union_good = set(noise_cov['names']) & set(evoked.ch_names)
    union_bads = set(noise_cov['bads']) & set(evoked.info['bads'])
    evoked.info['bads'].append(bad)

    assert len(set(inv_['info']['ch_names']) - union_good) == 0
    assert len(set(inv_['info']['bads']) - union_bads) == 0


@testing.requires_testing_data
def test_inverse_ctf_comp():
    """Test interpolation with compensated CTF data."""
    raw = mne.io.read_raw_ctf(fname_raw_ctf).crop(0, 0)
    raw.apply_gradient_compensation(1)
    sphere = make_sphere_model()
    cov = make_ad_hoc_cov(raw.info)
    src = mne.setup_volume_source_space(
        pos=dict(rr=[[0., 0., 0.01]], nn=[[0., 1., 0.]]))
    fwd = make_forward_solution(raw.info, None, src, sphere, eeg=False)
    raw.apply_gradient_compensation(0)
    with pytest.raises(RuntimeError, match='Compensation grade .* not match'):
        make_inverse_operator(raw.info, fwd, cov, loose=1.)
    raw.apply_gradient_compensation(1)
    inv = make_inverse_operator(raw.info, fwd, cov, loose=1.)
    apply_inverse_raw(raw, inv, 1. / 9.)  # smoke test
    raw.apply_gradient_compensation(0)
    with pytest.raises(RuntimeError, match='Compensation grade .* not match'):
        apply_inverse_raw(raw, inv, 1. / 9.)


run_tests_if_main()
