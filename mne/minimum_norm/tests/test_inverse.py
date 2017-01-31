from __future__ import print_function
import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_allclose, assert_array_equal)
from scipy import sparse
from nose.tools import assert_true, assert_raises
import copy
import warnings

from mne.datasets import testing
from mne.label import read_label, label_sign_flip
from mne.event import read_events
from mne.epochs import Epochs
from mne.source_estimate import read_source_estimate, VolSourceEstimate
from mne import (read_cov, read_forward_solution, read_evokeds, pick_types,
                 pick_types_forward, make_forward_solution,
                 convert_forward_solution, Covariance)
from mne.io import read_raw_fif, Info
from mne.minimum_norm.inverse import (apply_inverse, read_inverse_operator,
                                      apply_inverse_raw, apply_inverse_epochs,
                                      make_inverse_operator,
                                      write_inverse_operator,
                                      compute_rank_inverse,
                                      prepare_inverse_operator)
from mne.tests.common import assert_naming
from mne.utils import _TempDir, run_tests_if_main, slow_test
from mne.externals import six

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
# Four inverses:
fname_full = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_inv = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
fname_inv_fixed_nodepth = op.join(s_path,
                                  'sample_audvis_trunc-meg-eeg-oct-4-meg'
                                  '-nodepth-fixed-inv.fif')
fname_inv_meeg_diag = op.join(s_path,
                              'sample_audvis_trunc-'
                              'meg-eeg-oct-4-meg-eeg-diagnoise-inv.fif')

fname_data = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_raw = op.join(s_path, 'sample_audvis_trunc_raw.fif')
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


def read_forward_solution_meg(*args, **kwargs):
    """Read MEG forward."""
    fwd = read_forward_solution(*args, **kwargs)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    return fwd


def read_forward_solution_eeg(*args, **kwargs):
    """Read EEG forward."""
    fwd = read_forward_solution(*args, **kwargs)
    fwd = pick_types_forward(fwd, meg=False, eeg=True)
    return fwd


def _get_evoked():
    """Get evoked data."""
    evoked = read_evokeds(fname_data, condition=0, baseline=(None, 0))
    evoked.crop(0, 0.2)
    return evoked


def _compare(a, b):
    """Compare two python objects."""
    global last_keys
    skip_types = ['whitener', 'proj', 'reginv', 'noisenorm', 'nchan',
                  'command_line', 'working_dir', 'mri_file', 'mri_id']
    try:
        if isinstance(a, (dict, Info)):
            assert_true(isinstance(b, (dict, Info)))
            for k, v in six.iteritems(a):
                if k not in b and k not in skip_types:
                    raise ValueError('First one had one second one didn\'t:\n'
                                     '%s not in %s' % (k, b.keys()))
                if k not in skip_types:
                    last_keys.pop()
                    last_keys = [k] + last_keys
                    _compare(v, b[k])
            for k, v in six.iteritems(b):
                if k not in a and k not in skip_types:
                    raise ValueError('Second one had one first one didn\'t:\n'
                                     '%s not in %s' % (k, a.keys()))
        elif isinstance(a, list):
            assert_true(len(a) == len(b))
            for i, j in zip(a, b):
                _compare(i, j)
        elif isinstance(a, sparse.csr.csr_matrix):
            assert_array_almost_equal(a.data, b.data)
            assert_equal(a.indices, b.indices)
            assert_equal(a.indptr, b.indptr)
        elif isinstance(a, np.ndarray):
            assert_array_almost_equal(a, b)
        else:
            assert_true(a == b)
    except Exception:
        print(last_keys)
        raise


def _compare_inverses_approx(inv_1, inv_2, evoked, rtol, atol,
                             check_depth=True):
    """Compare inverses."""
    # depth prior
    if check_depth:
        if inv_1['depth_prior'] is not None:
            assert_array_almost_equal(inv_1['depth_prior']['data'],
                                      inv_2['depth_prior']['data'], 5)
        else:
            assert_true(inv_2['depth_prior'] is None)
    # orient prior
    if inv_1['orient_prior'] is not None:
        assert_array_almost_equal(inv_1['orient_prior']['data'],
                                  inv_2['orient_prior']['data'])
    else:
        assert_true(inv_2['orient_prior'] is None)
    # source cov
    assert_array_almost_equal(inv_1['source_cov']['data'],
                              inv_2['source_cov']['data'])

    # These are not as close as we'd like XXX
    assert_array_almost_equal(np.abs(inv_1['eigen_fields']['data']),
                              np.abs(inv_2['eigen_fields']['data']), 0)
    assert_array_almost_equal(np.abs(inv_1['eigen_leads']['data']),
                              np.abs(inv_2['eigen_leads']['data']), 0)

    stc_1 = apply_inverse(evoked, inv_1, lambda2, "dSPM")
    stc_2 = apply_inverse(evoked, inv_2, lambda2, "dSPM")

    assert_true(stc_1.subject == stc_2.subject)
    assert_equal(stc_1.times, stc_2.times)
    assert_allclose(stc_1.data, stc_2.data, rtol=rtol, atol=atol)
    assert_true(inv_1['units'] == inv_2['units'])


def _compare_io(inv_op, out_file_ext='.fif'):
    """Compare inverse IO."""
    tempdir = _TempDir()
    if out_file_ext == '.fif':
        out_file = op.join(tempdir, 'test-inv.fif')
    elif out_file_ext == '.gz':
        out_file = op.join(tempdir, 'test-inv.fif.gz')
    else:
        raise ValueError('IO test could not complete')
    # Test io operations
    inv_init = copy.deepcopy(inv_op)
    write_inverse_operator(out_file, inv_op)
    read_inv_op = read_inverse_operator(out_file)
    _compare(inv_init, read_inv_op)
    _compare(inv_init, inv_op)


@testing.requires_testing_data
def test_warn_inverse_operator():
    """Test MNE inverse warning without average EEG projection."""
    bad_info = copy.deepcopy(_get_evoked().info)
    bad_info['projs'] = list()
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    noise_cov = read_cov(fname_cov)
    with warnings.catch_warnings(record=True) as w:
        make_inverse_operator(bad_info, fwd_op, noise_cov)
    assert_equal(len(w), 1)


@slow_test
@testing.requires_testing_data
def test_make_inverse_operator():
    """Test MNE inverse computation (precomputed and non-precomputed)
    """
    # Test old version of inverse computation starting from forward operator
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)
    inverse_operator = read_inverse_operator(fname_inv)
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8,
                                      limit_depth_chs=False)
    _compare_io(my_inv_op)
    assert_true(inverse_operator['units'] == 'Am')
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked, 1e-2, 1e-2,
                             check_depth=False)
    # Test MNE inverse computation starting from forward operator
    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8)
    _compare_io(my_inv_op)
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked, 1e-2, 1e-2)
    assert_true('dev_head_t' in my_inv_op['info'])
    assert_true('mri_head_t' in my_inv_op)


@slow_test
@testing.requires_testing_data
def test_inverse_operator_channel_ordering():
    """Test MNE inverse computation is immune to channel reorderings
    """
    # These are with original ordering
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)

    fwd_orig = make_forward_solution(evoked.info, fname_trans, src_fname,
                                     fname_bem, eeg=True, mindist=5.0)
    fwd_orig = convert_forward_solution(fwd_orig, surf_ori=True)
    inv_orig = make_inverse_operator(evoked.info, fwd_orig, noise_cov,
                                     loose=0.2, depth=0.8,
                                     limit_depth_chs=False)
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
                                        loose=0.2, depth=0.8,
                                        limit_depth_chs=False)

    stc_2 = apply_inverse(evoked, inv_reorder, lambda2, "dSPM")

    assert_equal(stc_1.subject, stc_2.subject)
    assert_array_equal(stc_1.times, stc_2.times)
    assert_allclose(stc_1.data, stc_2.data, rtol=1e-5, atol=1e-5)
    assert_true(inv_orig['units'] == inv_reorder['units'])

    # Reload with original ordering & apply reordered inverse
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)

    stc_3 = apply_inverse(evoked, inv_reorder, lambda2, "dSPM")
    assert_allclose(stc_1.data, stc_3.data, rtol=1e-5, atol=1e-5)


@slow_test
@testing.requires_testing_data
def test_apply_inverse_operator():
    """Test MNE inverse application
    """
    inverse_operator = read_inverse_operator(fname_full)
    evoked = _get_evoked()

    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator) == 302)

    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator) == 302)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "MNE")
    assert_true(stc.subject == 'sample')
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 10e-9)
    assert_true(stc.data.mean() > 1e-11)

    # test if using prepared and not prepared inverse operator give the same
    # result
    inv_op = prepare_inverse_operator(inverse_operator, nave=evoked.nave,
                                      lambda2=lambda2, method="MNE")
    stc2 = apply_inverse(evoked, inv_op, lambda2, "MNE")
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "sLORETA")
    assert_true(stc.subject == 'sample')
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 10.0)
    assert_true(stc.data.mean() > 0.1)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM")
    assert_true(stc.subject == 'sample')
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 35)
    assert_true(stc.data.mean() > 0.1)

    # test without using a label (so delayed computation is used)
    label = read_label(fname_label % 'Aud-lh')
    stc = apply_inverse(evoked, inv_op, lambda2, "MNE")
    stc_label = apply_inverse(evoked, inv_op, lambda2, "MNE",
                              label=label)
    assert_equal(stc_label.subject, 'sample')
    label_stc = stc.in_label(label)
    assert_true(label_stc.subject == 'sample')
    assert_array_almost_equal(stc_label.data, label_stc.data)

    # Test we get errors when using custom ref or no average proj is present
    evoked.info['custom_ref_applied'] = True
    assert_raises(ValueError, apply_inverse, evoked, inv_op, lambda2, "MNE")
    evoked.info['custom_ref_applied'] = False
    evoked.info['projs'] = []  # remove EEG proj
    assert_raises(ValueError, apply_inverse, evoked, inv_op, lambda2, "MNE")


@testing.requires_testing_data
def test_make_inverse_operator_fixed():
    """Test MNE inverse computation (fixed orientation)
    """
    fwd_1 = read_forward_solution_meg(fname_fwd, surf_ori=False,
                                      force_fixed=False)
    fwd_2 = read_forward_solution_meg(fname_fwd, surf_ori=False,
                                      force_fixed=True)
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)

    # can't make depth-weighted fixed inv without surf ori fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_1,
                  noise_cov, depth=0.8, loose=None, fixed=True)
    # can't make fixed inv with depth weighting without free ori fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_2,
                  noise_cov, depth=0.8, loose=None, fixed=True)

    # now compare to C solution
    # note that the forward solution must not be surface-oriented
    # to get equivalency (surf_ori=True changes the normals)
    inv_op = make_inverse_operator(evoked.info, fwd_2, noise_cov, depth=None,
                                   loose=None, fixed=True)
    inverse_operator_nodepth = read_inverse_operator(fname_inv_fixed_nodepth)
    _compare_inverses_approx(inverse_operator_nodepth, inv_op, evoked, 0, 1e-2)
    # Inverse has 306 channels - 6 proj = 302
    assert_true(compute_rank_inverse(inverse_operator_nodepth) == 302)


@testing.requires_testing_data
def test_make_inverse_operator_free():
    """Test MNE inverse computation (free orientation)
    """
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    fwd_1 = read_forward_solution_meg(fname_fwd, surf_ori=False,
                                      force_fixed=False)
    fwd_2 = read_forward_solution_meg(fname_fwd, surf_ori=False,
                                      force_fixed=True)
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)

    # can't make free inv with fixed fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_2,
                  noise_cov, depth=None)

    # for free ori inv, loose=None and loose=1 should be equivalent
    inv_1 = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=None)
    inv_2 = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=1)
    _compare_inverses_approx(inv_1, inv_2, evoked, 0, 1e-2)

    # for depth=None, surf_ori of the fwd should not matter
    inv_3 = make_inverse_operator(evoked.info, fwd_op, noise_cov, depth=None,
                                  loose=None)
    inv_4 = make_inverse_operator(evoked.info, fwd_1, noise_cov, depth=None,
                                  loose=None)
    _compare_inverses_approx(inv_3, inv_4, evoked, 0, 1e-2)


@testing.requires_testing_data
def test_make_inverse_operator_diag():
    """Test MNE inverse computation with diagonal noise cov
    """
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov).as_diag()
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                   loose=0.2, depth=0.8)
    _compare_io(inv_op)
    inverse_operator_diag = read_inverse_operator(fname_inv_meeg_diag)
    # This one's only good to zero decimal places, roundoff error (?)
    _compare_inverses_approx(inverse_operator_diag, inv_op, evoked, 0, 1e0)
    # Inverse has 366 channels - 6 proj = 360
    assert_true(compute_rank_inverse(inverse_operator_diag) == 360)


@testing.requires_testing_data
def test_inverse_operator_noise_cov_rank():
    """Test MNE inverse operator with a specified noise cov rank
    """
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)
    inv = make_inverse_operator(evoked.info, fwd_op, noise_cov, rank=64)
    assert_true(compute_rank_inverse(inv) == 64)

    fwd_op = read_forward_solution_eeg(fname_fwd, surf_ori=True)
    inv = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                rank=dict(eeg=20))
    assert_true(compute_rank_inverse(inv) == 20)


@testing.requires_testing_data
def test_inverse_operator_volume():
    """Test MNE inverse computation on volume source space
    """
    tempdir = _TempDir()
    evoked = _get_evoked()
    inverse_operator_vol = read_inverse_operator(fname_vol_inv)
    assert_true(repr(inverse_operator_vol))
    stc = apply_inverse(evoked, inverse_operator_vol, lambda2, "dSPM")
    assert_true(isinstance(stc, VolSourceEstimate))
    # volume inverses don't have associated subject IDs
    assert_true(stc.subject is None)
    stc.save(op.join(tempdir, 'tmp-vl.stc'))
    stc2 = read_source_estimate(op.join(tempdir, 'tmp-vl.stc'))
    assert_true(np.all(stc.data > 0))
    assert_true(np.all(stc.data < 35))
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)


@slow_test
@testing.requires_testing_data
def test_io_inverse_operator():
    """Test IO of inverse_operator
    """
    tempdir = _TempDir()
    inverse_operator = read_inverse_operator(fname_inv)
    x = repr(inverse_operator)
    assert_true(x)
    assert_true(isinstance(inverse_operator['noise_cov'], Covariance))
    # just do one example for .gz, as it should generalize
    _compare_io(inverse_operator, '.gz')

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        inv_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        write_inverse_operator(inv_badname, inverse_operator)
        read_inverse_operator(inv_badname)
    assert_naming(w, 'test_inverse.py', 2)

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


@testing.requires_testing_data
def test_apply_mne_inverse_raw():
    """Test MNE with precomputed inverse operator on Raw."""
    start = 3
    stop = 10
    raw = read_raw_fif(fname_raw)
    label_lh = read_label(fname_label % 'Aud-lh')
    _, times = raw[0, start:stop]
    inverse_operator = read_inverse_operator(fname_full)
    inverse_operator = prepare_inverse_operator(inverse_operator, nave=1,
                                                lambda2=lambda2, method="dSPM")
    for pick_ori in [None, "normal"]:
        stc = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                label=label_lh, start=start, stop=stop, nave=1,
                                pick_ori=pick_ori, buffer_size=None,
                                prepared=True)

        stc2 = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                 label=label_lh, start=start, stop=stop,
                                 nave=1, pick_ori=pick_ori,
                                 buffer_size=3, prepared=True)

        if pick_ori is None:
            assert_true(np.all(stc.data > 0))
            assert_true(np.all(stc2.data > 0))

        assert_true(stc.subject == 'sample')
        assert_true(stc2.subject == 'sample')
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
    inv_op = make_inverse_operator(raw.info, fwd, noise_cov,
                                   loose=None, depth=0.8, fixed=True)

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

    assert_true(stc.subject == 'sample')
    assert_true(stc2.subject == 'sample')
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
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                label=label_lh, pick_ori="normal")
    inverse_operator = prepare_inverse_operator(inverse_operator, nave=1,
                                                lambda2=lambda2, method="dSPM")
    stcs2 = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                 label=label_lh, pick_ori="normal",
                                 prepared=True)
    # test if using prepared and not prepared inverse operator give the same
    # result
    assert_array_almost_equal(stcs[0].data, stcs2[0].data)
    assert_array_almost_equal(stcs[0].times, stcs2[0].times)

    assert_true(len(stcs) == 2)
    assert_true(3 < stcs[0].data.max() < 10)
    assert_true(stcs[0].subject == 'sample')

    data = sum(stc.data for stc in stcs) / len(stcs)
    flip = label_sign_flip(label_lh, inverse_operator['src'])

    label_mean = np.mean(data, axis=0)
    label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)

    assert_true(label_mean.max() < label_mean_flip.max())

    # test extracting a BiHemiLabel

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
    assert_true(stcs[0].subject == 'sample')
    label_stc = stcs[0].in_label(label_rh)
    assert_true(label_stc.subject == 'sample')
    assert_array_almost_equal(stcs_rh[0].data, label_stc.data)


@testing.requires_testing_data
def test_make_inverse_operator_bads():
    """Test MNE inverse computation given a mismatch of bad channels."""
    fwd_op = read_forward_solution_meg(fname_fwd, surf_ori=True)
    evoked = _get_evoked()
    noise_cov = read_cov(fname_cov)

    # test bads
    bad = evoked.info['bads'].pop()
    inv_ = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=None)
    union_good = set(noise_cov['names']) & set(evoked.ch_names)
    union_bads = set(noise_cov['bads']) & set(evoked.info['bads'])
    evoked.info['bads'].append(bad)

    assert_true(len(set(inv_['info']['ch_names']) - union_good) == 0)
    assert_true(len(set(inv_['info']['bads']) - union_bads) == 0)


run_tests_if_main()
