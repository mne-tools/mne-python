import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
from scipy import sparse
from nose.tools import assert_true
import nose
import copy

from mne.datasets import sample
from mne.label import read_label, label_sign_flip
from mne.event import read_events
from mne.epochs import Epochs
from mne.source_estimate import read_source_estimate
from mne import fiff, read_cov, read_forward_solution
from mne.minimum_norm.inverse import apply_inverse, read_inverse_operator, \
    apply_inverse_raw, apply_inverse_epochs, make_inverse_operator, \
    write_inverse_operator, compute_rank
from mne.utils import _TempDir

s_path = op.join(sample.data_path(), 'MEG', 'sample')
fname_inv = op.join(s_path, 'sample_audvis-meg-oct-6-meg-inv.fif')
fname_inv_fixed = op.join(s_path, 'sample_audvis-meg-oct-6-meg-fixed-inv.fif')
fname_vol_inv = op.join(s_path, 'sample_audvis-meg-vol-7-meg-inv.fif')
fname_data = op.join(s_path, 'sample_audvis-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis-cov.fif')
fname_fwd = op.join(s_path, 'sample_audvis-meg-oct-6-fwd.fif')
fname_raw = op.join(s_path, 'sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(s_path, 'sample_audvis_filt-0-40_raw-eve.fif')
fname_label = op.join(s_path, 'labels', '%s.label')

inverse_operator = read_inverse_operator(fname_inv)
label_lh = read_label(fname_label % 'Aud-lh')
label_rh = read_label(fname_label % 'Aud-rh')
noise_cov = read_cov(fname_cov)
raw = fiff.Raw(fname_raw)
snr = 3.0
lambda2 = 1.0 / snr ** 2

tempdir = _TempDir()


def _compare(a, b):
    if isinstance(a, dict):
        assert_true(isinstance(b, dict))
        for k, v in a.iteritems():
            if not k in b:
                raise ValueError('%s not in %s' % (k, b))
            _compare(v, b[k])
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


def test_io_inverse_operator():
    """Test IO of inverse_operator
    """
    inverse_operator_vol = read_inverse_operator(fname_vol_inv)
    for inv in [inverse_operator, inverse_operator_vol]:
        inv_init = copy.deepcopy(inv)
        out_file = op.join(tempdir, 'test-inv.fif')
        write_inverse_operator(out_file, inv)
        this_inv = read_inverse_operator(out_file)
        _compare(inv, inv_init)
        _compare(inv, this_inv)

    # just do one example for .gz, as it should generalize
    inv = inverse_operator
    out_file = op.join(tempdir, 'test-inv.fif.gz')
    write_inverse_operator(out_file, inv)
    this_inv = read_inverse_operator(out_file)
    _compare(inv, this_inv)


def test_apply_inverse_operator():
    """Test MNE inverse computation

    With and without precomputed inverse operator.
    """
    evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
    evoked.crop(0, 0.1)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "MNE")
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 10e-10)
    assert_true(stc.data.mean() > 1e-11)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "sLORETA")
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 9.0)
    assert_true(stc.data.mean() > 0.1)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM")
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 35)
    assert_true(stc.data.mean() > 0.1)

    # Test MNE inverse computation starting from forward operator
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)

    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8)
    out_file = op.join(tempdir, 'test-inv.fif')
    write_inverse_operator(out_file, my_inv_op)
    read_my_inv_op = read_inverse_operator(out_file)
    _compare(my_inv_op, read_my_inv_op)

    my_stc = apply_inverse(evoked, my_inv_op, lambda2, "dSPM")

    assert_true('dev_head_t' in my_inv_op['info'])
    assert_true('mri_head_t' in my_inv_op)

    assert_equal(stc.times, my_stc.times)
    assert_array_almost_equal(stc.data, my_stc.data, 2)

    # The inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank(inverse_operator) == 302)


def test_make_inverse_operator_fixed():
    """Test MNE inverse computation with fixed orientation"""
    # XXX : should be fixed and not skipped
    raise nose.SkipTest("XFailed Test")

    evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
    fwd_op = read_forward_solution(fname_fwd, force_fixed=True)
    inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov, depth=0.8,
                                   loose=None)
    inverse_operator_fixed = read_inverse_operator(fname_inv_fixed)

    assert_array_almost_equal(inverse_operator_fixed['depth_prior']['data'],
                              inv_op['depth_prior']['data'])
    assert_equal(inverse_operator_fixed['orient_prior'],
                 inv_op['orient_prior'])
    assert_array_almost_equal(inverse_operator_fixed['source_cov']['data'],
                              inv_op['source_cov']['data'])

    stc_fixed = apply_inverse(evoked, inverse_operator_fixed, lambda2, "dSPM")
    my_stc = apply_inverse(evoked, inv_op, lambda2, "dSPM")

    assert_equal(stc_fixed.times, my_stc.times)
    assert_array_almost_equal(stc_fixed.data, my_stc.data, 2)

    # assert_array_almost_equal(inverse_operator_fixed['eigen_fields']['data'],
    #                           inv_op['eigen_fields']['data'])
    # assert_array_almost_equal(inverse_operator_fixed['eigen_leads']['data'],
    #                           inv_op['eigen_leads']['data'])


def test_inverse_operator_volume():
    """Test MNE inverse computation on volume source space"""
    evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
    inverse_operator_vol = read_inverse_operator(fname_vol_inv)
    stc = apply_inverse(evoked, inverse_operator_vol, lambda2, "dSPM")
    stc.save(op.join(tempdir, 'tmp-vl.stc'))
    stc2 = read_source_estimate(op.join(tempdir, 'tmp-vl.stc'))
    assert_true(np.all(stc.data > 0))
    assert_true(np.all(stc.data < 35))
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)


def test_apply_mne_inverse_raw():
    """Test MNE with precomputed inverse operator on Raw"""
    start = 3
    stop = 10
    _, times = raw[0, start:stop]
    for pick_normal in [False, True]:
        stc = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                label=label_lh, start=start, stop=stop, nave=1,
                                pick_normal=pick_normal, buffer_size=None)

        stc2 = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                 label=label_lh, start=start, stop=stop, nave=1,
                                 pick_normal=pick_normal, buffer_size=3)

        if not pick_normal:
            assert_true(np.all(stc.data > 0))
            assert_true(np.all(stc2.data > 0))

        assert_array_almost_equal(stc.times, times)
        assert_array_almost_equal(stc2.times, times)

        assert_array_almost_equal(stc.data, stc2.data)


def test_apply_mne_inverse_fixed_raw():
    """Test MNE with fixed-orientation inverse operator on Raw"""
    start = 3
    stop = 10
    _, times = raw[0, start:stop]

    # create a fixed-orientation inverse operator
    fwd = read_forward_solution(fname_fwd, force_fixed=True)
    inv_op = make_inverse_operator(raw.info, fwd, noise_cov,
                                   loose=None, depth=0.8)

    stc = apply_inverse_raw(raw, inv_op, lambda2, "dSPM",
                            label=label_lh, start=start, stop=stop, nave=1,
                            pick_normal=False, buffer_size=None)

    stc2 = apply_inverse_raw(raw, inv_op, lambda2, "dSPM",
                             label=label_lh, start=start, stop=stop, nave=1,
                             pick_normal=False, buffer_size=3)

    assert_array_almost_equal(stc.times, times)
    assert_array_almost_equal(stc2.times, times)
    assert_array_almost_equal(stc.data, stc2.data)


def test_apply_mne_inverse_epochs():
    """Test MNE with precomputed inverse operator on Epochs
    """
    event_id, tmin, tmax = 1, -0.2, 0.5

    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                            ecg=True, eog=True, include=['STI 014'])
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    flat = dict(grad=1e-15, mag=1e-15)

    events = read_events(fname_event)[:15]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                label=label_lh, pick_normal=True)

    assert_true(len(stcs) == 4)
    assert_true(3 < stcs[0].data.max() < 10)

    data = sum(stc.data for stc in stcs) / len(stcs)
    flip = label_sign_flip(label_lh, inverse_operator['src'])

    label_mean = np.mean(data, axis=0)
    label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)

    assert_true(label_mean.max() < label_mean_flip.max())

    # test extracting a BiHemiLabel
    stcs_rh = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                   label=label_rh, pick_normal=True)
    stcs_bh = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                   label=label_lh + label_rh, pick_normal=True)

    n_lh = len(stcs[0].data)
    assert_array_almost_equal(stcs[0].data, stcs_bh[0].data[:n_lh])
    assert_array_almost_equal(stcs_rh[0].data, stcs_bh[0].data[n_lh:])
