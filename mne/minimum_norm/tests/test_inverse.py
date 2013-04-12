import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
from scipy import sparse
from nose.tools import assert_true, assert_raises
import copy

from mne.datasets import sample
from mne.label import read_label, label_sign_flip
from mne.event import read_events
from mne.epochs import Epochs
from mne.source_estimate import read_source_estimate
from mne import fiff, read_cov, read_forward_solution
from mne.minimum_norm.inverse import apply_inverse, read_inverse_operator, \
    apply_inverse_raw, apply_inverse_epochs, make_inverse_operator, \
    write_inverse_operator, compute_rank_inverse
from mne.utils import _TempDir

s_path = op.join(sample.data_path(), 'MEG', 'sample')
fname_inv = op.join(s_path, 'sample_audvis-meg-oct-6-meg-inv.fif')
fname_inv_fixed = op.join(s_path, 'sample_audvis-meg-oct-6-meg-fixed-inv.fif')
fname_inv_nodepth = op.join(s_path,
                           'sample_audvis-meg-oct-6-meg-nodepth-fixed-inv.fif')
fname_inv_diag = op.join(s_path,
                         'sample_audvis-meg-oct-6-meg-diagnoise-inv.fif')
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
evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
evoked.crop(0, 0.2)
snr = 3.0
lambda2 = 1.0 / snr ** 2

tempdir = _TempDir()
last_keys = [None] * 10


def _compare(a, b):
    global last_keys
    skip_types = ['whitener', 'proj', 'reginv', 'noisenorm', 'nchan',
                  'command_line', 'working_dir', 'mri_file', 'mri_id']
    try:
        if isinstance(a, dict):
            assert_true(isinstance(b, dict))
            for k, v in a.iteritems():
                if not k in b and k not in skip_types:
                    raise ValueError('First one had one second one didn\'t:\n'
                                     '%s not in %s' % (k, b.keys()))
                if k not in skip_types:
                    last_keys.pop()
                    last_keys = [k] + last_keys
                    _compare(v, b[k])
            for k, v in b.iteritems():
                if not k in a and k not in skip_types:
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
    except Exception as exptn:
        print last_keys
        raise exptn


def _compare_inverses_approx(inv_1, inv_2, evoked, stc_decimals,
                             check_depth=True):
    # depth prior
    if check_depth:
        if inv_1['depth_prior'] is not None:
            assert_array_almost_equal(inv_1['depth_prior']['data'],
                                      inv_2['depth_prior']['data'])
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
    assert_array_almost_equal(stc_1.data, stc_2.data, stc_decimals)


def _compare_io(inv_op, out_file_ext='.fif'):
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


def test_apply_inverse_operator():
    """Test MNE inverse computation (precomputed and non-precomputed)
    """

    # Test old version of inverse computation starting from forward operator
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8,
                                      limit_depth_chs=False)
    _compare_io(my_inv_op)
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked, 2,
                             check_depth=False)
    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator) == 302)

    # Test MNE inverse computation starting from forward operator
    my_inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov,
                                      loose=0.2, depth=0.8)
    _compare_io(my_inv_op)
    _compare_inverses_approx(my_inv_op, inverse_operator, evoked, 2)
    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator) == 302)

    stc = apply_inverse(evoked, inverse_operator, lambda2, "MNE")
    assert_true(stc.subject == 'sample')
    assert_true(stc.data.min() > 0)
    assert_true(stc.data.max() < 10e-10)
    assert_true(stc.data.mean() > 1e-11)

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

    my_stc = apply_inverse(evoked, my_inv_op, lambda2, "dSPM")

    assert_true('dev_head_t' in my_inv_op['info'])
    assert_true('mri_head_t' in my_inv_op)

    assert_true(my_stc.subject == 'sample')
    assert_equal(stc.times, my_stc.times)
    assert_array_almost_equal(stc.data, my_stc.data, 2)


def test_make_inverse_operator_fixed():
    """Test MNE inverse computation (fixed orientation)
    """
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    fwd_1 = read_forward_solution(fname_fwd, surf_ori=False, force_fixed=False)
    fwd_2 = read_forward_solution(fname_fwd, surf_ori=False, force_fixed=True)

    # can't make depth-weighted fixed inv without surf ori fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_1,
                  noise_cov, depth=0.8, loose=None, fixed=True)
    # can't make fixed inv with depth weighting without free ori fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_2,
                  noise_cov, depth=0.8, loose=None, fixed=True)
    # can't make non-depth-weighted fixed inv with surf_ori fwd
    # (otherwise the average normal could be employed)
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_op,
                  noise_cov, depth=None, loose=None, fixed=True)

    # compare to C solution w/fixed
    inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov, depth=0.8,
                                   loose=None, fixed=True)
    _compare_io(inv_op)
    inverse_operator_fixed = read_inverse_operator(fname_inv_fixed)
    _compare_inverses_approx(inverse_operator_fixed, inv_op, evoked, 2)
    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator_fixed) == 302)

    # now compare to C solution
    # note that the forward solution must not be surface-oriented
    # to get equivalency (surf_ori=True changes the normals)
    inv_op = make_inverse_operator(evoked.info, fwd_2, noise_cov, depth=None,
                                   loose=None, fixed=True)
    inverse_operator_nodepth = read_inverse_operator(fname_inv_nodepth)
    _compare_inverses_approx(inverse_operator_nodepth, inv_op, evoked, 2)
    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator_fixed) == 302)


def test_make_inverse_operator_free():
    """Test MNE inverse computation (free orientation)
    """
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    fwd_1 = read_forward_solution(fname_fwd, surf_ori=False, force_fixed=False)
    fwd_2 = read_forward_solution(fname_fwd, surf_ori=False, force_fixed=True)

    # can't make free inv with fixed fwd
    assert_raises(ValueError, make_inverse_operator, evoked.info, fwd_2,
                  noise_cov, depth=None)

    # for free ori inv, loose=None and loose=1 should be equivalent
    inv_1 = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=None)
    inv_2 = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=1)
    _compare_inverses_approx(inv_1, inv_2, evoked, 2)

    # for depth=None, surf_ori of the fwd should not matter
    inv_3 = make_inverse_operator(evoked.info, fwd_op, noise_cov, depth=None,
                                  loose=None)
    inv_4 = make_inverse_operator(evoked.info, fwd_1, noise_cov, depth=None,
                                  loose=None)
    _compare_inverses_approx(inv_3, inv_4, evoked, 2)


def test_make_inverse_operator_diag():
    """Test MNE inverse computation with diagonal noise cov
    """
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)
    inv_op = make_inverse_operator(evoked.info, fwd_op, noise_cov.as_diag(),
                                   loose=0.2, depth=0.8)
    _compare_io(inv_op)
    inverse_operator_diag = read_inverse_operator(fname_inv_diag)
    # This one's only good to zero decimal places, roundoff error (?)
    _compare_inverses_approx(inverse_operator_diag, inv_op, evoked, 0)
    # Inverse has 306 channels - 4 proj = 302
    assert_true(compute_rank_inverse(inverse_operator_diag) == 302)


def test_inverse_operator_volume():
    """Test MNE inverse computation on volume source space
    """
    inverse_operator_vol = read_inverse_operator(fname_vol_inv)
    _compare_io(inverse_operator_vol)
    stc = apply_inverse(evoked, inverse_operator_vol, lambda2, "dSPM")
    # volume inverses don't have associated subject IDs
    assert_true(stc.subject is None)
    stc.save(op.join(tempdir, 'tmp-vl.stc'))
    stc2 = read_source_estimate(op.join(tempdir, 'tmp-vl.stc'))
    assert_true(np.all(stc.data > 0))
    assert_true(np.all(stc.data < 35))
    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.times, stc2.times)


def test_io_inverse_operator():
    """Test IO of inverse_operator with GZip
    """
    # just do one example for .gz, as it should generalize
    _compare_io(inverse_operator, '.gz')


def test_apply_mne_inverse_raw():
    """Test MNE with precomputed inverse operator on Raw
    """
    start = 3
    stop = 10
    _, times = raw[0, start:stop]
    for pick_normal in [False, True]:
        stc = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                label=label_lh, start=start, stop=stop, nave=1,
                                pick_normal=pick_normal, buffer_size=None)

        stc2 = apply_inverse_raw(raw, inverse_operator, lambda2, "dSPM",
                                 label=label_lh, start=start, stop=stop,
                                 nave=1, pick_normal=pick_normal,
                                 buffer_size=3)

        if not pick_normal:
            assert_true(np.all(stc.data > 0))
            assert_true(np.all(stc2.data > 0))

        assert_true(stc.subject == 'sample')
        assert_true(stc2.subject == 'sample')
        assert_array_almost_equal(stc.times, times)
        assert_array_almost_equal(stc2.times, times)
        assert_array_almost_equal(stc.data, stc2.data)


def test_apply_mne_inverse_fixed_raw():
    """Test MNE with fixed-orientation inverse operator on Raw
    """
    start = 3
    stop = 10
    _, times = raw[0, start:stop]

    # create a fixed-orientation inverse operator
    fwd = read_forward_solution(fname_fwd, force_fixed=False, surf_ori=True)
    inv_op = make_inverse_operator(raw.info, fwd, noise_cov,
                                   loose=None, depth=0.8, fixed=True)

    stc = apply_inverse_raw(raw, inv_op, lambda2, "dSPM",
                            label=label_lh, start=start, stop=stop, nave=1,
                            pick_normal=False, buffer_size=None)

    stc2 = apply_inverse_raw(raw, inv_op, lambda2, "dSPM",
                             label=label_lh, start=start, stop=stop, nave=1,
                             pick_normal=False, buffer_size=3)

    assert_true(stc.subject == 'sample')
    assert_true(stc2.subject == 'sample')
    assert_array_almost_equal(stc.times, times)
    assert_array_almost_equal(stc2.times, times)
    assert_array_almost_equal(stc.data, stc2.data)


def test_apply_mne_inverse_epochs():
    """Test MNE with precomputed inverse operator on Epochs
    """
    event_id, tmin, tmax = 1, -0.2, 0.5

    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                            ecg=True, eog=True, include=['STI 014'],
                            exclude='bads')
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    flat = dict(grad=1e-15, mag=1e-15)

    events = read_events(fname_event)[:15]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                label=label_lh, pick_normal=True)

    assert_true(len(stcs) == 4)
    assert_true(3 < stcs[0].data.max() < 10)
    assert_true(stcs[0].subject == 'sample')

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

    # test without using a label (so delayed computation is used)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, "dSPM",
                                pick_normal=True)
    assert_true(stcs[0].subject == 'sample')
    label_stc = stcs[0].in_label(label_rh)
    assert_true(label_stc.subject == 'sample')
    assert_array_almost_equal(stcs_rh[0].data, label_stc.data)


def test_make_inverse_operator_bads():
    """Test MNE inverse computation given a mismatch of bad channels
    """
    fwd_op = read_forward_solution(fname_fwd, surf_ori=True)

    # test bads
    bad = evoked.info['bads'].pop()
    inv_ = make_inverse_operator(evoked.info, fwd_op, noise_cov, loose=None)
    union_good = set(noise_cov['names']) & set(evoked.ch_names)
    union_bads = set(noise_cov['bads']) & set(evoked.info['bads'])
    evoked.info['bads'].append(bad)

    assert_true(len(set(inv_['info']['ch_names']) - union_good) == 0)

    assert_true(len(set(inv_['info']['bads']) - union_bads) == 0)
