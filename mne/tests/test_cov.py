# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import itertools as itt

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
import pytest
import numpy as np
from scipy import linalg

from mne.cov import (regularize, whiten_evoked,
                     _auto_low_rank_model,
                     prepare_noise_cov, compute_whitener,
                     _regularized_covariance)

from mne import (read_cov, write_cov, Epochs, merge_events,
                 find_events, compute_raw_covariance,
                 compute_covariance, read_evokeds, compute_proj_raw,
                 pick_channels_cov, pick_types, make_ad_hoc_cov,
                 make_fixed_length_events, create_info)
from mne.channels import equalize_channels
from mne.datasets import testing
from mne.fixes import _get_args
from mne.io import read_raw_fif, RawArray, read_raw_ctf
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.preprocessing import maxwell_filter
from mne.rank import _compute_rank_int
from mne.utils import (requires_sklearn, run_tests_if_main,
                       catch_logging, assert_snr)

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
cov_fname = op.join(base_dir, 'test-cov.fif')
cov_gz_fname = op.join(base_dir, 'test-cov.fif.gz')
cov_km_fname = op.join(base_dir, 'test-km-cov.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
erm_cov_fname = op.join(base_dir, 'test_erm-cov.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')

ctf_fname = op.join(testing.data_path(download=False), 'CTF',
                    'testdata_ctf.ds')


@pytest.mark.parametrize('proj', (True, False))
@pytest.mark.parametrize('pca', (True, 'white', False))
def test_compute_whitener(proj, pca):
    """Test properties of compute_whitener."""
    raw = read_raw_fif(raw_fname).crop(0, 3).load_data()
    raw.pick_types(meg=True, eeg=True, exclude=())
    if proj:
        raw.apply_proj()
    else:
        raw.del_proj()
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        cov = compute_raw_covariance(raw)
    W, _, C = compute_whitener(cov, raw.info, pca=pca, return_colorer=True,
                               verbose='error')
    n_channels = len(raw.ch_names)
    n_reduced = len(raw.ch_names)
    rank = n_channels - len(raw.info['projs'])
    n_reduced = rank if pca is True else n_channels
    assert W.shape == C.shape[::-1] == (n_reduced, n_channels)
    # round-trip mults
    round_trip = np.dot(W, C)
    if pca is True:
        assert_allclose(round_trip, np.eye(n_reduced), atol=1e-7)
    elif pca == 'white':
        # Our first few rows/cols are zeroed out in the white space
        assert_allclose(round_trip[-rank:, -rank:],
                        np.eye(rank), atol=1e-7)
    else:
        assert pca is False
        assert_allclose(round_trip, np.eye(n_channels), atol=0.05)


def test_cov_mismatch():
    """Test estimation with MEG<->Head mismatch."""
    raw = read_raw_fif(raw_fname).crop(0, 5).load_data()
    events = find_events(raw, stim_channel='STI 014')
    raw.pick_channels(raw.ch_names[:5])
    raw.add_proj([], remove_existing=True)
    epochs = Epochs(raw, events, None, tmin=-0.2, tmax=0., preload=True)
    for kind in ('shift', 'None'):
        epochs_2 = epochs.copy()
        # This should be fine
        compute_covariance([epochs, epochs_2])
        if kind == 'shift':
            epochs_2.info['dev_head_t']['trans'][:3, 3] += 0.001
        else:  # None
            epochs_2.info['dev_head_t'] = None
        pytest.raises(ValueError, compute_covariance, [epochs, epochs_2])
        compute_covariance([epochs, epochs_2], on_mismatch='ignore')
        with pytest.raises(RuntimeWarning, match='transform mismatch'):
            compute_covariance([epochs, epochs_2], on_mismatch='warn')
        pytest.raises(ValueError, compute_covariance, epochs,
                      on_mismatch='x')
    # This should work
    epochs.info['dev_head_t'] = None
    epochs_2.info['dev_head_t'] = None
    compute_covariance([epochs, epochs_2], method=None)


def test_cov_order():
    """Test covariance ordering."""
    raw = read_raw_fif(raw_fname)
    raw.set_eeg_reference(projection=True)
    info = raw.info
    # add MEG channel with low enough index number to affect EEG if
    # order is incorrect
    info['bads'] += ['MEG 0113']
    ch_names = [info['ch_names'][pick]
                for pick in pick_types(info, meg=False, eeg=True)]
    cov = read_cov(cov_fname)
    # no avg ref present warning
    prepare_noise_cov(cov, info, ch_names, verbose='error')
    # big reordering
    cov_reorder = cov.copy()
    order = np.random.RandomState(0).permutation(np.arange(len(cov.ch_names)))
    cov_reorder['names'] = [cov['names'][ii] for ii in order]
    cov_reorder['data'] = cov['data'][order][:, order]
    # Make sure we did this properly
    _assert_reorder(cov_reorder, cov, order)
    # Now check some functions that should get the same result for both
    # regularize
    with pytest.raises(ValueError, match='rank, if str'):
        regularize(cov, info, rank='foo')
    with pytest.raises(TypeError, match='rank must be'):
        regularize(cov, info, rank=False)
    with pytest.raises(TypeError, match='rank must be'):
        regularize(cov, info, rank=1.)
    cov_reg = regularize(cov, info, rank='full')
    cov_reg_reorder = regularize(cov_reorder, info, rank='full')
    _assert_reorder(cov_reg_reorder, cov_reg, order)
    # prepare_noise_cov
    cov_prep = prepare_noise_cov(cov, info, ch_names)
    cov_prep_reorder = prepare_noise_cov(cov, info, ch_names)
    _assert_reorder(cov_prep, cov_prep_reorder,
                    order=np.arange(len(cov_prep['names'])))
    # compute_whitener
    whitener, w_ch_names, n_nzero = compute_whitener(
        cov, info, return_rank=True)
    assert whitener.shape[0] == whitener.shape[1]
    whitener_2, w_ch_names_2, n_nzero_2 = compute_whitener(
        cov_reorder, info, return_rank=True)
    assert_array_equal(w_ch_names_2, w_ch_names)
    assert_allclose(whitener_2, whitener, rtol=1e-6)
    assert n_nzero == n_nzero_2
    # with pca
    assert n_nzero < whitener.shape[0]
    whitener_pca, w_ch_names_pca, n_nzero_pca = compute_whitener(
        cov, info, pca=True, return_rank=True)
    assert_array_equal(w_ch_names_pca, w_ch_names)
    assert n_nzero_pca == n_nzero
    assert whitener_pca.shape == (n_nzero_pca, len(w_ch_names))
    # whiten_evoked
    evoked = read_evokeds(ave_fname)[0]
    evoked_white = whiten_evoked(evoked, cov)
    evoked_white_2 = whiten_evoked(evoked, cov_reorder)
    assert_allclose(evoked_white_2.data, evoked_white.data, atol=1e-7)


def _assert_reorder(cov_new, cov_orig, order):
    """Check that we get the same result under reordering."""
    inv_order = np.argsort(order)
    assert_array_equal([cov_new['names'][ii] for ii in inv_order],
                       cov_orig['names'])
    assert_allclose(cov_new['data'][inv_order][:, inv_order],
                    cov_orig['data'], atol=1e-20)


def test_ad_hoc_cov(tmpdir):
    """Test ad hoc cov creation and I/O."""
    out_fname = tmpdir.join('test-cov.fif')
    evoked = read_evokeds(ave_fname)[0]
    cov = make_ad_hoc_cov(evoked.info)
    cov.save(out_fname)
    assert 'Covariance' in repr(cov)
    cov2 = read_cov(out_fname)
    assert_array_almost_equal(cov['data'], cov2['data'])
    std = dict(grad=2e-13, mag=10e-15, eeg=0.1e-6)
    cov = make_ad_hoc_cov(evoked.info, std)
    cov.save(out_fname)
    assert 'Covariance' in repr(cov)
    cov2 = read_cov(out_fname)
    assert_array_almost_equal(cov['data'], cov2['data'])
    cov['data'] = np.diag(cov['data'])
    with pytest.raises(RuntimeError, match='attributes inconsistent'):
        cov._get_square()
    cov['diag'] = False
    cov._get_square()
    cov['data'] = np.diag(cov['data'])
    with pytest.raises(RuntimeError, match='attributes inconsistent'):
        cov._get_square()


def test_io_cov(tmpdir):
    """Test IO for noise covariance matrices."""
    cov = read_cov(cov_fname)
    cov['method'] = 'empirical'
    cov['loglik'] = -np.inf
    cov.save(tmpdir.join('test-cov.fif'))
    cov2 = read_cov(tmpdir.join('test-cov.fif'))
    assert_array_almost_equal(cov.data, cov2.data)
    assert_equal(cov['method'], cov2['method'])
    assert_equal(cov['loglik'], cov2['loglik'])
    assert 'Covariance' in repr(cov)

    cov2 = read_cov(cov_gz_fname)
    assert_array_almost_equal(cov.data, cov2.data)
    cov2.save(tmpdir.join('test-cov.fif.gz'))
    cov2 = read_cov(tmpdir.join('test-cov.fif.gz'))
    assert_array_almost_equal(cov.data, cov2.data)

    cov['bads'] = ['EEG 039']
    cov_sel = pick_channels_cov(cov, exclude=cov['bads'])
    assert cov_sel['dim'] == (len(cov['data']) - len(cov['bads']))
    assert cov_sel['data'].shape == (cov_sel['dim'], cov_sel['dim'])
    cov_sel.save(tmpdir.join('test-cov.fif'))

    cov2 = read_cov(cov_gz_fname)
    assert_array_almost_equal(cov.data, cov2.data)
    cov2.save(tmpdir.join('test-cov.fif.gz'))
    cov2 = read_cov(tmpdir.join('test-cov.fif.gz'))
    assert_array_almost_equal(cov.data, cov2.data)

    # test warnings on bad filenames
    cov_badname = tmpdir.join('test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='-cov.fif'):
        write_cov(cov_badname, cov)
    with pytest.warns(RuntimeWarning, match='-cov.fif'):
        read_cov(cov_badname)


@pytest.mark.parametrize('method', (None, 'empirical', 'shrunk'))
def test_cov_estimation_on_raw(method, tmpdir):
    """Test estimation from raw (typically empty room)."""
    if method == 'shrunk':
        try:
            import sklearn  # noqa: F401
        except Exception as exp:
            pytest.skip('sklearn is required, got %s' % (exp,))
    raw = read_raw_fif(raw_fname, preload=True)
    cov_mne = read_cov(erm_cov_fname)
    method_params = dict(shrunk=dict(shrinkage=[0]))

    # The pure-string uses the more efficient numpy-based method, the
    # the list gets triaged to compute_covariance (should be equivalent
    # but use more memory)
    with pytest.warns(None):  # can warn about EEG ref
        cov = compute_raw_covariance(
            raw, tstep=None, method=method, rank='full',
            method_params=method_params)
    assert_equal(cov.ch_names, cov_mne.ch_names)
    assert_equal(cov.nfree, cov_mne.nfree)
    assert_snr(cov.data, cov_mne.data, 1e6)

    # test equivalence with np.cov
    cov_np = np.cov(raw.copy().pick_channels(cov['names']).get_data(), ddof=1)
    if method != 'shrunk':  # can check all
        off_diag = np.triu_indices(cov_np.shape[0])
    else:
        # We explicitly zero out off-diag entries between channel types,
        # so let's just check MEG off-diag entries
        off_diag = np.triu_indices(len(pick_types(raw.info, meg=True,
                                                  exclude=())))
    for other in (cov_mne, cov):
        assert_allclose(np.diag(cov_np), np.diag(other.data), rtol=5e-6)
        assert_allclose(cov_np[off_diag], other.data[off_diag], rtol=4e-3)
        assert_snr(cov.data, other.data, 1e6)

    # tstep=0.2 (default)
    with pytest.warns(None):  # can warn about EEG ref
        cov = compute_raw_covariance(raw, method=method, rank='full',
                                     method_params=method_params)
    assert_equal(cov.nfree, cov_mne.nfree - 120)  # cutoff some samples
    assert_snr(cov.data, cov_mne.data, 170)

    # test IO when computation done in Python
    cov.save(tmpdir.join('test-cov.fif'))  # test saving
    cov_read = read_cov(tmpdir.join('test-cov.fif'))
    assert cov_read.ch_names == cov.ch_names
    assert cov_read.nfree == cov.nfree
    assert_array_almost_equal(cov.data, cov_read.data)

    # test with a subset of channels
    raw_pick = raw.copy().pick_channels(raw.ch_names[:5])
    raw_pick.info.normalize_proj()
    cov = compute_raw_covariance(raw_pick, tstep=None, method=method,
                                 rank='full', method_params=method_params)
    assert cov_mne.ch_names[:5] == cov.ch_names
    assert_snr(cov.data, cov_mne.data[:5, :5], 5e6)
    cov = compute_raw_covariance(raw_pick, method=method, rank='full',
                                 method_params=method_params)
    assert_snr(cov.data, cov_mne.data[:5, :5], 90)  # cutoff samps
    # make sure we get a warning with too short a segment
    raw_2 = read_raw_fif(raw_fname).crop(0, 1)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        cov = compute_raw_covariance(raw_2, method=method,
                                     method_params=method_params)
    # no epochs found due to rejection
    pytest.raises(ValueError, compute_raw_covariance, raw, tstep=None,
                  method='empirical', reject=dict(eog=200e-6))
    # but this should work
    with pytest.warns(None):  # sklearn
        cov = compute_raw_covariance(
            raw.copy().crop(0, 10.), tstep=None, method=method,
            reject=dict(eog=1000e-6), method_params=method_params,
            verbose='error')


@pytest.mark.slowtest
@requires_sklearn
def test_cov_estimation_on_raw_reg():
    """Test estimation from raw with regularization."""
    raw = read_raw_fif(raw_fname, preload=True)
    raw.info['sfreq'] /= 10.
    raw = RawArray(raw._data[:, ::10].copy(), raw.info)  # decimate for speed
    cov_mne = read_cov(erm_cov_fname)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        # XXX don't use "shrunk" here, for some reason it makes Travis 2.7
        # hang... "diagonal_fixed" is much faster. Use long epochs for speed.
        cov = compute_raw_covariance(raw, tstep=5., method='diagonal_fixed')
    assert_snr(cov.data, cov_mne.data, 5)


def _assert_cov(cov, cov_desired, tol=0.005, nfree=True):
    assert_equal(cov.ch_names, cov_desired.ch_names)
    err = (linalg.norm(cov.data - cov_desired.data, ord='fro') /
           linalg.norm(cov.data, ord='fro'))
    assert err < tol, '%s >= %s' % (err, tol)
    if nfree:
        assert_equal(cov.nfree, cov_desired.nfree)


@pytest.mark.slowtest
@pytest.mark.parametrize('rank', ('full', None))
def test_cov_estimation_with_triggers(rank, tmpdir):
    """Test estimation from raw with triggers."""
    raw = read_raw_fif(raw_fname)
    raw.set_eeg_reference(projection=True).load_data()
    events = find_events(raw, stim_channel='STI 014')
    event_ids = [1, 2, 3, 4]
    reject = dict(grad=10000e-13, mag=4e-12, eeg=80e-6, eog=150e-6)

    # cov with merged events and keep_sample_mean=True
    events_merged = merge_events(events, event_ids, 1234)
    epochs = Epochs(raw, events_merged, 1234, tmin=-0.2, tmax=0,
                    baseline=(-0.2, -0.1), proj=True,
                    reject=reject, preload=True)

    cov = compute_covariance(epochs, keep_sample_mean=True)
    cov_km = read_cov(cov_km_fname)
    # adjust for nfree bug
    cov_km['nfree'] -= 1
    _assert_cov(cov, cov_km)

    # Test with tmin and tmax (different but not too much)
    cov_tmin_tmax = compute_covariance(epochs, tmin=-0.19, tmax=-0.01)
    assert np.all(cov.data != cov_tmin_tmax.data)
    err = (linalg.norm(cov.data - cov_tmin_tmax.data, ord='fro') /
           linalg.norm(cov_tmin_tmax.data, ord='fro'))
    assert err < 0.05

    # cov using a list of epochs and keep_sample_mean=True
    epochs = [Epochs(raw, events, ev_id, tmin=-0.2, tmax=0,
                     baseline=(-0.2, -0.1), proj=True, reject=reject)
              for ev_id in event_ids]
    cov2 = compute_covariance(epochs, keep_sample_mean=True)
    assert_array_almost_equal(cov.data, cov2.data)
    assert cov.ch_names == cov2.ch_names

    # cov with keep_sample_mean=False using a list of epochs
    cov = compute_covariance(epochs, keep_sample_mean=False)
    assert cov_km.nfree == cov.nfree
    _assert_cov(cov, read_cov(cov_fname), nfree=False)

    method_params = {'empirical': {'assume_centered': False}}
    pytest.raises(ValueError, compute_covariance, epochs,
                  keep_sample_mean=False, method_params=method_params)
    pytest.raises(ValueError, compute_covariance, epochs,
                  keep_sample_mean=False, method='shrunk', rank=rank)

    # test IO when computation done in Python
    cov.save(tmpdir.join('test-cov.fif'))  # test saving
    cov_read = read_cov(tmpdir.join('test-cov.fif'))
    _assert_cov(cov, cov_read, 1e-5)

    # cov with list of epochs with different projectors
    epochs = [Epochs(raw, events[:1], None, tmin=-0.2, tmax=0,
                     baseline=(-0.2, -0.1), proj=True),
              Epochs(raw, events[:1], None, tmin=-0.2, tmax=0,
                     baseline=(-0.2, -0.1), proj=False)]
    # these should fail
    pytest.raises(ValueError, compute_covariance, epochs)
    pytest.raises(ValueError, compute_covariance, epochs, projs=None)
    # these should work, but won't be equal to above
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        cov = compute_covariance(epochs, projs=epochs[0].info['projs'])
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        cov = compute_covariance(epochs, projs=[])

    # test new dict support
    epochs = Epochs(raw, events, dict(a=1, b=2, c=3, d=4), tmin=-0.01, tmax=0,
                    proj=True, reject=reject, preload=True)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        compute_covariance(epochs)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        compute_covariance(epochs, projs=[])
    pytest.raises(TypeError, compute_covariance, epochs, projs='foo')
    pytest.raises(TypeError, compute_covariance, epochs, projs=['foo'])


def test_arithmetic_cov():
    """Test arithmetic with noise covariance matrices."""
    cov = read_cov(cov_fname)
    cov_sum = cov + cov
    assert_array_almost_equal(2 * cov.nfree, cov_sum.nfree)
    assert_array_almost_equal(2 * cov.data, cov_sum.data)
    assert cov.ch_names == cov_sum.ch_names

    cov += cov
    assert_array_almost_equal(cov_sum.nfree, cov.nfree)
    assert_array_almost_equal(cov_sum.data, cov.data)
    assert cov_sum.ch_names == cov.ch_names


def test_regularize_cov():
    """Test cov regularization."""
    raw = read_raw_fif(raw_fname)
    raw.info['bads'].append(raw.ch_names[0])  # test with bad channels
    noise_cov = read_cov(cov_fname)
    # Regularize noise cov
    reg_noise_cov = regularize(noise_cov, raw.info,
                               mag=0.1, grad=0.1, eeg=0.1, proj=True,
                               exclude='bads', rank='full')
    assert noise_cov['dim'] == reg_noise_cov['dim']
    assert noise_cov['data'].shape == reg_noise_cov['data'].shape
    assert np.mean(noise_cov['data'] < reg_noise_cov['data']) < 0.08
    # make sure all args are represented
    assert set(_DATA_CH_TYPES_SPLIT) - set(_get_args(regularize)) == set()


def test_whiten_evoked():
    """Test whitening of evoked data."""
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=True)
    cov = read_cov(cov_fname)

    ###########################################################################
    # Show result
    picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=False,
                       exclude='bads')

    noise_cov = regularize(cov, evoked.info, grad=0.1, mag=0.1, eeg=0.1,
                           exclude='bads', rank='full')

    evoked_white = whiten_evoked(evoked, noise_cov, picks, diag=True)
    whiten_baseline_data = evoked_white.data[picks][:, evoked.times < 0]
    mean_baseline = np.mean(np.abs(whiten_baseline_data), axis=1)
    assert np.all(mean_baseline < 1.)
    assert np.all(mean_baseline > 0.2)

    # degenerate
    cov_bad = pick_channels_cov(cov, include=evoked.ch_names[:10])
    pytest.raises(RuntimeError, whiten_evoked, evoked, cov_bad, picks)


def test_regularized_covariance():
    """Test unchanged data with regularized_covariance."""
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=True)
    data = evoked.data.copy()
    # check that input data remain unchanged. gh-5698
    _regularized_covariance(data)
    assert_allclose(data, evoked.data, atol=1e-20)


@requires_sklearn
def test_auto_low_rank():
    """Test probabilistic low rank estimators."""
    n_samples, n_features, rank = 400, 10, 5
    sigma = 0.1

    def get_data(n_samples, n_features, rank, sigma):
        rng = np.random.RandomState(42)
        W = rng.randn(n_features, n_features)
        X = rng.randn(n_samples, rank)
        U, _, _ = linalg.svd(W.copy())
        X = np.dot(X, U[:, :rank].T)

        sigmas = sigma * rng.rand(n_features) + sigma / 2.
        X += rng.randn(n_samples, n_features) * sigmas
        return X

    X = get_data(n_samples=n_samples, n_features=n_features, rank=rank,
                 sigma=sigma)
    method_params = {'iter_n_components': [4, 5, 6]}
    cv = 3
    n_jobs = 1
    mode = 'factor_analysis'
    rescale = 1e8
    X *= rescale
    est, info = _auto_low_rank_model(X, mode=mode, n_jobs=n_jobs,
                                     method_params=method_params,
                                     cv=cv)
    assert_equal(info['best'], rank)

    X = get_data(n_samples=n_samples, n_features=n_features, rank=rank,
                 sigma=sigma)
    method_params = {'iter_n_components': [n_features + 5]}
    msg = ('You are trying to estimate %i components on matrix '
           'with %i features.') % (n_features + 5, n_features)
    with pytest.warns(RuntimeWarning, match=msg):
        _auto_low_rank_model(X, mode=mode, n_jobs=n_jobs,
                             method_params=method_params, cv=cv)


@pytest.mark.slowtest
@pytest.mark.parametrize('rank', ('full', None, 'info'))
@requires_sklearn
def test_compute_covariance_auto_reg(rank):
    """Test automated regularization."""
    raw = read_raw_fif(raw_fname, preload=True)
    raw.resample(100, npad='auto')  # much faster estimation
    events = find_events(raw, stim_channel='STI 014')
    event_ids = [1, 2, 3, 4]
    reject = dict(mag=4e-12)

    # cov with merged events and keep_sample_mean=True
    events_merged = merge_events(events, event_ids, 1234)
    # we need a few channels for numerical reasons in PCA/FA
    picks = pick_types(raw.info, meg='mag', eeg=False)[:10]
    raw.pick_channels([raw.ch_names[pick] for pick in picks])
    raw.info.normalize_proj()
    epochs = Epochs(
        raw, events_merged, 1234, tmin=-0.2, tmax=0,
        baseline=(-0.2, -0.1), proj=True, reject=reject, preload=True)
    epochs = epochs.crop(None, 0)[:5]

    method_params = dict(factor_analysis=dict(iter_n_components=[3]),
                         pca=dict(iter_n_components=[3]))

    covs = compute_covariance(epochs, method='auto',
                              method_params=method_params,
                              return_estimators=True, rank=rank)
    # make sure regularization produces structured differencess
    diag_mask = np.eye(len(epochs.ch_names)).astype(bool)
    off_diag_mask = np.invert(diag_mask)
    for cov_a, cov_b in itt.combinations(covs, 2):
        if (cov_a['method'] == 'diagonal_fixed' and
                # here we have diagnoal or no regularization.
                cov_b['method'] == 'empirical' and rank == 'full'):

            assert not np.any(cov_a['data'][diag_mask] ==
                              cov_b['data'][diag_mask])

            # but the rest is the same
            assert_allclose(cov_a['data'][off_diag_mask],
                            cov_b['data'][off_diag_mask], rtol=1e-12)

        else:
            # and here we have shrinkage everywhere.
            assert not np.any(cov_a['data'][diag_mask] ==
                              cov_b['data'][diag_mask])

            assert not np.any(cov_a['data'][diag_mask] ==
                              cov_b['data'][diag_mask])

    logliks = [c['loglik'] for c in covs]
    assert np.diff(logliks).max() <= 0  # descending order

    methods = ['empirical', 'ledoit_wolf', 'oas', 'shrunk', 'shrinkage']
    if rank == 'full':
        methods.extend(['factor_analysis', 'pca'])
    with catch_logging() as log:
        cov3 = compute_covariance(epochs, method=methods,
                                  method_params=method_params, projs=None,
                                  return_estimators=True, rank=rank,
                                  verbose=True)
    log = log.getvalue().split('\n')
    if rank is None:
        assert '    Setting small MAG eigenvalues to zero (without PCA)' in log
        assert 'Reducing data rank from 10 -> 7' in log
    else:
        assert 'Reducing' not in log
    method_names = [cov['method'] for cov in cov3]
    best_bounds = [-45, -35]
    bounds = [-55, -45] if rank == 'full' else best_bounds
    for method in set(methods) - {'empirical', 'shrunk'}:
        this_lik = cov3[method_names.index(method)]['loglik']
        assert bounds[0] < this_lik < bounds[1]
    this_lik = cov3[method_names.index('shrunk')]['loglik']
    assert best_bounds[0] < this_lik < best_bounds[1]
    this_lik = cov3[method_names.index('empirical')]['loglik']
    bounds = [-110, -100] if rank == 'full' else best_bounds
    assert bounds[0] < this_lik < bounds[1]

    assert_equal({c['method'] for c in cov3}, set(methods))

    cov4 = compute_covariance(epochs, method=methods,
                              method_params=method_params, projs=None,
                              return_estimators=False, rank=rank)
    assert cov3[0]['method'] == cov4['method']  # ordering

    # invalid prespecified method
    pytest.raises(ValueError, compute_covariance, epochs, method='pizza')

    # invalid scalings
    pytest.raises(ValueError, compute_covariance, epochs, method='shrunk',
                  scalings=dict(misc=123))


def _cov_rank(cov, info, proj=True):
    # ignore warnings about rank mismatches: sometimes we will intentionally
    # violate the computed/info assumption, such as when using SSS with
    # `rank='full'`
    with pytest.warns(None):
        return _compute_rank_int(cov, info=info, proj=proj)


@pytest.fixture(scope='module')
def raw_epochs_events():
    """Create raw, epochs, and events for tests."""
    raw = read_raw_fif(raw_fname).set_eeg_reference(projection=True).crop(0, 3)
    raw = maxwell_filter(raw, regularize=None)  # heavily reduce the rank
    assert raw.info['bads'] == []  # no bads
    events = make_fixed_length_events(raw)
    epochs = Epochs(raw, events, tmin=-0.2, tmax=0, preload=True)
    return (raw, epochs, events)


@requires_sklearn
@pytest.mark.parametrize('rank', (None, 'full', 'info'))
def test_low_rank_methods(rank, raw_epochs_events):
    """Test low-rank covariance matrix estimation."""
    epochs = raw_epochs_events[1]
    sss_proj_rank = 139  # 80 MEG + 60 EEG - 1 proj
    n_ch = 366
    methods = ('empirical', 'diagonal_fixed', 'oas')
    bounds = {
        'None': dict(empirical=(-15000, -5000),
                     diagonal_fixed=(-1500, -500),
                     oas=(-700, -600)),
        'full': dict(empirical=(-18000, -8000),
                     diagonal_fixed=(-2000, -1600),
                     oas=(-1600, -1000)),
        'info': dict(empirical=(-15000, -5000),
                     diagonal_fixed=(-700, -600),
                     oas=(-700, -600)),
    }
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        covs = compute_covariance(
            epochs, method=methods, return_estimators=True, rank=rank,
            verbose=True)
    for cov in covs:
        method = cov['method']
        these_bounds = bounds[str(rank)][method]
        this_rank = _cov_rank(cov, epochs.info, proj=(rank != 'full'))
        if rank == 'full' and method != 'empirical':
            assert this_rank == n_ch
        else:
            assert this_rank == sss_proj_rank
        assert these_bounds[0] < cov['loglik'] < these_bounds[1], \
            (rank, method)


@requires_sklearn
def test_low_rank_cov(raw_epochs_events):
    """Test additional properties of low rank computations."""
    raw, epochs, events = raw_epochs_events
    sss_proj_rank = 139  # 80 MEG + 60 EEG - 1 proj
    n_ch = 366
    proj_rank = 365  # one EEG proj
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        emp_cov = compute_covariance(epochs)
    # Test equivalence with mne.cov.regularize subspace
    with pytest.raises(ValueError, match='are dependent.*must equal'):
        regularize(emp_cov, epochs.info, rank=None, mag=0.1, grad=0.2)
    assert _cov_rank(emp_cov, epochs.info) == sss_proj_rank
    reg_cov = regularize(emp_cov, epochs.info, proj=True, rank='full')
    assert _cov_rank(reg_cov, epochs.info) == proj_rank
    with pytest.warns(RuntimeWarning, match='exceeds the theoretical'):
        _compute_rank_int(reg_cov, info=epochs.info)
    del reg_cov
    with catch_logging() as log:
        reg_r_cov = regularize(emp_cov, epochs.info, proj=True, rank=None,
                               verbose=True)
    log = log.getvalue()
    assert 'jointly' in log
    assert _cov_rank(reg_r_cov, epochs.info) == sss_proj_rank
    reg_r_only_cov = regularize(emp_cov, epochs.info, proj=False, rank=None)
    assert _cov_rank(reg_r_only_cov, epochs.info) == sss_proj_rank
    assert_allclose(reg_r_only_cov['data'], reg_r_cov['data'])
    del reg_r_only_cov, reg_r_cov

    # test that rank=306 is same as rank='full'
    epochs_meg = epochs.copy().pick_types(meg=True)
    assert len(epochs_meg.ch_names) == 306
    epochs_meg.info.update(bads=[], projs=[])
    cov_full = compute_covariance(epochs_meg, method='oas',
                                  rank='full', verbose='error')
    assert _cov_rank(cov_full, epochs_meg.info) == 306
    with pytest.warns(RuntimeWarning, match='few samples'):
        cov_dict = compute_covariance(epochs_meg, method='oas',
                                      rank=dict(meg=306))
    assert _cov_rank(cov_dict, epochs_meg.info) == 306
    assert_allclose(cov_full['data'], cov_dict['data'])
    cov_dict = compute_covariance(epochs_meg, method='oas',
                                  rank=dict(meg=306), verbose='error')
    assert _cov_rank(cov_dict, epochs_meg.info) == 306
    assert_allclose(cov_full['data'], cov_dict['data'])

    # Work with just EEG data to simplify projection / rank reduction
    raw = raw.copy().pick_types(meg=False, eeg=True)
    n_proj = 2
    raw.add_proj(compute_proj_raw(raw, n_eeg=n_proj))
    n_ch = len(raw.ch_names)
    rank = n_ch - n_proj - 1  # plus avg proj
    assert len(raw.info['projs']) == 3
    epochs = Epochs(raw, events, tmin=-0.2, tmax=0, preload=True)
    assert len(raw.ch_names) == n_ch
    emp_cov = compute_covariance(epochs, rank='full', verbose='error')
    assert _cov_rank(emp_cov, epochs.info) == rank
    reg_cov = regularize(emp_cov, epochs.info, proj=True, rank='full')
    assert _cov_rank(reg_cov, epochs.info) == rank
    reg_r_cov = regularize(emp_cov, epochs.info, proj=False, rank=None)
    assert _cov_rank(reg_r_cov, epochs.info) == rank
    dia_cov = compute_covariance(epochs, rank=None, method='diagonal_fixed',
                                 verbose='error')
    assert _cov_rank(dia_cov, epochs.info) == rank
    assert_allclose(dia_cov['data'], reg_cov['data'])
    epochs.pick_channels(epochs.ch_names[:103])
    # degenerate
    with pytest.raises(ValueError, match='can.*only be used with rank="full"'):
        compute_covariance(epochs, rank=None, method='pca')
    with pytest.raises(ValueError, match='can.*only be used with rank="full"'):
        compute_covariance(epochs, rank=None, method='factor_analysis')


@testing.requires_testing_data
@requires_sklearn
def test_cov_ctf():
    """Test basic cov computation on ctf data with/without compensation."""
    raw = read_raw_ctf(ctf_fname).crop(0., 2.).load_data()
    events = make_fixed_length_events(raw, 99999)
    assert len(events) == 2
    ch_names = [raw.info['ch_names'][pick]
                for pick in pick_types(raw.info, meg=True, eeg=False,
                                       ref_meg=False)]

    for comp in [0, 1]:
        raw.apply_gradient_compensation(comp)
        epochs = Epochs(raw, events, None, -0.2, 0.2, preload=True)
        with pytest.warns(RuntimeWarning, match='Too few samples'):
            noise_cov = compute_covariance(epochs, tmax=0.,
                                           method=['empirical'])
        prepare_noise_cov(noise_cov, raw.info, ch_names)

    raw.apply_gradient_compensation(0)
    epochs = Epochs(raw, events, None, -0.2, 0.2, preload=True)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        noise_cov = compute_covariance(epochs, tmax=0., method=['empirical'])
    raw.apply_gradient_compensation(1)

    # TODO This next call in principle should fail.
    prepare_noise_cov(noise_cov, raw.info, ch_names)

    # make sure comps matrices was not removed from raw
    assert raw.info['comps'], 'Comps matrices removed'


def test_equalize_channels():
    """Test equalization of channels for instances of Covariance."""
    cov1 = make_ad_hoc_cov(create_info(['CH1', 'CH2', 'CH3', 'CH4'], sfreq=1.0,
                                       ch_types='eeg'))
    cov2 = make_ad_hoc_cov(create_info(['CH5', 'CH1', 'CH2'], sfreq=1.0,
                                       ch_types='eeg'))
    cov1, cov2 = equalize_channels([cov1, cov2])
    assert cov1.ch_names == ['CH1', 'CH2']
    assert cov2.ch_names == ['CH1', 'CH2']


run_tests_if_main()
