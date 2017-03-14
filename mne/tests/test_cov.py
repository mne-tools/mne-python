# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
from nose.tools import assert_raises
import numpy as np
from scipy import linalg
import warnings
import itertools as itt

from mne.cov import (regularize, whiten_evoked, _estimate_rank_meeg_cov,
                     _auto_low_rank_model, _apply_scaling_cov,
                     _undo_scaling_cov, prepare_noise_cov, compute_whitener,
                     _apply_scaling_array, _undo_scaling_array)

from mne import (read_cov, write_cov, Epochs, merge_events,
                 find_events, compute_raw_covariance,
                 compute_covariance, read_evokeds, compute_proj_raw,
                 pick_channels_cov, pick_channels, pick_types, pick_info,
                 make_ad_hoc_cov)
from mne.io import read_raw_fif, RawArray, read_info
from mne.tests.common import assert_naming, assert_snr
from mne.utils import (_TempDir, slow_test, requires_sklearn_0_15,
                       run_tests_if_main)
from mne.io.proc_history import _get_sss_rank
from mne.io.pick import channel_type, _picks_by_type

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
cov_fname = op.join(base_dir, 'test-cov.fif')
cov_gz_fname = op.join(base_dir, 'test-cov.fif.gz')
cov_km_fname = op.join(base_dir, 'test-km-cov.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
erm_cov_fname = op.join(base_dir, 'test_erm-cov.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


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
        with warnings.catch_warnings(record=True) as w:
            compute_covariance([epochs, epochs_2])
            assert_equal(len(w), 0)
            if kind == 'shift':
                epochs_2.info['dev_head_t']['trans'][:3, 3] += 0.001
            else:  # None
                epochs_2.info['dev_head_t'] = None
            assert_raises(ValueError, compute_covariance, [epochs, epochs_2])
            assert_equal(len(w), 0)
            compute_covariance([epochs, epochs_2], on_mismatch='ignore')
            assert_equal(len(w), 0)
            compute_covariance([epochs, epochs_2], on_mismatch='warn')
            assert_raises(ValueError, compute_covariance, epochs,
                          on_mismatch='x')
        assert_true(any('transform mismatch' in str(ww.message) for ww in w))
    # This should work
    epochs.info['dev_head_t'] = None
    epochs_2.info['dev_head_t'] = None
    compute_covariance([epochs, epochs_2], method=None)


def test_cov_order():
    """Test covariance ordering."""
    info = read_info(raw_fname)
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
    cov_reg = regularize(cov, info)
    cov_reg_reorder = regularize(cov_reorder, info)
    _assert_reorder(cov_reg_reorder, cov_reg, order)
    # prepare_noise_cov
    cov_prep = prepare_noise_cov(cov, info, ch_names)
    cov_prep_reorder = prepare_noise_cov(cov, info, ch_names)
    _assert_reorder(cov_prep, cov_prep_reorder,
                    order=np.arange(len(cov_prep['names'])))
    # compute_whitener
    whitener, w_ch_names = compute_whitener(cov, info)
    whitener_2, w_ch_names_2 = compute_whitener(cov_reorder, info)
    assert_array_equal(w_ch_names_2, w_ch_names)
    assert_allclose(whitener_2, whitener)
    # whiten_evoked
    evoked = read_evokeds(ave_fname)[0]
    evoked_white = whiten_evoked(evoked, cov)
    evoked_white_2 = whiten_evoked(evoked, cov_reorder)
    assert_allclose(evoked_white_2.data, evoked_white.data)


def _assert_reorder(cov_new, cov_orig, order):
    """Check that we get the same result under reordering."""
    inv_order = np.argsort(order)
    assert_array_equal([cov_new['names'][ii] for ii in inv_order],
                       cov_orig['names'])
    assert_allclose(cov_new['data'][inv_order][:, inv_order],
                    cov_orig['data'], atol=1e-20)


def test_ad_hoc_cov():
    """Test ad hoc cov creation and I/O."""
    tempdir = _TempDir()
    out_fname = op.join(tempdir, 'test-cov.fif')
    evoked = read_evokeds(ave_fname)[0]
    cov = make_ad_hoc_cov(evoked.info)
    cov.save(out_fname)
    assert_true('Covariance' in repr(cov))
    cov2 = read_cov(out_fname)
    assert_array_almost_equal(cov['data'], cov2['data'])


def test_io_cov():
    """Test IO for noise covariance matrices."""
    tempdir = _TempDir()
    cov = read_cov(cov_fname)
    cov['method'] = 'empirical'
    cov['loglik'] = -np.inf
    cov.save(op.join(tempdir, 'test-cov.fif'))
    cov2 = read_cov(op.join(tempdir, 'test-cov.fif'))
    assert_array_almost_equal(cov.data, cov2.data)
    assert_equal(cov['method'], cov2['method'])
    assert_equal(cov['loglik'], cov2['loglik'])
    assert_true('Covariance' in repr(cov))

    cov2 = read_cov(cov_gz_fname)
    assert_array_almost_equal(cov.data, cov2.data)
    cov2.save(op.join(tempdir, 'test-cov.fif.gz'))
    cov2 = read_cov(op.join(tempdir, 'test-cov.fif.gz'))
    assert_array_almost_equal(cov.data, cov2.data)

    cov['bads'] = ['EEG 039']
    cov_sel = pick_channels_cov(cov, exclude=cov['bads'])
    assert_true(cov_sel['dim'] == (len(cov['data']) - len(cov['bads'])))
    assert_true(cov_sel['data'].shape == (cov_sel['dim'], cov_sel['dim']))
    cov_sel.save(op.join(tempdir, 'test-cov.fif'))

    cov2 = read_cov(cov_gz_fname)
    assert_array_almost_equal(cov.data, cov2.data)
    cov2.save(op.join(tempdir, 'test-cov.fif.gz'))
    cov2 = read_cov(op.join(tempdir, 'test-cov.fif.gz'))
    assert_array_almost_equal(cov.data, cov2.data)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cov_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        write_cov(cov_badname, cov)
        read_cov(cov_badname)
    assert_naming(w, 'test_cov.py', 2)


def test_cov_estimation_on_raw():
    """Test estimation from raw (typically empty room)."""
    tempdir = _TempDir()
    raw = read_raw_fif(raw_fname, preload=True)
    cov_mne = read_cov(erm_cov_fname)

    # The pure-string uses the more efficient numpy-based method, the
    # the list gets triaged to compute_covariance (should be equivalent
    # but use more memory)
    for method in (None, ['empirical']):  # None is cast to 'empirical'
        cov = compute_raw_covariance(raw, tstep=None, method=method)
        assert_equal(cov.ch_names, cov_mne.ch_names)
        assert_equal(cov.nfree, cov_mne.nfree)
        assert_snr(cov.data, cov_mne.data, 1e4)

        cov = compute_raw_covariance(raw, method=method)  # tstep=0.2 (default)
        assert_equal(cov.nfree, cov_mne.nfree - 119)  # cutoff some samples
        assert_snr(cov.data, cov_mne.data, 1e2)

        # test IO when computation done in Python
        cov.save(op.join(tempdir, 'test-cov.fif'))  # test saving
        cov_read = read_cov(op.join(tempdir, 'test-cov.fif'))
        assert_true(cov_read.ch_names == cov.ch_names)
        assert_true(cov_read.nfree == cov.nfree)
        assert_array_almost_equal(cov.data, cov_read.data)

        # test with a subset of channels
        picks = pick_channels(raw.ch_names, include=raw.ch_names[:5])
        raw_pick = raw.copy().pick_channels(
            [raw.ch_names[pick] for pick in picks])
        raw_pick.info.normalize_proj()
        cov = compute_raw_covariance(raw_pick, picks=picks, tstep=None,
                                     method=method)
        assert_true(cov_mne.ch_names[:5] == cov.ch_names)
        assert_snr(cov.data, cov_mne.data[picks][:, picks], 1e4)
        cov = compute_raw_covariance(raw_pick, picks=picks, method=method)
        assert_snr(cov.data, cov_mne.data[picks][:, picks], 90)  # cutoff samps
        # make sure we get a warning with too short a segment
        raw_2 = read_raw_fif(raw_fname).crop(0, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cov = compute_raw_covariance(raw_2, method=method)
        assert_true(any('Too few samples' in str(ww.message) for ww in w))
        # no epochs found due to rejection
        assert_raises(ValueError, compute_raw_covariance, raw, tstep=None,
                      method='empirical', reject=dict(eog=200e-6))
        # but this should work
        cov = compute_raw_covariance(raw.copy().crop(0, 10.),
                                     tstep=None, method=method,
                                     reject=dict(eog=1000e-6))


@slow_test
@requires_sklearn_0_15
def test_cov_estimation_on_raw_reg():
    """Test estimation from raw with regularization."""
    raw = read_raw_fif(raw_fname, preload=True)
    raw.info['sfreq'] /= 10.
    raw = RawArray(raw._data[:, ::10].copy(), raw.info)  # decimate for speed
    cov_mne = read_cov(erm_cov_fname)
    with warnings.catch_warnings(record=True):  # too few samples
        warnings.simplefilter('always')
        # XXX don't use "shrunk" here, for some reason it makes Travis 2.7
        # hang... "diagonal_fixed" is much faster. Use long epochs for speed.
        cov = compute_raw_covariance(raw, tstep=5., method='diagonal_fixed')
    assert_snr(cov.data, cov_mne.data, 5)


@slow_test
def test_cov_estimation_with_triggers():
    """Test estimation from raw with triggers."""
    tempdir = _TempDir()
    raw = read_raw_fif(raw_fname)
    raw.set_eeg_reference()
    events = find_events(raw, stim_channel='STI 014')
    event_ids = [1, 2, 3, 4]
    reject = dict(grad=10000e-13, mag=4e-12, eeg=80e-6, eog=150e-6)

    # cov with merged events and keep_sample_mean=True
    events_merged = merge_events(events, event_ids, 1234)
    epochs = Epochs(raw, events_merged, 1234, tmin=-0.2, tmax=0,
                    baseline=(-0.2, -0.1), proj=True,
                    reject=reject, preload=True)

    cov = compute_covariance(epochs, keep_sample_mean=True)
    cov_mne = read_cov(cov_km_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro') /
                linalg.norm(cov.data, ord='fro')) < 0.005)

    # Test with tmin and tmax (different but not too much)
    cov_tmin_tmax = compute_covariance(epochs, tmin=-0.19, tmax=-0.01)
    assert_true(np.all(cov.data != cov_tmin_tmax.data))
    assert_true((linalg.norm(cov.data - cov_tmin_tmax.data, ord='fro') /
                 linalg.norm(cov_tmin_tmax.data, ord='fro')) < 0.05)

    # cov using a list of epochs and keep_sample_mean=True
    epochs = [Epochs(raw, events, ev_id, tmin=-0.2, tmax=0,
              baseline=(-0.2, -0.1), proj=True, reject=reject)
              for ev_id in event_ids]

    cov2 = compute_covariance(epochs, keep_sample_mean=True)
    assert_array_almost_equal(cov.data, cov2.data)
    assert_true(cov.ch_names == cov2.ch_names)

    # cov with keep_sample_mean=False using a list of epochs
    cov = compute_covariance(epochs, keep_sample_mean=False)
    cov_mne = read_cov(cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro') /
                 linalg.norm(cov.data, ord='fro')) < 0.005)

    method_params = {'empirical': {'assume_centered': False}}
    assert_raises(ValueError, compute_covariance, epochs,
                  keep_sample_mean=False, method_params=method_params)

    assert_raises(ValueError, compute_covariance, epochs,
                  keep_sample_mean=False, method='factor_analysis')

    # test IO when computation done in Python
    cov.save(op.join(tempdir, 'test-cov.fif'))  # test saving
    cov_read = read_cov(op.join(tempdir, 'test-cov.fif'))
    assert_true(cov_read.ch_names == cov.ch_names)
    assert_true(cov_read.nfree == cov.nfree)
    assert_true((linalg.norm(cov.data - cov_read.data, ord='fro') /
                 linalg.norm(cov.data, ord='fro')) < 1e-5)

    # cov with list of epochs with different projectors
    epochs = [Epochs(raw, events[:4], event_ids[0], tmin=-0.2, tmax=0,
                     baseline=(-0.2, -0.1), proj=True, reject=reject),
              Epochs(raw, events[:4], event_ids[0], tmin=-0.2, tmax=0,
                     baseline=(-0.2, -0.1), proj=False, reject=reject)]
    # these should fail
    assert_raises(ValueError, compute_covariance, epochs)
    assert_raises(ValueError, compute_covariance, epochs, projs=None)
    # these should work, but won't be equal to above
    with warnings.catch_warnings(record=True) as w:  # too few samples warning
        warnings.simplefilter('always')
        cov = compute_covariance(epochs, projs=epochs[0].info['projs'])
        cov = compute_covariance(epochs, projs=[])
    assert_true(len(w) == 2)

    # test new dict support
    epochs = Epochs(raw, events, dict(a=1, b=2, c=3, d=4), tmin=-0.2, tmax=0,
                    baseline=(-0.2, -0.1), proj=True, reject=reject)
    compute_covariance(epochs)

    # projs checking
    compute_covariance(epochs, projs=[])
    assert_raises(TypeError, compute_covariance, epochs, projs='foo')
    assert_raises(TypeError, compute_covariance, epochs, projs=['foo'])


def test_arithmetic_cov():
    """Test arithmetic with noise covariance matrices."""
    cov = read_cov(cov_fname)
    cov_sum = cov + cov
    assert_array_almost_equal(2 * cov.nfree, cov_sum.nfree)
    assert_array_almost_equal(2 * cov.data, cov_sum.data)
    assert_true(cov.ch_names == cov_sum.ch_names)

    cov += cov
    assert_array_almost_equal(cov_sum.nfree, cov.nfree)
    assert_array_almost_equal(cov_sum.data, cov.data)
    assert_true(cov_sum.ch_names == cov.ch_names)


def test_regularize_cov():
    """Test cov regularization."""
    raw = read_raw_fif(raw_fname)
    raw.info['bads'].append(raw.ch_names[0])  # test with bad channels
    noise_cov = read_cov(cov_fname)
    # Regularize noise cov
    reg_noise_cov = regularize(noise_cov, raw.info,
                               mag=0.1, grad=0.1, eeg=0.1, proj=True,
                               exclude='bads')
    assert_true(noise_cov['dim'] == reg_noise_cov['dim'])
    assert_true(noise_cov['data'].shape == reg_noise_cov['data'].shape)
    assert_true(np.mean(noise_cov['data'] < reg_noise_cov['data']) < 0.08)


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
                           exclude='bads')

    evoked_white = whiten_evoked(evoked, noise_cov, picks, diag=True)
    whiten_baseline_data = evoked_white.data[picks][:, evoked.times < 0]
    mean_baseline = np.mean(np.abs(whiten_baseline_data), axis=1)
    assert_true(np.all(mean_baseline < 1.))
    assert_true(np.all(mean_baseline > 0.2))

    # degenerate
    cov_bad = pick_channels_cov(cov, include=evoked.ch_names[:10])
    assert_raises(RuntimeError, whiten_evoked, evoked, cov_bad, picks)


@slow_test
def test_rank():
    """Test cov rank estimation."""
    # Test that our rank estimation works properly on a simple case
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=False)
    cov = read_cov(cov_fname)
    ch_names = [ch for ch in evoked.info['ch_names'] if '053' not in ch and
                ch.startswith('EEG')]
    cov = prepare_noise_cov(cov, evoked.info, ch_names, None)
    assert_equal(cov['eig'][0], 0.)  # avg projector should set this to zero
    assert_true((cov['eig'][1:] > 0).all())  # all else should be > 0

    # Now do some more comprehensive tests
    raw_sample = read_raw_fif(raw_fname)

    raw_sss = read_raw_fif(hp_fif_fname)
    raw_sss.add_proj(compute_proj_raw(raw_sss))

    cov_sample = compute_raw_covariance(raw_sample)
    cov_sample_proj = compute_raw_covariance(
        raw_sample.copy().apply_proj())

    cov_sss = compute_raw_covariance(raw_sss)
    cov_sss_proj = compute_raw_covariance(
        raw_sss.copy().apply_proj())

    picks_all_sample = pick_types(raw_sample.info, meg=True, eeg=True)
    picks_all_sss = pick_types(raw_sss.info, meg=True, eeg=True)

    info_sample = pick_info(raw_sample.info, picks_all_sample)
    picks_stack_sample = [('eeg', pick_types(info_sample, meg=False,
                                             eeg=True))]
    picks_stack_sample += [('meg', pick_types(info_sample, meg=True))]
    picks_stack_sample += [('all',
                            pick_types(info_sample, meg=True, eeg=True))]

    info_sss = pick_info(raw_sss.info, picks_all_sss)
    picks_stack_somato = [('eeg', pick_types(info_sss, meg=False, eeg=True))]
    picks_stack_somato += [('meg', pick_types(info_sss, meg=True))]
    picks_stack_somato += [('all',
                            pick_types(info_sss, meg=True, eeg=True))]

    iter_tests = list(itt.product(
        [(cov_sample, picks_stack_sample, info_sample),
         (cov_sample_proj, picks_stack_sample, info_sample),
         (cov_sss, picks_stack_somato, info_sss),
         (cov_sss_proj, picks_stack_somato, info_sss)],  # sss
        [dict(mag=1e15, grad=1e13, eeg=1e6)]
    ))

    for (cov, picks_list, this_info), scalings in iter_tests:
        for ch_type, picks in picks_list:

            this_very_info = pick_info(this_info, picks)

            # compute subset of projs
            this_projs = [c['active'] and
                          len(set(c['data']['col_names'])
                              .intersection(set(this_very_info['ch_names']))) >
                          0 for c in cov['projs']]
            n_projs = sum(this_projs)

            # count channel types
            ch_types = [channel_type(this_very_info, idx)
                        for idx in range(len(picks))]
            n_eeg, n_mag, n_grad = [ch_types.count(k) for k in
                                    ['eeg', 'mag', 'grad']]
            n_meg = n_mag + n_grad
            if ch_type in ('all', 'eeg'):
                n_projs_eeg = 1
            else:
                n_projs_eeg = 0

            # check sss
            if 'proc_history' in this_very_info:
                mf = this_very_info['proc_history'][0]['max_info']
                n_free = _get_sss_rank(mf)
                if 'mag' not in ch_types and 'grad' not in ch_types:
                    n_free = 0
                # - n_projs XXX clarify
                expected_rank = n_free + n_eeg
                if n_projs > 0 and ch_type in ('all', 'eeg'):
                    expected_rank -= n_projs_eeg
            else:
                expected_rank = n_meg + n_eeg - n_projs

            C = cov['data'][np.ix_(picks, picks)]
            est_rank = _estimate_rank_meeg_cov(C, this_very_info,
                                               scalings=scalings)

            assert_equal(expected_rank, est_rank)


def test_cov_scaling():
    """Test rescaling covs"""
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=True)
    cov = read_cov(cov_fname)['data']
    cov2 = read_cov(cov_fname)['data']

    assert_array_equal(cov, cov2)
    evoked.pick_channels([evoked.ch_names[k] for k in pick_types(
        evoked.info, meg=True, eeg=True
    )])
    picks_list = _picks_by_type(evoked.info)
    scalings = dict(mag=1e15, grad=1e13, eeg=1e6)

    _apply_scaling_cov(cov2, picks_list, scalings=scalings)
    _apply_scaling_cov(cov, picks_list, scalings=scalings)
    assert_array_equal(cov, cov2)
    assert_true(cov.max() > 1)

    _undo_scaling_cov(cov2, picks_list, scalings=scalings)
    _undo_scaling_cov(cov, picks_list, scalings=scalings)
    assert_array_equal(cov, cov2)
    assert_true(cov.max() < 1)

    data = evoked.data.copy()
    _apply_scaling_array(data, picks_list, scalings=scalings)
    _undo_scaling_array(data, picks_list, scalings=scalings)
    assert_allclose(data, evoked.data, atol=1e-20)


@requires_sklearn_0_15
def test_auto_low_rank():
    """Test probabilistic low rank estimators."""
    n_samples, n_features, rank = 400, 20, 10
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
    method_params = {'iter_n_components': [9, 10, 11]}
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
           'with %i features.')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _auto_low_rank_model(X, mode=mode, n_jobs=n_jobs,
                             method_params=method_params, cv=cv)
        assert_equal(len(w), 1)
        assert_equal(msg % (n_features + 5, n_features), '%s' % w[0].message)

    method_params = {'iter_n_components': [n_features + 5]}
    assert_raises(ValueError, _auto_low_rank_model, X, mode='foo',
                  n_jobs=n_jobs, method_params=method_params, cv=cv)


@slow_test
@requires_sklearn_0_15
def test_compute_covariance_auto_reg():
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
    epochs = epochs.crop(None, 0)[:10]

    method_params = dict(factor_analysis=dict(iter_n_components=[3]),
                         pca=dict(iter_n_components=[3]))

    covs = compute_covariance(epochs, method='auto',
                              method_params=method_params,
                              return_estimators=True)

    logliks = [c['loglik'] for c in covs]
    assert_true(np.diff(logliks).max() <= 0)  # descending order

    methods = ['empirical',
               'factor_analysis',
               'ledoit_wolf',
               'pca']
    cov3 = compute_covariance(epochs, method=methods,
                              method_params=method_params, projs=None,
                              return_estimators=True)

    assert_equal(set([c['method'] for c in cov3]),
                 set(methods))

    # invalid prespecified method
    assert_raises(ValueError, compute_covariance, epochs, method='pizza')

    # invalid scalings
    assert_raises(ValueError, compute_covariance, epochs, method='shrunk',
                  scalings=dict(misc=123))

run_tests_if_main()
