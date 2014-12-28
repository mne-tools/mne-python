# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true, assert_almost_equal
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises
import numpy as np
from scipy import linalg
import warnings
import itertools as itt

from mne.cov import regularize, whiten_evoked, _estimate_rank_meeg_cov
from mne import (read_cov, write_cov, Epochs, merge_events,
                 find_events, compute_raw_data_covariance,
                 compute_covariance, read_evokeds, compute_proj_raw,
                 pick_channels_cov, pick_channels, pick_types, pick_info)
from mne.io import Raw
from mne.io.pick import channel_type
from mne.utils import _TempDir, slow_test
from mne.io.proc_history import _get_sss_rank

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
cov_fname = op.join(base_dir, 'test-cov.fif')
cov_gz_fname = op.join(base_dir, 'test-cov.fif.gz')
cov_km_fname = op.join(base_dir, 'test-km-cov.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
erm_cov_fname = op.join(base_dir, 'test_erm-cov.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    tempdir = _TempDir()
    cov = read_cov(cov_fname)
    cov.save(op.join(tempdir, 'test-cov.fif'))
    cov2 = read_cov(op.join(tempdir, 'test-cov.fif'))
    assert_array_almost_equal(cov.data, cov2.data)

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
    assert_true(len(w) == 2)


def test_cov_estimation_on_raw_segment():
    """Test estimation from raw on continuous recordings (typically empty room)
    """
    tempdir = _TempDir()
    raw = Raw(raw_fname, preload=False)
    cov = compute_raw_data_covariance(raw)
    cov_mne = read_cov(erm_cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true(linalg.norm(cov.data - cov_mne.data, ord='fro')
                / linalg.norm(cov.data, ord='fro') < 1e-4)

    # test IO when computation done in Python
    cov.save(op.join(tempdir, 'test-cov.fif'))  # test saving
    cov_read = read_cov(op.join(tempdir, 'test-cov.fif'))
    assert_true(cov_read.ch_names == cov.ch_names)
    assert_true(cov_read.nfree == cov.nfree)
    assert_array_almost_equal(cov.data, cov_read.data)

    # test with a subset of channels
    picks = pick_channels(raw.ch_names, include=raw.ch_names[:5])
    cov = compute_raw_data_covariance(raw, picks=picks)
    assert_true(cov_mne.ch_names[:5] == cov.ch_names)
    assert_true(linalg.norm(cov.data - cov_mne.data[picks][:, picks],
                ord='fro') / linalg.norm(cov.data, ord='fro') < 1e-4)
    # make sure we get a warning with too short a segment
    raw_2 = raw.crop(0, 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cov = compute_raw_data_covariance(raw_2)
    assert_true(len(w) == 1)


def test_cov_estimation_with_triggers():
    """Test estimation from raw with triggers
    """
    tempdir = _TempDir()
    raw = Raw(raw_fname, preload=False)
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
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
                 / linalg.norm(cov.data, ord='fro')) < 0.005)

    # Test with tmin and tmax (different but not too much)
    cov_tmin_tmax = compute_covariance(epochs, tmin=-0.19, tmax=-0.01)
    assert_true(np.all(cov.data != cov_tmin_tmax.data))
    assert_true((linalg.norm(cov.data - cov_tmin_tmax.data, ord='fro')
                 / linalg.norm(cov_tmin_tmax.data, ord='fro')) < 0.05)

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
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
                 / linalg.norm(cov.data, ord='fro')) < 0.005)

    # test IO when computation done in Python
    cov.save(op.join(tempdir, 'test-cov.fif'))  # test saving
    cov_read = read_cov(op.join(tempdir, 'test-cov.fif'))
    assert_true(cov_read.ch_names == cov.ch_names)
    assert_true(cov_read.nfree == cov.nfree)
    assert_true((linalg.norm(cov.data - cov_read.data, ord='fro')
                 / linalg.norm(cov.data, ord='fro')) < 1e-5)

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


def test_arithmetic_cov():
    """Test arithmetic with noise covariance matrices
    """
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
    """Test cov regularization
    """
    raw = Raw(raw_fname, preload=False)
    raw.info['bads'].append(raw.ch_names[0])  # test with bad channels
    noise_cov = read_cov(cov_fname)
    # Regularize noise cov
    reg_noise_cov = regularize(noise_cov, raw.info,
                               mag=0.1, grad=0.1, eeg=0.1, proj=True,
                               exclude='bads')
    assert_true(noise_cov['dim'] == reg_noise_cov['dim'])
    assert_true(noise_cov['data'].shape == reg_noise_cov['data'].shape)
    assert_true(np.mean(noise_cov['data'] < reg_noise_cov['data']) < 0.08)


def test_evoked_whiten():
    """Test whitening of evoked data"""
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


@slow_test
def test_rank():
    """Test cov rank estimation"""
    raw_sample = Raw(raw_fname)

    raw_sss = Raw(hp_fif_fname)
    raw_sss.add_proj(compute_proj_raw(raw_sss))

    cov_sample = compute_raw_data_covariance(raw_sample)
    cov_sample_proj = compute_raw_data_covariance(
        raw_sample.copy().apply_proj())

    cov_sss = compute_raw_data_covariance(raw_sss)
    cov_sss_proj = compute_raw_data_covariance(
        raw_sss.copy().apply_proj())

    picks_all_sample = pick_types(raw_sample.info, meg=True, eeg=True)
    picks_all_sss = pick_types(raw_sss.info, meg=True, eeg=True)

    info_sample = pick_info(raw_sample.info, picks_all_sample)
    picks_stack_sample = [('eeg', pick_types(info_sample, meg=False, eeg=True))]
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
                              .intersection(set(this_very_info['ch_names']))) > 0
                          for c in cov['projs']]
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

            assert_almost_equal(expected_rank, est_rank)
