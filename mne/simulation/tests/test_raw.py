# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings
from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import assert_true, assert_raises

from mne import (read_source_spaces, pick_types, read_trans, read_cov,
                 make_sphere_model, create_info)
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.io import Raw, RawArray
from mne.utils import _TempDir, slow_test, run_tests_if_main

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
cov_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-cov.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
bem_path = op.join(data_path, 'subjects', 'sample', 'bem')
src_fname = op.join(bem_path, 'sample-oct-2-src.fif')
bem_fname = op.join(bem_path, 'sample-320-320-320-bem-sol.fif')


def _get_data():
    """Helper to get some starting data"""
    seed = 42
    # raw with ECG channel
    raw = Raw(raw_fname).crop(0., 5.0).preload_data()
    ecg = RawArray(np.zeros((1, len(raw.times))),
                   create_info(['ECG 063'], raw.info['sfreq'], 'ecg'))
    for key in ('dev_head_t', 'buffer_size_sec', 'highpass', 'lowpass',
                'filename', 'dig'):
        ecg.info[key] = raw.info[key]
    raw.add_channels([ecg])

    src = read_source_spaces(src_fname)
    trans = read_trans(trans_fname)
    sphere = make_sphere_model('auto', 'auto', raw.info)

    sfreq = raw.info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = len(raw.times) // 10
    times = np.arange(0, n_samples) * tstep
    stc = simulate_sparse_stc(src, 10, times, random_state=seed)
    return raw, src, stc, trans, sphere


@slow_test
@testing.requires_testing_data
def test_simulate_raw_sphere():
    """Test simulation of raw data with sphere model"""
    seed = 42
    raw, src, stc, trans, sphere = _get_data()
    assert_true(len(pick_types(raw.info, meg=False, ecg=True)) == 1)

    # head pos
    head_pos_sim = dict()
    shifts = [[0.001, 0., -0.001], [-0.001, 0.001, 0.]]

    for time_key, shift in zip(raw.times[:len(shifts)], shifts):
        # Create 4x4 matrix transform and normalize
        temp_trans = deepcopy(raw.info['dev_head_t'])
        temp_trans['trans'][:3, 3] += shift
        head_pos_sim[time_key] = temp_trans['trans']

    #
    # Test raw simulation with basic parameters
    #
    raw_sim = simulate_raw(raw, stc, trans, src, sphere, read_cov(cov_fname),
                           head_pos=head_pos_sim,
                           blink=True, ecg=True, random_state=seed)
    raw_sim_2 = simulate_raw(raw, stc, trans_fname, src_fname, sphere,
                             cov_fname, head_pos=head_pos_sim,
                             blink=True, ecg=True, random_state=seed)
    assert_array_equal(raw_sim_2[:][0], raw_sim[:][0])
    # Test IO on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'sim_test_raw.fif')
    raw_sim.save(test_outname)

    raw_sim_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)
    assert_allclose(raw_sim_loaded[:][0], raw_sim[:][0], rtol=1e-6, atol=1e-20)
    del raw_sim, raw_sim_2
    # with no cov (no noise) but with artifacts, most time periods should match
    # but the EOG/ECG channels should not
    for ecg, eog in ((True, False), (False, True), (True, True)):
        raw_sim_3 = simulate_raw(raw, stc, trans, src, sphere,
                                 cov=None, head_pos=head_pos_sim,
                                 blink=eog, ecg=ecg, random_state=seed)
        raw_sim_4 = simulate_raw(raw, stc, trans, src, sphere,
                                 cov=None, head_pos=head_pos_sim,
                                 blink=False, ecg=False, random_state=seed)
        picks = np.arange(len(raw.ch_names))
        diff_picks = pick_types(raw.info, meg=False, ecg=ecg, eog=eog)
        these_picks = np.setdiff1d(picks, diff_picks)
        close = np.isclose(raw_sim_3[these_picks][0],
                           raw_sim_4[these_picks][0], atol=1e-20)
        assert_true(np.mean(close) > 0.7)
        far = ~np.isclose(raw_sim_3[diff_picks][0],
                          raw_sim_4[diff_picks][0], atol=1e-20)
        assert_true(np.mean(far) > 0.99)
    del raw_sim_3, raw_sim_4

    # make sure it works with EEG-only and MEG-only
    raw_sim_meg = simulate_raw(raw.pick_types(meg=True, eeg=False, copy=True),
                               stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_eeg = simulate_raw(raw.pick_types(meg=False, eeg=True, copy=True),
                               stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_meeg = simulate_raw(raw.pick_types(meg=True, eeg=True, copy=True),
                                stc, trans, src, sphere, cov=None,
                                ecg=True, blink=True, random_state=seed)
    assert_allclose(np.concatenate((raw_sim_meg[:][0], raw_sim_eeg[:][0])),
                    raw_sim_meeg[:][0], rtol=1e-7, atol=1e-20)
    del raw_sim_meg, raw_sim_eeg, raw_sim_meeg

    # Make impossible transform (translate up into helmet) and ensure failure
    head_pos_sim_err = deepcopy(head_pos_sim)
    head_pos_sim_err[0.][2, 3] -= 0.1  # z trans upward 10cm
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src, sphere,
                  ecg=False, blink=False, head_pos=head_pos_sim_err)
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src,
                  bem_fname, ecg=False, blink=False,
                  head_pos=head_pos_sim_err)


@testing.requires_testing_data
def test_simulate_raw_bem():
    """Test simulation of raw data with BEM"""
    seed = 42
    raw, src, stc, trans, sphere = _get_data()
    raw = Raw(raw_fname).crop(0., 5.0).preload_data()
    raw_sim_sph = simulate_raw(raw, stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_bem = simulate_raw(raw, stc, trans, src, bem_fname, cov=None,
                               ecg=True, blink=True, random_state=seed,
                               n_jobs=2)
    # some radial components might not match that well, so just make sure that
    # most components have high correlation
    assert_array_equal(raw_sim_sph.ch_names, raw_sim_bem.ch_names)
    picks = pick_types(raw.info, meg=True, eeg=True)
    n_ch = len(picks)
    corr = np.corrcoef(raw_sim_sph[picks][0], raw_sim_bem[picks][0])
    assert_array_equal(corr.shape, (2 * n_ch, 2 * n_ch))
    assert_true(np.median(np.diag(corr[:n_ch, -n_ch:])) > 0.9)

run_tests_if_main()
