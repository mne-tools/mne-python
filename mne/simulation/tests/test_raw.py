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
                 make_sphere_model, create_info, setup_volume_source_space)
from mne.chpi import (_calculate_chpi_positions, read_head_pos,
                      _get_hpi_info, head_pos_to_trans_rot_t)
from mne.tests.test_chpi import _compare_positions
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.io import Raw, RawArray
from mne.time_frequency import psd_welch
from mne.utils import _TempDir, run_tests_if_main, requires_version, slow_test
from mne.fixes import isclose


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

raw_chpi_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_subsampled.pos')


def _make_stc(raw, src):
    """Helper to make a STC"""
    seed = 42
    sfreq = raw.info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = len(raw.times) // 10
    times = np.arange(0, n_samples) * tstep
    stc = simulate_sparse_stc(src, 10, times, random_state=seed)
    return stc


def _get_data():
    """Helper to get some starting data"""
    # raw with ECG channel
    raw = Raw(raw_fname).crop(0., 5.0, copy=False).load_data()
    data_picks = pick_types(raw.info, meg=True, eeg=True)
    other_picks = pick_types(raw.info, meg=False, stim=True, eog=True)
    picks = np.sort(np.concatenate((data_picks[::16], other_picks)))
    raw = raw.pick_channels([raw.ch_names[p] for p in picks])
    ecg = RawArray(np.zeros((1, len(raw.times))),
                   create_info(['ECG 063'], raw.info['sfreq'], 'ecg'))
    for key in ('dev_head_t', 'buffer_size_sec', 'highpass', 'lowpass',
                'filename', 'dig'):
        ecg.info[key] = raw.info[key]
    raw.add_channels([ecg])

    src = read_source_spaces(src_fname)
    trans = read_trans(trans_fname)
    sphere = make_sphere_model('auto', 'auto', raw.info)
    stc = _make_stc(raw, src)
    return raw, src, stc, trans, sphere


@testing.requires_testing_data
def test_simulate_raw_sphere():
    """Test simulation of raw data with sphere model"""
    seed = 42
    raw, src, stc, trans, sphere = _get_data()
    assert_true(len(pick_types(raw.info, meg=False, ecg=True)) == 1)

    # head pos
    head_pos_sim = dict()
    # these will be at 1., 2., ... sec
    shifts = [[0.001, 0., -0.001], [-0.001, 0.001, 0.]]

    for time_key, shift in enumerate(shifts):
        # Create 4x4 matrix transform and normalize
        temp_trans = deepcopy(raw.info['dev_head_t'])
        temp_trans['trans'][:3, 3] += shift
        head_pos_sim[time_key + 1.] = temp_trans['trans']

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

    raw_sim_loaded = Raw(test_outname, preload=True, proj=False)
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
        close = isclose(raw_sim_3[these_picks][0],
                        raw_sim_4[these_picks][0], atol=1e-20)
        assert_true(np.mean(close) > 0.7)
        far = ~isclose(raw_sim_3[diff_picks][0],
                       raw_sim_4[diff_picks][0], atol=1e-20)
        assert_true(np.mean(far) > 0.99)
    del raw_sim_3, raw_sim_4

    # make sure it works with EEG-only and MEG-only
    raw_sim_meg = simulate_raw(raw.copy().pick_types(meg=True, eeg=False),
                               stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_eeg = simulate_raw(raw.copy().pick_types(meg=False, eeg=True),
                               stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_meeg = simulate_raw(raw.copy().pick_types(meg=True, eeg=True),
                                stc, trans, src, sphere, cov=None,
                                ecg=True, blink=True, random_state=seed)
    assert_allclose(np.concatenate((raw_sim_meg[:][0], raw_sim_eeg[:][0])),
                    raw_sim_meeg[:][0], rtol=1e-7, atol=1e-20)
    del raw_sim_meg, raw_sim_eeg, raw_sim_meeg

    # check that different interpolations are similar given small movements
    raw_sim_cos = simulate_raw(raw, stc, trans, src, sphere,
                               head_pos=head_pos_sim,
                               random_state=seed)
    raw_sim_lin = simulate_raw(raw, stc, trans, src, sphere,
                               head_pos=head_pos_sim, interp='linear',
                               random_state=seed)
    assert_allclose(raw_sim_cos[:][0], raw_sim_lin[:][0],
                    rtol=1e-5, atol=1e-20)
    del raw_sim_cos, raw_sim_lin

    # Make impossible transform (translate up into helmet) and ensure failure
    head_pos_sim_err = deepcopy(head_pos_sim)
    head_pos_sim_err[1.][2, 3] -= 0.1  # z trans upward 10cm
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src, sphere,
                  ecg=False, blink=False, head_pos=head_pos_sim_err)
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src,
                  bem_fname, ecg=False, blink=False,
                  head_pos=head_pos_sim_err)
    # other degenerate conditions
    assert_raises(TypeError, simulate_raw, 'foo', stc, trans, src, sphere)
    assert_raises(TypeError, simulate_raw, raw, 'foo', trans, src, sphere)
    assert_raises(ValueError, simulate_raw, raw, stc.copy().crop(0, 0),
                  trans, src, sphere)
    stc_bad = stc.copy()
    stc_bad.tstep += 0.1
    assert_raises(ValueError, simulate_raw, raw, stc_bad, trans, src, sphere)
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src, sphere,
                  chpi=True)  # no cHPI info
    assert_raises(ValueError, simulate_raw, raw, stc, trans, src, sphere,
                  interp='foo')
    assert_raises(TypeError, simulate_raw, raw, stc, trans, src, sphere,
                  head_pos=1.)
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src, sphere,
                  head_pos=pos_fname)  # ends up with t>t_end
    head_pos_sim_err = deepcopy(head_pos_sim)
    head_pos_sim_err[-1.] = head_pos_sim_err[1.]  # negative time
    assert_raises(RuntimeError, simulate_raw, raw, stc, trans, src, sphere,
                  head_pos=head_pos_sim_err)
    raw_bad = raw.copy()
    raw_bad.info['dig'] = None
    assert_raises(RuntimeError, simulate_raw, raw_bad, stc, trans, src, sphere,
                  blink=True)


@testing.requires_testing_data
def test_simulate_raw_bem():
    """Test simulation of raw data with BEM"""
    seed = 42
    raw, src, stc, trans, sphere = _get_data()
    raw_sim_sph = simulate_raw(raw, stc, trans, src, sphere, cov=None,
                               ecg=True, blink=True, random_state=seed)
    raw_sim_bem = simulate_raw(raw, stc, trans, src, bem_fname, cov=None,
                               ecg=True, blink=True, random_state=seed,
                               n_jobs=2)
    # some components (especially radial) might not match that well,
    # so just make sure that most components have high correlation
    assert_array_equal(raw_sim_sph.ch_names, raw_sim_bem.ch_names)
    picks = pick_types(raw.info, meg=True, eeg=True)
    n_ch = len(picks)
    corr = np.corrcoef(raw_sim_sph[picks][0], raw_sim_bem[picks][0])
    assert_array_equal(corr.shape, (2 * n_ch, 2 * n_ch))
    assert_true(np.median(np.diag(corr[:n_ch, -n_ch:])) > 0.9)


@slow_test
@requires_version('numpy', '1.7')
@requires_version('scipy', '0.12')
@testing.requires_testing_data
def test_simulate_raw_chpi():
    """Test simulation of raw data with cHPI"""
    raw = Raw(raw_chpi_fname, allow_maxshield='yes')
    sphere = make_sphere_model('auto', 'auto', raw.info)
    # make sparse spherical source space
    sphere_vol = tuple(sphere['r0'] * 1000.) + (sphere.radius * 1000.,)
    src = setup_volume_source_space('sample', sphere=sphere_vol, pos=70.)
    stc = _make_stc(raw, src)
    # simulate data with cHPI on
    raw_sim = simulate_raw(raw, stc, None, src, sphere, cov=None, chpi=False)
    # need to trim extra samples off this one
    raw_chpi = simulate_raw(raw, stc, None, src, sphere, cov=None, chpi=True,
                            head_pos=pos_fname)
    # test cHPI indication
    hpi_freqs, _, hpi_pick, hpi_ons = _get_hpi_info(raw.info)[:4]
    assert_allclose(raw_sim[hpi_pick][0], 0.)
    assert_allclose(raw_chpi[hpi_pick][0], hpi_ons.sum())
    # test that the cHPI signals make some reasonable values
    picks_meg = pick_types(raw.info, meg=True, eeg=False)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)

    for picks in [picks_meg, picks_eeg]:
        psd_sim, freqs_sim = psd_welch(raw_sim, picks=picks)
        psd_chpi, freqs_chpi = psd_welch(raw_chpi, picks=picks)

        assert_array_equal(freqs_sim, freqs_chpi)
        freq_idx = np.sort([np.argmin(np.abs(freqs_sim - f))
                           for f in hpi_freqs])
        if picks is picks_meg:
            assert_true((psd_chpi[:, freq_idx] >
                         100 * psd_sim[:, freq_idx]).all())
        else:
            assert_allclose(psd_sim, psd_chpi, atol=1e-20)

    # test localization based on cHPI information
    quats_sim = _calculate_chpi_positions(raw_chpi)
    trans_sim, rot_sim, t_sim = head_pos_to_trans_rot_t(quats_sim)
    trans, rot, t = head_pos_to_trans_rot_t(read_head_pos(pos_fname))
    t -= raw.first_samp / raw.info['sfreq']
    _compare_positions((trans, rot, t), (trans_sim, rot_sim, t_sim),
                       max_dist=0.005)

run_tests_if_main()
