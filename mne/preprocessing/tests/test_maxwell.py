# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal)
from nose.tools import assert_true, assert_raises

from mne import compute_raw_covariance, pick_types
from mne.cov import _estimate_rank_meeg_cov
from mne.datasets import testing
from mne.forward._make_forward import _prep_meg_channels
from mne.io import Raw, proc_history
from mne.preprocessing.maxwell import (_maxwell_filter as maxwell_filter,
                                       get_num_moments, _sss_basis)
from mne.utils import _TempDir, run_tests_if_main, slow_test

warnings.simplefilter('always')  # Always throw warnings

data_path = op.join(testing.data_path(download=False))
raw_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
sss_std_fname = op.join(data_path, 'SSS',
                        'test_move_anon_raw_simp_stdOrigin_sss.fif')
sss_nonstd_fname = op.join(data_path, 'SSS',
                           'test_move_anon_raw_simp_nonStdOrigin_sss.fif')
sss_bad_recon_fname = op.join(data_path, 'SSS',
                              'test_move_anon_raw_bad_recon_sss.fif')
sss_fine_cal_fname = op.join(data_path, 'SSS',
                             'test_move_anon_raw_fineCal_sss.fif')

fine_cal_fname = op.join(data_path, 'SSS', 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(data_path, 'SSS', 'sss_cal_3053_3d.dat')

int_order, ext_order = 8, 3


@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter

    Notes
    -----
    Elekta MaxFilter uses single precision, so expect filtered results to
    differ slightly.
    """

    # TODO: Future tests integrate with mne/io/tests/test_proc_history

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    raw.load_data()
    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_std = Raw(sss_std_fname, allow_maxshield=True)
        sss_nonStd = Raw(sss_nonstd_fname, allow_maxshield=True)
        raw_err = Raw(raw_fname, proj=True,
                      allow_maxshield=True).crop(0., 0.1, False)
    assert_raises(RuntimeError, maxwell_filter, raw_err)

    # Create coils
    all_coils, _, _, meg_info = _prep_meg_channels(raw.info, ignore_ref=True,
                                                   elekta_defs=True)
    picks = [raw.info['ch_names'].index(ch) for ch in [coil['chname']
                                                       for coil in all_coils]]
    coils = [all_coils[ci] for ci in picks]
    ncoils = len(coils)

    int_order, ext_order = 8, 3
    n_int_bases = int_order ** 2 + 2 * int_order
    n_ext_bases = ext_order ** 2 + 2 * ext_order
    nbases = n_int_bases + n_ext_bases

    # Check number of bases computed correctly
    assert_equal(get_num_moments(int_order, ext_order), nbases)

    # Check multipolar moment basis set
    S_in, S_out = _sss_basis(origin=np.array([0, 0, 40]), coils=coils,
                             int_order=int_order, ext_order=ext_order)
    assert_equal(S_in.shape, (ncoils, n_int_bases), 'S_in has incorrect shape')
    assert_equal(S_out.shape, (ncoils, n_ext_bases),
                 'S_out has incorrect shape')

    # Test sss computation at the standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 0., 40.],
                             int_order=int_order, ext_order=ext_order)

    sss_std_data = sss_std[picks][0]
    assert_array_almost_equal(raw_sss[picks][0], sss_std_data,
                              decimal=11, err_msg='Maxwell filtered data at '
                              'standard origin incorrect.')

    # Confirm SNR is above 1000
    bench_rms = np.sqrt(np.mean(sss_std_data * sss_std_data, axis=1))
    error = raw_sss[picks][0] - sss_std_data
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 1000, 'SNR < 1000')

    # Test SSS computation at non-standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 20., 20.],
                             int_order=int_order, ext_order=ext_order)
    sss_nonStd_data = sss_nonStd[picks][0]
    assert_array_almost_equal(raw_sss[picks][0], sss_nonStd_data, decimal=11,
                              err_msg='Maxwell filtered data at non-std '
                              'origin incorrect.')

    # Confirm SNR is above 1000
    bench_rms = np.sqrt(np.mean(sss_nonStd_data * sss_nonStd_data, axis=1))
    error = raw_sss[picks][0] - sss_nonStd_data
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 1000, 'SNR < 1000')

    # Check against SSS functions from proc_history
    sss_info = raw_sss.info['proc_history'][0]['max_info']
    assert_equal(get_num_moments(int_order, 0),
                 proc_history._get_sss_rank(sss_info))

    # Degenerate cases
    raw_bad = raw.copy()
    raw_bad.info['comps'] = [0]
    assert_raises(RuntimeError, maxwell_filter, raw_bad)


@testing.requires_testing_data
def test_maxwell_filter_additional():
    """Test processing of Maxwell filtered data"""

    # TODO: Future tests integrate with mne/io/tests/test_proc_history

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    data_path = op.join(testing.data_path(download=False))

    file_name = 'test_move_anon'

    raw_fname = op.join(data_path, 'SSS', file_name + '_raw.fif')

    with warnings.catch_warnings(record=True):  # maxshield
        # Use 2.0 seconds of data to get stable cov. estimate
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 2., False)

    # Get MEG channels, compute Maxwell filtered data
    raw.load_data()
    raw.pick_types(meg=True, eeg=False)
    int_order, ext_order = 8, 3
    raw_sss = maxwell_filter(raw, int_order=int_order, ext_order=ext_order)

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sss.fif')
    raw_sss.save(test_outname)
    raw_sss_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)

    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sss_loaded[:][0], raw_sss[:][0],
                    rtol=1e-6, atol=1e-20)

    # Test rank of covariance matrices for raw and SSS processed data
    cov_raw = compute_raw_covariance(raw)
    cov_sss = compute_raw_covariance(raw_sss)

    scalings = None
    cov_raw_rank = _estimate_rank_meeg_cov(cov_raw['data'], raw.info, scalings)
    cov_sss_rank = _estimate_rank_meeg_cov(cov_sss['data'], raw_sss.info,
                                           scalings)

    assert_equal(cov_raw_rank, raw.info['nchan'])
    assert_equal(cov_sss_rank, get_num_moments(int_order, 0))


@slow_test
@testing.requires_testing_data
def test_bads_reconstruction():
    """Test reconstruction of channels marked as bad"""

    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_bench = Raw(sss_bad_recon_fname, allow_maxshield=True)

    raw_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')

    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)

    # Set 30 random bad MEG channels (20 grad, 10 mag)
    bads = ['MEG0912', 'MEG1722', 'MEG2213', 'MEG0132', 'MEG1312', 'MEG0432',
            'MEG2433', 'MEG1022', 'MEG0442', 'MEG2332', 'MEG0633', 'MEG1043',
            'MEG1713', 'MEG0422', 'MEG0932', 'MEG1622', 'MEG1343', 'MEG0943',
            'MEG0643', 'MEG0143', 'MEG2142', 'MEG0813', 'MEG2143', 'MEG1323',
            'MEG0522', 'MEG1123', 'MEG0423', 'MEG2122', 'MEG2532', 'MEG0812']
    raw.info['bads'] = bads

    # Compute Maxwell filtered data
    raw_sss = maxwell_filter(raw)
    meg_chs = pick_types(raw_sss.info)
    non_meg_chs = np.setdiff1d(np.arange(len(raw.ch_names)), meg_chs)
    sss_bench_data = sss_bench[meg_chs][0]

    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sss[meg_chs][0], sss_bench_data,
                    rtol=1e-12, atol=1e-4, err_msg='Maxwell filtered data '
                    'with reconstructed bads is incorrect.')

    # Confirm SNR is above 1000
    bench_rms = np.sqrt(np.mean(raw_sss[meg_chs][0] ** 2, axis=1))
    error = raw_sss[meg_chs][0] - sss_bench_data
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 1000,
                'SNR (%0.1f) < 1000' % np.mean(bench_rms / error_rms))
    assert_allclose(raw_sss[non_meg_chs][0], raw[non_meg_chs][0])


@testing.requires_testing_data
def test_spatiotemporal_maxwell():
    """Test spatiotemporal (tSSS) processing"""
    # Load raw testing data
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True)

    # Create coils
    picks = pick_types(raw.info)

    # Test that window is less than length of data
    assert_raises(ValueError, maxwell_filter, raw, st_dur=1000.)

    # Check both 4 and 10 seconds because Elekta handles them differently
    # This is to ensure that std/non-std tSSS windows are correctly handled
    st_durs = [4., 10.]
    for st_dur in st_durs:
        # Load tSSS data depending on st_dur and get data
        tSSS_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_' +
                             'spatiotemporal_%0ds_sss.fif' % st_dur)

        with warnings.catch_warnings(record=True):  # maxshield, naming
            tsss_bench = Raw(tSSS_fname, allow_maxshield=True)
            # Because Elekta's tSSS sometimes(!) lumps the tail window of data
            # onto the previous buffer if it's shorter than st_dur, we have to
            # crop the data here to compensate for Elekta's tSSS behavior.
            if st_dur == 10.:
                tsss_bench.crop(0, st_dur, copy=False)
        tsss_bench_data = tsss_bench[picks, :][0]
        del tsss_bench

        # Test sss computation at the standard head origin. Same cropping issue
        # as mentioned above.
        if st_dur == 10.:
            raw_tsss = maxwell_filter(raw.crop(0, st_dur), st_dur=st_dur)
        else:
            raw_tsss = maxwell_filter(raw, st_dur=st_dur)
        assert_allclose(raw_tsss[picks][0], tsss_bench_data,
                        rtol=1e-12, atol=1e-4, err_msg='Spatiotemporal (tSSS) '
                        'maxwell filtered data at standard origin incorrect.')

        # Confirm SNR is above 500. Single precision is part of discrepancy
        bench_rms = np.sqrt(np.mean(tsss_bench_data * tsss_bench_data, axis=1))
        error = raw_tsss[picks][0] - tsss_bench_data
        error_rms = np.sqrt(np.mean(error * error, axis=1))
        assert_true(np.mean(bench_rms / error_rms) >= 500,
                    'SNR (%0.1f) < 500' % np.mean(bench_rms / error_rms))

    # Confirm we didn't modify other channels (like EEG chs)
    non_picks = np.setdiff1d(np.arange(len(raw.ch_names)), picks)
    assert_allclose(raw[non_picks, 0:raw_tsss.n_times][0],
                    raw_tsss[non_picks, 0:raw_tsss.n_times][0])

    # Degenerate cases
    assert_raises(ValueError, maxwell_filter, raw, st_dur=10., st_corr=0.)


@testing.requires_testing_data
def test_maxwell_filter_fine_calibration():
    """Test fine calibration feature of Maxwell filter

    Notes
    -----
    Elekta MaxFilter uses single precision, so expect filtered results to
    differ slightly.
    """

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    raw.preload_data()
    raw.pick_types(meg=True, eeg=False)
    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_fine_cal = Raw(sss_fine_cal_fname, preload=True, proj=False,
                           allow_maxshield=True)

    # Create coils
    all_coils, _, _, meg_info = _prep_meg_channels(raw.info, ignore_ref=True,
                                                   elekta_defs=True)
    picks = [raw.info['ch_names'].index(ch) for ch in [coil['chname']
                                                       for coil in all_coils]]

    # Test 1D SSS fine calibration
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 0., 40.],
                                     int_order=int_order, ext_order=ext_order,
                                     fine_cal_fname=fine_cal_fname)
    assert_array_almost_equal(raw_sss._data[picks, :],
                              sss_fine_cal._data[picks, :], decimal=11,
                              err_msg='Maxwell filtered data with fine '
                              'calibration incorrect.')

    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_fine_cal._data[picks, :] ** 2, axis=1))
    error = raw_sss._data[picks, :] - sss_fine_cal._data[picks, :]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 100, 'SNR < 100')

    # Test 3D SSS fine calibration
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 0., 40.],
                                     int_order=int_order, ext_order=ext_order,
                                     fine_cal_fname=fine_cal_fname_3d)
    assert_array_almost_equal(raw_sss._data[picks, :],
                              sss_fine_cal._data[picks, :], decimal=11,
                              err_msg='Maxwell filtered data with fine '
                              'calibration incorrect.')

    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_fine_cal._data[picks, :] ** 2, axis=1))
    error = raw_sss._data[picks, :] - sss_fine_cal._data[picks, :]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 100, 'SNR < 100')


# TODO: Eventually add simulation tests mirroring Taulu's original papers
#@testing.requires_testing_data
#def test_maxwell_noise_rejection():

run_tests_if_main()
