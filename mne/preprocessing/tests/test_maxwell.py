# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from nose.tools import assert_true, assert_raises

from mne import compute_raw_covariance, pick_types
from mne.cov import _estimate_rank_meeg_cov
from mne.datasets import testing
from mne.io import Raw, proc_history
from mne.preprocessing.maxwell import maxwell_filter, _get_n_moments
from mne.utils import _TempDir, run_tests_if_main, slow_test

# Note: Elekta MaxFilter uses single precision, so expect filtered results to
# differ slightly.

warnings.simplefilter('always')  # Always throw warnings

sss_path = op.join(testing.data_path(download=False), 'SSS')
raw_fname = op.join(sss_path, 'test_move_anon_raw.fif')
sss_std_fname = op.join(sss_path, 'test_move_anon_stdOrigin_raw_sss.fif')
sss_nonstd_fname = op.join(sss_path, 'test_move_anon_nonStdOrigin_raw_sss.fif')
sss_bad_recon_fname = op.join(sss_path, 'test_move_anon_badRecon_raw_sss.fif')
sss_fine_cal_fname = op.join(sss_path, 'test_move_anon_fineCal_raw_sss.fif')
sss_ctc_fname = op.join(sss_path, 'test_move_anon_crossTalk_raw_sss.fif')
fine_cal_fname = op.join(sss_path, 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(sss_path, 'sss_cal_3053_3d.dat')
ctc_fname = op.join(sss_path, 'ct_sparse.fif')

int_order, ext_order = 8, 3

# 30 random bad MEG channels (20 grad, 10 mag) that were used in generation
bads = ['MEG0912', 'MEG1722', 'MEG2213', 'MEG0132', 'MEG1312', 'MEG0432',
        'MEG2433', 'MEG1022', 'MEG0442', 'MEG2332', 'MEG0633', 'MEG1043',
        'MEG1713', 'MEG0422', 'MEG0932', 'MEG1622', 'MEG1343', 'MEG0943',
        'MEG0643', 'MEG0143', 'MEG2142', 'MEG0813', 'MEG2143', 'MEG1323',
        'MEG0522', 'MEG1123', 'MEG0423', 'MEG2122', 'MEG2532', 'MEG0812']


def _assert_snr(actual, desired, snr_tol=1000.):
    """Helper to assert SNR of a certain level"""
    picks = pick_types(desired.info, meg=True, exclude=[])
    others = np.setdiff1d(np.arange(len(actual.ch_names)), picks)
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(actual[others][0], desired[others][0])
    actual = actual[picks][0]
    desired = desired[picks][0]
    bench_rms = np.sqrt(np.mean(desired * desired, axis=1))
    error = actual - desired
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    snr = np.mean(bench_rms / error_rms)
    assert_true(snr >= snr_tol, 'SNR (%0.1f) < %0.1f' % (snr, snr_tol))


@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter"""
    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
        raw_err = Raw(raw_fname, proj=True,
                      allow_maxshield=True).crop(0., 0.1, False)
    sss_std = Raw(sss_std_fname, allow_maxshield=True)
    sss_nonStd = Raw(sss_nonstd_fname, allow_maxshield=True)
    assert_raises(RuntimeError, maxwell_filter, raw_err)

    n_int_bases = int_order ** 2 + 2 * int_order
    n_ext_bases = ext_order ** 2 + 2 * ext_order
    nbases = n_int_bases + n_ext_bases

    # Check number of bases computed correctly
    assert_equal(_get_n_moments([int_order, ext_order]).sum(), nbases)

    # Test SSS computation at the standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 0., 40.],
                             int_order=int_order, ext_order=ext_order)
    _assert_snr(raw_sss, sss_std)

    # Test SSS computation at non-standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 20., 20.],
                             int_order=int_order, ext_order=ext_order)
    _assert_snr(raw_sss, sss_nonStd)

    # Check against SSS functions from proc_history
    sss_info = raw_sss.info['proc_history'][0]['max_info']
    assert_equal(_get_n_moments(int_order),
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
    assert_equal(cov_sss_rank, _get_n_moments(int_order))


@slow_test
@testing.requires_testing_data
def test_bads_reconstruction():
    """Test reconstruction of channels marked as bad"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    sss_bench = Raw(sss_bad_recon_fname, allow_maxshield=True)
    raw.info['bads'] = bads
    raw_sss = maxwell_filter(raw)
    _assert_snr(raw_sss, sss_bench)


@testing.requires_testing_data
def test_spatiotemporal_maxwell():
    """Test spatiotemporal (tSSS) processing"""
    # Load raw testing data
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True)

    # Test that window is less than length of data
    assert_raises(ValueError, maxwell_filter, raw, st_dur=1000.)

    # Check both 4 and 10 seconds because Elekta handles them differently
    # This is to ensure that std/non-std tSSS windows are correctly handled
    st_durs = [4., 10.]
    for st_dur in st_durs:
        # Load tSSS data depending on st_dur and get data
        tSSS_fname = op.join(sss_path,
                             'test_move_anon_st%0ds_raw_sss.fif' % st_dur)
        tsss_bench = Raw(tSSS_fname, allow_maxshield=True)
        # Because Elekta's tSSS sometimes(!) lumps the tail window of data
        # onto the previous buffer if it's shorter than st_dur, we have to
        # crop the data here to compensate for Elekta's tSSS behavior.
        if st_dur == 10.:
            tsss_bench.crop(0, st_dur, copy=False)

        # Test sss computation at the standard head origin. Same cropping issue
        # as mentioned above.
        if st_dur == 10.:
            raw_tsss = maxwell_filter(raw.crop(0, st_dur), st_dur=st_dur)
        else:
            raw_tsss = maxwell_filter(raw, st_dur=st_dur)

        # Confirm SNR > 500, single precision is part of discrepancy
        _assert_snr(raw_tsss, tsss_bench, 500.)

    # Degenerate cases
    assert_raises(ValueError, maxwell_filter, raw, st_dur=10., st_corr=0.)


@testing.requires_testing_data
def test_maxwell_filter_fine_calibration():
    """Test fine calibration feature of Maxwell filter"""

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    sss_fine_cal = Raw(sss_fine_cal_fname, proj=False,
                       allow_maxshield=True)

    # Test 1D SSS fine calibration
    raw_sss = maxwell_filter(raw, fine_cal=fine_cal_fname)
    _assert_snr(raw_sss, sss_fine_cal, 30.)  # XXX should be higher

    # Test 3D SSS fine calibration
    raw_sss = maxwell_filter(raw, fine_cal=fine_cal_fname_3d, verbose=True)
    _assert_snr(raw_sss, sss_fine_cal, 10.)  # XXX should be higher


@testing.requires_testing_data
def test_maxwell_filter_cross_talk():
    """Test cross-talk cancellation feature of Maxwell filter"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    raw.info['bads'] = bads
    sss_ctc = Raw(sss_ctc_fname)
    raw_sss = maxwell_filter(raw, ctc=ctc_fname)
    _assert_snr(raw_sss, sss_ctc)


# TODO: Eventually add simulation tests mirroring Taulu's original paper
# that calculates the localization error and shielding factor:
# http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1495874

# @testing.requires_testing_data
# def test_maxwell_noise_rejection():

run_tests_if_main()
