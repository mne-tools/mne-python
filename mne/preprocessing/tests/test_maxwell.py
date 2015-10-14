# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from nose.tools import assert_true, assert_raises

from mne import compute_raw_covariance, pick_types
from mne.forward import _prep_meg_channels
from mne.cov import _estimate_rank_meeg_cov
from mne.datasets import testing
from mne.io import Raw, proc_history
from mne.preprocessing.maxwell import (maxwell_filter, _get_n_moments,
                                       _sss_basis, _sh_complex_to_real,
                                       _sh_real_to_complex, _sh_negate,
                                       _bases_complex_to_real,
                                       _bases_real_to_complex, _sph_harm)
from mne.utils import _TempDir, run_tests_if_main, slow_test

# Note: Elekta MaxFilter uses single precision, so expect filtered results to
# differ slightly.

warnings.simplefilter('always')  # Always throw warnings

sss_path = op.join(testing.data_path(download=False), 'SSS')
raw_fname = op.join(sss_path, 'test_move_anon_raw.fif')
sss_std_fname = op.join(sss_path, 'test_move_anon_stdOrigin_raw_sss.fif')
bases_fname = op.join(sss_path, 'sss_data.mat')
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


def _assert_snr(actual, desired, min_tol, med_tol=500.):
    """Helper to assert SNR of a certain level"""
    picks = pick_types(desired.info, meg=True, exclude=[])
    others = np.setdiff1d(np.arange(len(actual.ch_names)), picks)
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(actual[others][0], desired[others][0])
    actual_data = actual[picks][0]
    desired_data = desired[picks][0]
    bench_rms = np.sqrt(np.mean(desired_data * desired_data, axis=1))
    error = actual_data - desired_data
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    snrs = bench_rms / error_rms
    # min tol
    snr = snrs.min()
    bad_count = (snrs < min_tol).sum()
    assert_true(bad_count == 0, 'SNR (worst %0.1f) < %0.1f for %s/%s channels'
                % (snr, min_tol, bad_count, len(picks)))
    # median tol
    snr = np.median(snrs)
    assert_true(snr >= med_tol, 'SNR median %0.1f < %0.1f' % (snr, med_tol))


@testing.requires_testing_data
def test_spherical_harmonics():
    """Test spherical harmonic basis functions"""
    from scipy.io import loadmat
    # Test our real<->complex conversion functions
    az, pol = np.meshgrid(np.linspace(0, 2 * np.pi, 30),
                          np.linspace(0, np.pi, 20), indexing='ij')
    for deg in range(1, int_order):
        for order in range(0, deg + 1):
            sph = _sph_harm(order, deg, az, pol)
            # ensure that we satisfy the conjugation property
            assert_allclose(_sh_negate(sph, order),
                            _sph_harm(-order, deg, az, pol))
            # ensure our conversion functions work
            sph_real_pos = _sh_complex_to_real(sph, order)
            sph_real_neg = _sh_complex_to_real(sph, -order)
            sph_2 = _sh_real_to_complex([sph_real_pos, sph_real_neg], order)
            assert_allclose(sph, sph_2, atol=1e-7)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True)
    coils = _prep_meg_channels(raw.info, accurate=True, elekta_defs=True,
                               verbose=False)[0]
    S_tot = _sss_basis(np.array([0., 0., 40e-3]), coils, int_order, ext_order)
    # Test our real<->complex conversion functions
    S_tot_complex = _bases_real_to_complex(S_tot, int_order, ext_order)
    S_tot_round = _bases_complex_to_real(S_tot_complex, int_order, ext_order)
    assert_allclose(S_tot, S_tot_round, atol=1e-7)
    # Now normalize our columns
    S_tot /= np.sqrt(np.sum(S_tot * S_tot, axis=0))[np.newaxis]
    S_tot_complex /= np.sqrt(np.sum(
        (S_tot_complex * S_tot_complex.conj()).real, axis=0))[np.newaxis]
    # Now check against a known benchmark
    sss_data = loadmat(bases_fname)
    S_tot_mat = np.concatenate([sss_data['SNin0040'], sss_data['SNout0040']],
                               axis=1)
    # Check this roundtrip
    S_tot_mat_real = _bases_complex_to_real(S_tot_mat, int_order, ext_order)
    S_tot_mat_round = _bases_real_to_complex(S_tot_mat_real,
                                             int_order, ext_order)
    assert_allclose(S_tot_mat, S_tot_mat_round, atol=1e-7)
    # XXX These should really be better...
    assert_allclose(S_tot_complex, S_tot_mat, rtol=1e0, atol=1e0)
    assert_allclose(S_tot, S_tot_mat_real, rtol=1e0, atol=1e0)


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
    _assert_snr(raw_sss, sss_std, 200.)

    # Test SSS computation at non-standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 20., 20.],
                             int_order=int_order, ext_order=ext_order)
    _assert_snr(raw_sss, sss_nonStd, 250.)

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
    _assert_snr(raw_sss, sss_bench, 300.)


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
    tols = [325., 200.]
    for st_dur, tol in zip(st_durs, tols):
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
        _assert_snr(raw_tsss, tsss_bench, tol)

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
    _assert_snr(raw_sss, sss_fine_cal, 1.5, 25.)  # XXX should be much higher

    # Test 3D SSS fine calibration (no equivalent func in MaxFilter yet!)
    # very low SNR as proc differs, eventually we should add a better test
    raw_sss_3D = maxwell_filter(raw, fine_cal=fine_cal_fname_3d, verbose=True)
    _assert_snr(raw_sss_3D, sss_fine_cal, 0.75, 5.)


@testing.requires_testing_data
def test_maxwell_filter_cross_talk():
    """Test cross-talk cancellation feature of Maxwell filter"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    raw.info['bads'] = bads
    sss_ctc = Raw(sss_ctc_fname)
    raw_sss = maxwell_filter(raw, ctc=ctc_fname)
    _assert_snr(raw_sss, sss_ctc, 275.)


# TODO: Eventually add simulation tests mirroring Taulu's original paper
# that calculates the localization error and shielding factor:
# http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1495874

# @testing.requires_testing_data
# def test_maxwell_noise_rejection():

run_tests_if_main()
