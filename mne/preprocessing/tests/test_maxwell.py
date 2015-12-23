# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
import sys
import scipy
from numpy.testing import assert_equal, assert_allclose
from nose.tools import assert_true, assert_raises
from nose.plugins.skip import SkipTest
from distutils.version import LooseVersion

from mne import compute_raw_covariance, pick_types
from mne.forward import _prep_meg_channels
from mne.cov import _estimate_rank_meeg_cov
from mne.datasets import testing
from mne.io import Raw, proc_history, read_info, read_raw_bti, read_raw_kit
from mne.preprocessing.maxwell import (maxwell_filter, _get_n_moments,
                                       _sss_basis_basic, _sh_complex_to_real,
                                       _sh_real_to_complex, _sh_negate,
                                       _bases_complex_to_real, _sss_basis,
                                       _bases_real_to_complex, _sph_harm,
                                       _get_coil_scale)
from mne.tests.common import assert_meg_snr
from mne.utils import (_TempDir, run_tests_if_main, slow_test, catch_logging,
                       requires_version, object_diff)
from mne.externals.six import PY3

warnings.simplefilter('always')  # Always throw warnings

data_path = testing.data_path(download=False)
sss_path = op.join(data_path, 'SSS')
pre = op.join(sss_path, 'test_move_anon_')
raw_fname = pre + 'raw.fif'
sss_std_fname = pre + 'stdOrigin_raw_sss.fif'
sss_nonstd_fname = pre + 'nonStdOrigin_raw_sss.fif'
sss_bad_recon_fname = pre + 'badRecon_raw_sss.fif'
sss_reg_in_fname = pre + 'regIn_raw_sss.fif'
sss_fine_cal_fname = pre + 'fineCal_raw_sss.fif'
sss_ctc_fname = pre + 'crossTalk_raw_sss.fif'
sss_trans_default_fname = pre + 'transDefault_raw_sss.fif'
sss_trans_sample_fname = pre + 'transSample_raw_sss.fif'
sss_st1FineCalCrossTalkRegIn_fname = \
    pre + 'st1FineCalCrossTalkRegIn_raw_sss.fif'
sss_st1FineCalCrossTalkRegInTransSample_fname = \
    pre + 'st1FineCalCrossTalkRegInTransSample_raw_sss.fif'

erm_fname = pre + 'erm_raw.fif'
sss_erm_std_fname = pre + 'erm_devOrigin_raw_sss.fif'
sss_erm_reg_in_fname = pre + 'erm_regIn_raw_sss.fif'
sss_erm_fine_cal_fname = pre + 'erm_fineCal_raw_sss.fif'
sss_erm_ctc_fname = pre + 'erm_crossTalk_raw_sss.fif'
sss_erm_st_fname = pre + 'erm_st1_raw_sss.fif'
sss_erm_st1FineCalCrossTalk_fname = pre + 'erm_st1FineCalCrossTalk_raw_sss.fif'
sss_erm_st1FineCalCrossTalkRegIn_fname = \
    pre + 'erm_st1FineCalCrossTalkRegIn_raw_sss.fif'

sample_fname = op.join(data_path, 'MEG', 'sample_audvis_trunc_raw.fif')
sss_samp_reg_in_fname = op.join(data_path, 'SSS',
                                'sample_audvis_trunc_regIn_raw_sss.fif')
sss_samp_fname = op.join(data_path, 'SSS', 'sample_audvis_trunc_raw_sss.fif')

bases_fname = op.join(sss_path, 'sss_data.mat')
fine_cal_fname = op.join(sss_path, 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(sss_path, 'sss_cal_3053_3d.dat')
ctc_fname = op.join(sss_path, 'ct_sparse.fif')
fine_cal_mgh_fname = op.join(sss_path, 'sss_cal_mgh.dat')
ctc_mgh_fname = op.join(sss_path, 'ct_sparse_mgh.fif')

sample_fname = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc_raw.fif')

int_order, ext_order = 8, 3
mf_head_origin = (0., 0., 0.04)
mf_meg_origin = (0., 0.013, -0.006)

# otherwise we can get SVD error
requires_svd_convergence = requires_version('scipy', '0.12')

# 30 random bad MEG channels (20 grad, 10 mag) that were used in generation
bads = ['MEG0912', 'MEG1722', 'MEG2213', 'MEG0132', 'MEG1312', 'MEG0432',
        'MEG2433', 'MEG1022', 'MEG0442', 'MEG2332', 'MEG0633', 'MEG1043',
        'MEG1713', 'MEG0422', 'MEG0932', 'MEG1622', 'MEG1343', 'MEG0943',
        'MEG0643', 'MEG0143', 'MEG2142', 'MEG0813', 'MEG2143', 'MEG1323',
        'MEG0522', 'MEG1123', 'MEG0423', 'MEG2122', 'MEG2532', 'MEG0812']


def _assert_n_free(raw_sss, lower, upper=None):
    """Helper to check the DOF"""
    upper = lower if upper is None else upper
    n_free = raw_sss.info['proc_history'][0]['max_info']['sss_info']['nfree']
    assert_true(lower <= n_free <= upper,
                'nfree fail: %s <= %s <= %s' % (lower, n_free, upper))


@slow_test
def test_other_systems():
    """Test Maxwell filtering on KIT, BTI, and CTF files
    """
    io_dir = op.join(op.dirname(__file__), '..', '..', 'io')

    # KIT
    kit_dir = op.join(io_dir, 'kit', 'tests', 'data')
    sqd_path = op.join(kit_dir, 'test.sqd')
    mrk_path = op.join(kit_dir, 'test_mrk.sqd')
    elp_path = op.join(kit_dir, 'test_elp.txt')
    hsp_path = op.join(kit_dir, 'test_hsp.txt')
    raw_kit = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
    assert_raises(RuntimeError, maxwell_filter, raw_kit)
    raw_sss = maxwell_filter(raw_kit, origin=(0., 0., 0.04), ignore_ref=True)
    _assert_n_free(raw_sss, 65)
    # XXX this KIT origin fit is terrible! Eventually we should get a
    # corrected HSP file with proper coverage
    with catch_logging() as log_file:
        assert_raises(RuntimeError, maxwell_filter, raw_kit,
                      ignore_ref=True, regularize=None)  # bad condition
        raw_sss = maxwell_filter(raw_kit, origin='auto',
                                 ignore_ref=True, bad_condition='warning',
                                 verbose='warning')
    log_file = log_file.getvalue()
    assert_true('badly conditioned' in log_file)
    assert_true('more than 20 mm from' in log_file)
    _assert_n_free(raw_sss, 28, 32)  # bad origin == brutal reg
    # Let's set the origin
    with catch_logging() as log_file:
        raw_sss = maxwell_filter(raw_kit, origin=(0., 0., 0.04),
                                 ignore_ref=True, bad_condition='warning',
                                 regularize=None, verbose='warning')
    log_file = log_file.getvalue()
    assert_true('badly conditioned' in log_file)
    _assert_n_free(raw_sss, 80)
    # Now with reg
    with catch_logging() as log_file:
        raw_sss = maxwell_filter(raw_kit, origin=(0., 0., 0.04),
                                 ignore_ref=True, verbose=True)
    log_file = log_file.getvalue()
    assert_true('badly conditioned' not in log_file)
    _assert_n_free(raw_sss, 65)

    # BTi
    bti_dir = op.join(io_dir, 'bti', 'tests', 'data')
    bti_pdf = op.join(bti_dir, 'test_pdf_linux')
    bti_config = op.join(bti_dir, 'test_config_linux')
    bti_hs = op.join(bti_dir, 'test_hs_linux')
    raw_bti = read_raw_bti(bti_pdf, bti_config, bti_hs, preload=False)
    raw_sss = maxwell_filter(raw_bti)
    _assert_n_free(raw_sss, 70)

    # CTF
    fname_ctf_raw = op.join(io_dir, 'tests', 'data', 'test_ctf_comp_raw.fif')
    raw_ctf = Raw(fname_ctf_raw, compensation=2)
    assert_raises(RuntimeError, maxwell_filter, raw_ctf)  # compensated
    raw_ctf = Raw(fname_ctf_raw)
    assert_raises(ValueError, maxwell_filter, raw_ctf)  # cannot fit headshape
    raw_sss = maxwell_filter(raw_ctf, origin=(0., 0., 0.04))
    _assert_n_free(raw_sss, 68)
    raw_sss = maxwell_filter(raw_ctf, origin=(0., 0., 0.04), ignore_ref=True)
    _assert_n_free(raw_sss, 70)


def test_spherical_harmonics():
    """Test spherical harmonic functions"""
    from scipy.special import sph_harm
    az, pol = np.meshgrid(np.linspace(0, 2 * np.pi, 30),
                          np.linspace(0, np.pi, 20))
    # As of Oct 16, 2015, Anancoda has a bug in scipy due to old compilers (?):
    # https://github.com/ContinuumIO/anaconda-issues/issues/479
    if (PY3 and
            LooseVersion(scipy.__version__) >= LooseVersion('0.15') and
            'Continuum Analytics' in sys.version):
        raise SkipTest('scipy sph_harm bad in Py3k on Anaconda')

    # Test our basic spherical harmonics
    for degree in range(1, int_order):
        for order in range(0, degree + 1):
            sph = _sph_harm(order, degree, az, pol)
            sph_scipy = sph_harm(order, degree, az, pol)
            assert_allclose(sph, sph_scipy, atol=1e-7)


def test_spherical_conversions():
    """Test spherical harmonic conversions"""
    # Test our real<->complex conversion functions
    az, pol = np.meshgrid(np.linspace(0, 2 * np.pi, 30),
                          np.linspace(0, np.pi, 20))
    for degree in range(1, int_order):
        for order in range(0, degree + 1):
            sph = _sph_harm(order, degree, az, pol)
            # ensure that we satisfy the conjugation property
            assert_allclose(_sh_negate(sph, order),
                            _sph_harm(-order, degree, az, pol))
            # ensure our conversion functions work
            sph_real_pos = _sh_complex_to_real(sph, order)
            sph_real_neg = _sh_complex_to_real(sph, -order)
            sph_2 = _sh_real_to_complex([sph_real_pos, sph_real_neg], order)
            assert_allclose(sph, sph_2, atol=1e-7)


@testing.requires_testing_data
def test_multipolar_bases():
    """Test multipolar moment basis calculation using sensor information"""
    from scipy.io import loadmat
    # Test our basis calculations
    info = read_info(raw_fname)
    coils = _prep_meg_channels(info, accurate=True, elekta_defs=True)[0]
    # Check against a known benchmark
    sss_data = loadmat(bases_fname)
    for origin in ((0, 0, 0.04), (0, 0.02, 0.02)):
        o_str = ''.join('%d' % (1000 * n) for n in origin)

        S_tot = _sss_basis_basic(origin, coils, int_order, ext_order,
                                 method='alternative')
        # Test our real<->complex conversion functions
        S_tot_complex = _bases_real_to_complex(S_tot, int_order, ext_order)
        S_tot_round = _bases_complex_to_real(S_tot_complex,
                                             int_order, ext_order)
        assert_allclose(S_tot, S_tot_round, atol=1e-7)

        S_tot_mat = np.concatenate([sss_data['Sin' + o_str],
                                    sss_data['Sout' + o_str]], axis=1)
        S_tot_mat_real = _bases_complex_to_real(S_tot_mat,
                                                int_order, ext_order)
        S_tot_mat_round = _bases_real_to_complex(S_tot_mat_real,
                                                 int_order, ext_order)
        assert_allclose(S_tot_mat, S_tot_mat_round, atol=1e-7)
        assert_allclose(S_tot_complex, S_tot_mat, rtol=1e-4, atol=1e-8)
        assert_allclose(S_tot, S_tot_mat_real, rtol=1e-4, atol=1e-8)

        # Now normalize our columns
        S_tot /= np.sqrt(np.sum(S_tot * S_tot, axis=0))[np.newaxis]
        S_tot_complex /= np.sqrt(np.sum(
            (S_tot_complex * S_tot_complex.conj()).real, axis=0))[np.newaxis]
        # Check against a known benchmark
        S_tot_mat = np.concatenate([sss_data['SNin' + o_str],
                                    sss_data['SNout' + o_str]], axis=1)
        # Check this roundtrip
        S_tot_mat_real = _bases_complex_to_real(S_tot_mat,
                                                int_order, ext_order)
        S_tot_mat_round = _bases_real_to_complex(S_tot_mat_real,
                                                 int_order, ext_order)
        assert_allclose(S_tot_mat, S_tot_mat_round, atol=1e-7)
        assert_allclose(S_tot_complex, S_tot_mat, rtol=1e-4, atol=1e-8)

        # Now test our optimized version
        S_tot = _sss_basis_basic(origin, coils, int_order, ext_order)
        S_tot_fast = _sss_basis(origin, coils, int_order, ext_order)
        S_tot_fast *= _get_coil_scale(coils)
        # there are some sign differences for columns (order/degrees)
        # in here, likely due to Condon-Shortley. Here we use a
        # Magnetometer channel to figure out the flips because the
        # gradiometer channels have effectively zero values for first three
        # external components (i.e., S_tot[grad_picks, 80:83])
        flips = (np.sign(S_tot_fast[2]) != np.sign(S_tot[2]))
        flips = 1 - 2 * flips
        assert_allclose(S_tot, S_tot_fast * flips, atol=1e-16)


@testing.requires_testing_data
def test_basic():
    """Test Maxwell filter basic version"""
    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
        raw_err = Raw(raw_fname, proj=True, allow_maxshield=True)
        raw_erm = Raw(erm_fname, allow_maxshield=True)
    assert_raises(RuntimeError, maxwell_filter, raw_err)
    assert_raises(TypeError, maxwell_filter, 1.)  # not a raw
    assert_raises(ValueError, maxwell_filter, raw, int_order=20)  # too many

    n_int_bases = int_order ** 2 + 2 * int_order
    n_ext_bases = ext_order ** 2 + 2 * ext_order
    nbases = n_int_bases + n_ext_bases

    # Check number of bases computed correctly
    assert_equal(_get_n_moments([int_order, ext_order]).sum(), nbases)

    # Test SSS computation at the standard head origin
    raw_sss = maxwell_filter(raw, origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, Raw(sss_std_fname), 200., 1000.)
    py_cal = raw_sss.info['proc_history'][0]['max_info']['sss_cal']
    assert_equal(len(py_cal), 0)
    py_ctc = raw_sss.info['proc_history'][0]['max_info']['sss_ctc']
    assert_equal(len(py_ctc), 0)
    py_st = raw_sss.info['proc_history'][0]['max_info']['max_st']
    assert_equal(len(py_st), 0)
    assert_raises(RuntimeError, maxwell_filter, raw_sss)

    # Test SSS computation at non-standard head origin
    raw_sss = maxwell_filter(raw, origin=[0., 0.02, 0.02], regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, Raw(sss_nonstd_fname), 250., 700.)

    # Test SSS computation at device origin
    sss_erm_std = Raw(sss_erm_std_fname)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg',
                             origin=mf_meg_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, sss_erm_std, 100., 900.)
    for key in ('job', 'frame'):
        vals = [x.info['proc_history'][0]['max_info']['sss_info'][key]
                for x in [raw_sss, sss_erm_std]]
        assert_equal(vals[0], vals[1])

    # Check against SSS functions from proc_history
    sss_info = raw_sss.info['proc_history'][0]['max_info']
    assert_equal(_get_n_moments(int_order),
                 proc_history._get_sss_rank(sss_info))

    # Degenerate cases
    raw_bad = raw.copy()
    raw_bad.comp = True
    assert_raises(RuntimeError, maxwell_filter, raw_bad)
    del raw_bad
    assert_raises(ValueError, maxwell_filter, raw, coord_frame='foo')
    assert_raises(ValueError, maxwell_filter, raw, origin='foo')
    assert_raises(ValueError, maxwell_filter, raw, origin=[0] * 4)


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
    int_order = 8
    raw_sss = maxwell_filter(raw, origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sss.fif')
    raw_sss.save(test_outname)
    raw_sss_loaded = Raw(test_outname, preload=True)

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
    """Test Maxwell filter reconstruction of bad channels"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    raw.info['bads'] = bads
    raw_sss = maxwell_filter(raw, origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, Raw(sss_bad_recon_fname), 300.)


@requires_svd_convergence
@testing.requires_testing_data
def test_spatiotemporal_maxwell():
    """Test Maxwell filter (tSSS) spatiotemporal processing"""
    # Load raw testing data
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True)

    # Test that window is less than length of data
    assert_raises(ValueError, maxwell_filter, raw, st_duration=1000.)

    # Check both 4 and 10 seconds because Elekta handles them differently
    # This is to ensure that std/non-std tSSS windows are correctly handled
    st_durations = [4., 10.]
    tols = [325., 200.]
    for st_duration, tol in zip(st_durations, tols):
        # Load tSSS data depending on st_duration and get data
        tSSS_fname = op.join(sss_path,
                             'test_move_anon_st%0ds_raw_sss.fif' % st_duration)
        tsss_bench = Raw(tSSS_fname)
        # Because Elekta's tSSS sometimes(!) lumps the tail window of data
        # onto the previous buffer if it's shorter than st_duration, we have to
        # crop the data here to compensate for Elekta's tSSS behavior.
        if st_duration == 10.:
            tsss_bench.crop(0, st_duration, copy=False)

        # Test sss computation at the standard head origin. Same cropping issue
        # as mentioned above.
        if st_duration == 10.:
            raw_tsss = maxwell_filter(raw.crop(0, st_duration),
                                      origin=mf_head_origin,
                                      st_duration=st_duration, regularize=None,
                                      bad_condition='ignore')
        else:
            raw_tsss = maxwell_filter(raw, st_duration=st_duration,
                                      origin=mf_head_origin, regularize=None,
                                      bad_condition='ignore')
        assert_meg_snr(raw_tsss, tsss_bench, tol)
        py_st = raw_tsss.info['proc_history'][0]['max_info']['max_st']
        assert_true(len(py_st) > 0)
        assert_equal(py_st['buflen'], st_duration)
        assert_equal(py_st['subspcorr'], 0.98)

    # Degenerate cases
    assert_raises(ValueError, maxwell_filter, raw, st_duration=10.,
                  st_correlation=0.)


@testing.requires_testing_data
def test_maxwell_filter_fine_calibration():
    """Test Maxwell filter fine calibration"""

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    sss_fine_cal = Raw(sss_fine_cal_fname)

    # Test 1D SSS fine calibration
    raw_sss = maxwell_filter(raw, calibration=fine_cal_fname,
                             origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, sss_fine_cal, 70, 500)
    py_cal = raw_sss.info['proc_history'][0]['max_info']['sss_cal']
    assert_true(py_cal is not None)
    assert_true(len(py_cal) > 0)
    mf_cal = sss_fine_cal.info['proc_history'][0]['max_info']['sss_cal']
    # we identify these differently
    mf_cal['cal_chans'][mf_cal['cal_chans'][:, 1] == 3022, 1] = 3024
    assert_allclose(py_cal['cal_chans'], mf_cal['cal_chans'])
    assert_allclose(py_cal['cal_corrs'], mf_cal['cal_corrs'],
                    rtol=1e-3, atol=1e-3)

    # Test 3D SSS fine calibration (no equivalent func in MaxFilter yet!)
    # very low SNR as proc differs, eventually we should add a better test
    raw_sss_3D = maxwell_filter(raw, calibration=fine_cal_fname_3d,
                                origin=mf_head_origin, regularize=None,
                                bad_condition='ignore')
    assert_meg_snr(raw_sss_3D, sss_fine_cal, 1.0, 6.)


@slow_test
@testing.requires_testing_data
def test_maxwell_filter_regularization():
    """Test Maxwell filter regularization"""
    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    min_tols = (100., 2.6, 1.0)
    med_tols = (1000., 21.4, 3.7)
    origins = ((0., 0., 0.04), (0.,) * 3, (0., 0.02, 0.02))
    coord_frames = ('head', 'meg', 'head')
    raw_fnames = (raw_fname, erm_fname, sample_fname)
    sss_fnames = (sss_reg_in_fname, sss_erm_reg_in_fname,
                  sss_samp_reg_in_fname)
    comp_tols = [0, 1, 4]
    for ii, rf in enumerate(raw_fnames):
        with warnings.catch_warnings(record=True):  # maxshield
            raw = Raw(rf, allow_maxshield=True).crop(0., 1., False)
        sss_reg_in = Raw(sss_fnames[ii])

        # Test "in" regularization
        raw_sss = maxwell_filter(raw, coord_frame=coord_frames[ii],
                                 origin=origins[ii])
        assert_meg_snr(raw_sss, sss_reg_in, min_tols[ii], med_tols[ii], msg=rf)

        # check components match
        py_info = raw_sss.info['proc_history'][0]['max_info']['sss_info']
        assert_true(py_info is not None)
        assert_true(len(py_info) > 0)
        mf_info = sss_reg_in.info['proc_history'][0]['max_info']['sss_info']
        n_in = None
        for inf in py_info, mf_info:
            if n_in is None:
                n_in = _get_n_moments(inf['in_order'])
            else:
                assert_equal(n_in, _get_n_moments(inf['in_order']))
            assert_equal(inf['components'][:n_in].sum(), inf['nfree'])
        assert_allclose(py_info['nfree'], mf_info['nfree'],
                        atol=comp_tols[ii], err_msg=rf)


@testing.requires_testing_data
def test_cross_talk():
    """Test Maxwell filter cross-talk cancellation"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    raw.info['bads'] = bads
    sss_ctc = Raw(sss_ctc_fname)
    raw_sss = maxwell_filter(raw, cross_talk=ctc_fname,
                             origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, sss_ctc, 275.)
    py_ctc = raw_sss.info['proc_history'][0]['max_info']['sss_ctc']
    assert_true(len(py_ctc) > 0)
    assert_raises(ValueError, maxwell_filter, raw, cross_talk=raw)
    assert_raises(ValueError, maxwell_filter, raw, cross_talk=raw_fname)
    mf_ctc = sss_ctc.info['proc_history'][0]['max_info']['sss_ctc']
    del mf_ctc['block_id']  # we don't write this
    assert_equal(object_diff(py_ctc, mf_ctc), '')


@testing.requires_testing_data
def test_head_translation():
    """Test Maxwell filter head translation"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, allow_maxshield=True).crop(0., 1., False)
    # First try with an unchanged destination
    raw_sss = maxwell_filter(raw, destination=raw_fname,
                             origin=mf_head_origin, regularize=None,
                             bad_condition='ignore')
    assert_meg_snr(raw_sss, Raw(sss_std_fname).crop(0., 1., False), 200.)
    # Now with default
    with catch_logging() as log:
        raw_sss = maxwell_filter(raw, destination=mf_head_origin,
                                 origin=mf_head_origin, regularize=None,
                                 bad_condition='ignore', verbose='warning')
    assert_true('over 25 mm' in log.getvalue())
    assert_meg_snr(raw_sss, Raw(sss_trans_default_fname), 125.)
    # Now to sample's head pos
    with catch_logging() as log:
        raw_sss = maxwell_filter(raw, destination=sample_fname,
                                 origin=mf_head_origin, regularize=None,
                                 bad_condition='ignore', verbose='warning')
    assert_true('= 25.6 mm' in log.getvalue())
    assert_meg_snr(raw_sss, Raw(sss_trans_sample_fname), 350.)
    # Degenerate cases
    assert_raises(RuntimeError, maxwell_filter, raw,
                  destination=mf_head_origin, coord_frame='meg')
    assert_raises(ValueError, maxwell_filter, raw, destination=[0.] * 4)


# TODO: Eventually add simulation tests mirroring Taulu's original paper
# that calculates the localization error:
# http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1495874

def _assert_shielding(raw_sss, erm_power, shielding_factor, meg='mag'):
    """Helper to assert a minimum shielding factor using empty-room power"""
    picks = pick_types(raw_sss.info, meg=meg)
    sss_power = raw_sss[picks][0].ravel()
    sss_power = np.sqrt(np.sum(sss_power * sss_power))
    factor = erm_power / sss_power
    assert_true(factor >= shielding_factor,
                'Shielding factor %0.3f < %0.3f' % (factor, shielding_factor))


@slow_test
@requires_svd_convergence
@testing.requires_testing_data
def test_noise_rejection():
    """Test Maxwell filter shielding factor using empty room"""
    with warnings.catch_warnings(record=True):  # maxshield
        raw_erm = Raw(erm_fname, allow_maxshield=True, preload=True)
    picks = pick_types(raw_erm.info, meg='mag')
    erm_power = raw_erm[picks][0].ravel()
    erm_power = np.sqrt(np.sum(erm_power * erm_power))

    # Vanilla SSS (second value would be for meg=True instead of meg='mag')
    _assert_shielding(Raw(sss_erm_std_fname), erm_power, 10)  # 1.5)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None)
    _assert_shielding(raw_sss, erm_power, 12)  # 1.5)

    # Fine cal
    _assert_shielding(Raw(sss_erm_fine_cal_fname), erm_power, 12)  # 2.0)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             origin=mf_meg_origin,
                             calibration=fine_cal_fname)
    _assert_shielding(raw_sss, erm_power, 12)  # 2.0)

    # Crosstalk
    _assert_shielding(Raw(sss_erm_ctc_fname), erm_power, 12)  # 2.1)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             origin=mf_meg_origin,
                             cross_talk=ctc_fname)
    _assert_shielding(raw_sss, erm_power, 12)  # 2.1)

    # Fine cal + Crosstalk
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             calibration=fine_cal_fname,
                             origin=mf_meg_origin,
                             cross_talk=ctc_fname)
    _assert_shielding(raw_sss, erm_power, 13)  # 2.2)

    # tSSS
    _assert_shielding(Raw(sss_erm_st_fname), erm_power, 37)  # 5.8)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             origin=mf_meg_origin, st_duration=1.)
    _assert_shielding(raw_sss, erm_power, 37)  # 5.8)

    # Crosstalk + tSSS
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             cross_talk=ctc_fname, origin=mf_meg_origin,
                             st_duration=1.)
    _assert_shielding(raw_sss, erm_power, 38)  # 5.91)

    # Fine cal + tSSS
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             calibration=fine_cal_fname,
                             origin=mf_meg_origin, st_duration=1.)
    _assert_shielding(raw_sss, erm_power, 38)  # 5.98)

    # Fine cal + Crosstalk + tSSS
    _assert_shielding(Raw(sss_erm_st1FineCalCrossTalk_fname),
                      erm_power, 39)  # 6.07)
    raw_sss = maxwell_filter(raw_erm, coord_frame='meg', regularize=None,
                             calibration=fine_cal_fname, origin=mf_meg_origin,
                             cross_talk=ctc_fname, st_duration=1.)
    _assert_shielding(raw_sss, erm_power, 39)  # 6.05)

    # Fine cal + Crosstalk + tSSS + Reg-in
    _assert_shielding(Raw(sss_erm_st1FineCalCrossTalkRegIn_fname), erm_power,
                      57)  # 6.97)
    raw_sss = maxwell_filter(raw_erm, calibration=fine_cal_fname,
                             cross_talk=ctc_fname, st_duration=1.,
                             origin=mf_meg_origin,
                             coord_frame='meg', regularize='in')
    _assert_shielding(raw_sss, erm_power, 53)  # 6.64)
    raw_sss = maxwell_filter(raw_erm, calibration=fine_cal_fname,
                             cross_talk=ctc_fname, st_duration=1.,
                             coord_frame='meg', regularize='in')
    _assert_shielding(raw_sss, erm_power, 58)  # 7.0)
    raw_sss = maxwell_filter(raw_erm, calibration=fine_cal_fname_3d,
                             cross_talk=ctc_fname, st_duration=1.,
                             coord_frame='meg', regularize='in')

    # Our 3D cal has worse defaults for this ERM than the 1D file
    _assert_shielding(raw_sss, erm_power, 54)
    # Show it by rewriting the 3D as 1D and testing it
    temp_dir = _TempDir()
    temp_fname = op.join(temp_dir, 'test_cal.dat')
    with open(fine_cal_fname_3d, 'r') as fid:
        with open(temp_fname, 'w') as fid_out:
            for line in fid:
                fid_out.write(' '.join(line.strip().split(' ')[:14]) + '\n')
    raw_sss = maxwell_filter(raw_erm, calibration=temp_fname,
                             cross_talk=ctc_fname, st_duration=1.,
                             coord_frame='meg', regularize='in')
    # Our 3D cal has worse defaults for this ERM than the 1D file
    _assert_shielding(raw_sss, erm_power, 44)


@slow_test
@requires_svd_convergence
@testing.requires_testing_data
def test_all():
    """Test maxwell filter using all options"""
    raw_fnames = (raw_fname, raw_fname, erm_fname, sample_fname)
    sss_fnames = (sss_st1FineCalCrossTalkRegIn_fname,
                  sss_st1FineCalCrossTalkRegInTransSample_fname,
                  sss_erm_st1FineCalCrossTalkRegIn_fname,
                  sss_samp_fname)
    fine_cals = (fine_cal_fname,
                 fine_cal_fname,
                 fine_cal_fname,
                 fine_cal_mgh_fname)
    coord_frames = ('head', 'head', 'meg', 'head')
    ctcs = (ctc_fname, ctc_fname, ctc_fname, ctc_mgh_fname)
    mins = (3.5, 3.5, 1.2, 0.9)
    meds = (10.9, 10.4, 3.2, 6.)
    st_durs = (1., 1., 1., None)
    destinations = (None, sample_fname, None, None)
    origins = (mf_head_origin,
               mf_head_origin,
               mf_meg_origin,
               mf_head_origin)
    for ii, rf in enumerate(raw_fnames):
        with warnings.catch_warnings(record=True):  # maxshield
            raw = Raw(rf, allow_maxshield=True).crop(0., 1., copy=False)
        sss_py = maxwell_filter(raw, calibration=fine_cals[ii],
                                cross_talk=ctcs[ii], st_duration=st_durs[ii],
                                coord_frame=coord_frames[ii],
                                destination=destinations[ii],
                                origin=origins[ii])
        sss_mf = Raw(sss_fnames[ii])
        assert_meg_snr(sss_py, sss_mf, mins[ii], meds[ii], msg=rf)

run_tests_if_main()
