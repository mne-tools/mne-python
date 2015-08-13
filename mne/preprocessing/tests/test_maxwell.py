# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal)
from nose.tools import assert_true, assert_raises

from mne import compute_raw_data_covariance, pick_types
from mne.cov import _estimate_rank_meeg_cov
from mne.datasets import testing
from mne.forward._make_forward import _prep_meg_channels
from mne.io import Raw, proc_history
from mne.preprocessing import maxwell
from mne.utils import _TempDir, run_tests_if_main

warnings.simplefilter('always')  # Always throw warnings

data_path = op.join(testing.data_path(download=False))
raw_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
sss_std_fname = op.join(data_path, 'SSS',
                        'test_move_anon_raw_simp_stdOrigin_sss.fif')
sss_nonstd_fname = op.join(data_path, 'SSS',
                           'test_move_anon_raw_simp_nonStdOrigin_sss.fif')
sss_bad_recon_fname = op.join(data_path, 'SSS',
                              'test_move_anon_raw_bad_recon_sss.fif')


@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter"""

    # TODO: Future tests integrate with mne/io/tests/test_proc_history

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    raw.preload_data()
    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_std = Raw(sss_std_fname, preload=True, proj=False,
                      allow_maxshield=True)
        sss_nonStd = Raw(sss_nonstd_fname, preload=True, proj=False,
                         allow_maxshield=True)
        raw_err = Raw(raw_fname, preload=False, proj=True,
                      allow_maxshield=True).crop(0., 0.1, False)
    assert_raises(RuntimeError, maxwell.maxwell_filter, raw_err)

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
    assert_equal(maxwell.get_num_moments(int_order, ext_order), nbases)

    # Check multipolar moment basis set
    S_in, S_out = maxwell._sss_basis(origin=np.array([0, 0, 40]), coils=coils,
                                     int_order=int_order, ext_order=ext_order)
    assert_equal(S_in.shape, (ncoils, n_int_bases), 'S_in has incorrect shape')
    assert_equal(S_out.shape, (ncoils, n_ext_bases),
                 'S_out has incorrect shape')

    # Check normalization
    assert_allclose(np.sum(S_in ** 2, axis=0), np.ones((S_in.shape[1])),
                    rtol=1e-12, err_msg='S_in normalization error')
    assert_allclose(np.sum(S_out ** 2, axis=0), np.ones((S_out.shape[1])),
                    rtol=1e-12, err_msg='S_out normalization error')

    # Test sss computation at the standard head origin
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 0., 40.],
                                     int_order=int_order, ext_order=ext_order)

    assert_array_almost_equal(raw_sss._data[picks, :], sss_std._data[picks, :],
                              decimal=11, err_msg='Maxwell filtered data at '
                              'standard origin incorrect.')

    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_std._data[picks, :] ** 2, axis=1))
    error = raw_sss._data[picks, :] - sss_std._data[picks, :]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) > 1000, 'SNR < 1000')

    # Test sss computation at non-standard head origin
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 20., 20.],
                                     int_order=int_order, ext_order=ext_order)
    assert_array_almost_equal(raw_sss._data[picks, :],
                              sss_nonStd._data[picks, :], decimal=11,
                              err_msg='Maxwell filtered data at non-std '
                              'origin incorrect.')
    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_nonStd._data[picks, :] ** 2, axis=1))
    error = raw_sss._data[picks, :] - sss_nonStd._data[picks, :]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) > 1000, 'SNR < 1000')

    # Check against SSS functions from proc_history
    sss_info = raw_sss.info['proc_history'][0]['max_info']
    assert_equal(maxwell.get_num_moments(int_order, 0),
                 proc_history._get_sss_rank(sss_info))


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
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 2., False)

    # Get MEG channels, compute Maxwell filtered data
    raw.preload_data()
    raw.pick_types(meg=True, eeg=False)
    int_order, ext_order = 8, 3
    raw_sss = maxwell.maxwell_filter(raw, int_order=int_order,
                                     ext_order=ext_order)

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sss.fif')
    raw_sss.save(test_outname)
    raw_sss_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)

    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sss_loaded._data[:, :], raw_sss._data[:, :],
                    rtol=1e-6, atol=1e-20)

    # Test rank of covariance matrices for raw and SSS processed data
    cov_raw = compute_raw_data_covariance(raw)
    cov_sss = compute_raw_data_covariance(raw_sss)

    scalings = None
    cov_raw_rank = _estimate_rank_meeg_cov(cov_raw['data'], raw.info, scalings)
    cov_sss_rank = _estimate_rank_meeg_cov(cov_sss['data'], raw_sss.info,
                                           scalings)

    assert_equal(cov_raw_rank, raw.info['nchan'])
    assert_equal(cov_sss_rank, maxwell.get_num_moments(int_order, 0))


@testing.requires_testing_data
def test_bads_reconstruction():
    """Test reconstruction of channels marked as bad"""

    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_bench = Raw(sss_bad_recon_fname, preload=True, proj=False,
                        allow_maxshield=True)

    raw_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')

    with warnings.catch_warnings(record=True):  # maxshield
        # Use 2.0 seconds of data to get stable cov. estimate
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)

    raw.preload_data()

    # Set bad MEG channels, compute Maxwell filtered data
    raw.info['bads'] = raw.info['ch_names'][0:9]
    raw_sss = maxwell.maxwell_filter(raw)
    meg_chs = pick_types(raw_sss.info)

    assert_array_almost_equal(raw_sss._data[meg_chs, :],
                              sss_bench._data[meg_chs, :], decimal=11,
                              err_msg='Maxwell filtered data with '
                              ' reconstructed bads is incorrect.')

    # Confirm SNR is above 1000
    bench_rms = np.sqrt(np.mean(raw_sss._data[meg_chs, :] ** 2, axis=1))
    error = raw_sss._data[meg_chs, :] - sss_bench._data[meg_chs, :]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) >= 1000, 'SNR < 1000')

run_tests_if_main()
