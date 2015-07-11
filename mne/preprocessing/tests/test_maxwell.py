# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal)
from nose.tools import assert_true, assert_raises

from mne.preprocessing import maxwell
from mne.datasets import testing
from mne.io import Raw, proc_history
from mne.utils import slow_test, _TempDir
warnings.simplefilter('always')  # Always throw warnings


@slow_test
@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter"""

    # TODO: Future tests integrate with mne/io/tests/test_proc_history

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    data_path = op.join(testing.data_path(download=False))

    file_name = 'test_move_anon'

    raw_fname = op.join(data_path, 'SSS', file_name + '_raw.fif')
    sss_std_fname = op.join(data_path, 'SSS', file_name +
                            '_raw_simp_stdOrigin_sss.fif')
    sss_nonStd_fname = op.join(data_path, 'SSS', file_name +
                               '_raw_simp_nonStdOrigin_sss.fif')

    with warnings.catch_warnings(record=True):  # maxshield
        raw = Raw(raw_fname, preload=False, proj=False,
                  allow_maxshield=True).crop(0., 1., False)
    raw.preload_data()
    with warnings.catch_warnings(record=True):  # maxshield, naming
        sss_std = Raw(sss_std_fname, preload=True, proj=False,
                      allow_maxshield=True)
        sss_nonStd = Raw(sss_nonStd_fname, preload=True, proj=False,
                         allow_maxshield=True)
        raw_err = Raw(raw_fname, preload=False, proj=True,
                      allow_maxshield=True).crop(0., 0.1, False)
    assert_raises(RuntimeError, maxwell.maxwell_filter, raw_err)

    # Create coils
    all_coils, meg_info = maxwell._make_coils(raw.info)
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

    # Test io on processed data
    tempdir = _TempDir()
    test_outname = op.join(tempdir, 'test_raw_sss.fif')
    raw_sss.save(test_outname)
    raw_sss_loaded = Raw(test_outname, preload=True, proj=False,
                         allow_maxshield=True)
    # Some numerical imprecision since save uses 'single' fmt
    assert_allclose(raw_sss_loaded._data[:, :], raw_sss._data[:, :],
                    rtol=1e-6, atol=1e-20)

    # Check against SSS functions from proc_history
    sss_info = raw_sss.info['proc_history'][0]['max_info']
    assert_equal(maxwell.get_num_moments(int_order, 0),
                 proc_history._get_sss_rank(sss_info))
