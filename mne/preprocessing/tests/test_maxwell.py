# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal)
from nose.tools import assert_true

from mne.preprocessing import maxwell
from mne.datasets import testing
from mne.io import Raw
from mne.utils import slow_test
warnings.simplefilter('always')  # Always throw warnings


def test_spherical_harmonic():
    """Test for spherical harmonics"""
    from scipy.special import sph_harm as scipy_sph_harm

    deg = 1
    order = -1
    azimuth = np.random.random_sample(size=(50, 1)) * 2 * np.pi
    polar = np.random.random_sample(size=(50, 1)) * np.pi

    # Internal calculation
    sph_harmonic = maxwell._sph_harmonic(deg, order, azimuth, polar)
    # Check against scipy
    sph_harmonic_scipy = scipy_sph_harm(order, deg, azimuth, polar)

    assert_array_almost_equal(sph_harmonic, sph_harmonic_scipy, decimal=15,
                              err_msg='Spherical harmonic mismatch')


@slow_test
@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter"""

    # Load testing data (raw, SSS std origin, SSS non-standard origin)
    data_path = op.join(testing.data_path(download=False))

    file_name = 'test_move_anon'

    raw_fname = op.join(data_path, 'SSS', file_name + '_raw.fif')
    sss_std_fname = op.join(data_path, 'SSS', file_name +
                            '_raw_simp_stdOrigin_sss.fif')
    sss_nonStd_fname = op.join(data_path, 'SSS', file_name +
                               '_raw_simp_nonStdOrigin_sss.fif')

    raw = Raw(raw_fname, preload=False, proj=False,
              allow_maxshield=True).crop(0., 1., False)
    raw.preload_data()
    sss_std = Raw(sss_std_fname, preload=True, proj=False,
                  allow_maxshield=True)
    sss_nonStd = Raw(sss_nonStd_fname, preload=True, proj=False,
                     allow_maxshield=True)

    # Create coils
    all_coils, meg_info = maxwell._make_coils(raw.info)
    picks = [raw.info['ch_names'].index(ch) for ch in [coil['chname']
                                                       for coil in all_coils]]
    coils = [all_coils[ci] for ci in picks]
    ncoils = len(coils)

    raw.pick_channels(all_coils[ci]['chname'] for ci in picks)
    sss_std.pick_channels(all_coils[ci]['chname'] for ci in picks)
    sss_nonStd.pick_channels(all_coils[ci]['chname'] for ci in picks)

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
    assert_allclose(np.linalg.norm(S_in, axis=0), np.ones((S_in.shape[1])),
                    rtol=1e-12, err_msg='S_in normalization error')
    assert_allclose(np.linalg.norm(S_out, axis=0), np.ones((S_out.shape[1])),
                    rtol=1e-12, err_msg='S_out normalization error')

    # Test sss computation at the standard head origin
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 0., 40.],
                                     int_order=int_order, ext_order=ext_order)

    assert_array_almost_equal(raw_sss[:, :][0], sss_std[:, :][0], decimal=11,
                              err_msg='Maxwell filtered data at standard '
                              ' origin incorrect.')

    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_std[:, :][0] ** 2, axis=1))
    error = raw_sss[:, :][0] - sss_std[:, :][0]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) > 1000, 'SNR < 1000')

    # Test sss computation at non-standard head origin
    raw_sss = maxwell.maxwell_filter(raw, origin=[0., 20., 20.],
                                     int_order=int_order, ext_order=ext_order)
    assert_array_almost_equal(raw_sss[:, :][0], sss_nonStd[:, :][0],
                              decimal=11, err_msg='Maxwell filtered data at '
                              'non-std origin incorrect.')
    # Confirm SNR is above 100
    bench_rms = np.sqrt(np.mean(sss_nonStd[:, :][0] ** 2, axis=1))
    error = raw_sss[:, :][0] - sss_nonStd[:, :][0]
    error_rms = np.sqrt(np.mean(error ** 2, axis=1))
    assert_true(np.mean(bench_rms / error_rms) > 1000, 'SNR < 1000')

