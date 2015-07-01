# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings
from mne.datasets import sample
from mne.io import Raw
from mne.datasets import testing

from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal)
import numpy as np

from mne.preprocessing import maxwell

from scipy.special import sph_harm as scipy_sph_harm

warnings.simplefilter('always')  # Always throw warnings


def test_spherical_harmonic():
    """Test for spherical harmonics"""
    deg = 1
    order = -1
    azimuth = np.random.random_sample(size=(50, 1)) * 2 * np.pi
    polar = np.random.random_sample(size=(50, 1)) * np.pi

    # Internal calculation
    sph_harmonic = maxwell._sph_harmonic(deg, order, azimuth, polar)
    # Check against scipy
    sph_harmonic_scipy = scipy_sph_harm(order, deg, azimuth, polar)

    assert_array_almost_equal(sph_harmonic, sph_harmonic_scipy, decimal=15,
                              err_msg='Spherical harmonic calculation mismatch')


@testing.requires_testing_data
def test_maxwell_filter():
    """Test multipolar moment and Maxwell filter"""

    # Load data
    data_path = sample.data_path()
    raw_fname = op.join(data_path, 'MEG/sample', 'sample_audvis_raw.fif')
    raw = Raw(raw_fname, preload=False, proj=False).crop(0., 1., False)
    raw.preload_data()

    #sss_proc_fname = op.join(data_path, 'sample_audvis_raw_sss.fif')
    #sss_benchmark = mne.io.Raw(sss_proc_fname, preload=False,
    #                           proj=False).crop(0., 1., False)
    #sss_benchmark.preload_data()

    int_order, ext_order = 8, 3
    origin = np.array([0, 0, 40.])  # Test with brain center in head coords

    all_coils, meg_info = maxwell._make_coils(raw.info)
    picks = [raw.info['ch_names'].index(ch) for ch in [coil['chname']
                                                       for coil in all_coils]]
    coils = [all_coils[ci] for ci in picks]
    ncoils = len(coils)
    n_int_bases = int_order ** 2 + 2 * int_order
    n_ext_bases = ext_order ** 2 + 2 * ext_order
    nbases = n_int_bases + n_ext_bases

    # Check number of bases computed correctly
    assert_equal(maxwell.get_num_moments(int_order, ext_order), nbases)

    # Compute multipolar moments calculated correctly
    S_in, S_out = maxwell._sss_basis(origin, coils, int_order=8, ext_order=3)
    assert_equal(S_in.shape, (ncoils, n_int_bases), 'S_in has incorrect shape')
    assert_equal(S_out.shape, (ncoils, n_ext_bases),
                 'S_out has incorrect shape')

    # Check normalization
    assert_allclose(np.linalg.norm(S_in, axis=0), np.ones((S_in.shape[1])),
                    atol=1e-15, err_msg='S_in normalization error')
    assert_allclose(np.linalg.norm(S_out, axis=0), np.ones((S_out.shape[1])),
                    atol=1e-15, err_msg='S_out normalization error')

    # Test sss computation
    raw_sss = maxwell.maxwell_filter(raw, coils, origin, int_order=int_order,
                                     ext_order=ext_order)
    #assert_array_almost_equal(raw_sss, sss_benchmark, decimal=15,
    #                          err_msg='Maxwell filtered data incorrect.)
