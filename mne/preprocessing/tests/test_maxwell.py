# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal
import numpy as np

from ...io import Raw
from ..maxwell import _sss_basis, maxwell_filter, _sph_harmonic

from scipy.special import sph_harm as scipy_sph_harm

warnings.simplefilter('always')  # Always throw warnings


def test_sss_basis():
    """Test that the multipolar moment basis is computed correctly"""
    deg = 1
    order = 1
    polar = np.array([np.pi / 7., np.pi / 9., np.pi / 20.])
    azimuth = np.array([np.pi / 11., np.pi * 1.9, np.pi * 1.3])

    # Internal calculation: _sph_harmonic(degree, order, azimuth, polar)
    sph_harmonic = _sph_harmonic(deg, order, azimuth, polar)
    # Check against scipy: sph_harm(order, degree, azimuth, polar)
    sph_harmonic_scipy = np.real(scipy_sph_harm(deg, order, azimuth, polar))

    assert_array_almost_equal(sph_harmonic, sph_harmonic_scipy)


def test_maxwell_filter():
    """Test Maxwell filter against precomputed test set"""
    pass
