# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from mne.preprocessing import peak_finder


def test_peak_finder():
    """Test the peak detection method."""
    # check for random data
    rng = np.random.default_rng(42)
    peak_inds, peak_mags = peak_finder(rng.standard_normal(20))

    assert_equal(peak_inds.dtype, np.dtype("int64"))
    assert_equal(peak_mags.dtype, np.dtype("float64"))

    # check for empty array as created in the #5025
    with pytest.raises(ValueError):
        peak_finder(np.arange(2, 1, 0.05))

    # check for empty array
    with pytest.raises(ValueError):
        peak_finder([])

    # check for monotonic function
    peak_inds, peak_mags = peak_finder(np.arange(1, 2, 0.05))

    assert_equal(peak_inds.dtype, np.dtype("int64"))
    assert_equal(peak_mags.dtype, np.dtype("float64"))

    # check for no peaks
    peak_inds, peak_mags = peak_finder(np.zeros(20))

    assert_equal(peak_inds.dtype, np.dtype("int64"))
    assert_equal(peak_mags.dtype, np.dtype("float64"))

    # check values
    peak_inds, peak_mags = peak_finder([0, 2, 5, 0, 6, -1])
    assert_array_equal(peak_inds, [2, 4])

    # integer data with plateaus gives the same peaks as float data
    x = [0, 0, 3, 3, 0, 0, 5, 5, 5, 0, 1]
    peak_inds, peak_mags = peak_finder(np.array(x, int), thresh=1)
    peak_inds_float, peak_mags_float = peak_finder(np.array(x, float), thresh=1)
    assert_array_equal(peak_inds, [2, 6])
    assert_array_equal(peak_inds, peak_inds_float)
    assert_array_equal(peak_mags, peak_mags_float)

    # no minimum found in a monotonic function
    peak_inds, peak_mags = peak_finder(np.arange(5.0), thresh=10, extrema=-1)
    assert len(peak_inds) == len(peak_mags) == 0
    assert_equal(peak_mags.dtype, np.dtype("float64"))
