import numpy as np
from ..peak_finder import peak_finder
from numpy.testing import assert_array_equal


def test_peak_finder():
    """Test the peak detection method"""
    x = [0, 2, 5, 0, 6, -1]
    peak_inds, peak_mags = peak_finder(x)
    assert_array_equal(peak_inds, [2, 4])
