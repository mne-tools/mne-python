import numpy as np
from numpy.testing import assert_array_almost_equal

from ..filter import band_pass_filter, high_pass_filter, low_pass_filter

def test_filters():
    a = np.random.randn(1000)
    Fs = 1000
    bp = band_pass_filter(a, Fs, 4, 8)
    lp = low_pass_filter(a, Fs, 8)
    hp = high_pass_filter(lp, Fs, 4)
    assert_array_almost_equal(hp, bp, 2)
