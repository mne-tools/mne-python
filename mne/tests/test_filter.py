import numpy as np
from numpy.testing import assert_array_almost_equal

from ..filter import band_pass_filter, high_pass_filter, low_pass_filter

def test_filters():
    Fs = 250
    # Test short and long signals (direct FFT and overlap-add FFT filtering)
    for sig_len_secs in [10, 90]:
        a = np.random.randn(sig_len_secs * Fs)
        bp = band_pass_filter(a, Fs, 4, 8)
        lp = low_pass_filter(a, Fs, 8)
        hp = high_pass_filter(lp, Fs, 4)
        assert_array_almost_equal(hp, bp, 2)
