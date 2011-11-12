import numpy as np
from numpy.testing import assert_array_almost_equal

from ..filter import band_pass_filter, high_pass_filter, low_pass_filter


def test_filters():
    Fs = 500
    sig_len_secs = 60

    # Filtering of short signals (filter length = len(a))
    a = np.random.randn(sig_len_secs * Fs)
    bp = band_pass_filter(a, Fs, 4, 8)
    lp = low_pass_filter(a, Fs, 8)
    hp = high_pass_filter(lp, Fs, 4)
    assert_array_almost_equal(hp, bp, 2)

    # Overlap-add filtering with a fixed filter length
    filter_length = 8192
    bp_oa = band_pass_filter(a, Fs, 4, 8, filter_length)
    lp_oa = low_pass_filter(a, Fs, 8, filter_length)
    hp_oa = high_pass_filter(lp_oa, Fs, 4, filter_length)
    assert_array_almost_equal(hp_oa, bp_oa, 2)

    # The two methods should give the same result
    # As filtering for short signals uses a circular convolution (FFT) and
    # the overlap-add filter implements a linear convolution, the signal
    # boundary will be slightly different and we ignore it
    n_edge_ignore = 1000
    assert_array_almost_equal(hp[n_edge_ignore:-n_edge_ignore],
                              hp_oa[n_edge_ignore:-n_edge_ignore], 2)
