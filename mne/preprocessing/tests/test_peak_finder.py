from numpy.testing import assert_array_equal, assert_equal
import pytest
import numpy as np

from mne.preprocessing import peak_finder


def test_peak_finder():
    """Test the peak detection method."""
    # check for random data
    rng = np.random.RandomState(42)
    peak_inds, peak_mags = peak_finder(rng.randn(20))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

    # check for empty array as created in the #5025
    with pytest.raises(ValueError):
        peak_finder(np.arange(2, 1, 0.05))

    # check for empty array
    with pytest.raises(ValueError):
        peak_finder([])

    # check for monotonic function
    peak_inds, peak_mags = peak_finder(np.arange(1, 2, 0.05))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

    # check for no peaks
    peak_inds, peak_mags = peak_finder(np.zeros(20))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

    # check values
    peak_inds, peak_mags = peak_finder([0, 2, 5, 0, 6, -1])
    assert_array_equal(peak_inds, [2, 4])
