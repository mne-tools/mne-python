import numpy as np
from mne.preprocessing.peak_finder import peak_finder
from numpy.testing import assert_raises, assert_equal
from mne.utils import run_tests_if_main


def test_peak_finder():

    # check for random data
    peak_inds, peak_mags = peak_finder(np.asarray(np.random.random(20)))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

    # check for empty array as created in the #5025
    with assert_raises(ValueError):
        peak_finder(np.arange(2, 1, 0.05))

    # check for empty array
    with assert_raises(ValueError):
        peak_finder([])

    # check for monotonic function
    peak_inds, peak_mags = peak_finder(np.arange(1, 2, 0.05))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

    # check for no peaks
    peak_inds, peak_mags = peak_finder(np.asarray(np.zeros(20)))

    assert_equal(peak_inds.dtype, np.dtype('int64'))
    assert_equal(peak_mags.dtype, np.dtype('float64'))

run_tests_if_main()
