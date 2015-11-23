import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.connectivity import phase_slope_index


def test_psi():
    """Test Phase Slope Index (PSI) estimation"""
    sfreq = 50.
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, n_signals, n_times)

    # simulate time shifts
    for i in range(n_epochs):
        data[i, 1, 10:] = data[i, 0, :-10]  # signal 0 is ahead
        data[i, 2, :-10] = data[i, 0, 10:]  # signal 2 is ahead

    psi, freqs, times, n_epochs, n_tapers = phase_slope_index(
        data, mode='fourier', sfreq=sfreq)
    assert_true(psi[1, 0, 0] < 0)
    assert_true(psi[2, 0, 0] > 0)

    indices = (np.array([0]), np.array([1]))
    psi_2, freqs, times, n_epochs, n_tapers = phase_slope_index(
        data, mode='fourier', sfreq=sfreq, indices=indices)

    # the measure is symmetric (sign flip)
    assert_array_almost_equal(psi_2[0, 0], -psi[1, 0, 0])

    cwt_freqs = np.arange(5., 20, 0.5)
    psi_cwt, freqs, times, n_epochs, n_tapers = phase_slope_index(
        data, mode='cwt_morlet', sfreq=sfreq, cwt_frequencies=cwt_freqs,
        indices=indices)

    assert_true(np.all(psi_cwt > 0))
    assert_true(psi_cwt.shape[-1] == n_times)
