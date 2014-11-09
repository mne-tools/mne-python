import numpy as np
from nose.tools import assert_true
from mne.utils import check_random_state
from scipy import stats
from scipy.signal import hilbert
from mne.connectivity.cfc import modulation_index


def test_phase_amplitude_coupling():

    fs, times, trials = 100., 1, 10
    times = np.linspace(0, times, fs * times)

    rng = check_random_state(42)
    white_noise = rng.normal(0, 0.2, len(times))

    phase_series_fp = np.angle(hilbert(white_noise)) + np.pi
    bin_size = 2 * np.pi / 18  # for 18 bins
    phase_bins = np.arange(phase_series_fp.min(),
                           phase_series_fp.max() + bin_size, bin_size)
    assert_true(len(phase_bins) - 1 == 18)

    amplitude_bin_means = np.zeros((trials, 18))
    amplitude_bin_means[:, rng.randint(0, 19)] = 1
    for i in range(trials):
        assert_true(modulation_index(amplitude_bin_means)[i] == 1.)


def test_modulation_index():

    dist = np.random.rand(1, 18)
    dist /= dist.sum()
    mi = 1. - (stats.entropy(dist[0]) / np.log(len(dist[0])))
    assert_true(0. <= mi <= 1.)


def test_cross_frequency_coupling():
    pass
