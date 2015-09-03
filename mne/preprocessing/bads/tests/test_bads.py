import numpy as np
import mne

from nose.tools import (assert_true, assert_almost_equal,
                        assert_raises, assert_equal)
from numpy.testing import (assert_allclose)

from mne.preprocessing.bads.faster_ import (_hurst, _freqs_power)
from mne.preprocessing.bads import (find_bad_channels, find_bad_epochs,
                                    find_bad_channels_in_epochs)

# Signal properties used in the tests
length = 2  # in seconds
srate = 200. # in Hertz
n_channels = 32
n_epochs = 100
n_samples = int(length * srate)
time = np.arange(n_samples) / srate

# Fix the seed
np.random.seed(123)


def test_hurst():
    """Test internal hurst exponent function."""
    np.random.seed(123)

    # Hurst exponent of a sine wave
    p = np.atleast_2d(np.sin(1000))
    assert_almost_equal(p, 0.82687954)

    # Positive first derivative, hurst > 1
    p = np.atleast_2d(np.log10(np.cumsum(np.random.randn(1000) + 100)))
    assert_true(_hurst(p) > 1)

    # First derivative alternating around zero, hurst ~ 0
    p = np.atleast_2d(np.log10(np.random.randn(1000) + 1000))
    assert_allclose(_hurst(p), 0, atol=0.1)

    # Positive, but fluctuating first derivative, hurst ~ 0.5
    p = np.atleast_2d(np.log10(np.cumsum(np.random.randn(1000)) + 1000))
    assert_allclose(_hurst(p), 0.5, atol=0.1)


# This function also implicitly tests _efficient_welch
def test_freqs_power():
    """Test internal function for frequency power estimation."""
    # Create signal with different frequency components
    freqs = [1, 5, 12.8, 23.4, 40]  # in Hertz
    srate = 100.0
    time = np.arange(10 * srate) / srate
    signal = np.sum([np.sin(2 * np.pi * f * time) for f in freqs], axis=0)
    signal = np.atleast_2d(signal)

    # These frequencies should be present
    for f in freqs:
        assert_almost_equal(_freqs_power(signal, srate, [f]), 3 + 1/3.)

    # The function should sum the individual frequency  powers
    assert_almost_equal(_freqs_power(signal, srate, freqs),
                        len(freqs) * (3 + 1/3.))

    # These frequencies should not be present
    assert_almost_equal(_freqs_power(signal, srate, [2, 4, 13, 23, 35]), 0)

    # Insufficient sample rate to calculate this frequency
    assert_raises(ValueError, _freqs_power, signal, srate, [51])


def _baseline_signal():
    """Helper function to create the baseline signal"""
    signal = np.tile(np.sin(time), (n_epochs, n_channels, 1))
    noise = np.random.randn(n_epochs, n_channels, n_samples)
    return signal, noise


def _to_epochs(signal, noise):
    """Helper function to create the epochs object"""
    events = np.tile(np.arange(n_epochs)[:, np.newaxis], (1, 3))
    return mne.EpochsArray(signal + noise,
                           mne.create_info(n_channels, srate, 'eeg'),
                           events)


def test_find_bad_channels():
    """Test detecting bad channels through outlier detection"""
    signal, noise = _baseline_signal()

    # This channel has more noise
    noise[:, 0, :] *= 2

    # This channel does not correlate with the others
    signal[:, 1, :] = np.sin(time + 0.68)

    # This channel has excessive 50 Hz line noise
    signal[:, 2, :] += np.sin(50 * 2 * np.pi * time)

    # This channel has excessive 60 Hz line noise
    signal[:, 3, :] += 1.2 * np.sin(60 * 2 * np.pi * time)

    # This channel has a different noise signature (kurtosis)
    noise[:, 4, :] = 4 * np.random.rand(n_epochs, n_samples)

    # TODO: deviant hurst
    epochs = _to_epochs(signal, noise)
    bads = find_bad_channels(epochs, method_params={'max_iter': 1},
                             return_by_metric=True)
    assert_equal(bads, {
        'variance': ['0'],
        'correlation': ['1'],
        'line_noise': ['2', '3'],
        'kurtosis': ['4'],
        'hurst': ['2'],
    })

    # Test picks
    bads = find_bad_channels(epochs, return_by_metric=True,
                             picks=range(3, n_channels))
    assert_equal(bads['line_noise'], ['3'])


def test_find_bad_epochs():
    """Test detecting bad epochs through outlier detection"""
    signal, noise = _baseline_signal()

    # This epoch has more noise
    noise[0, :, :] *= 2

    # This epoch has some deviation
    signal[1, :, :] += 20

    # This epoch has a single spike across channels
    signal[2, :, 0] += 10

    epochs = _to_epochs(signal, noise)

    bads = find_bad_epochs(epochs, method_params={'max_iter': 1},
                           return_by_metric=True)
    assert_equal(bads, {
        'variance': [0],
        'deviation': [1],
        'amplitude': [0, 2],
    })

    # Test picks
    bads = find_bad_epochs(epochs, return_by_metric=True,
                           picks=range(3, n_channels))
    assert_equal(bads, {
        'variance': [0],
        'deviation': [1],
        'amplitude': [0, 2],
    })


def test_find_bad_channels_in_epochs():
    """Test detecting bad channels in each epoch through outlier detection"""
    signal, noise = _baseline_signal()

    # This channel/epoch combination has more noise
    noise[0, 0, :] *= 2

    # This channel/epoch combination has some deviation
    signal[1, 1, :] += 20

    # This channel/epoch combination has a single spike
    signal[2, 2, 0] += 10

    # This channel/epoch combination has excessive 50 Hz line noise
    signal[3, 3, :] += np.sin(50 * 2 * np.pi * time)

    epochs = _to_epochs(signal, noise)

    bads = find_bad_channels_in_epochs(epochs, return_by_metric=True,
                                       method_params={'thresh': 5})
    assert_equal(bads['variance'][0], ['0'])
    assert_equal(bads['deviation'][1], ['1'])
    assert_equal(bads['amplitude'][2], ['2'])
    assert_equal(bads['median_gradient'][0], ['0'])
    assert_equal(bads['line_noise'][3], ['3'])

    # Test picks
    bads = find_bad_channels_in_epochs(epochs, return_by_metric=True,
                                       method_params={'thresh': 5},
                                       picks=range(3, n_channels))
    assert_equal(bads['variance'][0], [])
    assert_equal(bads['deviation'][1], [])
    assert_equal(bads['amplitude'][2], [])
    assert_equal(bads['median_gradient'][0], [])
    assert_equal(bads['line_noise'][3], ['3'])
