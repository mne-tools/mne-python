import numpy as np
from pytest import raises
from numpy.testing import assert_array_equal, assert_allclose
from os import path as op
import warnings
import pickle
from itertools import product

import mne
from mne.io import read_raw_fif
from mne.utils import sum_squared, run_tests_if_main, _TempDir, requires_h5py
from mne.time_frequency import (csd_fourier, csd_multitaper,
                                csd_morlet, csd_array_fourier,
                                csd_array_multitaper, csd_array_morlet,
                                csd_epochs, csd_array, tfr_morlet,
                                CrossSpectralDensity, read_csd,
                                pick_channels_csd, psd_multitaper)
from mne.time_frequency.csd import _sym_mat_to_vector, _vector_to_sym_mat

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


def _make_csd():
    """Make a simple CrossSpectralDensity object."""
    frequencies = [1., 2., 3., 4.]
    n_freqs = len(frequencies)
    names = ['CH1', 'CH2', 'CH3']
    tmin, tmax = (0., 1.)
    data = np.arange(6. * n_freqs).reshape(n_freqs, 6).T
    return CrossSpectralDensity(data, names, frequencies, 1, tmin, tmax)


def test_csd():
    """Test constructing a CrossSpectralDensity."""
    csd = CrossSpectralDensity([1, 2, 3], ['CH1', 'CH2'], frequencies=1,
                               n_fft=1, tmin=0, tmax=1)
    assert_array_equal(csd._data, [[1], [2], [3]])  # Conversion to 2D array
    assert_array_equal(csd.frequencies, [1])  # Conversion to 1D array

    # Channels don't match
    raises(ValueError, CrossSpectralDensity, [1, 2, 3],
           ['CH1', 'CH2', 'Too many!'], tmin=0, tmax=1, frequencies=1, n_fft=1)
    raises(ValueError, CrossSpectralDensity, [1, 2, 3], ['too little'],
           tmin=0, tmax=1, frequencies=1, n_fft=1)

    # Frequencies don't match
    raises(ValueError, CrossSpectralDensity,
           [[1, 2], [3, 4], [5, 6]], ['CH1', 'CH2'],
           tmin=0, tmax=1, frequencies=1, n_fft=1)

    # Invalid dims
    raises(ValueError, CrossSpectralDensity, [[[1]]], ['CH1'], frequencies=1,
           n_fft=1, tmin=0, tmax=1)


def test_csd_repr():
    """Test string representation of CrossSpectralDensity."""
    csd = _make_csd()
    assert str(csd) == ('<CrossSpectralDensity  |  n_channels=3, time=0.0 to '
                        '1.0 s, frequencies=1.0, 2.0, 3.0, 4.0 Hz.>')

    assert str(csd.mean()) == ('<CrossSpectralDensity  |  n_channels=3, '
                               'time=0.0 to 1.0 s, frequencies=1.0-4.0 Hz.>')

    csd_binned = csd.mean(fmin=[1, 3], fmax=[2, 4])
    assert str(csd_binned) == ('<CrossSpectralDensity  |  n_channels=3, '
                               'time=0.0 to 1.0 s, frequencies=1.0-2.0, '
                               '3.0-4.0 Hz.>')

    csd_binned = csd.mean(fmin=[1, 2], fmax=[1, 4])
    assert str(csd_binned) == ('<CrossSpectralDensity  |  n_channels=3, '
                               'time=0.0 to 1.0 s, frequencies=1.0, 2.0-4.0 '
                               'Hz.>')

    csd_no_time = csd.copy()
    csd_no_time.tmin = None
    csd_no_time.tmax = None
    assert str(csd_no_time) == (
        '<CrossSpectralDensity  |  n_channels=3, time=unknown, '
        'frequencies=1.0, 2.0, 3.0, 4.0 Hz.>'
    )


def test_csd_mean():
    """Test averaging frequency bins of CrossSpectralDensity."""
    csd = _make_csd()

    # Test different ways to average across all frequencies
    avg = [[9], [10], [11], [12], [13], [14]]
    assert_array_equal(csd.mean()._data, avg)
    assert_array_equal(csd.mean(fmin=None, fmax=4)._data, avg)
    assert_array_equal(csd.mean(fmin=1, fmax=None)._data, avg)
    assert_array_equal(csd.mean(fmin=0, fmax=None)._data, avg)
    assert_array_equal(csd.mean(fmin=1, fmax=4)._data, avg)

    # Test averaging across frequency bins
    csd_binned = csd.mean(fmin=[1, 3], fmax=[2, 4])
    assert_array_equal(
        csd_binned._data,
        [[3, 15],
         [4, 16],
         [5, 17],
         [6, 18],
         [7, 19],
         [8, 20]],
    )

    csd_binned = csd.mean(fmin=[1, 3], fmax=[1, 4])
    assert_array_equal(
        csd_binned._data,
        [[0, 15],
         [1, 16],
         [2, 17],
         [3, 18],
         [4, 19],
         [5, 20]],
    )

    # This flag should be set after averaging
    assert csd.mean()._is_sum

    # Test construction of .frequency attribute
    assert csd.mean().frequencies == [[1, 2, 3, 4]]
    assert (csd.mean(fmin=[1, 3], fmax=[2, 4]).frequencies ==
            [[1, 2], [3, 4]])

    # Test invalid inputs
    raises(ValueError, csd.mean, fmin=1, fmax=[2, 3])
    raises(ValueError, csd.mean, fmin=[1, 2], fmax=[3])
    raises(ValueError, csd.mean, fmin=[1, 2], fmax=[1, 1])

    # Taking the mean twice should raise an error
    raises(RuntimeError, csd.mean().mean)


def test_csd_get_frequency_index():
    """Test the _get_frequency_index method of CrossSpectralDensity."""
    csd = _make_csd()

    assert csd._get_frequency_index(1) == 0
    assert csd._get_frequency_index(2) == 1
    assert csd._get_frequency_index(4) == 3

    assert csd._get_frequency_index(0.9) == 0
    assert csd._get_frequency_index(2.1) == 1
    assert csd._get_frequency_index(4.1) == 3

    # Frequency can be off by a maximum of 1
    raises(IndexError, csd._get_frequency_index, csd.frequencies[-1] + 1.0001)


def test_csd_pick_frequency():
    """Test the pick_frequency method of CrossSpectralDensity."""
    csd = _make_csd()

    csd2 = csd.pick_frequency(freq=2)
    assert csd2.frequencies == [2]
    assert_array_equal(
        csd2.get_data(),
        [[6, 7, 8],
         [7, 9, 10],
         [8, 10, 11]]
    )

    csd2 = csd.pick_frequency(index=1)
    assert csd2.frequencies == [2]
    assert_array_equal(
        csd2.get_data(),
        [[6, 7, 8],
         [7, 9, 10],
         [8, 10, 11]]
    )

    # Nonexistent frequency
    raises(IndexError, csd.pick_frequency, -1)

    # Nonexistent index
    raises(IndexError, csd.pick_frequency, index=10)

    # Invalid parameters
    raises(ValueError, csd.pick_frequency)
    raises(ValueError, csd.pick_frequency, freq=2, index=1)


def test_csd_get_data():
    """Test the get_data method of CrossSpectralDensity."""
    csd = _make_csd()

    # CSD matrix corresponding to 2 Hz.
    assert_array_equal(
        csd.get_data(frequency=2),
        [[6, 7, 8],
         [7, 9, 10],
         [8, 10, 11]]
    )

    # Mean CSD matrix
    assert_array_equal(
        csd.mean().get_data(),
        [[9, 10, 11],
         [10, 12, 13],
         [11, 13, 14]]
    )

    # Average across frequency bins, select bin
    assert_array_equal(
        csd.mean(fmin=[1, 3], fmax=[2, 4]).get_data(index=1),
        [[15, 16, 17],
         [16, 18, 19],
         [17, 19, 20]]
    )

    # Invalid inputs
    raises(ValueError, csd.get_data)
    raises(ValueError, csd.get_data, frequency=1, index=1)
    raises(IndexError, csd.get_data, frequency=15)
    raises(ValueError, csd.mean().get_data, frequency=1)
    raises(IndexError, csd.mean().get_data, index=15)


@requires_h5py
def test_csd_save():
    """Test saving and loading a CrossSpectralDensity."""
    csd = _make_csd()
    tempdir = _TempDir()
    fname = op.join(tempdir, 'csd.h5')
    csd.save(fname)
    csd2 = read_csd(fname)
    assert_array_equal(csd._data, csd2._data)
    assert csd.tmin == csd2.tmin
    assert csd.tmax == csd2.tmax
    assert csd.ch_names == csd2.ch_names
    assert csd.frequencies == csd2.frequencies
    assert csd._is_sum == csd2._is_sum


def test_csd_pickle():
    """Test pickling and unpickling a CrossSpectralDensity."""
    csd = _make_csd()
    tempdir = _TempDir()
    fname = op.join(tempdir, 'csd.dat')
    with open(fname, 'wb') as f:
        pickle.dump(csd, f)
    with open(fname, 'rb') as f:
        csd2 = pickle.load(f)
    assert_array_equal(csd._data, csd2._data)
    assert csd.tmin == csd2.tmin
    assert csd.tmax == csd2.tmax
    assert csd.ch_names == csd2.ch_names
    assert csd.frequencies == csd2.frequencies
    assert csd._is_sum == csd2._is_sum


def test_pick_channels_csd():
    """Test selecting channels from a CrossSpectralDensity."""
    csd = _make_csd()
    csd = pick_channels_csd(csd, ['CH1', 'CH3'])
    assert csd.ch_names == ['CH1', 'CH3']
    assert_array_equal(csd._data, [[0, 6, 12, 18],
                                   [2, 8, 14, 20],
                                   [5, 11, 17, 23]])


def test_sym_mat_to_vector():
    """Test converting between vectors and symmetric matrices."""
    mat = np.array([[0, 1, 2, 3],
                    [1, 4, 5, 6],
                    [2, 5, 7, 8],
                    [3, 6, 8, 9]])
    assert_array_equal(_sym_mat_to_vector(mat),
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    vec = np.arange(10)
    assert_array_equal(_vector_to_sym_mat(vec),
                       [[0, 1, 2, 3],
                        [1, 4, 5, 6],
                        [2, 5, 7, 8],
                        [3, 6, 8, 9]])

    # Test complex values: diagonals should be complex conjugates
    comp_vec = np.arange(3) + 1j
    assert_array_equal(_vector_to_sym_mat(comp_vec),
                       [[0. + 0.j,  1. + 1.j],
                        [1. - 1.j,  2. + 0.j]])

    # Test preservation of data type
    assert _sym_mat_to_vector(mat.astype(np.int8)).dtype == np.int8
    assert _vector_to_sym_mat(vec.astype(np.int8)).dtype == np.int8
    assert _sym_mat_to_vector(mat.astype(np.float16)).dtype == np.float16
    assert _vector_to_sym_mat(vec.astype(np.float16)).dtype == np.float16


def _generate_coherence_data():
    """Create an epochs object with coherence at 22Hz between channels 1 and 3.

    A base 10 Hz sine wave is generated for all channels, but with different
    phases, which means no actual coherence. A  22Hz sine wave is laid on top
    for channels 1 and 3, with the same phase, so there is coherence between
    these channels.
    """
    ch_names = ['CH1', 'CH2', 'CH3']
    sfreq = 50.
    info = mne.create_info(ch_names, sfreq, 'eeg')
    tstep = 1. / sfreq
    n_samples = int(10 * sfreq)  # 10 seconds of data
    times = np.arange(n_samples) * tstep
    events = np.array([[0, 1, 1]])  # one event

    # Phases for the signals
    phases = np.arange(info['nchan']) * 0.3 * np.pi

    # Generate 10 Hz sine waves with different phases
    signal = np.vstack([np.sin(times * 2 * np.pi * 10 + phase)
                        for phase in phases])

    data = np.zeros((1, info['nchan'], n_samples))
    data[0, :, :] = signal

    # Generate 22Hz sine wave at the first and last electrodes with the same
    # phase.
    signal = np.sin(times * 2 * np.pi * 22)
    data[0, [0, -1], :] += signal

    return mne.EpochsArray(data, info, events, baseline=(0, times[-1]))


def _test_csd_matrix(csd):
    """Perform a suite of tests on a CSD matrix."""
    # Check shape of the CSD matrix
    n_chan = len(csd.ch_names)
    assert n_chan == 3
    assert csd.ch_names == ['CH1', 'CH2', 'CH3']
    n_freqs = len(csd.frequencies)
    assert n_freqs == 3
    assert csd._data.shape == (6, 3)  # Only upper triangle of CSD matrix

    # Extract CSD ndarrays. Diagonals are PSDs.
    csd_10 = csd.get_data(index=0)
    csd_22 = csd.get_data(index=2)
    power_10 = np.diag(csd_10)
    power_22 = np.diag(csd_22)

    # Check if the CSD matrices are hermitian
    assert np.all(np.tril(csd_10).T.conj() == np.triu(csd_10))
    assert np.all(np.tril(csd_22).T.conj() == np.triu(csd_22))

    # Off-diagonals show phase difference
    assert np.abs(csd_10[0, 1].imag) > 0.4
    assert np.abs(csd_10[0, 2].imag) > 0.4
    assert np.abs(csd_10[1, 2].imag) > 0.4

    # No phase differences at 22 Hz
    assert np.all(np.abs(csd_22[0, 2].imag) < 1E-3)

    # Test CSD between the two channels that have a 20Hz signal and the one
    # that has only a 10 Hz signal
    assert np.abs(csd_22[0, 2]) > np.abs(csd_22[0, 1])
    assert np.abs(csd_22[0, 2]) > np.abs(csd_22[1, 2])

    # Check that electrodes/frequency combinations with signal have more
    # power than frequencies without signal.
    power_15 = np.diag(csd.get_data(index=1))
    assert np.all(power_10 > power_15)
    assert np.all(power_22[[0, -1]] > power_15[[0, -1]])


def _test_fourier_multitaper_parameters(epochs, csd_epochs, csd_array):
    """Parameter tests for csd_*_fourier and csd_*_multitaper."""
    with warnings.catch_warnings(record=True):  # deprecation
        raises(ValueError, csd_epochs, epochs, fmin=20, fmax=10)
        raises(ValueError, csd_array, epochs._data, epochs.info['sfreq'],
               epochs.tmin, fmin=20, fmax=10)
        raises(ValueError, csd_epochs, epochs, fmin=20, fmax=20.1)
        raises(ValueError, csd_array, epochs._data, epochs.info['sfreq'],
               epochs.tmin, fmin=20, fmax=20.1)
        raises(ValueError, csd_epochs, epochs, tmin=0.15, tmax=0.1)
        raises(ValueError, csd_array, epochs._data, epochs.info['sfreq'],
               epochs.tmin, tmin=0.15, tmax=0.1)
        raises(ValueError, csd_epochs, epochs, tmin=-1, tmax=10)
        raises(ValueError, csd_array, epochs._data, epochs.info['sfreq'],
               epochs.tmin, tmin=-1, tmax=10)
        raises(ValueError, csd_epochs, epochs, tmin=10, tmax=11)
        raises(ValueError, csd_array, epochs._data, epochs.info['sfreq'],
               epochs.tmin, tmin=10, tmax=11)

    # Test checks for data types and sizes
    diff_types = [np.random.randn(3, 5), "error"]
    err_data = [np.random.randn(3, 5), np.random.randn(2, 4)]
    with warnings.catch_warnings(record=True):  # deprecation
        raises(ValueError, csd_array, err_data, sfreq=1)
        raises(ValueError, csd_array, diff_types, sfreq=1)
        raises(ValueError, csd_array, np.random.randn(3), sfreq=1)


def test_csd_fourier():
    """Test computing cross-spectral density using short-term Fourier."""
    epochs = _generate_coherence_data()
    sfreq = epochs.info['sfreq']
    _test_fourier_multitaper_parameters(epochs, csd_fourier, csd_array_fourier)

    # Compute CSDs using various parameters
    times = [(None, None), (1, 9)]
    as_arrays = [False, True]
    parameters = product(times, as_arrays)
    for (tmin, tmax), as_array in parameters:
        if as_array:
            csd = csd_array_fourier(epochs.get_data(), sfreq, epochs.tmin,
                                    fmin=9, fmax=23, tmin=tmin, tmax=tmax,
                                    ch_names=epochs.ch_names)
        else:
            csd = csd_fourier(epochs, fmin=9, fmax=23, tmin=tmin, tmax=tmax)

        if tmin is None and tmax is None:
            assert csd.tmin == 0 and csd.tmax == 9.98
        else:
            assert csd.tmin == tmin and csd.tmax == tmax
        csd = csd.mean([9.9, 14.9, 21.9], [10.1, 15.1, 22.1])
        _test_csd_matrix(csd)

    # For the next test, generate a simple sine wave with a known power
    times = np.arange(20 * sfreq) / sfreq  # 20 seconds of signal
    signal = np.sin(2 * np.pi * 10 * times)[None, None, :]  # 10 Hz wave
    signal_power_per_sample = sum_squared(signal) / len(times)

    # Power per sample should not depend on time window length
    for tmax in [12, 18]:
        t_mask = (times <= tmax)
        n_samples = sum(t_mask)

        # Power per sample should not depend on number of FFT points
        for add_n_fft in [0, 30]:
            n_fft = n_samples + add_n_fft
            csd = csd_array_fourier(signal, sfreq, tmax=tmax,
                                    n_fft=n_fft).sum().get_data()
            first_samp = csd[0, 0]
            fourier_power_per_sample = np.abs(first_samp) * sfreq / n_fft
            assert abs(signal_power_per_sample -
                       fourier_power_per_sample) < 0.001


def test_csd_multitaper():
    """Test computing cross-spectral density using multitapers."""
    epochs = _generate_coherence_data()
    sfreq = epochs.info['sfreq']
    _test_fourier_multitaper_parameters(epochs, csd_multitaper,
                                        csd_array_multitaper)

    # Compute CSDs using various parameters
    times = [(None, None), (1, 9)]
    as_arrays = [False, True]
    adaptives = [False, True]
    parameters = product(times, as_arrays, adaptives)
    for (tmin, tmax), as_array, adaptive in parameters:
        if as_array:
            csd = csd_array_multitaper(epochs.get_data(), sfreq, epochs.tmin,
                                       adaptive=adaptive, fmin=9, fmax=23,
                                       tmin=tmin, tmax=tmax,
                                       ch_names=epochs.ch_names)
        else:
            csd = csd_multitaper(epochs, adaptive=adaptive, fmin=9, fmax=23,
                                 tmin=tmin, tmax=tmax)
        if tmin is None and tmax is None:
            assert csd.tmin == 0 and csd.tmax == 9.98
        else:
            assert csd.tmin == tmin and csd.tmax == tmax
        csd = csd.mean([9.9, 14.9, 21.9], [10.1, 15.1, 22.1])
        _test_csd_matrix(csd)

    # Test equivalence with PSD
    psd, psd_freqs = psd_multitaper(epochs, fmin=1e-3,
                                    normalization='full')  # omit DC
    csd = csd_multitaper(epochs)
    assert_allclose(psd_freqs, csd.frequencies)
    csd = np.array([np.diag(csd.get_data(index=ii))
                    for ii in range(len(csd))]).T
    assert_allclose(psd[0], csd)

    # For the next test, generate a simple sine wave with a known power
    times = np.arange(20 * sfreq) / sfreq  # 20 seconds of signal
    signal = np.sin(2 * np.pi * 10 * times)[None, None, :]  # 10 Hz wave
    signal_power_per_sample = sum_squared(signal) / len(times)

    # Power per sample should not depend on time window length
    for tmax in [12, 18]:
        t_mask = (times <= tmax)
        n_samples = sum(t_mask)
        n_fft = len(times)

        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 5]:
            bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
            csd_mt = csd_array_multitaper(signal, sfreq, tmax=tmax,
                                          bandwidth=bandwidth,
                                          n_fft=n_fft).sum().get_data()
            mt_power_per_sample = np.abs(csd_mt[0, 0]) * sfreq / n_fft
            assert abs(signal_power_per_sample - mt_power_per_sample) < 0.001


def test_csd_morlet():
    """Test computing cross-spectral density using Morlet wavelets."""
    epochs = _generate_coherence_data()
    sfreq = epochs.info['sfreq']

    # Compute CSDs by a variety of methods
    freqs = [10, 15, 22]
    n_cycles = [20, 30, 44]
    times = [(None, None), (1, 9)]
    as_arrays = [False, True]
    parameters = product(times, as_arrays)
    for (tmin, tmax), as_array in parameters:
        if as_array:
            csd = csd_array_morlet(epochs.get_data(), sfreq, freqs,
                                   t0=epochs.tmin, n_cycles=n_cycles,
                                   tmin=tmin, tmax=tmax,
                                   ch_names=epochs.ch_names)
        else:
            csd = csd_morlet(epochs, frequencies=freqs, n_cycles=n_cycles,
                             tmin=tmin, tmax=tmax)
        if tmin is None and tmax is None:
            assert csd.tmin == 0 and csd.tmax == 9.98
        else:
            assert csd.tmin == tmin and csd.tmax == tmax
        _test_csd_matrix(csd)

    # CSD diagonals should contain PSD
    tfr = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    power = np.mean(tfr.data, 2)
    csd = csd_morlet(epochs, frequencies=freqs, n_cycles=n_cycles)
    assert_allclose(csd._data[[0, 3, 5]] * sfreq, power)

    # Test using plain convolution instead of FFT
    csd = csd_morlet(epochs, frequencies=freqs, n_cycles=n_cycles,
                     use_fft=False)
    assert_allclose(csd._data[[0, 3, 5]] * sfreq, power)

    # Test baselining warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs_nobase = epochs.copy()
        epochs_nobase.baseline = None
        epochs_nobase.info['highpass'] = 0
        csd = csd_morlet(epochs_nobase, frequencies=[10], decim=20)
    assert len(w) == 1


###############################################################################
# Old CSD tests. These can be removed for version 0.17.

def _get_real_data():
    """Get some real MEG data."""
    raw = read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)[0:100]

    # Read raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set picks
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

    # Read several epochs
    event_id, tmin, tmax = 1, -0.2, 0.5
    return mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                      preload=True, reject=dict(grad=4000e-13, mag=4e-12))


def _generate_simple_data():
    """Create an epochs object with one epoch of 10 Hz sine wave."""
    event_id, tmin, tmax = 1, 0.0, 1.0
    raw = read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)[0:5]
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=[0],
                        preload=True, reject=dict(grad=4000e-13))
    freq = 10
    epochs._data = np.sin(2 * np.pi * freq *
                          epochs.times)[None, None, :]
    return epochs


def test_csd_epochs():
    """Test computing cross-spectral density from epochs."""
    epochs = _get_real_data()

    # Check that wrong parameters are recognized
    with warnings.catch_warnings(record=True):  # deprecation
        raises(ValueError, csd_epochs, epochs, mode='notamode')
        raises(ValueError, csd_epochs, epochs, fmin=20, fmax=10)
        raises(ValueError, csd_epochs, epochs, fmin=20, fmax=20.1)
        raises(ValueError, csd_epochs, epochs, tmin=0.15, tmax=0.1)
        raises(ValueError, csd_epochs, epochs, tmin=0, tmax=10)
        raises(ValueError, csd_epochs, epochs, tmin=10, tmax=11)

    # Test deprecation warning
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        csd_mt = csd_epochs(epochs, mode='multitaper', fmin=8, fmax=12,
                            tmin=0.04, tmax=0.15)
    assert len([w for w in ws
                if issubclass(w.category, DeprecationWarning)]) == 1

    with warnings.catch_warnings(record=True):  # deprecation
        csd_fourier = csd_epochs(epochs, mode='fourier', fmin=8, fmax=12,
                                 tmin=0.04, tmax=0.15)

    # Check shape of the CSD matrix
    n_chan = len(csd_mt.ch_names)
    csd_mt_data = csd_mt.get_data()
    csd_fourier_data = csd_fourier.get_data()
    assert csd_mt_data.shape == (n_chan, n_chan)
    assert csd_fourier_data.shape == (n_chan, n_chan)

    # Check if the CSD matrix is hermitian
    assert_array_equal(np.tril(csd_mt_data).T.conj(),
                       np.triu(csd_mt_data))
    assert_array_equal(np.tril(csd_fourier_data).T.conj(),
                       np.triu(csd_fourier_data))

    # Computing induced power for comparison
    epochs.crop(tmin=0.04, tmax=0.15)
    tfr = tfr_morlet(epochs, freqs=[10], n_cycles=0.6, return_itc=False)
    power = np.mean(tfr.data, 2)

    # Maximum PSD should occur for specific channel
    max_ch_power = power.argmax()
    max_ch_mt = csd_mt_data.diagonal().argmax()
    max_ch_fourier = csd_fourier_data.diagonal().argmax()
    assert max_ch_mt == max_ch_power
    assert max_ch_fourier == max_ch_power

    # Maximum CSD should occur for specific channel
    ch_csd_mt = np.abs(csd_mt_data[max_ch_power])
    ch_csd_mt[max_ch_power] = 0.
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = np.abs(csd_fourier_data[max_ch_power])
    ch_csd_fourier[max_ch_power] = 0.
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert max_ch_csd_mt == max_ch_csd_fourier

    # Check a list of CSD matrices is returned for multiple frequencies within
    # a given range when fsum=False
    with warnings.catch_warnings(record=True):  # deprecation
        csd_fsum = csd_epochs(epochs, mode='fourier', fmin=8, fmax=20,
                              fsum=True)
        csds = csd_epochs(epochs, mode='fourier', fmin=8, fmax=20, fsum=False)
    assert len(csd_fsum.frequencies) == 1
    assert len(csds.frequencies) == 2
    assert_array_equal(csd_fsum.frequencies[0], csds.frequencies)

    csd_sum = csds._data.sum(axis=1, keepdims=True)
    assert_array_equal(csd_fsum._data, csd_sum)


def test_csd_epochs_on_artificial_data():
    """Test computing CSD on artificial data."""
    epochs = _generate_simple_data()
    sfreq = epochs.info['sfreq']

    # Computing signal power in the time domain
    signal_power = sum_squared(epochs._data)
    signal_power_per_sample = signal_power / len(epochs.times)

    # Computing signal power in the frequency domain
    with warnings.catch_warnings(record=True):  # deprecation
        csd_fourier = csd_epochs(epochs, mode='fourier').get_data()
        csd_mt = csd_epochs(epochs, mode='multitaper').get_data()

    fourier_power = np.abs(csd_fourier[0, 0]) * sfreq
    mt_power = np.abs(csd_mt[0, 0]) * sfreq
    assert abs(fourier_power - signal_power) <= 0.5
    assert abs(mt_power - signal_power) <= 1

    # Power per sample should not depend on time window length
    for tmax in [0.2, 0.8]:
        t_mask = (epochs.times >= 0) & (epochs.times <= tmax)
        n_samples = sum(t_mask)
        for add_n_fft in [0, 30]:
            n_fft = n_samples + add_n_fft
            with warnings.catch_warnings(record=True):  # deprecation
                csd_fourier = csd_epochs(
                    epochs, mode='fourier', tmin=None, tmax=tmax, fmin=0,
                    fmax=np.inf, n_fft=n_fft
                ).get_data()
            first_samp = csd_fourier[0, 0]
            fourier_power_per_sample = np.abs(first_samp) * sfreq / n_fft
            assert abs(signal_power_per_sample -
                       fourier_power_per_sample) < 0.003
        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 5]:
            mt_bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
            with warnings.catch_warnings(record=True):  # deprecation
                csd_mt = csd_epochs(
                    epochs, mode='multitaper', tmin=None, tmax=tmax, fmin=0,
                    fmax=np.inf, mt_bandwidth=mt_bandwidth, n_fft=n_fft
                ).get_data()
            mt_power_per_sample = np.abs(csd_mt[0, 0]) * sfreq / n_fft
            # The estimate of power gets worse for small time windows when more
            # tapers are used
            if n_tapers == 5 and tmax == 0.2:
                delta = 0.05
            else:
                delta = 0.004
            assert abs(signal_power_per_sample -
                       mt_power_per_sample) < delta


def test_compute_csd():
    """Test computing cross-spectral density from ndarray."""
    epochs = _get_real_data()

    tmin = 0.04
    tmax = 0.15
    tmp = np.where(np.logical_and(epochs.times >= tmin,
                                  epochs.times <= tmax))[0]

    picks_meeg = mne.pick_types(epochs[0].info, meg=True, eeg=True, eog=False,
                                ref_meg=False, exclude='bads')

    epochs_data = [e[picks_meeg][:, tmp].copy() for e in epochs]
    n_trials = len(epochs)
    n_series = len(picks_meeg)
    X = np.concatenate(epochs_data, axis=0)
    X = np.reshape(X, (n_trials, n_series, -1))
    X_list = epochs_data

    sfreq = epochs.info['sfreq']

    # Check data types and sizes are checked
    diff_types = [np.random.randn(3, 5), "error"]
    err_data = [np.random.randn(3, 5), np.random.randn(2, 4)]
    with warnings.catch_warnings(record=True):  # deprecation
        raises(ValueError, csd_array, err_data, sfreq)
        raises(ValueError, csd_array, diff_types, sfreq)
        raises(ValueError, csd_array, np.random.randn(3), sfreq)

        # Check that wrong parameters are recognized
        raises(ValueError, csd_array, X, sfreq, mode='notamode')
        raises(ValueError, csd_array, X, sfreq, fmin=20, fmax=10)
        raises(ValueError, csd_array, X, sfreq, fmin=20, fmax=20.1)

    # Test deprecation warning
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        csd_mt = csd_array(X, sfreq, mode='multitaper', fmin=8, fmax=12)
    assert len([w for w in ws
                if issubclass(w.category, DeprecationWarning)]) == 1

    with warnings.catch_warnings(record=True):  # deprecation
        csd_fourier = csd_array(X, sfreq, mode='fourier', fmin=8, fmax=12)

    # Test as list too
    with warnings.catch_warnings(record=True):  # deprecation
        csd_mt_list = csd_array(X_list, sfreq, mode='multitaper',
                                fmin=8, fmax=12)
        csd_fourier_list = csd_array(X_list, sfreq, mode='fourier', fmin=8,
                                     fmax=12)

    assert_array_equal(csd_mt._data, csd_mt_list._data)
    assert_array_equal(csd_fourier._data, csd_fourier_list._data)
    assert_array_equal(csd_mt.frequencies, csd_mt_list.frequencies)
    assert_array_equal(csd_fourier.frequencies, csd_fourier_list.frequencies)

    # Check shape of the CSD matrix
    n_chan = len(epochs.ch_names)
    csd_mt_data = csd_mt.get_data()
    csd_fourier_data = csd_fourier.get_data()
    assert csd_mt_data.shape == (n_chan, n_chan)
    assert csd_fourier_data.shape == (n_chan, n_chan)

    # Check if the CSD matrix is hermitian
    assert_array_equal(np.tril(csd_mt_data).T.conj(),
                       np.triu(csd_mt_data))
    assert_array_equal(np.tril(csd_fourier_data).T.conj(),
                       np.triu(csd_fourier_data))

    # Computing induced power for comparison
    epochs.crop(tmin=0.04, tmax=0.15)
    tfr = tfr_morlet(epochs, freqs=[10], n_cycles=0.6, return_itc=False)
    power = np.mean(tfr.data, 2)

    # Maximum PSD should occur for specific channel
    max_ch_power = power.argmax()
    max_ch_mt = csd_mt_data.diagonal().argmax()
    max_ch_fourier = csd_fourier_data.diagonal().argmax()
    assert max_ch_mt == max_ch_power
    assert max_ch_fourier == max_ch_power

    # Maximum CSD should occur for specific channel
    ch_csd_mt = np.abs(csd_mt_data[max_ch_power])
    ch_csd_mt[max_ch_power] = 0.
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = np.abs(csd_fourier_data[max_ch_power])
    ch_csd_fourier[max_ch_power] = 0.
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert max_ch_csd_mt == max_ch_csd_fourier

    # Check a list of CSD matrices is returned for multiple frequencies within
    # a given range when fsum=False
    with warnings.catch_warnings(record=True):  # deprecation
        csd_fsum = csd_array(X, sfreq, mode='fourier', fmin=8, fmax=20,
                             fsum=True)
        csds = csd_array(X, sfreq, mode='fourier', fmin=8, fmax=20,
                         fsum=False)

    assert csds._data.shape[1] == 2
    assert len(csds.frequencies) == 2
    assert_array_equal(csd_fsum.frequencies[0], csds.frequencies)
    assert_array_equal(csd_fsum._data, csds._data.sum(axis=1, keepdims=True))


def test_csd_on_artificial_data():
    """Test computing CSD on artificial data. """
    # Ignore deprecation warnings for this test
    epochs = _generate_simple_data()
    sfreq = epochs.info['sfreq']

    # Computing signal power in the time domain
    signal_power = sum_squared(epochs._data)
    signal_power_per_sample = signal_power / len(epochs.times)

    # Computing signal power in the frequency domain
    with warnings.catch_warnings(record=True):  # deprecation
        csd_mt = csd_array(epochs._data, sfreq, mode='multitaper').get_data()
        csd_fourier = csd_array(epochs._data, sfreq, mode='fourier').get_data()

    fourier_power = np.abs(csd_fourier[0, 0]) * sfreq
    mt_power = np.abs(csd_mt[0, 0]) * sfreq
    assert abs(fourier_power - signal_power) <= 0.5
    assert abs(mt_power - signal_power) <= 1

    # Power per sample should not depend on time window length
    for tmax in [0.2, 0.8]:
        tslice = np.where(epochs.times <= tmax)[0]

        for add_n_fft in [0, 30]:
            t_mask = (epochs.times >= 0) & (epochs.times <= tmax)
            n_samples = sum(t_mask)
            n_fft = n_samples + add_n_fft

            with warnings.catch_warnings(record=True):  # deprecation
                csd_fourier = csd_array(epochs._data[:, :, tslice], sfreq,
                                        mode='fourier', fmin=0, fmax=np.inf,
                                        n_fft=n_fft).get_data()

            first_samp = csd_fourier[0, 0]
            fourier_power_per_sample = np.abs(first_samp) * sfreq / n_fft
            assert abs(signal_power_per_sample -
                       fourier_power_per_sample) < 0.003
        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 5]:
            mt_bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
            with warnings.catch_warnings(record=True):  # deprecation
                csd_mt = csd_array(
                    epochs._data[:, :, tslice], sfreq,
                    mt_bandwidth=mt_bandwidth, n_fft=n_fft
                ).get_data()
            mt_power_per_sample = np.abs(csd_mt[0, 0]) * sfreq / n_fft
            # The estimate of power gets worse for small time windows when more
            # tapers are used
            if n_tapers == 5 and tmax == 0.2:
                delta = 0.05
            else:
                delta = 0.004
            assert abs(signal_power_per_sample -
                       mt_power_per_sample) < delta


run_tests_if_main()
