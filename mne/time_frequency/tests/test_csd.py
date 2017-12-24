# -*- coding: utf-8 -*-
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Susanna Aro <susanna.aro@aalto.fi>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)
from os import path as op
import warnings
import pickle
from itertools import product

from pytest import raises
from numpy.testing import assert_array_equal
import numpy as np

import mne
from mne.utils import run_tests_if_main, _TempDir, requires_h5py
from mne.time_frequency import (csd_epochs, csd_array, CrossSpectralDensity,
                                read_csd, pick_channels_csd)
from mne.time_frequency.csd import _sym_mat_to_vector, _vector_to_sym_mat

warnings.simplefilter('always')


def _get_data():
    """Create an epochs object with some simulated data."""
    ch_names = ['CH1', 'CH2', 'CH3']
    sfreq = 50.
    info = mne.create_info(ch_names, sfreq, 'eeg')
    tstep = 1. / sfreq
    n_samples = int(10 * sfreq)
    times = np.arange(n_samples) * tstep
    events = np.array([[0, 1, 1]])

    # Phases for the signals
    phases = np.arange(info['nchan']) * 0.3 * np.pi

    # Generate 10 Hz sine waves with different phases
    signal = np.vstack([
        np.sin(times * 2 * np.pi * 10 + phase)
        for phase in phases
    ])

    data = np.zeros((1, info['nchan'], n_samples))
    data[0, :, :] = signal

    # Generate 22Hz sine wave at the first and last electrodes with the same
    # phase.
    signal = np.sin(times * 2 * np.pi * 22)
    data[0, [0, -1], :] += signal

    epochs = mne.EpochsArray(data, info, events, baseline=(0, times[-1]))

    return epochs


def _make_csd():
    """Make a simple CrossSpectralDensity object."""
    frequencies = [1., 2., 3., 4.]
    n_freqs = len(frequencies)
    names = ['CH1', 'CH2', 'CH3']
    tmin, tmax = (0., 1.)
    data = np.arange(6. * n_freqs).reshape(n_freqs, 6).T
    return CrossSpectralDensity(data, names, tmin, tmax, frequencies, 1)


def test_csd():
    """Test constructing a CrossSpectralDensity."""
    csd = CrossSpectralDensity([1, 2, 3], ['CH1', 'CH2'], tmin=0, tmax=1,
                               frequencies=1, n_fft=1)
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
    raises(ValueError, CrossSpectralDensity, [[[1]]], ['CH1'], tmin=0, tmax=1,
           frequencies=1, n_fft=1)


def test_csd_repr():
    """Test string representation of CrossSpectralDensity."""
    csd = _make_csd()
    assert str(csd) == ('<CrossSpectralDensity  |  n_series=3, time=0.0 to 1.0'
                        ' s, frequencies=1.0, 2.0, 3.0, 4.0 Hz.>')

    assert str(csd.mean()) == ('<CrossSpectralDensity  |  n_series=3, time=0.0'
                               ' to 1.0 s, frequencies=1.0-4.0 Hz.>')

    csd_binned = csd.mean(fmin=[1, 3], fmax=[2, 4])
    assert str(csd_binned) == ('<CrossSpectralDensity  |  n_series=3, time=0.0'
                               ' to 1.0 s, frequencies=1.0-2.0, 3.0-4.0 Hz.>')

    csd_binned = csd.mean(fmin=[1, 2], fmax=[1, 4])
    assert str(csd_binned) == ('<CrossSpectralDensity  |  n_series=3, time=0.0'
                               ' to 1.0 s, frequencies=1.0, 2.0-4.0 Hz.>')


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
    assert csd.mean().is_sum

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
        csd2.get_matrix(),
        [[6, 7, 8],
         [7, 9, 10],
         [8, 10, 11]]
    )

    csd2 = csd.pick_frequency(index=1)
    assert csd2.frequencies == [2]
    assert_array_equal(
        csd2.get_matrix(),
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


def test_csd_get_matrix():
    """Test the get_matrix method of CrossSpectralDensity."""
    csd = _make_csd()

    # CSD matrix corresponding to 2 Hz.
    assert_array_equal(
        csd.get_matrix(frequency=2),
        [[6, 7, 8],
         [7, 9, 10],
         [8, 10, 11]]
    )

    # Mean CSD matrix
    assert_array_equal(
        csd.mean().get_matrix(),
        [[9, 10, 11],
         [10, 12, 13],
         [11, 13, 14]]
    )

    # Average across frequency bins, select bin
    assert_array_equal(
        csd.mean(fmin=[1, 3], fmax=[2, 4]).get_matrix(index=1),
        [[15, 16, 17],
         [16, 18, 19],
         [17, 19, 20]]
    )

    # Invalid inputs
    raises(ValueError, csd.get_matrix)
    raises(ValueError, csd.get_matrix, frequency=1, index=1)
    raises(IndexError, csd.get_matrix, frequency=15)
    raises(ValueError, csd.mean().get_matrix, frequency=1)
    raises(IndexError, csd.mean().get_matrix, index=15)


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
    assert csd.names == csd2.names
    assert csd.frequencies == csd2.frequencies
    assert csd.is_sum == csd2.is_sum


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
    assert csd.names == csd2.names
    assert csd.frequencies == csd2.frequencies
    assert csd.is_sum == csd2.is_sum


def test_pick_channels_csd():
    """Test selecting channels from a CrossSpectralDensity."""
    csd = _make_csd()
    csd = pick_channels_csd(csd, ['CH1', 'CH3'])
    assert csd.names == ['CH1', 'CH3']
    assert_array_equal(csd._data, [[0, 6, 12, 18],
                                   [2, 8, 14, 20],
                                   [5, 11, 17, 23]])


def _test_csd_on_artificial_data(csd):
    """Perform a suite of tests on a CSD matrix."""
    # Check shape of the CSD matrix
    n_chan = len(csd.names)
    n_freqs = len(csd.frequencies)
    assert n_chan == 3
    assert n_freqs == 3
    assert csd._data.shape == (6, 3)  # Only upper triangle of CSD matrix

    # Extract CSD ndarrays. Diagonals are PSDs.
    csd_10 = csd.get_matrix(index=0)
    csd_22 = csd.get_matrix(index=2)
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
    power_15 = np.diag(csd.get_matrix(index=1))
    assert np.all(power_10 > power_15)
    assert np.all(power_22[[0, -1]] > power_15[[0, -1]])


def test_csd_epochs():
    """Test computing cross-spectral density from epochs."""
    epochs = _get_data()

    # Check that wrong parameters are recognized
    raises(ValueError, csd_epochs, epochs, mode='notamode')
    raises(ValueError, csd_epochs, epochs, mode='fourier', fmin=20, fmax=10)
    raises(ValueError, csd_epochs, epochs, mode='fourier', fmin=20, fmax=20.1)
    raises(ValueError, csd_epochs, epochs, mode='fourier', tmin=0.15, tmax=0.1)
    raises(ValueError, csd_epochs, epochs, mode='fourier', tmin=-1, tmax=10)
    raises(ValueError, csd_epochs, epochs, mode='fourier', tmin=10, tmax=11)
    raises(ValueError, csd_epochs, epochs, mode='cwt_morlet', frequencies=None)

    # Check data types and sizes are checked
    diff_types = [np.random.randn(3, 5), "error"]
    err_data = [np.random.randn(3, 5), np.random.randn(2, 4)]
    raises(ValueError, csd_array, err_data, sfreq=1000)
    raises(ValueError, csd_array, diff_types, sfreq=1000)
    raises(ValueError, csd_array, np.random.randn(3), sfreq=1000)

    # Compute CSDs by a variety of methods
    freqs = [10, 15, 22]
    n_cycles = [20, 30, 44]
    modes = ['cwt_morlet', 'multitaper', 'fourier']
    times = [(None, None), (1, 9)]
    as_arrays = [False, True]
    mt_adaptives = [False, True]
    parameters = product(modes, times, as_arrays, mt_adaptives)
    for mode, (tmin, tmax), as_array, mt_adaptive in parameters:
        if mode != 'multitaper' and mt_adaptive:
            continue  # Skip this combination of parameters

        if as_array:
            csd = csd_array(epochs.get_data(), epochs.info['sfreq'],
                            epochs.tmin, mode=mode, mt_adaptive=mt_adaptive,
                            frequencies=freqs, fsum=False,
                            cwt_n_cycles=n_cycles, fmin=9, fmax=23, tmin=tmin,
                            tmax=tmax)
        else:
            csd = csd_epochs(epochs, mode=mode, mt_adaptive=mt_adaptive,
                             frequencies=freqs, fsum=False,
                             cwt_n_cycles=n_cycles, fmin=9, fmax=23, tmin=tmin,
                             tmax=tmax)
            assert csd.names == ['CH1', 'CH2', 'CH3']

        # Narrow down the CSD to three frequency bands: 10, 15 and 22 Hz
        if mode != 'cwt_morlet':
            csd = csd.mean([9.9, 14.9, 21.9], [10.1, 15.1, 22.1])

        _test_csd_on_artificial_data(csd)

    # Test summing across frequencies
    csd1 = csd_epochs(epochs, mode='cwt_morlet', frequencies=freqs, fsum=False,
                      cwt_n_cycles=n_cycles)
    csd2 = csd_epochs(epochs, mode='cwt_morlet', frequencies=freqs, fsum=True,
                      cwt_n_cycles=n_cycles)
    assert_array_equal(csd1.sum()._data, csd2._data)

    # Test baselining warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs_nobase = epochs.copy()
        epochs_nobase.baseline = None
        epochs_nobase.info['highpass'] = 0
        csd = csd_epochs(epochs_nobase, mode='cwt_morlet', frequencies=[10],
                         decim=20)
    assert len(w) == 1


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


run_tests_if_main()
