import numpy as np
from nose.tools import assert_raises, assert_equal, assert_true
from numpy.testing import assert_array_equal
from os import path as op
import warnings

import mne

from mne.io import read_raw_fif
from mne.utils import sum_squared, run_tests_if_main
from mne.time_frequency import csd_epochs, csd_array, tfr_morlet

warnings.simplefilter('always')
base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


def _get_data(mode='real'):
    """Get data."""
    raw = read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)[0:100]
    if mode == 'real':
        # Read raw data
        raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

        # Set picks
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')

        # Read several epochs
        event_id, tmin, tmax = 1, -0.2, 0.5
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            preload=True,
                            reject=dict(grad=4000e-13, mag=4e-12))
    elif mode == 'sin':
        # Create an epochs object with one epoch and one channel of artificial
        # data
        event_id, tmin, tmax = 1, 0.0, 1.0
        epochs = mne.Epochs(raw, events[0:5], event_id, tmin, tmax,  picks=[0],
                            preload=True, reject=dict(grad=4000e-13))
        freq = 10
        epochs._data = np.sin(2 * np.pi * freq *
                              epochs.times)[None, None, :]

    return epochs


def test_csd_epochs():
    """Test computing cross-spectral density from epochs."""
    epochs = _get_data(mode='real')
    # Check that wrong parameters are recognized
    assert_raises(ValueError, csd_epochs, epochs, mode='notamode')
    assert_raises(ValueError, csd_epochs, epochs, fmin=20, fmax=10)
    assert_raises(ValueError, csd_epochs, epochs, fmin=20, fmax=20.1)
    assert_raises(ValueError, csd_epochs, epochs, tmin=0.15, tmax=0.1)
    assert_raises(ValueError, csd_epochs, epochs, tmin=0, tmax=10)
    assert_raises(ValueError, csd_epochs, epochs, tmin=10, tmax=11)

    data_csd_mt = csd_epochs(epochs, mode='multitaper', fmin=8, fmax=12,
                             tmin=0.04, tmax=0.15)
    data_csd_fourier = csd_epochs(epochs, mode='fourier', fmin=8, fmax=12,
                                  tmin=0.04, tmax=0.15)

    # Check shape of the CSD matrix
    n_chan = len(data_csd_mt.ch_names)
    assert_equal(data_csd_mt.data.shape, (n_chan, n_chan))
    assert_equal(data_csd_fourier.data.shape, (n_chan, n_chan))

    # Check if the CSD matrix is hermitian
    assert_array_equal(np.tril(data_csd_mt.data).T.conj(),
                       np.triu(data_csd_mt.data))
    assert_array_equal(np.tril(data_csd_fourier.data).T.conj(),
                       np.triu(data_csd_fourier.data))

    # Computing induced power for comparison
    epochs.crop(tmin=0.04, tmax=0.15)
    tfr = tfr_morlet(epochs, freqs=[10], n_cycles=0.6, return_itc=False)
    power = np.mean(tfr.data, 2)

    # Maximum PSD should occur for specific channel
    max_ch_power = power.argmax()
    max_ch_mt = data_csd_mt.data.diagonal().argmax()
    max_ch_fourier = data_csd_fourier.data.diagonal().argmax()
    assert_equal(max_ch_mt, max_ch_power)
    assert_equal(max_ch_fourier, max_ch_power)

    # Maximum CSD should occur for specific channel
    ch_csd_mt = np.abs(data_csd_mt.data[max_ch_power])
    ch_csd_mt[max_ch_power] = 0.
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = np.abs(data_csd_fourier.data[max_ch_power])
    ch_csd_fourier[max_ch_power] = 0.
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert_equal(max_ch_csd_mt, max_ch_csd_fourier)

    # Check a list of CSD matrices is returned for multiple frequencies within
    # a given range when fsum=False
    csd_fsum = csd_epochs(epochs, mode='fourier', fmin=8, fmax=20, fsum=True)
    csds = csd_epochs(epochs, mode='fourier', fmin=8, fmax=20, fsum=False)
    freqs = [csd.frequencies[0] for csd in csds]

    csd_sum = np.zeros_like(csd_fsum.data)
    for csd in csds:
        csd_sum += csd.data

    assert_equal(len(csds), 2)
    assert_equal(len(csd_fsum.frequencies), 2)
    assert_array_equal(csd_fsum.frequencies, freqs)
    assert_array_equal(csd_fsum.data, csd_sum)


def test_csd_epochs_on_artificial_data():
    """Test computing CSD on artificial data."""
    epochs = _get_data(mode='sin')
    sfreq = epochs.info['sfreq']

    # Computing signal power in the time domain
    signal_power = sum_squared(epochs._data)
    signal_power_per_sample = signal_power / len(epochs.times)

    # Computing signal power in the frequency domain
    data_csd_fourier = csd_epochs(epochs, mode='fourier')
    data_csd_mt = csd_epochs(epochs, mode='multitaper')
    fourier_power = np.abs(data_csd_fourier.data[0, 0]) * sfreq
    mt_power = np.abs(data_csd_mt.data[0, 0]) * sfreq
    assert_true(abs(fourier_power - signal_power) <= 0.5)
    assert_true(abs(mt_power - signal_power) <= 1)

    # Power per sample should not depend on time window length
    for tmax in [0.2, 0.8]:
        for add_n_fft in [0, 30]:
            t_mask = (epochs.times >= 0) & (epochs.times <= tmax)
            n_samples = sum(t_mask)
            n_fft = n_samples + add_n_fft

            data_csd_fourier = csd_epochs(epochs, mode='fourier',
                                          tmin=None, tmax=tmax, fmin=0,
                                          fmax=np.inf, n_fft=n_fft)
            first_samp = data_csd_fourier.data[0, 0]
            fourier_power_per_sample = np.abs(first_samp) * sfreq / n_fft
            assert_true(abs(signal_power_per_sample -
                            fourier_power_per_sample) < 0.003)
        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 5]:
            for add_n_fft in [0, 30]:
                mt_bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
                data_csd_mt = csd_epochs(epochs, mode='multitaper',
                                         tmin=None, tmax=tmax, fmin=0,
                                         fmax=np.inf,
                                         mt_bandwidth=mt_bandwidth,
                                         n_fft=n_fft)
                mt_power_per_sample = np.abs(data_csd_mt.data[0, 0]) *\
                    sfreq / data_csd_mt.n_fft
                # The estimate of power gets worse for small time windows when
                # more tapers are used
                if n_tapers == 5 and tmax == 0.2:
                    delta = 0.05
                else:
                    delta = 0.004
                assert_true(abs(signal_power_per_sample -
                                mt_power_per_sample) < delta)


def test_compute_csd():
    """Test computing cross-spectral density from ndarray."""
    epochs = _get_data(mode='real')

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
    assert_raises(ValueError, csd_array, err_data, sfreq)
    assert_raises(ValueError, csd_array, diff_types, sfreq)
    assert_raises(ValueError, csd_array, np.random.randn(3), sfreq)

    # Check that wrong parameters are recognized
    assert_raises(ValueError, csd_array, X, sfreq, mode='notamode')
    assert_raises(ValueError, csd_array, X, sfreq, fmin=20, fmax=10)
    assert_raises(ValueError, csd_array, X, sfreq, fmin=20, fmax=20.1)

    data_csd_mt, freqs_mt = csd_array(X, sfreq, mode='multitaper',
                                      fmin=8, fmax=12)
    data_csd_fourier, freqs_fft = csd_array(X, sfreq, mode='fourier',
                                            fmin=8, fmax=12)

    # Test as list too
    data_csd_mt_list, freqs_mt_list = csd_array(X_list, sfreq,
                                                mode='multitaper',
                                                fmin=8, fmax=12)
    data_csd_fourier_list, freqs_fft_list = csd_array(X_list, sfreq,
                                                      mode='fourier',
                                                      fmin=8, fmax=12)

    assert_array_equal(data_csd_mt, data_csd_mt_list)
    assert_array_equal(data_csd_fourier, data_csd_fourier_list)
    assert_array_equal(freqs_mt, freqs_mt_list)
    assert_array_equal(freqs_fft, freqs_fft_list)

    # Check shape of the CSD matrix
    n_chan = len(epochs.ch_names)
    assert_equal(data_csd_mt.shape, (n_chan, n_chan))
    assert_equal(data_csd_fourier.shape, (n_chan, n_chan))

    # Check if the CSD matrix is hermitian
    assert_array_equal(np.tril(data_csd_mt).T.conj(),
                       np.triu(data_csd_mt))
    assert_array_equal(np.tril(data_csd_fourier).T.conj(),
                       np.triu(data_csd_fourier))

    # Computing induced power for comparison
    epochs.crop(tmin=0.04, tmax=0.15)
    tfr = tfr_morlet(epochs, freqs=[10], n_cycles=0.6, return_itc=False)
    power = np.mean(tfr.data, 2)

    # Maximum PSD should occur for specific channel
    max_ch_power = power.argmax()
    max_ch_mt = data_csd_mt.diagonal().argmax()
    max_ch_fourier = data_csd_fourier.diagonal().argmax()
    assert_equal(max_ch_mt, max_ch_power)
    assert_equal(max_ch_fourier, max_ch_power)

    # Maximum CSD should occur for specific channel
    ch_csd_mt = np.abs(data_csd_mt[max_ch_power])
    ch_csd_mt[max_ch_power] = 0.
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = np.abs(data_csd_fourier[max_ch_power])
    ch_csd_fourier[max_ch_power] = 0.
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert_equal(max_ch_csd_mt, max_ch_csd_fourier)

    # Check a list of CSD matrices is returned for multiple frequencies within
    # a given range when fsum=False
    csd_fsum, freqs_fsum = csd_array(X, sfreq, mode='fourier', fmin=8,
                                     fmax=20, fsum=True)
    csds, freqs = csd_array(X, sfreq, mode='fourier', fmin=8, fmax=20,
                            fsum=False)

    csd_sum = np.sum(csds, axis=2)

    assert_equal(csds.shape[2], 2)
    assert_equal(len(freqs), 2)
    assert_array_equal(freqs_fsum, freqs)
    assert_array_equal(csd_fsum, csd_sum)


def test_csd_on_artificial_data():
    """Test computing CSD on artificial data. """
    epochs = _get_data(mode='sin')
    sfreq = epochs.info['sfreq']

    # Computing signal power in the time domain
    signal_power = sum_squared(epochs._data)
    signal_power_per_sample = signal_power / len(epochs.times)

    # Computing signal power in the frequency domain
    data_csd_mt, freqs_mt = csd_array(epochs._data, sfreq,
                                      mode='multitaper')
    data_csd_fourier, freqs_fft = csd_array(epochs._data, sfreq,
                                            mode='fourier')

    fourier_power = np.abs(data_csd_fourier[0, 0]) * sfreq
    mt_power = np.abs(data_csd_mt[0, 0]) * sfreq
    assert_true(abs(fourier_power - signal_power) <= 0.5)
    assert_true(abs(mt_power - signal_power) <= 1)

    # Power per sample should not depend on time window length
    for tmax in [0.2, 0.8]:
        tslice = np.where(epochs.times <= tmax)[0]

        for add_n_fft in [0, 30]:
            t_mask = (epochs.times >= 0) & (epochs.times <= tmax)
            n_samples = sum(t_mask)
            n_fft = n_samples + add_n_fft

            data_csd_fourier, _ = csd_array(epochs._data[:, :, tslice],
                                            sfreq, mode='fourier',
                                            fmin=0, fmax=np.inf, n_fft=n_fft)

            first_samp = data_csd_fourier[0, 0]
            fourier_power_per_sample = np.abs(first_samp) * sfreq / n_fft
            assert_true(abs(signal_power_per_sample -
                            fourier_power_per_sample) < 0.003)
        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 5]:
            for add_n_fft in [0, 30]:
                mt_bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
                data_csd_mt, _ = csd_array(epochs._data[:, :, tslice],
                                           sfreq, mt_bandwidth=mt_bandwidth,
                                           n_fft=n_fft)
                mt_power_per_sample = np.abs(data_csd_mt[0, 0]) *\
                    sfreq / n_fft
                # The estimate of power gets worse for small time windows when
                # more tapers are used
                if n_tapers == 5 and tmax == 0.2:
                    delta = 0.05
                else:
                    delta = 0.004
                assert_true(abs(signal_power_per_sample -
                                mt_power_per_sample) < delta)

run_tests_if_main()
