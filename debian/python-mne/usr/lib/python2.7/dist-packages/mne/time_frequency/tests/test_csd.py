import numpy as np
from nose.tools import assert_raises, assert_equal, assert_true
from numpy.testing import assert_array_equal
from os import path as op
import warnings

import mne

from mne.io import Raw
from mne.utils import sum_squared
from mne.time_frequency import compute_epochs_csd, tfr_morlet

warnings.simplefilter('always')
base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


def _get_data():
    # Read raw data
    raw = Raw(raw_fname)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set picks
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

    # Read several epochs
    event_id, tmin, tmax = 1, -0.2, 0.5
    events = mne.read_events(event_fname)[0:100]
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=dict(grad=4000e-13, mag=4e-12))

    # Create an epochs object with one epoch and one channel of artificial data
    event_id, tmin, tmax = 1, 0.0, 1.0
    epochs_sin = mne.Epochs(raw, events[0:5], event_id, tmin, tmax, proj=True,
                            picks=[0], baseline=(None, 0), preload=True,
                            reject=dict(grad=4000e-13))
    freq = 10
    epochs_sin._data = np.sin(2 * np.pi * freq *
                              epochs_sin.times)[None, None, :]
    return epochs, epochs_sin


def test_compute_epochs_csd():
    """Test computing cross-spectral density from epochs
    """
    epochs, epochs_sin = _get_data()
    # Check that wrong parameters are recognized
    assert_raises(ValueError, compute_epochs_csd, epochs, mode='notamode')
    assert_raises(ValueError, compute_epochs_csd, epochs, fmin=20, fmax=10)
    assert_raises(ValueError, compute_epochs_csd, epochs, fmin=20, fmax=20.1)
    assert_raises(ValueError, compute_epochs_csd, epochs, tmin=0.15, tmax=0.1)
    assert_raises(ValueError, compute_epochs_csd, epochs, tmin=0, tmax=10)
    assert_raises(ValueError, compute_epochs_csd, epochs, tmin=10, tmax=11)

    data_csd_mt = compute_epochs_csd(epochs, mode='multitaper', fmin=8,
                                     fmax=12, tmin=0.04, tmax=0.15)
    data_csd_fourier = compute_epochs_csd(epochs, mode='fourier', fmin=8,
                                          fmax=12, tmin=0.04, tmax=0.15)

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
    ch_csd_mt = [np.abs(data_csd_mt.data[max_ch_power][i])
                 if i != max_ch_power else 0 for i in range(n_chan)]
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = [np.abs(data_csd_fourier.data[max_ch_power][i])
                      if i != max_ch_power else 0 for i in range(n_chan)]
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert_equal(max_ch_csd_mt, max_ch_csd_fourier)

    # Check a list of CSD matrices is returned for multiple frequencies within
    # a given range when fsum=False
    csd_fsum = compute_epochs_csd(epochs, mode='fourier', fmin=8, fmax=20,
                                  fsum=True)
    csds = compute_epochs_csd(epochs, mode='fourier', fmin=8, fmax=20,
                              fsum=False)
    freqs = [csd.frequencies[0] for csd in csds]

    csd_sum = np.zeros_like(csd_fsum.data)
    for csd in csds:
        csd_sum += csd.data

    assert(len(csds) == 2)
    assert(len(csd_fsum.frequencies) == 2)
    assert_array_equal(csd_fsum.frequencies, freqs)
    assert_array_equal(csd_fsum.data, csd_sum)


def test_compute_epochs_csd_on_artificial_data():
    """Test computing CSD on artificial data
    """
    epochs, epochs_sin = _get_data()
    sfreq = epochs_sin.info['sfreq']

    # Computing signal power in the time domain
    signal_power = sum_squared(epochs_sin._data)
    signal_power_per_sample = signal_power / len(epochs_sin.times)

    # Computing signal power in the frequency domain
    data_csd_fourier = compute_epochs_csd(epochs_sin, mode='fourier')
    data_csd_mt = compute_epochs_csd(epochs_sin, mode='multitaper')
    fourier_power = np.abs(data_csd_fourier.data[0, 0]) * sfreq
    mt_power = np.abs(data_csd_mt.data[0, 0]) * sfreq
    assert_true(abs(fourier_power - signal_power) <= 0.5)
    assert_true(abs(mt_power - signal_power) <= 1)

    # Power per sample should not depend on time window length
    for tmax in [0.2, 0.4, 0.6, 0.8]:
        for add_n_fft in [30, 0, 30]:
            t_mask = (epochs_sin.times >= 0) & (epochs_sin.times <= tmax)
            n_samples = sum(t_mask)
            n_fft = n_samples + add_n_fft

            data_csd_fourier = compute_epochs_csd(epochs_sin, mode='fourier',
                                                  tmin=None, tmax=tmax, fmin=0,
                                                  fmax=np.inf, n_fft=n_fft)
            fourier_power_per_sample = np.abs(data_csd_fourier.data[0, 0]) *\
                sfreq / data_csd_fourier.n_fft
            assert_true(abs(signal_power_per_sample -
                            fourier_power_per_sample) < 0.003)
        # Power per sample should not depend on number of tapers
        for n_tapers in [1, 2, 3, 5]:
            for add_n_fft in [30, 0, 30]:
                mt_bandwidth = sfreq / float(n_samples) * (n_tapers + 1)
                data_csd_mt = compute_epochs_csd(epochs_sin, mode='multitaper',
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
