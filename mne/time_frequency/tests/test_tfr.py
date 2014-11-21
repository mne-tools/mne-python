import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal

import mne
from mne import io, Epochs, read_events, pick_types, create_info, EpochsArray
from mne.utils import _TempDir
from mne.time_frequency import single_trial_power
from mne.time_frequency.tfr import cwt_morlet, morlet, tfr_morlet
from mne.time_frequency.tfr import _dpss_wavelet, tfr_multitaper
from mne.time_frequency.tfr import AverageTFR, read_tfr

raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
event_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                      'data', 'test-eve.fif')


def test_morlet():
    """Test morlet with and without zero mean"""
    Wz = morlet(1000, [10], 2., zero_mean=True)
    W = morlet(1000, [10], 2., zero_mean=False)

    assert_true(np.abs(np.mean(np.real(Wz[0]))) < 1e-5)
    assert_true(np.abs(np.mean(np.real(W[0]))) > 1e-3)


def test_time_frequency():
    """Test time frequency transform (PSD and phase lock)
    """
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = io.Raw(raw_fname)
    events = read_events(event_fname)

    include = []
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False,
                       stim=False, include=include, exclude=exclude)

    picks = picks[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    data = epochs.get_data()
    times = epochs.times
    nave = len(data)

    freqs = np.arange(6, 20, 5)  # define frequencies of interest
    n_cycles = freqs / 4.

    # Test first with a single epoch
    power, itc = tfr_morlet(epochs[0], freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True)

    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True)

    print(itc)  # test repr
    print(itc.ch_names)  # test property
    itc += power  # test add
    itc -= power  # test add

    power.apply_baseline(baseline=(-0.1, 0), mode='logratio')

    assert_true('meg' in power)
    assert_true('grad' in power)
    assert_false('mag' in power)
    assert_false('eeg' in power)

    assert_equal(power.nave, nave)
    assert_equal(itc.nave, nave)
    assert_true(power.data.shape == (len(picks), len(freqs), len(times)))
    assert_true(power.data.shape == itc.data.shape)
    assert_true(np.sum(itc.data >= 1) == 0)
    assert_true(np.sum(itc.data <= 0) == 0)

    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=2, use_fft=False,
                            return_itc=True)

    assert_true(power.data.shape == (len(picks), len(freqs), len(times)))
    assert_true(power.data.shape == itc.data.shape)
    assert_true(np.sum(itc.data >= 1) == 0)
    assert_true(np.sum(itc.data <= 0) == 0)

    Fs = raw.info['sfreq']  # sampling in Hz
    tfr = cwt_morlet(data[0], Fs, freqs, use_fft=True, n_cycles=2)
    assert_true(tfr.shape == (len(picks), len(freqs), len(times)))

    single_power = single_trial_power(data, Fs, freqs, use_fft=False,
                                      n_cycles=2)

    assert_array_almost_equal(np.mean(single_power), power.data)

    power_pick = power.pick_channels(power.ch_names[:10:2])
    assert_equal(len(power_pick.ch_names), len(power.ch_names[:10:2]))
    assert_equal(power_pick.data.shape[0], len(power.ch_names[:10:2]))
    power_drop = power.drop_channels(power.ch_names[1:10:2])
    assert_equal(power_drop.ch_names, power_pick.ch_names)
    assert_equal(power_pick.data.shape[0], len(power_drop.ch_names))

    mne.equalize_channels([power_pick, power_drop])
    assert_equal(power_pick.ch_names, power_drop.ch_names)
    assert_equal(power_pick.data.shape, power_drop.data.shape)


def test_dpsswavelet():
    """Some tests for DPSS wavelet"""
    freqs = np.arange(5, 25, 3)
    Ws = _dpss_wavelet(1000, freqs=freqs, n_cycles=freqs/2.,
                       time_bandwidth=4.0, zero_mean=True)

    assert_true(len(Ws) == 3)  # 3 tapers expected

    # Check that zero mean is true
    assert_true(np.abs(np.mean(np.real(Ws[0][0]))) < 1e-5)

    assert_true(len(Ws[0]) == len(freqs))  # As many wavelets as asked for


def test_tfr_multitaper():
    """ Some tests for tfr_multitaper() """
    sfreq = 200.0
    ch_names = ['SIM0001', 'SIM0002', 'SIM0003']
    ch_types = ['grad', 'grad', 'grad']
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    n_times = int(sfreq)  # Second long epochs
    n_epochs = 3
    seed = 42
    rng = np.random.RandomState(seed)
    noise = 0.1 * rng.randn(n_epochs, len(ch_names), n_times)
    t = np.arange(n_times, dtype=np.float) / sfreq
    signal = np.sin(np.pi * 2. * 50. * t)  # 50 Hz sinusoid signal
    signal[np.logical_or(t < 0.45, t > 0.55)] = 0.  # Hard windowing
    on_time = np.logical_and(t >= 0.45, t <= 0.55)
    signal[on_time] *= np.hanning(on_time.sum())  # Ramping
    dat = noise + signal

    reject = dict(grad=4000.)
    events = np.empty((n_epochs, 3))
    first_event_sample = 100
    event_id = dict(sin50hz=1)
    for k in range(n_epochs):
        events[k, :] = first_event_sample + k * n_times, 0, event_id['sin50hz']

    epochs = EpochsArray(data=dat, info=info, events=events, event_id=event_id,
                         reject=reject)

    freqs = np.arange(5, 100, 3, dtype=np.float)
    power, itc = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2.,
                                time_bandwidth=4.0)
    tmax = t[np.argmax(itc.data[0, freqs == 50, :])]
    fmax = freqs[np.argmax(power.data[1, :, t == 0.5])]
    assert_true(tmax > 0.3 and tmax < 0.7)
    assert_false(np.any(itc.data < 0.))
    assert_true(fmax > 40 and fmax < 60)


def test_io():
    """Test TFR IO capacities"""
    tempdir = _TempDir()
    fname = op.join(tempdir, 'test-tfr.h5')
    raw = io.Raw(raw_fname)
    data = np.zeros((raw.info['nchan'], 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])
    tfr = AverageTFR(raw.info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.save(fname)
    tfr2 = read_tfr(fname)
    assert_array_equal(tfr.data, tfr2.data)
    assert_array_equal(tfr.times, tfr2.times)
    assert_array_equal(tfr.freqs, tfr2.freqs)
    assert_equal(tfr.comment, tfr2.comment)
    assert_equal(tfr.nave, tfr2.nave)

test_io()
