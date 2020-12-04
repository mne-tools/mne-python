from itertools import product
import datetime
import os.path as op

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_allclose)
import pytest
import matplotlib.pyplot as plt

import mne
from mne import (Epochs, read_events, pick_types, create_info, EpochsArray,
                 Info, Transform)
from mne.io import read_raw_fif
from mne.utils import (_TempDir, run_tests_if_main, requires_h5py,
                       requires_pandas, grand_average, catch_logging)
from mne.time_frequency.tfr import (morlet, tfr_morlet, _make_dpss,
                                    tfr_multitaper, AverageTFR, read_tfrs,
                                    write_tfrs, combine_tfr, cwt, _compute_tfr,
                                    EpochsTFR)
from mne.time_frequency import tfr_array_multitaper, tfr_array_morlet
from mne.viz.utils import _fake_click
from mne.tests.test_epochs import assert_metadata_equal

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
raw_ctf_fname = op.join(data_path, 'test_ctf_raw.fif')


def test_tfr_ctf():
    """Test that TFRs can be calculated on CTF data."""
    raw = read_raw_fif(raw_ctf_fname).crop(0, 1)
    raw.apply_gradient_compensation(3)
    events = mne.make_fixed_length_events(raw, duration=0.5)
    epochs = mne.Epochs(raw, events)
    for method in (tfr_multitaper, tfr_morlet):
        method(epochs, [10], 1)  # smoke test


def test_morlet():
    """Test morlet with and without zero mean."""
    Wz = morlet(1000, [10], 2., zero_mean=True)
    W = morlet(1000, [10], 2., zero_mean=False)

    assert (np.abs(np.mean(np.real(Wz[0]))) < 1e-5)
    assert (np.abs(np.mean(np.real(W[0]))) > 1e-3)


def test_time_frequency():
    """Test time-frequency transform (PSD and ITC)."""
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    # Setup for reading the raw data
    raw = read_raw_fif(raw_fname)
    events = read_events(event_fname)

    include = []
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False,
                       stim=False, include=include, exclude=exclude)

    picks = picks[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()
    times = epochs.times
    nave = len(data)

    epochs_nopicks = Epochs(raw, events, event_id, tmin, tmax)

    freqs = np.arange(6, 20, 5)  # define frequencies of interest
    n_cycles = freqs / 4.

    # Test first with a single epoch
    power, itc = tfr_morlet(epochs[0], freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True)
    # Now compute evoked
    evoked = epochs.average()
    pytest.raises(ValueError, tfr_morlet, evoked, freqs, 1., return_itc=True)
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True)
    power_, itc_ = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                              use_fft=True, return_itc=True, decim=slice(0, 2))
    # Test picks argument and average parameter
    pytest.raises(ValueError, tfr_morlet, epochs, freqs=freqs,
                  n_cycles=n_cycles, return_itc=True, average=False)

    power_picks, itc_picks = \
        tfr_morlet(epochs_nopicks,
                   freqs=freqs, n_cycles=n_cycles, use_fft=True,
                   return_itc=True, picks=picks, average=True)

    epochs_power_picks = \
        tfr_morlet(epochs_nopicks,
                   freqs=freqs, n_cycles=n_cycles, use_fft=True,
                   return_itc=False, picks=picks, average=False)
    power_picks_avg = epochs_power_picks.average()
    # the actual data arrays here are equivalent, too...
    assert_allclose(power.data, power_picks.data)
    assert_allclose(power.data, power_picks_avg.data)
    assert_allclose(itc.data, itc_picks.data)

    # test on evoked
    power_evoked = tfr_morlet(evoked, freqs, n_cycles, use_fft=True,
                              return_itc=False)
    # one is squared magnitude of the average (evoked) and
    # the other is average of the squared magnitudes (epochs PSD)
    # so values shouldn't match, but shapes should
    assert_array_equal(power.data.shape, power_evoked.data.shape)
    pytest.raises(AssertionError, assert_allclose,
                  power.data, power_evoked.data)

    # complex output
    pytest.raises(ValueError, tfr_morlet, epochs, freqs, n_cycles,
                  return_itc=False, average=True, output="complex")
    pytest.raises(ValueError, tfr_morlet, epochs, freqs, n_cycles,
                  output="complex", average=False, return_itc=True)
    epochs_power_complex = tfr_morlet(epochs, freqs, n_cycles,
                                      output="complex", average=False,
                                      return_itc=False)
    epochs_amplitude_2 = abs(epochs_power_complex)
    epochs_amplitude_3 = epochs_amplitude_2.copy()
    epochs_amplitude_3.data[:] = np.inf  # test that it's actually copied

    # test that the power computed via `complex` is equivalent to power
    # computed within the method.
    assert_allclose(epochs_amplitude_2.data**2, epochs_power_picks.data)

    print(itc)  # test repr
    print(itc.ch_names)  # test property
    itc += power  # test add
    itc -= power  # test sub
    ret = itc * 23  # test mult
    itc = ret / 23  # test dic

    power = power.apply_baseline(baseline=(-0.1, 0), mode='logratio')

    assert 'meg' in power
    assert 'grad' in power
    assert 'mag' not in power
    assert 'eeg' not in power

    assert power.nave == nave
    assert itc.nave == nave
    assert (power.data.shape == (len(picks), len(freqs), len(times)))
    assert (power.data.shape == itc.data.shape)
    assert (power_.data.shape == (len(picks), len(freqs), 2))
    assert (power_.data.shape == itc_.data.shape)
    assert (np.sum(itc.data >= 1) == 0)
    assert (np.sum(itc.data <= 0) == 0)

    # grand average
    itc2 = itc.copy()
    itc2.info['bads'] = [itc2.ch_names[0]]  # test channel drop
    gave = grand_average([itc2, itc])
    assert gave.data.shape == (itc2.data.shape[0] - 1,
                               itc2.data.shape[1],
                               itc2.data.shape[2])
    assert itc2.ch_names[1:] == gave.ch_names
    assert gave.nave == 2
    itc2.drop_channels(itc2.info["bads"])
    assert_allclose(gave.data, itc2.data)
    itc2.data = np.ones(itc2.data.shape)
    itc.data = np.zeros(itc.data.shape)
    itc2.nave = 2
    itc.nave = 1
    itc.drop_channels([itc.ch_names[0]])
    combined_itc = combine_tfr([itc2, itc])
    assert_allclose(combined_itc.data,
                    np.ones(combined_itc.data.shape) * 2 / 3)

    # more tests
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=2, use_fft=False,
                            return_itc=True)

    assert (power.data.shape == (len(picks), len(freqs), len(times)))
    assert (power.data.shape == itc.data.shape)
    assert (np.sum(itc.data >= 1) == 0)
    assert (np.sum(itc.data <= 0) == 0)

    tfr = tfr_morlet(epochs[0], freqs, use_fft=True, n_cycles=2, average=False,
                     return_itc=False)
    tfr_data = tfr.data[0]
    assert (tfr_data.shape == (len(picks), len(freqs), len(times)))
    tfr2 = tfr_morlet(epochs[0], freqs, use_fft=True, n_cycles=2,
                      decim=slice(0, 2), average=False,
                      return_itc=False).data[0]
    assert (tfr2.shape == (len(picks), len(freqs), 2))

    single_power = tfr_morlet(epochs, freqs, 2, average=False,
                              return_itc=False).data
    single_power2 = tfr_morlet(epochs, freqs, 2, decim=slice(0, 2),
                               average=False, return_itc=False).data
    single_power3 = tfr_morlet(epochs, freqs, 2, decim=slice(1, 3),
                               average=False, return_itc=False).data
    single_power4 = tfr_morlet(epochs, freqs, 2, decim=slice(2, 4),
                               average=False, return_itc=False).data

    assert_allclose(np.mean(single_power, axis=0), power.data)
    assert_allclose(np.mean(single_power2, axis=0), power.data[:, :, :2])
    assert_allclose(np.mean(single_power3, axis=0), power.data[:, :, 1:3])
    assert_allclose(np.mean(single_power4, axis=0), power.data[:, :, 2:4])

    power_pick = power.pick_channels(power.ch_names[:10:2])
    assert_equal(len(power_pick.ch_names), len(power.ch_names[:10:2]))
    assert_equal(power_pick.data.shape[0], len(power.ch_names[:10:2]))
    power_drop = power.drop_channels(power.ch_names[1:10:2])
    assert_equal(power_drop.ch_names, power_pick.ch_names)
    assert_equal(power_pick.data.shape[0], len(power_drop.ch_names))

    power_pick, power_drop = mne.equalize_channels([power_pick, power_drop])
    assert_equal(power_pick.ch_names, power_drop.ch_names)
    assert_equal(power_pick.data.shape, power_drop.data.shape)

    # Test decimation:
    # 2: multiple of len(times) even
    # 3: multiple odd
    # 8: not multiple, even
    # 9: not multiple, odd
    for decim in [2, 3, 8, 9]:
        for use_fft in [True, False]:
            power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=2,
                                    use_fft=use_fft, return_itc=True,
                                    decim=decim)
            assert_equal(power.data.shape[2],
                         np.ceil(float(len(times)) / decim))
    freqs = list(range(50, 55))
    decim = 2
    _, n_chan, n_time = data.shape
    tfr = tfr_morlet(epochs[0], freqs, 2., decim=decim, average=False,
                     return_itc=False).data[0]
    assert_equal(tfr.shape, (n_chan, len(freqs), n_time // decim))

    # Test cwt modes
    Ws = morlet(512, [10, 20], n_cycles=2)
    pytest.raises(ValueError, cwt, data[0, :, :], Ws, mode='foo')
    for use_fft in [True, False]:
        for mode in ['same', 'valid', 'full']:
            cwt(data[0], Ws, use_fft=use_fft, mode=mode)

    # Test invalid frequency arguments
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_morlet(epochs, freqs=np.arange(0, 3), n_cycles=7)
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_morlet(epochs, freqs=np.arange(-4, -1), n_cycles=7)

    # Test decim parameter checks
    pytest.raises(TypeError, tfr_morlet, epochs, freqs=freqs,
                  n_cycles=n_cycles, use_fft=True, return_itc=True,
                  decim='decim')

    # When convolving in time, wavelets must not be longer than the data
    pytest.raises(ValueError, cwt, data[0, :, :Ws[0].size - 1], Ws,
                  use_fft=False)
    with pytest.warns(UserWarning, match='one of the wavelets.*is longer'):
        cwt(data[0, :, :Ws[0].size - 1], Ws, use_fft=True)

    # Check for off-by-one errors when using wavelets with an even number of
    # samples
    psd = cwt(data[0], [Ws[0][:-1]], use_fft=False, mode='full')
    assert_equal(psd.shape, (2, 1, 420))


def test_dpsswavelet():
    """Test DPSS tapers."""
    freqs = np.arange(5, 25, 3)
    Ws = _make_dpss(1000, freqs=freqs, n_cycles=freqs / 2., time_bandwidth=4.0,
                    zero_mean=True)

    assert (len(Ws) == 3)  # 3 tapers expected

    # Check that zero mean is true
    assert (np.abs(np.mean(np.real(Ws[0][0]))) < 1e-5)

    assert (len(Ws[0]) == len(freqs))  # As many wavelets as asked for


@pytest.mark.slowtest
def test_tfr_multitaper():
    """Test tfr_multitaper."""
    sfreq = 200.0
    ch_names = ['SIM0001', 'SIM0002']
    ch_types = ['grad', 'grad']
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    n_times = int(sfreq)  # Second long epochs
    n_epochs = 3
    seed = 42
    rng = np.random.RandomState(seed)
    noise = 0.1 * rng.randn(n_epochs, len(ch_names), n_times)
    t = np.arange(n_times, dtype=np.float64) / sfreq
    signal = np.sin(np.pi * 2. * 50. * t)  # 50 Hz sinusoid signal
    signal[np.logical_or(t < 0.45, t > 0.55)] = 0.  # Hard windowing
    on_time = np.logical_and(t >= 0.45, t <= 0.55)
    signal[on_time] *= np.hanning(on_time.sum())  # Ramping
    dat = noise + signal

    reject = dict(grad=4000.)
    events = np.empty((n_epochs, 3), int)
    first_event_sample = 100
    event_id = dict(sin50hz=1)
    for k in range(n_epochs):
        events[k, :] = first_event_sample + k * n_times, 0, event_id['sin50hz']

    epochs = EpochsArray(data=dat, info=info, events=events, event_id=event_id,
                         reject=reject)

    freqs = np.arange(35, 70, 5, dtype=np.float64)

    power, itc = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2.,
                                time_bandwidth=4.0)
    power2, itc2 = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2.,
                                  time_bandwidth=4.0, decim=slice(0, 2))
    picks = np.arange(len(ch_names))
    power_picks, itc_picks = tfr_multitaper(epochs, freqs=freqs,
                                            n_cycles=freqs / 2.,
                                            time_bandwidth=4.0, picks=picks)
    power_epochs = tfr_multitaper(epochs, freqs=freqs,
                                  n_cycles=freqs / 2., time_bandwidth=4.0,
                                  return_itc=False, average=False)
    power_averaged = power_epochs.average()
    power_evoked = tfr_multitaper(epochs.average(), freqs=freqs,
                                  n_cycles=freqs / 2., time_bandwidth=4.0,
                                  return_itc=False, average=False).average()

    print(power_evoked)  # test repr for EpochsTFR

    # Test channel picking
    power_epochs_picked = power_epochs.copy().drop_channels(['SIM0002'])
    assert_equal(power_epochs_picked.data.shape, (3, 1, 7, 200))
    assert_equal(power_epochs_picked.ch_names, ['SIM0001'])

    pytest.raises(ValueError, tfr_multitaper, epochs,
                  freqs=freqs, n_cycles=freqs / 2.,
                  return_itc=True, average=False)

    # test picks argument
    assert_allclose(power.data, power_picks.data)
    assert_allclose(power.data, power_averaged.data)
    assert_allclose(power.times, power_epochs.times)
    assert_allclose(power.times, power_averaged.times)
    assert_equal(power.nave, power_averaged.nave)
    assert_equal(power_epochs.data.shape, (3, 2, 7, 200))
    assert_allclose(itc.data, itc_picks.data)
    # one is squared magnitude of the average (evoked) and
    # the other is average of the squared magnitudes (epochs PSD)
    # so values shouldn't match, but shapes should
    assert_array_equal(power.data.shape, power_evoked.data.shape)
    pytest.raises(AssertionError, assert_allclose,
                  power.data, power_evoked.data)

    tmax = t[np.argmax(itc.data[0, freqs == 50, :])]
    fmax = freqs[np.argmax(power.data[1, :, t == 0.5])]
    assert (tmax > 0.3 and tmax < 0.7)
    assert not np.any(itc.data < 0.)
    assert (fmax > 40 and fmax < 60)
    assert (power2.data.shape == (len(picks), len(freqs), 2))
    assert (power2.data.shape == itc2.data.shape)

    # Test decim parameter checks and compatibility between wavelets length
    # and instance length in the time dimension.
    pytest.raises(TypeError, tfr_multitaper, epochs, freqs=freqs,
                  n_cycles=freqs / 2., time_bandwidth=4.0, decim=(1,))
    pytest.raises(ValueError, tfr_multitaper, epochs, freqs=freqs,
                  n_cycles=1000, time_bandwidth=4.0)

    # Test invalid frequency arguments
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_multitaper(epochs, freqs=np.arange(0, 3), n_cycles=7)
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_multitaper(epochs, freqs=np.arange(-4, -1), n_cycles=7)


def test_crop():
    """Test TFR cropping."""
    data = np.zeros((3, 4, 5))
    times = np.array([.1, .2, .3, .4, .5])
    freqs = np.array([.10, .20, .30, .40])
    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')

    tfr.crop(tmin=0.2)
    assert_array_equal(tfr.times, [0.2, 0.3, 0.4, 0.5])
    assert tfr.data.ndim == 3
    assert tfr.data.shape[-1] == 4

    tfr.crop(fmax=0.3)
    assert_array_equal(tfr.freqs, [0.1, 0.2, 0.3])
    assert tfr.data.ndim == 3
    assert tfr.data.shape[-2] == 3

    tfr.crop(tmin=0.3, tmax=0.4, fmin=0.1, fmax=0.2)
    assert_array_equal(tfr.times, [0.3, 0.4])
    assert tfr.data.ndim == 3
    assert tfr.data.shape[-1] == 2
    assert_array_equal(tfr.freqs, [0.1, 0.2])
    assert tfr.data.shape[-2] == 2


@requires_h5py
@requires_pandas
def test_io():
    """Test TFR IO capacities."""
    from pandas import DataFrame
    tempdir = _TempDir()
    fname = op.join(tempdir, 'test-tfr.h5')
    data = np.zeros((3, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])

    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    info['meas_date'] = datetime.datetime(year=2020, month=2, day=5,
                                          tzinfo=datetime.timezone.utc)
    info._check_consistency()
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.save(fname)
    tfr2 = read_tfrs(fname, condition='test')
    assert isinstance(tfr2.info, Info)
    assert isinstance(tfr2.info['dev_head_t'], Transform)

    assert_array_equal(tfr.data, tfr2.data)
    assert_array_equal(tfr.times, tfr2.times)
    assert_array_equal(tfr.freqs, tfr2.freqs)
    assert_equal(tfr.comment, tfr2.comment)
    assert_equal(tfr.nave, tfr2.nave)

    pytest.raises(IOError, tfr.save, fname)

    tfr.comment = None
    # test old meas_date
    info['meas_date'] = (1, 2)
    tfr.save(fname, overwrite=True)
    assert_equal(read_tfrs(fname, condition=0).comment, tfr.comment)
    tfr.comment = 'test-A'
    tfr2.comment = 'test-B'

    fname = op.join(tempdir, 'test2-tfr.h5')
    write_tfrs(fname, [tfr, tfr2])
    tfr3 = read_tfrs(fname, condition='test-A')
    assert_equal(tfr.comment, tfr3.comment)

    assert (isinstance(tfr.info, mne.Info))

    tfrs = read_tfrs(fname, condition=None)
    assert_equal(len(tfrs), 2)
    tfr4 = tfrs[1]
    assert_equal(tfr2.comment, tfr4.comment)

    pytest.raises(ValueError, read_tfrs, fname, condition='nonono')
    # Test save of EpochsTFR.
    n_events = 5
    data = np.zeros((n_events, 3, 2, 3))

    # create fake metadata
    rng = np.random.RandomState(42)
    rt = np.round(rng.uniform(size=(n_events,)), 3)
    trialtypes = np.array(['face', 'place'])
    trial = trialtypes[(rng.uniform(size=(n_events,)) > .5).astype(int)]
    meta = DataFrame(dict(RT=rt, Trial=trial))
    # fake events and event_id
    events = np.zeros([n_events, 3])
    events[:, 0] = np.arange(n_events)
    events[:, 2] = np.ones(n_events)
    event_id = {'a/b': 1}

    tfr = EpochsTFR(info, data=data, times=times, freqs=freqs,
                    comment='test', method='crazy-tfr', events=events,
                    event_id=event_id, metadata=meta)
    tfr.save(fname, True)
    read_tfr = read_tfrs(fname)[0]
    assert_array_equal(tfr.data, read_tfr.data)
    assert_metadata_equal(tfr.metadata, read_tfr.metadata)
    assert_array_equal(tfr.events, read_tfr.events)
    assert tfr.event_id == read_tfr.event_id


def test_plot():
    """Test TFR plotting."""
    data = np.zeros((3, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])
    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.plot([1, 2], title='title', colorbar=False,
             mask=np.ones(tfr.data.shape[1:], bool))
    plt.close('all')
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax3 = plt.subplot2grid((2, 2), (0, 1))
    tfr.plot(picks=[0, 1, 2], axes=[ax, ax2, ax3])
    plt.close('all')

    tfr.plot([1, 2], title='title', colorbar=False, exclude='bads')
    plt.close('all')

    tfr.plot_topo(picks=[1, 2])
    plt.close('all')

    fig = tfr.plot(picks=[1], cmap='RdBu_r')  # interactive mode on by default
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('down')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('+')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('-')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('pageup')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('pagedown')

    cbar = fig.get_axes()[0].CB  # Fake dragging with mouse.
    ax = cbar.cbar.ax
    _fake_click(fig, ax, (0.1, 0.1))
    _fake_click(fig, ax, (0.1, 0.2), kind='motion')
    _fake_click(fig, ax, (0.1, 0.3), kind='release')

    _fake_click(fig, ax, (0.1, 0.1), button=3)
    _fake_click(fig, ax, (0.1, 0.2), button=3, kind='motion')
    _fake_click(fig, ax, (0.1, 0.3), kind='release')

    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up

    plt.close('all')


def test_plot_joint():
    """Test TFR joint plotting."""
    raw = read_raw_fif(raw_fname)
    times = np.linspace(-0.1, 0.1, 200)
    n_freqs = 3
    nave = 1
    rng = np.random.RandomState(42)
    data = rng.randn(len(raw.ch_names), n_freqs, len(times))
    tfr = AverageTFR(raw.info, data, times, np.arange(n_freqs), nave)

    topomap_args = {'res': 8, 'contours': 0, 'sensors': False}

    for combine in ('mean', 'rms', None):
        with catch_logging() as log:
            tfr.plot_joint(title='auto', colorbar=True,
                           combine=combine, topomap_args=topomap_args,
                           verbose='debug')
        plt.close('all')
        log = log.getvalue()
        assert 'Plotting topomap for grad data' in log

    # check various timefreqs
    for timefreqs in (
            {(tfr.times[0], tfr.freqs[1]): (0.1, 0.5),
             (tfr.times[-1], tfr.freqs[-1]): (0.2, 0.6)},
            [(tfr.times[1], tfr.freqs[1])]):
        tfr.plot_joint(timefreqs=timefreqs, topomap_args=topomap_args)
        plt.close('all')

    # test bad timefreqs
    timefreqs = ([(-100, 1)], tfr.times[1], [1],
                 [(tfr.times[1], tfr.freqs[1], tfr.freqs[1])])
    for these_timefreqs in timefreqs:
        pytest.raises(ValueError, tfr.plot_joint, these_timefreqs)

    # test that the object is not internally modified
    tfr_orig = tfr.copy()
    tfr.plot_joint(baseline=(0, None), exclude=[tfr.ch_names[0]],
                   topomap_args=topomap_args)
    plt.close('all')
    assert_array_equal(tfr.data, tfr_orig.data)
    assert set(tfr.ch_names) == set(tfr_orig.ch_names)
    assert set(tfr.times) == set(tfr_orig.times)

    # test tfr with picked channels
    tfr.pick_channels(tfr.ch_names[:-1])
    tfr.plot_joint(title='auto', colorbar=True, topomap_args=topomap_args)


def test_add_channels():
    """Test tfr splitting / re-appending channel types."""
    data = np.zeros((6, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])
    info = mne.create_info(
        ['MEG 001', 'MEG 002', 'MEG 003', 'EEG 001', 'EEG 002', 'STIM 001'],
        1000., ['mag', 'mag', 'mag', 'eeg', 'eeg', 'stim'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr_eeg = tfr.copy().pick_types(meg=False, eeg=True)
    tfr_meg = tfr.copy().pick_types(meg=True)
    tfr_stim = tfr.copy().pick_types(meg=False, stim=True)
    tfr_eeg_meg = tfr.copy().pick_types(meg=True, eeg=True)
    tfr_new = tfr_meg.copy().add_channels([tfr_eeg, tfr_stim])
    assert all(ch in tfr_new.ch_names
               for ch in tfr_stim.ch_names + tfr_meg.ch_names)
    tfr_new = tfr_meg.copy().add_channels([tfr_eeg])

    have_all = all(ch in tfr_new.ch_names
                   for ch in tfr.ch_names if ch != 'STIM 001')
    assert have_all
    assert_array_equal(tfr_new.data, tfr_eeg_meg.data)
    assert all(ch not in tfr_new.ch_names for ch in tfr_stim.ch_names)

    # Now test errors
    tfr_badsf = tfr_eeg.copy()
    tfr_badsf.info['sfreq'] = 3.1415927
    tfr_eeg = tfr_eeg.crop(-.1, .1)

    pytest.raises(RuntimeError, tfr_meg.add_channels, [tfr_badsf])
    pytest.raises(AssertionError, tfr_meg.add_channels, [tfr_eeg])
    pytest.raises(ValueError, tfr_meg.add_channels, [tfr_meg])
    pytest.raises(TypeError, tfr_meg.add_channels, tfr_badsf)


def test_compute_tfr():
    """Test _compute_tfr function."""
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    # Setup for reading the raw data
    raw = read_raw_fif(raw_fname)
    events = read_events(event_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False,
                       stim=False, include=[], exclude=exclude)

    picks = picks[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    freqs = np.arange(10, 20, 3).astype(float)

    # Check all combination of options
    for func, use_fft, zero_mean, output in product(
        (tfr_array_multitaper, tfr_array_morlet), (False, True), (False, True),
        ('complex', 'power', 'phase',
         'avg_power_itc', 'avg_power', 'itc')):
        # Check exception
        if (func == tfr_array_multitaper) and (output == 'phase'):
            pytest.raises(NotImplementedError, func, data, sfreq=sfreq,
                          freqs=freqs, output=output)
            continue

        # Check runs
        out = func(data, sfreq=sfreq, freqs=freqs, use_fft=use_fft,
                   zero_mean=zero_mean, n_cycles=2., output=output)
        # Check shapes
        shape = np.r_[data.shape[:2], len(freqs), data.shape[2]]
        if ('avg' in output) or ('itc' in output):
            assert_array_equal(shape[1:], out.shape)
        else:
            assert_array_equal(shape, out.shape)

        # Check types
        if output in ('complex', 'avg_power_itc'):
            assert_equal(np.complex128, out.dtype)
        else:
            assert_equal(np.float64, out.dtype)
        assert (np.all(np.isfinite(out)))

    # Check errors params
    for _data in (None, 'foo', data[0]):
        pytest.raises(ValueError, _compute_tfr, _data, freqs, sfreq)
    for _freqs in (None, 'foo', [[0]]):
        pytest.raises(ValueError, _compute_tfr, data, _freqs, sfreq)
    for _sfreq in (None, 'foo'):
        pytest.raises(ValueError, _compute_tfr, data, freqs, _sfreq)
    for key in ('output', 'method', 'use_fft', 'decim', 'n_jobs'):
        for value in (None, 'foo'):
            kwargs = {key: value}  # FIXME pep8
            pytest.raises(ValueError, _compute_tfr, data, freqs, sfreq,
                          **kwargs)
    with pytest.raises(ValueError, match='above Nyquist'):
        _compute_tfr(data, [sfreq], sfreq)

    # No time_bandwidth param in morlet
    pytest.raises(ValueError, _compute_tfr, data, freqs, sfreq,
                  method='morlet', time_bandwidth=1)
    # No phase in multitaper XXX Check ?
    pytest.raises(NotImplementedError, _compute_tfr, data, freqs, sfreq,
                  method='multitaper', output='phase')

    # Inter-trial coherence tests
    out = _compute_tfr(data, freqs, sfreq, output='itc', n_cycles=2.)
    assert np.sum(out >= 1) == 0
    assert np.sum(out <= 0) == 0

    # Check decim shapes
    # 2: multiple of len(times) even
    # 3: multiple odd
    # 8: not multiple, even
    # 9: not multiple, odd
    for decim in (2, 3, 8, 9, slice(0, 2), slice(1, 3), slice(2, 4)):
        _decim = slice(None, None, decim) if isinstance(decim, int) else decim
        n_time = len(np.arange(data.shape[2])[_decim])
        shape = np.r_[data.shape[:2], len(freqs), n_time]
        for method in ('multitaper', 'morlet'):
            # Single trials
            out = _compute_tfr(data, freqs, sfreq, method=method, decim=decim,
                               n_cycles=2.)
            assert_array_equal(shape, out.shape)
            # Averages
            out = _compute_tfr(data, freqs, sfreq, method=method, decim=decim,
                               output='avg_power', n_cycles=2.)
            assert_array_equal(shape[1:], out.shape)


@pytest.mark.parametrize('method', ('multitaper', 'morlet'))
@pytest.mark.parametrize('decim', (1, slice(1, None, 2), 3))
def test_compute_tfr_correct(method, decim):
    """Test that TFR actually gets us our freq back."""
    sfreq = 1000.
    t = np.arange(1000) / sfreq
    f = 50.
    data = np.sin(2 * np.pi * 50. * t)
    data *= np.hanning(data.size)
    data = data[np.newaxis, np.newaxis]
    freqs = np.arange(10, 111, 10)
    assert f in freqs
    tfr = _compute_tfr(data, freqs, sfreq, method=method, decim=decim,
                       n_cycles=2)[0, 0]
    assert freqs[np.argmax(np.abs(tfr).mean(-1))] == f


@requires_pandas
def test_getitem_epochsTFR():
    """Test GetEpochsMixin in the context of EpochsTFR."""
    from pandas import DataFrame
    # Setup for reading the raw data and select a few trials
    raw = read_raw_fif(raw_fname)
    events = read_events(event_fname)
    n_events = 10

    # create fake metadata
    rng = np.random.RandomState(42)
    rt = rng.uniform(size=(n_events,))
    trialtypes = np.array(['face', 'place'])
    trial = trialtypes[(rng.uniform(size=(n_events,)) > .5).astype(int)]
    meta = DataFrame(dict(RT=rt, Trial=trial))
    event_id = dict(a=1, b=2, c=3, d=4)
    epochs = Epochs(raw, events[:n_events], event_id=event_id, metadata=meta,
                    decim=1)

    freqs = np.arange(12., 17., 2.)  # define frequencies of interest
    n_cycles = freqs / 2.  # 0.5 second time windows for all frequencies

    # Choose time x (full) bandwidth product
    time_bandwidth = 4.0  # With 0.5 s time windows, this gives 8 Hz smoothing
    kwargs = dict(freqs=freqs, n_cycles=n_cycles, use_fft=True,
                  time_bandwidth=time_bandwidth, return_itc=False,
                  average=False, n_jobs=1)
    power = tfr_multitaper(epochs, **kwargs)

    # Check decim affects sfreq
    power_decim = tfr_multitaper(epochs, decim=2, **kwargs)
    assert power.info['sfreq'] / 2. == power_decim.info['sfreq']

    # Check that power and epochs metadata is the same
    assert_metadata_equal(epochs.metadata, power.metadata)
    assert_metadata_equal(epochs[::2].metadata, power[::2].metadata)
    assert_metadata_equal(epochs['RT < .5'].metadata,
                          power['RT < .5'].metadata)

    # Check that get power is functioning
    assert_array_equal(power[3:6].data, power.data[3:6])
    assert_array_equal(power[3:6].events, power.events[3:6])

    indx_check = (power.metadata['Trial'] == 'face')
    try:
        indx_check = indx_check.to_numpy()
    except Exception:
        pass  # older Pandas
    indx_check = indx_check.nonzero()
    assert_array_equal(power['Trial == "face"'].events,
                       power.events[indx_check])
    assert_array_equal(power['Trial == "face"'].data,
                       power.data[indx_check])

    # Check that the wrong Key generates a Key Error for Metadata search
    with pytest.raises(KeyError):
        power['Trialz == "place"']

    # Test length function
    assert len(power) == n_events
    assert len(power[3:6]) == 3

    # Test iteration function
    for ind, power_ep in enumerate(power):
        assert_array_equal(power_ep, power.data[ind])
        if ind == 5:
            break

    # Test that current state is maintained
    assert_array_equal(power.next(), power.data[ind + 1])


run_tests_if_main()
