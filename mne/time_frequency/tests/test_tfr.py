import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

import mne
from mne import Epochs, read_events, pick_types, create_info, EpochsArray
from mne.io import read_raw_fif
from mne.utils import (_TempDir, run_tests_if_main, slow_test, requires_h5py,
                       grand_average)
from mne.time_frequency.tfr import (morlet, tfr_morlet, _make_dpss,
                                    tfr_multitaper, AverageTFR, read_tfrs,
                                    write_tfrs, combine_tfr, cwt, _compute_tfr)
from mne.time_frequency import tfr_array_multitaper, tfr_array_morlet
from mne.viz.utils import _fake_click
from itertools import product
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')
event_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                      'data', 'test-eve.fif')


def test_morlet():
    """Test morlet with and without zero mean."""
    Wz = morlet(1000, [10], 2., zero_mean=True)
    W = morlet(1000, [10], 2., zero_mean=False)

    assert_true(np.abs(np.mean(np.real(Wz[0]))) < 1e-5)
    assert_true(np.abs(np.mean(np.real(W[0]))) > 1e-3)


def test_time_frequency():
    """Test the to-be-deprecated time-frequency transform (PSD and ITC)."""
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
    power_evoked = tfr_morlet(evoked, freqs, n_cycles, use_fft=True,
                              return_itc=False)
    assert_raises(ValueError, tfr_morlet, evoked, freqs, 1., return_itc=True)
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True)
    power_, itc_ = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                              use_fft=True, return_itc=True, decim=slice(0, 2))
    # Test picks argument and average parameter
    assert_raises(ValueError, tfr_morlet, epochs, freqs=freqs,
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
    assert_array_almost_equal(power.data, power_picks.data)
    assert_array_almost_equal(power.data, power_picks_avg.data)
    assert_array_almost_equal(itc.data, itc_picks.data)
    assert_array_almost_equal(power.data, power_evoked.data)

    print(itc)  # test repr
    print(itc.ch_names)  # test property
    itc += power  # test add
    itc -= power  # test sub

    power = power.apply_baseline(baseline=(-0.1, 0), mode='logratio')

    assert_true('meg' in power)
    assert_true('grad' in power)
    assert_false('mag' in power)
    assert_false('eeg' in power)

    assert_equal(power.nave, nave)
    assert_equal(itc.nave, nave)
    assert_true(power.data.shape == (len(picks), len(freqs), len(times)))
    assert_true(power.data.shape == itc.data.shape)
    assert_true(power_.data.shape == (len(picks), len(freqs), 2))
    assert_true(power_.data.shape == itc_.data.shape)
    assert_true(np.sum(itc.data >= 1) == 0)
    assert_true(np.sum(itc.data <= 0) == 0)

    # grand average
    itc2 = itc.copy()
    itc2.info['bads'] = [itc2.ch_names[0]]  # test channel drop
    gave = grand_average([itc2, itc])
    assert_equal(gave.data.shape, (itc2.data.shape[0] - 1,
                                   itc2.data.shape[1],
                                   itc2.data.shape[2]))
    assert_equal(itc2.ch_names[1:], gave.ch_names)
    assert_equal(gave.nave, 2)
    itc2.drop_channels(itc2.info["bads"])
    assert_array_almost_equal(gave.data, itc2.data)
    itc2.data = np.ones(itc2.data.shape)
    itc.data = np.zeros(itc.data.shape)
    itc2.nave = 2
    itc.nave = 1
    itc.drop_channels([itc.ch_names[0]])
    combined_itc = combine_tfr([itc2, itc])
    assert_array_almost_equal(combined_itc.data,
                              np.ones(combined_itc.data.shape) * 2 / 3)

    # more tests
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=2, use_fft=False,
                            return_itc=True)

    assert_true(power.data.shape == (len(picks), len(freqs), len(times)))
    assert_true(power.data.shape == itc.data.shape)
    assert_true(np.sum(itc.data >= 1) == 0)
    assert_true(np.sum(itc.data <= 0) == 0)

    tfr = tfr_morlet(epochs[0], freqs, use_fft=True, n_cycles=2, average=False,
                     return_itc=False).data[0]
    assert_true(tfr.shape == (len(picks), len(freqs), len(times)))
    tfr2 = tfr_morlet(epochs[0], freqs, use_fft=True, n_cycles=2,
                      decim=slice(0, 2), average=False,
                      return_itc=False).data[0]
    assert_true(tfr2.shape == (len(picks), len(freqs), 2))

    single_power = tfr_morlet(epochs, freqs, 2, average=False,
                              return_itc=False).data
    single_power2 = tfr_morlet(epochs, freqs, 2, decim=slice(0, 2),
                               average=False, return_itc=False).data
    single_power3 = tfr_morlet(epochs, freqs, 2, decim=slice(1, 3),
                               average=False, return_itc=False).data
    single_power4 = tfr_morlet(epochs, freqs, 2, decim=slice(2, 4),
                               average=False, return_itc=False).data

    assert_array_almost_equal(np.mean(single_power, axis=0), power.data)
    assert_array_almost_equal(np.mean(single_power2, axis=0),
                              power.data[:, :, :2])
    assert_array_almost_equal(np.mean(single_power3, axis=0),
                              power.data[:, :, 1:3])
    assert_array_almost_equal(np.mean(single_power4, axis=0),
                              power.data[:, :, 2:4])

    power_pick = power.pick_channels(power.ch_names[:10:2])
    assert_equal(len(power_pick.ch_names), len(power.ch_names[:10:2]))
    assert_equal(power_pick.data.shape[0], len(power.ch_names[:10:2]))
    power_drop = power.drop_channels(power.ch_names[1:10:2])
    assert_equal(power_drop.ch_names, power_pick.ch_names)
    assert_equal(power_pick.data.shape[0], len(power_drop.ch_names))

    mne.equalize_channels([power_pick, power_drop])
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
    assert_raises(ValueError, cwt, data[0, :, :], Ws, mode='foo')
    for use_fft in [True, False]:
        for mode in ['same', 'valid', 'full']:
            # XXX JRK: full wavelet decomposition needs to be implemented
            if (not use_fft) and mode == 'full':
                assert_raises(ValueError, cwt, data[0, :, :], Ws,
                              use_fft=use_fft, mode=mode)
                continue
            cwt(data[0, :, :], Ws, use_fft=use_fft, mode=mode)

    # Test decim parameter checks
    assert_raises(TypeError, tfr_morlet, epochs, freqs=freqs,
                  n_cycles=n_cycles, use_fft=True, return_itc=True,
                  decim='decim')


def test_dpsswavelet():
    """Test DPSS tapers."""
    freqs = np.arange(5, 25, 3)
    Ws = _make_dpss(1000, freqs=freqs, n_cycles=freqs / 2., time_bandwidth=4.0,
                    zero_mean=True)

    assert_true(len(Ws) == 3)  # 3 tapers expected

    # Check that zero mean is true
    assert_true(np.abs(np.mean(np.real(Ws[0][0]))) < 1e-5)

    assert_true(len(Ws[0]) == len(freqs))  # As many wavelets as asked for


@slow_test
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
    t = np.arange(n_times, dtype=np.float) / sfreq
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

    freqs = np.arange(35, 70, 5, dtype=np.float)

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

    assert_raises(ValueError, tfr_multitaper, epochs,
                  freqs=freqs, n_cycles=freqs / 2.,
                  return_itc=True, average=False)

    # test picks argument
    assert_array_almost_equal(power.data, power_picks.data)
    assert_array_almost_equal(power.data, power_averaged.data)
    assert_array_almost_equal(power.times, power_epochs.times)
    assert_array_almost_equal(power.times, power_averaged.times)
    assert_equal(power.nave, power_averaged.nave)
    assert_equal(power_epochs.data.shape, (3, 2, 7, 200))
    assert_array_almost_equal(itc.data, itc_picks.data)
    # one is squared magnitude of the average (evoked) and
    # the other is average of the squared magnitudes (epochs PSD)
    # so values shouldn't match, but shapes should
    assert_array_equal(power.data.shape, power_evoked.data.shape)
    assert_raises(AssertionError, assert_array_almost_equal,
                  power.data, power_evoked.data)

    tmax = t[np.argmax(itc.data[0, freqs == 50, :])]
    fmax = freqs[np.argmax(power.data[1, :, t == 0.5])]
    assert_true(tmax > 0.3 and tmax < 0.7)
    assert_false(np.any(itc.data < 0.))
    assert_true(fmax > 40 and fmax < 60)
    assert_true(power2.data.shape == (len(picks), len(freqs), 2))
    assert_true(power2.data.shape == itc2.data.shape)

    # Test decim parameter checks and compatibility between wavelets length
    # and instance length in the time dimension.
    assert_raises(TypeError, tfr_multitaper, epochs, freqs=freqs,
                  n_cycles=freqs / 2., time_bandwidth=4.0, decim=(1,))
    assert_raises(ValueError, tfr_multitaper, epochs, freqs=freqs,
                  n_cycles=1000, time_bandwidth=4.0)


def test_crop():
    """Test TFR cropping."""
    data = np.zeros((3, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])
    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.crop(0.2, 0.3)
    assert_array_equal(tfr.times, [0.2, 0.3])
    assert_equal(tfr.data.shape[-1], 2)


@requires_h5py
def test_io():
    """Test TFR IO capacities."""

    tempdir = _TempDir()
    fname = op.join(tempdir, 'test-tfr.h5')
    data = np.zeros((3, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])

    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.save(fname)
    tfr2 = read_tfrs(fname, condition='test')

    assert_array_equal(tfr.data, tfr2.data)
    assert_array_equal(tfr.times, tfr2.times)
    assert_array_equal(tfr.freqs, tfr2.freqs)
    assert_equal(tfr.comment, tfr2.comment)
    assert_equal(tfr.nave, tfr2.nave)

    assert_raises(IOError, tfr.save, fname)

    tfr.comment = None
    tfr.save(fname, overwrite=True)
    assert_equal(read_tfrs(fname, condition=0).comment, tfr.comment)
    tfr.comment = 'test-A'
    tfr2.comment = 'test-B'

    fname = op.join(tempdir, 'test2-tfr.h5')
    write_tfrs(fname, [tfr, tfr2])
    tfr3 = read_tfrs(fname, condition='test-A')
    assert_equal(tfr.comment, tfr3.comment)

    assert_true(isinstance(tfr.info, mne.Info))

    tfrs = read_tfrs(fname, condition=None)
    assert_equal(len(tfrs), 2)
    tfr4 = tfrs[1]
    assert_equal(tfr2.comment, tfr4.comment)

    assert_raises(ValueError, read_tfrs, fname, condition='nonono')


def test_plot():
    """Test TFR plotting."""
    import matplotlib.pyplot as plt

    data = np.zeros((3, 2, 3))
    times = np.array([.1, .2, .3])
    freqs = np.array([.10, .20])
    info = mne.create_info(['MEG 001', 'MEG 002', 'MEG 003'], 1000.,
                           ['mag', 'mag', 'mag'])
    tfr = AverageTFR(info, data=data, times=times, freqs=freqs,
                     nave=20, comment='test', method='crazy-tfr')
    tfr.plot([1, 2], title='title', colorbar=False)
    plt.close('all')
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax3 = plt.subplot2grid((2, 2), (0, 1))
    tfr.plot(picks=[0, 1, 2], axes=[ax, ax2, ax3])
    plt.close('all')

    tfr.plot_topo(picks=[1, 2])
    plt.close('all')

    tfr.plot_topo(picks=[1, 2])
    plt.close('all')

    fig = tfr.plot(picks=[1], cmap='RdBu_r')  # interactive mode on by default
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event(' ')
    fig.canvas.key_press_event('down')

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
    assert_true(all(ch in tfr_new.ch_names
                    for ch in tfr_stim.ch_names + tfr_meg.ch_names))
    tfr_new = tfr_meg.copy().add_channels([tfr_eeg])

    assert_true(ch in tfr_new.ch_names for ch in tfr.ch_names)
    assert_array_equal(tfr_new.data, tfr_eeg_meg.data)
    assert_true(all(ch not in tfr_new.ch_names
                    for ch in tfr_stim.ch_names))

    # Now test errors
    tfr_badsf = tfr_eeg.copy()
    tfr_badsf.info['sfreq'] = 3.1415927
    tfr_eeg = tfr_eeg.crop(-.1, .1)

    assert_raises(RuntimeError, tfr_meg.add_channels, [tfr_badsf])
    assert_raises(AssertionError, tfr_meg.add_channels, [tfr_eeg])
    assert_raises(ValueError, tfr_meg.add_channels, [tfr_meg])
    assert_raises(AssertionError, tfr_meg.add_channels, tfr_badsf)


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
            assert_raises(NotImplementedError, func, data, sfreq=sfreq,
                          frequencies=freqs, output=output)
            continue

        # Check runs
        out = func(data, sfreq=sfreq, frequencies=freqs, use_fft=use_fft,
                   zero_mean=zero_mean, n_cycles=2., output=output)
        # Check shapes
        shape = np.r_[data.shape[:2], len(freqs), data.shape[2]]
        if ('avg' in output) or ('itc' in output):
            assert_array_equal(shape[1:], out.shape)
        else:
            assert_array_equal(shape, out.shape)

        # Check types
        if output in ('complex', 'avg_power_itc'):
            assert_equal(np.complex, out.dtype)
        else:
            assert_equal(np.float, out.dtype)
        assert_true(np.all(np.isfinite(out)))

    # Check errors params
    for _data in (None, 'foo', data[0]):
        assert_raises(ValueError, _compute_tfr, _data, freqs, sfreq)
    for _freqs in (None, 'foo', [[0]]):
        assert_raises(ValueError, _compute_tfr, data, _freqs, sfreq)
    for _sfreq in (None, 'foo'):
        assert_raises(ValueError, _compute_tfr, data, freqs, _sfreq)
    for key in ('output', 'method', 'use_fft', 'decim', 'n_jobs'):
        for value in (None, 'foo'):
            kwargs = {key: value}  # FIXME pep8
            assert_raises(ValueError, _compute_tfr, data, freqs, sfreq,
                          **kwargs)

    # No time_bandwidth param in morlet
    assert_raises(ValueError, _compute_tfr, data, freqs, sfreq,
                  method='morlet', time_bandwidth=1)
    # No phase in multitaper XXX Check ?
    assert_raises(NotImplementedError, _compute_tfr, data, freqs, sfreq,
                  method='multitaper', output='phase')

    # Inter-trial coherence tests
    out = _compute_tfr(data, freqs, sfreq, output='itc', n_cycles=2.)
    assert_true(np.sum(out >= 1) == 0)
    assert_true(np.sum(out <= 0) == 0)

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

run_tests_if_main()
