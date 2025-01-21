# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

import mne
from mne import (
    Epochs,
    EpochsArray,
    create_info,
    pick_types,
    read_events,
)
from mne.epochs import equalize_epoch_counts
from mne.io import read_raw_fif
from mne.time_frequency import (
    AverageTFR,
    AverageTFRArray,
    EpochsSpectrum,
    EpochsTFR,
    EpochsTFRArray,
    RawTFR,
    RawTFRArray,
    tfr_array_morlet,
    tfr_array_multitaper,
)
from mne.time_frequency.tfr import (
    _compute_tfr,
    _make_dpss,
    combine_tfr,
    cwt,
    fwhm,
    morlet,
    read_tfrs,
    tfr_morlet,
    tfr_multitaper,
    write_tfrs,
)
from mne.utils import catch_logging, grand_average
from mne.utils._testing import _get_suptitle
from mne.viz.utils import (
    _channel_type_prettyprint,
    _fake_click,
    _fake_keypress,
    _fake_scroll,
)

from .test_spectrum import _get_inst

data_path = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_path / "test_raw.fif"
event_fname = data_path / "test-eve.fif"
raw_ctf_fname = data_path / "test_ctf_raw.fif"

freqs_linspace = np.linspace(20, 40, num=5)
freqs_unsorted_list = [26, 33, 41, 20]
mag_names = [f"MEG 01{n}1" for n in (1, 2, 3)]

parametrize_morlet_multitaper = pytest.mark.parametrize(
    "method", ("morlet", "multitaper")
)
parametrize_power_phase_complex = pytest.mark.parametrize(
    "output", ("power", "phase", "complex")
)
parametrize_inst_and_ch_type = pytest.mark.parametrize(
    "inst,ch_type",
    (
        pytest.param("raw_tfr", "mag"),
        pytest.param("raw_tfr", "grad"),
        pytest.param("epochs_tfr", "mag"),  # no grad pairs in epochs fixture
        pytest.param("average_tfr", "mag"),
        pytest.param("average_tfr", "grad"),
    ),
)


def test_tfr_ctf():
    """Test that TFRs can be calculated on CTF data."""
    raw = read_raw_fif(raw_ctf_fname).crop(0, 1)
    raw.apply_gradient_compensation(3)
    events = mne.make_fixed_length_events(raw, duration=0.5)
    epochs = mne.Epochs(raw, events)
    for method in (tfr_multitaper, tfr_morlet):
        method(epochs, [10], 1)  # smoke test


# Copied from SciPy before it was removed
def _morlet2(M, s, w=5):
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi ** (-0.25)
    output = np.sqrt(1 / s) * wavelet
    return output


@pytest.mark.parametrize("sfreq", [1000.0, 100 + np.pi])
@pytest.mark.parametrize("freq", [10.0, np.pi])
@pytest.mark.parametrize("n_cycles", [7, 2])
def test_morlet(sfreq, freq, n_cycles):
    """Test morlet with and without zero mean."""
    Wz = morlet(sfreq, freq, n_cycles, zero_mean=True)
    W = morlet(sfreq, freq, n_cycles, zero_mean=False)

    assert np.abs(np.mean(np.real(Wz))) < 1e-5
    if n_cycles == 2:
        assert np.abs(np.mean(np.real(W))) > 1e-3
    else:
        assert np.abs(np.mean(np.real(W))) < 1e-5

    assert_allclose(np.linalg.norm(W), np.sqrt(2), atol=1e-6)

    # Convert to SciPy nomenclature and compare
    M = len(W)
    w = n_cycles
    s = w * sfreq / (2 * freq * np.pi)  # from SciPy docs
    Ws = _morlet2(M, s, w) * np.sqrt(2)
    assert_allclose(W, Ws)

    # Check FWHM
    fwhm_formula = fwhm(freq, n_cycles)
    half_max = np.abs(W).max() / 2.0
    fwhm_empirical = (np.abs(W) >= half_max).sum() / sfreq
    # Could be off by a few samples
    assert_allclose(fwhm_formula, fwhm_empirical, atol=3 / sfreq)


def test_tfr_morlet():
    """Test time-frequency transform (PSD and ITC)."""
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    # Setup for reading the raw data
    raw = read_raw_fif(raw_fname)
    events = read_events(event_fname)

    include = []
    exclude = raw.info["bads"] + ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg="grad", eeg=False, stim=False, include=include, exclude=exclude
    )

    picks = picks[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()
    times = epochs.times
    nave = len(data)

    epochs_nopicks = Epochs(raw, events, event_id, tmin, tmax)

    freqs = np.arange(6, 20, 5)  # define frequencies of interest
    n_cycles = freqs / 4.0

    # Test first with a single epoch
    power, itc = tfr_morlet(
        epochs[0], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True
    )

    # Now compute evoked
    evoked = epochs.average()
    with pytest.raises(ValueError, match="Inter-trial coherence is not supported with"):
        tfr_morlet(evoked, freqs, n_cycles=1.0, return_itc=True)
    power, itc = tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True
    )
    power_, itc_ = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        decim=slice(0, 2),
    )
    # Test picks argument and average parameter
    pytest.raises(
        ValueError,
        tfr_morlet,
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=True,
        average=False,
    )

    power_picks, itc_picks = tfr_morlet(
        epochs_nopicks,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        picks=picks,
        average=True,
    )

    epochs_power_picks = tfr_morlet(
        epochs_nopicks,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        picks=picks,
        average=False,
    )
    assert_allclose(epochs_power_picks.data[0, 0, 0, 0], 9.130315e-23, rtol=1e-4)
    power_picks_avg = epochs_power_picks.average()
    # the actual data arrays here are equivalent, too...
    assert_allclose(power.data, power_picks.data)
    assert_allclose(power.data, power_picks_avg.data)
    assert_allclose(itc.data, itc_picks.data)

    # test on evoked
    power_evoked = tfr_morlet(evoked, freqs, n_cycles, use_fft=True, return_itc=False)
    # one is squared magnitude of the average (evoked) and
    # the other is average of the squared magnitudes (epochs PSD)
    # so values shouldn't match, but shapes should
    assert_array_equal(power.data.shape, power_evoked.data.shape)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        assert_allclose(power.data, power_evoked.data)

    # complex output
    with pytest.raises(ValueError, match='must be "power" if average=True'):
        tfr_morlet(
            epochs, freqs, n_cycles, return_itc=False, average=True, output="complex"
        )
    with pytest.raises(ValueError, match="Inter-trial coher.*average=False"):
        tfr_morlet(
            epochs, freqs, n_cycles, return_itc=True, average=False, output="complex"
        )
    epochs_power_complex = tfr_morlet(
        epochs, freqs, n_cycles, return_itc=False, average=False, output="complex"
    )
    epochs_amplitude_2 = abs(epochs_power_complex)
    epochs_amplitude_3 = epochs_amplitude_2.copy()
    epochs_amplitude_3.data[:] = np.inf  # test that it's actually copied

    # test that the power computed via `complex` is equivalent to power
    # computed within the method.
    assert_allclose(epochs_amplitude_2.data**2, epochs_power_picks.data)

    # test that aggregating power across tapers when multitaper with
    # output='complex' gives the same as output='power'
    epoch_data = epochs.get_data()
    multitaper_power = tfr_array_multitaper(
        epoch_data, epochs.info["sfreq"], freqs, n_cycles, output="power"
    )
    multitaper_complex, weights = tfr_array_multitaper(
        epoch_data,
        epochs.info["sfreq"],
        freqs,
        n_cycles,
        output="complex",
        return_weights=True,
    )

    weights = np.expand_dims(weights, axis=(0, 1, -1))  # match shape of complex data
    tfr = weights * multitaper_complex
    tfr = (tfr * tfr.conj()).real.sum(axis=2)
    power_from_complex = tfr * (2 / (weights * weights.conj()).real.sum(axis=2))
    assert_allclose(power_from_complex, multitaper_power)

    print(itc)  # test repr
    print(itc.ch_names)  # test property
    itc += power  # test add
    itc -= power  # test sub
    ret = itc * 23  # test mult
    itc = ret / 23  # test dic

    power = power.apply_baseline(baseline=(-0.1, 0), mode="logratio")
    assert power.baseline == (-0.1, 0)

    assert "meg" in power
    assert "grad" in power
    assert "mag" not in power
    assert "eeg" not in power

    assert power.nave == nave
    assert itc.nave == nave
    assert power.data.shape == (len(picks), len(freqs), len(times))
    assert power.data.shape == itc.data.shape
    assert power_.data.shape == (len(picks), len(freqs), 2)
    assert power_.data.shape == itc_.data.shape
    assert np.sum(itc.data >= 1) == 0
    assert np.sum(itc.data <= 0) == 0

    # grand average
    itc2 = itc.copy()
    itc2.info["bads"] = [itc2.ch_names[0]]  # test channel drop
    gave = grand_average([itc2, itc])
    assert gave.data.shape == (
        itc2.data.shape[0] - 1,
        itc2.data.shape[1],
        itc2.data.shape[2],
    )
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
    assert_allclose(combined_itc.data, np.ones(combined_itc.data.shape) * 2 / 3)

    # more tests
    power, itc = tfr_morlet(
        epochs, freqs=freqs, n_cycles=2, use_fft=False, return_itc=True
    )

    assert power.data.shape == (len(picks), len(freqs), len(times))
    assert power.data.shape == itc.data.shape
    assert np.sum(itc.data >= 1) == 0
    assert np.sum(itc.data <= 0) == 0

    tfr = tfr_morlet(
        epochs[0], freqs, use_fft=True, n_cycles=2, average=False, return_itc=False
    )
    tfr_data = tfr.data[0]
    assert tfr_data.shape == (len(picks), len(freqs), len(times))
    tfr2 = tfr_morlet(
        epochs[0],
        freqs,
        use_fft=True,
        n_cycles=2,
        decim=slice(0, 2),
        average=False,
        return_itc=False,
    ).data[0]
    assert tfr2.shape == (len(picks), len(freqs), 2)

    single_power = tfr_morlet(epochs, freqs, 2, average=False, return_itc=False).data
    single_power2 = tfr_morlet(
        epochs, freqs, 2, decim=slice(0, 2), average=False, return_itc=False
    ).data
    single_power3 = tfr_morlet(
        epochs, freqs, 2, decim=slice(1, 3), average=False, return_itc=False
    ).data
    single_power4 = tfr_morlet(
        epochs, freqs, 2, decim=slice(2, 4), average=False, return_itc=False
    ).data

    assert_allclose(np.mean(single_power, axis=0), power.data)
    assert_allclose(np.mean(single_power2, axis=0), power.data[:, :, :2])
    assert_allclose(np.mean(single_power3, axis=0), power.data[:, :, 1:3])
    assert_allclose(np.mean(single_power4, axis=0), power.data[:, :, 2:4])

    power_pick = power.pick(power.ch_names[:10:2])
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
            power, itc = tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=2,
                use_fft=use_fft,
                return_itc=True,
                decim=decim,
            )
            assert_equal(power.data.shape[2], np.ceil(float(len(times)) / decim))
    freqs = list(range(50, 55))
    decim = 2
    _, n_chan, n_time = data.shape
    tfr = tfr_morlet(
        epochs[0], freqs, 2.0, decim=decim, average=False, return_itc=False
    ).data[0]
    assert_equal(tfr.shape, (n_chan, len(freqs), n_time // decim))

    # Test cwt modes
    Ws = morlet(512, [10, 20], n_cycles=2)
    pytest.raises(ValueError, cwt, data[0, :, :], Ws, mode="foo")
    for use_fft in [True, False]:
        for mode in ["same", "valid", "full"]:
            cwt(data[0], Ws, use_fft=use_fft, mode=mode)

    # Test invalid frequency arguments
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_morlet(epochs, freqs=np.arange(0, 3), n_cycles=7)
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_morlet(epochs, freqs=np.arange(-4, -1), n_cycles=7)

    # Test decim parameter checks
    pytest.raises(
        TypeError,
        tfr_morlet,
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        decim="decim",
    )

    # When convolving in time, wavelets must not be longer than the data
    pytest.raises(ValueError, cwt, data[0, :, : Ws[0].size - 1], Ws, use_fft=False)
    with pytest.warns(UserWarning, match="one of the wavelets.*is longer"):
        cwt(data[0, :, : Ws[0].size - 1], Ws, use_fft=True)

    # Check for off-by-one errors when using wavelets with an even number of
    # samples
    psd = cwt(data[0], [Ws[0][:-1]], use_fft=False, mode="full")
    assert_equal(psd.shape, (2, 1, 420))


def test_dpsswavelet():
    """Test DPSS tapers."""
    freqs = np.arange(5, 25, 3)
    Ws, weights = _make_dpss(
        1000,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        time_bandwidth=4.0,
        zero_mean=True,
        return_weights=True,
    )

    assert np.shape(Ws)[:2] == (3, len(freqs))  # 3 tapers expected
    assert np.shape(Ws)[:2] == np.shape(weights)  # weights of shape (tapers, freqs)

    # Check that zero mean is true
    assert np.abs(np.mean(np.real(Ws[0][0]))) < 1e-5


@pytest.mark.slowtest
def test_tfr_multitaper():
    """Test tfr_multitaper."""
    sfreq = 200.0
    ch_names = ["SIM0001", "SIM0002"]
    ch_types = ["grad", "grad"]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    n_times = int(sfreq)  # Second long epochs
    n_epochs = 3
    seed = 42
    rng = np.random.RandomState(seed)
    noise = 0.1 * rng.randn(n_epochs, len(ch_names), n_times)
    t = np.arange(n_times, dtype=np.float64) / sfreq
    signal = np.sin(np.pi * 2.0 * 50.0 * t)  # 50 Hz sinusoid signal
    signal[np.logical_or(t < 0.45, t > 0.55)] = 0.0  # Hard windowing
    on_time = np.logical_and(t >= 0.45, t <= 0.55)
    signal[on_time] *= np.hanning(on_time.sum())  # Ramping
    dat = noise + signal

    reject = dict(grad=4000.0)
    events = np.empty((n_epochs, 3), int)
    first_event_sample = 100
    event_id = dict(sin50hz=1)
    for k in range(n_epochs):
        events[k, :] = first_event_sample + k * n_times, 0, event_id["sin50hz"]

    epochs = EpochsArray(
        data=dat, info=info, events=events, event_id=event_id, reject=reject
    )

    freqs = np.arange(35, 70, 5, dtype=np.float64)

    power, itc = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=freqs / 2.0, time_bandwidth=4.0
    )
    power2, itc2 = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=freqs / 2.0, time_bandwidth=4.0, decim=slice(0, 2)
    )
    picks = np.arange(len(ch_names))
    power_picks, itc_picks = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=freqs / 2.0, time_bandwidth=4.0, picks=picks
    )
    power_epochs = tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        time_bandwidth=4.0,
        return_itc=False,
        average=False,
    )
    power_averaged = power_epochs.average()
    power_evoked = tfr_multitaper(
        epochs.average(),
        freqs=freqs,
        n_cycles=freqs / 2.0,
        time_bandwidth=4.0,
        return_itc=False,
        average=False,
    ).average()

    print(power_evoked)  # test repr for EpochsTFR

    # Test channel picking
    power_epochs_picked = power_epochs.copy().drop_channels(["SIM0002"])
    assert_equal(power_epochs_picked.data.shape, (3, 1, 7, 200))
    assert_equal(power_epochs_picked.ch_names, ["SIM0001"])

    pytest.raises(
        ValueError,
        tfr_multitaper,
        epochs,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        return_itc=True,
        average=False,
    )

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
    pytest.raises(AssertionError, assert_allclose, power.data, power_evoked.data)

    tmax = t[np.argmax(itc.data[0, freqs == 50, :])]
    fmax = freqs[np.argmax(power.data[1, :, t == 0.5])]
    assert tmax > 0.3 and tmax < 0.7
    assert not np.any(itc.data < 0.0)
    assert fmax > 40 and fmax < 60
    assert power2.data.shape == (len(picks), len(freqs), 2)
    assert power2.data.shape == itc2.data.shape

    # Test decim parameter checks and compatibility between wavelets length
    # and instance length in the time dimension.
    pytest.raises(
        TypeError,
        tfr_multitaper,
        epochs,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        time_bandwidth=4.0,
        decim=(1,),
    )
    pytest.raises(
        ValueError,
        tfr_multitaper,
        epochs,
        freqs=freqs,
        n_cycles=1000,
        time_bandwidth=4.0,
    )

    # Test invalid frequency arguments
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_multitaper(epochs, freqs=np.arange(0, 3), n_cycles=7)
    with pytest.raises(ValueError, match=" 'freqs' must be greater than 0"):
        tfr_multitaper(epochs, freqs=np.arange(-4, -1), n_cycles=7)


@pytest.mark.parametrize(
    "method,freqs",
    (
        pytest.param("morlet", freqs_linspace, id="morlet"),
        pytest.param("multitaper", freqs_linspace, id="multitaper"),
        pytest.param("stockwell", freqs_linspace[[0, -1]], id="stockwell"),
    ),
)
@pytest.mark.parametrize("decim", (4, slice(0, 200), slice(1, 200, 3)))
def test_tfr_decim_and_shift_time(epochs, method, freqs, decim):
    """Test TFR decimation; slices must be long-ish to be longer than the wavelets."""
    tfr = epochs.compute_tfr(method, freqs=freqs, decim=decim)
    if not isinstance(decim, slice):
        decim = slice(None, None, decim)
    # check n_times
    want = len(range(*decim.indices(len(epochs.times))))
    assert tfr.shape[-1] == want
    # Check that decim changes sfreq
    assert tfr.sfreq == epochs.info["sfreq"] / (decim.step or 1)
    # check after-the-fact decimation. The mixin .decimate method doesn't allow slices
    if isinstance(decim, int):
        tfr2 = epochs.compute_tfr(method, freqs=freqs, decim=1)
        tfr2.decimate(decim)
        assert tfr == tfr2
    # test .shift_time() too
    shift = -0.137
    data, times, freqs = tfr.get_data(return_times=True, return_freqs=True)
    tfr.shift_time(shift, relative=True)
    assert_allclose(times + shift, tfr.times, rtol=0, atol=0.5 / tfr.sfreq)
    # shift time should only affect times:
    assert_array_equal(data, tfr.get_data())
    assert_array_equal(freqs, tfr.freqs)


@pytest.mark.parametrize("inst", ("raw_tfr", "epochs_tfr", "average_tfr"))
def test_tfr_io(inst, average_tfr, request, tmp_path):
    """Test TFR I/O."""
    pytest.importorskip("h5io")
    pd = pytest.importorskip("pandas")

    tfr = _get_inst(inst, request, average_tfr=average_tfr)
    fname = tmp_path / "temp_tfr.hdf5"
    # test .save() method
    tfr.save(fname, overwrite=True)
    assert read_tfrs(fname) == tfr
    # test save single TFR with write_tfrs()
    write_tfrs(fname, tfr, overwrite=True)
    assert read_tfrs(fname) == tfr
    # test save multiple TFRs with write_tfrs()
    tfr2 = tfr.copy()
    tfr2._data = np.zeros_like(tfr._data)
    write_tfrs(fname, [tfr, tfr2], overwrite=True)
    tfr_list = read_tfrs(fname)
    assert tfr_list[0] == tfr
    assert tfr_list[1] == tfr2
    # test condition-related errors
    if isinstance(tfr, AverageTFR):
        # auto-generated keys: first TFR has comment, so `0` not assigned
        tfr2.comment = None
        write_tfrs(fname, [tfr, tfr2], overwrite=True)
        with pytest.raises(ValueError, match='Cannot find condition "0" in this'):
            read_tfrs(fname, condition=0)
        # second TFR had no comment, so should get auto-comment `1` assigned
        read_tfrs(fname, condition=1)
        return
    else:
        with pytest.raises(NotImplementedError, match="condition is only supported"):
            read_tfrs(fname, condition="foo")
    # the rest we only do for EpochsTFR (no need to parametrize)
    if isinstance(tfr, RawTFR):
        return
    # make sure everything still works if there's metadata
    tfr.metadata = pd.DataFrame(dict(foo=range(tfr.shape[0])), index=tfr.selection)
    # test old-style meas date
    sec_microsec_tuple = (1, 2)
    with tfr.info._unlock():
        tfr.info["meas_date"] = sec_microsec_tuple
    tfr.save(fname, overwrite=True)
    tfr_loaded = read_tfrs(fname)
    want = datetime.datetime(
        year=1970,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=sec_microsec_tuple[0],
        microsecond=sec_microsec_tuple[1],
        tzinfo=datetime.timezone.utc,
    )
    assert tfr_loaded.info["meas_date"] == want
    with tfr.info._unlock():
        tfr.info["meas_date"] = want
    assert tfr_loaded == tfr
    # test with taper dimension and weights
    n_tapers = 3  # anything >= 1 should do
    weights = np.ones((n_tapers, tfr.shape[2]))  # tapers x freqs
    state = tfr.__getstate__()
    state["data"] = np.repeat(np.expand_dims(tfr.data, 2), n_tapers, axis=2)  # add dim
    state["weights"] = weights  # add weights
    state["dims"] = ("epoch", "channel", "taper", "freq", "time")  # update dims
    tfr = EpochsTFR(inst=state)
    tfr.save(fname, overwrite=True)
    tfr_loaded = read_tfrs(fname)
    assert tfr_loaded == tfr
    # test overwrite
    with pytest.raises(OSError, match="Destination file exists."):
        tfr.save(fname, overwrite=False)


def test_roundtrip_from_legacy_func(epochs, tmp_path):
    """Test save/load with TFRs generated by legacy method (gh-12512)."""
    pytest.importorskip("h5io")

    fname = tmp_path / "temp_tfr.hdf5"
    tfr = tfr_morlet(
        epochs, freqs=freqs_linspace, n_cycles=7, average=True, return_itc=False
    )
    tfr.save(fname, overwrite=True)
    assert read_tfrs(fname) == tfr


def test_raw_tfr_init(raw):
    """Test the RawTFR and RawTFRArray constructors."""
    one = RawTFR(inst=raw, method="morlet", freqs=freqs_linspace)
    two = RawTFRArray(one.info, one.data, one.times, one.freqs, method="morlet")
    # some attributes we know won't match:
    for attr in ("_data_type", "_inst_type"):
        assert getattr(one, attr) != getattr(two, attr)
        delattr(one, attr)
        delattr(two, attr)
    assert one == two
    # test RawTFR.__getitem__
    data = one[:5]
    assert data.shape == (5,) + one.shape[1:]
    # test missing method/freqs
    with pytest.raises(ValueError, match="RawTFR got unsupported parameter value"):
        RawTFR(inst=raw)


def test_average_tfr_init(full_evoked):
    """Test the AverageTFR and AverageTFRArray constructors."""
    one = AverageTFR(inst=full_evoked, method="morlet", freqs=freqs_linspace)
    two = AverageTFRArray(
        one.info,
        one.data,
        one.times,
        one.freqs,
        method="morlet",
        comment=one.comment,
        nave=one.nave,
    )
    # some attributes we know won't match, otherwise should be identical
    assert one._data_type != two._data_type
    one._data_type = two._data_type
    assert one == two
    # test missing method, bad freqs
    with pytest.raises(ValueError, match="AverageTFR got unsupported parameter value"):
        AverageTFR(inst=full_evoked)
    with pytest.raises(ValueError, match='must be a length-2 iterable or "auto"'):
        AverageTFR(inst=full_evoked, method="stockwell", freqs=freqs_linspace)


@pytest.mark.parametrize("inst", ("raw_tfr", "epochs_tfr", "average_tfr"))
def test_tfr_init_errors(inst, request, average_tfr):
    """Test __init__ for {Raw,Epochs,Average}TFR."""
    # Load data
    inst = _get_inst(inst, request, average_tfr=average_tfr)
    state = inst.__getstate__()
    # Prepare for TFRArray object instantiation
    inst_name = inst.__class__.__name__
    class_mapping = dict(RawTFR=RawTFR, EpochsTFR=EpochsTFR, AverageTFR=AverageTFR)
    ndims_mapping = dict(
        RawTFR=("3D or 4D"), EpochsTFR=("4D or 5D"), AverageTFR=("3D or 4D")
    )
    TFR = class_mapping[inst_name]
    allowed_ndims = ndims_mapping[inst_name]
    # Check errors caught
    with pytest.raises(ValueError, match=f".*TFR data should be {allowed_ndims}"):
        TFR(inst=state | dict(data=inst.data[..., 0]))
    with pytest.raises(ValueError, match=f".*TFR data should be {allowed_ndims}"):
        TFR(inst=state | dict(data=np.expand_dims(inst.data, axis=(0, 1))))
    with pytest.raises(ValueError, match="Channel axis of data .* doesn't match info"):
        TFR(inst=state | dict(data=inst.data[..., :-1, :, :]))
    with pytest.raises(ValueError, match="Time axis of data.*doesn't match times attr"):
        TFR(inst=state | dict(times=inst.times[:-1]))
    with pytest.raises(ValueError, match="Frequency axis of.*doesn't match freqs attr"):
        TFR(inst=state | dict(freqs=inst.freqs[:-1]))


@pytest.mark.parametrize(
    "method,freqs,match",
    (
        ("morlet", None, "EpochsTFR got unsupported parameter value freqs=None."),
        (None, freqs_linspace, "got unsupported parameter value method=None."),
        (None, None, "got unsupported parameter values method=None and freqs=None."),
    ),
)
def test_compute_tfr_init_errors(epochs, method, freqs, match):
    """Test that method and freqs are always passed (if not using __setstate__)."""
    with pytest.raises(ValueError, match=match):
        epochs.compute_tfr(method=method, freqs=freqs)


def test_equalize_epochs_tfr_counts(epochs_tfr):
    """Test equalize_epoch_counts for EpochsTFR."""
    # make the fixture have 3 epochs instead of 1
    epochs_tfr._data = np.vstack((epochs_tfr._data, epochs_tfr._data, epochs_tfr._data))
    tfr2 = epochs_tfr.copy()
    tfr2 = tfr2[:-1]
    equalize_epoch_counts([epochs_tfr, tfr2])
    assert epochs_tfr.shape == tfr2.shape


def test_dB_computation():
    """Test dB computation in plot methods (gh 11091)."""
    ampl = 2.0
    data = np.full((3, 2, 3), ampl**2)  # already power
    complex_data = np.full((3, 2, 3), ampl + 0j)  # ampl → power when plotting
    times = np.array([0.1, 0.2, 0.3])
    freqs = np.array([0.10, 0.20])
    info = mne.create_info(
        ["MEG 001", "MEG 002", "MEG 003"], 1000.0, ["mag", "mag", "mag"]
    )
    kwargs = dict(times=times, freqs=freqs, nave=20, comment="test", method="crazy-tfr")
    tfr = AverageTFRArray(info=info, data=data, **kwargs)
    complex_tfr = AverageTFRArray(info=info, data=complex_data, **kwargs)
    plot_kwargs = dict(dB=True, combine="mean", vlim=(0, 7))
    fig1 = tfr.plot(**plot_kwargs)[0]
    fig2 = complex_tfr.plot(**plot_kwargs)[0]
    # since we're fixing vmin/vmax, equal colors should mean ~equal input data
    quadmesh1 = fig1.axes[0].collections[0]
    quadmesh2 = fig2.axes[0].collections[0]
    if hasattr(quadmesh1, "_mapped_colors"):  # fails on compat/old
        assert_array_equal(quadmesh1._mapped_colors, quadmesh2._mapped_colors)


def test_plot():
    """Test TFR plotting."""
    data = np.zeros((3, 2, 3))
    times = np.array([0.1, 0.2, 0.3])
    freqs = np.array([0.10, 0.20])
    info = mne.create_info(
        ["MEG 001", "MEG 002", "MEG 003"], 1000.0, ["mag", "mag", "mag"]
    )
    tfr = AverageTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        nave=20,
        comment="test",
        method="crazy-tfr",
    )

    # interactive mode on by default
    fig = tfr.plot(picks=[1], cmap="RdBu_r")[0]
    _fake_keypress(fig, "up")
    _fake_keypress(fig, " ")
    _fake_keypress(fig, "down")
    _fake_keypress(fig, " ")
    _fake_keypress(fig, "+")
    _fake_keypress(fig, " ")
    _fake_keypress(fig, "-")
    _fake_keypress(fig, " ")
    _fake_keypress(fig, "pageup")
    _fake_keypress(fig, " ")
    _fake_keypress(fig, "pagedown")

    cbar = fig.get_axes()[0].CB  # Fake dragging with mouse.
    ax = cbar.cbar.ax
    _fake_click(fig, ax, (0.1, 0.1))
    _fake_click(fig, ax, (0.1, 0.2), kind="motion")
    _fake_click(fig, ax, (0.1, 0.3), kind="release")

    _fake_click(fig, ax, (0.1, 0.1), button=3)
    _fake_click(fig, ax, (0.1, 0.2), button=3, kind="motion")
    _fake_click(fig, ax, (0.1, 0.3), kind="release")

    _fake_scroll(fig, 0.5, 0.5, -0.5)  # scroll down
    _fake_scroll(fig, 0.5, 0.5, 0.5)  # scroll up

    plt.close("all")


@pytest.mark.parametrize("output", ("complex", "phase"))
def test_plot_multitaper_complex_phase(output):
    """Test TFR plotting of data with a taper dimension."""
    # Create example data with a taper dimension
    n_chans, n_tapers, n_freqs, n_times = (3, 4, 2, 3)
    data = np.random.rand(n_chans, n_tapers, n_freqs, n_times)
    if output == "complex":
        data = data + np.random.rand(*data.shape) * 1j  # add imaginary data
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)
    weights = np.random.rand(n_tapers, n_freqs)
    info = mne.create_info(n_chans, 1000.0, "eeg")
    tfr = AverageTFRArray(
        info=info, data=data, times=times, freqs=freqs, weights=weights
    )
    # Check that plotting works
    tfr.plot()


@pytest.mark.parametrize(
    "timefreqs,title,combine",
    (
        pytest.param(
            {(0.33, 23): (0, 0), (0.25, 30): (0.1, 2)},
            "0.25 ± 0.05 s,\n30.0 ± 1.0 Hz",
            "mean",
            id="dict,mean",
        ),
        pytest.param([(0.25, 30)], "0.25 s,\n30.0 Hz", "rms", id="list,rms"),
        pytest.param(None, None, lambda x: x.mean(axis=0), id="none,lambda"),
    ),
)
@parametrize_inst_and_ch_type
def test_tfr_plot_joint(
    inst, ch_type, combine, timefreqs, title, full_average_tfr, request
):
    """Test {Raw,Epochs,Average}TFR.plot_joint()."""
    tfr = _get_inst(inst, request, average_tfr=full_average_tfr)
    with catch_logging() as log:
        fig = tfr.plot_joint(
            picks=ch_type,
            timefreqs=timefreqs,
            combine=combine,
            topomap_args=dict(res=8, contours=0, sensors=False),  # for speed
            verbose="debug",
        )
        assert f"Plotting topomap for {ch_type} data" in log.getvalue()
    # check for correct number of axes
    n_topomaps = 1 if timefreqs is None else len(timefreqs)
    assert len(fig.axes) == n_topomaps + 2  # n_topomaps + 1 image + 1 colorbar
    # title varies by `ch_type` when `timefreqs=None`, so we don't test that here
    if title is not None:
        assert fig.axes[0].get_title() == title
    # test interactivity
    ax = [ax for ax in fig.axes if ax.get_xlabel() == "Time (s)"][0]
    kw = dict(fig=fig, ax=ax, xform="ax")
    _fake_click(**kw, kind="press", point=(0.4, 0.4))
    _fake_click(**kw, kind="motion", point=(0.5, 0.5))
    _fake_click(**kw, kind="release", point=(0.6, 0.6))
    # make sure we actually got a pop-up figure, and it has a plausible title
    fignums = plt.get_fignums()
    assert len(fignums) == 2
    popup_fig = plt.figure(fignums[-1])
    assert re.match(
        r"-?\d{1,2}\.\d{3} - -?\d{1,2}\.\d{3} s,\n\d{1,2}\.\d{2} - \d{1,2}\.\d{2} Hz",
        _get_suptitle(popup_fig),
    )


@pytest.mark.parametrize(
    "match,timefreqs,topomap_args",
    (
        (r"Requested time point \(-88.000 s\) exceeds the range of", [(-88, 1)], None),
        (r"Requested frequency \(99.0 Hz\) exceeds the range of", [(0.0, 99)], None),
        ("list of tuple pairs, or a dict of such tuple pairs, not 0", [0.0], None),
        ("does not match the channel type present in", None, dict(ch_type="eeg")),
    ),
)
def test_tfr_plot_joint_errors(full_average_tfr, match, timefreqs, topomap_args):
    """Test AverageTFR.plot_joint() error messages."""
    with pytest.raises(ValueError, match=match):
        full_average_tfr.plot_joint(timefreqs=timefreqs, topomap_args=topomap_args)


def test_tfr_plot_joint_doesnt_modify(full_average_tfr):
    """Test that the object is unchanged after plot_joint()."""
    tfr = full_average_tfr.copy()
    full_average_tfr.plot_joint()
    assert tfr == full_average_tfr


def test_add_channels():
    """Test tfr splitting / re-appending channel types."""
    data = np.zeros((6, 2, 3))
    times = np.array([0.1, 0.2, 0.3])
    freqs = np.array([0.10, 0.20])
    info = mne.create_info(
        ["MEG 001", "MEG 002", "MEG 003", "EEG 001", "EEG 002", "STIM 001"],
        1000.0,
        ["mag", "mag", "mag", "eeg", "eeg", "stim"],
    )
    tfr = AverageTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        nave=20,
        comment="test",
        method="crazy-tfr",
    )
    tfr_eeg = tfr.copy().pick(picks="eeg")
    tfr_meg = tfr.copy().pick(picks="meg")
    tfr_stim = tfr.copy().pick(picks="stim")
    tfr_eeg_meg = tfr.copy().pick(picks=["meg", "eeg"])
    tfr_new = tfr_meg.copy().add_channels([tfr_eeg, tfr_stim])
    assert all(ch in tfr_new.ch_names for ch in tfr_stim.ch_names + tfr_meg.ch_names)
    tfr_new = tfr_meg.copy().add_channels([tfr_eeg])

    have_all = all(ch in tfr_new.ch_names for ch in tfr.ch_names if ch != "STIM 001")
    assert have_all
    assert_array_equal(tfr_new.data, tfr_eeg_meg.data)
    assert all(ch not in tfr_new.ch_names for ch in tfr_stim.ch_names)

    # Now test errors
    tfr_badsf = tfr_eeg.copy()
    with tfr_badsf.info._unlock():
        tfr_badsf.info["sfreq"] = 3.1415927
    tfr_eeg = tfr_eeg.crop(0.1, 0.1)

    pytest.raises(RuntimeError, tfr_meg.add_channels, [tfr_badsf])
    pytest.raises(ValueError, tfr_meg.add_channels, [tfr_eeg])
    pytest.raises(ValueError, tfr_meg.add_channels, [tfr_meg])
    pytest.raises(TypeError, tfr_meg.add_channels, tfr_badsf)

    # Test for EpochsTFR(Array)
    tfr1 = EpochsTFRArray(
        info=mne.create_info(["EEG 001"], 1000, "eeg"),
        data=np.zeros((5, 1, 2, 3)),  # epochs, channels, freqs, times
        times=[0.1, 0.2, 0.3],
        freqs=[0.1, 0.2],
    )
    tfr2 = EpochsTFRArray(
        info=mne.create_info(["EEG 002", "EEG 003"], 1000, "eeg"),
        data=np.zeros((5, 2, 2, 3)),  # epochs, channels, freqs, times
        times=[0.1, 0.2, 0.3],
        freqs=[0.1, 0.2],
    )
    tfr1.add_channels([tfr2])
    assert tfr1.ch_names == ["EEG 001", "EEG 002", "EEG 003"]
    assert tfr1.data.shape == (5, 3, 2, 3)


def test_compute_tfr():
    """Test _compute_tfr function."""
    # Set parameters
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    # Setup for reading the raw data
    raw = read_raw_fif(raw_fname)
    events = read_events(event_fname)

    exclude = raw.info["bads"] + ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg="grad", eeg=False, stim=False, include=[], exclude=exclude
    )

    picks = picks[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    freqs = np.arange(10, 20, 3).astype(float)

    # Check all combination of options
    for func, use_fft, zero_mean, output in product(
        (tfr_array_multitaper, tfr_array_morlet),
        (False, True),
        (False, True),
        ("complex", "power", "phase", "avg_power_itc", "avg_power", "itc"),
    ):
        # Check runs
        out = func(
            data,
            sfreq=sfreq,
            freqs=freqs,
            use_fft=use_fft,
            zero_mean=zero_mean,
            n_cycles=2.0,
            output=output,
        )
        # Check shapes
        if func == tfr_array_multitaper and output in ["complex", "phase"]:
            n_tapers = 3
            shape = np.r_[data.shape[:2], n_tapers, len(freqs), data.shape[2]]
        else:
            shape = np.r_[data.shape[:2], len(freqs), data.shape[2]]
        if ("avg" in output) or ("itc" in output):
            assert_array_equal(shape[1:], out.shape)
        else:
            assert_array_equal(shape, out.shape)

        # Check types
        if output in ("complex", "avg_power_itc"):
            assert_equal(np.complex128, out.dtype)
        else:
            assert_equal(np.float64, out.dtype)
        assert np.all(np.isfinite(out))

    # Check errors params
    for _data in (None, "foo", data[0]):
        pytest.raises(ValueError, _compute_tfr, _data, freqs, sfreq)
    for _freqs in (None, "foo", [[0]]):
        pytest.raises(ValueError, _compute_tfr, data, _freqs, sfreq)
    for _sfreq in (None, "foo"):
        pytest.raises(ValueError, _compute_tfr, data, freqs, _sfreq)
    for key in ("output", "method", "use_fft", "decim", "n_jobs"):
        for value in (None, "foo"):
            kwargs = {key: value}  # FIXME pep8
            pytest.raises(ValueError, _compute_tfr, data, freqs, sfreq, **kwargs)
    with pytest.raises(ValueError, match="above Nyquist"):
        _compute_tfr(data, [sfreq], sfreq)

    # No time_bandwidth param in morlet
    pytest.raises(
        ValueError, _compute_tfr, data, freqs, sfreq, method="morlet", time_bandwidth=1
    )

    # Inter-trial coherence tests
    out = _compute_tfr(data, freqs, sfreq, output="itc", n_cycles=2.0)
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

        for method in ("multitaper", "morlet"):
            # Single trials
            out = _compute_tfr(
                data,
                freqs,
                sfreq,
                method=method,
                decim=decim,
                output="power",
                n_cycles=2.0,
            )
            assert_array_equal(shape, out.shape)
            # Averages
            out = _compute_tfr(
                data,
                freqs,
                sfreq,
                method=method,
                decim=decim,
                output="avg_power",
                n_cycles=2.0,
            )
            assert_array_equal(shape[1:], out.shape)


@pytest.mark.parametrize("method", ("multitaper", "morlet"))
@pytest.mark.parametrize("decim", (1, slice(1, None, 2), 3))
def test_compute_tfr_correct(method, decim):
    """Test that TFR actually gets us our freq back."""
    sfreq = 1000.0
    t = np.arange(1000) / sfreq
    f = 50.0
    data = np.sin(2 * np.pi * f * t)
    data *= np.hanning(data.size)
    data = data[np.newaxis, np.newaxis]
    freqs = np.arange(10, 111, 4)
    assert f in freqs

    # previous n_cycles=2 gives weird results for multitaper
    n_cycles = freqs * 0.25
    tfr = _compute_tfr(
        data,
        freqs,
        sfreq,
        method=method,
        decim=decim,
        n_cycles=n_cycles,
        output="power",
    )[0, 0]
    assert freqs[np.argmax(tfr.mean(-1))] == f


def test_averaging_epochsTFR():
    """Test that EpochsTFR averaging methods work."""
    # Setup for reading the raw data
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    freqs = np.arange(6, 20, 5)  # define frequencies of interest
    n_cycles = freqs / 4.0

    raw = read_raw_fif(raw_fname)
    # only pick a few events for speed
    events = read_events(event_fname)[:4]

    include = []
    exclude = raw.info["bads"] + ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg="grad", eeg=False, stim=False, include=include, exclude=exclude
    )
    picks = picks[:2]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)

    # Obtain EpochsTFR
    power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        use_fft=True,
        return_itc=False,
    )

    # Test average methods
    for func, method in zip(
        [np.mean, np.median, np.mean], ["mean", "median", lambda x: np.mean(x, axis=0)]
    ):
        avgpower = power.average(method=method)
        assert_array_equal(func(power.data, axis=0), avgpower.data)
    with pytest.raises(
        RuntimeError, match=r"EpochsTFR.average\(\) got .* shape \(\), but it should be"
    ):
        power.average(method=np.mean)

    # Check it doesn't run for taper spectra
    tapered = epochs.compute_tfr(
        method="multitaper", freqs=freqs, n_cycles=n_cycles, output="complex"
    )
    with pytest.raises(
        NotImplementedError, match=r"Averaging multitaper tapers .* is not supported."
    ):
        tapered.average()


def test_averaging_freqsandtimes_epochsTFR():
    """Test that EpochsTFR averaging freqs methods work."""
    # Setup for reading the raw data
    event_id = 1
    tmin = -0.2
    tmax = 0.498  # Allows exhaustive decimation testing

    freqs = np.arange(6, 20, 5)  # define frequencies of interest
    n_cycles = freqs / 4.0

    raw = read_raw_fif(raw_fname)
    # only pick a few events for speed
    events = read_events(event_fname)[:4]

    include = []
    exclude = raw.info["bads"] + ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg="grad", eeg=False, stim=False, include=include, exclude=exclude
    )
    picks = picks[:2]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)

    # Obtain EpochsTFR
    power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        use_fft=True,
        return_itc=False,
    )

    # Test averaging over freqs
    kwargs = dict(dim="freqs", copy=True)
    for method, func in zip(
        ("mean", "median", lambda x: np.mean(x, axis=2)), (np.mean, np.median, np.mean)
    ):
        avgpower = power.average(method=method, **kwargs)
        assert_array_equal(avgpower.data, func(power.data, axis=2, keepdims=True))
        assert_array_equal(avgpower.freqs, func(power.freqs, keepdims=True))
        assert isinstance(avgpower, EpochsTFR)
        avgpower = avgpower.average()  # average over epochs
        assert isinstance(avgpower, AverageTFR)
    with pytest.raises(RuntimeError, match=r"shape \(1, 2, 3\), but it should"):
        # collapsing wrong axis (time instead of freq)
        avgpower = power.average(method=lambda x: np.mean(x, axis=3), **kwargs)

    # Test averaging over times
    kwargs = dict(dim="times", copy=False)
    for method, func in zip(
        ("mean", "median", lambda x: np.mean(x, axis=3)), (np.mean, np.median, np.mean)
    ):
        avgpower = power.average(method=method, **kwargs)
        assert_array_equal(avgpower.data, func(power.data, axis=-1, keepdims=False))
        assert isinstance(avgpower, EpochsSpectrum)
    with pytest.raises(RuntimeError, match=r"shape \(1, 2, 420\), but it should"):
        # collapsing wrong axis (freq instead of time)
        avgpower = power.average(method=lambda x: np.mean(x, axis=2), **kwargs)


@pytest.mark.parametrize("n_drop, as_tfr_array", ((0, False), (0, True), (2, False)))
def test_epochstfr_getitem(epochs_full, n_drop, as_tfr_array):
    """Test EpochsTFR.__getitem__()."""
    pd = pytest.importorskip("pandas")
    from pandas.testing import assert_frame_equal

    epochs_full.metadata = pd.DataFrame(dict(foo=list("aaaabbb"), bar=np.arange(7)))
    epochs_full.drop(np.arange(n_drop))
    tfr = epochs_full.compute_tfr(method="morlet", freqs=freqs_linspace)
    if not as_tfr_array:  # check that various attributes are preserved
        assert_frame_equal(tfr.metadata, epochs_full.metadata)
        assert epochs_full.drop_log == tfr.drop_log
        for attr in ("events", "selection", "times"):
            assert_array_equal(getattr(epochs_full, attr), getattr(tfr, attr))
        # test pandas query
        foo_a = tfr["foo == 'a'"]
        bar_3 = tfr["bar <= 3"]
        assert foo_a == bar_3
        assert foo_a.shape[0] == 4 - n_drop
    else:  # repackage to check __getitem__ also works with unspecified events, etc...
        tfr = EpochsTFRArray(
            info=tfr.info, data=tfr.data, times=tfr.times, freqs=tfr.freqs
        )
    # test integer and slice
    subset_ints = tfr[[0, 1, 2]]
    subset_slice = tfr[:3]
    assert subset_ints == subset_slice
    # test iteration
    for ix, epo in enumerate(tfr):
        assert_array_equal(tfr[ix].data, epo.data.obj[np.newaxis])


def test_to_data_frame():
    """Test EpochsTFR Pandas exporter."""
    # Create fake EpochsTFR data:
    pytest.importorskip("pandas")
    n_epos = 3
    ch_names = ["EEG 001", "EEG 002", "EEG 003", "EEG 004"]
    n_picks = len(ch_names)
    ch_types = ["eeg"] * n_picks
    n_tapers = 2
    n_freqs = 5
    n_times = 6
    data = np.random.rand(n_epos, n_picks, n_tapers, n_freqs, n_times)
    times = np.arange(n_times)
    srate = 1000.0
    freqs = np.arange(n_freqs)
    tapers = np.arange(n_tapers)
    weights = np.ones((n_tapers, n_freqs))
    events = np.zeros((n_epos, 3), dtype=int)
    events[:, 0] = np.arange(n_epos)
    events[:, 2] = np.arange(5, 5 + n_epos)
    event_id = {k: v for v, k in zip(events[:, 2], ["ha", "he", "hu"])}
    info = mne.create_info(ch_names, srate, ch_types)
    tfr = EpochsTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        events=events,
        event_id=event_id,
        weights=weights,
    )
    # test index checking
    with pytest.raises(ValueError, match="options. Valid index options are"):
        tfr.to_data_frame(index=["foo", "bar"])
    with pytest.raises(ValueError, match='"qux" is not a valid option'):
        tfr.to_data_frame(index="qux")
    with pytest.raises(TypeError, match="index must be `None` or a string "):
        tfr.to_data_frame(index=np.arange(400))
    # test wide format
    df_wide = tfr.to_data_frame()
    assert all(np.isin(tfr.ch_names, df_wide.columns))
    assert all(
        np.isin(["time", "condition", "freq", "epoch", "taper"], df_wide.columns)
    )
    # test long format
    df_long = tfr.to_data_frame(long_format=True)
    expected = (
        "condition",
        "epoch",
        "freq",
        "time",
        "channel",
        "ch_type",
        "value",
        "taper",
    )
    assert set(expected) == set(df_long.columns)
    assert set(tfr.ch_names) == set(df_long["channel"])
    assert len(df_long) == tfr.data.size
    # test long format w/ index
    df_long = tfr.to_data_frame(long_format=True, index=["freq"])
    del df_wide, df_long
    # test whether data is in correct shape
    df = tfr.to_data_frame(index=["condition", "epoch", "taper", "freq", "time"])
    data = tfr.data
    assert_array_equal(df.values[:, 0], data[:, 0, :, :].reshape(1, -1).squeeze())
    # compare arbitrary observation:
    assert (
        df.loc[("he", slice(None), tapers[1], freqs[1], times[2]), ch_names[3]].iat[0]
        == data[1, 3, 1, 1, 2]
    )

    # Check also for AverageTFR:
    # (remove taper dimension before averaging)
    state = tfr.__getstate__()
    state["data"] = state["data"][:, :, 0]
    state["dims"] = ("epoch", "channel", "freq", "time")
    state["weights"] = None
    tfr = EpochsTFR(inst=state)
    tfr = tfr.average()
    with pytest.raises(ValueError, match="options. Valid index options are"):
        tfr.to_data_frame(index=["epoch", "condition"])
    with pytest.raises(ValueError, match='"epoch" is not a valid option'):
        tfr.to_data_frame(index="epoch")
    with pytest.raises(ValueError, match='"taper" is not a valid option'):
        tfr.to_data_frame(index="taper")
    with pytest.raises(TypeError, match="index must be `None` or a string "):
        tfr.to_data_frame(index=np.arange(400))
    # test wide format
    df_wide = tfr.to_data_frame()
    assert all(np.isin(tfr.ch_names, df_wide.columns))
    assert all(np.isin(["time", "freq"], df_wide.columns))
    # test long format
    df_long = tfr.to_data_frame(long_format=True)
    expected = ("freq", "time", "channel", "ch_type", "value")
    assert set(expected) == set(df_long.columns)
    assert set(tfr.ch_names) == set(df_long["channel"])
    assert len(df_long) == tfr.data.size
    # test long format w/ index
    df_long = tfr.to_data_frame(long_format=True, index=["freq"])
    del df_wide, df_long
    # test whether data is in correct shape
    df = tfr.to_data_frame(index=["freq", "time"])
    data = tfr.data
    assert_array_equal(df.values[:, 0], data[0, :, :].reshape(1, -1).squeeze())
    # compare arbitrary observation:
    assert df.loc[(freqs[1], times[2]), ch_names[3]] == data[3, 1, 2]


@pytest.mark.parametrize(
    "index",
    ("time", ["condition", "time", "freq"], ["freq", "time"], ["time", "freq"], None),
)
def test_to_data_frame_index(index):
    """Test index creation in epochs Pandas exporter."""
    # Create fake EpochsTFR data:
    pytest.importorskip("pandas")
    n_epos = 3
    ch_names = ["EEG 001", "EEG 002", "EEG 003", "EEG 004"]
    n_picks = len(ch_names)
    ch_types = ["eeg"] * n_picks
    n_tapers = 2
    n_freqs = 5
    n_times = 6
    data = np.random.rand(n_epos, n_picks, n_tapers, n_freqs, n_times)
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)
    weights = np.ones((n_tapers, n_freqs))
    events = np.zeros((n_epos, 3), dtype=int)
    events[:, 0] = np.arange(n_epos)
    events[:, 2] = np.arange(5, 8)
    event_id = {k: v for v, k in zip(events[:, 2], ["ha", "he", "hu"])}
    info = mne.create_info(ch_names, 1000.0, ch_types)
    tfr = EpochsTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        events=events,
        event_id=event_id,
        weights=weights,
    )
    df = tfr.to_data_frame(picks=[0, 2, 3], index=index)
    # test index order/hierarchy preservation
    if not isinstance(index, list):
        index = [index]
    assert list(df.index.names) == index
    # test that non-indexed data were present as columns
    non_index = list(set(["condition", "time", "freq", "taper", "epoch"]) - set(index))
    if len(non_index):
        assert all(np.isin(non_index, df.columns))


@pytest.mark.parametrize("time_format", (None, "ms", "timedelta"))
def test_to_data_frame_time_format(time_format):
    """Test time conversion in epochs Pandas exporter."""
    pd = pytest.importorskip("pandas")
    n_epos = 3
    ch_names = ["EEG 001", "EEG 002", "EEG 003", "EEG 004"]
    n_picks = len(ch_names)
    ch_types = ["eeg"] * n_picks
    n_freqs = 5
    n_times = 6
    data = np.random.rand(n_epos, n_picks, n_freqs, n_times)
    times = np.arange(6, dtype=float)
    freqs = np.arange(5)
    events = np.zeros((n_epos, 3), dtype=int)
    events[:, 0] = np.arange(n_epos)
    events[:, 2] = np.arange(5, 8)
    event_id = {k: v for v, k in zip(events[:, 2], ["ha", "he", "hu"])}
    info = mne.create_info(ch_names, 1000.0, ch_types)
    tfr = EpochsTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        events=events,
        event_id=event_id,
    )
    # test time_format
    df = tfr.to_data_frame(time_format=time_format)
    dtypes = {None: np.float64, "ms": np.int64, "timedelta": pd.Timedelta}
    assert isinstance(df["time"].iloc[0], dtypes[time_format])


@parametrize_morlet_multitaper
@parametrize_power_phase_complex
@pytest.mark.parametrize("picks", ("mag", mag_names, [2, 5, 8]))  # all 3 equivalent
def test_raw_compute_tfr(raw, method, output, picks, tmp_path):
    """Test Raw.compute_tfr() and picks handling."""
    full_tfr = raw.compute_tfr(method, output=output, freqs=freqs_linspace)
    pick_tfr = raw.compute_tfr(method, output=output, freqs=freqs_linspace, picks=picks)
    assert isinstance(pick_tfr, RawTFR), type(pick_tfr)
    # ↓↓↓ can't use [2,5,8] because ch0 is IAS, so indices change between raw and TFR
    want = full_tfr.get_data(picks=mag_names)
    got = pick_tfr.get_data()
    assert_array_equal(want, got)
    # make sure save/load works for phase/complex data
    if output in ("phase", "complex"):
        pytest.importorskip("h5io")
        fname = tmp_path / "temp_tfr.hdf5"
        full_tfr.save(fname, overwrite=True)
        assert read_tfrs(fname) == full_tfr


@parametrize_morlet_multitaper
@parametrize_power_phase_complex
@pytest.mark.parametrize("freqs", (freqs_linspace, freqs_unsorted_list))
def test_evoked_compute_tfr(full_evoked, method, output, freqs):
    """Test Evoked.compute_tfr(), with a few different ways of specifying freqs."""
    tfr = full_evoked.compute_tfr(method, freqs, output=output)
    assert isinstance(tfr, AverageTFR), type(tfr)
    assert tfr.nave == full_evoked.nave
    assert tfr.comment == full_evoked.comment


@parametrize_morlet_multitaper
@pytest.mark.parametrize(
    "average,return_itc,dim,want_class",
    (
        pytest.param(True, False, None, None, id="average,no_itc"),
        pytest.param(True, True, None, None, id="average,itc"),
        pytest.param(False, False, "freqs", EpochsTFR, id="no_average,agg_freqs"),
        pytest.param(False, False, "epochs", AverageTFR, id="no_average,agg_epochs"),
        pytest.param(False, False, "times", EpochsSpectrum, id="no_average,agg_times"),
    ),
)
def test_epochs_compute_tfr_average_itc(
    epochs, method, average, return_itc, dim, want_class
):
    """Test Epochs.compute_tfr(), averaging (at call time and afterward), and ITC."""
    tfr = epochs.compute_tfr(
        method, freqs=freqs_linspace, average=average, return_itc=return_itc
    )
    if return_itc:
        tfr, itc = tfr
        assert isinstance(itc, AverageTFR), type(itc)
        # for single-epoch input, ITC should be (nearly) unity
        assert_array_almost_equal(itc.get_data(), 1.0, decimal=15)
    # if not averaging initially, make sure the post-facto .average() works too
    if average:
        assert isinstance(tfr, AverageTFR), type(tfr)
        assert tfr.nave == 1
        assert tfr.comment == "1"
    else:
        assert isinstance(tfr, EpochsTFR), type(tfr)
        avg = tfr.average(dim=dim)
        assert isinstance(avg, want_class), type(avg)
        if dim == "epochs":
            assert avg.nave == len(epochs)
            assert avg.comment.startswith(f"mean of {len(epochs)} EpochsTFR")


def test_epochs_vs_evoked_compute_tfr(epochs):
    """Compare result of averaging before or after the TFR computation.

    This is mostly a test of object structure / attribute preservation. In normal cases,
    the data should not match:
        - epochs.compute_tfr().average() is average of squared magnitudes
        - epochs.average().compute_tfr() is squared magnitude of average
    But the `epochs` fixture has only one epoch, so here data should be identical too.

    The three things that will always end up different are `._comment`, `._inst_type`,
    and `._data_type`, so we ignore those here.
    """
    avg_first = epochs.average().compute_tfr(method="morlet", freqs=freqs_linspace)
    avg_second = epochs.compute_tfr(method="morlet", freqs=freqs_linspace).average()
    for attr in ("_comment", "_inst_type", "_data_type"):
        assert getattr(avg_first, attr) != getattr(avg_second, attr)
        delattr(avg_first, attr)
        delattr(avg_second, attr)
    assert avg_first == avg_second


morlet_kw = dict(n_cycles=freqs_linspace / 4, use_fft=False, zero_mean=True)
mt_kw = morlet_kw | dict(zero_mean=False, time_bandwidth=6)
stockwell_kw = dict(n_fft=1024, width=2)


@pytest.mark.parametrize(
    "method,freqs,method_kw",
    (
        pytest.param("morlet", freqs_linspace, morlet_kw, id="morlet-nondefaults"),
        pytest.param("multitaper", freqs_linspace, mt_kw, id="multitaper-nondefaults"),
        pytest.param("stockwell", "auto", stockwell_kw, id="stockwell-nondefaults"),
    ),
)
def test_epochs_compute_tfr_method_kw(epochs, method, freqs, method_kw):
    """Test Epochs.compute_tfr(**method_kw)."""
    tfr = epochs.compute_tfr(method, freqs=freqs, average=True, **method_kw)
    assert isinstance(tfr, AverageTFR), type(tfr)


@pytest.mark.parametrize(
    "freqs",
    (pytest.param("auto", id="freqauto"), pytest.param([20, 41], id="fminfmax")),
)
@pytest.mark.parametrize("return_itc", (False, True))
def test_epochs_compute_tfr_stockwell(epochs, freqs, return_itc):
    """Test Epochs.compute_tfr(method="stockwell")."""
    tfr = epochs.compute_tfr("stockwell", freqs, return_itc=return_itc)
    if return_itc:
        tfr, itc = tfr
        assert isinstance(itc, AverageTFR)
        # for single-epoch input, ITC should be (nearly) unity
        assert_array_almost_equal(itc.get_data(), 1.0, decimal=15)
    assert isinstance(tfr, AverageTFR)
    assert tfr.comment == "1"


@pytest.mark.parametrize("output", ("complex", "phase"))
def test_epochs_compute_tfr_multitaper_complex_phase(epochs, output):
    """Test Epochs.compute_tfr(output="complex"/"phase")."""
    tfr = epochs.compute_tfr("multitaper", freqs_linspace, output=output)
    assert len(tfr.shape) == 5  # epoch x channel x taper x freq x time
    assert tfr.weights.shape == tfr.shape[2:4]  # check weights and coeffs shapes match


@pytest.mark.parametrize("copy", (False, True))
def test_epochstfr_iter_evoked(epochs_tfr, copy):
    """Test EpochsTFR.iter_evoked()."""
    avgs = list(epochs_tfr.iter_evoked(copy=copy))
    assert len(avgs) == len(epochs_tfr)
    assert all(avg.nave == 1 for avg in avgs)
    assert avgs[0].comment == str(epochs_tfr.events[0, -1])


@pytest.mark.parametrize("obj_type", ("raw", "epochs", "evoked"))
def test_tfrarray_tapered_spectra(obj_type):
    """Test {Raw,Epochs,Average}TFRArray instantiation with tapered spectra."""
    # Create example data with a taper dimension
    n_epochs, n_chans, n_tapers, n_freqs, n_times = (5, 3, 4, 2, 6)
    data_shape = (n_chans, n_tapers, n_freqs, n_times)
    if obj_type == "epochs":
        data_shape = (n_epochs,) + data_shape
    data = np.random.rand(*data_shape)
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)
    weights = np.random.rand(n_tapers, n_freqs)
    info = mne.create_info(n_chans, 1000.0, "eeg")
    # Prepare for TFRArray object instantiation
    defaults = dict(info=info, data=data, times=times, freqs=freqs)
    class_mapping = dict(raw=RawTFRArray, epochs=EpochsTFRArray, evoked=AverageTFRArray)
    TFRArray = class_mapping[obj_type]
    # Check TFRArray instantiation runs with good data
    TFRArray(**defaults, weights=weights)
    # Check taper dimension but no weights caught
    with pytest.raises(
        ValueError, match="Taper dimension in data, but no weights found."
    ):
        TFRArray(**defaults)
    # Check mismatching n_taper in weights caught
    with pytest.raises(
        ValueError, match=r"Taper axis .* doesn't match weights attribute"
    ):
        TFRArray(**defaults, weights=weights[:-1])
    # Check mismatching n_freq in weights caught
    with pytest.raises(
        ValueError, match=r"Frequency axis .* doesn't match weights attribute"
    ):
        TFRArray(**defaults, weights=weights[:, :-1])


def test_tfr_proj(epochs):
    """Test `compute_tfr(proj=True)`."""
    epochs.compute_tfr(method="morlet", freqs=freqs_linspace, proj=True)


def test_tfr_copy(average_tfr):
    """Test BaseTFR.copy() method."""
    tfr_copy = average_tfr.copy()
    # check that info is independent
    tfr_copy.info["bads"] = tfr_copy.ch_names
    assert average_tfr.info["bads"] == []
    # check that data is independent
    tfr_copy.data = np.inf
    assert np.isfinite(average_tfr.get_data()).all()


@pytest.mark.parametrize(
    "mode", ("mean", "ratio", "logratio", "percent", "zscore", "zlogratio")
)
def test_tfr_apply_baseline(average_tfr, mode):
    """Test TFR baselining."""
    average_tfr.apply_baseline((-0.1, -0.05), mode=mode)


def test_tfr_arithmetic(epochs):
    """Test TFR arithmetic operations."""
    tfr, itc = epochs.compute_tfr(
        "morlet", freqs=freqs_linspace, average=True, return_itc=True
    )
    itc_copy = itc.copy()
    # addition / subtraction of objects
    double = tfr + tfr
    double -= tfr
    assert tfr == double
    itc_copy += tfr
    assert itc == itc_copy - tfr
    # multiplication / division with scalars
    bigger_itc = itc * 23
    assert_array_almost_equal(itc.data, (bigger_itc / 23).data, decimal=15)
    # multiplication / division with arrays
    arr = np.full_like(itc.data, 23)
    assert_array_equal(bigger_itc.data, (itc * arr).data)
    # in-place multiplication/division
    bigger_itc *= 2
    bigger_itc /= 46
    assert_array_almost_equal(itc.data, bigger_itc.data, decimal=15)
    # check errors
    with pytest.raises(RuntimeError, match="types do not match"):
        tfr + epochs
    with pytest.raises(RuntimeError, match="times do not match"):
        tfr + tfr.copy().crop(tmax=0.2)
    with pytest.raises(RuntimeError, match="freqs do not match"):
        tfr + tfr.copy().crop(fmax=33)


def test_tfr_repr_html(epochs_tfr):
    """Test TFR._repr_html_()."""
    result = epochs_tfr._repr_html_(caption="Foo")
    for heading in ("Data type", "Data source", "Estimation method"):
        assert f"<th>{heading}</th>" in result
    for data in ("Power Estimates", "Epochs", "morlet"):
        assert f"<td>{data}</td>" in result


@pytest.mark.parametrize(
    "picks,combine",
    (
        pytest.param("mag", "mean", id="mean_of_mags"),
        pytest.param("grad", "rms", id="rms_of_grads"),
        pytest.param([1], "mean", id="single_channel"),
        pytest.param([1, 2], None, id="two_separate_channels"),
    ),
)
def test_tfr_plot_combine(epochs_tfr, picks, combine):
    """Test TFR.plot() picks, combine, and title="auto".

    No need to parametrize over {Raw,Epochs,Evoked}TFR, the code path is shared.
    """
    fig = epochs_tfr.plot(picks=picks, combine=combine, title="auto")
    assert len(fig) == 1 if isinstance(picks, str) else len(picks)
    # test `title="auto"`
    for ix, _fig in enumerate(fig):
        if isinstance(picks, str):
            ch_type = _channel_type_prettyprint[picks]
            want = rf"{'RMS' if combine == 'rms' else 'Mean'} of \d{{1,3}} {ch_type}s"
        else:
            want = epochs_tfr.ch_names[picks[ix]]
        assert re.search(want, _get_suptitle(_fig))


def test_tfr_plot_extras(epochs_tfr):
    """Test other options of TFR.plot()."""
    # test mask and custom title
    picks = [1]
    mask = np.ones(epochs_tfr.data.shape[2:], bool)
    fig = epochs_tfr.plot(picks=picks, mask=mask, title="Foo")
    assert _get_suptitle(fig[0]) == "Foo"
    mask = np.ones(epochs_tfr.data.shape[1:], bool)
    with pytest.raises(ValueError, match="mask must have the same shape as the data"):
        epochs_tfr.plot(picks=picks, mask=mask)
    # test combine-related errors
    with pytest.raises(ValueError, match='"combine" must be None, a callable, or one'):
        epochs_tfr.plot(picks=picks, combine="foo")
    with pytest.raises(RuntimeError, match="Wrong type yielded by callable"):
        epochs_tfr.plot(picks=picks, combine=lambda x: 777)
    with pytest.raises(RuntimeError, match="Wrong shape yielded by callable"):
        epochs_tfr.plot(picks=picks, combine=lambda x: np.array([777]))
    with pytest.raises(ValueError, match="wrong with the callable passed to 'combine'"):
        epochs_tfr.plot(picks=picks, combine=lambda x, y: x.mean(axis=0))
    # test custom Axes
    fig, axs = plt.subplots(1, 5)
    fig2 = epochs_tfr.plot(picks=[1, 2], combine=lambda x: x.mean(axis=0), axes=axs[0])
    fig3 = epochs_tfr.plot(picks=[1, 2, 3], axes=axs[1:-1])
    fig4 = epochs_tfr.plot(picks=[1], axes=axs[-1:].tolist())
    for _fig in fig2 + fig3 + fig4:
        assert fig == _fig
    with pytest.raises(ValueError, match="axes must be None"):
        epochs_tfr.plot(picks=picks, axes={})
    with pytest.raises(RuntimeError, match="must be one axes for each picked channel"):
        epochs_tfr.plot(picks=[1, 2], axes=axs[-1:])
    # test singleton check by faking having 2 epochs
    epochs_tfr._data = np.vstack((epochs_tfr._data, epochs_tfr._data))
    with pytest.raises(NotImplementedError, match=r"Cannot call plot\(\) from"):
        epochs_tfr.plot()


def test_tfr_plot_interactivity(epochs_tfr):
    """Test interactivity of TFR.plot()."""
    fig = epochs_tfr.plot(picks="mag", combine="mean")[0]
    assert len(plt.get_fignums()) == 1
    # press and release in same spot (should do nothing)
    kw = dict(fig=fig, ax=fig.axes[0], xform="ax")
    _fake_click(**kw, point=(0.5, 0.5), kind="press")
    _fake_click(**kw, point=(0.5, 0.5), kind="motion")
    _fake_click(**kw, point=(0.5, 0.5), kind="release")
    assert len(plt.get_fignums()) == 1
    # click and drag (should create popup topomap)
    _fake_click(**kw, point=(0.4, 0.4), kind="press")
    _fake_click(**kw, point=(0.5, 0.5), kind="motion")
    _fake_click(**kw, point=(0.6, 0.6), kind="release")
    assert len(plt.get_fignums()) == 2


@parametrize_inst_and_ch_type
def test_tfr_plot_topo(inst, ch_type, average_tfr, request):
    """Test {Raw,Epochs,Average}TFR.plot_topo()."""
    tfr = _get_inst(inst, request, average_tfr=average_tfr)
    fig = tfr.plot_topo(picks=ch_type)
    assert fig is not None


@parametrize_inst_and_ch_type
def test_tfr_plot_topomap(inst, ch_type, full_average_tfr, request):
    """Test {Raw,Epochs,Average}TFR.plot_topomap()."""
    tfr = _get_inst(inst, request, average_tfr=full_average_tfr)
    fig = tfr.plot_topomap(ch_type=ch_type)
    # fake a click-drag-release to select all sensors & generate a pop-up TFR image
    ax = fig.axes[0]
    pts = [
        coll.get_offsets()
        for coll in ax.collections
        if isinstance(coll, PathCollection)
    ][0]
    # sometimes sensors are outside axes; make sure our click starts inside axes
    lims = np.vstack((ax.get_xlim(), ax.get_ylim()))
    pad = np.diff(lims, axis=1).ravel() / 100
    start = np.clip(pts.min(axis=0) - pad, *(lims.min(axis=1) + pad))
    stop = np.clip(pts.max(axis=0) + pad, *(lims.max(axis=1) - pad))
    kw = dict(fig=fig, ax=ax, xform="data")
    _fake_click(**kw, kind="press", point=tuple(start))
    # ↓↓↓ possible bug? using (start+stop)/2 for the motion event causes the motion
    # ↓↓↓ event (not release event) coords to propagate → fails to select sensors
    _fake_click(**kw, kind="motion", point=tuple(stop))
    _fake_click(**kw, kind="release", point=tuple(stop))
    # make sure we actually got a pop-up figure, and it has a plausible title
    fignums = plt.get_fignums()
    assert len(fignums) == 2
    popup_fig = plt.figure(fignums[-1])
    assert re.match(
        rf"Average over \d{{1,3}} {ch_type} channels\.", popup_fig.axes[0].get_title()
    )


@pytest.mark.parametrize("output", ("complex", "phase"))
def test_tfr_topo_plotting_multitaper_complex_phase(output, evoked):
    """Test plot_joint/topo/topomap() for data with a taper dimension."""
    # Compute TFR with taper dimension
    tfr = evoked.compute_tfr(
        method="multitaper", freqs=freqs_linspace, n_cycles=4, output=output
    )
    # Check that plotting works
    tfr.plot_joint(topomap_args=dict(res=8, contours=0, sensors=False))  # for speed
    tfr.plot_topo()
    tfr.plot_topomap()


def test_combine_tfr_error_catch(average_tfr):
    """Test combine_tfr() catches errors."""
    # check unrecognised weights string caught
    with pytest.raises(ValueError, match='Weights must be .* "nave" or "equal"'):
        combine_tfr([average_tfr, average_tfr], weights="foo")
    # check bad weights size caught
    with pytest.raises(ValueError, match="Weights must be the same size as all_tfr"):
        combine_tfr([average_tfr, average_tfr], weights=[1, 1, 1])
    # check different channel names caught
    state = average_tfr.__getstate__()
    new_info = average_tfr.info.copy()
    average_tfr_bad = AverageTFR(
        inst=state | dict(info=new_info.rename_channels({new_info.ch_names[0]: "foo"}))
    )
    with pytest.raises(AssertionError, match=".* do not contain the same channels"):
        combine_tfr([average_tfr, average_tfr_bad])
    # check different times caught
    average_tfr_bad = AverageTFR(inst=state | dict(times=average_tfr.times + 1))
    with pytest.raises(
        AssertionError, match=".* do not contain the same time instants"
    ):
        combine_tfr([average_tfr, average_tfr_bad])
    # check taper dim caught
    n_tapers = 3  # anything >= 1 should do
    weights = np.ones((n_tapers, average_tfr.shape[1]))  # tapers x freqs
    state["data"] = np.repeat(np.expand_dims(average_tfr.data, 1), n_tapers, axis=1)
    state["weights"] = weights
    state["dims"] = ("channel", "taper", "freq", "time")
    average_tfr_taper = AverageTFR(inst=state)
    with pytest.raises(
        NotImplementedError,
        match="Aggregating multitaper tapers across TFR datasets is not supported.",
    ):
        combine_tfr([average_tfr_taper, average_tfr_taper])
