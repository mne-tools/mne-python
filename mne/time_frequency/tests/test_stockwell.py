# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License : BSD-3-Clause

import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_equal, assert_array_less)

from scipy import fftpack

from mne import read_events, Epochs, make_fixed_length_events
from mne.io import read_raw_fif
from mne.time_frequency._stockwell import (tfr_stockwell, _st,
                                           _precompute_st_windows,
                                           _check_input_st,
                                           _st_power_itc)

from mne.time_frequency import AverageTFR, tfr_array_stockwell
from mne.utils import _record_warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
raw_ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')


def test_stockwell_ctf():
    """Test that Stockwell can be calculated on CTF data."""
    raw = read_raw_fif(raw_ctf_fname)
    raw.apply_gradient_compensation(3)
    events = make_fixed_length_events(raw, duration=0.5)
    evoked = Epochs(raw, events, tmin=-0.2, tmax=0.3, decim=10,
                    preload=True, verbose='error').average()
    tfr_stockwell(evoked, verbose='error')  # smoke test


def test_stockwell_check_input():
    """Test input checker for stockwell."""
    # check for data size equal and unequal to a power of 2

    for last_dim in (127, 128):
        data = np.zeros((2, 10, last_dim))
        with _record_warnings():  # n_fft sometimes
            x_in, n_fft, zero_pad = _check_input_st(data, None)

        assert_equal(x_in.shape, (2, 10, 128))
        assert_equal(n_fft, 128)
        assert_equal(zero_pad, 128 - last_dim)


def test_stockwell_st_no_zero_pad():
    """Test stockwell power itc."""
    data = np.zeros((20, 128))
    start_f = 1
    stop_f = 10
    sfreq = 30
    width = 2
    W = _precompute_st_windows(data.shape[-1], start_f, stop_f, sfreq, width)
    _st_power_itc(data, 10, True, 0, 1, W)


def test_stockwell_core():
    """Test stockwell transform."""
    # adapted from
    # http://vcs.ynic.york.ac.uk/docs/naf/intro/concepts/timefreq.html
    sfreq = 1000.0  # make things easy to understand
    dur = 0.5
    onset, offset = 0.175, 0.275
    n_samp = int(sfreq * dur)
    t = np.arange(n_samp) / sfreq   # make an array for time
    pulse_freq = 15.
    pulse = np.cos(2. * np.pi * pulse_freq * t)
    pulse[0:int(onset * sfreq)] = 0.        # Zero before our desired pulse
    pulse[int(offset * sfreq):] = 0.         # and zero after our desired pulse

    width = 0.5
    freqs = fftpack.fftfreq(len(pulse), 1. / sfreq)
    fmin, fmax = 1.0, 100.0
    start_f, stop_f = [np.abs(freqs - f).argmin() for f in (fmin, fmax)]
    W = _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width)

    st_pulse = _st(pulse, start_f, W)
    st_pulse = np.abs(st_pulse) ** 2
    assert_equal(st_pulse.shape[-1], len(pulse))
    st_max_freq = freqs[st_pulse.max(axis=1).argmax(axis=0)]  # max freq
    assert_allclose(st_max_freq, pulse_freq, atol=1.0)
    assert (onset < t[st_pulse.max(axis=0).argmax(axis=0)] < offset)

    # test inversion to FFT, by averaging local spectra, see eq. 5 in
    # Moukadem, A., Bouguila, Z., Ould Abdeslam, D. and Alain Dieterlen.
    # "Stockwell transform optimization applied on the detection of split in
    # heart sounds."

    width = 1.0
    start_f, stop_f = 0, len(pulse)
    W = _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width)
    y = _st(pulse, start_f, W)
    # invert stockwell
    y_inv = fftpack.ifft(np.sum(y, axis=1)).real
    assert_array_almost_equal(pulse, y_inv)


def test_stockwell_api():
    """Test stockwell functions."""
    raw = read_raw_fif(raw_fname)
    event_id, tmin, tmax = 1, -0.2, 0.5
    event_name = op.join(base_dir, 'test-eve.fif')
    events = read_events(event_name)
    epochs = Epochs(raw, events,  # XXX pick 2 has epochs of zeros.
                    event_id, tmin, tmax, picks=[0, 1, 3])
    for fmin, fmax in [(None, 50), (5, 50), (5, None)]:
        power, itc = tfr_stockwell(epochs, fmin=fmin, fmax=fmax,
                                   return_itc=True)
        if fmax is not None:
            assert (power.freqs.max() <= fmax)
        power_evoked = tfr_stockwell(epochs.average(), fmin=fmin,
                                     fmax=fmax, return_itc=False)
        # for multitaper these don't necessarily match, but they seem to
        # for stockwell... if this fails, this maybe could be changed
        # just to check the shape
        assert_array_almost_equal(power_evoked.data, power.data)
    assert (isinstance(power, AverageTFR))
    assert (isinstance(itc, AverageTFR))
    assert_equal(power.data.shape, itc.data.shape)
    assert (itc.data.min() >= 0.0)
    assert (itc.data.max() <= 1.0)
    assert (np.log(power.data.max()) * 20 <= 0.0)
    assert (np.log(power.data.max()) * 20 <= 0.0)
    with pytest.raises(TypeError, match='ndarray'):
        tfr_array_stockwell('foo', 1000.)
    data = np.random.RandomState(0).randn(1, 1024)
    with pytest.raises(ValueError, match='3D with shape'):
        tfr_array_stockwell(data, 1000.)
    data = data[np.newaxis]
    power, itc, freqs = tfr_array_stockwell(data, 1000., return_itc=True)
    assert_allclose(itc, np.ones_like(itc))
    assert power.shape == (1, len(freqs), data.shape[-1])
    assert_array_less(0, power)
