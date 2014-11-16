# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License : BSD 3-clause

import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_equals

from scipy import fftpack

from mne import io, read_events, Epochs, pick_types
from mne.time_frequency._stockwell import (tfr_stockwell, _st,
                                           _check_input_st)
from mne.time_frequency.tfr import AverageTFR
from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
raw = io.Raw(raw_fname, add_eeg_ref=False)
event_name = op.join(base_dir, 'test-eve.fif')
events = read_events(event_name)
picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                   ecg=True, eog=True, include=['STI 014'],
                   exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

tempdir = _TempDir()


def test_stockwell_core():
    """Test stockwell transform"""
    # taken from

    def stockwell(data, sfreq, fmin=0, fmax=np.inf, n_fft=None):
        n_fft, x_in, x_outer_shape = _check_input_st(data, n_fft)

        freqs = fftpack.fftfreq(n_fft, 1. / sfreq)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        st = _st(x_in, n_fft, freqs)[freq_mask]
        return st

    # http://vcs.ynic.york.ac.uk/docs/naf/intro/concepts/timefreq.html
    sfreq = 1e3  # make things easy to understand
    t = np.arange(sfreq)   # make an array for time
    t /= sfreq        # scale it so it goes to 1, i.e. 1 sec of time
    pulse_freq = 10.
    pulse = np.cos(2. * np.pi * pulse_freq * t)
    pulse[0:175] = 0.        # Zero before our desired pulse
    pulse[275:] = 0.         # and zero after our desired pulse

    # test with ndim
    st_pulse = stockwell(pulse, sfreq=sfreq)
    st_pulse_2d = stockwell(pulse[None, :], sfreq=sfreq)
    st_pulse_3d = stockwell(pulse[None, None, :], sfreq=sfreq)

    assert_array_almost_equal(st_pulse, st_pulse_2d[0])
    assert_array_almost_equal(st_pulse, st_pulse_3d[0, 0])

    for n_fft in [None, len(pulse)]:
        st_pulse = stockwell(pulse, sfreq=sfreq, n_fft=n_fft)
        st_pulse = stockwell(pulse, sfreq=sfreq)  # with next power of 2

        assert_equals(st_pulse.shape[-1], len(pulse))  # max freq
        assert_equals(st_pulse.max(axis=1).argmax(axis=0), pulse_freq)
        assert_true(175 < st_pulse.max(0).argmax(0) < 275)  # max time


def test_stockwell_api():
    """test stockwell functions"""
    epochs = Epochs(raw, events,
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))
    power, itc = tfr_stockwell(epochs, return_itc=True)
    assert_true(isinstance(power, AverageTFR))
    assert_true(isinstance(itc, AverageTFR))
    assert_equals(power.data.shape, itc.data.shape)
    assert_true(itc.data.min() >= 0.0)
    assert_true(itc.data.max() <= 1.0)
    assert_true(itc.data.max() < 0.0)
