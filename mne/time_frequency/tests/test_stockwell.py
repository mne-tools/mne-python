# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#
# License : BSD 3-clause

import numpy as np
import os.path as op
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_equals

from mne import fiff, read_events, Epochs
from mne.time_frequency import stockwell, stockwell_power
from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
raw = fiff.Raw(raw_fname, add_eeg_ref=False)
event_name = op.join(base_dir, 'test-eve.fif')
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                        ecg=True, eog=True, include=['STI 014'],
                        exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

tempdir = _TempDir()


def test_stockwell_core():
    """Test stockwell transform"""

    # taken from
    # http://vcs.ynic.york.ac.uk/docs/naf/intro/concepts/timefreq.html
    sfreq = 1e3  # make things easy to understand
    t = np.arange(sfreq)   # make an array for time
    t /= sfreq        # scale it so it goes to 1, i.e. 1 sec of time
    pulse10Hz = np.cos(2 * np.pi * 10 * t)
    pulse10Hz[0:175] = 0        # Zero before our desired pulse
    pulse10Hz[275:] = 0         # and zero after our desired pulse
    stpulse10Hz = stockwell(pulse10Hz, sfreq=sfreq, n_fft=int(sfreq))

    assert_equals(stpulse10Hz.max(1).argmax(0), 10)  # max freq
    assert_true(175 < stpulse10Hz.max(0).argmax(0) < 275)  # max time


def test_stockwell_api():
    """test stockwell functions"""
    epochs = Epochs(raw, events,
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))

    data = epochs.get_data()[:2, :3, :100]

    n_fft = data.shape[-1]  # make interpretation simpler
    st_args = dict(n_fft=n_fft, sfreq=epochs.info['sfreq'], n_jobs=1)

    fun = stockwell
    st1 = fun(data, **st_args)

    # outer dimensions should be the same as input
    assert_equals(st1.shape[:-2], data.shape[:-1])

    # one dimension only
    st2 = fun(data[0, 0, :], **st_args)
    assert_equals(st1.shape[-2:], st2.shape)
    assert_equals(st1.shape[:2], data.shape[:2])

    # two dimensions dimensions
    st3 = fun(data[0, :, :], **st_args)
    assert_equals(st2.shape, st3.shape[1:])
    assert_equals(data.shape[1], st3.shape[0])

    # test joblib
    st_args.update({'n_jobs': 2})
    st4 = fun(data, **st_args)

    assert_array_equal(st1, st4)

    st_power_args = dict(n_fft=n_fft, sfreq=epochs.info['sfreq'], n_jobs=1)

    fun = stockwell_power
    st_power1 = fun(data, **st_power_args)
    # outer dimensions should be the same as input
    assert_equals(st_power1.shape[:-2], data.shape[:-1])
    # one dimension only
    st_power2 = fun(data[0, 0, :], **st_power_args)
    assert_equals(st_power1.shape[-2:], st_power2.shape)
    assert_equals(st_power1.shape[:2], data.shape[:2])
    # two dimensions dimensions
    st_power3 = fun(data[0, :, :], **st_power_args)
    assert_equals(st_power2.shape, st_power3.shape[1:])
    assert_equals(data.shape[1], st_power3.shape[0])
    # test joblib
    st_power_args.update({'n_jobs': 2})
    st_power4 = fun(data, **st_power_args)
    assert_array_equal(st_power1, st_power4)

    # taper == 0 just returns the power
    st_power_args.update({'n_jobs': 1, 'n_tapers': 0})
    st_power5 = fun(data, **st_power_args)

    assert_array_equal(st_power5, np.abs(st1) ** 2)
