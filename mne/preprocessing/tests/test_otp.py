# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op

import pytest

import numpy as np
from numpy.fft import rfft, rfftfreq

from mne import create_info
from mne.datasets import testing
from mne.io import RawArray, read_raw_fif
from mne.io.pick import _pick_data_channels
from mne.preprocessing import oversampled_temporal_projection
from mne.utils import catch_logging

data_path = testing.data_path(download=False)
erm_fname = op.join(data_path, 'SSS', 'test_move_anon_erm_raw.fif')
triux_fname = op.join(data_path, 'SSS', 'TRIUX', 'triux_bmlhus_erm_raw.fif')
skip_fname = op.join(data_path, 'misc', 'intervalrecording_raw.fif')


def test_otp_array():
    """Test the OTP algorithm on artificial data."""
    n_channels, n_time, sfreq = 10, 2000, 1000.
    signal_f = 2.
    rng = np.random.RandomState(0)
    data = rng.randn(n_channels, n_time)
    raw = RawArray(data, create_info(n_channels, sfreq, 'eeg'))
    raw.info['bads'] = [raw.ch_names[0]]
    signal = np.sin(2 * np.pi * signal_f * raw.times)
    raw._data += signal

    # Check SNR
    def snr(data):
        """Check SNR according to the simulation model."""
        data_fft = rfft(data)
        freqs = rfftfreq(n_time, 1. / 1000.)
        sidx = np.where(freqs == signal_f)[0][0]
        oidx = list(range(sidx)) + list(range(sidx + 1, len(freqs)))
        snr = ((data_fft[:, sidx] * data_fft[:, sidx].conj()).real.sum() /
               (data_fft[:, oidx] * data_fft[:, oidx].conj()).real.sum())
        return snr

    orig_snr = snr(raw[:][0])
    with catch_logging() as log:
        raw_otp = oversampled_temporal_projection(
            raw, duration=2., verbose=True)
        otp_2_snr = snr(raw_otp[:][0])
        assert otp_2_snr > 3 + orig_snr
        assert '1 data chunk' in log.getvalue()
    with catch_logging() as log:
        raw_otp = oversampled_temporal_projection(
            raw, duration=1.2, verbose=True)
        otp_1p5_snr = snr(raw_otp[:][0])
        assert otp_1p5_snr > 3 + orig_snr
        assert '2 data chunks' in log.getvalue()
    with catch_logging() as log:
        raw_otp = oversampled_temporal_projection(
            raw, duration=1., verbose=True)
        otp_1_snr = snr(raw_otp[:][0])
        assert otp_1_snr > 2 + orig_snr
        assert '3 data chunks' in log.getvalue()

    # Pure-noise test
    raw._data -= signal
    raw_otp = oversampled_temporal_projection(raw, 2.)
    reduction = (np.linalg.norm(raw[:][0], axis=-1) /
                 np.linalg.norm(raw_otp[:][0], axis=-1))
    assert reduction.min() > 9.

    # Degenerate conditions
    raw = RawArray(np.zeros((200, 1000)), create_info(200, sfreq, 'eeg'))
    with pytest.raises(ValueError):  # too short
        oversampled_temporal_projection(raw, duration=198. / sfreq)
    with pytest.raises(ValueError):  # duration invalid
        oversampled_temporal_projection(
            raw, duration=raw.times[-1] + 2. / raw.info['sfreq'], verbose=True)
    raw._data[0, 0] = np.inf
    with pytest.raises(RuntimeError):  # too short
        oversampled_temporal_projection(raw, duration=1.)


@testing.requires_testing_data
def test_otp_real():
    """Test OTP on real data."""
    for fname in (erm_fname, triux_fname):
        raw = read_raw_fif(fname, allow_maxshield='yes').crop(0, 1)
        raw.load_data().pick_channels(raw.ch_names[:10])
        raw_otp = oversampled_temporal_projection(raw, 1.)
        picks = _pick_data_channels(raw.info)
        reduction = (np.linalg.norm(raw[picks][0], axis=-1) /
                     np.linalg.norm(raw_otp[picks][0], axis=-1))
        assert reduction.min() > 1

    # Handling of acquisition skips
    raw = read_raw_fif(skip_fname, preload=True)
    raw.pick_channels(raw.ch_names[:10])
    raw_otp = oversampled_temporal_projection(raw, duration=1.)
