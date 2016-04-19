# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import mne
from itertools import combinations
from nose.tools import assert_true, assert_raises, assert_equal
from mne.connectivity.cfc import (phase_amplitude_coupling,
                                  phase_locked_amplitude,
                                  phase_binned_amplitude)


def _create_rand_data():
    """Quickly create some random data."""
    # Set params
    n_sig = 3
    n_ep = 4
    sfreq, n_t = 1000., 40
    ev = np.linspace(0, sfreq * n_t, n_ep).astype(int)
    ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).T
    t = np.linspace(0, n_t, sfreq * n_t)

    ixs_conn = np.array(list(combinations(range(n_sig), 2)))
    info = mne.create_info(n_sig, sfreq, 'eeg')

    # Test random signals
    ev = np.vstack([np.linspace(0, len(t), n_ep),
                    np.zeros(n_ep),
                    np.ones(n_ep)]).astype(int).T
    rng = np.random.RandomState(42)
    rand_data = rng.randn(n_ep, n_sig, len(t))
    rand_raw = mne.io.RawArray(np.hstack(rand_data), info)
    rand_epochs = mne.Epochs(rand_raw, ev, {'ev': 1}, -1, 8, preload=True)
    return rand_raw, rand_epochs, ev, ixs_conn


def test_phase_amplitude_coupling():
    """ Test phase amplitude coupling. """
    from scipy.signal import hilbert
    flo = [4, 8]
    fhi = [80, 150]
    rand_raw, rand_epochs, ev, ixs_conn = _create_rand_data()
    conn = phase_amplitude_coupling(rand_epochs, flo, fhi, ixs_conn)
    assert_true(conn.mean() < .2)  # More variable because not as many times

    rand_raw_test = rand_raw.crop(0, 15, copy=True)  # To speed things up
    conn = phase_amplitude_coupling(rand_raw_test, flo, fhi, ixs_conn)
    assert_true(conn.mean() < .2)

    # Test events handling
    conn = phase_amplitude_coupling(rand_raw, flo, fhi, ixs_conn, ev=ev[:, 0],
                                    tmin=0, tmax=2)
    assert_true(conn.mean() < .2)
    # events ndim > 1
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, tmin=0, tmax=2, ev=ev)
    # No tmin/tmax
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, ev=ev)

    # Test low frequency carrier / modulated oscillation
    n_t = 10
    sfreq = 1000.
    t = np.linspace(0, n_t, sfreq * n_t)
    lo = np.sin(t * 2 * np.pi * 6)
    hi = np.sin(t * 2 * np.pi * 100)

    # Clip one signal so it only exists on certain phases of the cycle
    hi[np.angle(hilbert(lo)) > -np.pi * .5] = 0

    # Create Raw array for testing
    data = np.vstack([lo, hi])
    info = mne.create_info(['lo', 'hi'], sfreq, 'eeg')
    data_raw = mne.io.RawArray(data, info)
    conn = phase_amplitude_coupling(data_raw, flo, fhi, [0, 1])
    assert_true(conn > .98)

    # Tests for Raw + events functionality
    conn = phase_amplitude_coupling(data_raw, flo, fhi, [0, 1], )

    # Check that arrays don't work and correct ixs/freqs must be given
    assert_raises(ValueError, phase_amplitude_coupling, data, flo, fhi, [0, 1])
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, fhi, [0])
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, [1], [0, 1])


def test_phase_amplitude_viz_funcs():
    """Test helper functions for visualization"""
    freqs_ph = np.linspace(8, 12, 2)
    freqs_amp = np.linspace(40, 60, 5)
    rand_raw, rand_epochs, ev, ixs_conn = _create_rand_data()

    # Phase locked viz
    ix_ph, ix_amp = [ixs_conn[0][i] for i in [0, 1]]
    amp, phase, times = phase_locked_amplitude(
        rand_epochs, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    amp, phase, times = phase_locked_amplitude(
        rand_raw, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    use_times = rand_raw.times < 3
    amp, phase, times = phase_locked_amplitude(
        rand_raw, freqs_ph, freqs_amp, ix_ph, ix_amp, mask_times=use_times,
        tmin=-.5, tmax=.5)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    # Phase binning
    amp_binned, bins = phase_binned_amplitude(rand_epochs, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(rand_raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(rand_raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20,
                                              mask_times=use_times)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)
