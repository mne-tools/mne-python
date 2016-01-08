# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import mne
from scipy.signal import hilbert
from itertools import combinations
from nose.tools import assert_true, assert_raises
from mne.connectivity.cfc import phase_amplitude_coupling


def test_phase_amplitude_coupling():
    """ Test phase amplitude coupling. """
    # Set params
    n_sig = 3
    n_ep = 5
    sfreq, n_t = 1000., 40
    ev = np.linspace(0, sfreq * n_t, n_ep).astype(int)
    ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).T
    t = np.linspace(0, n_t, sfreq * n_t)
    flo = [4, 8]
    fhi = [80, 150]
    ixs_conn = np.array(list(combinations(range(n_sig), 2)))
    info = mne.create_info([str(i) for i in range(n_sig)],
                           sfreq, 'eeg')

    # Test random signals
    ev = np.vstack([np.linspace(0, len(t), n_ep),
                    np.zeros(n_ep),
                    np.ones(n_ep)]).astype(int).T
    rand_data = np.random.randn(n_ep, n_sig, len(t))
    rand_raw = mne.io.RawArray(np.hstack(rand_data), info)
    rand_epochs = mne.Epochs(rand_raw, ev, {'ev': 1}, -1, 8)

    for data in [rand_epochs, rand_raw]:
        conn = phase_amplitude_coupling(data, flo, fhi, ixs_conn)
        assert_true(conn.mean() < .3)

    # Test events handling
    conn = phase_amplitude_coupling(rand_raw, flo, fhi, ixs_conn, ev=ev[:, 0],
                                    tmin=0, tmax=2)
    assert_true(conn.mean() < .05)
    # events ndim > 1
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, tmin=0, tmax=2, ev=ev)
    # No tmin/tmax
    assert_raises(ValueError, phase_amplitude_coupling, rand_raw, flo, fhi,
                  ixs_conn, ev=ev)

    # Test low frequency carrier / modulated oscillation
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

    # Check that arrays don't work and correct ixs/freqs must be given
    assert_raises(ValueError, phase_amplitude_coupling, data, flo, fhi, [0, 1])
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, fhi, [0])
    assert_raises(
        ValueError, phase_amplitude_coupling, data_raw, flo, [1], [0, 1])
