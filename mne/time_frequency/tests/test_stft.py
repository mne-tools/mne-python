import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from nose.tools import assert_true

from ..stft import stft, istft, stftfreq


def test_stft():
    "Test stft and istft tight frame property"
    sfreq = 1000.  # Hz
    f = 7.  # Hz
    for T in [253, 256]:  # try with even and odd numbers
        t = np.arange(T).astype(np.float)
        x = np.sin(2 * np.pi * f * t / sfreq)
        x = np.array([x, x + 1.])
        wsize = 128
        tstep = 4
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, Tx=T)

        freqs = stftfreq(wsize, sfreq=1000)

        max_freq = freqs[np.argmax(np.sum(np.abs(X[0]) ** 2, axis=1))]

        assert_true(X.shape[1] == len(freqs))
        assert_true(np.all(freqs >= 0.))
        assert_true(np.abs(max_freq - f) < 1.)

        assert_array_almost_equal(x, xp, decimal=6)

        # Symmetrize X to get also negative frequencies to guarantee
        # norm conservation thanks to tight frame property
        X = np.concatenate([X[:, 1:, :][:, ::-1, :], X], axis=1)

        assert_almost_equal(linalg.norm(X.ravel()), linalg.norm(x.ravel()),
                            decimal=2)

        # Try with empty array
        x = np.zeros((0, T))
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, T)
        assert_true(xp.shape == x.shape)
