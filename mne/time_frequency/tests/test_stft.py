import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from mne.time_frequency import stft, istft, stftfreq
from mne.time_frequency._stft import stft_norm2


def test_stft():
    """Test stft and istft tight frame property."""
    sfreq = 1000.  # Hz
    f = 7.  # Hz
    for T in [127, 128]:  # try with even and odd numbers
        # Test with low frequency signal
        t = np.arange(T).astype(np.float64)
        x = np.sin(2 * np.pi * f * t / sfreq)
        x = np.array([x, x + 1.])
        wsize = 128
        tstep = 4
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, Tx=T)

        freqs = stftfreq(wsize, sfreq=1000)

        max_freq = freqs[np.argmax(np.sum(np.abs(X[0]) ** 2, axis=1))]

        assert X.shape[1] == len(freqs)
        assert np.all(freqs >= 0.)
        assert np.abs(max_freq - f) < 1.
        assert_array_almost_equal(x, xp, decimal=6)

        # norm conservation thanks to tight frame property
        assert_almost_equal(np.sqrt(stft_norm2(X)),
                            [linalg.norm(xx) for xx in x], decimal=6)

        # Test with random signal
        x = np.random.randn(2, T)
        wsize = 16
        tstep = 8
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, Tx=T)

        freqs = stftfreq(wsize, sfreq=1000)

        max_freq = freqs[np.argmax(np.sum(np.abs(X[0]) ** 2, axis=1))]

        assert X.shape[1] == len(freqs)
        assert np.all(freqs >= 0.)
        assert_array_almost_equal(x, xp, decimal=6)

        # norm conservation thanks to tight frame property
        assert_almost_equal(np.sqrt(stft_norm2(X)),
                            [linalg.norm(xx) for xx in x],
                            decimal=6)

        # Try with empty array
        x = np.zeros((0, T))
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, T)
        assert xp.shape == x.shape
