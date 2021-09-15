# Authors : Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Eric Larson <larson.eric.d@gmail.com>
#
# License : BSD-3-Clause

import pytest
import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from mne.time_frequency import stft, istft, stftfreq
from mne.time_frequency._stft import stft_norm2


@pytest.mark.parametrize('T', (127, 128, 255, 256, 1337))
@pytest.mark.parametrize('wsize', (128, 256))
@pytest.mark.parametrize('tstep', (4, 64))
@pytest.mark.parametrize('f', (7., 23.))  # should be close to fftfreqs
def test_stft(T, wsize, tstep, f):
    """Test stft and istft tight frame property."""
    sfreq = 1000.  # Hz
    if True:  # just to minimize diff
        # Test with low frequency signal
        t = np.arange(T).astype(np.float64)
        x = np.sin(2 * np.pi * f * t / sfreq)
        x = np.array([x, x + 1.])
        X = stft(x, wsize, tstep)
        xp = istft(X, tstep, Tx=T)

        freqs = stftfreq(wsize, sfreq=sfreq)

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
