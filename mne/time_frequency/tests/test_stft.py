# Authors : Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Eric Larson <larson.eric.d@gmail.com>
#
# License : BSD 3-clause

import pytest
import os.path as op
import numpy as np
from scipy import linalg
from numpy.testing import (assert_almost_equal, assert_array_almost_equal)

from mne import read_events, Epochs
from mne.io import read_raw_fif
from mne.time_frequency import (stft, istft, stftfreq, tfr_stft,
                                tfr_array_stft, AverageTFR)
from mne.time_frequency._stft import stft_norm2


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
raw_ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')


def test_stft_api():
    """Test STFT functions."""
    raw = read_raw_fif(raw_fname)
    event_id, tmin, tmax = 1, -0.2, 0.5
    event_name = op.join(base_dir, 'test-eve.fif')
    events = read_events(event_name)
    epochs = Epochs(raw, events,  # XXX pick 2 has epochs of zeros.
                    event_id, tmin, tmax, picks=[0, 1, 3])
    sfreq = raw.info['sfreq']
    window_size = 200

    # error if window size is not a multiple of 4
    with pytest.raises(ValueError, match='The window length must '
                                         'be a multiple of 4.'):
        tfr_stft(epochs, window_size=333,
                 average=True)

    power = tfr_stft(epochs, window_size=window_size,
                            average=True)

    assert (isinstance(power, AverageTFR))
    assert (np.log(power.data.max()) * 20 <= 0.0)
    assert (np.log(power.data.min()) * 20 <= 0.0)

    with pytest.raises(TypeError, match='ndarray'):
        tfr_array_stft('foo', 1000., window_size=window_size)

    data = np.random.RandomState(0).randn(1, 1024)
    with pytest.raises(ValueError, match='3D with shape'):
        tfr_array_stft(data, 1000., window_size=window_size)
    data = data[np.newaxis]

    power, freqs, times = tfr_array_stft(data, sfreq=sfreq,
                                         window_size=window_size,
                                         return_times=True)
    assert power.shape == (1, 1, len(freqs), times.size)


@pytest.mark.filterwarnings('ignore:This function call for STFT is deprecated:DeprecationWarning')
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
