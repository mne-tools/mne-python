# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD-3-Clause


import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne.utils import requires_sklearn
from mne.decoding.time_frequency import TimeFrequency


@requires_sklearn
def test_timefrequency_params():
    freqs = [20, 21, 22]
    try:
        from mne.utils import _check_sklearn_estimator
        _check_sklearn_estimator(TimeFrequency(freqs, sfreq=100))
    except ImportError:
        pytest.xfail('Cannot find sklearn needed for checking parameters')


@requires_sklearn
def test_timefrequency():
    """Test TimeFrequency."""
    # Init
    n_freqs = 3
    freqs = [20, 21, 22]
    tf = TimeFrequency(freqs, sfreq=100)
    for output in ['avg_power', 'foo', None]:
        pytest.raises(ValueError, TimeFrequency(freqs, output=output).fit,
                      np.random.rand(10, 2, 100))

    # Clone estimator
    freqs_array = np.array(np.asarray(freqs))
    tf = TimeFrequency(freqs_array, 100, "morlet", freqs_array / 5.)

    # Fit
    n_epochs, n_chans, n_times = 10, 2, 100
    X = np.random.rand(n_epochs, n_chans, n_times)
    tf.fit(X, None)

    # Transform
    tf = TimeFrequency(freqs, sfreq=100)
    tf.fit_transform(X, None)
    # 3-D X
    Xt = tf.transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times])
    # 2-D X
    Xt = tf.transform(X[:, 0, :])
    assert_array_equal(Xt.shape, [n_epochs, n_freqs, n_times])
    # 3-D with decim
    tf = TimeFrequency(freqs, sfreq=100, decim=2)
    Xt = tf.transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times // 2])
