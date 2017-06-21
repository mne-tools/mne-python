# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from mne.utils import requires_sklearn
from mne.decoding.time_frequency import TimeFrequency


@requires_sklearn
def test_timefrequency():
    from sklearn.base import clone
    # Init
    n_freqs = 3
    frequencies = np.linspace(20, 30, n_freqs)
    tf = TimeFrequency(frequencies, sfreq=100)
    for output in ['avg_power', 'foo', None]:
        assert_raises(ValueError, TimeFrequency, frequencies, output=output)
    tf = clone(tf)

    # Fit
    n_epochs, n_chans, n_times = 10, 2, 100
    X = np.random.rand(n_epochs, n_chans, n_times)
    tf.fit(X, None)

    # Transform
    tf = TimeFrequency(frequencies, sfreq=100)
    tf.fit_transform(X, None)
    # 3-D X
    Xt = tf.transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times])
    # 2-D X
    Xt = tf.transform(X[:, 0, :])
    assert_array_equal(Xt.shape, [n_epochs, n_freqs, n_times])
    # 3-D with decim
    tf = TimeFrequency(frequencies, sfreq=100, decim=2)
    Xt = tf.transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times // 2])
