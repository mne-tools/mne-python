# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne.decoding.time_frequency import TimeFrequency


def test_timefrequency_basic():
    """Test TimeFrequency."""
    # Init
    n_freqs = 3
    freqs = [20, 21, 22]
    tf = TimeFrequency(freqs, sfreq=100)
    n_epochs, n_chans, n_times = 10, 2, 100
    X = np.random.rand(n_epochs, n_chans, n_times)
    for output in ["avg_power", "foo", None]:
        tf = TimeFrequency(freqs, output=output)
        with pytest.raises(ValueError, match="Invalid value"):
            tf.fit(X)
    tf = clone(tf)

    # Clone estimator
    freqs_array = np.array(np.asarray(freqs))
    tf = TimeFrequency(freqs_array, 100, "morlet", freqs_array / 5.0)
    clone(tf)

    # Fit
    assert not hasattr(tf, "fitted_")
    tf.fit(X, None)
    assert tf.fitted_

    # Transform
    tf = TimeFrequency(freqs, sfreq=100)
    tf.fit_transform(X, None)
    # 3-D X
    Xt = tf.transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times])
    # 2-D X
    Xt = tf.fit_transform(X[:, 0, :])
    assert_array_equal(Xt.shape, [n_epochs, n_freqs, n_times])
    # 3-D with decim
    tf = TimeFrequency(freqs, sfreq=100, decim=2)
    Xt = tf.fit_transform(X)
    assert_array_equal(Xt.shape, [n_epochs, n_chans, n_freqs, n_times // 2])


@parametrize_with_checks([TimeFrequency([300, 400], 1000.0, n_cycles=0.25)])
def test_sklearn_compliance(estimator, check):
    """Test LinearModel compliance with sklearn."""
    check(estimator)
