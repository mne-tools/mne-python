import numpy as np

from mne.decoding import TimeDelayingRidge
from mne.utils import _check_sklearn_estimator


def test_time_delaying_ridge():
    rng = np.random.RandomState(3)
    x = rng.randn(1000, 3)
    y = np.zeros((1000, 2))
    smin, smax = 0, 5

    td = TimeDelayingRidge(smin, smax, 1)
    _check_sklearn_estimator(td, x.shape, y.shape)