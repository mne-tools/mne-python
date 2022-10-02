import numpy as np

from mne.decoding import TimeDelayingRidge
from mne.utils import requires_sklearn
import pytest


@requires_sklearn
def test_time_delaying_ridge_params():
    rng = np.random.RandomState(3)
    x = rng.randn(1000, 3)
    y = np.zeros((1000, 2))
    smin, smax = 0, 5
    try:
        from mne.utils import _check_sklearn_estimator
        _check_sklearn_estimator(TimeDelayingRidge(smin, smax, 1),
                                 x.shape, y.shape)
    except ImportError:
        pytest.xfail('Cannot find sklearn utils needed for checking parameters')