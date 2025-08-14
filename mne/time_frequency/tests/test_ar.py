# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.signal import lfilter

from mne import io
from mne.time_frequency.ar import _yule_walker, fit_iir_model_raw

raw_fname = Path(__file__).parents[2] / "io" / "tests" / "data" / "test_raw.fif"


def test_yule_walker():
    """Test Yule-Walker against statsmodels."""
    pytest.importorskip("patsy", "0.4")
    pytest.importorskip("statsmodels", "0.8")
    from statsmodels.regression.linear_model import yule_walker as sm_yw

    d = np.random.randn(100)
    sm_rho, sm_sigma = sm_yw(d, order=2)
    rho, sigma = _yule_walker(d[np.newaxis], order=2)
    assert_array_almost_equal(sm_sigma, sigma)
    assert_array_almost_equal(sm_rho, rho)


def test_ar_raw():
    """Test fitting AR model on raw data."""
    raw = io.read_raw_fif(raw_fname).crop(0, 2).load_data()
    raw.pick(picks="grad")
    # pick MEG gradiometers
    for order in (2, 5, 10):
        coeffs = fit_iir_model_raw(raw, order)[1][1:]
        assert coeffs.shape == (order,)
        assert_allclose(-coeffs[0], 1.0, atol=0.5)
    # let's make sure we're doing something reasonable: first, white noise
    rng = np.random.RandomState(0)
    raw._data = rng.randn(*raw._data.shape)
    raw._data *= 1e-15
    for order in (2, 5, 10):
        coeffs = fit_iir_model_raw(raw, order)[1]
        assert_allclose(coeffs, [1.0] + [0.0] * order, atol=2e-2)
    # Now let's try pink noise
    iir = [1, -1, 0.2]
    raw._data = lfilter([1.0], iir, raw._data)
    for order in (2, 5, 10):
        coeffs = fit_iir_model_raw(raw, order)[1]
        assert_allclose(coeffs, iir + [0.0] * (order - 2), atol=5e-2)
