import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne import io, pick_types
from mne.time_frequency import yule_walker, ar_raw
from mne.utils import requires_statsmodels, requires_patsy


raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data',
                    'test_raw.fif')

@requires_patsy
@requires_statsmodels
def test_yule_walker():
    """Test Yule-Walker against statsmodels
    """
    from statsmodels.regression.linear_model import yule_walker as sm_yw
    d = np.random.randn(100)
    sm_rho, sm_sigma = sm_yw(d, order=2)
    rho, sigma = yule_walker(d, order=2)
    assert_array_almost_equal(sm_sigma, sigma)
    assert_array_almost_equal(sm_rho, rho)


def test_ar_raw():
    """Test fitting AR model on raw data
    """
    raw = io.Raw(raw_fname)

    # picks MEG gradiometers
    picks = pick_types(raw.info, meg='grad', exclude='bads')

    picks = picks[:2]

    tmin, tmax = 0, 10  # use the first s of data
    order = 2
    coefs = ar_raw(raw, picks=picks, order=order, tmin=tmin, tmax=tmax)
    mean_coefs = np.mean(coefs, axis=0)

    assert_true(coefs.shape == (len(picks), order))
    assert_true(0.9 < mean_coefs[0] < 1.1)
