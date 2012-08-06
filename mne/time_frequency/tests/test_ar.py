import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true
import nose

from mne import fiff
from mne.time_frequency import yule_walker, ar_raw

raw_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data',
                'test_raw.fif')


def test_yule_walker():
    """Test Yule-Walker against statsmodels
    """
    try:
        from statsmodels.regression.linear_model import yule_walker as sm_yw
        d = np.random.randn(100)
        sm_rho, sm_sigma = sm_yw(d, order=2)
        rho, sigma = yule_walker(d, order=2)
        assert_array_almost_equal(sm_sigma, sigma)
        assert_array_almost_equal(sm_rho, rho)
    except ImportError:
        raise nose.SkipTest("XFailed Test")


def test_ar_raw():
    raw = fiff.Raw(raw_fname)

    # picks MEG gradiometers
    picks = fiff.pick_types(raw.info, meg='grad')

    picks = picks[:2]

    tmin, tmax = 0, 10  # use the first s of data
    order = 2
    coefs = ar_raw(raw, picks=picks, order=order, tmin=tmin, tmax=tmax)
    mean_coefs = np.mean(coefs, axis=0)

    assert_true(coefs.shape == (len(picks), order))
    assert_true(0.9 < mean_coefs[0] < 1.1)
