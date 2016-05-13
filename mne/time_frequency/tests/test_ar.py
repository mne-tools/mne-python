import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_equal

from mne import io, pick_types
from mne.time_frequency.ar import yule_walker, fit_iir_model_raw
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
    raw = io.read_raw_fif(raw_fname)
    # pick MEG gradiometers
    picks = pick_types(raw.info, meg='grad', exclude='bads')
    picks = picks[:2]
    tmin, tmax, order = 0, 10, 2
    coefs = fit_iir_model_raw(raw, order, picks, tmin, tmax)[1][1:]
    assert_equal(coefs.shape, (order,))
    assert_true(0.9 < -coefs[0] < 1.1)
