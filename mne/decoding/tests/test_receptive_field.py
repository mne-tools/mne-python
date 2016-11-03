# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne import io, read_events, pick_types
from mne.utils import (requires_sklearn, run_tests_if_main)
from mne.encoding import (SubsetEstimator, EventsBinarizer,
                          FeatureDelayer, get_coefs)
from mne.encoding.model import _check_regressor, _get_final_est


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

np.random.seed(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data
raw = io.read_raw_fif(raw_fname, preload=True)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[0:2]


@requires_sklearn
def test_feature():
    from sklearn.cross_validation import cross_val_score, cross_val_predict
    from sklearn.linear_model import Ridge
    events = read_events(event_name)
    events[:, 0] -= raw.first_samp

    # --- Feature Delayer ---
    # Explicit delays + sfreq
    X = np.random.randn(1000, 2)
    X_sp = np.zeros([10, 2])
    X_sp[1, 0] = 1
    X_sp = csr_matrix(X_sp)
    for idel, isfreq in [[[0, 1, 2], 1], [[0, .1, .2], 10]]:
        delayer = FeatureDelayer(delays=idel, sfreq=isfreq)
        # sfreq must be int/float
        assert_raises(ValueError, FeatureDelayer, sfreq=[1])
        # Delays must be 1D
        assert_raises(ValueError, FeatureDelayer, delays=[idel])
        # Delays must be int/float
        assert_raises(ValueError, FeatureDelayer,
                      delays=np.array(idel, dtype=np.complex))

        for idata in [X, X_sp]:
            X_delayed = delayer.transform(X)
            assert_array_equal(X_delayed[:, 0, 0], X[:, 0])
            assert_array_equal(X_delayed[:-1, 0, 1], X[1:, 0])
            assert_equal(X_delayed.shape[-1], len(idel))


@requires_sklearn
def test_encoding():
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()
    pipe = make_pipeline(Ridge())
    samp = SubsetEstimator(pipe)
    est = _get_final_est(pipe)
    assert_true(isinstance(est, type(mod)))
    est = _get_final_est(samp)
    assert_true(isinstance(est, type(mod)))
    # Est must be fit first
    assert_raises(ValueError, get_coefs, est)
    # Coefs are correctly taken
    est.fit([[1, 2], [3, 4]], [1, 2])
    coefs = get_coefs(samp, 'coef_')
    assert_equal(coefs.shape[-1], 2)
    # Incorrect coefficient name
    assert_raises(ValueError, get_coefs, est, 'foo')

    # Make sure the checks are working
    mod = _check_regressor(Ridge())
    assert_true(isinstance(mod, Ridge))
    # None returns Ridge instance
    assert_true(isinstance(_check_regressor(None), Ridge))
    assert_true(isinstance(_get_final_est(_check_regressor(None)), Ridge))
    # Correct ridge solver
    assert_equal(_get_final_est(_check_regressor('lsqr')).solver, 'lsqr')
    assert_true(isinstance(_get_final_est(_check_regressor('lsqr')), Ridge))
    # Incorrect string type
    assert_raises(ValueError, _check_regressor, 'foo')
    # Estimator must have fit/predict methods
    assert_raises(ValueError, _check_regressor, lambda a: a + 1)

run_tests_if_main()
