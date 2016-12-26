# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne import io, pick_types
from mne.utils import requires_sklearn_0_15, run_tests_if_main
from mne.decoding import ReceptiveField, get_coefs, delay_time_series
from mne.decoding.base import _get_final_est


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


def test_time_delay():
    from scipy.sparse import csr_matrix

    # Explicit delays + sfreq
    X = np.random.randn(2, 1000)
    X_sp = np.zeros([2, 10])
    X_sp[0, 1] = 1
    X_sp = csr_matrix(X_sp)
    for idel, isfreq in [([0, 1, 2], 1), ([0, .1, .2], 10)]:
        # sfreq must be int/float
        assert_raises(ValueError, delay_time_series, X, idel, sfreq=[1])
        # Delays must be 1D
        assert_raises(ValueError, delay_time_series, X, [idel])
        # Delays must be int/float
        assert_raises(ValueError, delay_time_series, X,
                      np.array(idel, dtype=np.complex))
        # Make sure swapaxes works
        delayed = delay_time_series(X, idel, isfreq, newaxis=2)
        assert_equal(delayed.shape[2], len(idel))

        for idata in [X, X_sp]:
            X_delayed = delay_time_series(X, idel, isfreq)
            assert_array_equal(X_delayed[0], X)
            assert_array_equal(X_delayed[1][0, :-1], X[0, 1:])
            assert_equal(len(X_delayed), len(idel))


@requires_sklearn_0_15
def test_receptive_field():
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()
    pipe = make_pipeline(Ridge())
    est = _get_final_est(pipe)
    assert_true(isinstance(est, type(mod)))

    # Est must be fit first
    assert_raises(ValueError, get_coefs, est)
    # Coefs are correctly taken
    est.fit([[1, 2], [3, 4]], [1, 2])
    coefs = get_coefs(pipe, 'coef_')
    assert_equal(coefs.shape[-1], 2)
    # Incorrect coefficient name
    assert_raises(ValueError, get_coefs, est, 'foo')

    # Test the receptive field model
    # Define parameters for the model and simulate inputs + weights
    delays = np.array([0, 2, 4])
    w = np.random.randn(9)
    X = np.random.randn(3, 10000)
    # Delay inputs and cut off first 4 values since they'll be cut in the fit
    X_del = np.vstack(delay_time_series(X, delays))
    y = np.dot(w, X_del)
    # Fit the model and test values
    rf = ReceptiveField(delays, ['one', 'two', 'three'], model=mod)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    assert_array_almost_equal(y[:-4], y_pred.squeeze(), 3)
    assert_array_almost_equal(np.hstack(rf.coef_), w, 3)
    # stim features must match length of input data
    assert_raises(ValueError, rf.fit, X[:1], y)
    # auto-naming features
    rf = ReceptiveField(delays, model=mod)
    rf.fit(X, y)
    assert_equal(rf.feature_names, ['feature_%s' % ii for ii in [0, 1, 2]])

run_tests_if_main()
