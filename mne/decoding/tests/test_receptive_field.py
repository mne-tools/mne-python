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
from mne.decoding import ReceptiveField
from mne.decoding.receptive_field import (_delay_time_series, _SCORERS,
                                          _times_to_delays, _delays_to_slice)


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

rng = np.random.RandomState(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data
raw = io.read_raw_fif(raw_fname, preload=True)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[:2]


def test_time_delay():
    """Test that time-delaying w/ times and samples works properly."""
    from scipy.sparse import csr_matrix

    # Explicit delays + sfreq
    X = rng.randn(2, 1000)
    X_sp = np.zeros([2, 10])
    X_sp[0, 1] = 1
    X_sp = csr_matrix(X_sp)
    test_tlims = [((0, 2), 1), ((0, .2), 10), ((-.1, .1), 10)]
    for (tmin, tmax), isfreq in test_tlims:
        # sfreq must be int/float
        assert_raises(ValueError, _delay_time_series, X, tmin, tmax,
                      sfreq=[1])
        # Delays must be int/float
        assert_raises(ValueError, _delay_time_series, X,
                      np.complex(tmin), tmax, 1)
        # Make sure swapaxes works
        delayed = _delay_time_series(X, tmin, tmax, isfreq,
                                     newaxis=2, axis=-1)
        assert_equal(delayed.shape[2], 3)
        # Make sure delay slice is correct
        delays = _times_to_delays(tmin, tmax, isfreq)
        keep = _delays_to_slice(delays)
        assert_true(delayed[..., keep].shape[-1] > 0)
        assert_true(np.isnan(delayed[..., keep]).sum() == 0)

        for idata in [X, X_sp]:
            if tmin < 0:
                continue
            X_delayed = _delay_time_series(X, tmin, tmax, isfreq, axis=-1)
            assert_array_equal(X_delayed[0], X)
            assert_array_equal(X_delayed[1][0, :-1], X[0, 1:])
            assert_equal(len(X_delayed), (tmax - tmin) * isfreq + 1)


@requires_sklearn_0_15
def test_receptive_field():
    """Test model prep and fitting"""
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()

    # Test the receptive field model
    # Define parameters for the model and simulate inputs + weights
    tmin, tmax = 0., 10.
    n_feats = 3
    X = rng.randn(n_feats, 10000)
    w = rng.randn(int((tmax - tmin) + 1) * n_feats)

    # Delay inputs and cut off first 4 values since they'll be cut in the fit
    X_del = np.vstack(_delay_time_series(X, tmin, tmax, 1., axis=-1))
    y = np.dot(w, X_del)
    X = np.rollaxis(X, -1, 0)  # time to first dimension

    # Fit the model and test values
    feature_names = ['feature_%i' % ii for ii in [0, 1, 2]]
    rf = ReceptiveField(tmin, tmax, 1, feature_names, estimator=mod)
    rf.fit(X, y)
    assert_array_equal(rf.delays_, np.arange(tmin, tmax + 1))

    y_pred = rf.predict(X)
    assert_array_almost_equal(y[rf.keep_samples_],
                              y_pred.squeeze()[rf.keep_samples_], 2)
    scores = rf.score(X, y)
    assert_true(scores > .99)
    assert_array_almost_equal(rf.coef_.reshape(-1, order='F'), w, 2)
    # Make sure different input shapes work
    rf.fit(X[:, np.newaxis:, ], y[:, np.newaxis])
    rf.fit(X, y[:, np.newaxis])
    assert_raises(ValueError, rf.fit, X[..., np.newaxis], y)
    assert_raises(ValueError, rf.fit, X[:, 0], y)
    assert_raises(ValueError, rf.fit, X[..., np.newaxis],
                  np.tile(y[..., np.newaxis], [2, 1, 1]))
    # stim features must match length of input data
    assert_raises(ValueError, rf.fit, X[:, :1], y)
    # auto-naming features
    rf = ReceptiveField(tmin, tmax, 1, estimator=mod)
    rf.fit(X, y)
    assert_equal(rf.feature_names, ['feature_%s' % ii for ii in [0, 1, 2]])
    # X/y same n timepoints
    assert_raises(ValueError, rf.fit, X, y[:-2])
    # Float becomes ridge
    rf = ReceptiveField(tmin, tmax, 1, ['one', 'two', 'three'],
                        estimator=0)
    str(rf)  # repr works before fit
    rf.fit(X, y)
    assert_true(isinstance(rf.estimator_, Ridge))
    str(rf)  # repr works after fit
    rf = ReceptiveField(tmin, tmax, 1, ['one'], estimator=0)
    rf.fit(X[:, [0]], y)
    str(rf)  # repr with one feature
    # Should only accept estimators or floats
    rf = ReceptiveField(tmin, tmax, 1, estimator='foo')
    assert_raises(ValueError, rf.fit, X, y)
    rf = ReceptiveField(tmin, tmax, 1, estimator=np.array([1, 2, 3]))
    assert_raises(ValueError, rf.fit, X, y)
    # tmin must be <= tmax
    rf = ReceptiveField(5, 4, 1)
    assert_raises(ValueError, rf.fit, X, y)
    # scorers
    for key, val in _SCORERS.items():
        rf = ReceptiveField(tmin, tmax, 1, ['one'],
                            estimator=0, scoring=key)
        rf.fit(X[:, [0]], y)
        y_pred = rf.predict(X[:, [0]])
        assert_array_almost_equal(val(y[:, np.newaxis], y_pred),
                                  rf.score(X[:, [0]], y), 4)
    # Need 2D input
    assert_raises(ValueError, _SCORERS['corrcoef'], y.squeeze(), y_pred)
    # Need correct scorers
    assert_raises(ValueError, ReceptiveField, tmin, tmax, 1, scoring='foo')


run_tests_if_main()
