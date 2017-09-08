# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
import numpy as np
from scipy.sparse import csr_matrix

from numpy.testing import assert_array_equal, assert_allclose

from mne import io, pick_types
from mne.utils import requires_sklearn_0_15, run_tests_if_main
from mne.decoding import ReceptiveField, TimeDelayingRidge
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


@requires_sklearn_0_15
def test_rank_deficiency():
    """Test signals that are rank deficient."""
    # See GH#4253
    from sklearn.linear_model import Ridge
    N = 256
    fs = 1.
    tmin, tmax = -50, 100
    reg = 0.1
    rng = np.random.RandomState(0)
    eeg = rng.randn(N, 1)
    eeg *= 100
    eeg = np.fft.rfft(eeg, axis=0)
    eeg[N // 4:] = 0  # rank-deficient lowpass
    eeg = np.fft.irfft(eeg, axis=0)
    win = np.hanning(N // 8)
    win /= win.mean()
    y = np.apply_along_axis(np.convolve, 0, eeg, win, mode='same')
    y += rng.randn(*y.shape) * 100

    for est in (Ridge(reg), reg):
        rf = ReceptiveField(tmin, tmax, fs, estimator=est)
        rf.fit(eeg, y)
        pred = rf.predict(eeg)
        assert_equal(y.shape, pred.shape)
        corr = np.corrcoef(y.ravel(), pred.ravel())[0, 1]
        assert_true(corr > 0.995, msg=corr)


def test_time_delay():
    """Test that time-delaying w/ times and samples works properly."""
    # Explicit delays + sfreq
    X = rng.randn(1000, 2)
    test_tlims = [((0, 2), 1), ((0, .2), 10), ((-.1, .1), 10)]
    for (tmin, tmax), isfreq in test_tlims:
        # sfreq must be int/float
        assert_raises(ValueError, _delay_time_series, X, tmin, tmax,
                      sfreq=[1])
        # Delays must be int/float
        assert_raises(ValueError, _delay_time_series, X,
                      np.complex(tmin), tmax, 1)
        # Make sure swapaxes works
        delayed = _delay_time_series(X, tmin, tmax, isfreq)
        assert_equal(delayed.shape, (1000, 2, 3))
        # Make sure delay slice is correct
        delays = _times_to_delays(tmin, tmax, isfreq)
        keep = _delays_to_slice(delays)
        assert_true(delayed[..., keep].shape[-1] > 0)
        assert_true(np.isnan(delayed[..., keep]).sum() == 0)

        if tmin < 0:
            continue
        X_delayed = _delay_time_series(X, tmin, tmax, isfreq)
        assert_array_equal(X_delayed[:, :, 0], X)
        assert_array_equal(X_delayed[:-1, 0, 1], X[1:, 0])
        assert_equal(X_delayed.shape[-1], (tmax - tmin) * isfreq + 1)


@requires_sklearn_0_15
def test_receptive_field():
    """Test model prep and fitting."""
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()

    # Test the receptive field model
    # Define parameters for the model and simulate inputs + weights
    tmin, tmax = 0., 10.
    n_feats = 3
    X = rng.randn(10000, n_feats)
    w = rng.randn(int((tmax - tmin) + 1) * n_feats)

    # Delay inputs and cut off first 4 values since they'll be cut in the fit
    X_del = np.concatenate(
        _delay_time_series(X, tmin, tmax, 1.).transpose(2, 0, 1), axis=1)
    y = np.dot(X_del, w)

    # Fit the model and test values
    feature_names = ['feature_%i' % ii for ii in [0, 1, 2]]
    rf = ReceptiveField(tmin, tmax, 1, feature_names, estimator=mod)
    rf.fit(X, y)
    assert_array_equal(rf.delays_, np.arange(tmin, tmax + 1))

    y_pred = rf.predict(X)
    assert_allclose(y[rf.valid_samples_], y_pred[rf.valid_samples_], atol=1e-2)
    scores = rf.score(X, y)
    assert_true(scores > .99)
    assert_allclose(rf.coef_.T.ravel(), w, atol=1e-2)
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
    assert_true(isinstance(rf.estimator_, TimeDelayingRidge))
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
        y_pred = rf.predict(X[:, [0]]).T.ravel()[:, np.newaxis]
        assert_allclose(val(y[:, np.newaxis], y_pred),
                        rf.score(X[:, [0]], y), rtol=1e-3)
    # Need 2D input
    assert_raises(ValueError, _SCORERS['corrcoef'], y.ravel(), y_pred)
    # Need correct scorers
    rf = ReceptiveField(tmin, tmax, 1., scoring='foo')
    assert_raises(ValueError, rf.fit, X, y)


@requires_sklearn_0_15
def test_receptive_field_fast():
    """Test that the fast solving works like Ridge."""
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(0)
    x = rng.randn(500, 1)
    for delay in range(-2, 3):
        y = np.zeros(500)
        slims = [(-4, 2)]
        if delay == 0:
            y[:] = x[:, 0]
        elif delay < 0:
            y[-delay:] = x[:delay, 0]
            slims += [(-4, -1)]
        else:
            y[:-delay] = x[delay:, 0]
            slims += [(1, 2)]
        for ndim in (1, 2):
            y.shape = (y.shape[0],) + (1,) * (ndim - 1)
            for slim in slims:
                tdr = TimeDelayingRidge(slim[0], slim[1], 1., 0.1, 'laplacian',
                                        fit_intercept=False)
                for estimator in (Ridge(alpha=0.), 0., 0.1, tdr):
                    model = ReceptiveField(slim[0], slim[1], 1.,
                                           estimator=estimator)
                    model.fit(x, y)
                    assert_array_equal(model.delays_,
                                       np.arange(slim[0], slim[1] + 1))
                    expected = (model.delays_ == delay).astype(float)
                    expected = expected[np.newaxis]  # features
                    if y.ndim == 2:
                        expected = expected[np.newaxis]  # outputs
                    assert_equal(model.coef_.ndim, ndim + 1)
                    assert_allclose(model.coef_, expected, atol=1e-2)
                    p = model.predict(x)
                    assert_allclose(y, p, atol=5e-2)
                    score = model.score(x, y)
                    assert_true(score > 0.9999, msg=score)

    # multidimensional
    x = rng.randn(1000, 3)
    y = np.zeros((1000, 2))
    slim = [-5, 0]
    # This is a weird assignment, but it's just a way to distribute some
    # unique values at various delays, and "expected" explains how they
    # should appear in the resulting RF
    for ii in range(1, 5):
        y[ii:, ii % 2] += (-1) ** ii * ii * x[:-ii, ii % 3]
    expected = [
        [[0, 0, 0, 0, 0, 0],
         [0, 4, 0, 0, 0, 0],
         [0, 0, 0, 2, 0, 0]],
        [[0, 0, -3, 0, 0, 0],
         [0, 0, 0, 0, -1, 0],
         [0, 0, 0, 0, 0, 0]],
    ]
    tdr = TimeDelayingRidge(slim[0], slim[1], 1., 0.1, 'laplacian')
    for estimator in (Ridge(alpha=0.), 0., 0.01, tdr):
        model = ReceptiveField(slim[0], slim[1], 1.,
                               estimator=estimator)
        model.fit(x, y)
        assert_array_equal(model.delays_,
                           np.arange(slim[0], slim[1] + 1))
        assert_allclose(model.coef_, expected, atol=1e-1)
    tdr = TimeDelayingRidge(slim[0], slim[1], 1., 0.01, reg_type='foo')
    model = ReceptiveField(slim[0], slim[1], 1., estimator=tdr)
    assert_raises(ValueError, model.fit, x, y)
    tdr = TimeDelayingRidge(slim[0], slim[1], 1., 0.01, reg_type=['laplacian'])
    model = ReceptiveField(slim[0], slim[1], 1., estimator=tdr)
    assert_raises(ValueError, model.fit, x, y)

    # Now check the intercept_
    tdr = TimeDelayingRidge(slim[0], slim[1], 1., 0.)
    tdr_no = TimeDelayingRidge(slim[0], slim[1], 1., 0., fit_intercept=False)
    for estimator in (Ridge(alpha=0.), tdr,
                      Ridge(alpha=0., fit_intercept=False), tdr_no):
        x -= np.mean(x, axis=0)
        y -= np.mean(y, axis=0)
        model = ReceptiveField(slim[0], slim[1], 1., estimator=estimator)
        model.fit(x, y)
        assert_allclose(model.estimator_.intercept_, 0., atol=1e-2)
        assert_allclose(model.coef_, expected, atol=1e-1)
        x += 1e3
        model.fit(x, y)
        if estimator.fit_intercept:
            val = [-6000, 4000]
            # XXX This suggests there is some bug in the intercept code of TDR
            itol = 15 if estimator is tdr else 0.5
            ctol = 1e-2 if estimator is tdr else 5e-4
        else:
            val = itol = 0.
            # Same tolerances for TDR and Ridge, so the non-intercept code
            # is approximately equivalent
            ctol = 2.
        assert_allclose(model.estimator_.intercept_, val, atol=itol,
                        err_msg=repr(estimator))
        assert_allclose(model.coef_, expected, atol=ctol, rtol=ctol,
                        err_msg=repr(estimator))
        model = ReceptiveField(slim[0], slim[1], 1., fit_intercept=False)
        model.fit(x, y)
        assert_allclose(model.estimator_.intercept_, 0., atol=1e-7)
        score = model.score(x, y)
        assert_true(score > 0.6, msg=score)


run_tests_if_main()
