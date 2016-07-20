# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_true
from ...utils import requires_sklearn
from ..search_light import SearchLight, GeneralizationLight


def make_data():
    n_epochs, n_chan, n_time = 50, 32, 10
    X = np.random.rand(n_epochs, n_chan, n_time)
    y = np.arange(n_epochs) % 2
    for ii in range(n_time):
        coef = np.random.randn(n_chan)
        X[y == 0, :, ii] += coef
        X[y == 1, :, ii] -= coef
    return X, y


@requires_sklearn
def test_searchlight():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    X, y = make_data()
    n_epochs, _, n_time = X.shape
    # fit
    sl = SearchLight(LogisticRegression())
    sl.fit(X, y)
    assert_raises(ValueError, sl.fit, X[1:], y)
    assert_raises(ValueError, sl.fit, X[:, :, 0], y)

    # transforms
    assert_raises(ValueError, sl.predict, X[:, :, :2])
    y_pred = sl.predict(X)
    assert_true(y_pred.dtype == int)
    assert_array_equal(y_pred.shape, [n_epochs, n_time])
    y_proba = sl.predict_proba(X)
    assert_true(y_proba.dtype == float)
    assert_array_equal(y_proba.shape, [n_epochs, n_time, 2])

    # score
    score = sl.score(X, y)
    assert_array_equal(score.shape, [n_time])
    assert_true(np.sum(np.abs(score)) != 0)
    assert_true(score.dtype == float)

    # n_jobs
    sl = SearchLight(LogisticRegression(), n_jobs=2)
    sl.fit(X, y)
    sl.predict(X)
    sl.score(X, y)

    # pipeline

    class _LogRegTransformer(LogisticRegression):
        # XXX needs transformer in pipeline to get first proba only
        def transform(self, X):
            return super(_LogRegTransformer, self).predict_proba(X)[..., 1]

    pipe = make_pipeline(SearchLight(_LogRegTransformer()),
                         LogisticRegression())
    pipe.fit(X, y)
    pipe.predict(X)


@requires_sklearn
def test_generalizationlight():
    from sklearn.linear_model import LogisticRegression
    X, y = make_data()
    n_epochs, _, n_time = X.shape
    # fit
    gl = GeneralizationLight(LogisticRegression())
    gl.fit(X, y)

    # transforms
    y_pred = gl.predict(X)
    assert_array_equal(y_pred.shape, [n_epochs, n_time, n_time])
    assert_true(y_pred.dtype == int)
    y_proba = gl.predict_proba(X)
    assert_true(y_proba.dtype == float)
    assert_array_equal(y_proba.shape, [n_epochs, n_time, n_time, 2])

    # transform to different datasize
    y_pred = gl.predict(X[:, :, :2])
    assert_array_equal(y_pred.shape, [n_epochs, n_time, 2])

    # score
    score = gl.score(X[:, :, :3], y)
    assert_array_equal(score.shape, [n_time, 3])
    assert_true(np.sum(np.abs(score)) != 0)
    assert_true(score.dtype == float)

    # n_jobs
    gl = GeneralizationLight(LogisticRegression(), n_jobs=2)
    gl.fit(X, y)
    y_pred = gl.predict(X)
    assert_array_equal(y_pred.shape, [n_epochs, n_time, n_time])
    score = gl.score(X, y)
    assert_array_equal(score.shape, [n_time, n_time])
