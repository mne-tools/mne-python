# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_true, assert_equal
from ...utils import requires_sklearn_0_15
from ..search_light import _SearchLight, _GeneralizationLight
from .. import Vectorizer


def make_data():
    n_epochs, n_chan, n_time = 50, 32, 10
    X = np.random.rand(n_epochs, n_chan, n_time)
    y = np.arange(n_epochs) % 2
    for ii in range(n_time):
        coef = np.random.randn(n_chan)
        X[y == 0, :, ii] += coef
        X[y == 1, :, ii] -= coef
    return X, y


@requires_sklearn_0_15
def test_SearchLight():
    """Test _SearchLight"""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, get_scorer, make_scorer

    X, y = make_data()
    n_epochs, _, n_time = X.shape
    # init
    assert_raises(ValueError, _SearchLight, 'foo')
    sl = _SearchLight(Ridge())
    sl = _SearchLight(LogisticRegression())
    # fit
    assert_equal(sl.__repr__()[:14], '<_SearchLight(')
    sl.fit(X, y)
    assert_equal(sl.__repr__()[-28:], ', fitted with 10 estimators>')
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

    sl = _SearchLight(LogisticRegression())
    assert_equal(sl.scoring, None)

    # Scoring method
    for err, scoring in [(ValueError, 'foo'), (TypeError, 999)]:
        sl = _SearchLight(LogisticRegression(), scoring=scoring)
        sl.fit(X, y)
        assert_raises(err, sl.score, X, y)

    # Check sklearn's roc_auc fix: scikit-learn/scikit-learn#6874
    # -- 3 class problem
    sl = _SearchLight(LogisticRegression(random_state=0), scoring='roc_auc')
    y = np.arange(len(X)) % 3
    sl.fit(X, y)
    assert_raises(ValueError, sl.score, X, y)
    # -- 2 class problem not in [0, 1]
    y = np.arange(len(X)) % 2 + 1
    sl.fit(X, y)
    score = sl.score(X, y)
    assert_array_equal(score, [roc_auc_score(y - 1, _y_pred - 1)
                               for _y_pred in sl.decision_function(X).T])
    y = np.arange(len(X)) % 2

    for method, scoring in [
            ('predict_proba', 'roc_auc'), ('predict', roc_auc_score)]:
        sl1 = _SearchLight(LogisticRegression(), scoring=scoring)
        sl1.fit(X, y)
        np.random.seed(0)
        X = np.random.randn(*X.shape)  # randomize X to avoid AUCs in [0, 1]
        score_sl = sl1.score(X, y)
        assert_array_equal(score_sl.shape, [n_time])
        assert_true(score_sl.dtype == float)

        # Check that scoring was applied adequately
        if isinstance(scoring, str):
            scoring = get_scorer(scoring)
        else:
            scoring = make_scorer(scoring)

        score_manual = [scoring(est, x, y) for est, x in zip(
                        sl1.estimators_, X.transpose(2, 0, 1))]
        assert_array_equal(score_manual, score_sl)

    # n_jobs
    sl = _SearchLight(LogisticRegression(random_state=0), n_jobs=1,
                      scoring='roc_auc')
    score_1job = sl.fit(X, y).score(X, y)
    sl.n_jobs = 2
    score_njobs = sl.fit(X, y).score(X, y)
    assert_array_equal(score_1job, score_njobs)
    sl.predict(X)

    # n_jobs > n_estimators
    sl.fit(X[..., [0]], y)
    sl.predict(X[..., [0]])

    # pipeline

    class _LogRegTransformer(LogisticRegression):
        # XXX needs transformer in pipeline to get first proba only
        def transform(self, X):
            return super(_LogRegTransformer, self).predict_proba(X)[..., 1]

    pipe = make_pipeline(_SearchLight(_LogRegTransformer()),
                         LogisticRegression())
    pipe.fit(X, y)
    pipe.predict(X)

    # n-dimensional feature space
    X = np.random.rand(10, 3, 4, 2)
    y = np.arange(10) % 2
    y_preds = list()
    for n_jobs in [1, 2]:
        pipe = _SearchLight(make_pipeline(Vectorizer(), LogisticRegression()),
                            n_jobs=n_jobs)
        y_preds.append(pipe.fit(X, y).predict(X))
        features_shape = pipe.estimators_[0].steps[0][1].features_shape_
        assert_array_equal(features_shape, [3, 4])
    assert_array_equal(y_preds[0], y_preds[1])


@requires_sklearn_0_15
def test_GeneralizationLight():
    """Test _GeneralizationLight"""
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = make_data()
    n_epochs, _, n_time = X.shape
    # fit
    gl = _GeneralizationLight(LogisticRegression())
    assert_equal(gl.__repr__()[:22], '<_GeneralizationLight(')
    gl.fit(X, y)

    assert_equal(gl.__repr__()[-28:], ', fitted with 10 estimators>')
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

    gl = _GeneralizationLight(LogisticRegression(), scoring='roc_auc')
    gl.fit(X, y)
    score = gl.score(X, y)
    auc = roc_auc_score(y, gl.estimators_[0].predict_proba(X[..., 0])[..., 1])
    assert_equal(score[0, 0], auc)

    for err, scoring in [(ValueError, 'foo'), (TypeError, 999)]:
        gl = _GeneralizationLight(LogisticRegression(), scoring=scoring)
        gl.fit(X, y)
        assert_raises(err, gl.score, X, y)

    # Check sklearn's roc_auc fix: scikit-learn/scikit-learn#6874
    # -- 3 class problem
    gl = _GeneralizationLight(LogisticRegression(), scoring='roc_auc')
    y = np.arange(len(X)) % 3
    gl.fit(X, y)
    assert_raises(ValueError, gl.score, X, y)
    # -- 2 class problem not in [0, 1]
    y = np.arange(len(X)) % 2 + 1
    gl.fit(X, y)
    score = gl.score(X, y)
    manual_score = [[roc_auc_score(y - 1, _y_pred) for _y_pred in _y_preds]
                    for _y_preds in gl.decision_function(X).transpose(1, 2, 0)]
    assert_array_equal(score, manual_score)

    # n_jobs
    gl = _GeneralizationLight(LogisticRegression(), n_jobs=2)
    gl.fit(X, y)
    y_pred = gl.predict(X)
    assert_array_equal(y_pred.shape, [n_epochs, n_time, n_time])
    score = gl.score(X, y)
    assert_array_equal(score.shape, [n_time, n_time])

    # n_jobs > n_estimators
    gl.fit(X[..., [0]], y)
    gl.predict(X[..., [0]])

    # n-dimensional feature space
    X = np.random.rand(10, 3, 4, 2)
    y = np.arange(10) % 2
    y_preds = list()
    for n_jobs in [1, 2]:
        pipe = _GeneralizationLight(
            make_pipeline(Vectorizer(), LogisticRegression()), n_jobs=n_jobs)
        y_preds.append(pipe.fit(X, y).predict(X))
        features_shape = pipe.estimators_[0].steps[0][1].features_shape_
        assert_array_equal(features_shape, [3, 4])
    assert_array_equal(y_preds[0], y_preds[1])
