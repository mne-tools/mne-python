# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_equal, assert_raises
from mne.utils import requires_sklearn_0_15
from mne.decoding.base import (_get_inverse_funcs, LinearModel, get_coef,
                               cross_val_multiscore)
from mne.decoding.search_light import SlidingEstimator
from mne.decoding import Scaler


@requires_sklearn_0_15
def test_get_coef():
    """Test the retrieval of linear coefficients (filters and patterns) from
    simple and pipeline estimators.
    """
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    scale = lambda x: (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True)

    # Define a classifier, an invertible transformer and an non-invertible one.

    class Clf(BaseEstimator):
        def fit(self, X, y):
            return self

    class NoInv(TransformerMixin):
        def fit(self, X, y):
            return self

        def transform(self, X):
            return X

    class Inv(NoInv):
        def inverse_transform(self, X):
            return X * 666.

    np.random.RandomState(0)
    n_samples, n_features = 20, 3
    y = scale(np.arange(n_samples) % 2)
    w = np.random.randn(n_features, 1)
    X = w.dot(y[np.newaxis, :]).T + np.random.randn(n_samples, n_features)
    X = scale(X)

    # I. Test inverse function

    # Check that we retrieve the right number of inverse functions even if
    # there are nested pipelines
    good_estimators = [
        (1, make_pipeline(Inv(), Clf())),
        (2, make_pipeline(Inv(), Inv(), Clf())),
        (3, make_pipeline(Inv(), make_pipeline(Inv(), Inv()), Clf())),
    ]

    for expected_n, est in good_estimators:
        est.fit(X, y)
        assert_true(expected_n == len(_get_inverse_funcs(est)))

    bad_estimators = [
        Clf(),  # no preprocessing
        Inv(),  # final estimator isn't classifier
        make_pipeline(NoInv(), Clf()),  # first step isn't invertible
        make_pipeline(Inv(), make_pipeline(
            Inv(), NoInv()), Clf()),  # nested step isn't invertible
    ]
    for est in bad_estimators:
        est.fit(X, y)
        invs = _get_inverse_funcs(est)
        assert_equal(invs, list())

    # II. Test get coef for simple estimator and pipelines
    for clf in (LinearModel(), make_pipeline(Inv(), StandardScaler(), LinearModel())):
        clf.fit(X, y)
        # Retrieve final linear model
        filters = get_coef(clf, 'filters_', False)
        if hasattr(clf, 'steps'):
            coefs = clf.steps[-1][-1].model.coef_
        else:
            coefs = clf.model.coef_
        assert_array_equal(filters, coefs[0])
        patterns = get_coef(clf, 'patterns_', False)
        assert_true(filters[0] != patterns[0])
        n_chans = X.shape[1]
        assert_array_equal(filters.shape, patterns.shape, [n_chans, n_chans])

    # Inverse transform linear model
    filters_inv = get_coef(clf, 'filters_', True)
    assert_true(filters[0] != filters_inv[0])
    patterns_inv = get_coef(clf, 'patterns_', True)
    assert_true(patterns[0] != patterns_inv[0])

    # Check patterns values
    clf = make_pipeline(StandardScaler(), LinearModel(LinearRegression()))
    clf.fit(X, y)
    patterns = get_coef(clf, 'patterns_', True)
    coef = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    patterns_manual = np.cov(X.T).dot(coef)
    assert_array_almost_equal(patterns, patterns_manual)

    # Check with search_light and combination of preprocessing ending with sl:
    n_samples, n_features, n_times = 20, 3, 5
    y = scale(np.arange(n_samples) % 2)
    X = scale(np.random.rand(n_samples, n_features, n_times))
    slider = SlidingEstimator(make_pipeline(StandardScaler(), LinearModel()))

    clfs = (make_pipeline(Scaler(None, scalings='mean'), slider), slider)
    for clf in clfs:
        clf.fit(X, y)
        for inverse in (True, False):
            patterns = get_coef(clf, 'patterns_', inverse)
            filters = get_coef(clf, 'filters_', inverse)
            assert_array_equal(filters.shape, patterns.shape,
                               [n_features, n_times])
    for t in [0, 1]:
        assert_array_equal(get_coef(clf.estimators_[t], 'filters_', False),
                           filters[:, t])

    # Check patterns with more than 1 regressor
    n_samples, n_features, n_regressors = 2000, 3, 2
    y = np.transpose([(np.arange(n_samples) % 2) * 2 - 1] * n_regressors)
    noise = np.random.randn(n_samples, n_features)
    X = np.random.randn(n_features, n_regressors).dot(y.T).T + noise
    # normalization is necessary for filters and patterns to be dual
    X, y = scale(X), scale(y)
    # We normalize outside the pipeline to check that we find the same results
    # as a subtraction.
    lm = LinearModel(LinearRegression())
    lm.fit(X, y)
    assert_array_equal(lm.filters_.shape, lm.patterns_.shape)
    assert_array_equal(lm.filters_.shape, [n_regressors, n_features])
    subtraction = (X[y[:, 0]==1].mean(0) - X[y[:, 0]==-1].mean(0)) / 4.
    assert_array_almost_equal(subtraction, lm.patterns_[0])


@requires_sklearn_0_15
def test_linearmodel():
    """Test LinearModel class for computing filters and patterns.
    """
    clf = LinearModel()
    X = np.random.rand(20, 3)
    X = (X - X.mean(0)) / X.std(0)
    y = np.arange(20) % 2
    y = (y - y.mean(0)) / y.std(0)
    clf.fit(X, y)
    assert_equal(clf.filters_.shape, (3,))
    assert_equal(clf.patterns_.shape, (3,))
    assert_raises(ValueError, clf.fit, np.random.rand(20, 3, 2), y)


@requires_sklearn_0_15
def test_cross_val_multiscore():
    """Test cross_val_multiscore for computing scores on decoding over time.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LogisticRegression

    # compare to cross-val-score
    X = np.random.rand(20, 3)
    y = np.arange(20) % 2
    clf = LogisticRegression()
    cv = KFold(2, random_state=0)
    assert_array_equal(cross_val_score(clf, X, y, cv=cv),
                       cross_val_multiscore(clf, X, y, cv=cv))

    # Test with search light
    X = np.random.rand(20, 4, 3)
    y = np.arange(20) % 2
    clf = SlidingEstimator(LogisticRegression(), scoring='accuracy')
    scores_acc = cross_val_multiscore(clf, X, y, cv=cv)
    assert_array_equal(np.shape(scores_acc), [2, 3])

    # check values
    scores_acc_manual = list()
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        scores_acc_manual.append(clf.score(X[test], y[test]))
    assert_array_equal(scores_acc, scores_acc_manual)

    # check scoring metric
    # raise an error if scoring is defined at cross-val-score level and
    # search light, because search light does not return a 1-dimensional
    # prediction.
    assert_raises(ValueError, cross_val_multiscore, clf, X, y, cv=cv,
                  scoring='roc_auc')
    clf = SlidingEstimator(LogisticRegression(), scoring='roc_auc')
    scores_auc = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=1)
    scores_auc_manual = list()
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        scores_auc_manual.append(clf.score(X[test], y[test]))
    assert_array_equal(scores_auc, scores_auc_manual)
