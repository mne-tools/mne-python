# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_equal, assert_raises
from ...utils import requires_sklearn_0_15
from ..base import _get_inverse_funcs, LinearModel, get_coef
from ..search_light import _SearchLight


@requires_sklearn_0_15
def test_get_coef():
    """Test the retrieval of linear coefficients (filters and patterns) from
    simple and pipeline estimators.
    """
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

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
            return X

    np.random.RandomState(0)
    n_samples, n_features = 20, 3
    y = (np.arange(n_samples) % 2) * 2 - 1
    w = np.random.randn(n_features, 1)
    X = w.dot(y[np.newaxis, :]).T + np.random.randn(n_samples, n_features)

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
    for clf in (LinearModel(), make_pipeline(StandardScaler(), LinearModel())):
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
    mean, std = X.mean(0), X.std(0)
    X = (X - mean) / std
    coef = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    patterns_manual = np.cov(X.T).dot(coef)
    assert_array_almost_equal(patterns, patterns_manual * std + mean)

    # Check with search_light:
    n_samples, n_features, n_times = 20, 3, 5
    y = np.arange(n_samples) % 2
    X = np.random.rand(n_samples, n_features, n_times)
    clf = _SearchLight(make_pipeline(StandardScaler(), LinearModel()))
    clf.fit(X, y)
    for inverse in (True, False):
        patterns = get_coef(clf, 'patterns_', inverse)
        filters = get_coef(clf, 'filters_', inverse)
        assert_array_equal(filters.shape, patterns.shape,
                           [n_features, n_times])
    for t in [0, 1]:
        assert_array_equal(get_coef(clf.estimators_[t], 'filters_', False),
                           filters[t])


@requires_sklearn_0_15
def test_linearmodel():
    """Test LinearModel class for computing filters and patterns.
    """
    clf = LinearModel()
    X = np.random.rand(20, 3)
    y = np.arange(20) % 2
    clf.fit(X, y)
    assert_equal(clf.filters_.shape, (3,))
    assert_equal(clf.patterns_.shape, (3,))
    assert_raises(ValueError, clf.fit, np.random.rand(20, 3, 2), y)
