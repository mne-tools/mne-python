# Author: Jean-Remi King, <jeanremi.king@gmail.com>
#         Marijn van Vliet, <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_allclose, assert_array_less)
import pytest

from mne import create_info, EpochsArray
from mne.fixes import is_regressor, is_classifier
from mne.utils import requires_sklearn, requires_version
from mne.decoding.base import (_get_inverse_funcs, LinearModel, get_coef,
                               cross_val_multiscore, BaseEstimator)
from mne.decoding.search_light import SlidingEstimator
from mne.decoding import (Scaler, TransformerMixin, Vectorizer,
                          GeneralizingEstimator)


def _make_data(n_samples=1000, n_features=5, n_targets=3):
    """Generate some testing data.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_targets : int
        The number of targets.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The measured data.
    Y : ndarray, shape (n_samples, n_targets)
        The latent variables generating the data.
    A : ndarray, shape (n_features, n_targets)
        The forward model, mapping the latent variables (=Y) to the measured
        data (=X).
    """
    # Define Y latent factors
    np.random.seed(0)
    cov_Y = np.eye(n_targets) * 10 + np.random.rand(n_targets, n_targets)
    cov_Y = (cov_Y + cov_Y.T) / 2.
    mean_Y = np.random.rand(n_targets)
    Y = np.random.multivariate_normal(mean_Y, cov_Y, size=n_samples)

    # The Forward model
    A = np.random.randn(n_features, n_targets)

    X = Y.dot(A.T)
    X += np.random.randn(n_samples, n_features)  # add noise
    X += np.random.rand(n_features)  # Put an offset

    return X, Y, A


@requires_sklearn
def test_get_coef():
    """Test getting linear coefficients (filters/patterns) from estimators."""
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    lm_classification = LinearModel()
    assert (is_classifier(lm_classification))

    lm_regression = LinearModel(Ridge())
    assert (is_regressor(lm_regression))

    parameters = {'kernel': ['linear'], 'C': [1, 10]}
    lm_gs_classification = LinearModel(
        GridSearchCV(svm.SVC(), parameters, cv=2, refit=True, n_jobs=1))
    assert (is_classifier(lm_gs_classification))

    lm_gs_regression = LinearModel(
        GridSearchCV(svm.SVR(), parameters, cv=2, refit=True, n_jobs=1))
    assert (is_regressor(lm_gs_regression))

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

    X, y, A = _make_data(n_samples=1000, n_features=3, n_targets=1)

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
        assert (expected_n == len(_get_inverse_funcs(est)))

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

    # II. Test get coef for classification/regression estimators and pipelines
    rng = np.random.RandomState(0)
    for clf in (lm_regression,
                lm_gs_classification,
                make_pipeline(StandardScaler(), lm_classification),
                make_pipeline(StandardScaler(), lm_gs_regression)):

        # generate some categorical/continuous data
        # according to the type of estimator.
        if is_classifier(clf):
            n, n_features = 1000, 3
            X = rng.rand(n, n_features)
            y = np.arange(n) % 2
        else:
            X, y, A = _make_data(n_samples=1000, n_features=3, n_targets=1)
            y = np.ravel(y)

        clf.fit(X, y)

        # Retrieve final linear model
        filters = get_coef(clf, 'filters_', False)
        if hasattr(clf, 'steps'):
            if hasattr(clf.steps[-1][-1].model, 'best_estimator_'):
                # Linear Model with GridSearchCV
                coefs = clf.steps[-1][-1].model.best_estimator_.coef_
            else:
                # Standard Linear Model
                coefs = clf.steps[-1][-1].model.coef_
        else:
            if hasattr(clf.model, 'best_estimator_'):
                # Linear Model with GridSearchCV
                coefs = clf.model.best_estimator_.coef_
            else:
                # Standard Linear Model
                coefs = clf.model.coef_
        if coefs.ndim == 2 and coefs.shape[0] == 1:
            coefs = coefs[0]
        assert_array_equal(filters, coefs)
        patterns = get_coef(clf, 'patterns_', False)
        assert (filters[0] != patterns[0])
        n_chans = X.shape[1]
        assert_array_equal(filters.shape, patterns.shape, [n_chans, n_chans])

    # Inverse transform linear model
    filters_inv = get_coef(clf, 'filters_', True)
    assert (filters[0] != filters_inv[0])
    patterns_inv = get_coef(clf, 'patterns_', True)
    assert (patterns[0] != patterns_inv[0])


class _Noop(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()

    inverse_transform = transform


@requires_sklearn
@pytest.mark.parametrize('inverse', (True, False))
@pytest.mark.parametrize('Scale, kwargs', [
    (Scaler, dict(info=None, scalings='mean')),
    (_Noop, dict()),
])
def test_get_coef_inverse_transform(inverse, Scale, kwargs):
    """Test get_coef with and without inverse_transform."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    lm_regression = LinearModel(Ridge())
    X, y, A = _make_data(n_samples=1000, n_features=3, n_targets=1)
    # Check with search_light and combination of preprocessing ending with sl:
    # slider = SlidingEstimator(make_pipeline(StandardScaler(), lm_regression))
    # XXX : line above should work but does not as only last step is
    # used in get_coef ...
    slider = SlidingEstimator(make_pipeline(lm_regression))
    X = np.transpose([X, -X], [1, 2, 0])  # invert X across 2 time samples
    clf = make_pipeline(Scale(**kwargs), slider)
    clf.fit(X, y)
    patterns = get_coef(clf, 'patterns_', inverse)
    filters = get_coef(clf, 'filters_', inverse)
    assert_array_equal(filters.shape, patterns.shape, X.shape[1:])
    # the two time samples get inverted patterns
    assert_equal(patterns[0, 0], -patterns[0, 1])
    for t in [0, 1]:
        filters_t = get_coef(
            clf.named_steps['slidingestimator'].estimators_[t],
            'filters_', False)
        if Scale is _Noop:
            assert_array_equal(filters_t, filters[:, t])


@requires_sklearn
@pytest.mark.parametrize('n_features', [1, 5])
@pytest.mark.parametrize('n_targets', [1, 3])
def test_get_coef_multiclass(n_features, n_targets):
    """Test get_coef on multiclass problems."""
    # Check patterns with more than 1 regressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    X, Y, A = _make_data(
        n_samples=30000, n_features=n_features, n_targets=n_targets)
    lm = LinearModel(LinearRegression()).fit(X, Y)
    assert_array_equal(lm.filters_.shape, lm.patterns_.shape)
    if n_targets == 1:
        want_shape = (n_features,)
    else:
        want_shape = (n_targets, n_features)
    assert_array_equal(lm.filters_.shape, want_shape)
    if n_features > 1 and n_targets > 1:
        assert_array_almost_equal(A, lm.patterns_.T, decimal=2)
    lm = LinearModel(Ridge(alpha=0))
    clf = make_pipeline(lm)
    clf.fit(X, Y)
    if n_features > 1 and n_targets > 1:
        assert_allclose(A, lm.patterns_.T, atol=2e-2)
    coef = get_coef(clf, 'patterns_', inverse_transform=True)
    assert_allclose(lm.patterns_, coef, atol=1e-5)

    # With epochs, scaler, and vectorizer (typical use case)
    X_epo = X.reshape(X.shape + (1,))
    info = create_info(n_features, 1000., 'eeg')
    lm = LinearModel(Ridge(alpha=1))
    clf = make_pipeline(
        Scaler(info, scalings=dict(eeg=1.)),  # XXX adding this step breaks
        Vectorizer(),
        lm,
    )
    clf.fit(X_epo, Y)
    if n_features > 1 and n_targets > 1:
        assert_allclose(A, lm.patterns_.T, atol=2e-2)
    coef = get_coef(clf, 'patterns_', inverse_transform=True)
    lm_patterns_ = lm.patterns_[..., np.newaxis]
    assert_allclose(lm_patterns_, coef, atol=1e-5)

    # Check can pass fitting parameters
    lm.fit(X, Y, sample_weight=np.ones(len(Y)))


@requires_version('sklearn', '0.22')  # roc_auc_ovr_weighted
@pytest.mark.parametrize('n_classes, n_channels, n_times', [
    (4, 10, 2),
    (4, 3, 2),
    (3, 2, 1),
    (3, 1, 2),
])
def test_get_coef_multiclass_full(n_classes, n_channels, n_times):
    """Test a full example with pattern extraction."""
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    data = np.zeros((10 * n_classes, n_channels, n_times))
    # Make only the first channel informative
    for ii in range(n_classes):
        data[ii * 10:(ii + 1) * 10, 0] = ii
    events = np.zeros((len(data), 3), int)
    events[:, 0] = np.arange(len(events))
    events[:, 2] = data[:, 0, 0]
    info = create_info(n_channels, 1000., 'eeg')
    epochs = EpochsArray(data, info, events, tmin=0)
    clf = make_pipeline(
        Scaler(epochs.info), Vectorizer(),
        LinearModel(LogisticRegression(random_state=0, multi_class='ovr')),
    )
    scorer = 'roc_auc_ovr_weighted'
    time_gen = GeneralizingEstimator(clf, scorer, verbose=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    n_splits = 3
    cv = StratifiedKFold(n_splits=n_splits)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, verbose=True)
    want = (n_splits,)
    if n_times > 1:
        want += (n_times, n_times)
    assert scores.shape == want
    assert_array_less(0.8, scores)
    clf.fit(X, y)
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)
    assert patterns.shape == (n_classes, n_channels, n_times)
    assert_allclose(patterns[:, 1:], 0., atol=1e-7)  # no other channels useful


@requires_sklearn
def test_linearmodel():
    """Test LinearModel class for computing filters and patterns."""
    # check categorical target fit in standard linear model
    from sklearn.linear_model import LinearRegression
    rng = np.random.RandomState(0)
    clf = LinearModel()
    n, n_features = 20, 3
    X = rng.rand(n, n_features)
    y = np.arange(n) % 2
    clf.fit(X, y)
    assert_equal(clf.filters_.shape, (n_features,))
    assert_equal(clf.patterns_.shape, (n_features,))
    with pytest.raises(ValueError):
        wrong_X = rng.rand(n, n_features, 99)
        clf.fit(wrong_X, y)

    # check categorical target fit in standard linear model with GridSearchCV
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel': ['linear'], 'C': [1, 10]}
    clf = LinearModel(
        GridSearchCV(svm.SVC(), parameters, cv=2, refit=True, n_jobs=1))
    clf.fit(X, y)
    assert_equal(clf.filters_.shape, (n_features,))
    assert_equal(clf.patterns_.shape, (n_features,))
    with pytest.raises(ValueError):
        wrong_X = rng.rand(n, n_features, 99)
        clf.fit(wrong_X, y)

    # check continuous target fit in standard linear model with GridSearchCV
    n_targets = 1
    Y = rng.rand(n, n_targets)
    clf = LinearModel(
        GridSearchCV(svm.SVR(), parameters, cv=2, refit=True, n_jobs=1))
    clf.fit(X, y)
    assert_equal(clf.filters_.shape, (n_features, ))
    assert_equal(clf.patterns_.shape, (n_features, ))
    with pytest.raises(ValueError):
        wrong_y = rng.rand(n, n_features, 99)
        clf.fit(X, wrong_y)

    # check multi-target fit in standard linear model
    n_targets = 5
    Y = rng.rand(n, n_targets)
    clf = LinearModel(LinearRegression())
    clf.fit(X, Y)
    assert_equal(clf.filters_.shape, (n_targets, n_features))
    assert_equal(clf.patterns_.shape, (n_targets, n_features))
    with pytest.raises(ValueError):
        wrong_y = rng.rand(n, n_features, 99)
        clf.fit(X, wrong_y)


@requires_sklearn
def test_cross_val_multiscore():
    """Test cross_val_multiscore for computing scores on decoding over time."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression, LinearRegression

    logreg = LogisticRegression(solver='liblinear', random_state=0)

    # compare to cross-val-score
    X = np.random.rand(20, 3)
    y = np.arange(20) % 2
    cv = KFold(2, random_state=0, shuffle=True)
    clf = logreg
    assert_array_equal(cross_val_score(clf, X, y, cv=cv),
                       cross_val_multiscore(clf, X, y, cv=cv))

    # Test with search light
    X = np.random.rand(20, 4, 3)
    y = np.arange(20) % 2
    clf = SlidingEstimator(logreg, scoring='accuracy')
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
    pytest.raises(ValueError, cross_val_multiscore, clf, X, y, cv=cv,
                  scoring='roc_auc')
    clf = SlidingEstimator(logreg, scoring='roc_auc')
    scores_auc = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=1)
    scores_auc_manual = list()
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        scores_auc_manual.append(clf.score(X[test], y[test]))
    assert_array_equal(scores_auc, scores_auc_manual)

    # indirectly test that cross_val_multiscore rightly detects the type of
    # estimator and generates a StratifiedKFold for classiers and a KFold
    # otherwise
    X = np.random.randn(1000, 3)
    y = np.ones(1000, dtype=int)
    y[::2] = 0
    clf = logreg
    reg = LinearRegression()
    for cross_val in (cross_val_score, cross_val_multiscore):
        manual = cross_val(clf, X, y, cv=StratifiedKFold(2))
        auto = cross_val(clf, X, y, cv=2)
        assert_array_equal(manual, auto)

        manual = cross_val(reg, X, y, cv=KFold(2))
        auto = cross_val(reg, X, y, cv=2)
        assert_array_equal(manual, auto)
