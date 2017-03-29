# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .mixin import TransformerMixin
from .base import BaseEstimator, _check_estimator, _make_scorer
from ..parallel import parallel_func


class _SearchLight(BaseEstimator, TransformerMixin):
    """Search Light.

    Fit, predict and score a series of models to each subset of the dataset
    along the last dimension.

    Parameters
    ----------
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    scoring : callable, string, defaults to None
        Score function (or loss function) with signature
        score_func(y, y_pred, **kwargs).
        Note that the predict_method is automatically identified if scoring is
        a string (e.g. scoring="roc_auc" calls predict_proba) but is not
        automatically set if scoring is a callable (e.g.
        scoring=sklearn.metrics.roc_auc_score).
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    """

    def __init__(self, base_estimator, scoring=None, n_jobs=1):  # noqa: D102
        _check_estimator(base_estimator)
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.scoring = scoring

        if not isinstance(self.n_jobs, int):
            raise ValueError('n_jobs must be int, got %s' % n_jobs)

    def __repr__(self):  # noqa: D105
        repr_str = '<' + super(_SearchLight, self).__repr__()
        if hasattr(self, 'estimators_'):
            repr_str = repr_str[:-1]
            repr_str += ', fitted with %i estimators' % len(self.estimators_)
        return repr_str + '>'

    def fit_transform(self, X, y):
        """Fit and transform a series of independent estimators to the dataset.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The training input samples. For each data slice, a clone estimator
            is fitted independently. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators) | (n_samples, n_estimators, n_targets)
            The predicted values for each estimator.
        """  # noqa: E501
        return self.fit(X, y).transform(X)

    def fit(self, X, y):
        """Fit a series of independent estimators to the dataset.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The training input samples. For each data slice, a clone estimator
            is fitted independently. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        self : object
            Return self.
        """
        self._check_Xy(X, y)
        self.estimators_ = list()
        # For fitting, the parallelization is across estimators.
        parallel, p_func, n_jobs = parallel_func(_sl_fit, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        estimators = parallel(
            p_func(self.base_estimator, split, y)
            for split in np.array_split(X, n_jobs, axis=-1))
        self.estimators_ = np.concatenate(estimators, 0)
        return self

    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        if X.shape[-1] != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        parallel, p_func, n_jobs = parallel_func(_sl_transform, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        est_splits = np.array_split(self.estimators_, n_jobs)
        y_pred = parallel(p_func(est, x, method)
                          for (est, x) in zip(est_splits, X_splits))

        if n_jobs > 1:
            y_pred = np.concatenate(y_pred, axis=1)
        else:
            y_pred = y_pred[0]
        return y_pred

    def transform(self, X):
        """Transform each data slice with a series of independent estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The input samples. For each data slice, the corresponding estimator
            makes a transformation of the data:
            e.g. [estimators[ii].transform(X[..., ii])
                  for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        Xt : array, shape (n_samples, n_estimators)
            The transformed values generated by each estimator.
        """
        return self._transform(X, 'transform')

    def predict(self, X):
        """Predict each data slice with a series of independent estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The input samples. For each data slice, the corresponding estimator
            makes the sample predictions:
            e.g. [estimators[ii].predict(X[..., ii])
                  for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators) | (n_samples, n_estimators, n_targets)
            Predicted values for each estimator/data slice.
        """  # noqa: E501
        return self._transform(X, 'predict')

    def predict_proba(self, X):
        """Predict each data slice with a series of independent estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The input samples. For each data slice, the corresponding estimator
            makes the sample probabilistic predictions:
            e.g. [estimators[ii].predict_proba(X[..., ii])
                  for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)


        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_classes)
            Predicted probabilities for each estimator/data slice.
        """
        return self._transform(X, 'predict_proba')

    def decision_function(self, X):
        """Estimate distances of each data slice to the hyperplanes.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The input samples. For each data slice, the corresponding estimator
            outputs the distance to the hyperplane:
            e.g. [estimators[ii].decision_function(X[..., ii])
                  for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_classes * (n_classes-1) // 2)
            Predicted distances for each estimator/data slice.

        Notes
        -----
        This requires base_estimator to have a `decision_function` method.
        """  # noqa: E501
        return self._transform(X, 'decision_function')

    def _check_Xy(self, X, y=None):
        """Aux. function to check input data."""
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def score(self, X, y):
        """Score each estimator/data slice couple.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_estimators)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction: e.g. [estimators[ii].score(X[..., ii], y)
                                         for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        score : array, shape (n_samples, n_estimators)
            Score for each estimator / data slice couple.
        """
        self._check_Xy(X)
        if X.shape[-1] != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')

        scoring = _make_scorer(self.scoring)
        y = _fix_auc(scoring, y)

        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        parallel, p_func, n_jobs = parallel_func(_sl_score, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        est_splits = np.array_split(self.estimators_, n_jobs)
        score = parallel(p_func(est, scoring, x, y)
                         for (est, x) in zip(est_splits, X_splits))

        if n_jobs > 1:
            score = np.concatenate(score, axis=0)
        else:
            score = score[0]
        return score


def _sl_fit(estimator, X, y):
    """Aux. function to fit _SearchLight in parallel.

    Fit a clone estimator to each slice of data.

    Parameters
    ----------
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    y : array, shape (n_sample, )
        The target values.

    Returns
    -------
    estimators_ : list of estimators
        The fitted estimators.
    """
    from sklearn.base import clone
    estimators_ = list()
    for ii in range(X.shape[-1]):
        est = clone(estimator)
        est.fit(X[..., ii], y)
        estimators_.append(est)
    return estimators_


def _sl_transform(estimators, X, method):
    """Aux. function to transform _SearchLight in parallel.

    Applies transform/predict/decision_function etc for each slice of data.

    Parameters
    ----------
    estimators : list of estimators
        The fitted estimators.
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    method : str
        The estimator method to use (e.g. 'predict', 'transform').

    Returns
    -------
    y_pred : array, shape (n_samples, n_estimators, n_classes * (n_classes-1) // 2)
        The transformations for each slice of data.
    """  # noqa: E501
    for ii, est in enumerate(estimators):
        transform = getattr(est, method)
        _y_pred = transform(X[..., ii])
        # Initialize array of predictions on the first transform iteration
        if ii == 0:
            y_pred = _sl_init_pred(_y_pred, X)
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _sl_init_pred(y_pred, X):
    """Aux. function to _SearchLight to initialize y_pred."""
    n_sample, n_iter = X.shape[0], X.shape[-1]
    if y_pred.ndim > 1:
        # for estimator that generate multidimensional y_pred,
        # e.g. clf.predict_proba()
        y_pred = np.zeros(np.r_[n_sample, n_iter, y_pred.shape[1:]],
                          y_pred.dtype)
    else:
        # for estimator that generate unidimensional y_pred,
        # e.g. clf.predict()
        y_pred = np.zeros((n_sample, n_iter), y_pred.dtype)
    return y_pred


def _sl_score(estimators, scoring, X, y):
    """Aux. function to score _SearchLight in parallel.

    Predict and score each slice of data.

    Parameters
    ----------
    estimators : list of estimators
        The fitted estimators.
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    scoring : callable, string or None
        If scoring is None (default), the predictions are internally
        generated by estimator.score(). Else, we must first get the
        predictions to pass them to ad-hoc scorer.
    y : array, shape (n_samples,) | (n_samples, n_targets)
        The target values.

    Returns
    -------
    score : array, shape (n_estimators,)
        The score for each slice of data.
    """
    n_iter = X.shape[-1]
    for ii, est in enumerate(estimators):
        if scoring is not None:
            _score = scoring(est, X[..., ii], y)
        else:
            _score = est.score(X[..., ii], y)
        # Initialize array of scores on the first score iteration
        if ii == 0:
            if isinstance(_score, np.ndarray):
                dtype = _score.dtype
                shape = _score.shape
                np.r_[n_iter, _score.shape]
            else:
                dtype = type(_score)
                shape = n_iter
            score = np.zeros(shape, dtype)
        score[ii] = _score
    return score


def _check_method(estimator, method):
    """Check that an estimator has the method attribute.

    If method == 'transform'  and estimator does not have 'transform', use
    'predict' instead.
    """
    if method == 'transform' and not hasattr(estimator, 'transform'):
        method = 'predict'
    if not hasattr(estimator, method):
        ValueError('base_estimator does not have `%s` method.' % method)
    return method


class _GeneralizationLight(_SearchLight):
    """Generalization Light.

    Fit a search-light along the last dimension and use them to apply a
    systematic cross-feature generalization.

    Parameters
    ----------
    base_estimator : object
        The base estimator to iteratively fit on a subset of the dataset.
    scoring : callable | string | None
        Score function (or loss function) with signature
        score_func(y, y_pred, **kwargs).
        Note that the predict_method is automatically identified if scoring is
        a string (e.g. scoring="roc_auc" calls predict_proba) but is not
        automatically set if scoring is a callable (e.g.
        scoring=sklearn.metrics.roc_auc_score).
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    """

    def __repr__(self):  # noqa: D105
        repr_str = super(_GeneralizationLight, self).__repr__()
        if hasattr(self, 'estimators_'):
            repr_str = repr_str[:-1]
            repr_str += ', fitted with %i estimators>' % len(self.estimators_)
        return repr_str

    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        parallel, p_func, n_jobs = parallel_func(_gl_transform, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        y_pred = parallel(
            p_func(self.estimators_, x_split, method)
            for x_split in np.array_split(X, n_jobs, axis=-1))

        y_pred = np.concatenate(y_pred, axis=2)
        return y_pred

    def transform(self, X):
        """Transform each data slice with all possible estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The input samples. For estimator the corresponding data slice is
            used to make a transformation. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        Xt : array, shape (n_samples, n_estimators, n_slices)
            The transformed values generated by each estimator.
        """
        return self._transform(X, 'transform')

    def predict(self, X):
        """Predict each data slice with all possible estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. For each data slice, a fitted estimator
            predicts each slice of the data independently. The feature
            dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices) | (n_samples, n_estimators, n_slices, n_targets)
            The predicted values for each estimator.
        """  # noqa: E501
        return self._transform(X, 'predict')

    def predict_proba(self, X):
        """Estimate probabilistic estimates of each data slice with all possible estimators.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. For each data slice, a fitted estimator
            predicts a slice of the data. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes)
            The predicted values for each estimator.

        Notes
        -----
        This requires base_estimator to have a `predict_proba` method.
        """  # noqa: E501
        return self._transform(X, 'predict_proba')

    def decision_function(self, X):
        """Estimate distances of each data slice to all hyperplanes.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. Each estimator outputs the distance to
            its hyperplane: e.g. [estimators[ii].decision_function(X[..., ii])
                                  for ii in range(n_estimators)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes * (n_classes-1) // 2)
            The predicted values for each estimator.

        Notes
        -----
        This requires base_estimator to have a `decision_function` method.
        """  # noqa: E501
        return self._transform(X, 'decision_function')

    def score(self, X, y):
        """Score each of the estimators on the tested dimensions.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction: e.g. [estimators[ii].score(X[..., ii], y)
                                         for ii in range(n_slices)]
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        score : array, shape (n_samples, n_estimators, n_slices)
            Score for each estimator / data slice couple.
        """
        self._check_Xy(X)
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        parallel, p_func, n_jobs = parallel_func(_gl_score, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        scoring = _make_scorer(self.scoring)
        y = _fix_auc(scoring, y)

        score = parallel(p_func(self.estimators_, scoring, x, y)
                         for x in X_splits)

        if n_jobs > 1:
            score = np.concatenate(score, axis=1)
        else:
            score = score[0]
        return score


def _gl_transform(estimators, X, method):
    """Transform the dataset.

    This will apply each estimator to all slices of the data.

    Parameters
    ----------
    X : array, shape (n_samples, nd_features, n_slices)
        The training input samples. For each data slice, a clone estimator
        is fitted independently. The feature dimension can be multidimensional
        e.g. X.shape = (n_samples, n_features_1, n_features_2, n_estimators)

    Returns
    -------
    Xt : array, shape (n_samples, n_slices)
        The transformed values generated by each estimator.
    """
    n_sample, n_iter = X.shape[0], X.shape[-1]
    for ii, est in enumerate(estimators):
        # stack generalized data for faster prediction
        X_stack = X.transpose(np.r_[0, X.ndim - 1, range(1, X.ndim - 1)])
        X_stack = X_stack.reshape(np.r_[n_sample * n_iter, X_stack.shape[2:]])
        transform = getattr(est, method)
        _y_pred = transform(X_stack)
        # unstack generalizations
        if _y_pred.ndim == 2:
            _y_pred = np.reshape(_y_pred, [n_sample, n_iter, _y_pred.shape[1]])
        else:
            shape = np.r_[n_sample, n_iter, _y_pred.shape[1:]].astype(int)
            _y_pred = np.reshape(_y_pred, shape)
        # Initialize array of predictions on the first transform iteration
        if ii == 0:
            y_pred = _gl_init_pred(_y_pred, X, len(estimators))
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _gl_init_pred(y_pred, X, n_train):
    """Aux. function to _GeneralizationLight to initialize y_pred."""
    n_sample, n_iter = X.shape[0], X.shape[-1]
    if y_pred.ndim == 3:
        y_pred = np.zeros((n_sample, n_train, n_iter, y_pred.shape[-1]),
                          y_pred.dtype)
    else:
        y_pred = np.zeros((n_sample, n_train, n_iter), y_pred.dtype)
    return y_pred


def _gl_score(estimators, scoring, X, y):
    """Score _GeneralizationLight in parallel.

    Predict and score each slice of data.

    Parameters
    ----------
    estimators : list of estimators
        The fitted estimators.
    scoring : callable, string or None
        If scoring is None (default), the predictions are internally
        generated by estimator.score(). Else, we must first get the
        predictions to pass them to ad-hoc scorer.
    X : array, shape (n_samples, nd_features, n_slices)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    y : array, shape (n_samples,) | (n_samples, n_targets)
        The target values.

    Returns
    -------
    score : array, shape (n_estimators, n_slices)
        The score for each slice of data.
    """
    # FIXME: The level parallization may be a bit high, and might be memory
    # consuming. Perhaps need to lower it down to the loop across X slices.
    n_iter = X.shape[-1]
    n_est = len(estimators)
    for ii, est in enumerate(estimators):
        for jj in range(X.shape[-1]):
            if scoring is not None:
                _score = scoring(est, X[..., jj], y)
            else:
                _score = est.score(X[..., jj], y)

            # Initialize array of predictions on the first score iteration
            if (ii == 0) & (jj == 0):
                if isinstance(_score, np.ndarray):
                    dtype = _score.dtype
                    shape = np.r_[n_est, n_iter, _score.shape]
                else:
                    dtype = type(_score)
                    shape = [n_est, n_iter]
                score = np.zeros(shape, dtype)
            score[ii, jj, ...] = _score
    return score


def _fix_auc(scoring, y):
    from sklearn.preprocessing import LabelEncoder
    # This fixes sklearn's inability to compute roc_auc when y not in [0, 1]
    # scikit-learn/scikit-learn#6874
    if scoring is not None:
        if (
            hasattr(scoring, '_score_func') and
            hasattr(scoring._score_func, '__name__') and
            scoring._score_func.__name__ == 'roc_auc_score'
        ):
            if np.ndim(y) != 1 or len(set(y)) != 2:
                raise ValueError('roc_auc scoring can only be computed for '
                                 'two-class problems.')
            y = LabelEncoder().fit_transform(y)
    return y
