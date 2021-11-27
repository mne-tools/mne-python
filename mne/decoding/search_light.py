# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from .mixin import TransformerMixin
from .base import BaseEstimator, _check_estimator
from ..fixes import _get_check_scoring
from ..parallel import parallel_func
from ..utils import (_validate_type, array_split_idx, ProgressBar,
                     verbose, fill_doc)


@fill_doc
class SlidingEstimator(BaseEstimator, TransformerMixin):
    """Search Light.

    Fit, predict and score a series of models to each subset of the dataset
    along the last dimension. Each entry in the last dimension is referred
    to as a task.

    Parameters
    ----------
    %(base_estimator)s
    %(scoring)s
    %(n_jobs)s
    %(verbose)s

    Attributes
    ----------
    estimators_ : array-like, shape (n_tasks,)
        List of fitted scikit-learn estimators (one per task).
    """

    def __init__(self, base_estimator, scoring=None, n_jobs=1,
                 verbose=None):  # noqa: D102
        _check_estimator(base_estimator)
        self._estimator_type = getattr(base_estimator, "_estimator_type", None)
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose

        _validate_type(self.n_jobs, 'int', 'n_jobs')

    def __repr__(self):  # noqa: D105
        repr_str = '<' + super(SlidingEstimator, self).__repr__()
        if hasattr(self, 'estimators_'):
            repr_str = repr_str[:-1]
            repr_str += ', fitted with %i estimators' % len(self.estimators_)
        return repr_str + '>'

    @verbose  # to use class value
    def fit(self, X, y, **fit_params):
        """Fit a series of independent estimators to the dataset.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The training input samples. For each data slice, a clone estimator
            is fitted independently. The feature dimension can be
            multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks).
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.
        **fit_params : dict of string -> object
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        self : object
            Return self.
        """
        self._check_Xy(X, y)
        self.estimators_ = list()
        self.fit_params = fit_params
        # For fitting, the parallelization is across estimators.
        parallel, p_func, n_jobs = parallel_func(_sl_fit, self.n_jobs,
                                                 verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        mesg = 'Fitting %s' % (self.__class__.__name__,)
        with ProgressBar(X.shape[-1], mesg=mesg) as pb:
            estimators = parallel(
                p_func(self.base_estimator, split, y, pb.subset(pb_idx),
                       **fit_params)
                for pb_idx, split in array_split_idx(X, n_jobs, axis=-1))

        # Each parallel job can have a different number of training estimators
        # We can't directly concatenate them because of sklearn's Bagging API
        # (see scikit-learn #9720)
        self.estimators_ = np.empty(X.shape[-1], dtype=object)
        idx = 0
        for job_estimators in estimators:
            for est in job_estimators:
                self.estimators_[idx] = est
                idx += 1
        return self

    def fit_transform(self, X, y, **fit_params):
        """Fit and transform a series of independent estimators to the dataset.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The training input samples. For each task, a clone estimator
            is fitted independently. The feature dimension can be
            multidimensional, e.g.::

                X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.
        **fit_params : dict of string -> object
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        y_pred : array, shape (n_samples, n_tasks) | (n_samples, n_tasks, n_targets)
            The predicted values for each estimator.
        """  # noqa: E501
        return self.fit(X, y, **fit_params).transform(X)

    @verbose  # to use the class value
    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        if X.shape[-1] != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        mesg = 'Transforming %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(
            _sl_transform, self.n_jobs, verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        idx, est_splits = zip(*array_split_idx(self.estimators_, n_jobs))
        with ProgressBar(X.shape[-1], mesg=mesg) as pb:
            y_pred = parallel(p_func(est, x, method, pb.subset(pb_idx))
                              for pb_idx, est, x in zip(
                                  idx, est_splits, X_splits))

        y_pred = np.concatenate(y_pred, axis=1)
        return y_pred

    def transform(self, X):
        """Transform each data slice/task with a series of independent estimators.

        The number of tasks in X should match the number of tasks/estimators
        given at fit time.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice/task, the corresponding
            estimator makes a transformation of the data, e.g.
            ``[estimators[ii].transform(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks).

        Returns
        -------
        Xt : array, shape (n_samples, n_estimators)
            The transformed values generated by each estimator.
        """  # noqa: E501
        return self._transform(X, 'transform')

    def predict(self, X):
        """Predict each data slice/task with a series of independent estimators.

        The number of tasks in X should match the number of tasks/estimators
        given at fit time.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice, the corresponding estimator
            makes the sample predictions, e.g.:
            ``[estimators[ii].predict(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks).

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators) | (n_samples, n_tasks, n_targets)
            Predicted values for each estimator/data slice.
        """  # noqa: E501
        return self._transform(X, 'predict')

    def predict_proba(self, X):
        """Predict each data slice with a series of independent estimators.

        The number of tasks in X should match the number of tasks/estimators
        given at fit time.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice, the corresponding estimator
            makes the sample probabilistic predictions, e.g.:
            ``[estimators[ii].predict_proba(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks).

        Returns
        -------
        y_pred : array, shape (n_samples, n_tasks, n_classes)
            Predicted probabilities for each estimator/data slice/task.
        """  # noqa: E501
        return self._transform(X, 'predict_proba')

    def decision_function(self, X):
        """Estimate distances of each data slice to the hyperplanes.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice, the corresponding estimator
            outputs the distance to the hyperplane, e.g.:
            ``[estimators[ii].decision_function(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators).

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_classes * (n_classes-1) // 2)
            Predicted distances for each estimator/data slice.

        Notes
        -----
        This requires base_estimator to have a ``decision_function`` method.
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
        """Score each estimator on each task.

        The number of tasks in X should match the number of tasks/estimators
        given at fit time, i.e. we need
        ``X.shape[-1] == len(self.estimators_)``.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_tasks)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction, e.g.:
            ``[estimators[ii].score(X[..., ii], y) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            X.shape = (n_samples, n_features_1, n_features_2, n_tasks).
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        score : array, shape (n_samples, n_estimators)
            Score for each estimator/task.
        """  # noqa: E501
        check_scoring = _get_check_scoring()

        self._check_Xy(X)
        if X.shape[-1] != len(self.estimators_):
            raise ValueError('The number of estimators does not match '
                             'X.shape[-1]')

        scoring = check_scoring(self.base_estimator, self.scoring)
        y = _fix_auc(scoring, y)

        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        parallel, p_func, n_jobs = parallel_func(_sl_score, self.n_jobs)
        n_jobs = min(n_jobs, X.shape[-1])
        X_splits = np.array_split(X, n_jobs, axis=-1)
        est_splits = np.array_split(self.estimators_, n_jobs)
        score = parallel(p_func(est, scoring, x, y)
                         for (est, x) in zip(est_splits, X_splits))

        score = np.concatenate(score, axis=0)
        return score

    @property
    def classes_(self):
        if not hasattr(self.estimators_[0], 'classes_'):
            raise AttributeError('classes_ attribute available only if '
                                 'base_estimator has it, and estimator %s does'
                                 ' not' % (self.estimators_[0],))
        return self.estimators_[0].classes_


@fill_doc
def _sl_fit(estimator, X, y, pb, **fit_params):
    """Aux. function to fit SlidingEstimator in parallel.

    Fit a clone estimator to each slice of data.

    Parameters
    ----------
    %(base_estimator)s
    X : array, shape (n_samples, nd_features, n_estimators)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_estimators)
    y : array, shape (n_sample, )
        The target values.
    fit_params : dict | None
        Parameters to pass to the fit method of the estimator.

    Returns
    -------
    estimators_ : list of estimators
        The fitted estimators.
    """
    from sklearn.base import clone
    estimators_ = list()
    for ii in range(X.shape[-1]):
        est = clone(estimator)
        est.fit(X[..., ii], y, **fit_params)
        estimators_.append(est)
        pb.update(ii + 1)
    return estimators_


def _sl_transform(estimators, X, method, pb):
    """Aux. function to transform SlidingEstimator in parallel.

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
        pb.update(ii + 1)
    return y_pred


def _sl_init_pred(y_pred, X):
    """Aux. function to SlidingEstimator to initialize y_pred."""
    n_sample, n_tasks = X.shape[0], X.shape[-1]
    y_pred = np.zeros((n_sample, n_tasks) + y_pred.shape[1:], y_pred.dtype)
    return y_pred


def _sl_score(estimators, scoring, X, y):
    """Aux. function to score SlidingEstimator in parallel.

    Predict and score each slice of data.

    Parameters
    ----------
    estimators : list, shape (n_tasks,)
        The fitted estimators.
    X : array, shape (n_samples, nd_features, n_tasks)
        The target data. The feature dimension can be multidimensional e.g.
        X.shape = (n_samples, n_features_1, n_features_2, n_tasks)
    scoring : callable, str or None
        If scoring is None (default), the predictions are internally
        generated by estimator.score(). Else, we must first get the
        predictions to pass them to ad-hoc scorer.
    y : array, shape (n_samples,) | (n_samples, n_targets)
        The target values.

    Returns
    -------
    score : array, shape (n_tasks,)
        The score for each task / slice of data.
    """
    n_tasks = X.shape[-1]
    score = np.zeros(n_tasks)
    for ii, est in enumerate(estimators):
        score[ii] = scoring(est, X[..., ii], y)
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


@fill_doc
class GeneralizingEstimator(SlidingEstimator):
    """Generalization Light.

    Fit a search-light along the last dimension and use them to apply a
    systematic cross-tasks generalization.

    Parameters
    ----------
    %(base_estimator)s
    %(scoring)s
    %(n_jobs)s
    %(verbose)s
    """

    def __repr__(self):  # noqa: D105
        repr_str = super(GeneralizingEstimator, self).__repr__()
        if hasattr(self, 'estimators_'):
            repr_str = repr_str[:-1]
            repr_str += ', fitted with %i estimators>' % len(self.estimators_)
        return repr_str

    @verbose  # use class value
    def _transform(self, X, method):
        """Aux. function to make parallel predictions/transformation."""
        self._check_Xy(X)
        method = _check_method(self.base_estimator, method)
        mesg = 'Transforming %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(
            _gl_transform, self.n_jobs, verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        with ProgressBar(X.shape[-1] * len(self.estimators_), mesg=mesg) as pb:
            y_pred = parallel(
                p_func(self.estimators_, x_split, method, pb.subset(pb_idx))
                for pb_idx, x_split in array_split_idx(
                    X, n_jobs, axis=-1, n_per_split=len(self.estimators_)))

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
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators).

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
            X.shape = (n_samples, n_features_1, n_features_2, n_estimators).

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
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes)
            The predicted values for each estimator.

        Notes
        -----
        This requires ``base_estimator`` to have a ``predict_proba`` method.
        """  # noqa: E501
        return self._transform(X, 'predict_proba')

    def decision_function(self, X):
        """Estimate distances of each data slice to all hyperplanes.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The training input samples. Each estimator outputs the distance to
            its hyperplane, e.g.:
            ``[estimators[ii].decision_function(X[..., ii]) for ii in range(n_estimators)]``.
            The feature dimension can be multidimensional e.g.
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.

        Returns
        -------
        y_pred : array, shape (n_samples, n_estimators, n_slices, n_classes * (n_classes-1) // 2)
            The predicted values for each estimator.

        Notes
        -----
        This requires ``base_estimator`` to have a ``decision_function``
        method.
        """  # noqa: E501
        return self._transform(X, 'decision_function')

    @verbose  # to use class value
    def score(self, X, y):
        """Score each of the estimators on the tested dimensions.

        Parameters
        ----------
        X : array, shape (n_samples, nd_features, n_slices)
            The input samples. For each data slice, the corresponding estimator
            scores the prediction, e.g.:
            ``[estimators[ii].score(X[..., ii], y) for ii in range(n_slices)]``.
            The feature dimension can be multidimensional e.g.
            ``X.shape = (n_samples, n_features_1, n_features_2, n_estimators)``.
        y : array, shape (n_samples,) | (n_samples, n_targets)
            The target values.

        Returns
        -------
        score : array, shape (n_samples, n_estimators, n_slices)
            Score for each estimator / data slice couple.
        """  # noqa: E501
        check_scoring = _get_check_scoring()
        self._check_Xy(X)
        # For predictions/transforms the parallelization is across the data and
        # not across the estimators to avoid memory load.
        mesg = 'Scoring %s' % (self.__class__.__name__,)
        parallel, p_func, n_jobs = parallel_func(_gl_score, self.n_jobs,
                                                 verbose=False)
        n_jobs = min(n_jobs, X.shape[-1])
        scoring = check_scoring(self.base_estimator, self.scoring)
        y = _fix_auc(scoring, y)
        with ProgressBar(X.shape[-1] * len(self.estimators_), mesg=mesg) as pb:
            score = parallel(p_func(self.estimators_, scoring, x, y,
                                    pb.subset(pb_idx))
                             for pb_idx, x in array_split_idx(
                                 X, n_jobs, axis=-1,
                                 n_per_split=len(self.estimators_)))

        score = np.concatenate(score, axis=1)
        return score


def _gl_transform(estimators, X, method, pb):
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
        pb.update((ii + 1) * n_iter)
    return y_pred


def _gl_init_pred(y_pred, X, n_train):
    """Aux. function to GeneralizingEstimator to initialize y_pred."""
    n_sample, n_iter = X.shape[0], X.shape[-1]
    if y_pred.ndim == 3:
        y_pred = np.zeros((n_sample, n_train, n_iter, y_pred.shape[-1]),
                          y_pred.dtype)
    else:
        y_pred = np.zeros((n_sample, n_train, n_iter), y_pred.dtype)
    return y_pred


def _gl_score(estimators, scoring, X, y, pb):
    """Score GeneralizingEstimator in parallel.

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
    # FIXME: The level parallelization may be a bit high, and might be memory
    # consuming. Perhaps need to lower it down to the loop across X slices.
    score_shape = [len(estimators), X.shape[-1]]
    for jj in range(X.shape[-1]):
        for ii, est in enumerate(estimators):
            _score = scoring(est, X[..., jj], y)
            # Initialize array of predictions on the first score iteration
            if (ii == 0) and (jj == 0):
                dtype = type(_score)
                score = np.zeros(score_shape, dtype)
            score[ii, jj, ...] = _score
            pb.update(jj * len(estimators) + ii + 1)
    return score


def _fix_auc(scoring, y):
    from sklearn.preprocessing import LabelEncoder
    # This fixes sklearn's inability to compute roc_auc when y not in [0, 1]
    # scikit-learn/scikit-learn#6874
    if scoring is not None:
        score_func = getattr(scoring, '_score_func', None)
        kwargs = getattr(scoring, '_kwargs', {})
        if (getattr(score_func, '__name__', '') == 'roc_auc_score' and
                kwargs.get('multi_class', 'raise') == 'raise'):
            if np.ndim(y) != 1 or len(set(y)) != 2:
                raise ValueError('roc_auc scoring can only be computed for '
                                 'two-class problems.')
            y = LabelEncoder().fit_transform(y)
    return y
