"""Base class copy from sklearn.base."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt
import numbers

import numpy as np
from sklearn import model_selection as models
from sklearn.base import (  # noqa: F401
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    is_classifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils import check_array, check_X_y, indexable

from ..parallel import parallel_func
from ..utils import _pl, logger, verbose, warn


class LinearModel(MetaEstimatorMixin, BaseEstimator):
    """Compute and store patterns from linear models.

    The linear model coefficients (filters) are used to extract discriminant
    neural sources from the measured data. This class computes the
    corresponding patterns of these linear filters to make them more
    interpretable :footcite:`HaufeEtAl2014`.

    Parameters
    ----------
    model : object | None
        A linear model from scikit-learn with a fit method
        that updates a ``coef_`` attribute.
        If None the model will be LogisticRegression.

    Attributes
    ----------
    filters_ : ndarray, shape ([n_targets], n_features)
        If fit, the filters used to decompose the data.
    patterns_ : ndarray, shape ([n_targets], n_features)
        If fit, the patterns used to restore M/EEG signals.

    See Also
    --------
    CSP
    mne.preprocessing.ICA
    mne.preprocessing.Xdawn

    Notes
    -----
    .. versionadded:: 0.10

    References
    ----------
    .. footbibliography::
    """

    # TODO: Properly refactor this using
    # https://github.com/scikit-learn/scikit-learn/issues/30237#issuecomment-2465572885
    _model_attr_wrap = (
        "transform",
        "predict",
        "predict_proba",
        "_estimator_type",
        "__tags__",
        "decision_function",
        "score",
        "classes_",
    )

    def __init__(self, model=None):
        # TODO: We need to set this to get our tag checking to work properly
        if model is None:
            model = LogisticRegression(solver="liblinear")
        self.model = model

    def __sklearn_tags__(self):
        """Get sklearn tags."""
        from sklearn.utils import get_tags  # added in 1.6

        # fit method below does not allow sparse data via check_data, we could
        # eventually make it smarter if we had to
        tags = get_tags(self.model)
        tags.input_tags.sparse = False
        return tags

    def __getattr__(self, attr):
        """Wrap to model for some attributes."""
        if attr in LinearModel._model_attr_wrap:
            return getattr(self.model, attr)
        elif attr == "fit_transform" and hasattr(self.model, "fit_transform"):
            return super().__getattr__(self, "_fit_transform")
        return super().__getattr__(self, attr)

    def _fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def fit(self, X, y, **fit_params):
        """Estimate the coefficients of the linear model.

        Save the coefficients in the attribute ``filters_`` and
        computes the attribute ``patterns_``.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples, [n_targets])
            The target values.
        **fit_params : dict of string -> object
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        self : instance of LinearModel
            Returns the modified instance.
        """
        if y is not None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        if y is not None:
            y = check_array(y, dtype=None, ensure_2d=False, input_name="y")
            if y.ndim > 2:
                raise ValueError(
                    f"LinearModel only accepts up to 2-dimensional y, got {y.shape} "
                    "instead."
                )

        # fit the Model
        self.model.fit(X, y, **fit_params)
        self.model_ = self.model  # for better sklearn compat

        # Computes patterns using Haufe's trick: A = Cov_X . W . Precision_Y

        inv_Y = 1.0
        X = X - X.mean(0, keepdims=True)
        if y.ndim == 2 and y.shape[1] != 1:
            y = y - y.mean(0, keepdims=True)
            inv_Y = np.linalg.pinv(np.cov(y.T))
        self.patterns_ = np.cov(X.T).dot(self.filters_.T.dot(inv_Y)).T

        return self

    @property
    def filters_(self):
        if hasattr(self.model, "coef_"):
            # Standard Linear Model
            filters = self.model.coef_
        elif hasattr(self.model.best_estimator_, "coef_"):
            # Linear Model with GridSearchCV
            filters = self.model.best_estimator_.coef_
        else:
            raise ValueError("model does not have a `coef_` attribute.")
        if filters.ndim == 2 and filters.shape[0] == 1:
            filters = filters[0]
        return filters


def _set_cv(cv, estimator=None, X=None, y=None):
    """Set the default CV depending on whether clf is classifier/regressor."""
    # Detect whether classification or regression

    if estimator in ["classifier", "regressor"]:
        est_is_classifier = estimator == "classifier"
    else:
        est_is_classifier = is_classifier(estimator)
    # Setup CV
    if isinstance(cv, int | np.int64):
        XFold = StratifiedKFold if est_is_classifier else KFold
        cv = XFold(n_splits=cv)
    elif isinstance(cv, str):
        if not hasattr(models, cv):
            raise ValueError("Unknown cross-validation")
        cv = getattr(models, cv)
        cv = cv()
    cv = check_cv(cv=cv, y=y, classifier=est_is_classifier)

    # Extract train and test set to retrieve them at predict time
    cv_splits = [(train, test) for train, test in cv.split(X=np.zeros_like(y), y=y)]

    if not np.all([len(train) for train, _ in cv_splits]):
        raise ValueError("Some folds do not have any train epochs.")

    return cv, cv_splits


def _check_estimator(estimator, get_params=True):
    """Check whether an object has the methods required by sklearn."""
    valid_methods = ("predict", "transform", "predict_proba", "decision_function")
    if (not hasattr(estimator, "fit")) or (
        not any(hasattr(estimator, method) for method in valid_methods)
    ):
        raise ValueError(
            "estimator must be a scikit-learn transformer or "
            "an estimator with the fit and a predict-like (e.g. "
            "predict_proba) or a transform method."
        )

    if get_params and not hasattr(estimator, "get_params"):
        raise ValueError(
            "estimator must be a scikit-learn transformer or an "
            "estimator with the get_params method that allows "
            "cloning."
        )


def _get_inverse_funcs(estimator, terminal=True):
    """Retrieve the inverse functions of an pipeline or an estimator."""
    inverse_func = list()
    estimators = list()
    if hasattr(estimator, "steps"):
        # if pipeline, retrieve all steps by nesting
        for _, est in estimator.steps:
            inverse_func.extend(_get_inverse_funcs(est, terminal=False))
            estimators.append(est.__class__.__name__)
    elif hasattr(estimator, "inverse_transform"):
        # if not pipeline attempt to retrieve inverse function
        inverse_func.append(estimator.inverse_transform)
        estimators.append(estimator.__class__.__name__)
    else:
        inverse_func.append(False)
        estimators.append("Unknown")

    # If terminal node, check that that the last estimator is a classifier,
    # and remove it from the transformers.
    if terminal:
        last_is_estimator = inverse_func[-1] is False
        logger.debug(f"  Last estimator is an estimator: {last_is_estimator}")
        non_invertible = np.where(
            [inv_func is False for inv_func in inverse_func[:-1]]
        )[0]
        if last_is_estimator and len(non_invertible) == 0:
            # keep all inverse transformation and remove last estimation
            logger.debug("  Removing inverse transformation from inverse list.")
            inverse_func = inverse_func[:-1]
        else:
            if len(non_invertible):
                bad = ", ".join(estimators[ni] for ni in non_invertible)
                warn(
                    f"Cannot inverse transform non-invertible "
                    f"estimator{_pl(non_invertible)}: {bad}."
                )
            inverse_func = list()

    return inverse_func


@verbose
def get_coef(estimator, attr="filters_", inverse_transform=False, *, verbose=None):
    """Retrieve the coefficients of an estimator ending with a Linear Model.

    This is typically useful to retrieve "spatial filters" or "spatial
    patterns" of decoding models :footcite:`HaufeEtAl2014`.

    Parameters
    ----------
    estimator : object | None
        An estimator from scikit-learn.
    attr : str
        The name of the coefficient attribute to retrieve, typically
        ``'filters_'`` (default) or ``'patterns_'``.
    inverse_transform : bool
        If True, returns the coefficients after inverse transforming them with
        the transformer steps of the estimator.
    %(verbose)s

    Returns
    -------
    coef : array
        The coefficients.

    References
    ----------
    .. footbibliography::
    """
    # Get the coefficients of the last estimator in case of nested pipeline
    est = estimator
    logger.debug(f"Getting coefficients from estimator: {est.__class__.__name__}")
    while hasattr(est, "steps"):
        est = est.steps[-1][1]

    squeeze_first_dim = False

    # If SlidingEstimator, loop across estimators
    if hasattr(est, "estimators_"):
        coef = list()
        for ei, this_est in enumerate(est.estimators_):
            if ei == 0:
                logger.debug("  Extracting coefficients from SlidingEstimator.")
            coef.append(get_coef(this_est, attr, inverse_transform))
        coef = np.transpose(coef)
        coef = coef[np.newaxis]  # fake a sample dimension
        squeeze_first_dim = True
    elif not hasattr(est, attr):
        raise ValueError(f"This estimator does not have a {attr} attribute:\n{est}")
    else:
        coef = getattr(est, attr)

    if coef.ndim == 1:
        coef = coef[np.newaxis]
        squeeze_first_dim = True

    # inverse pattern e.g. to get back physical units
    if inverse_transform:
        if not hasattr(estimator, "steps") and not hasattr(est, "estimators_"):
            raise ValueError(
                "inverse_transform can only be applied onto pipeline estimators."
            )
        # The inverse_transform parameter will call this method on any
        # estimator contained in the pipeline, in reverse order.
        for inverse_func in _get_inverse_funcs(estimator)[::-1]:
            logger.debug(f"  Applying inverse transformation: {inverse_func}.")
            coef = inverse_func(coef)

    if squeeze_first_dim:
        logger.debug("  Squeezing first dimension of coefficients.")
        coef = coef[0]

    return coef


@verbose
def cross_val_multiscore(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=None,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    """Evaluate a score by cross-validation.

    Parameters
    ----------
    estimator : instance of sklearn.base.BaseEstimator
        The object to use to fit the data.
        Must implement the 'fit' method.
    X : array-like, shape (n_samples, n_dimensional_features,)
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like, shape (n_samples, n_targets,)
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like, with shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : str, callable | None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        Note that when using an estimator which inherently returns
        multidimensional output - in particular, SlidingEstimator
        or GeneralizingEstimator - you should set the scorer
        there, not here.
    cv : int, cross-validation generator | iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a ``(Stratified)KFold``,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.
    %(n_jobs)s
    %(verbose)s
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or str, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are
          spawned
        - A string, giving an expression as a function of n_jobs,
          as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape (n_splits,) | shape (n_splits, n_scores)
        Array of scores of the estimator for each run of the cross validation.
    """
    # This code is copied from sklearn
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    # Note: this parallelization is implemented using MNE Parallel
    parallel, p_func, n_jobs = parallel_func(
        _fit_and_score, n_jobs, pre_dispatch=pre_dispatch
    )
    position = hasattr(estimator, "position")
    scores = parallel(
        p_func(
            estimator=clone(estimator),
            X=X,
            y=y,
            scorer=scorer,
            train=train,
            test=test,
            fit_params=fit_params,
            verbose=verbose,
            parameters=dict(position=ii % n_jobs) if position else None,
        )
        for ii, (train, test) in enumerate(cv_iter)
    )
    return np.array(scores)[:, 0, ...]  # flatten over joblib output.


# This verbose is necessary to properly set the verbosity level
# during parallelization
@verbose
def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    error_score="raise",
    *,
    verbose=None,
    position=0,
):
    """Fit estimator and compute scores for a given dataset split."""
    #  This code is adapted from sklearn
    from sklearn.model_selection import _validation
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.utils.validation import _num_samples

    # Adjust length of sample weights

    fit_params = fit_params if fit_params is not None else {}
    fit_params = {
        k: _validation._index_param_value(X, v, train) for k, v in fit_params.items()
    }

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = dt.datetime.now()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_duration = dt.datetime.now() - start_time
        score_duration = dt.timedelta(0)
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warn(
                "Classifier fit failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n{e!r}"
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a numeric value. (Hint: if "
                "using 'raise', please make sure that it has been spelled correctly.)"
            )

    else:
        fit_duration = dt.datetime.now() - start_time
        test_score = _score(estimator, X_test, y_test, scorer)
        score_duration = dt.datetime.now() - start_time - fit_duration
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_duration.total_seconds(), score_duration.total_seconds()])
    if return_parameters:
        ret.append(parameters)
    return ret


def _score(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set.

    This code is the same as sklearn.model_selection._validation._score
    but accepts to output arrays instead of floats.
    """
    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, "item"):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    return score
