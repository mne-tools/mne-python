"""Base class copy from sklearn.base."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt
import numbers
from functools import partial
from inspect import Parameter, signature

import numpy as np
from sklearn import model_selection as models
from sklearn.base import (  # noqa: F401
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils import indexable
from sklearn.utils.validation import check_is_fitted

from ..parallel import parallel_func
from ..utils import (
    _check_option,
    _pl,
    _validate_type,
    logger,
    pinv,
    verbose,
    warn,
)
from ._fixes import validate_data
from ._ged import (
    _handle_restr_mat,
    _is_cov_pos_semidef,
    _is_cov_symm,
    _smart_ajd,
    _smart_ged,
)
from ._mod_ged import _no_op_mod
from .transformer import MNETransformerMixin, Vectorizer


class _GEDTransformer(MNETransformerMixin, BaseEstimator):
    """M/EEG signal decomposition using the generalized eigenvalue decomposition (GED).

    Given two channel covariance matrices S and R, the goal is to find spatial filters
    that maximise contrast between S and R.

    Parameters
    ----------
    n_components : int | None
        The number of spatial filters to decompose M/EEG signals.
        If None, all of the components will be used for transformation.
        Defaults to None.
    cov_callable : callable
        Function used to estimate covariances and reference matrix (C_ref) from the
        data. The only required arguments should be 'X' and optionally 'y'. The function
        should return covs, C_ref, info, rank and additional kwargs passed further
        to mod_ged_callable. C_ref, info, rank can be None and kwargs can be empty dict.
    mod_ged_callable : callable | None
        Function used to modify (e.g. sort or normalize) generalized
        eigenvalues and eigenvectors. It should accept as arguments evals, evecs
        and also covs and optional kwargs returned by cov_callable. It should return
        sorted and/or modified evals and evecs and the list of indices according
        to which the first two were sorted. If None, evals and evecs will be
        ordered according to :func:`~scipy.linalg.eigh` default. Defaults to None.
    dec_type : "single" | "multi"
        When "single" and cov_callable returns > 2 covariances,
        approximate joint diagonalization based on Pham's algorithm
        will be used instead of GED.
        When 'multi', GED is performed separately for each class, i.e. each covariance
        (except the last) returned by cov_callable is decomposed with the last
        covariance. In this case, number of covariances should be number of classes + 1.
        Defaults to "single".
    restr_type : "restricting" | "whitening" | None
        Restricting transformation for covariance matrices before performing GED.
        If "restricting" only restriction to the principal subspace of the C_ref
        will be performed.
        If "whitening", covariance matrices will be additionally rescaled according
        to the whitening for the C_ref.
        If None, no restriction will be applied. Defaults to None.
    R_func : callable | None
        If provided, GED will be performed on (S, R_func([S,R])). When dec_type is
        "single", R_func applicable only if two covariances returned by cov_callable.
        If None, GED is performed on (S, R). Defaults to None.

    Attributes
    ----------
    evals_ : ndarray, shape (n_channels)
        If fit, generalized eigenvalues used to decompose S and R, else None.
    filters_ :  ndarray, shape (n_channels or less, n_channels)
        If fit, spatial filters (unmixing matrix) used to decompose the data,
        else None.
    patterns_ : ndarray, shape (n_channels or less, n_channels)
        If fit, spatial patterns (mixing matrix) used to restore M/EEG signals,
        else None.

    See Also
    --------
    CSP
    SPoC
    SSD

    Notes
    -----
    .. versionadded:: 1.11
    """

    def __init__(
        self,
        cov_callable=None,
        n_components=None,
        mod_ged_callable=None,
        dec_type="single",
        restr_type=None,
        R_func=None,
    ):
        self.n_components = n_components
        self.cov_callable = cov_callable
        self.mod_ged_callable = mod_ged_callable
        self.dec_type = dec_type
        self.restr_type = restr_type
        self.R_func = R_func

    _is_base_ged = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._is_base_ged = False

    def fit(self, X, y=None):
        """..."""
        # Let the inheriting transformers check data by themselves
        if self._is_base_ged:
            X, y = self._check_data(
                X,
                y=y,
                fit=True,
                return_y=True,
            )
        self._validate_ged_params()
        covs, C_ref, info, rank, kwargs = self.cov_callable(X, y)
        covs = np.stack(covs)
        self._validate_covariances(covs)
        if C_ref is not None:
            self._validate_covariances([C_ref])
        mod_ged_callable = (
            self.mod_ged_callable if self.mod_ged_callable is not None else _no_op_mod
        )
        restr_mat = _handle_restr_mat(C_ref, self.restr_type, info, rank)

        if self.dec_type == "single":
            if len(covs) > 2:
                weights = kwargs.get("sample_weights", None)
                evecs = _smart_ajd(covs, restr_mat, weights=weights)
                evals = None
            else:
                S = covs[0]
                R = covs[1]
                evals, evecs = _smart_ged(S, R, restr_mat, R_func=self.R_func)

            evals, evecs, self.sorter_ = mod_ged_callable(evals, evecs, covs, **kwargs)
            self.evals_ = evals
            self.filters_ = evecs.T
            self.patterns_ = pinv(evecs)

        elif self.dec_type == "multi":
            self.classes_ = np.unique(y)
            R = covs[-1]
            all_evals, all_evecs = list(), list()
            all_patterns, all_sorters = list(), list()
            for i in range(len(self.classes_)):
                S = covs[i]

                evals, evecs = _smart_ged(S, R, restr_mat, R_func=self.R_func)

                evals, evecs, sorter = mod_ged_callable(evals, evecs, covs, **kwargs)
                all_evals.append(evals)
                all_evecs.append(evecs.T)
                all_patterns.append(pinv(evecs))
                all_sorters.append(sorter)
            self.sorter_ = np.array(all_sorters)
            self.evals_ = np.array(all_evals)
            self.filters_ = np.array(all_evecs)
            self.patterns_ = np.array(all_patterns)

        return self

    def transform(self, X):
        """..."""
        check_is_fitted(self, "filters_")
        # Let the inheriting transformers check data by themselves
        if self._is_base_ged:
            X = self._check_data(X)
        if self.dec_type == "single":
            pick_filters = self.filters_[: self.n_components]
        elif self.dec_type == "multi":
            pick_filters = self._subset_multi_components()
        X = pick_filters @ X
        return X

    def _subset_multi_components(self, name="filters"):
        # The shape of stored filters and patterns is
        # is (n_classes, n_evecs, n_chs)
        # Transform and subset into (n_classes*n_components, n_chs)
        if name == "filters":
            return self.filters_[:, : self.n_components, :].reshape(
                -1, self.filters_.shape[2]
            )
        elif name == "patterns":
            return self.patterns_[:, : self.n_components, :].reshape(
                -1, self.patterns_.shape[2]
            )
        return None

    def _validate_required_args(self, func, desired_required_args):
        sig = signature(func)
        actual_required_args = [
            param.name
            for param in sig.parameters.values()
            if param.default is Parameter.empty
        ]
        func_name = func.func.__name__ if isinstance(func, partial) else func.__name__
        if not all(arg in desired_required_args for arg in actual_required_args):
            raise ValueError(
                f"Invalid required arguments for '{func_name}'. "
                f"The only allowed required arguments are {desired_required_args}, "
                f"but got {actual_required_args} instead."
            )

    def _validate_ged_params(self):
        # Naming is GED-specific so that the validation is still executed
        # when child classes run super().fit()

        _validate_type(self.n_components, (int, None), "n_components")
        if self.n_components is not None and self.n_components <= 0:
            raise ValueError(
                "Invalid value for the 'n_components' parameter. "
                "Allowed are positive integers or None, "
                "but got a non-positive integer instead."
            )

        self._validate_required_args(
            self.cov_callable, desired_required_args=["X", "y"]
        )

        _check_option(
            "dec_type",
            self.dec_type,
            ("single", "multi"),
        )

        _check_option(
            "restr_type",
            self.restr_type,
            ("restricting", "whitening", None),
        )

    def _validate_covariances(self, covs):
        error_template = (
            "{matrix} is not {prop}, but required to be for {decomp}. "
            "Check your cov_callable"
        )
        if len(covs) == 1:
            C_ref = covs[0]
            is_C_ref_symm = _is_cov_symm(C_ref)
            if not is_C_ref_symm:
                raise ValueError(
                    error_template.format(
                        matrix="C_ref covariance",
                        prop="symmetric",
                        decomp="decomposition",
                    )
                )
        elif self.dec_type == "single" and len(covs) > 2:
            # make only lenient symmetric check here.
            # positive semidefiniteness/definiteness will be
            # checked inside _smart_ajd
            for ci, cov in enumerate(covs):
                if not _is_cov_symm(cov):
                    raise ValueError(
                        error_template.format(
                            matrix=f"cov[{ci}]",
                            prop="symmetric",
                            decomp="approximate joint diagonalization",
                        )
                    )
        else:
            if len(covs) == 2:
                S_covs = [covs[0]]
                R = covs[1]
            elif self.dec_type == "multi":
                S_covs = covs[:-1]
                R = covs[-1]

            are_all_S_symm = all([_is_cov_symm(S) for S in S_covs])
            if not are_all_S_symm:
                raise ValueError(
                    error_template.format(
                        matrix="S covariance",
                        prop="symmetric",
                        decomp="generalized eigendecomposition",
                    )
                )
            if not _is_cov_symm(R):
                raise ValueError(
                    error_template.format(
                        matrix="R covariance",
                        prop="symmetric",
                        decomp="generalized eigendecomposition",
                    )
                )
            if not _is_cov_pos_semidef(R):
                raise ValueError(
                    error_template.format(
                        matrix="R covariance",
                        prop="positive semi-definite",
                        decomp="generalized eigendecomposition",
                    )
                )

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        # Can be a transformer where S and R covs are not based on y classes.
        tags.target_tags.required = False
        tags.target_tags.one_d_labels = True
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = True
        tags.requires_fit = True
        return tags


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
        If None the model will be
        :class:`sklearn.linear_model.LogisticRegression`.

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

    _model_attr_wrap = (
        "transform",
        "fit_transform",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "_estimator_type",  # remove after sklearn 1.6
        "decision_function",
        "score",
        "classes_",
    )

    def __init__(self, model=None):
        self.model = model

    def __sklearn_tags__(self):
        """Get sklearn tags."""
        tags = super().__sklearn_tags__()
        model = self.model if self.model is not None else LogisticRegression()
        model_tags = model.__sklearn_tags__()
        tags.estimator_type = model_tags.estimator_type
        if tags.estimator_type is not None:
            model_type_tags = getattr(model_tags, f"{tags.estimator_type}_tags")
            setattr(tags, f"{tags.estimator_type}_tags", model_type_tags)
        return tags

    def __getattr__(self, attr):
        """Wrap to model for some attributes."""
        if attr in LinearModel._model_attr_wrap:
            model = self.model_ if "model_" in self.__dict__ else self.model
            if attr == "fit_transform" and hasattr(model, "fit_transform"):
                return self._fit_transform
            else:
                return getattr(model, attr)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def _fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _validate_params(self, X):
        if self.model is not None:
            model = self.model
            if isinstance(model, MetaEstimatorMixin):
                model = model.estimator
            is_predictor = is_regressor(model) or is_classifier(model)
            if not is_predictor:
                raise ValueError(
                    "Linear model should be a supervised predictor "
                    "(classifier or regressor)"
                )

        # For sklearn < 1.6
        try:
            self._check_n_features(X, reset=True)
        except AttributeError:
            pass

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
        self._validate_params(X)
        X, y = validate_data(self, X, y, multi_output=True)

        # fit the Model
        self.model_ = (
            clone(self.model)
            if self.model is not None
            else LogisticRegression(solver="liblinear")
        )
        self.model_.fit(X, y, **fit_params)

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
        if hasattr(self.model_, "coef_"):
            # Standard Linear Model
            filters = self.model_.coef_
        elif hasattr(self.model_, "estimators_"):
            # Linear model with OneVsRestClassifier
            filters = np.vstack([est.coef_ for est in self.model_.estimators_])
        elif hasattr(self.model_, "best_estimator_") and hasattr(
            self.model_.best_estimator_, "coef_"
        ):
            # Linear Model with GridSearchCV
            filters = self.model_.best_estimator_.coef_
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


def _get_inverse_funcs_before_step(estimator, step_name):
    """Get the inverse_transform methods for all steps before a target step."""
    # in case step_name is nested with __
    parts = step_name.split("__")
    inverse_funcs = list()
    current_pipeline = estimator
    for part_name in parts:
        all_names = [name for name, _ in current_pipeline.steps]
        part_idx = all_names.index(part_name)
        # get all preceding steps for the current step
        for prec_name, prec_step in current_pipeline.steps[:part_idx]:
            if hasattr(prec_step, "inverse_transform"):
                inverse_funcs.append(prec_step.inverse_transform)
            else:
                warn(
                    f"Preceding step '{prec_name}' is not invertible "
                    f"and will be skipped."
                )
        current_pipeline = current_pipeline.named_steps[part_name]
    return inverse_funcs


@verbose
def get_coef(
    estimator, attr="filters_", inverse_transform=False, *, step_name=None, verbose=None
):
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
    step_name : str | None
        Name of the sklearn's pipeline step to get the coef from.
        If inverse_transform is True, the inverse transformations
        will be applied using transformers before this step.
        If None, the last step will be used. Defaults to None.

        .. versionadded:: 1.11
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

    if step_name is not None:
        if not hasattr(estimator, "named_steps"):
            raise ValueError("step_name can only be used with a pipeline estimator.")
        try:
            est = est.get_params(deep=True)[step_name]
        except KeyError:
            raise ValueError(f"Step '{step_name}' is not part of the pipeline.")
    else:
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
        if step_name is None:
            inverse_funcs = _get_inverse_funcs(estimator)
        else:
            inverse_funcs = _get_inverse_funcs_before_step(estimator, step_name)

        # The inverse_transform parameter will call this method on any
        # estimator contained in the pipeline, in reverse order.
        for inverse_func in inverse_funcs[::-1]:
            logger.debug(f"  Applying inverse transformation: {inverse_func}.")
            coef = inverse_func(coef)

    if squeeze_first_dim:
        logger.debug("  Squeezing first dimension of coefficients.")
        coef = coef[0]

    # inverse_transform with Vectorizer returns shape (n_channels, n_components).
    # we should transpose to be consistent with how spatial filters
    # store filters and patterns: (n_components, n_channels)
    if inverse_transform and hasattr(estimator, "steps"):
        is_vectorizer = any(
            isinstance(param_value, Vectorizer)
            for param_value in estimator.get_params(deep=True).values()
        )
        if is_vectorizer and coef.ndim == 2:
            coef = coef.T

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
