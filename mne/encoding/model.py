"""Base and high-level classes for fitting encoding models."""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.de>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from ..externals.six import string_types
from ..decoding import BaseEstimator


class SubsetEstimator(BaseEstimator):
    """Remove samples with a particular value before fitting.

    Parameters
    ----------
    base_estimator : instance of sklearn-style base_estimator
        The base_estimator to be called after samples are removed.
    remove_value : float, int, nan | callable | None
        The value that will be removed from the dataset. If a callable,
        it should return a boolean mask of length n_samples that takes
        X an array of shape (n_samples, n_features) as input.
    samples_train : array, shape (n_samples,) | None
        An optional array of indices for manually specifying which samples
        to keep during training. Defaults to None.
    samples_pred : array, shape (n_samples,) | None
        An optional array of indices for manually specifying which samples
        to keep during prediction. Defaults to None.
    remove_condition : 'all', 'any'
        Whether only one `remove_value` will trigger removal of a row, or if
        all values per row must be `remove_value` for removal.
        Defaults to 'all'.
    remove_condition_pred : 'same', 'none'
        Whether to remove rows that contain `remove_value` before doing
        prediction. If `same`, will use the strategy used in estimator fitting.
        Defaults is 'same'.
    """
    def __init__(self, base_estimator, samples_train=None, samples_pred=None,
                 remove_value=None, remove_condition='all',
                 remove_condition_pred='same'):
        if all(ii is not None for ii in [remove_value, samples_train]):
            raise ValueError('Supply at least one of remove_value | '
                             'samples_train')
        if remove_condition not in ['all', 'any']:
            raise ValueError('removal condition must be one of "all" or "any"')
        if remove_condition_pred not in ['same', 'none']:
            raise ValueError('prediction removal condition must be one of'
                             ' "same", "none"')

        if samples_train is not None:
            samples_train = np.asarray(samples_train).astype(int)
        if samples_pred is not None:
            samples_pred = np.asarray(samples_pred).astype(int)
        self.samples_train = samples_train
        self.samples_pred = samples_pred
        self.remove_value = remove_value
        self.remove_condition = remove_condition
        self.remove_condition_pred = remove_condition_pred
        self.base_estimator = _check_regressor(base_estimator)

    def fit(self, X, y):
        """Fit an estimator on a subsample of X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Samples will be
            removed prior to fit according to `remove_value` and
            `samples_train`.
        y : array, shape (n_samples, n_targets)
            The output array for the model. Samples will be
            removed prior to fit according to `remove_value` and
            `samples_train`.

        Returns
        -------
        self : an instance of SubsetEstimator
            The instance with modified attributes corresponding to masks.
        """
        X, y = self._mask_data(X, y, ixs=self.samples_train,
                               remove_value=self.remove_value,
                               remove_condition=self.remove_condition,
                               kind='train')
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions on a subsample of X.

        This will call `self.base_estimator.predict`. If you do not wish
        to make predictions on a subsample of X, `remove_condition_pred`
        must be 'none'.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Samples will be
            removed prior to fit according to `samples_pred`.
        """
        if self.remove_condition_pred == 'same':
            remove_value = self.remove_value
            remove_condition = self.remove_condition
        else:
            remove_value = None
            remove_condition = None
        X = self._mask_data(X, ixs=self.samples_pred,
                            remove_value=remove_value,
                            remove_condition=remove_condition, kind='predict')
        return self.base_estimator.predict(X)

    def transform(self, X, y=None):
        """Return a subset of data points.

        Returns
        -------
        data : array, shape (n_masked_samples, n_features)
            A subset of rows from the input array.
        """
        return self._mask_data(X, y)

    def fit_transform(self, X, y=None):
        """Fit an estimator on a subsample of X and y.

        This does the same thing as the `fit` method.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Samples (rows) will be
            removed prior to fit according to `remove_value` and
            `samples_train`.
        y : array, shape (n_samples, n_targets)
            The output array for the model. Samples (rows) will be
            removed prior to fit according to `remove_value` and
            `samples_train`.

        Returns
        -------
        self : an instance of SubsetEstimator
            The instance with modified attributes corresponding to masks.
        """
        return self.fit(X, y)

    def score(self, X, y):
        """Score model predictions on a subsample of X.

        This will call `self.base_estimator.score`. If you do not wish
        to make predictions on a subsample of X, `remove_condition_pred`
        must be 'none'.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Samples (rows) will be
            removed prior to fit according to `samples_pred`.
        """
        if self.remove_condition_pred == 'same':
            remove_value = self.remove_value
            remove_condition = self.remove_condition
        else:
            remove_value = None
            remove_condition = None
        X, y = self._mask_data(
            X, y, ixs=self.samples_pred, remove_value=remove_value,
            remove_condition=remove_condition, kind='predict')
        return self.base_estimator.score(X, y)

    def get_params(self, deep=True):
        """Return base_estimator parameters."""
        params = dict(samples_train=self.samples_train,
                      samples_pred=self.samples_pred,
                      remove_value=self.remove_value,
                      remove_condition=self.remove_condition,
                      remove_condition_pred=self.remove_condition_pred,
                      base_estimator=self.base_estimator)
        return params

    def _mask_data(self, X, y=None, ixs=None, remove_value=None,
                   remove_condition=None, kind='train'):
        """Remove datapoints according to indices or masked values.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Rows will be removed.
        y : array, shape (n_samples, n_targets)
            The output array for the model. Rows will be removed.
        ixs : array, shape (n_samples_to_keep,)
            An optional array of indices to keep before removing
            any samples based on their values.
        remove_value : float, int, nan | callable | None
            The value that will be removed from the dataset. If a callable,
            it should return a boolean mask of length n_samples that takes
            X an array of shape (n_samples, n_features) as input.
        remove_condition : 'all', 'any'
            Whether only one `remove_value` will trigger removal of a row,
            or if all values per row must be `remove_value` for removal.
            Defaults to 'all'.
        kind : 'train', 'predict', 'transform'
            Whether the estimator is being used for training or prediction.
            This affects the name of the attribute created by the masking
            process.

        Returns
        -------
        X : array, shape (n_samples_after_removal, n_features)
            The input X array after removing rows based on ixs, then
            based on the values in X.
        y : array, shape (n_samples_after_removal, n_targets) (optional)
            The output y array after removing rows based on ixs, then
            based on the values in X. Only returned if `y` is passed.
        """

        # First remove indices if we've manually specified them
        if ixs is not None:
            if X.shape[0] < ixs.max():
                raise ValueError('ixs exceed data shape')
            X = X[ixs]
            if y is not None:
                y = y[ixs]

        # Find data points with value we want to remove
        if remove_value is not None:
            if hasattr(remove_value, '__call__'):
                # If a callable, pass X as an input, assume bool output
                mask = remove_value(X)
                if mask.ndim > 1:
                    raise ValueError('Output mask must be of shape'
                                     ' (n_samples,)')
            else:
                if remove_value is np.nan:
                    mask = ~np.isnan(X)
                else:
                    mask = X != remove_value

                # Decide which rows to remove
                if remove_condition == 'all':
                    # Remove rows w/ NO good values
                    mask = mask.sum(axis=1) == 0
                else:
                    # Remove rows w/ AT LEAST 1 bad value
                    mask = mask.sum(axis=1) < mask.shape[-1]

            # Now change mask to rows that we wish to *keep*
            mask = ~mask
            mask_ixs = np.where(mask)[0]
            mask_ixs = np.asarray(mask_ixs).squeeze()  # In case its a matrix
        else:
            mask = np.ones(X.shape[0], dtype=bool)
            mask_ixs = np.arange(X.shape[0])

        # Create a mask attribute
        if kind == 'train':
            self.mask_train_ = mask
        elif kind == 'predict':
            self.mask_predict_ = mask
        elif kind != 'transform':
            # Don't create a mask attribute if we're just calling transform
            raise ValueError('kind must be one of "train" | "predict"')

        # Return masked values
        X = X[mask_ixs]
        if y is not None:
            y = y[mask_ixs]
            return X, y
        return X


def _get_final_est(estimator):
    """Return the final component of a sklearn estimator/pipeline.

    Parameters
    ----------
    estimator : pipeline, or sklearn / MNE estimator
        An estimator chain from which you wish to pull the final estimator.

    Returns
    -------
    estimator : the sklearn-style estimator at the end of the input chain.
    """
    # Define classes where we pull `estimator` manually
    import sklearn
    from distutils.version import LooseVersion
    if LooseVersion(sklearn.__version__) < '0.16':
        raise ValueError('Encoding models require sklearn version >= 0.16')

    attributes = [ii for ii in ['_final_estimator', 'base_estimator']
                  if hasattr(estimator, ii)]
    if len(attributes) > 0:
        # In the event that both are present, `_final_estimator` is prioritized
        estimator = getattr(estimator, attributes[0])
        estimator = _get_final_est(estimator)
    return estimator


def get_coefs(estimator, coef_name='coef_'):
    """Pull coefficients from an estimator object.

    Parameters
    ----------
    estimator : a sklearn estimator
        The estimator from which to pull the coefficients.
    coef_name : string
        The name of the attribute corresponding to the coefficients created
        after fitting. Defaults to 'coef_'

    Returns
    -------
    coefs : array, shape (n_targets, n_coefs)
        The output coefficients.
    """
    estimator = _get_final_est(estimator)
    if not hasattr(estimator, coef_name):
        raise ValueError('Estimator either is not fit or does not use'
                         ' coefficient name: %s' % coef_name)
    coefs = getattr(estimator, coef_name)
    return coefs


def _check_regressor(regressor):
    """Ensure the estimator will work for regression data."""
    from ..decoding.base import _check_estimator
    from sklearn.linear_model import Ridge

    # Define string-based solvers
    _ridge_solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']

    regressor = 'auto' if regressor is None else regressor
    if regressor in _ridge_solvers:
        regressor = Ridge(alpha=0, solver=regressor, fit_intercept=False)
    elif isinstance(regressor, string_types):
        raise ValueError("estimator must be a scikit-learn estimator or one of"
                         " {}".format(_ridge_solvers))
    else:
        _check_estimator(regressor)
    return regressor
