# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc as abc
from functools import partial

import numpy as np

from .._fiff.meas_info import Info
from ..cov import Covariance
from ..decoding._covs_ged import _xdawn_estimate
from ..decoding._mod_ged import _xdawn_mod
from ..decoding.base import _GEDTransformer
from ..utils import _validate_type


class XdawnTransformer(_GEDTransformer):
    """Implementation of the Xdawn Algorithm compatible with scikit-learn.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the event related responses. Xdawn was
    originally designed for P300 evoked potential by enhancing the target
    response with respect to the non-target response. This implementation is a
    generalization to any type of event related response.

    .. note:: _XdawnTransformer does not correct for epochs overlap. To correct
              overlaps see ``Xdawn``.

    Parameters
    ----------
    n_components : int (default 2)
        The number of components to decompose the signals.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    signal_cov : None | Covariance | array, shape (n_channels, n_channels)
        The signal covariance used for whitening of the data.
        if None, the covariance is estimated from the epochs signal.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    restr_type : "restricting" | "whitening" | None
        Restricting transformation for covariance matrices before performing
        generalized eigendecomposition.
        If "restricting" only restriction to the principal subspace of signal_cov
        will be performed.
        If "whitening", covariance matrices will be additionally rescaled according
        to the whitening for the signal_cov.
        If None, no restriction will be applied. Defaults to None.

        .. versionadded:: 1.10
    info : mne.Info | None
        The mne.Info object with information about the sensors and methods of
        measurement used for covariance estimation and generalized
        eigendecomposition.
        If None, one channel type and no projections will be assumed and if
        rank is dict, it will be sum of ranks per channel type.
        Defaults to None.

        .. versionadded:: 1.10
    %(rank)s
        Defaults to "full".

        .. versionadded:: 1.10


    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The event indices of the classes.
    filters_ : array, shape (n_channels, n_channels)
        The Xdawn components used to decompose the data for each event type.
    patterns_ : array, shape (n_channels, n_channels)
        The Xdawn patterns used to restore the signals for each event type.
    """

    def __init__(
        self,
        n_components=2,
        reg=None,
        signal_cov=None,
        cov_method_params=None,
        restr_type=None,
        info=None,
        rank="full",
    ):
        """Init."""
        self.n_components = n_components
        self.signal_cov = signal_cov
        self.reg = reg
        self.cov_method_params = cov_method_params
        self.restr_type = restr_type
        self.info = info
        self.rank = rank

        cov_callable = partial(
            _xdawn_estimate,
            reg=reg,
            cov_method_params=cov_method_params,
            R=signal_cov,
            info=info,
            rank=rank,
        )
        super().__init__(
            n_components=n_components,
            cov_callable=cov_callable,
            mod_ged_callable=_xdawn_mod,
            dec_type="multi",
            restr_type=restr_type,
        )

    def _validate_params(self, X):
        _validate_type(self.n_components, int, "n_components")

        # reg is validated in _regularized_covariance

        if self.signal_cov is not None:
            if isinstance(self.signal_cov, Covariance):
                self.signal_cov = self.signal_cov.data
            elif not isinstance(self.signal_cov, np.ndarray):
                raise ValueError("signal_cov should be mne.Covariance or np.ndarray")
            if not np.array_equal(self.signal_cov.shape, np.tile(X.shape[1], 2)):
                raise ValueError(
                    "signal_cov data should be of shape (n_channels, n_channels)"
                )
        _validate_type(self.cov_method_params, (abc.Mapping, None), "cov_method_params")
        _validate_type(self.info, (Info, None), "info")

    def fit(self, X, y=None):
        """Fit Xdawn spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_samples)
            The target data.
        y : array, shape (n_epochs,) | None
            The target labels. If None, Xdawn fit on the average evoked.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        from ..preprocessing.xdawn import _fit_xdawn

        X, y = self._check_Xy(X, y)
        self._validate_params(X)
        # Main function
        self.classes_ = np.unique(y)
        self.filters_, self.patterns_, _ = _fit_xdawn(
            X,
            y,
            n_components=self.n_components,
            reg=self.reg,
            signal_cov=self.signal_cov,
            method_params=self.cov_method_params,
        )
        old_filters = self.filters_
        old_patterns = self.patterns_
        super().fit(X, y)

        # Hack for assert_allclose in transform
        self.new_filters_ = self.filters_.copy()
        # Xdawn performs separate GED for each class.
        # filters_ returned by _fit_xdawn are subset per
        # n_components and then appended and are of shape
        # (n_classes*n_components, n_chs).
        # GEDTransformer creates new dimension per class without subsetting
        # for easier analysis and visualisations.
        # So it needs to be performed post-hoc to conform with Xdawn.
        # The shape returned by GED here is (n_classes, n_evecs, n_chs)
        # Need to transform and subset into (n_classes*n_components, n_chs)
        self.filters_ = self.filters_[:, : self.n_components, :].reshape(
            -1, self.filters_.shape[2]
        )
        self.patterns_ = self.patterns_[:, : self.n_components, :].reshape(
            -1, self.patterns_.shape[2]
        )
        np.testing.assert_allclose(old_filters, self.filters_)
        np.testing.assert_allclose(old_patterns, self.patterns_)

        return self

    def transform(self, X):
        """Transform data with spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_samples)
            The target data.

        Returns
        -------
        X : array, shape (n_epochs, n_components * n_classes, n_samples)
            The transformed data.
        """
        X, _ = self._check_Xy(X)
        orig_X = X.copy()

        # Check size
        if self.filters_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X must have {self.filters_.shape[1]} channels, got {X.shape[1]} "
                "instead."
            )

        # Transform
        X = np.dot(self.filters_, X)
        X = X.transpose((1, 0, 2))
        ged_X = super().transform(orig_X)
        np.testing.assert_allclose(X, ged_X)
        return X

    def inverse_transform(self, X):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data, zero out components,
        and inverse transform the data. This procedure will reconstruct
        the signals from which the dynamics described by the excluded
        components is subtracted.

        Parameters
        ----------
        X : array, shape (n_epochs, n_components * n_classes, n_times)
            The transformed data.

        Returns
        -------
        X : array, shape (n_epochs, n_channels * n_classes, n_times)
            The inverse transform data.
        """
        # Check size
        X, _ = self._check_Xy(X)
        n_epochs, n_comp, n_times = X.shape
        if n_comp != (self.n_components * len(self.classes_)):
            raise ValueError(
                f"X must have {self.n_components * len(self.classes_)} components, "
                f"got {n_comp} instead."
            )

        # Transform
        return np.dot(self.patterns_.T, X).transpose(1, 0, 2)

    def _check_Xy(self, X, y=None):
        """Check X and y types and dimensions."""
        # Check data
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(
                "X must be an array of shape (n_epochs, n_channels, n_samples)."
            )
        if y is None:
            y = np.ones(len(X))
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        return X, y
