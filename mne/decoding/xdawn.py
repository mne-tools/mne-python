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
from ..utils import (
    _check_fname,
    _import_h5io_funcs,
    _validate_type,
    fill_doc,
    verbose,
)
from ..utils.check import check_fname


@fill_doc
class XdawnTransformer(_GEDTransformer):
    """Implementation of the Xdawn Algorithm compatible with scikit-learn.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the event related responses. Xdawn was
    originally designed for P300 evoked potential by enhancing the target
    response with respect to the non-target response. This implementation is a
    generalization to any type of event related response.

    .. note:: XdawnTransformer does not correct for epochs overlap. To correct
              overlaps see `mne.preprocessing.Xdawn`.

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

        .. versionadded:: 1.11
    info : mne.Info | None
        The mne.Info object with information about the sensors and methods of
        measurement used for covariance estimation and generalized
        eigendecomposition.
        If None, one channel type and no projections will be assumed and if
        rank is dict, it will be sum of ranks per channel type.
        Defaults to None.

        .. versionadded:: 1.11
    %(rank_full)s

        .. versionadded:: 1.11

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The event indices of the classes.
    filters_ : array, shape (n_channels, n_channels)
        The Xdawn components used to decompose the data for each event type.
    patterns_ : array, shape (n_channels, n_channels)
        The Xdawn patterns used to restore the signals for each event type.

    See Also
    --------
    CSP, SPoC, SSD
    """

    def __init__(
        self,
        n_components=2,
        reg=None,
        signal_cov=None,
        cov_method_params=None,
        *,
        restr_type=None,
        info=None,
        rank="full",
    ):
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

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def __getstate__(self):
        """Prepare state for serialization."""
        state = self.__dict__.copy()
        state.pop("cov_callable", None)
        state.pop("mod_ged_callable", None)
        return state

    _required_state_keys = (
        "n_components",
        "reg",
        "signal_cov",
        "cov_method_params",
        "restr_type",
        "info",
        "rank",
    )

    def __setstate__(self, state):
        """Restore state from serialization."""
        missing = [k for k in self._required_state_keys if k not in state]
        if missing:
            raise ValueError(
                f"State dict is missing required keys: {missing}. "
                "The state may be from an incompatible version of MNE."
            )
        if state["info"] is not None:
            state["info"] = Info(**state["info"])
        self.__dict__.update(state)
        self.cov_callable = partial(
            _xdawn_estimate,
            reg=self.reg,
            cov_method_params=self.cov_method_params,
            R=self.signal_cov,
            info=self.info,
            rank=self.rank,
        )
        self.mod_ged_callable = _xdawn_mod

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save the XdawnTransformer object to disk (in HDF5 format).

        Parameters
        ----------
        fname : path-like
            The file path to save to. Should end with ``'.h5'`` or ``'.hdf5'``.
        %(overwrite)s
        %(verbose)s

        See Also
        --------
        mne.decoding.read_xdawn_transformer
        """
        _, write_hdf5 = _import_h5io_funcs()
        check_fname(fname, "xdawn_transformer", (".h5", ".hdf5"))
        fname = _check_fname(fname, overwrite=overwrite, verbose=verbose)
        write_hdf5(
            fname,
            self.__getstate__(),
            overwrite=overwrite,
            title="mnepython",
            slash="replace",
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
        X, y = self._check_data(X, y=y, fit=True, return_y=True)
        # For test purposes
        if y is None:
            y = np.ones(len(X))
        self._validate_params(X)

        super().fit(X, y)

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
        X = self._check_data(X)
        X = super().transform(X)
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
        X = self._check_data(X, check_n_features=False)
        n_epochs, n_comp, n_times = X.shape
        if n_comp != (self.n_components * len(self.classes_)):
            raise ValueError(
                f"X must have {self.n_components * len(self.classes_)} components, "
                f"got {n_comp} instead."
            )
        pick_patterns = self._subset_multi_components(name="patterns")
        # Transform
        return np.dot(pick_patterns.T, X).transpose(1, 0, 2)


def read_xdawn_transformer(fname):
    """Load a saved :class:`~mne.decoding.XdawnTransformer` object from disk.

    Parameters
    ----------
    fname : path-like
        Path to an XdawnTransformer file in HDF5 format, which should end with
        ``.h5`` or ``.hdf5``.

    Returns
    -------
    xdawn_transformer : instance of :class:`~mne.decoding.XdawnTransformer`
        The loaded XdawnTransformer object with all fitted attributes restored.

    See Also
    --------
    mne.decoding.XdawnTransformer.save
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, "path-like", "fname")
    fname = _check_fname(fname=fname, overwrite="read", must_exist=False)
    state = read_hdf5(fname, title="mnepython", slash="replace")
    xdawn_transformer = object.__new__(XdawnTransformer)
    xdawn_transformer.__setstate__(state)
    return xdawn_transformer
