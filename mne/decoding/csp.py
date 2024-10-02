# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy as cp

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from .._fiff.meas_info import create_info
from ..cov import _compute_rank_raw_array, _regularized_covariance, _smart_eigh
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..evoked import EvokedArray
from ..utils import (
    _check_option,
    _validate_type,
    _verbose_safe_false,
    fill_doc,
    pinv,
    warn,
)


@fill_doc
class CSP(TransformerMixin, BaseEstimator):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    This class can be used as a supervised decomposition to estimate spatial
    filters for feature extraction. CSP in the context of EEG was first
    described in :footcite:`KolesEtAl1990`; a comprehensive tutorial on CSP can
    be found in :footcite:`BlankertzEtAl2008`. Multi-class solving is
    implemented from :footcite:`Grosse-WentrupBuss2008`.

    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals. This number should
        be set by cross-validation.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow regularization
        for covariance estimation. If float (between 0 and 1), shrinkage is
        used. For str values, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If ``transform_into`` equals ``'average_power'`` and ``log`` is None or
        True, then apply a log transform to standardize features, else features
        are z-scored. If ``transform_into`` is ``'csp_space'``, ``log`` must be
        None.
    cov_est : 'concat' | 'epoch' (default 'concat')
        If ``'concat'``, covariance matrices are estimated on concatenated
        epochs for each class. If ``'epoch'``, covariance matrices are
        estimated on each epoch separately and then averaged over each class.
    transform_into : 'average_power' | 'csp_space' (default 'average_power')
        If 'average_power' then ``self.transform`` will return the average
        power of each spatial filter. If ``'csp_space'``, ``self.transform``
        will return the data in CSP space.
    norm_trace : bool (default False)
        Normalize class covariance by its trace. Trace normalization is a step
        of the original CSP algorithm :footcite:`KolesEtAl1990` to eliminate
        magnitude variations in the EEG between individuals. It is not applied
        in more recent work :footcite:`BlankertzEtAl2008`,
        :footcite:`Grosse-WentrupBuss2008` and can have a negative impact on
        pattern order.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    %(rank_none)s

        .. versionadded:: 0.17
    component_order : 'mutual_info' | 'alternate' (default 'mutual_info')
        If ``'mutual_info'`` order components by decreasing mutual information
        (in the two-class case this uses a simplification which orders
        components by decreasing absolute deviation of the eigenvalues from 0.5
        :footcite:`BarachantEtAl2010`). For the two-class case, ``'alternate'``
        orders components by starting with the largest eigenvalue, followed by
        the smallest, the second-to-largest, the second-to-smallest, and so on
        :footcite:`BlankertzEtAl2008`.

        .. versionadded:: 0.21

    Attributes
    ----------
    filters_ :  ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_components,)
        If fit, the std squared power for each component.

    See Also
    --------
    mne.preprocessing.Xdawn, SPoC

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_components=4,
        reg=None,
        log=None,
        cov_est="concat",
        transform_into="average_power",
        norm_trace=False,
        cov_method_params=None,
        rank=None,
        component_order="mutual_info",
    ):
        # Init default CSP
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an integer.")
        self.n_components = n_components
        self.rank = rank
        self.reg = reg

        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        # Init default transform_into
        self.transform_into = _check_option(
            "transform_into", transform_into, ["average_power", "csp_space"]
        )

        # Init default log
        if transform_into == "average_power":
            if log is not None and not isinstance(log, bool):
                raise ValueError(
                    'log must be a boolean if transform_into == "average_power".'
                )
        else:
            if log is not None:
                raise ValueError('log must be a None if transform_into == "csp_space".')
        self.log = log

        _validate_type(norm_trace, bool, "norm_trace")
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = _check_option(
            "component_order", component_order, ("mutual_info", "alternate")
        )

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be of type ndarray (got {type(X)}).")
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError("X and y must have the same length.")
        if X.ndim < 3:
            raise ValueError("X must have at least 3 dimensions.")

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        self._check_Xy(X, y)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        if n_classes > 2 and self.component_order == "alternate":
            raise ValueError(
                "component_order='alternate' requires two classes, but data contains "
                f"{n_classes} classes; use component_order='mutual_info' instead."
            )

        # Convert rank to one that will run
        _validate_type(self.rank, (dict, None, str), "rank")

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs, sample_weights)
        ix = self._order_components(
            covs, sample_weights, eigen_vectors, eigen_values, self.component_order
        )

        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = pinv(eigen_vectors)

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
        X = (X**2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the CSP filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_components)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_components, n_times).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be of type ndarray (got {type(X)}).")
        if self.filters_ is None:
            raise RuntimeError(
                "No filters available. Please first fit CSP decomposition."
            )

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == "average_power":
            X = (X**2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X

    def inverse_transform(self, X):
        """Project CSP features back to sensor space.

        Parameters
        ----------
        X : array, shape (n_epochs, n_components)
            The data in CSP power space.

        Returns
        -------
        X : ndarray
            The data in sensor space and shape (n_epochs, n_channels, n_components).
        """
        if self.transform_into != "average_power":
            raise NotImplementedError(
                "Can only inverse transform CSP features when transform_into is "
                "'average_power'."
            )
        if not (X.ndim == 2 and X.shape[1] == self.n_components):
            raise ValueError(
                f"X must be 2D with X[1]={self.n_components}, got {X.shape=}"
            )
        return X[:, np.newaxis, :] * self.patterns_[: self.n_components].T

    def fit_transform(self, X, y=None, **fit_params):
        """Fit CSP to data, then transform it.

        Fits transformer to ``X`` and ``y`` with optional parameters ``fit_params``, and
        returns a transformed version of ``X``.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        **fit_params : dict
            Additional fitting parameters passed to the :meth:`mne.decoding.CSP.fit`
            method. Not used for this class.

        Returns
        -------
        X_csp : array, shape (n_epochs, n_components[, n_times])
            If ``self.transform_into == 'average_power'`` then returns the power of CSP
            features averaged over time and shape is ``(n_epochs, n_components)``. If
            ``self.transform_into == 'csp_space'`` then returns the data in CSP space
            and shape is ``(n_epochs, n_components, n_times)``.
        """
        # use parent TransformerMixin method but with custom docstring
        return super().fit_transform(X, y=y, **fit_params)

    @fill_doc
    def plot_patterns(
        self,
        info,
        components=None,
        *,
        average=None,
        ch_type=None,
        scalings=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap="RdBu_r",
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%3.1f",
        units=None,
        axes=None,
        name_format="CSP%01d",
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic patterns of components.

        The patterns explain how the measured data was generated from the
        neural sources (a.k.a. the forward model).

        Parameters
        ----------
        %(info_not_none)s Used for fitting. If not available, consider using
            :func:`mne.create_info`.
        components : float | array of float | None
           The patterns to plot. If ``None``, all components will be shown.
        %(average_plot_evoked_topomap)s
        %(ch_type_topomap)s
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_patterns_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s

            .. versionadded:: 1.3
        %(border_topomap)s

            .. versionadded:: 1.3
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap)s

            .. versionadded:: 1.3
        %(cnorm)s

            .. versionadded:: 1.3
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap)s
        %(axes_evoked_plot_topomap)s
        name_format : str
            String format for topomap values. Defaults to "CSP%%01d".
        %(nrows_ncols_topomap)s

            .. versionadded:: 1.3
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        if units is None:
            units = "AU"
        if components is None:
            components = np.arange(self.n_components)

        if average is not None:
            warn("`average` is deprecated and will be removed in 1.10.", FutureWarning)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        with info._unlock():
            info["sfreq"] = 1.0
        # create an evoked
        patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        # the call plot_topomap
        fig = patterns.plot_topomap(
            times=components,
            average=average,
            ch_type=ch_type,
            scalings=scalings,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            units=units,
            axes=axes,
            time_format=name_format,
            nrows=nrows,
            ncols=ncols,
            show=show,
        )
        return fig

    @fill_doc
    def plot_filters(
        self,
        info,
        components=None,
        *,
        average=None,
        ch_type=None,
        scalings=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap="RdBu_r",
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%3.1f",
        units=None,
        axes=None,
        name_format="CSP%01d",
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic filters of components.

        The filters are used to extract discriminant neural sources from
        the measured data (a.k.a. the backward model).

        Parameters
        ----------
        %(info_not_none)s Used for fitting. If not available, consider using
            :func:`mne.create_info`.
        components : float | array of float | None
           The patterns to plot. If ``None``, all components will be shown.
        %(average_plot_evoked_topomap)s
        %(ch_type_topomap)s
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_patterns_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s

            .. versionadded:: 1.3
        %(border_topomap)s

            .. versionadded:: 1.3
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap_psd)s

            .. versionadded:: 1.3
        %(cnorm)s

            .. versionadded:: 1.3
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap)s
        %(axes_evoked_plot_topomap)s
        name_format : str
            String format for topomap values. Defaults to "CSP%%01d".
        %(nrows_ncols_topomap)s

            .. versionadded:: 1.3
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        if units is None:
            units = "AU"
        if components is None:
            components = np.arange(self.n_components)

        if average is not None:
            warn("`average` is deprecated and will be removed in 1.10.", FutureWarning)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        with info._unlock():
            info["sfreq"] = 1.0
        # create an evoked
        filters = EvokedArray(self.filters_.T, info, tmin=0)
        # the call plot_topomap
        fig = filters.plot_topomap(
            times=components,
            average=average,
            ch_type=ch_type,
            scalings=scalings,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            units=units,
            axes=axes,
            time_format=name_format,
            nrows=nrows,
            ncols=ncols,
            show=show,
        )
        return fig

    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        if self.cov_est == "concat":
            cov_estimator = self._concat_cov
        elif self.cov_est == "epoch":
            cov_estimator = self._epoch_cov

        # Someday we could allow the user to pass this, then we wouldn't need to convert
        # but in the meantime they can use a pipeline with a scaler
        self._info = create_info(n_channels, 1000.0, "mag")
        if isinstance(self.rank, dict):
            self._rank = {"mag": sum(self.rank.values())}
        else:
            self._rank = _compute_rank_raw_array(
                X.transpose(1, 0, 2).reshape(X.shape[1], -1),
                self._info,
                rank=self.rank,
                scalings=None,
                log_ch_type="data",
            )

        covs = []
        sample_weights = []
        for ci, this_class in enumerate(self._classes):
            cov, weight = cov_estimator(
                X[y == this_class],
                cov_kind=f"class={this_class}",
                log_rank=ci == 0,
            )

            if self.norm_trace:
                cov /= np.trace(cov)

            covs.append(cov)
            sample_weights.append(weight)

        return np.stack(covs), np.array(sample_weights)

    def _concat_cov(self, x_class, *, cov_kind, log_rank):
        """Concatenate epochs before computing the covariance."""
        _, n_channels, _ = x_class.shape

        x_class = x_class.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = _regularized_covariance(
            x_class,
            reg=self.reg,
            method_params=self.cov_method_params,
            rank=self._rank,
            info=self._info,
            cov_kind=cov_kind,
            log_rank=log_rank,
            log_ch_type="data",
        )
        weight = x_class.shape[0]

        return cov, weight

    def _epoch_cov(self, x_class, *, cov_kind, log_rank):
        """Mean of per-epoch covariances."""
        cov = sum(
            _regularized_covariance(
                this_X,
                reg=self.reg,
                method_params=self.cov_method_params,
                rank=self._rank,
                info=self._info,
                cov_kind=cov_kind,
                log_rank=log_rank and ii == 0,
                log_ch_type="data",
            )
            for ii, this_X in enumerate(x_class)
        )
        cov /= len(x_class)
        weight = len(x_class)

        return cov, weight

    def _decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        n_channels = covs[0].shape[0]
        assert self._rank is not None  # should happen in _compute_covariance_matrices
        _, sub_vec, mask = _smart_eigh(
            covs.mean(0),
            self._info,
            self._rank,
            proj_subspace=True,
            do_compute_rank=False,
            log_ch_type="data",
            verbose=_verbose_safe_false(),
        )
        sub_vec = sub_vec[mask]
        covs = np.array([sub_vec @ cov @ sub_vec.T for cov in covs], float)
        assert covs[0].shape == (mask.sum(),) * 2
        if n_classes == 2:
            eigen_values, eigen_vectors = eigh(covs[0], covs.sum(0))
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)
            eigen_vectors = self._normalize_eigenvectors(
                eigen_vectors.T, covs, sample_weights
            )
            eigen_values = None
        # project back
        eigen_vectors = sub_vec.T @ eigen_vectors
        assert eigen_vectors.shape == (n_channels, mask.sum())
        return eigen_vectors, eigen_values

    def _compute_mutual_info(self, covs, sample_weights, eigen_vectors):
        class_probas = sample_weights / sample_weights.sum()

        mutual_info = []
        for jj in range(eigen_vectors.shape[1]):
            aa, bb = 0, 0
            for cov, prob in zip(covs, class_probas):
                tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov), eigen_vectors[:, jj])
                aa += prob * np.log(np.sqrt(tmp))
                bb += prob * (tmp**2 - 1)
            mi = -(aa + (3.0 / 16) * (bb**2))
            mutual_info.append(mi)

        return mutual_info

    def _normalize_eigenvectors(self, eigen_vectors, covs, sample_weights):
        # Here we apply an euclidean mean. See pyRiemann for other metrics
        mean_cov = np.average(covs, axis=0, weights=sample_weights)

        for ii in range(eigen_vectors.shape[1]):
            tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov), eigen_vectors[:, ii])
            eigen_vectors[:, ii] /= np.sqrt(tmp)
        return eigen_vectors

    def _order_components(
        self, covs, sample_weights, eigen_vectors, eigen_values, component_order
    ):
        n_classes = len(self._classes)
        if component_order == "mutual_info" and n_classes > 2:
            mutual_info = self._compute_mutual_info(covs, sample_weights, eigen_vectors)
            ix = np.argsort(mutual_info)[::-1]
        elif component_order == "mutual_info" and n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        elif component_order == "alternate" and n_classes == 2:
            i = np.argsort(eigen_values)
            ix = np.empty_like(i)
            ix[1::2] = i[: len(i) // 2]
            ix[0::2] = i[len(i) // 2 :][::-1]
        return ix


def _ajd_pham(X, eps=1e-6, max_iter=15):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    eps : float, default 1e-6
        The tolerance for stopping criterion.
    max_iter : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(max_iter):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.0j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp**2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order="F")
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order="F")
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D


@fill_doc
class SPoC(CSP):
    """Implementation of the SPoC spatial filtering.

    Source Power Comodulation (SPoC) :footcite:`DahneEtAl2014` allows to
    extract spatial filters and
    patterns by using a target (continuous) variable in the decomposition
    process in order to give preference to components whose power correlates
    with the target variable.

    SPoC can be seen as an extension of the CSP driven by a continuous
    variable rather than a discrete variable. Typical applications include
    extraction of motor patterns using EMG power or audio patterns using sound
    envelope.

    Parameters
    ----------
    n_components : int
        The number of components to decompose M/EEG signals.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If transform_into == 'average_power' and log is None or True, then
        applies a log transform to standardize the features, else the features
        are z-scored. If transform_into == 'csp_space', then log must be None.
    transform_into : {'average_power', 'csp_space'}
        If 'average_power' then self.transform will return the average power of
        each spatial filter. If 'csp_space' self.transform will return the data
        in CSP space. Defaults to 'average_power'.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    %(rank_none)s

        .. versionadded:: 0.17

    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_channels)
        If fit, the SPoC spatial filters, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the SPoC spatial patterns, else None.
    mean_ : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_components,)
        If fit, the std squared power for each component.

    See Also
    --------
    mne.preprocessing.Xdawn, CSP

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_components=4,
        reg=None,
        log=None,
        transform_into="average_power",
        cov_method_params=None,
        rank=None,
    ):
        """Init of SPoC."""
        super().__init__(
            n_components=n_components,
            reg=reg,
            log=log,
            cov_est="epoch",
            norm_trace=False,
            transform_into=transform_into,
            rank=rank,
            cov_method_params=cov_method_params,
        )
        # Covariance estimation have to be done on the single epoch level,
        # unlike CSP where covariance estimation can also be achieved through
        # concatenation of all epochs from the same class.
        delattr(self, "cov_est")
        delattr(self, "norm_trace")

    def fit(self, X, y):
        """Estimate the SPoC decomposition on epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the SPoC.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of SPoC
            Returns the modified instance.
        """
        self._check_Xy(X, y)

        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")

        # The following code is directly copied from pyRiemann

        # Normalize target variable
        target = y.astype(np.float64)
        target -= target.mean()
        target /= target.std()

        n_epochs, n_channels = X.shape[:2]

        # Estimate single trial covariance
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X):
            covs[ii] = _regularized_covariance(
                epoch,
                reg=self.reg,
                method_params=self.cov_method_params,
                rank=self.rank,
                log_ch_type="data",
                log_rank=ii == 0,
            )

        C = covs.mean(0)
        Cz = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)

        # solve eigenvalue decomposition
        evals, evecs = eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels

        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X**2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the SPoC filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_components)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_components, n_times).
        """
        return super().transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit SPoC to data, then transform it.

        Fits transformer to ``X`` and ``y`` with optional parameters ``fit_params``, and
        returns a transformed version of ``X``.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the SPoC.
        y : array, shape (n_epochs,)
            The class for each epoch.
        **fit_params : dict
            Additional fitting parameters passed to the :meth:`mne.decoding.CSP.fit`
            method. Not used for this class.

        Returns
        -------
        X : array, shape (n_epochs, n_components[, n_times])
            If ``self.transform_into == 'average_power'`` then returns the power of CSP
            features averaged over time and shape is ``(n_epochs, n_components)``. If
            ``self.transform_into == 'csp_space'`` then returns the data in CSP space
            and shape is ``(n_epochs, n_components, n_times)``.
        """
        # use parent TransformerMixin method but with custom docstring
        return super().fit_transform(X, y=y, **fit_params)
