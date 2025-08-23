# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc as abc
from functools import partial

import numpy as np

from .._fiff.meas_info import Info
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..utils import _check_option, _validate_type, fill_doc, legacy
from ._covs_ged import _csp_estimate, _spoc_estimate
from ._mod_ged import _csp_mod, _spoc_mod
from .base import _GEDTransformer
from .spatial_filter import get_spatial_filter_from_estimator


@fill_doc
class CSP(_GEDTransformer):
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

    restr_type : "restricting" | "whitening" | None
        Restricting transformation for covariance matrices before performing
        generalized eigendecomposition.
        If "restricting" only restriction to the principal subspace of signal_cov
        will be performed.
        If "whitening", covariance matrices will be additionally rescaled according
        to the whitening for the signal_cov.
        If None, no restriction will be applied. Defaults to "restricting".

        .. versionadded:: 1.11
    info : mne.Info | None
        The mne.Info object with information about the sensors and methods of
        measurement used for covariance estimation and generalized
        eigendecomposition.
        If None, one channel type and no projections will be assumed and if
        rank is dict, it will be sum of ranks per channel type.
        Defaults to None.

        .. versionadded:: 1.11
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
    XdawnTransformer, SPoC, SSD

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
        *,
        restr_type="restricting",
        info=None,
        rank=None,
        component_order="mutual_info",
    ):
        # Init default CSP
        self.n_components = n_components
        self.info = info
        self.rank = rank
        self.reg = reg
        self.cov_est = cov_est
        self.transform_into = transform_into
        self.log = log
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.restr_type = restr_type

        cov_callable = partial(
            _csp_estimate,
            reg=reg,
            cov_method_params=cov_method_params,
            cov_est=cov_est,
            info=info,
            rank=rank,
            norm_trace=norm_trace,
        )
        mod_ged_callable = partial(_csp_mod, evecs_order=component_order)
        super().__init__(
            n_components=n_components,
            cov_callable=cov_callable,
            mod_ged_callable=mod_ged_callable,
            restr_type=restr_type,
            R_func=sum,
        )

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        return tags

    def _validate_params(self, *, y):
        _validate_type(self.n_components, int, "n_components")
        if hasattr(self, "cov_est"):
            _validate_type(self.cov_est, str, "cov_est")
            _check_option("cov_est", self.cov_est, ("concat", "epoch"))
        if hasattr(self, "norm_trace"):
            _validate_type(self.norm_trace, bool, "norm_trace")
        _check_option(
            "transform_into", self.transform_into, ["average_power", "csp_space"]
        )
        if self.transform_into == "average_power":
            _validate_type(
                self.log,
                (bool, None),
                "log",
                extra="when transform_into is 'average_power'",
            )
        else:
            _validate_type(
                self.log, None, "log", extra="when transform_into is 'csp_space'"
            )
        _check_option(
            "component_order", self.component_order, ("mutual_info", "alternate")
        )
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "y should be a 1d array with more than two classes, "
                f"but got {n_classes} class from {y}"
            )
        elif n_classes > 2 and self.component_order == "alternate":
            raise ValueError(
                "component_order='alternate' requires two classes, but data contains "
                f"{n_classes} classes; use component_order='mutual_info' instead."
            )
        _validate_type(self.rank, (dict, None, str), "rank")
        _validate_type(self.info, (Info, None), "info")
        _validate_type(self.cov_method_params, (abc.Mapping, None), "cov_method_params")

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
        X, y = self._check_data(X, y=y, fit=True, return_y=True)
        self._validate_params(y=y)

        # Covariance estimation, GED/AJD
        # and evecs/evals sorting happen here
        super().fit(X, y)

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
        X = self._check_data(X)
        X = super().transform(X)
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

    @legacy(alt="get_spatial_filter_from_estimator(clf, info=info).plot_patterns()")
    @fill_doc
    def plot_patterns(
        self,
        info,
        components=None,
        *,
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
        spf = get_spatial_filter_from_estimator(self, info=info)
        return spf.plot_patterns(
            components,
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
            name_format=name_format,
            nrows=nrows,
            ncols=ncols,
            show=show,
        )

    @legacy(alt="get_spatial_filter_from_estimator(clf, info=info).plot_filters()")
    @fill_doc
    def plot_filters(
        self,
        info,
        components=None,
        *,
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
        spf = get_spatial_filter_from_estimator(self, info=info)
        return spf.plot_filters(
            components,
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
            name_format=name_format,
            nrows=nrows,
            ncols=ncols,
            show=show,
        )


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
        *,
        restr_type=None,
        info=None,
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
            restr_type=restr_type,
            info=info,
            rank=rank,
            cov_method_params=cov_method_params,
        )

        cov_callable = partial(
            _spoc_estimate,
            reg=reg,
            cov_method_params=cov_method_params,
            info=info,
            rank=rank,
        )
        super(CSP, self).__init__(
            n_components=n_components,
            cov_callable=cov_callable,
            mod_ged_callable=_spoc_mod,
            restr_type=restr_type,
        )

        # Covariance estimation have to be done on the single epoch level,
        # unlike CSP where covariance estimation can also be achieved through
        # concatenation of all epochs from the same class.
        delattr(self, "cov_est")
        delattr(self, "norm_trace")

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = False
        return tags

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
        X, y = self._check_data(X, y=y, fit=True, return_y=True)
        self._validate_params(y=y)

        super(CSP, self).fit(X, y)

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
