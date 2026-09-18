# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc as abc
from functools import partial

import numpy as np

from .._fiff.meas_info import Info
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..utils import (
    _check_option,
    _validate_type,
    fill_doc_static,
    legacy,
    verbose_static,
)
from ._covs_ged import _csp_estimate, _spoc_estimate
from ._mod_ged import _csp_mod, _spoc_mod
from .base import _GEDTransformer, _read_ged
from .spatial_filter import get_spatial_filter_from_estimator


@fill_doc_static("rank_none")
class CSP(_GEDTransformer):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    This class can be used as a supervised decomposition to estimate spatial
    filters for feature extraction. CSP in the context of EEG was first
    described in :footcite:`KolesEtAl1990`; a comprehensive tutorial on CSP can
    be found in :footcite:`BlankertzEtAl2008`. Multi-class solving is
    implemented from :footcite:`Grosse-WentrupBuss2008`.

    Parameters
    ----------
    n_components : int
        The number of components to decompose M/EEG signals. This number should
        be set by cross-validation.
    reg : float | str | None
        If not None (same as ``'empirical'``, default), allow regularization
        for covariance estimation. If float (between 0 and 1), shrinkage is
        used. For str values, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool
        If ``transform_into`` equals ``'average_power'`` and ``log`` is None or
        True, then apply a log transform to standardize features, else features
        are z-scored. If ``transform_into`` is ``'csp_space'``, ``log`` must be
        None.
    cov_est : 'concat' | 'epoch'
        If ``'concat'``, covariance matrices are estimated on concatenated
        epochs for each class. If ``'epoch'``, covariance matrices are
        estimated on each epoch separately and then averaged over each class.
    transform_into : 'average_power' | 'csp_space'
        If 'average_power' then ``self.transform`` will return the average
        power of each spatial filter. If ``'csp_space'``, ``self.transform``
        will return the data in CSP space.
    norm_trace : bool
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
    rank : None | 'info' | 'full' | dict
        This controls the rank computation that can be read from the
        measurement info or estimated from the data. When a noise covariance
        is used for whitening, this should reflect the rank of that covariance,
        otherwise amplification of noise components can occur in whitening (e.g.,
        often during source localization).

        :data:`python:None`
            The rank will be estimated from the data after proper scaling of
            different channel types.
        ``'info'``
            The rank is inferred from ``info``. If data have been processed
            with Maxwell filtering, the Maxwell filtering header is used.
            Otherwise, the channel counts themselves are used.
            In both cases, the number of projectors is subtracted from
            the (effective) number of channels in the data.
            For example, if Maxwell filtering reduces the rank to 68, with
            two projectors the returned value will be 66.
        ``'full'``
            The rank is assumed to be full, i.e. equal to the
            number of good channels. If a `~mne.Covariance` is passed, this can
            make sense if it has been (possibly improperly) regularized without
            taking into account the true data rank.
        :class:`dict`
            Calculate the rank only for a subset of channel types, and explicitly
            specify the rank for the remaining channel types. This can be
            extremely useful if you already **know** the rank of (part of) your
            data, for instance in case you have calculated it earlier.

            This parameter must be a dictionary whose **keys** correspond to
            channel types in the data (e.g. ``'meg'``, ``'mag'``, ``'grad'``,
            ``'eeg'``), and whose **values** are integers representing the
            respective ranks. For example, ``{'mag': 90, 'eeg': 45}`` will assume
            a rank of ``90`` and ``45`` for magnetometer data and EEG data,
            respectively.

            The ranks for all channel types present in the data, but
            **not** specified in the dictionary will be estimated empirically.
            That is, if you passed a dataset containing magnetometer, gradiometer,
            and EEG data together with the dictionary from the previous example,
            only the gradiometer rank would be determined, while the specified
            magnetometer and EEG ranks would be taken for granted.

        The default is ``None``.

        .. versionadded:: 0.17
    component_order : 'mutual_info' | 'alternate'
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

    _save_fname_type = "csp"

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        return tags

    _required_state_keys = (
        "component_order",
        "cov_est",
        "cov_method_params",
        "info",
        "log",
        "n_components",
        "norm_trace",
        "rank",
        "reg",
        "restr_type",
        "transform_into",
    )

    def _restore_callables(self):
        """Restore CSP-specific callables after loading state."""
        self.cov_callable = partial(
            _csp_estimate,
            reg=self.reg,
            cov_method_params=self.cov_method_params,
            cov_est=self.cov_est,
            info=self.info,
            rank=self.rank,
            norm_trace=self.norm_trace,
        )
        self.mod_ged_callable = partial(_csp_mod, evecs_order=self.component_order)
        self.R_func = sum

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
    @fill_doc_static(
        "info_not_none",
        "ch_type_topomap",
        "sensors_topomap",
        "show_names_topomap",
        "mask_patterns_topomap",
        "mask_params_topomap",
        "mask_label_params_topomap",
        "contours_topomap",
        "outlines_topomap",
        "sphere_topomap_auto",
        "image_interp_topomap",
        "extrapolate_topomap",
        "border_topomap",
        "res_topomap",
        "size_topomap",
        "cmap_topomap",
        "vlim_plot_topomap",
        "cnorm",
        "colorbar_topomap",
        "cbar_fmt_topomap",
        "units_topomap",
        "axes_evoked_plot_topomap",
        "nrows_ncols_topomap",
        "show",
    )
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
        mask_label_params=None,
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
        info : mne.Info
            The :class:`mne.Info` object with information about the
            sensors and methods of measurement.
            Used for fitting. If not available, consider using
            :func:`mne.create_info`.
        components : float | array of float | None
           The patterns to plot. If ``None``, all components will be shown.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the RMS for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        sensors : bool | str
            Whether to add markers for sensor locations. If :class:`str`, should be a
            valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
            Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
            default), black circles will be used.
        show_names : bool | callable
            If ``True``, show channel names next to each sensor marker. If callable,
            channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
            non-masked sensor names will be shown.
        mask : ndarray of bool, shape (n_channels, n_patterns) | None
            Array indicating channel-pattern combinations to highlight with a distinct
            plotting style.
            Array elements set to ``True`` will be plotted
            with the parameters given in ``mask_params``. Defaults to ``None``,
            equivalent to an array of all ``False`` elements.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4)
        mask_label_params : dict | None
            Additional plotting parameters for significant sensor labels.
            Default (None) equals::

                dict(fontsize='medium', fontweight='bold')

            .. versionadded:: 1.13
        contours : int | array-like
            The number of contour lines to draw. If ``0``, no contours will be drawn.
            If a positive integer, that number of contour levels are chosen using the
            matplotlib tick locator (may sometimes be inaccurate, use array for
            accuracy). If array-like, the array values are used as the contour levels.
            The values should be in µV for EEG, fT for magnetometers and fT/m for
            gradiometers. If ``colorbar=True``, the colorbar will have ticks
            corresponding to the contour levels. Default is ``6``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will be
            drawn. If dict, each key refers to a tuple of x and y positions, the values
            in 'mask_pos' will serve as image mask.
            Alternatively, a matplotlib patch object can be passed for advanced
            masking options, either directly or as a function that returns patches
            (required for multi-axis plots). If None, nothing will be drawn.
            Defaults to 'head'.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        image_interp : str
            The image interpolation to be used. Options are ``'cubic'`` (default)
            to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
            ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
            ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
        extrapolate : str
            Options:

            - ``'box'``
                Extrapolate to four points placed to form a square encompassing all
                data points, where each side of the square is three times the range
                of the data in the respective dimension.
            - ``'local'`` (default for MEG sensors)
                Extrapolate only to nearby points (approximately to points closer than
                median inter-electrode distance). This will also set the
                mask to be polygonal based on the convex hull of the sensors.
            - ``'head'`` (default for non-MEG sensors)
                Extrapolate out to the edges of the clipping circle. This will be on
                the head circle when the sensors are contained within the head circle,
                but it can extend beyond the head when sensors are plotted outside
                the head circle.

            .. versionadded:: 1.3
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.

            .. versionadded:: 1.3
        res : int
            The resolution of the topomap image (number of pixels along each side).
        size : float
            Side length of each subplot in inches.
        cmap : str | matplotlib.colors.Colormap | tuple | 'interactive' | None
            Colormap to use. If :class:`tuple`, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging the
            colorbar with left and right mouse button. Left mouse button moves the
            scale up and down and right mouse button adjusts the range. Hitting
            space bar resets the range. Up and down arrows can be used to change
            the colormap. If ``None``, ``'Reds'`` is used for data that is either
            all-positive or all-negative, and ``'RdBu_r'`` is used otherwise.
            ``'interactive'`` is equivalent to ``(None, True)``. Defaults to ``None``.

            .. warning::  Interactive mode works smoothly only for a small amount
                of topomaps. Interactive mode is disabled by default for more than
                2 topomaps.
        vlim : tuple of length 2
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data.
            If both entries are ``None``, the bounds are set at
            ``(min(data), max(data))``.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. Defaults to ``(None, None)``.

            .. versionadded:: 1.3
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.

            .. versionadded:: 1.3
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
        units : str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of ``times`` provided (unless ``times`` is
            ``None``). Default is ``None``.
        name_format : str
            String format for topomap values. Defaults to "CSP%01d".
        nrows, ncols : int | 'auto'
            The number of rows and columns of topographies to plot. If either ``nrows``
            or ``ncols`` is ``'auto'``, the necessary number will be inferred. Defaults
            to ``nrows=1, ncols='auto'``.

            .. versionadded:: 1.3
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """  # noqa: E501
        spf = get_spatial_filter_from_estimator(self, info=info)
        return spf.plot_patterns(
            components,
            ch_type=ch_type,
            scalings=scalings,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            mask_label_params=mask_label_params,
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
    @fill_doc_static(
        "info_not_none",
        "ch_type_topomap",
        "sensors_topomap",
        "show_names_topomap",
        "mask_patterns_topomap",
        "mask_params_topomap",
        "mask_label_params_topomap",
        "contours_topomap",
        "outlines_topomap",
        "sphere_topomap_auto",
        "image_interp_topomap",
        "extrapolate_topomap",
        "border_topomap",
        "res_topomap",
        "size_topomap",
        "cmap_topomap",
        "vlim_plot_topomap_psd",
        "cnorm",
        "colorbar_topomap",
        "cbar_fmt_topomap",
        "units_topomap",
        "axes_evoked_plot_topomap",
        "nrows_ncols_topomap",
        "show",
    )
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
        mask_label_params=None,
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
        info : mne.Info
            The :class:`mne.Info` object with information about the
            sensors and methods of measurement.
            Used for fitting. If not available, consider using
            :func:`mne.create_info`.
        components : float | array of float | None
           The patterns to plot. If ``None``, all components will be shown.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the RMS for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        sensors : bool | str
            Whether to add markers for sensor locations. If :class:`str`, should be a
            valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
            Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
            default), black circles will be used.
        show_names : bool | callable
            If ``True``, show channel names next to each sensor marker. If callable,
            channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
            non-masked sensor names will be shown.
        mask : ndarray of bool, shape (n_channels, n_patterns) | None
            Array indicating channel-pattern combinations to highlight with a distinct
            plotting style.
            Array elements set to ``True`` will be plotted
            with the parameters given in ``mask_params``. Defaults to ``None``,
            equivalent to an array of all ``False`` elements.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4)
        mask_label_params : dict | None
            Additional plotting parameters for significant sensor labels.
            Default (None) equals::

                dict(fontsize='medium', fontweight='bold')

            .. versionadded:: 1.13
        contours : int | array-like
            The number of contour lines to draw. If ``0``, no contours will be drawn.
            If a positive integer, that number of contour levels are chosen using the
            matplotlib tick locator (may sometimes be inaccurate, use array for
            accuracy). If array-like, the array values are used as the contour levels.
            The values should be in µV for EEG, fT for magnetometers and fT/m for
            gradiometers. If ``colorbar=True``, the colorbar will have ticks
            corresponding to the contour levels. Default is ``6``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will be
            drawn. If dict, each key refers to a tuple of x and y positions, the values
            in 'mask_pos' will serve as image mask.
            Alternatively, a matplotlib patch object can be passed for advanced
            masking options, either directly or as a function that returns patches
            (required for multi-axis plots). If None, nothing will be drawn.
            Defaults to 'head'.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        image_interp : str
            The image interpolation to be used. Options are ``'cubic'`` (default)
            to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
            ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
            ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
        extrapolate : str
            Options:

            - ``'box'``
                Extrapolate to four points placed to form a square encompassing all
                data points, where each side of the square is three times the range
                of the data in the respective dimension.
            - ``'local'`` (default for MEG sensors)
                Extrapolate only to nearby points (approximately to points closer than
                median inter-electrode distance). This will also set the
                mask to be polygonal based on the convex hull of the sensors.
            - ``'head'`` (default for non-MEG sensors)
                Extrapolate out to the edges of the clipping circle. This will be on
                the head circle when the sensors are contained within the head circle,
                but it can extend beyond the head when sensors are plotted outside
                the head circle.

            .. versionadded:: 1.3
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.

            .. versionadded:: 1.3
        res : int
            The resolution of the topomap image (number of pixels along each side).
        size : float
            Side length of each subplot in inches.
        cmap : str | matplotlib.colors.Colormap | tuple | 'interactive' | None
            Colormap to use. If :class:`tuple`, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging the
            colorbar with left and right mouse button. Left mouse button moves the
            scale up and down and right mouse button adjusts the range. Hitting
            space bar resets the range. Up and down arrows can be used to change
            the colormap. If ``None``, ``'Reds'`` is used for data that is either
            all-positive or all-negative, and ``'RdBu_r'`` is used otherwise.
            ``'interactive'`` is equivalent to ``(None, True)``. Defaults to ``None``.

            .. warning::  Interactive mode works smoothly only for a small amount
                of topomaps. Interactive mode is disabled by default for more than
                2 topomaps.
        vlim : tuple of length 2 | "joint"
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data. Elements of the :class:`tuple` may also be
            callable functions which take in a :class:`NumPy array <numpy.ndarray>` and
            return a scalar.

            If both entries are ``None``, the bounds are set at
            ± the maximum absolute value
            of the data (yielding a colormap with midpoint at 0), or
            ``(0, max(abs(data)))`` if the (possibly baselined) data are all-positive.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. If ``vlim="joint"``, will compute the colormap
            limits jointly across all topomaps of the same channel type (instead of
            separately for each topomap), using the min/max of the data for that
            channel type. Defaults to ``(None, None)``.

            .. versionadded:: 1.3
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.

            .. versionadded:: 1.3
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
        units : str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of ``times`` provided (unless ``times`` is
            ``None``). Default is ``None``.
        name_format : str
            String format for topomap values. Defaults to "CSP%01d".
        nrows, ncols : int | 'auto'
            The number of rows and columns of topographies to plot. If either ``nrows``
            or ``ncols`` is ``'auto'``, the necessary number will be inferred. Defaults
            to ``nrows=1, ncols='auto'``.

            .. versionadded:: 1.3
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """  # noqa: E501
        spf = get_spatial_filter_from_estimator(self, info=info)
        return spf.plot_filters(
            components,
            ch_type=ch_type,
            scalings=scalings,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            mask_label_params=mask_label_params,
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
    eps : float
        The tolerance for stopping criterion.
    max_iter : int
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


@fill_doc_static("rank_none")
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
    reg : float | str | None
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool
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
    rank : None | 'info' | 'full' | dict
        This controls the rank computation that can be read from the
        measurement info or estimated from the data. When a noise covariance
        is used for whitening, this should reflect the rank of that covariance,
        otherwise amplification of noise components can occur in whitening (e.g.,
        often during source localization).

        :data:`python:None`
            The rank will be estimated from the data after proper scaling of
            different channel types.
        ``'info'``
            The rank is inferred from ``info``. If data have been processed
            with Maxwell filtering, the Maxwell filtering header is used.
            Otherwise, the channel counts themselves are used.
            In both cases, the number of projectors is subtracted from
            the (effective) number of channels in the data.
            For example, if Maxwell filtering reduces the rank to 68, with
            two projectors the returned value will be 66.
        ``'full'``
            The rank is assumed to be full, i.e. equal to the
            number of good channels. If a `~mne.Covariance` is passed, this can
            make sense if it has been (possibly improperly) regularized without
            taking into account the true data rank.
        :class:`dict`
            Calculate the rank only for a subset of channel types, and explicitly
            specify the rank for the remaining channel types. This can be
            extremely useful if you already **know** the rank of (part of) your
            data, for instance in case you have calculated it earlier.

            This parameter must be a dictionary whose **keys** correspond to
            channel types in the data (e.g. ``'meg'``, ``'mag'``, ``'grad'``,
            ``'eeg'``), and whose **values** are integers representing the
            respective ranks. For example, ``{'mag': 90, 'eeg': 45}`` will assume
            a rank of ``90`` and ``45`` for magnetometer data and EEG data,
            respectively.

            The ranks for all channel types present in the data, but
            **not** specified in the dictionary will be estimated empirically.
            That is, if you passed a dataset containing magnetometer, gradiometer,
            and EEG data together with the dictionary from the previous example,
            only the gradiometer rank would be determined, while the specified
            magnetometer and EEG ranks would be taken for granted.

        The default is ``None``.

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

    _save_fname_type = "spoc"

    def __sklearn_tags__(self):
        """Tag the transformer."""
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = False
        return tags

    _required_state_keys = (
        "cov_method_params",
        "info",
        "log",
        "n_components",
        "rank",
        "reg",
        "restr_type",
        "transform_into",
    )

    def _restore_callables(self):
        """Restore SPoC-specific callables after loading state."""
        self.cov_callable = partial(
            _spoc_estimate,
            reg=self.reg,
            cov_method_params=self.cov_method_params,
            info=self.info,
            rank=self.rank,
        )
        self.mod_ged_callable = _spoc_mod
        self.R_func = None

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


@verbose_static()
def read_csp(fname, *, verbose=None):
    """Load a saved :class:`mne.decoding.CSP` object from disk.

    Parameters
    ----------
    fname : path-like
        Path to a CSP file in HDF5 format, which should end with ``.h5`` or
        ``.hdf5``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    csp : instance of :class:`~mne.decoding.CSP`
        The loaded CSP object with all fitted attributes restored.

    See Also
    --------
    mne.decoding.CSP.save

    Notes
    -----
    .. versionadded:: 1.12
    """
    return _read_ged(fname, CSP, verbose=verbose)


@verbose_static()
def read_spoc(fname, *, verbose=None):
    """Load a saved :class:`mne.decoding.SPoC` object from disk.

    Parameters
    ----------
    fname : path-like
        Path to a SPoC file in HDF5 format, which should end with ``.h5`` or
        ``.hdf5``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    spoc : instance of :class:`~mne.decoding.SPoC`
        The loaded SPoC object with all fitted attributes restored.

    See Also
    --------
    mne.decoding.SPoC.save

    Notes
    -----
    .. versionadded:: 1.12
    """
    return _read_ged(fname, SPoC, verbose=verbose)
