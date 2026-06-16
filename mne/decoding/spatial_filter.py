# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy as cp

import matplotlib.pyplot as plt
import numpy as np

from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..evoked import EvokedArray
from ..utils import _check_option, fill_doc, verbose
from ..viz.utils import plt_show
from .base import LinearModel, _GEDTransformer, get_coef


def _plot_model(
    model_array,
    info,
    components=None,
    *,
    evk_tmin=None,
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
    name_format=None,
    nrows=1,
    ncols="auto",
    show=True,
):
    if components is None:
        n_comps = model_array.shape[-2]
        components = np.arange(n_comps)
    kwargs = dict(
        # args set here
        times=components,
        average=None,
        proj=False,
        units="AU" if units is None else units,
        time_format=name_format,
        # args passed from the upstream
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
        nrows=nrows,
        ncols=ncols,
        show=show,
    )

    # set sampling frequency to have 1 component per time point

    if evk_tmin is None:
        info = cp.deepcopy(info)
        with info._unlock():
            info["sfreq"] = 1.0
        evk_tmin = 0

    if model_array.ndim == 3:
        n_classes = model_array.shape[0]
        figs = list()
        for class_idx in range(n_classes):
            model_evk = EvokedArray(model_array[class_idx].T, info, tmin=evk_tmin)
            fig = model_evk.plot_topomap(
                axes=axes[class_idx] if axes else None, **kwargs
            )
            figs.append(fig)
        return figs
    else:
        model_evk = EvokedArray(model_array.T, info, tmin=evk_tmin)
        fig = model_evk.plot_topomap(axes=axes, **kwargs)
        return fig


def _plot_scree_per_class(evals, add_cumul_evals, axes):
    component_numbers = np.arange(len(evals))
    cumul_evals = np.cumsum(evals) if add_cumul_evals else None
    # plot individual eigenvalues
    color_line = "cornflowerblue"
    axes.set_xlabel("Component Index", fontsize=18)
    axes.set_ylabel("Eigenvalue", fontsize=18)
    axes.plot(
        component_numbers,
        evals,
        color=color_line,
        marker="o",
        markersize=8,
    )
    axes.tick_params(axis="y", labelsize=16)
    axes.tick_params(axis="x", labelsize=16)

    if add_cumul_evals:
        # plot cumulative eigenvalues
        ax2 = axes.twinx()
        ax2.grid(False)
        color_line = "firebrick"
        ax2.set_ylabel("Cumulative Eigenvalues", fontsize=18)
        ax2.plot(
            component_numbers,
            cumul_evals,
            color=color_line,
            marker="o",
            markersize=6,
        )
        ax2.tick_params(axis="y", labelcolor=color_line, labelsize=16)
        ax2.set_ylim(0)


def _plot_scree(
    evals,
    title="Scree plot",
    add_cumul_evals=True,
    axes=None,
):
    evals_data = evals if evals.ndim == 2 else [evals]
    n_classes = len(evals_data)
    axes = [axes] if isinstance(axes, plt.Axes) else axes
    if axes is not None and n_classes != len(axes):
        raise ValueError(f"Received {len(axes)} axes, but expected {n_classes}")

    orig_axes = axes
    figs = list()
    for class_idx in range(n_classes):
        fig = None
        if orig_axes is None:
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        else:
            ax = axes[class_idx]
        _plot_scree_per_class(evals_data[class_idx], add_cumul_evals, ax)
        if fig is not None:
            fig.suptitle(title, fontsize=22)
            figs.append(fig)

    return figs[0] if len(figs) == 1 else figs


@verbose
def get_spatial_filter_from_estimator(
    estimator,
    info,
    *,
    inverse_transform=False,
    step_name=None,
    get_coefs=("filters_", "patterns_", "evals_"),
    patterns_method=None,
    verbose=None,
):
    """Instantiate a :class:`mne.decoding.SpatialFilter` object.

    Creates object from the fitted generalized eigendecomposition
    transformers or :class:`mne.decoding.LinearModel`.
    This object can be used to visualize spatial filters,
    patterns, and eigenvalues.

    Parameters
    ----------
    estimator : instance of sklearn.base.BaseEstimator
        Sklearn-based estimator or meta-estimator from which to initialize
        spatial filter. Use ``step_name`` to select relevant transformer
        from the pipeline object (works with nested names using ``__`` syntax).
    info : instance of mne.Info
        The measurement info object for plotting topomaps.
    inverse_transform : bool
        If True, returns filters and patterns after inverse transforming them with
        the transformer steps of the estimator. Defaults to False.
    step_name : str | None
        Name of the sklearn's pipeline step to get the coefs from.
        If inverse_transform is True, the inverse transformations
        will be applied using transformers before this step.
        If None, the last step will be used. Defaults to None.
    get_coefs : tuple
        The names of the coefficient attributes to retrieve, can include
        ``'filters_'``, ``'patterns_'`` and ``'evals_'``.
        If step is GEDTransformer, will use all.
        if step is LinearModel will only use ``'filters_'`` and ``'patterns_'``.
        Defaults to (``'filters_'``, ``'patterns_'``, ``'evals_'``).
    patterns_method : str
        The method used to compute the patterns. Can be None, ``'pinv'`` or ``'haufe'``.
        It will be set automatically to ``'pinv'`` if step is GEDTransformer,
        or to ``'haufe'`` if step is LinearModel. Defaults to None.
    %(verbose)s

    Returns
    -------
    sp_filter : instance of mne.decoding.SpatialFilter
        The spatial filter object.

    See Also
    --------
    SpatialFilter, mne.decoding.LinearModel, mne.decoding.CSP,
    mne.decoding.SSD, mne.decoding.XdawnTransformer, mne.decoding.SPoC

    Notes
    -----
    .. versionadded:: 1.11
    """
    for coef in get_coefs:
        if coef not in ("filters_", "patterns_", "evals_"):
            raise ValueError(
                f"'get_coefs' can only include 'filters_', "
                f"'patterns_' and 'evals_', but got {coef}."
            )
    if step_name is not None:
        model = estimator.get_params()[step_name]
    elif hasattr(estimator, "named_steps"):
        model = estimator[-1]
    else:
        model = estimator
    if isinstance(model, LinearModel):
        patterns_method = "haufe"
        get_coefs = ["filters_", "patterns_"]
    elif isinstance(model, _GEDTransformer):
        patterns_method = "pinv"
        get_coefs = ["filters_", "patterns_", "evals_"]

    coefs = {
        coef[:-1]: get_coef(
            estimator,
            coef,
            inverse_transform=False if coef == "evals_" else inverse_transform,
            step_name=step_name,
            verbose=verbose,
        )
        for coef in get_coefs
    }

    sp_filter = SpatialFilter(info, patterns_method=patterns_method, **coefs)
    return sp_filter


class SpatialFilter:
    r"""Container for spatial filter weights (evecs) and patterns.

    .. warning:: For MNE-Python decoding classes, this container should be
        instantiated with `mne.decoding.get_spatial_filter_from_estimator`.
        Direct instantiation with external spatial filters is possible
        at your own risk.

    This object is obtained either by generalized eigendecomposition (GED) algorithms
    such as :class:`mne.decoding.CSP`, :class:`mne.decoding.SPoC`,
    :class:`mne.decoding.SSD`, :class:`mne.decoding.XdawnTransformer` or by
    :class:`mne.decoding.LinearModel`, wrapping linear models like SVM or Logit.
    The object stores the filters that projects sensor data to a reduced component
    space, and the corresponding patterns (obtained by pseudoinverse in GED case or
    Haufe's trick in case of :class:`mne.decoding.LinearModel`). It can also be directly
    initialized using filters from other transformers (e.g. PyRiemann),
    but make sure that the dimensions match.

    Parameters
    ----------
    info : instance of Info
        The measurement info containing channel topography.
    filters : ndarray, shape ((n_classes), n_components, n_channels)
        The spatial filters (transposed eigenvectors of the decomposition).
    evals : ndarray, shape ((n_classes), n_components) | None
        The eigenvalues of the decomposition. Defaults to ``None``.
    patterns : ndarray, shape ((n_classes), n_components, n_channels) | None
        The patterns of the decomposition. If None, they will be computed
        from the filters using pseudoinverse. Defaults to ``None``.
    patterns_method : str
        The method used to compute the patterns. Can be ``'pinv'`` or ``'haufe'``.
        If ``patterns`` is None, it will be set to ``'pinv'``. Defaults to ``'pinv'``.

    Attributes
    ----------
    info : instance of Info
        The measurement info.
    filters : ndarray, shape (n_components, n_channels)
        The spatial filters (unmixing matrix). Applying these filters to the data
        gives the component time series.
    patterns : ndarray, shape (n_components, n_channels)
        The spatial patterns (mixing matrix/forward model).
        These represent the scalp topography of each component.
    evals : ndarray, shape (n_components,)
        The eigenvalues associated with each component.
    patterns_method : str
        The method used to compute the patterns from the filters.

    See Also
    --------
    get_spatial_filter_from_estimator, mne.decoding.LinearModel, mne.decoding.CSP,
    mne.decoding.SSD, mne.decoding.XdawnTransformer, mne.decoding.SPoC

    Notes
    -----
    The spatial filters and patterns are stored with shape
    ``(n_components, n_channels)``.

    Filters and patterns are related by the following equation:

    .. math::
        \mathbf{A} = \mathbf{W}^{-1}

    where :math:`\mathbf{A}` is the matrix of patterns (the mixing matrix) and
    :math:`\mathbf{W}` is the matrix of filters (the unmixing matrix).

    For a detailed discussion on the difference between filters and patterns for GED
    see :footcite:`Cohen2022` and for linear models in
    general see :footcite:`HaufeEtAl2014`.

    .. versionadded:: 1.11

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        info,
        filters,
        *,
        evals=None,
        patterns=None,
        patterns_method="pinv",
    ):
        _check_option(
            "patterns_method",
            patterns_method,
            ("pinv", "haufe"),
        )
        self.info = info
        self.evals = evals
        self.filters = filters
        n_comps, n_chs = self.filters.shape[-2:]
        if patterns is None:
            # XXX Using numpy's pinv here to handle 3D case seamlessly
            # Perhaps mne.linalg.pinv can be improved to handle 3D also
            # Then it could be changed here to be consistent with
            # GEDTransformer
            self.patterns = np.linalg.pinv(filters.T)
            self.patterns_method = "pinv"
        else:
            self.patterns = patterns
            self.patterns_method = patterns_method

        # In case of multi-target classification in LinearModel
        # number of targets can be greater than number of channels.
        if patterns_method != "haufe" and n_comps > n_chs:
            raise ValueError(
                "Number of components can't be greater "
                "than number of channels in filters, "
                "perhaps the provided matrix is transposed?"
            )
        if self.filters.shape != self.patterns.shape:
            raise ValueError(
                f"Shape mismatch between filters and patterns."
                f"Filters are {self.filters.shape},"
                f"while patterns are {self.patterns.shape}"
            )

    @fill_doc
    def plot_filters(
        self,
        components=None,
        tmin=None,
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
        name_format="Filter%01d",
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic maps of model filters.

        Parameters
        ----------
        components : float | array of float | 'auto' | None
            Indices of filters to plot. If "auto", the number of
            ``axes`` determines the amount of filters.
            If None, all filters will be plotted. Defaults to None.
        tmin : float | None
            In case filters are distributed temporally,
            this can be used to align them with times
            and frequency. Use ``epochs.tmin``, for example.
            Defaults to None.
        %(ch_type_topomap)s
        %(scalings_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap_psd)s
        %(cnorm)s
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap_evoked)s
        %(axes_evoked_plot_topomap)s
        name_format : str
            String format for topomap values. Defaults to ``'Filter%%01d'``.
        %(nrows_ncols_topomap)s
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        fig = _plot_model(
            self.filters,
            self.info,
            components=components,
            evk_tmin=tmin,
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
        return fig

    @fill_doc
    def plot_patterns(
        self,
        components=None,
        tmin=None,
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
        name_format="Pattern%01d",
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic maps of model patterns.

        Parameters
        ----------
        components : float | array of float | 'auto' | None
            Indices of patterns to plot. If "auto", the number of
            ``axes`` determines the amount of patterns.
            If None, all patterns will be plotted. Defaults to None.
        tmin : float | None
            In case patterns are distributed temporally,
            this can be used to align them with times
            and frequency. Use ``epochs.tmin``, for example.
            Defaults to None.
        %(ch_type_topomap)s
        %(scalings_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap_psd)s
        %(cnorm)s
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap_evoked)s
        %(axes_evoked_plot_topomap)s
        name_format : str
            String format for topomap values. Defaults to ``'Pattern%%01d'``.
        %(nrows_ncols_topomap)s
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        fig = _plot_model(
            self.patterns,
            self.info,
            components=components,
            evk_tmin=tmin,
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
        return fig

    @fill_doc
    def plot_scree(
        self,
        title="Scree plot",
        add_cumul_evals=False,
        axes=None,
        show=True,
    ):
        """Plot scree for GED eigenvalues.

        Parameters
        ----------
        title : str
            Title for the plot. Defaults to ``'Scree plot'``.
        add_cumul_evals : bool
            Whether to add second line and y-axis for cumulative eigenvalues.
            Defaults to ``True``.
        axes : instance of Axes | None
            The matplotlib axes to plot to. Defaults to ``None``.
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        if self.evals is None:
            raise AttributeError("Can't plot scree if eigenvalues are not provided.")

        fig = _plot_scree(
            self.evals,
            title=title,
            add_cumul_evals=add_cumul_evals,
            axes=axes,
        )
        plt_show(show, block=False)
        return fig
