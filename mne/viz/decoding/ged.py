# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy as cp

import matplotlib.pyplot as plt
import numpy as np

from ...defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ...evoked import EvokedArray
from ...utils import _check_option, fill_doc, pinv


def _plot_model(
    model_array,
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
    name_format=None,
    nrows=1,
    ncols="auto",
    show=True,
):
    if units is None:
        units = "AU"
    if components is None:
        # n_components are rows
        components = np.arange(model_array.shape[0])

    # set sampling frequency to have 1 component per time point
    info = cp.deepcopy(info)
    with info._unlock():
        info["sfreq"] = 1.0
    # create an evoked
    model_evk = EvokedArray(model_array.T, info, tmin=0)
    # the call plot_topomap
    fig = model_evk.plot_topomap(
        times=components,
        average=None,
        ch_type=ch_type,
        scalings=scalings,
        proj=False,
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


def _plot_scree(
    evals,
    title="Scree plot",
    add_cumul_evals=True,
    plt_style="seaborn-v0_8-whitegrid",
    axes=None,
):
    cumul_evals = np.cumsum(evals)
    n_components = len(evals)
    component_numbers = np.arange(n_components)

    with plt.style.context(plt_style):
        if axes is None:
            fig, axes = plt.subplots(figsize=(12, 7), layout="constrained")
        else:
            fig = None

        # plot individual eigenvalues
        color_line = "cornflowerblue"
        axes.set_xlabel("Component Index", fontsize=18)
        axes.set_ylabel("Eigenvalue", color=color_line, fontsize=18)
        axes.set_ylabel("Cumulative Eigenvalues", color=color_line, fontsize=18)
        axes.plot(component_numbers, evals, color=color_line, marker="o", markersize=10)
        axes.tick_params(axis="y", labelcolor=color_line, labelsize=16)

        if add_cumul_evals:
            # plot cumulative eigenvalues
            ax2 = axes.twinx()
            ax2.grid(False)
            color_line = "firebrick"
            ax2.set_ylabel("Cumulative Eigenvalues", color=color_line, fontsize=18)
            ax2.plot(
                component_numbers,
                cumul_evals,
                color=color_line,
                marker="o",
                markersize=6,
            )
            ax2.tick_params(axis="y", labelcolor=color_line, labelsize=16)
            ax2.set_ylim(0)

        if fig:
            fig.suptitle(title, fontsize=22, fontweight="bold")

    return fig


class SpatialFilter:
    r"""Visualization container for spatial filter weights and patterns.

    This object is obtained either by generalized eigendecomposition (GED) algorithms
    such as `mne.decoding.CSP`, `mne.decoding.SPoC`, `mne.decoding.SSD`,
    `mne.decoding.XdawnTransformer` or by `mne.decoding.LinearModel`
    wrapping linear models like SVM or Logit.
    The objects stores the filters that projects sensor data to a reduced component
    space, and the corresponding patterns (obtained by pseudoinverse in GED case or
    Haufe's trickin case of `mne.decoding.LinearModel`). It can also be directly
    initialized using filters from other transformers (e.g. PyRiemann).

    Parameters
    ----------
    info : instance of Info
        The measurement info containing channel topography.
    evecs : ndarray, shape (n_channels, n_components)
        The eigenvectors of the decomposition (transposed filters).
    evals : ndarray, shape (n_components,) | None
        The eigenvalues of the decomposition. Defaults to ``None``.
    patterns : ndarray, shape (n_components, n_channels) | None
        The patterns of the decomposition. If None, they will be computed
        from the eigenvectors using pseudoinverse. Defaults to ``None``.
    patterns_method : str
        The method used to compute the patterns. Can be ``'pinv'`` or ``'haufe'``.
        If `patterns` is None, it will be set to ``'pinv'``. Defaults to ``'pinv'``.

    Attributes
    ----------
    info : instance of Info
        The measurement info.
    filters : ndarray, shape (n_components, n_channels)
        The spatial filters (unmixing matrix). Applying these filters to the data
        gives the component time series.
    patterns : ndarray, shape (n_components, n_channels)
        The spatial patterns (forward model). These represent the scalp
        topography of each component.
    evals : ndarray, shape (n_components,)
        The eigenvalues associated with each component.
    patterns_method : str
        The method used to compute the patterns from the filters.

    Notes
    -----
    The spatial filters and patterns are stored with shape
    ``(n_components, n_channels)``.

    Filters and patterns are related by the following equation:

    .. math::
        \\mathbf{A} = \\mathbf{W}^{-1}

    where :math:`\\mathbf{A}` is the matrix of patterns (the mixing matrix) and
    :math:`\\mathbf{W}` is the matrix of filters (the unmixing matrix).

    For a detailed discussion on the difference between filters and patterns for GED
    see :footcite:`Cohen2022` and :footcite:`HaufeEtAl2014` for linear models in
    general.

    .. versionadded:: 1.11

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        info,
        evecs,
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
        self.filters = evecs.T

        if patterns is None:
            self.patterns = pinv(evecs)
            self.patterns_method = "pinv"
        else:
            self.patterns = patterns
            self.patterns_method = patterns_method

    @fill_doc
    def plot_filters(
        self,
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
        name_format="Filter%01d",
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic maps of model filters.

        Parameters
        ----------
        components : float | array of float
            Indices of filters to plot. If None, all filters will be plotted.
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
        components : float | array of float
            Indices of patterns to plot. If None, all patterns will be plotted.
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

    def plot_scree(
        self,
        title="Scree plot",
        add_cumul_evals=True,
        plt_style="seaborn-v0_8-whitegrid",
        axes=None,
    ):
        """Plot scree for GED eigenvalues.

        Parameters
        ----------
        title : str
            Title for the plot. Defaults to ``'Scree plot'``.
        add_cumul_evals : bool
            Whether to add second line and y-axis for cumulative eigenvalues.
            Defaults to ``True``.
        plt_style : str
            Matplotlib plot style.
            Check available styles with plt.style.available.
            Defaults to ``'seaborn-v0_8-whitegrid'``.
        axes : instance of Axes | None
            The matplotlib axes to plot to. Defaults to ``None``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        if self.evals is None:
            raise ValueError("Can't plot scree if eigenvalues are not provided.")
        fig = _plot_scree(
            self.evals,
            title=title,
            add_cumul_evals=add_cumul_evals,
            plt_style=plt_style,
            axes=axes,
        )
        return fig
