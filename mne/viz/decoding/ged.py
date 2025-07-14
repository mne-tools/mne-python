# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy as cp

import matplotlib.pyplot as plt
import numpy as np

from ...defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ...evoked import EvokedArray


def _plot_model(
    model_array,
    info,
    model="inverse",
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
    if name_format is None:
        name_format = f"{model}%01d"

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
    filters_evk = EvokedArray(model_array.T, info, tmin=0)
    # the call plot_topomap
    fig = filters_evk.plot_topomap(
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
    ax=None,
):
    cumul_evals = np.cumsum(evals)
    n_components = len(evals)
    component_numbers = np.arange(n_components)

    # check available styles with plt.style.available
    with plt.style.context(plt_style):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        else:
            fig = None

        # plot individual eigenvalues
        color_line = "cornflowerblue"
        ax.set_xlabel("Component Index", fontsize=18)
        ax.set_ylabel("Eigenvalue", color=color_line, fontsize=18)
        ax.set_ylabel("Cumulative Eigenvalues", color=color_line, fontsize=18)
        ax.plot(component_numbers, evals, color=color_line, marker="o", markersize=10)
        ax.tick_params(axis="y", labelcolor=color_line, labelsize=16)

        if add_cumul_evals:
            # plot cumulative eigenvalues
            ax2 = ax.twinx()
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
