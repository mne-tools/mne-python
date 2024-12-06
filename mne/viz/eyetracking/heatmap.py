# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.ndimage import gaussian_filter

from ..._fiff.constants import FIFF
from ...utils import _validate_type, fill_doc, logger
from ..utils import plt_show


@fill_doc
def plot_gaze(
    epochs,
    *,
    calibration=None,
    width=None,
    height=None,
    sigma=25,
    cmap=None,
    alpha=1.0,
    vlim=(None, None),
    axes=None,
    show=True,
):
    """Plot a heatmap of eyetracking gaze data.

    Parameters
    ----------
    epochs : instance of Epochs
        The :class:`~mne.Epochs` object containing eyegaze channels.
    calibration : instance of Calibration | None
        An instance of Calibration with information about the screen size, distance,
        and resolution. If ``None``, you must provide a width and height.
    width : int
        The width dimension of the plot canvas, only valid if eyegaze data are in
        pixels. For example, if the participant screen resolution was 1920x1080, then
        the width should be 1920.
    height : int
        The height dimension of the plot canvas, only valid if eyegaze data are in
        pixels. For example, if the participant screen resolution was 1920x1080, then
        the height should be 1080.
    sigma : float | None
        The amount of Gaussian smoothing applied to the heatmap data (standard
        deviation in pixels). If ``None``, no smoothing is applied. Default is 25.
    %(cmap)s
    alpha : float
        The opacity of the heatmap (default is 1).
    %(vlim_plot_topomap)s
    %(axes_plot_topomap)s
    %(show)s

    Returns
    -------
    fig : instance of Figure
        The resulting figure object for the heatmap plot.

    Notes
    -----
    .. versionadded:: 1.6
    """
    from mne import BaseEpochs
    from mne._fiff.pick import _picks_to_idx

    from ...preprocessing.eyetracking.utils import (
        _check_calibration,
        get_screen_visual_angle,
    )

    _validate_type(epochs, BaseEpochs, "epochs")
    _validate_type(alpha, "numeric", "alpha")
    _validate_type(sigma, ("numeric", None), "sigma")

    # Get the gaze data
    pos_picks = _picks_to_idx(epochs.info, "eyegaze")
    gaze_data = epochs.get_data(picks=pos_picks)
    gaze_ch_loc = np.array([epochs.info["chs"][idx]["loc"] for idx in pos_picks])
    x_data = gaze_data[:, np.where(gaze_ch_loc[:, 4] == -1)[0], :]
    y_data = gaze_data[:, np.where(gaze_ch_loc[:, 4] == 1)[0], :]
    unit = epochs.info["chs"][pos_picks[0]]["unit"]  # assumes all units are the same

    if x_data.shape[1] > 1:  # binocular recording. Average across eyes
        logger.info("Detected binocular recording. Averaging positions across eyes.")
        x_data = np.nanmean(x_data, axis=1)  # shape (n_epochs, n_samples)
        y_data = np.nanmean(y_data, axis=1)
    canvas = np.vstack((x_data.flatten(), y_data.flatten()))  # shape (2, n_samples)

    # Check that we have the right inputs
    if calibration is not None:
        if width is not None or height is not None:
            raise ValueError(
                "If a calibration is provided, you cannot provide a width or height"
                " to plot heatmaps. Please provide only the calibration object."
            )
        _check_calibration(calibration)
        if unit == FIFF.FIFF_UNIT_PX:
            width, height = calibration["screen_resolution"]
        elif unit == FIFF.FIFF_UNIT_RAD:
            width, height = calibration["screen_size"]
        else:
            raise ValueError(
                f"Invalid unit type: {unit}. gaze data Must be pixels or radians."
            )
    else:
        if width is None or height is None:
            raise ValueError(
                "If no calibration is provided, you must provide a width and height"
                " to plot heatmaps."
            )

    # Create 2D histogram
    # We need to set the histogram bins & bounds, and imshow extent, based on the units
    if unit == FIFF.FIFF_UNIT_PX:  # pixel on screen
        _range = [[0, height], [0, width]]
        bins_x, bins_y = width, height
        extent = [0, width, height, 0]
    elif unit == FIFF.FIFF_UNIT_RAD:  # radians of visual angle
        if not calibration:
            raise ValueError(
                "If gaze data are in Radians, you must provide a"
                " calibration instance to plot heatmaps."
            )
        width, height = get_screen_visual_angle(calibration)
        x_range = [-width / 2, width / 2]
        y_range = [-height / 2, height / 2]
        _range = [y_range, x_range]
        extent = (x_range[0], x_range[1], y_range[0], y_range[1])
        bins_x, bins_y = calibration["screen_resolution"]

    hist, _, _ = np.histogram2d(
        canvas[1, :],
        canvas[0, :],
        bins=(bins_y, bins_x),
        range=_range,
    )
    # Convert density from samples to seconds
    hist /= epochs.info["sfreq"]
    # Smooth the heatmap
    if sigma:
        hist = gaussian_filter(hist, sigma=sigma)

    return _plot_heatmap_array(
        hist,
        width=width,
        height=height,
        cmap=cmap,
        alpha=alpha,
        vmin=vlim[0],
        vmax=vlim[1],
        extent=extent,
        axes=axes,
        show=show,
    )


def _plot_heatmap_array(
    data,
    width,
    height,
    *,
    cmap=None,
    alpha=None,
    vmin=None,
    vmax=None,
    extent=None,
    axes=None,
    show=True,
):
    """Plot a heatmap of eyetracking gaze data from a numpy array."""
    import matplotlib.pyplot as plt

    # Prepare axes
    if axes is not None:
        from matplotlib.axes import Axes

        _validate_type(axes, Axes, "axes")
        ax = axes
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(constrained_layout=True)

    ax.set_title("Gaze heatmap")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    # Prepare the heatmap
    alphas = 1 if alpha is None else alpha
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    if extent is None:
        extent = [0, width, height, 0]

    # Plot heatmap
    im = ax.imshow(
        data,
        aspect="equal",
        cmap=cmap,
        alpha=alphas,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Prepare the colorbar
    fig.colorbar(im, ax=ax, shrink=0.6, label="Dwell time (seconds)")
    plt_show(show)
    return fig
