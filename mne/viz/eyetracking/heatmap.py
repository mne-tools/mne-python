# Authors: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from ..utils import plt_show
from ...utils import _validate_type, logger


def plot_gaze(
    epochs,
    width,
    height,
    n_bins=25,
    sigma=1,
    cmap=None,
    vmin=None,
    vmax=None,
    make_transparent=True,
    axes=None,
    show=True,
):
    """Plot a heatmap of eyetracking gaze data.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object containing the gaze data.
    width : int
        The width dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the display screens resolution was 1920x1080, then the width
        should be 1920.
    height : int
        The height dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the display screens resolution was 1920x1080, then the height
        should be 1080.
    n_bins : int
        The number of bins to use for the heatmap. Default is 25, which means the
        heatmap will be a 25x25 grid.
    sigma : float | None
        The sigma value for the gaussian kernel used to smooth the heatmap.
        If ``None``, no smoothing is applied. Default is 1.
    cmap : matplotlib colormap | str | None
        The matplotlib colormap to use. Defaults to None, which means the
        colormap will default to matplotlib's default.
    vmin : float | None
        The minimum value for the colormap. The unit is seconds, for dwell time
        to the pixel coordinate. If ``None``, the minimum value is set to the
        minimum value of the heatmap. Default is ``None``.
    vmax : float | None
        The maximum value for the colormap. The unit is seconds, for dwell time
        to the pixel coordinate. If ``None``, the maximum value is set to the
        maximum value of the heatmap. Default is ``None``.
    make_transparent : bool
        Whether to make the background transparent. Default is ``True``.
    axes : matplotlib.axes.Axes | None
        The axes to plot on. If ``None``, a new figure and axes are created.
        Default is ``None``.
    show : bool
        Whether to show the plot. Default is ``True``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object for the heatmap plot.
    """
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter
    from mne import BaseEpochs

    _validate_type(epochs, BaseEpochs, "epochs")

    # Find xpos and ypos channels:
    # In principle we could check the coil_type for eyetrack position channels,
    # which could be more robust if future readers use different channel names?
    xpos_indices = np.where(np.char.startswith(epochs.ch_names, "xpos"))[0]
    ypos_indices = np.where(np.char.startswith(epochs.ch_names, "ypos"))[0]

    data = epochs.get_data()
    x_data = data[:, xpos_indices, :]
    y_data = data[:, ypos_indices, :]
    if xpos_indices.size > 1:  # binocular recording. Average across eyes
        logger.info("Detected binocular recording. Averaging positions across eyes.")
        x_data = np.nanmean(x_data, axis=1)  # shape (n_epochs, n_samples)
        y_data = np.nanmean(y_data, axis=1)
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    gaze_data = np.vstack((x_data, y_data)).T  # shape (n_samples, 2)
    # Make sure gaze position data is within screen bounds
    mask = (
        (gaze_data[:, 0] > 0)
        & (gaze_data[:, 1] > 0)
        & (gaze_data[:, 0] < width)
        & (gaze_data[:, 1] < height)
    )
    canvas = gaze_data[mask].astype(float)

    # Create heatmap
    heatmap, _, _ = np.histogram2d(
        canvas[:, 0],
        canvas[:, 1],
        bins=n_bins,
        range=[[0, width], [0, height]],
    )
    heatmap = heatmap.T  # transpose to match screen coordinates
    # Convert density from samples to seconds
    heatmap /= epochs.info["sfreq"]
    if sigma:
        # Smooth heatmap
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Prepare axes
    if axes is not None:
        from matplotlib.axes import Axes

        _validate_type(axes, Axes, "axes")
        ax = axes
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    ax.set_title("Gaze heatmap")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    if make_transparent:
        # Make heatmap transparent
        norm = mcolors.Normalize(vmin=0, vmax=np.nanmax(heatmap))
        alphas = norm(heatmap)
    else:
        alphas = 1.0

    # Prepare the heatmap
    vmin = np.nanmin(heatmap) if vmin is None else vmin
    vmax = np.nanmax(heatmap) if vmax is None else vmax
    extent = [0, width, height, 0]  # origin is the top left of the screen
    # Plot heatmap
    im = ax.imshow(
        heatmap,
        aspect="equal",
        interpolation="gaussian",
        alpha=alphas,
        cmap=cmap,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Prepare the colorbar
    cbar = fig.colorbar(im, ax=ax, label="Dwell time (seconds)")
    # Prepare the colorbar transparency
    if make_transparent:
        cbar.set_alpha(1.0)
        cbar.solids.set(alpha=np.linspace(0, np.max(alphas), 256))

    plt_show(show)
    return fig
