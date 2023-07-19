# Authors: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

import numpy as np

from ..utils import plt_show
from ...utils import _validate_type, logger


def plot_heatmap_array(
    data,
    width,
    height,
    cmap=None,
    alpha=None,
    vmin=None,
    vmax=None,
    axes=None,
    show=True,
):
    """Plot a heatmap of eyetracking gaze data from a numpy array.

    Parameters
    ----------
    data : numpy array
        The heatmap data to plot.
    width : int
        The width dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the display screens resolution was 1920x1080, then the width
        should be 1920.
    height : int
        The height dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the display screens resolution was 1920x1080, then the height
        should be 1080.
    cmap : matplotlib colormap | str | None
        The matplotlib colormap to use. Defaults to None, which means the colormap will
        default to matplotlib's default.
    alpha : float | array-like | None
        The alpha value(s) to use for the colormap. If ``None``, the alpha value is set
        to 1. Default is ``None``. If an array-like object is passed, the shape must
        match the shape of the data array.
    vmin : float | None
        The minimum value for the colormap. The unit is seconds, for the dwell time to
        the pixel coordinate. If ``None``, the minimum value is set to the minimum value
        of the heatmap. Default is ``None``.
    vmax : float | None
        The maximum value for the colormap. The unit is seconds, for the dwell time to
        the pixel coordinate. If ``None``, the maximum value is set to the maximum value
        of the heatmap. Default is ``None``.
    axes : matplotlib.axes.Axes | None
        The axes to plot on. If ``None``, a new figure and axes are created.
    show : bool
        Whether to show the plot. Default is ``True``.
    """
    import matplotlib.pyplot as plt

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

    # Prepare the heatmap
    alphas = 1 if alpha is None else alpha
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    extent = [0, width, height, 0]  # origin is the top left of the screen
    # Plot heatmap
    im = ax.imshow(
        data,
        aspect="equal",
        interpolation="none",
        cmap=cmap,
        alpha=alphas,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Prepare the colorbar
    fig.colorbar(im, ax=ax, label="Dwell time (seconds)")
    plt_show(show)
    return fig


def plot_gaze(
    epochs,
    width,
    height,
    bin_width,
    sigma=1,
    cmap=None,
    vmin=None,
    vmax=None,
    axes=None,
    show=True,
    return_array=False,
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
    bin_width : int
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
    from scipy.ndimage import gaussian_filter
    from mne import BaseEpochs

    _validate_type(epochs, BaseEpochs, "epochs")

    # Find xpos and ypos channels:
    # We could check the channeltype for eyegaze channels, which could be more robust if
    # future readers use different channel names? However channel type will not
    # differentiate between x-position and y-position.
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
    # filter out gaze data that is outside screen bounds
    mask = (
        (gaze_data[:, 0] > 0)
        & (gaze_data[:, 1] > 0)
        & (gaze_data[:, 0] < width)
        & (gaze_data[:, 1] < height)
    )
    canvas = gaze_data[mask].astype(float)

    # Create 2D histogram
    x_bins = np.linspace(0, width, width // bin_width)
    y_bins = np.linspace(0, height, height // bin_width)
    hist, _, _ = np.histogram2d(
        canvas[:, 0],
        canvas[:, 1],
        bins=(x_bins, y_bins),
        range=[[0, width], [0, height]],
    )
    hist = hist.T  # transpose to match screen coordinates. i.e. width > height
    # Convert density from samples to seconds
    hist /= epochs.info["sfreq"]
    if sigma:
        # Smooth heatmap
        hist = gaussian_filter(hist, sigma=sigma)

    fig = plot_heatmap_array(hist, width, height, cmap, vmin, vmax, axes)
    if return_array:
        return fig, hist
    return fig
