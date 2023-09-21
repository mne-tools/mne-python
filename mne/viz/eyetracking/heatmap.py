# Authors: Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

import numpy as np
from scipy.ndimage import gaussian_filter


from ..utils import plt_show
from ...utils import _validate_type, logger, fill_doc


@fill_doc
def plot_gaze(
    epochs,
    width,
    height,
    bin_width=10,
    sigma=1,
    cmap=None,
    alpha=None,
    vmin=None,
    vmax=None,
    axes=None,
    show=True,
):
    """Plot a heatmap of eyetracking gaze data.

    Parameters
    ----------
    epochs : mne.Epochs
        The :class:`~mne.Epochs` object containing eyegaze channels.
    width : int
        The width dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the participant screen resolution was 1920x1080, then the width
        should be 1920.
    height : int
        The height dimension of the plot canvas. For example, if the eyegaze data units
        are pixels, and the participant screen resolution was 1920x1080, then the height
        should be 1080.
    bin_width : int
        The number of eyegaze units per square bin that are used to create the heatmap.
        Default is 10, meaning that if the eyegaze units are pixels, each bin is 10x10
        pixels. See the appendix section of :ref:`tut-eyetrack-heatmap` for more
        detail.
    sigma : int | float
        The amount of smoothing applied to the heatmap data. If ``None``,
        no smoothing is applied. Default is 1.
    cmap : matplotlib colormap | str | None
        The :class:`~matplotlib.colors.Colormap` to use. Defaults to ``None``, meaning
        the colormap will default to matplotlib's default.
    alpha : int | float | None
        The transparency value of the heatmap. If ``None``, the alpha value is set to 1,
        meaning the heatmap colors are fully opaque. Default is ``None``.
    vmin : float | None
        The minimum value for the colormap. The unit is seconds, for dwell time
        to the bin coordinate. If ``None``, the minimum value is set to the
        minimum value of the heatmap. Default is ``None``.
    vmax : float | None
        The maximum value for the colormap. The unit is seconds, for dwell time
        to the bin coordinate. If ``None``, the maximum value is set to the
        maximum value of the heatmap. Default is ``None``.
    axes : matplotlib.axes.Axes | None
        The axes to plot on. If ``None``, a new figure and axes are created.
        Default is ``None``.
    show : bool
        Whether to show the plot. Default is ``True``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object for the heatmap plot.

    Notes
    -----
    .. versionadded:: 1.6
    """
    from mne import BaseEpochs

    _validate_type(epochs, BaseEpochs, "epochs")

    # Find xpos and ypos channels. if future readers use different channel names than
    # xpos/ypos, we will need a different way to identify these channels
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
    # mask gaze data that is outside screen bounds
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
    # Smooth the heatmap
    if sigma:
        hist = gaussian_filter(hist, sigma=sigma)

    return _plot_heatmap_array(
        hist,
        width=width,
        height=height,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        axes=axes,
        show=show,
    )


def _plot_heatmap_array(
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
    """Plot a heatmap of eyetracking gaze data from a numpy array."""
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
        cmap=cmap,
        alpha=alphas,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Prepare the colorbar
    # stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fig.colorbar(im, ax=ax, label="Dwell time (seconds)", fraction=0.046, pad=0.04)
    plt_show(show)
    return fig
