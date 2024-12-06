# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Functions to plot EEG sensor montages or digitizer montages."""

from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist

from .._fiff._digitization import _get_fid_coords
from .._fiff.meas_info import create_info
from ..utils import _check_option, _validate_type, logger, verbose
from .utils import plot_sensors


@verbose
def plot_montage(
    montage,
    *,
    scale=None,
    scale_factor=None,
    show_names=True,
    kind="topomap",
    show=True,
    sphere=None,
    axes=None,
    verbose=None,
):
    """Plot a montage.

    Parameters
    ----------
    montage : instance of DigMontage
        The montage to visualize.
    scale : float
        Determines the scale of the channel points and labels; values < 1 will scale
        down, whereas values > 1 will scale up. Default to None, which implies 1.
    scale_factor : float
        Determines the size of the points. Deprecated, use scale instead.
    show_names : bool | list
        Whether to display all channel names. If a list, only the channel
        names in the list are shown. Defaults to True.
    kind : str
        Whether to plot the montage as '3d' or 'topomap' (default).
    show : bool
        Show figure if True.
    %(sphere_topomap_auto)s
    %(axes_montage)s

        .. versionadded:: 1.4
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    from ..channels import DigMontage, make_dig_montage

    if scale_factor is not None:
        msg = "scale_factor has been deprecated and will be removed. Use scale instead."
        if scale is not None:
            raise ValueError(
                " ".join(["scale and scale_factor cannot be used together.", msg])
            )
        logger.info(msg)
    if scale is None:
        scale = 1

    _check_option("kind", kind, ["topomap", "3d"])
    _validate_type(montage, DigMontage, item_name="montage")
    ch_names = montage.ch_names
    title = None

    if len(ch_names) == 0:
        raise RuntimeError("No valid channel positions found.")

    pos = np.array(list(montage._get_ch_pos().values()))

    dists = cdist(pos, pos)

    # only consider upper triangular part by setting the rest to np.nan
    dists[np.tril_indices(dists.shape[0])] = np.nan
    dupes = np.argwhere(np.isclose(dists, 0))
    if dupes.any():
        montage = deepcopy(montage)
        n_chans = pos.shape[0]
        n_dupes = dupes.shape[0]
        idx = np.setdiff1d(np.arange(len(pos)), dupes[:, 1]).tolist()
        logger.info(f"{n_dupes} duplicate electrode labels found:")
        logger.info(", ".join([ch_names[d[0]] + "/" + ch_names[d[1]] for d in dupes]))
        logger.info(f"Plotting {n_chans - n_dupes} unique labels.")
        ch_names = [ch_names[i] for i in idx]
        ch_pos = dict(zip(ch_names, pos[idx, :]))
        # XXX: this might cause trouble if montage was originally in head
        fid, _ = _get_fid_coords(montage.dig)
        montage = make_dig_montage(ch_pos=ch_pos, **fid)

    info = create_info(ch_names, sfreq=256, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    fig = plot_sensors(
        info,
        kind=kind,
        show_names=show_names,
        show=show,
        title=title,
        sphere=sphere,
        axes=axes,
    )

    if scale_factor is not None:
        # scale points
        collection = fig.axes[0].collections[0]
        collection.set_sizes([scale_factor])
    elif scale is not None:
        # scale points
        collection = fig.axes[0].collections[0]
        collection.set_sizes([scale * 10])

        # scale labels
        labels = fig.findobj(match=plt.Text)
        x_label, y_label = fig.axes[0].xaxis.label, fig.axes[0].yaxis.label
        z_label = fig.axes[0].zaxis.label if kind == "3d" else None
        tick_labels = fig.axes[0].get_xticklabels() + fig.axes[0].get_yticklabels()
        if kind == "3d":
            tick_labels += fig.axes[0].get_zticklabels()
        for label in labels:
            if label not in [x_label, y_label, z_label] + tick_labels:
                label.set_fontsize(label.get_fontsize() * scale)

    return fig
