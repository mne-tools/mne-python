# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Functions to plot EEG sensor montages or digitizer montages."""

from copy import deepcopy

import numpy as np

from .._fiff._digitization import _get_fid_coords
from .._fiff.meas_info import create_info
from ..utils import _check_option, _validate_type, logger, verbose_static
from .utils import plot_sensors


@verbose_static("sphere_topomap_auto", "axes_montage")
def plot_montage(
    montage,
    *,
    scale=1.0,
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
        down, whereas values > 1 will scale up.
    show_names : bool | list
        Whether to display all channel names. If a list, only the channel
        names in the list are shown. Defaults to True.
    kind : str
        Whether to plot the montage as '3d' or 'topomap' (default).
    show : bool
        Show figure if True.
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
    axes : instance of Axes | instance of Axes3D | None
        Axes to draw the sensors to. If ``kind='3d'``, axes must be an instance
        of Axes3D. If None (default), a new axes will be created.

        .. versionadded:: 1.4
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist

    from ..channels import DigMontage, make_dig_montage

    _validate_type(scale, "numeric", "scale")
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

    if scale != 1.0:
        axes = axes if axes else fig.axes[0]

        # scale points
        collection = axes.collections[0]
        collection.set_sizes([scale * 10])

        # scale labels
        labels = axes.findobj(match=plt.Text)
        x_label, y_label = axes.xaxis.label, axes.yaxis.label
        z_label = axes.zaxis.label if kind == "3d" else None
        tick_labels = axes.get_xticklabels() + axes.get_yticklabels()
        if kind == "3d":
            tick_labels += axes.get_zticklabels()
        for label in labels:
            if label not in [x_label, y_label, z_label] + tick_labels:
                label.set_fontsize(label.get_fontsize() * scale)

    return fig
