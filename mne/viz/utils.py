"""Utility functions for plotting M/EEG data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import difflib
import math
import os
import sys
import tempfile
import traceback
import webbrowser
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import numpy as np
from decorator import decorator
from scipy.signal import argrelmax

from .._fiff.constants import FIFF
from .._fiff.meas_info import Info
from .._fiff.open import show_fiff
from .._fiff.pick import (
    _DATA_CH_TYPES_ORDER_DEFAULT,
    _DATA_CH_TYPES_SPLIT,
    _VALID_CHANNEL_TYPES,
    _contains_ch_type,
    _pick_data_channels,
    _picks_by_type,
    channel_indices_by_type,
    channel_type,
    pick_channels,
    pick_channels_cov,
    pick_info,
)
from .._fiff.proj import Projection, setup_proj
from ..defaults import _handle_default
from ..fixes import _median_complex
from ..rank import compute_rank
from ..transforms import apply_trans
from ..utils import (
    _auto_weakref,
    _check_ch_locs,
    _check_decim,
    _check_option,
    _check_sphere,
    _ensure_int,
    _pl,
    _to_rgb,
    _validate_type,
    fill_doc,
    get_config,
    logger,
    verbose,
    warn,
)
from ..utils.misc import _identity_function
from .ui_events import ColormapRange, publish, subscribe

_channel_type_prettyprint = {
    "eeg": "EEG channel",
    "grad": "Gradiometer",
    "mag": "Magnetometer",
    "seeg": "sEEG channel",
    "dbs": "DBS channel",
    "eog": "EOG channel",
    "ecg": "ECG sensor",
    "emg": "EMG sensor",
    "ecog": "ECoG channel",
    "misc": "miscellaneous sensor",
}


@decorator
def safe_event(fun, *args, **kwargs):
    """Protect against Qt exiting on event-handling errors."""
    try:
        return fun(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stderr)


def _setup_vmin_vmax(data, vmin, vmax, norm=False):
    """Handle vmin and vmax parameters for visualizing topomaps.

    For the normal use-case (when `vmin` and `vmax` are None), the parameter
    `norm` drives the computation. When norm=False, data is supposed to come
    from a mag and the output tuple (vmin, vmax) is symmetric range
    (-x, x) where x is the max(abs(data)). When norm=True (a.k.a. data is the
    L2 norm of a gradiometer pair) the output tuple corresponds to (0, x).

    Otherwise, vmin and vmax are callables that drive the operation.
    """
    should_warn = False
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        vmin = 0.0 if norm else -vmax
        if vmin == 0 and np.min(data) < 0:
            should_warn = True

    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            vmin = 0.0 if norm else np.min(data)
            if vmin == 0 and np.min(data) < 0:
                should_warn = True

        if callable(vmax):
            vmax = vmax(data)
        elif vmax is None:
            vmax = np.max(data)

    if should_warn:
        warn_msg = (
            "_setup_vmin_vmax output a (min={vmin}, max={vmax})"
            " range whereas the minimum of data is {data_min}"
        )
        warn_val = {"vmin": vmin, "vmax": vmax, "data_min": np.min(data)}
        warn(warn_msg.format(**warn_val), UserWarning)

    return vmin, vmax


def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    import matplotlib.pyplot as plt
    from matplotlib import get_backend

    if hasattr(fig, "mne") and hasattr(fig.mne, "backend"):
        backend = fig.mne.backend
        # TODO: This is a hack to deal with the fact that the
        # with plt.ion():
        #     BACKEND = get_backend()
        # an the top of _mpl_figure detects QtAgg during testing even though
        # we've set the backend to Agg.
        if backend != "agg":
            gotten_backend = get_backend()
            if gotten_backend == "agg":
                backend = "agg"
    else:
        backend = get_backend()
    if show and backend != "agg":
        logger.debug(f"Showing plot for backend {repr(backend)}")
        (fig or plt).show(**kwargs)


def _show_browser(show=True, block=True, fig=None, **kwargs):
    """Show the browser considering different backends.

    Parameters
    ----------
    show : bool
        Show the figure.
    block : bool
        If to block execution on showing.
    fig : instance of Figure | None
        Needs to be passed for Qt backend,
        optional for matplotlib.
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from ._figure import get_browser_backend

    _validate_type(block, bool, "block")
    backend = get_browser_backend()
    if os.getenv("_MNE_BROWSER_NO_BLOCK", "false").lower() == "true":
        block = False
    if backend == "matplotlib":
        plt_show(show, block=block, **kwargs)
    else:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        from .backends._utils import _qt_app_exec

        if fig is not None and os.getenv("_MNE_BROWSER_BACK", "").lower() == "true":
            fig.setWindowFlags(fig.windowFlags() | Qt.WindowStaysOnBottomHint)
        if show:
            fig.show()
        # If block=False, a Qt-Event-Loop has to be started
        # somewhere else in the calling code.
        if block:
            _qt_app_exec(QApplication.instance())


def _check_delayed_ssp(container):
    """Handle interactive SSP selection."""
    if container.proj is True or all(p["active"] for p in container.info["projs"]):
        raise RuntimeError(
            "Projs are already applied. Please initialize"
            " the data with proj set to False."
        )
    elif len(container.info["projs"]) < 1:
        raise RuntimeError("No projs found in evoked.")


def _validate_if_list_of_axes(axes, obligatory_len=None, name="axes"):
    """Validate whether input is a list/array of axes."""
    from matplotlib.axes import Axes

    _validate_type(axes, (list, tuple, np.ndarray), name)
    if isinstance(axes, np.ndarray) and axes.ndim > 1:
        raise ValueError(
            f"if {name} is a numpy array, it must be one-dimensional, but "
            f"the received numpy array has {axes.ndim} dimensions. Try using "
            "ravel or flatten method of the array."
        )
    wrong_idx = np.where([not isinstance(x, Axes) for x in axes])[0]
    if len(wrong_idx):
        raise TypeError(
            f"{name} must be an array-like of matplotlib axes objects, but "
            f"{name}[{wrong_idx[0]}] is of type {type(axes[wrong_idx[0]])}"
        )
    if obligatory_len is not None:
        obligatory_len = _ensure_int(
            obligatory_len, "obligatory_len", extra="if not None"
        )
        if len(axes) != obligatory_len:
            raise ValueError(
                f"{name} must be an array-like of length {obligatory_len}, "
                f"but the length is {len(axes)}"
            )


def mne_analyze_colormap(limits=(5, 10, 15), format="vtk"):  # noqa: A002
    """Return a colormap similar to that used by mne_analyze.

    Parameters
    ----------
    limits : array-like of length 3 or 6
        Bounds for the colormap, which will be mirrored across zero if length
        3, or completely specified (and potentially asymmetric) if length 6.
    format : str
        Type of colormap to return. If 'matplotlib', will return a
        matplotlib.colors.LinearSegmentedColormap. If 'vtk', will
        return an RGBA array of shape (256, 4).

    Returns
    -------
    cmap : instance of colormap | array
        A teal->blue->gray->red->yellow colormap. See docstring of the 'format'
        argument for further details.

    Notes
    -----
    For this will return a colormap that will display correctly for data
    that are scaled by the plotting function to span [-fmax, fmax].
    """  # noqa: E501
    # Ensure limits is an array
    limits = np.asarray(limits, dtype="float")

    if len(limits) != 3 and len(limits) != 6:
        raise ValueError("limits must have 3 or 6 elements")
    if len(limits) == 3 and any(limits < 0.0):
        raise ValueError("if 3 elements, limits must all be non-negative")
    if any(np.diff(limits) <= 0):
        raise ValueError("limits must be monotonically increasing")
    if format == "matplotlib":
        from matplotlib import colors

        if len(limits) == 3:
            limits = (np.concatenate((-np.flipud(limits), limits)) + limits[-1]) / (
                2 * limits[-1]
            )
        else:
            limits = (limits - np.min(limits)) / np.max(limits - np.min(limits))

        cdict = {
            "red": (
                (limits[0], 0.0, 0.0),
                (limits[1], 0.0, 0.0),
                (limits[2], 0.5, 0.5),
                (limits[3], 0.5, 0.5),
                (limits[4], 1.0, 1.0),
                (limits[5], 1.0, 1.0),
            ),
            "green": (
                (limits[0], 1.0, 1.0),
                (limits[1], 0.0, 0.0),
                (limits[2], 0.5, 0.5),
                (limits[3], 0.5, 0.5),
                (limits[4], 0.0, 0.0),
                (limits[5], 1.0, 1.0),
            ),
            "blue": (
                (limits[0], 1.0, 1.0),
                (limits[1], 1.0, 1.0),
                (limits[2], 0.5, 0.5),
                (limits[3], 0.5, 0.5),
                (limits[4], 0.0, 0.0),
                (limits[5], 0.0, 0.0),
            ),
            "alpha": (
                (limits[0], 1.0, 1.0),
                (limits[1], 1.0, 1.0),
                (limits[2], 0.0, 0.0),
                (limits[3], 0.0, 0.0),
                (limits[4], 1.0, 1.0),
                (limits[5], 1.0, 1.0),
            ),
        }
        return colors.LinearSegmentedColormap("mne_analyze", cdict)
    elif format in ("vtk", "mayavi"):
        if len(limits) == 3:
            limits = np.concatenate((-np.flipud(limits), [0], limits)) / limits[-1]
        else:
            limits = np.concatenate((limits[:3], [0], limits[3:]))
            limits /= np.max(np.abs(limits))
        r = np.array([0, 0, 0, 0, 1, 1, 1])
        g = np.array([1, 0, 0, 0, 0, 0, 1])
        b = np.array([1, 1, 1, 0, 0, 0, 0])
        a = np.array([1, 1, 0, 0, 0, 1, 1])
        xp = (np.arange(256) - 128) / 128.0
        colormap = np.r_[[np.interp(xp, limits, 255 * c) for c in [r, g, b, a]]].T
        return colormap
    else:
        # Use this instead of check_option because we have a hidden option
        raise ValueError(f"format must be either matplotlib or vtk, got {repr(format)}")


@contextmanager
def _events_off(obj):
    obj.eventson = False
    try:
        yield
    finally:
        obj.eventson = True


def _toggle_proj(event, params, all_=False):
    """Perform operations when proj boxes clicked."""
    # read options if possible
    if "proj_checks" in params:
        bools = list(params["proj_checks"].get_status())
        if all_:
            new_bools = [not all(bools)] * len(bools)
            with _events_off(params["proj_checks"]):
                for bi, (old, new) in enumerate(zip(bools, new_bools)):
                    if old != new:
                        params["proj_checks"].set_active(bi)
                        bools[bi] = new
        for bi, (b, p) in enumerate(zip(bools, params["projs"])):
            # see if they tried to deactivate an active one
            if not b and p["active"]:
                bools[bi] = True
    else:
        proj = params.get("apply_proj", True)
        bools = [proj] * len(params["projs"])

    compute_proj = False
    if "proj_bools" not in params:
        compute_proj = True
    elif not np.array_equal(bools, params["proj_bools"]):
        compute_proj = True

    # if projectors changed, update plots
    if compute_proj is True:
        params["plot_update_proj_callback"](params, bools)


def _get_channel_plotting_order(order, ch_types, picks=None):
    """Determine channel plotting order for browse-style Raw/Epochs plots."""
    if order is None:
        # for backward compat, we swap the first two to keep grad before mag
        ch_type_order = list(_DATA_CH_TYPES_ORDER_DEFAULT)
        ch_type_order = tuple(["grad", "mag"] + ch_type_order[2:])
        order = [
            pick_idx
            for order_type in ch_type_order
            for pick_idx, pick_type in enumerate(ch_types)
            if order_type == pick_type
        ]
    elif not isinstance(order, np.ndarray | list | tuple):
        raise ValueError(f'order should be array-like; got "{order}" ({type(order)}).')
    if picks is not None:
        order = [ch for ch in order if ch in picks]
    return np.asarray(order, int)


def _make_event_color_dict(event_color, events=None, event_id=None):
    """Make or validate a dict mapping event ids to colors."""
    from .misc import _handle_event_colors

    if isinstance(event_color, dict):  # if event_color is a dict, validate it
        event_id = dict() if event_id is None else event_id
        event_color = {
            _ensure_int(event_id.get(key, key), "event_color key"): value
            for key, value in event_color.items()
        }
        default = event_color.pop(-1, None)
        default_factory = None if default is None else lambda: default
        new_dict = defaultdict(default_factory)
        for key, value in event_color.items():
            if key < 1:
                raise KeyError(
                    "event_color keys must be strictly positive, "
                    f"or -1 (cannot use {key})"
                )
            new_dict[key] = value
        return new_dict
    elif event_color is None:  # make a dict from color cycle
        uniq_events = set() if events is False else np.unique(events[:, 2])
        return _handle_event_colors(event_color, uniq_events, event_id)
    else:  # if event_color is a MPL color-like thing, use it for all events
        return defaultdict(lambda: event_color)


def _prepare_trellis(
    n_cells,
    ncols,
    nrows="auto",
    title=False,
    size=1.3,
    sharex=False,
    sharey=False,
):
    from matplotlib.gridspec import GridSpec

    from ._mpl_figure import _figure

    if n_cells == 1:
        nrows = ncols = 1
    elif isinstance(ncols, int) and n_cells <= ncols:
        nrows, ncols = 1, n_cells
    else:
        if ncols == "auto" and nrows == "auto":
            nrows = math.floor(math.sqrt(n_cells))
            ncols = math.ceil(n_cells / nrows)
        elif ncols == "auto":
            ncols = math.ceil(n_cells / nrows)
        elif nrows == "auto":
            nrows = math.ceil(n_cells / ncols)
        else:
            naxes = ncols * nrows
            if naxes < n_cells:
                raise ValueError(
                    f"Cannot plot {n_cells} axes in a {nrows} by {ncols} figure."
                )

    width = size * ncols
    height = (size + max(0, 0.1 * (4 - size))) * nrows + bool(title) * 0.5
    fig = _figure(toolbar=False, figsize=(width * 1.5, 0.25 + height * 1.5))
    gs = GridSpec(nrows, ncols, figure=fig)

    axes = []
    for ax_idx in range(n_cells):
        subplot_kw = dict()
        if ax_idx > 0:
            if sharex:
                subplot_kw.update(sharex=axes[0])
            if sharey:
                subplot_kw.update(sharey=axes[0])
        axes.append(fig.add_subplot(gs[ax_idx], **subplot_kw))

    return fig, axes, ncols, nrows


def _draw_proj_checkbox(event, params, draw_current_state=True):
    """Toggle options (projectors) dialog."""
    from matplotlib import widgets

    projs = params["projs"]
    # turn on options dialog

    labels = [p["desc"] for p in projs]
    actives = (
        [p["active"] for p in projs]
        if draw_current_state
        else params.get("proj_bools", [params["apply_proj"]] * len(projs))
    )

    width = max([4.0, max([len(p["desc"]) for p in projs]) / 6.0 + 0.5])
    height = (len(projs) + 1) / 6.0 + 1.5
    # We manually place everything here so avoid constrained layouts
    fig_proj = figure_nobar(figsize=(width, height), layout=None)
    _set_window_title(fig_proj, "SSP projection vectors")
    offset = 1.0 / 6.0 / height
    params["fig_proj"] = fig_proj  # necessary for proper toggling
    ax_temp = fig_proj.add_axes((0, offset, 1, 0.8 - offset), frameon=False)
    ax_temp.set_title('Projectors marked with "X" are active')

    # make edges around checkbox areas and change already-applied projectors
    # to red
    from ._mpl_figure import _OLD_BUTTONS

    check_kwargs = dict()
    if not _OLD_BUTTONS:
        checkcolor = ["#ff0000" if p["active"] else "k" for p in projs]
        check_kwargs["check_props"] = dict(facecolor=checkcolor)
        check_kwargs["frame_props"] = dict(edgecolor="0.5", linewidth=1)
    proj_checks = widgets.CheckButtons(
        ax_temp, labels=labels, actives=actives, **check_kwargs
    )
    if _OLD_BUTTONS:
        for rect in proj_checks.rectangles:
            rect.set_edgecolor("0.5")
            rect.set_linewidth(1.0)
        for ii, p in enumerate(projs):
            if p["active"]:
                for x in proj_checks.lines[ii]:
                    x.set_color("#ff0000")

    # make minimal size
    # pass key presses from option dialog over
    proj_checks.on_clicked(partial(_toggle_proj, params=params))
    params["proj_checks"] = proj_checks
    fig_proj.canvas.mpl_connect("key_press_event", _key_press)

    # Toggle all
    ax_temp = fig_proj.add_axes((0, 0, 1, offset), frameon=False)
    proj_all = widgets.Button(ax_temp, "Toggle all")
    proj_all.on_clicked(partial(_toggle_proj, params=params, all_=True))
    params["proj_all"] = proj_all

    # this should work for non-test cases
    try:
        fig_proj.canvas.draw()
        plt_show(fig=fig_proj, warn=False)
    except Exception:
        pass


def _simplify_float(label):
    # Heuristic to turn floats to ints where possible (e.g. -500.0 to -500)
    if (
        isinstance(label, float)
        and np.isfinite(label)
        and float(str(label)) != round(label)
    ):
        label = round(label, 2)
    return label


def _get_figsize_from_config():
    """Get default / most recent figure size from config."""
    figsize = get_config("MNE_BROWSE_RAW_SIZE")
    if figsize is not None:
        figsize = figsize.split(",")
        figsize = tuple([float(s) for s in figsize])
    return figsize


@verbose
def compare_fiff(
    fname_1,
    fname_2,
    fname_out=None,
    show=True,
    indent="    ",
    read_limit=np.inf,
    max_str=30,
    verbose=None,
):
    """Compare the contents of two fiff files using diff and show_fiff.

    Parameters
    ----------
    fname_1 : path-like
        First file to compare.
    fname_2 : path-like
        Second file to compare.
    fname_out : path-like | None
        Filename to store the resulting diff. If None, a temporary
        file will be created.
    show : bool
        If True, show the resulting diff in a new tab in a web browser.
    indent : str
        How to indent the lines.
    read_limit : int
        Max number of bytes of data to read from a tag. Can be np.inf
        to always read all data (helps test read completion).
    max_str : int
        Max number of characters of string representation to print for
        each tag's data.
    %(verbose)s

    Returns
    -------
    fname_out : str
        The filename used for storing the diff. Could be useful for
        when a temporary file is used.
    """
    file_1 = show_fiff(
        fname_1, output=list, indent=indent, read_limit=read_limit, max_str=max_str
    )
    file_2 = show_fiff(
        fname_2, output=list, indent=indent, read_limit=read_limit, max_str=max_str
    )
    diff = difflib.HtmlDiff().make_file(file_1, file_2, fname_1, fname_2)
    if fname_out is not None:
        f = open(fname_out, "wb")
    else:
        f = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".html")
        fname_out = f.name
    with f as fid:
        fid.write(diff.encode("utf-8"))
    if show is True:
        webbrowser.open_new_tab(fname_out)
    return fname_out


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar.

    Parameters
    ----------
    *args : list
        Arguments to pass to :func:`matplotlib.pyplot.figure`.
    **kwargs : dict
        Keyword arguments to pass to :func:`matplotlib.pyplot.figure`.

    Returns
    -------
    fig : instance of Figure
        The figure.
    """
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    old_val = rcParams["toolbar"]
    try:
        rcParams["toolbar"] = "none"
        if "layout" not in kwargs:
            kwargs["layout"] = "constrained"
        fig = plt.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        cbs = list(fig.canvas.callbacks.callbacks["key_press_event"].keys())
        for key in cbs:
            fig.canvas.callbacks.disconnect(key)
    finally:
        rcParams["toolbar"] = old_val
    return fig


def _show_help_fig(col1, col2, fig_help, ax, show):
    _set_window_title(fig_help, "Help")
    celltext = [
        [c1, c2] for c1, c2 in zip(col1.strip().split("\n"), col2.strip().split("\n"))
    ]
    table = ax.table(cellText=celltext, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax.set_axis_off()
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(None)  # remove cell borders
        # right justify, following:
        # https://stackoverflow.com/questions/48210749/matplotlib-table-assign-different-text-alignments-to-different-columns?rq=1  # noqa: E501
        if col == 0:
            cell._loc = "right"

    fig_help.canvas.mpl_connect("key_press_event", _key_press)

    if show:
        # this should work for non-test cases
        try:
            fig_help.canvas.draw()
            plt_show(fig=fig_help, warn=False)
        except Exception:
            pass


def _key_press(event):
    """Handle key press in dialog."""
    import matplotlib.pyplot as plt

    if event.key == "escape":
        plt.close(event.canvas.figure)


class ClickableImage:
    """Display an image so you can click on it and store x/y positions.

    Takes as input an image array (can be any array that works with imshow,
    but will work best with images.  Displays the image and lets you
    click on it.  Stores the xy coordinates of each click, so now you can
    superimpose something on top of it.

    Upon clicking, the x/y coordinate of the cursor will be stored in
    self.coords, which is a list of (x, y) tuples.

    Parameters
    ----------
    imdata : ndarray
        The image that you wish to click on for 2-d points.
    **kwargs : dict
        Keyword arguments. Passed to ax.imshow.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self, imdata, **kwargs):
        """Display the image for clicking."""
        import matplotlib.pyplot as plt

        self.coords = []
        self.imdata = imdata
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ymax = self.imdata.shape[0]
        self.xmax = self.imdata.shape[1]
        self.im = self.ax.imshow(
            imdata, extent=(0, self.xmax, 0, self.ymax), picker=True, **kwargs
        )
        self.ax.axis("off")
        self.fig.canvas.mpl_connect("pick_event", self.onclick)
        plt_show(block=True)

    def onclick(self, event):
        """Handle Mouse clicks.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The matplotlib object that we use to get x/y position.
        """
        mouseevent = event.mouseevent
        self.coords.append((mouseevent.xdata, mouseevent.ydata))

    def plot_clicks(self, **kwargs):
        """Plot the x/y positions stored in self.coords.

        Parameters
        ----------
        **kwargs : dict
            Arguments are passed to imshow in displaying the bg image.
        """
        import matplotlib.pyplot as plt

        if len(self.coords) == 0:
            raise ValueError(
                "No coordinates found, make sure you click "
                "on the image that is first shown."
            )
        f, ax = plt.subplots()
        ax.imshow(self.imdata, extent=(0, self.xmax, 0, self.ymax), **kwargs)
        xlim, ylim = [ax.get_xlim(), ax.get_ylim()]
        xcoords, ycoords = zip(*self.coords)
        ax.scatter(xcoords, ycoords, c="#ff0000")
        ann_text = np.arange(len(self.coords)).astype(str)
        for txt, coord in zip(ann_text, self.coords):
            ax.annotate(txt, coord, fontsize=20, color="#ff0000")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt_show()

    def to_layout(self, **kwargs):
        """Turn coordinates into an MNE Layout object.

        Normalizes by the image you used to generate clicks

        Parameters
        ----------
        **kwargs : dict
            Arguments are passed to generate_2d_layout.

        Returns
        -------
        layout : instance of Layout
            The layout.
        """
        from ..channels.layout import generate_2d_layout

        coords = np.array(self.coords)
        lt = generate_2d_layout(coords, bg_image=self.imdata, **kwargs)
        return lt


def _fake_click(fig, ax, point, xform="ax", button=1, kind="press", key=None):
    """Fake a click at a relative point within axes."""
    from matplotlib import backend_bases

    if xform == "ax":
        x, y = ax.transAxes.transform_point(point)
    elif xform == "data":
        x, y = ax.transData.transform_point(point)
    else:
        assert xform == "pix"
        x, y = point
    if kind in ("press", "release"):
        kind = f"button_{kind}_event"
    else:
        assert kind == "motion"
        kind = "motion_notify_event"
        button = None
    logger.debug(f"Faking {kind} @ ({x}, {y}) with button={button} and key={key}")
    fig.canvas.callbacks.process(
        kind,
        backend_bases.MouseEvent(
            name=kind, canvas=fig.canvas, x=x, y=y, button=button, key=key
        ),
    )


def _fake_keypress(fig, key):
    from matplotlib import backend_bases

    fig.canvas.callbacks.process(
        "key_press_event",
        backend_bases.KeyEvent(name="key_press_event", canvas=fig.canvas, key=key),
    )


def _fake_scroll(fig, x, y, step):
    from matplotlib import backend_bases

    button = "up" if step >= 0 else "down"
    fig.canvas.callbacks.process(
        "scroll_event",
        backend_bases.MouseEvent(
            name="scroll_event", canvas=fig.canvas, x=x, y=y, step=step, button=button
        ),
    )


def add_background_image(fig, im, set_ratios=None):
    """Add a background image to a plot.

    Adds the image specified in ``im`` to the
    figure ``fig``. This is generally meant to
    be done with topo plots, though it could work
    for any plot.

    .. note:: This modifies the figure and/or axes in place.

    Parameters
    ----------
    fig : Figure
        The figure you wish to add a bg image to.
    im : array, shape (M, N, {3, 4})
        A background image for the figure. This must be a valid input to
        `matplotlib.pyplot.imshow`. Defaults to None.
    set_ratios : None | str
        Set the aspect ratio of any axes in fig
        to the value in set_ratios. Defaults to None,
        which does nothing to axes.

    Returns
    -------
    ax_im : instance of Axes
        Axes created corresponding to the image you added.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    if im is None:
        # Don't do anything and return nothing
        return None
    if set_ratios is not None:
        for ax in fig.axes:
            ax.set_aspect(set_ratios)

    ax_im = fig.add_axes([0, 0, 1, 1], label="background")
    ax_im.imshow(im, aspect="auto")
    ax_im.set_zorder(-1)
    return ax_im


def _find_peaks(evoked, npeaks):
    """Find peaks from evoked data.

    Returns ``npeaks`` biggest peaks as a list of time points.
    """
    gfp = evoked.data.std(axis=0)
    order = len(evoked.times) // 30
    if order < 1:
        order = 1
    peaks = argrelmax(gfp, order=order, axis=0)[0]
    if len(peaks) > npeaks:
        max_indices = np.argsort(gfp[peaks])[-npeaks:]
        peaks = np.sort(peaks[max_indices])
    times = evoked.times[peaks]
    if len(times) == 0:
        times = [evoked.times[gfp.argmax()]]
    return times


def _process_times(inst, use_times, n_peaks=None, few=False):
    """Return a list of times for topomaps."""
    if isinstance(use_times, str):
        if use_times == "interactive":
            use_times, n_peaks = "peaks", 1
        if use_times == "peaks":
            if n_peaks is None:
                n_peaks = min(3 if few else 7, len(inst.times))
            use_times = _find_peaks(inst, n_peaks)
        elif use_times == "auto":
            if n_peaks is None:
                n_peaks = min(5 if few else 10, len(use_times))
            use_times = np.linspace(inst.times[0], inst.times[-1], n_peaks)
        else:
            raise ValueError(
                "Got an unrecognized method for `times`. Only "
                "'peaks', 'auto' and 'interactive' are supported "
                "(or directly passing numbers)."
            )
    elif np.isscalar(use_times):
        use_times = [use_times]

    use_times = np.array(use_times, float)

    if use_times.ndim != 1:
        raise ValueError(f"times must be 1D, got {use_times.ndim} dimensions")

    if len(use_times) > 25:
        warn("More than 25 topomaps plots requested. This might take a while.")

    return use_times


@verbose
def plot_sensors(
    info,
    kind="topomap",
    ch_type=None,
    title=None,
    show_names=False,
    ch_groups=None,
    to_sphere=True,
    axes=None,
    block=False,
    show=True,
    sphere=None,
    pointsize=None,
    linewidth=2,
    *,
    cmap=None,
    verbose=None,
):
    """Plot sensors positions.

    Parameters
    ----------
    %(info_not_none)s
    kind : str
        Whether to plot the sensors as 3d, topomap or as an interactive
        sensor selection dialog. Available options ``'topomap'``, ``'3d'``,
        ``'select'``. If ``'select'``, a set of channels can be selected
        interactively by using lasso selector or clicking while holding control
        key. The selected channels are returned along with the figure instance.
        Defaults to ``'topomap'``.
    ch_type : None | str
        The channel type to plot. Available options ``'mag'``, ``'grad'``,
        ``'eeg'``, ``'seeg'``, ``'dbs'``, ``'ecog'``, ``'all'``. If ``'all'``,
        all the available mag, grad, eeg, seeg, dbs and ecog channels are
        plotted. If None (default), then channels are chosen in the order given
        above.
    title : str | None
        Title for the figure. If None (default), equals to
        ``'Sensor positions (%%s)' %% ch_type``.
    show_names : bool | array of str
        Whether to display all channel names. If an array, only the channel
        names in the array are shown. Defaults to False.
    ch_groups : 'position' | list of list | None
        Channel groups for coloring the sensors. If None (default), default
        coloring scheme is used. If 'position', the sensors are divided
        into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
        array, the channels are divided by picks given in the array. Also
        accepts a list of lists to allow channel groups of the same or
        different sizes.

        .. versionadded:: 0.13.0
    to_sphere : bool
        Whether to project the 3d locations to a sphere. When False, the
        sensor array appears similar as to looking downwards straight above the
        subject's head. Has no effect when ``kind='3d'``. Defaults to True.

        .. versionadded:: 0.14.0
    %(axes_montage)s

        .. versionadded:: 0.13.0
    block : bool
        Whether to halt program execution until the figure is closed. Defaults
        to False.

        .. versionadded:: 0.13.0
    show : bool
        Show figure if True. Defaults to True.
    %(sphere_topomap_auto)s
    pointsize : float | None
        The size of the points. If None (default), will bet set to ``75`` if
        ``kind='3d'``, or ``25`` otherwise.
    linewidth : float
        The width of the outline. If ``0``, the outline will not be drawn.
    cmap : str | instance of matplotlib.colors.Colormap | None
        Colormap for coloring ch_groups. Has effect only when ``ch_groups``
        is list of list. If None, set to ``matplotlib.rcParams["image.cmap"]``.
        Defaults to None.
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure containing the sensor topography.
    selection : list
        A list of selected channels. Only returned if ``kind=='select'``.

    See Also
    --------
    mne.viz.plot_layout

    Notes
    -----
    This function plots the sensor locations from the info structure using
    matplotlib. For drawing the sensors using PyVista see
    :func:`mne.viz.plot_alignment`.

    .. versionadded:: 0.12.0
    """
    from .evoked import _rgb

    _check_option("kind", kind, ["topomap", "3d", "select"])
    if axes is not None:
        from matplotlib.axes import Axes
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        if kind == "3d":
            _validate_type(axes, Axes3D, "axes", extra="when 'kind' is '3d'")
        elif kind in ("topomap", "select"):
            _validate_type(
                axes, Axes, "axes", extra="when 'kind' is 'topomap' or 'select'"
            )
            if isinstance(axes, Axes3D):
                raise TypeError(
                    "axes must be an instance of Axes when 'kind' is "
                    f"'topomap' or 'select', got {type(axes)} instead."
                )
    _validate_type(info, Info, "info")
    ch_indices = channel_indices_by_type(info)
    allowed_types = _DATA_CH_TYPES_SPLIT
    if ch_type is None:
        for this_type in allowed_types:
            if _contains_ch_type(info, this_type):
                ch_type = this_type
                break
        picks = ch_indices[ch_type]
    elif ch_type == "all":
        picks = list()
        for this_type in allowed_types:
            picks += ch_indices[this_type]
    elif ch_type in allowed_types:
        picks = ch_indices[ch_type]
    else:
        raise ValueError(f"ch_type must be one of {allowed_types} not {ch_type}!")

    if len(picks) == 0:
        raise ValueError(f"Could not find any channels of type {ch_type}.")

    if not _check_ch_locs(info=info, picks=picks):
        raise RuntimeError("No valid channel positions found")

    dev_head_t = info["dev_head_t"]
    chs = [info["chs"][pick] for pick in picks]
    pos = np.empty((len(chs), 3))
    for ci, ch in enumerate(chs):
        pos[ci] = ch["loc"][:3]
        if ch["coord_frame"] == FIFF.FIFFV_COORD_DEVICE:
            if dev_head_t is None:
                warn(
                    "dev_head_t is None, transforming MEG sensors to head "
                    "coordinate frame using identity transform"
                )
                dev_head_t = np.eye(4)
            pos[ci] = apply_trans(dev_head_t, pos[ci])
    del dev_head_t

    ch_names = np.array([ch["ch_name"] for ch in chs])
    bads = [idx for idx, name in enumerate(ch_names) if name in info["bads"]]
    _validate_type(ch_groups, (list, np.ndarray, str, None), "ch_groups")
    if ch_groups is None:
        def_colors = _handle_default("color")
        colors = [
            "red" if i in bads else def_colors[channel_type(info, pick)]
            for i, pick in enumerate(picks)
        ]
    else:
        if isinstance(ch_groups, str):
            _check_option(
                "ch_groups", ch_groups, ["position", "selection"], extra="when str"
            )
            # Avoid circular import
            from ..channels import (
                _EEG_SELECTIONS,
                _SELECTIONS,
                _divide_to_regions,
                read_vectorview_selection,
            )

            if ch_groups == "position":
                ch_groups = _divide_to_regions(info, add_stim=False)
                ch_groups = list(ch_groups.values())
            else:
                ch_groups, color_vals = list(), list()
                for selection in _SELECTIONS + _EEG_SELECTIONS:
                    channels = pick_channels(
                        info["ch_names"],
                        read_vectorview_selection(selection, info=info),
                        ordered=False,
                    )
                    ch_groups.append(channels)
            color_vals = np.ones((len(ch_groups), 4))
            for idx, ch_group in enumerate(ch_groups):
                color_picks = [
                    np.where(picks == ch)[0][0] for ch in ch_group if ch in picks
                ]
                if len(color_picks) == 0:
                    continue
                x, y, z = pos[color_picks].T
                color = np.mean(_rgb(x, y, z), axis=0)
                color_vals[idx, :3] = color  # mean of spatial color
        else:  # array-like
            cmap = _get_cmap(cmap)
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [cmap(colors[i]) for i in range(len(ch_groups))]
        colors = np.zeros((len(picks), 4))
        for pick_idx, pick in enumerate(picks):
            for ind, value in enumerate(ch_groups):
                if pick in value:
                    colors[pick_idx] = color_vals[ind]
                    break
    title = f"Sensor positions ({ch_type})" if title is None else title
    fig = _plot_sensors_2d(
        pos,
        info,
        picks,
        colors,
        bads,
        ch_names,
        title,
        show_names,
        axes,
        show,
        kind,
        block,
        to_sphere,
        sphere,
        pointsize=pointsize,
        linewidth=linewidth,
    )
    if kind == "select":
        return fig, fig.lasso.selection
    return fig


def _onpick_sensor(event, fig, ax, pos, ch_names, show_names):
    """Pick a channel in plot_sensors."""
    if event.mouseevent.inaxes != ax:
        return

    if event.mouseevent.key == "control" and fig.lasso is not None:
        for ind in event.ind:
            fig.lasso.select_one(ind)

        return
    if show_names:
        return  # channel names already visible
    ind = event.ind[0]  # Just take the first sensor.
    ch_name = ch_names[ind]

    this_pos = pos[ind]

    # XXX: Bug in matplotlib won't allow setting the position of existing
    # text item, so we create a new one.
    ax.texts[0].remove()
    if len(this_pos) == 3:
        ax.text(this_pos[0], this_pos[1], this_pos[2], ch_name)
    else:
        ax.text(this_pos[0], this_pos[1], ch_name)
    fig.canvas.draw()


def _close_event(event, fig):
    """Listen for sensor plotter close event."""
    if getattr(fig, "lasso", None) is not None:
        fig.lasso.disconnect()


def _plot_sensors_2d(
    pos,
    info,
    picks,
    colors,
    bads,
    ch_names,
    title,
    show_names,
    ax,
    show,
    kind,
    block,
    to_sphere,
    sphere,
    pointsize=None,
    linewidth=2,
):
    """Plot sensors."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 analysis:ignore

    from .topomap import _draw_outlines, _get_pos_outlines

    ch_names = [str(ch_name) for ch_name in ch_names]
    sphere = _check_sphere(sphere, info)

    edgecolors = np.repeat(rcParams["axes.edgecolor"], len(colors))
    edgecolors[bads] = "red"
    axes_was_none = ax is None
    if axes_was_none:
        subplot_kw = dict()
        if kind == "3d":
            subplot_kw.update(projection="3d")
        fig, ax = plt.subplots(
            1,
            figsize=(max(rcParams["figure.figsize"]),) * 2,
            subplot_kw=subplot_kw,
            layout="constrained",
        )
    else:
        fig = ax.get_figure()

    if kind == "3d":
        pointsize = 75 if pointsize is None else pointsize
        ax.text(0, 0, 0, "", zorder=1)

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            picker=True,
            c=colors,
            s=pointsize,
            edgecolor=edgecolors,
            linewidth=linewidth,
        )

        ax.azim = 90
        ax.elev = 0
        ax.xaxis.set_label_text("x (m)")
        ax.yaxis.set_label_text("y (m)")
        ax.zaxis.set_label_text("z (m)")
    else:  # kind in 'select', 'topomap'
        pointsize = 25 if pointsize is None else pointsize
        ax.text(0, 0, "", zorder=1)

        pos, outlines = _get_pos_outlines(info, picks, sphere, to_sphere=to_sphere)
        _draw_outlines(ax, outlines)
        pts = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            picker=True,
            clip_on=False,
            c=colors,
            edgecolors=edgecolors,
            s=pointsize,
            lw=linewidth,
        )
        if kind == "select":
            fig.lasso = SelectFromCollection(ax, pts, ch_names)
        else:
            fig.lasso = None

        # Equal aspect for 3D looks bad, so only use for 2D
        ax.set(aspect="equal")
        ax.axis("off")  # remove border around figure
    del sphere

    connect_picker = True
    if show_names:
        if isinstance(show_names, list | np.ndarray):  # only given channels
            indices = [list(ch_names).index(name) for name in show_names]
        else:  # all channels
            indices = range(len(pos))
        for idx in indices:
            this_pos = pos[idx]
            if kind == "3d":
                ax.text(this_pos[0], this_pos[1], this_pos[2], ch_names[idx])
            else:
                ax.text(
                    this_pos[0] + 0.0025,
                    this_pos[1],
                    ch_names[idx],
                    ha="left",
                    va="center",
                )
        connect_picker = kind == "select"
        # make sure no names go off the edge of the canvas
        xmin, ymin, xmax, ymax = fig.get_window_extent().bounds
    if connect_picker:
        picker = partial(
            _onpick_sensor,
            fig=fig,
            ax=ax,
            pos=pos,
            ch_names=ch_names,
            show_names=show_names,
        )
        fig.canvas.mpl_connect("pick_event", picker)
    if axes_was_none:
        _set_window_title(fig, title)
    closed = partial(_close_event, fig=fig)
    fig.canvas.mpl_connect("close_event", closed)
    plt_show(show, block=block)
    return fig


def _compute_scalings(scalings, inst, remove_dc=False, duration=10):
    """Compute scalings for each channel type automatically.

    Parameters
    ----------
    scalings : dict
        The scalings for each channel type. If any values are
        'auto', this will automatically compute a reasonable
        scaling for that channel type. Any values that aren't
        'auto' will not be changed.
    inst : instance of Raw or Epochs
        The data for which you want to compute scalings. If data
        is not preloaded, this will read a subset of times / epochs
        up to 100mb in size in order to compute scalings.
    remove_dc : bool
        Whether to remove the mean (DC) before calculating the scalings. If
        True, the mean will be computed and subtracted for short epochs in
        order to compensate not only for global mean offset, but also for slow
        drifts in the signals.
    duration : float
        If remove_dc is True, the mean will be computed and subtracted on
        segments of length ``duration`` seconds.

    Returns
    -------
    scalings : dict
        A scalings dictionary with updated values
    """
    from ..epochs import BaseEpochs
    from ..io import BaseRaw

    scalings = _handle_default("scalings_plot_raw", scalings)
    if not isinstance(inst, BaseRaw | BaseEpochs):
        raise ValueError("Must supply either Raw or Epochs")

    for key, value in scalings.items():
        if not (isinstance(value, str) and value == "auto"):
            try:
                scalings[key] = float(value)
            except Exception:
                raise ValueError(
                    f'scalings must be "auto" or float, got '
                    f"scalings[{key!r}]={value!r} which could not be "
                    f"converted to float"
                )

    # If there are no "auto" scalings, we can return early!
    if all(
        [scalings[ch_type] != "auto" for ch_type in inst.get_channel_types(unique=True)]
    ):
        return scalings

    ch_types = channel_indices_by_type(inst.info)
    ch_types = {i_type: i_ixs for i_type, i_ixs in ch_types.items() if len(i_ixs) != 0}

    if inst.preload is False:
        if isinstance(inst, BaseRaw):
            # Load a window of data from the center up to 100mb in size
            n_times = 1e8 // (len(inst.ch_names) * 8)
            n_times = np.clip(n_times, 1, inst.n_times)
            n_secs = n_times / float(inst.info["sfreq"])
            time_middle = np.mean(inst.times)
            tmin = np.clip(time_middle - n_secs / 2.0, inst.times.min(), None)
            tmax = np.clip(time_middle + n_secs / 2.0, None, inst.times.max())
            smin, smax = (int(round(x * inst.info["sfreq"])) for x in (tmin, tmax))
            data = inst._read_segment(smin, smax)
        elif isinstance(inst, BaseEpochs):
            # Load a random subset of epochs up to 100mb in size
            n_epochs = 1e8 // (len(inst.ch_names) * len(inst.times) * 8)
            n_epochs = int(np.clip(n_epochs, 1, len(inst)))
            ixs_epochs = np.random.choice(range(len(inst)), n_epochs, False)
            inst = inst.copy()[ixs_epochs].load_data()
    else:
        data = inst._data
    if isinstance(inst, BaseEpochs):
        data = inst._data.swapaxes(0, 1).reshape([len(inst.ch_names), -1])
    # Iterate through ch types and update scaling if ' auto'
    for key, value in scalings.items():
        if key not in ch_types or value != "auto":
            continue
        this_data = data[ch_types[key]]
        if remove_dc and (this_data.shape[1] / inst.info["sfreq"] >= duration):
            length = int(duration * inst.info["sfreq"])  # segment length
            # truncate data so that we can divide into segments of equal length
            this_data = this_data[:, : this_data.shape[1] // length * length]
            shape = this_data.shape  # original shape
            this_data = this_data.T.reshape(-1, length, shape[0])  # segment
            this_data -= np.nanmean(this_data, 0)  # subtract segment means
            this_data = this_data.T.reshape(shape)  # reshape into original
        this_data = this_data.ravel()
        this_data = this_data[np.isfinite(this_data)]
        if this_data.size:
            iqr = np.diff(np.percentile(this_data, [25, 75]))[0]
        else:
            iqr = 1.0
        scalings[key] = iqr
    return scalings


def _setup_cmap(cmap, n_axes=1, norm=False):
    """Set color map interactivity."""
    if cmap == "interactive":
        cmap = ("Reds" if norm else "RdBu_r", True)
    elif not isinstance(cmap, tuple):
        if cmap is None:
            cmap = "Reds" if norm else "RdBu_r"
        cmap = (cmap, False if n_axes > 2 else True)
    return cmap


def _prepare_joint_axes(n_maps, figsize=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(2, n_maps, height_ratios=[1, 2], figure=fig)
    map_ax = [fig.add_subplot(gs[0, x]) for x in range(n_maps)]  # first row
    main_ax = fig.add_subplot(gs[1, :])  # second row
    return fig, main_ax, map_ax


class DraggableColorbar:
    """Enable interactive colorbar.

    See http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
    """  # noqa: E501

    def __init__(self, cbar, mappable, kind, ch_type):
        import matplotlib.pyplot as plt

        self.cbar = cbar
        self.mappable = mappable
        self.kind = kind
        self.ch_type = ch_type
        self.fig = self.cbar.ax.figure
        self.press = None
        self.cycle = sorted(
            [i for i in dir(plt.cm) if hasattr(getattr(plt.cm, i), "N")]
        )
        self.cycle += [mappable.get_cmap().name]
        self.index = self.cycle.index(mappable.get_cmap().name)
        self.lims = (self.cbar.norm.vmin, self.cbar.norm.vmax)
        self.connect()

        @_auto_weakref
        def _on_colormap_range(event):
            return self._on_colormap_range(event)

        subscribe(self.fig, "colormap_range", _on_colormap_range)

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.cbar.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.cbar.ax.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.cbar.ax.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.keypress = self.cbar.ax.figure.canvas.mpl_connect(
            "key_press_event", self.key_press
        )
        self.scroll = self.cbar.ax.figure.canvas.mpl_connect(
            "scroll_event", self.on_scroll
        )

    def on_press(self, event):
        """Handle button press."""
        if event.inaxes != self.cbar.ax:
            return
        self.press = event.y

    def key_press(self, event):
        """Handle key press."""
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.key == "down":
            self.index += 1
        elif event.key == "up":
            self.index -= 1
        elif event.key == " ":  # space key resets scale
            self.cbar.norm.vmin = self.lims[0]
            self.cbar.norm.vmax = self.lims[1]
        elif event.key == "+":
            self.cbar.norm.vmin -= (perc * scale) * -1
            self.cbar.norm.vmax += (perc * scale) * -1
        elif event.key == "-":
            self.cbar.norm.vmin -= (perc * scale) * 1
            self.cbar.norm.vmax += (perc * scale) * 1
        elif event.key == "pageup":
            self.cbar.norm.vmin -= (perc * scale) * 1
            self.cbar.norm.vmax -= (perc * scale) * 1
        elif event.key == "pagedown":
            self.cbar.norm.vmin -= (perc * scale) * -1
            self.cbar.norm.vmax -= (perc * scale) * -1
        else:
            return
        if self.index < 0:
            self.index = len(self.cycle) - 1
        elif self.index >= len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.mappable.set_cmap(cmap)
        self.cbar.ax.figure.draw_without_rendering()
        self.mappable.set_cmap(cmap)
        self._publish()

    def on_motion(self, event):
        """Handle mouse movements."""
        if self.press is None:
            return
        if event.inaxes != self.cbar.ax:
            return
        yprev = self.press
        dy = event.y - yprev
        self.press = event.y
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button == 1:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax -= (perc * scale) * np.sign(dy)
        elif event.button == 3:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax += (perc * scale) * np.sign(dy)
        self._publish()

    def on_release(self, event):
        """Handle release."""
        self.press = None
        self._update()

    def on_scroll(self, event):
        """Handle scroll."""
        scale = 1.1 if event.step < 0 else 1.0 / 1.1
        self.cbar.norm.vmin *= scale
        self.cbar.norm.vmax *= scale
        self._publish()

    def _on_colormap_range(self, event):
        if event.kind != self.kind or event.ch_type != self.ch_type:
            return
        if event.fmin is not None:
            self.cbar.norm.vmin = event.fmin
        if event.fmax is not None:
            self.cbar.norm.vmax = event.fmax
        if event.cmap is not None:
            self.cbar.mappable.set_cmap(event.cmap)
            self.mappable.set_cmap(event.cmap)
        self._update()

    def _publish(self):
        publish(
            self.fig,
            ColormapRange(
                kind=self.kind,
                ch_type=self.ch_type,
                fmin=self.cbar.norm.vmin,
                fmax=self.cbar.norm.vmax,
                cmap=self.mappable.get_cmap(),
            ),
        )

    def _update(self):
        from matplotlib.ticker import AutoLocator

        self.cbar.set_ticks(AutoLocator())
        self.cbar.update_ticks()
        self.cbar.ax.figure.draw_without_rendering()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.ax.figure.canvas.draw()


class SelectFromCollection:
    """Select channels from a matplotlib collection using ``LassoSelector``.

    Selected channels are saved in the ``selection`` attribute. This tool
    highlights selected points by fading other points out (i.e., reducing their
    alpha values).

    Parameters
    ----------
    ax : instance of Axes
        Axes to interact with.
    collection : instance of matplotlib collection
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to ``alpha_other``.
        Defaults to 0.3.
    linewidth_other : float
        Linewidth to use for non-selected sensors. Default is 1.

    Notes
    -----
    This tool selects collection objects based on their *origins*
    (i.e., ``offsets``). Calls all callbacks in self.callbacks when selection
    is ready.
    """

    def __init__(
        self,
        ax,
        collection,
        ch_names,
        alpha_other=0.5,
        linewidth_other=0.5,
        alpha_selected=1,
        linewidth_selected=1,
    ):
        from matplotlib.widgets import LassoSelector

        self.canvas = ax.figure.canvas
        self.collection = collection
        self.ch_names = ch_names
        self.alpha_other = alpha_other
        self.linewidth_other = linewidth_other
        self.alpha_selected = alpha_selected
        self.linewidth_selected = linewidth_selected

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        self.ec = collection.get_edgecolors()
        self.lw = collection.get_linewidths()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)
            self.ec = np.tile(self.ec, self.Npts).reshape(self.Npts, -1)
        self.fc[:, -1] = self.alpha_other  # deselect in the beginning
        self.ec[:, -1] = self.alpha_other
        self.lw = np.full(self.Npts, self.linewidth_other)

        self.lasso = LassoSelector(
            ax, onselect=self.on_select, props=dict(color="red", linewidth=0.5)
        )
        self.selection = list()
        self.callbacks = list()

    def on_select(self, verts):
        """Select a subset from the collection."""
        from matplotlib.path import Path

        if len(verts) <= 3:  # Seems to be a good way to exclude single clicks.
            return

        path = Path(verts)
        inds = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        if self.canvas._key == "control":  # Appending selection.
            sels = [np.where(self.ch_names == c)[0][0] for c in self.selection]
            inters = set(inds) - set(sels)
            inds = list(inters.union(set(sels) - set(inds)))

        self.selection[:] = np.array(self.ch_names)[inds].tolist()
        self.style_sensors(inds)
        self.notify()

    def select_one(self, ind):
        """Select or deselect one sensor."""
        ch_name = self.ch_names[ind]
        if ch_name in self.selection:
            sel_ind = self.selection.index(ch_name)
            self.selection.pop(sel_ind)
        else:
            self.selection.append(ch_name)
        inds = np.isin(self.ch_names, self.selection).nonzero()[0]
        self.style_sensors(inds)
        self.notify()

    def notify(self):
        """Notify listeners that a selection has been made."""
        for callback in self.callbacks:
            callback()

    def select_many(self, inds):
        """Select many sensors using indices (for predefined selections)."""
        self.selection[:] = np.array(self.ch_names)[inds].tolist()
        self.style_sensors(inds)

    def style_sensors(self, inds):
        """Style selected sensors as "active"."""
        # reset
        self.fc[:, -1] = self.alpha_other
        self.ec[:, -1] = self.alpha_other / 2
        self.lw[:] = self.linewidth_other
        # style sensors at `inds`
        self.fc[inds, -1] = self.alpha_selected
        self.ec[inds, -1] = self.alpha_selected
        self.lw[inds] = self.linewidth_selected
        self.collection.set_facecolors(self.fc)
        self.collection.set_edgecolors(self.ec)
        self.collection.set_linewidths(self.lw)
        self.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the lasso selector."""
        self.lasso.disconnect_events()
        self.fc[:, -1] = self.alpha_selected
        self.ec[:, -1] = self.alpha_selected
        self.collection.set_facecolors(self.fc)
        self.collection.set_edgecolors(self.ec)
        self.canvas.draw_idle()


def _get_color_list(annotations=False):
    """Get the current color list from matplotlib rcParams.

    Parameters
    ----------
    annotations : boolean
        Has no influence on the function if false. If true, check if color
        "red" (#ff0000) is in the cycle and remove it.

    Returns
    -------
    colors : list
    """
    from matplotlib import rcParams

    color_cycle = rcParams.get("axes.prop_cycle")
    colors = color_cycle.by_key()["color"]

    # If we want annotations, red is reserved ... remove if present. This
    # checks for the reddish color in MPL dark background style, normal style,
    # and MPL "red", and defaults to the last of those if none are present
    for red in ("#fa8174", "#d62728", "#ff0000"):
        if annotations and red in colors:
            colors.remove(red)
            break

    return (colors, red) if annotations else colors


def _merge_annotations(start, stop, description, annotations, current=()):
    """Handle drawn annotations."""
    ends = annotations.onset + annotations.duration
    idx = np.intersect1d(
        np.where(ends >= start)[0], np.where(annotations.onset <= stop)[0]
    )
    idx = np.intersect1d(idx, np.where(annotations.description == description)[0])
    new_idx = np.setdiff1d(idx, current)  # don't include modified annotation
    end = max(
        np.append((annotations.onset[new_idx] + annotations.duration[new_idx]), stop)
    )
    onset = min(np.append(annotations.onset[new_idx], start))
    duration = end - onset
    annotations.delete(idx)
    annotations.append(onset, duration, description)


class DraggableLine:
    """Custom matplotlib line for moving around by drag and drop.

    Parameters
    ----------
    line : instance of matplotlib Line2D
        Line to add interactivity to.
    callback : function
        Callback to call when line is released.
    """

    def __init__(self, line, modify_callback, drag_callback):
        self.line = line
        self.press = None
        self.x0 = line.get_xdata()[0]
        self.modify_callback = modify_callback
        self.drag_callback = drag_callback
        self.cidpress = self.line.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def set_x(self, x):
        """Repoisition the line."""
        self.line.set_xdata([x, x])
        self.x0 = x

    def on_press(self, event):
        """Store button press if on top of the line."""
        if event.inaxes != self.line.axes or not self.line.contains(event)[0]:
            return
        x0 = self.line.get_xdata()
        y0 = self.line.get_ydata()
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        """Move the line on drag."""
        if self.press is None:
            return
        if event.inaxes != self.line.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        self.line.set_xdata(x0 + dx)
        self.drag_callback((x0 + dx)[0])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        """Handle release."""
        if event.inaxes != self.line.axes or self.press is None:
            return
        self.press = None
        self.line.figure.canvas.draw()
        self.modify_callback(self.x0, event.xdata)
        self.x0 = event.xdata

    def remove(self):
        """Remove the line."""
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)
        self.line.remove()


def _setup_ax_spines(
    axes,
    vlines,
    xmin,
    xmax,
    ymin,
    ymax,
    invert_y=False,
    unit=None,
    truncate_xaxis=True,
    truncate_yaxis=True,
    skip_axlabel=False,
    hline=True,
    time_unit="s",
):
    # don't show zero line if it coincides with x-axis (even if hline=True)
    if hline and ymin != 0.0:
        axes.spines["top"].set_position("zero")
    else:
        axes.spines["top"].set_visible(False)
    # the axes can become very small with topo plotting. This prevents the
    # x-axis from shrinking to length zero if truncate_xaxis=True, by adding
    # new ticks that are nice round numbers close to (but less extreme than)
    # xmin and xmax
    vlines = [] if vlines is None else vlines
    xticks = _trim_ticks(axes.get_xticks(), round(xmin, 2), round(xmax, 2))
    xticks = np.array(sorted(set([x for x in xticks] + vlines)))
    if len(xticks) < 2:

        def log_fix(tval):
            exp = np.log10(np.abs(tval))
            return np.sign(tval) * 10 ** (np.fix(exp) - (exp < 0))

        xlims = np.array([xmin, xmax])
        temp_ticks = log_fix(xlims)
        closer_idx = np.argmin(np.abs(xlims - temp_ticks))
        further_idx = np.argmax(np.abs(xlims - temp_ticks))
        start_stop = [temp_ticks[closer_idx], xlims[further_idx]]
        step = np.sign(np.diff(start_stop)).item() * np.max(np.abs(temp_ticks))
        tts = np.arange(*start_stop, step)
        xticks = np.array(sorted(xticks + [tts[0], tts[-1]]))
    axes.set_xticks(xticks)
    # y-axis is simpler
    yticks = _trim_ticks(axes.get_yticks(), ymin, ymax)
    axes.set_yticks(yticks)
    # truncation case 1: truncate both
    if truncate_xaxis and truncate_yaxis:
        axes.spines["bottom"].set_bounds(*xticks[[0, -1]])
        axes.spines["left"].set_bounds(*yticks[[0, -1]])
    # case 2: truncate only x (only right side; connect to y at left)
    elif truncate_xaxis:
        xbounds = np.array(axes.get_xlim())
        xbounds[1] = axes.get_xticks()[-1]
        axes.spines["bottom"].set_bounds(*xbounds)
    # case 3: truncate only y (only top; connect to x at bottom)
    elif truncate_yaxis:
        ybounds = np.array(axes.get_ylim())
        if invert_y:
            ybounds[0] = axes.get_yticks()[0]
        else:
            ybounds[1] = axes.get_yticks()[-1]
        axes.spines["left"].set_bounds(*ybounds)
    # handle axis labels
    if skip_axlabel:
        axes.set_yticklabels([""] * len(yticks))
        axes.set_xticklabels([""] * len(xticks))
    else:
        if unit is not None:
            axes.set_ylabel(unit, rotation=90)
        axes.set_xlabel(f"Time ({time_unit})")
    # plot vertical lines
    if vlines:
        _ymin, _ymax = axes.get_ylim()
        axes.vlines(
            vlines, _ymax, _ymin, linestyles="--", colors="k", linewidth=1.0, zorder=1
        )
    # invert?
    if invert_y:
        axes.invert_yaxis()
    # changes we always make:
    axes.tick_params(direction="out")
    axes.tick_params(right=False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_zorder(0)


def _handle_decim(info, decim, lowpass):
    """Handle decim parameter for plotters."""
    if isinstance(decim, str) and decim == "auto":
        lp = info["sfreq"] if info["lowpass"] is None else info["lowpass"]
        lp = min(lp, info["sfreq"] if lowpass is None else lowpass)
        with info._unlock():
            info["lowpass"] = lp
        decim = max(int(info["sfreq"] / (lp * 3) + 1e-6), 1)
    decim = _ensure_int(decim, "decim", must_be='an int or "auto"')
    if decim <= 0:
        raise ValueError(f'decim must be "auto" or a positive integer, got {decim}')
    decim = _check_decim(info, decim, 0)[0]
    data_picks = _pick_data_channels(info, exclude=())
    return decim, data_picks


def _setup_plot_projector(info, noise_cov, proj=True, use_noise_cov=True, nave=1):
    from ..cov import compute_whitener

    projector = np.eye(len(info["ch_names"]))
    whitened_ch_names = []
    if noise_cov is not None and use_noise_cov:
        # any channels in noise_cov['bads'] but not in info['bads'] get
        # set to nan, which means that they are not plotted.
        data_picks = _pick_data_channels(info, with_ref_meg=False, exclude=())
        data_names = {info["ch_names"][pick] for pick in data_picks}
        # these can be toggled by the user
        bad_names = set(info["bads"])
        # these can't in standard pipelines be enabled (we always take the
        # union), so pretend they're not in cov at all
        cov_names = (set(noise_cov["names"]) & set(info["ch_names"])) - set(
            noise_cov["bads"]
        )
        # Actually compute the whitener only using the difference
        whiten_names = cov_names - bad_names
        whiten_picks = pick_channels(info["ch_names"], whiten_names, ordered=True)
        whiten_info = pick_info(info, whiten_picks)
        rank = _triage_rank_sss(whiten_info, [noise_cov])[1][0]
        whitener, whitened_ch_names = compute_whitener(
            noise_cov, whiten_info, rank=rank, verbose=False
        )
        whitener *= np.sqrt(nave)  # proper scaling for Evoked data
        assert set(whitened_ch_names) == whiten_names
        projector[whiten_picks, whiten_picks[:, np.newaxis]] = whitener
        # Now we need to change the set of "whitened" channels to include
        # all data channel names so that they are properly italicized.
        whitened_ch_names = data_names
        # We would need to set "bad_picks" to identity to show the traces
        # (but in gray), but here we don't need to because "projector"
        # starts out as identity. So all that is left to do is take any
        # *good* data channels that are not in the noise cov to be NaN
        nan_names = data_names - (bad_names | cov_names)
        # XXX conditional necessary because of annoying behavior of
        # pick_channels where an empty list means "all"!
        if len(nan_names) > 0:
            nan_picks = pick_channels(info["ch_names"], nan_names)
            projector[nan_picks] = np.nan
    elif proj:
        projector, _ = setup_proj(info, add_eeg_ref=False, verbose=False)
    return projector, whitened_ch_names


def _check_sss(info):
    """Check SSS history in info."""
    ch_used = [ch for ch in _DATA_CH_TYPES_SPLIT if _contains_ch_type(info, ch)]
    has_meg = "mag" in ch_used and "grad" in ch_used
    has_sss = (
        has_meg
        and len(info["proc_history"]) > 0
        and info["proc_history"][0].get("max_info") is not None
    )
    return ch_used, has_meg, has_sss


def _triage_rank_sss(info, covs, rank=None, scalings=None):
    rank = dict() if rank is None else rank
    scalings = _handle_default("scalings_cov_rank", scalings)

    # Only look at good channels
    picks = _pick_data_channels(info, with_ref_meg=False, exclude="bads")
    info = pick_info(info, picks)
    ch_used, has_meg, has_sss = _check_sss(info)
    if has_sss:
        if "mag" in rank or "grad" in rank:
            raise ValueError(
                'When using SSS, pass "meg" to set the rank '
                '(separate rank values for "mag" or "grad" are '
                "meaningless)."
            )
    elif "meg" in rank:
        raise ValueError(
            "When not using SSS, pass separate rank values "
            'for "mag" and "grad" (do not use "meg").'
        )

    picks_list = _picks_by_type(info, meg_combined=has_sss)
    if has_sss:
        # reduce ch_used to combined mag grad
        ch_used = list(zip(*picks_list))[0]
    # order pick list by ch_used (required for compat with plot_evoked)
    picks_list = [x for x, y in sorted(zip(picks_list, ch_used))]
    n_ch_used = len(ch_used)

    # make sure we use the same rank estimates for GFP and whitening

    picks_list2 = [k for k in picks_list]
    # add meg picks if needed.
    if has_meg:
        # append ("meg", picks_meg)
        picks_list2 += _picks_by_type(info, meg_combined=True)

    rank_list = []  # rank dict for each cov
    for cov in covs:
        # We need to add the covariance projectors, compute the projector,
        # and apply it, just like we will do in prepare_noise_cov, otherwise
        # we risk the rank estimates being incorrect (i.e., if the projectors
        # do not match).
        info_proj = info.copy()
        with info_proj._unlock():
            info_proj["projs"] += cov["projs"]
        this_rank = {}
        # assemble rank dict for this cov, such that we have meg
        for ch_type, this_picks in picks_list2:
            # if we have already estimates / values for mag/grad but not
            # a value for meg, combine grad and mag.
            if "mag" in this_rank and "grad" in this_rank and "meg" not in rank:
                this_rank["meg"] = this_rank["mag"] + this_rank["grad"]
                # and we're done here
                break
            if rank.get(ch_type) is None:
                ch_names = [info["ch_names"][pick] for pick in this_picks]
                this_C = pick_channels_cov(cov, ch_names, ordered=False)
                this_estimated_rank = compute_rank(
                    this_C, scalings=scalings, info=info_proj
                )[ch_type]
                this_rank[ch_type] = this_estimated_rank
            elif rank.get(ch_type) is not None:
                this_rank[ch_type] = rank[ch_type]

        rank_list.append(this_rank)
    return n_ch_used, rank_list, picks_list, has_sss


def _check_cov(noise_cov, info):
    """Check the noise_cov for whitening and issue an SSS warning."""
    from ..cov import _ensure_cov

    if noise_cov is None:
        return None
    noise_cov = _ensure_cov(noise_cov, name="noise_cov", verbose=False)
    if _check_sss(info)[2]:  # has_sss
        warn(
            "Data have been processed with SSS, which changes the relative "
            "scaling of magnetometers and gradiometers when viewing data "
            "whitened by a noise covariance"
        )
    return noise_cov


def _set_title_multiple_electrodes(
    title, combine, ch_names, max_chans=6, all_=False, ch_type=None
):
    """Prepare a title string for multiple electrodes."""
    if title is None:
        title = ", ".join(ch_names[:max_chans])
        ch_type = _channel_type_prettyprint.get(ch_type, ch_type)
        if ch_type is None:
            ch_type = "sensor"
        ch_type = f"{ch_type}{_pl(ch_names)}"
        if hasattr(combine, "func"):  # functools.partial
            combine = combine.func
        if callable(combine):
            combine = getattr(combine, "__name__", str(combine))
        if not isinstance(combine, str):
            combine = "Combination"
        # mean → Mean, but avoid RMS → Rms and GFP → Gfp
        if combine[0].islower():
            combine = combine.capitalize()
        if all_:
            title = f"{combine} of {len(ch_names)} {ch_type}"
        elif len(ch_names) > max_chans and combine != "gfp":
            logger.info(f"More than {max_chans} channels, truncating title ...")
            title += f", ...\n({combine} of {len(ch_names)} {ch_type})"
    return title


def _check_time_unit(time_unit, times):
    if not isinstance(time_unit, str):
        raise TypeError(f"time_unit must be str, got {type(time_unit)}")
    if time_unit == "s":
        pass
    elif time_unit == "ms":
        times = 1e3 * times
    else:
        raise ValueError(f"time_unit must be 's' or 'ms', got {time_unit!r}")
    return time_unit, times


def _plot_masked_image(
    ax,
    data,
    times,
    mask=None,
    yvals=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    ylim=None,
    mask_style="both",
    mask_alpha=0.25,
    mask_cmap="Greys",
    yscale="linear",
    cnorm=None,
):
    """Plot a potentially masked (evoked, TFR, ...) 2D image."""
    from matplotlib import ticker
    from matplotlib.colors import Normalize

    if mask_style is None and mask is not None:
        mask_style = "both"  # default
    draw_mask = mask_style in {"both", "mask"}
    draw_contour = mask_style in {"both", "contour"}
    if cmap is None:
        mask_cmap = cmap
    if cnorm is None:
        cnorm = Normalize(vmin=vmin, vmax=vmax)

    # mask param check and preparation
    if draw_mask is None:
        if mask is not None:
            draw_mask = True
        else:
            draw_mask = False
    if draw_contour is None:
        if mask is not None:
            draw_contour = True
        else:
            draw_contour = False
    if mask is None:
        if draw_mask:
            warn("`mask` is None, not masking the plot ...")
            draw_mask = False
        if draw_contour:
            warn("`mask` is None, not adding contour to the plot ...")
            draw_contour = False

    if draw_mask:
        if mask.shape != data.shape:
            raise ValueError(
                "The mask must have the same shape as the data, "
                f"i.e., {data.shape}, not {mask.shape}"
            )
        if draw_contour and yscale == "log":
            warn("Cannot draw contours with linear yscale yet ...")

    if yvals is None:  # for e.g. Evoked images
        yvals = np.arange(data.shape[0])
    # else, if TFR plot, yvals will be freqs

    # test yscale
    if yscale == "log" and not yvals[0] > 0:
        raise ValueError(
            "Using log scale for frequency axis requires all your"
            " frequencies to be positive (you cannot include"
            " the DC component (0 Hz) in the TFR)."
        )

    if len(yvals) < 2 or yvals[0] == 0:
        yscale = "linear"
    elif yscale != "linear":
        ratio = yvals[1:] / yvals[:-1]
    if yscale == "auto":
        if yvals[0] > 0 and np.allclose(ratio, ratio[0]):
            yscale = "log"
        else:
            yscale = "linear"

    if yscale == "log":  # pcolormesh for log scale
        # compute bounds between time samples
        (time_lims,) = centers_to_edges(times)
        log_yvals = np.concatenate(
            [[yvals[0] / ratio[0]], yvals, [yvals[-1] * ratio[0]]]
        )
        yval_lims = np.sqrt(log_yvals[:-1] * log_yvals[1:])

        # construct a time-yvaluency bounds grid
        time_mesh, yval_mesh = np.meshgrid(time_lims, yval_lims)

        if mask is not None:
            ax.pcolormesh(
                time_mesh, yval_mesh, data, cmap=mask_cmap, norm=cnorm, alpha=mask_alpha
            )
            im = ax.pcolormesh(
                time_mesh,
                yval_mesh,
                np.ma.masked_where(~mask, data),
                cmap=cmap,
                norm=cnorm,
                alpha=1,
            )
        else:
            im = ax.pcolormesh(time_mesh, yval_mesh, data, cmap=cmap, norm=cnorm)
        if ylim is None:
            ylim = yval_lims[[0, -1]]
        if yscale == "log":
            ax.set_yscale("log")
            ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # get rid of minor ticks
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        tick_vals = yvals[
            np.unique(np.linspace(0, len(yvals) - 1, 12).round().astype("int"))
        ]
        ax.set_yticks(tick_vals)

    else:
        # imshow for linear because the y ticks are nicer
        # and the masked areas look better
        dt = np.median(np.diff(times)) / 2.0 if len(times) > 1 else 0.1
        dy = np.median(np.diff(yvals)) / 2.0 if len(yvals) > 1 else 0.5
        extent = [times[0] - dt, times[-1] + dt, yvals[0] - dy, yvals[-1] + dy]
        im_args = dict(
            interpolation="nearest", origin="lower", extent=extent, aspect="auto"
        )
        if draw_mask:
            ax.imshow(data, alpha=mask_alpha, cmap=mask_cmap, norm=cnorm, **im_args)
            im = ax.imshow(
                np.ma.masked_where(~mask, data), cmap=cmap, norm=cnorm, **im_args
            )
        else:
            ax.imshow(data, cmap=cmap, norm=cnorm, **im_args)  # see #6481
            im = ax.imshow(data, cmap=cmap, norm=cnorm, **im_args)

        if draw_contour and np.unique(mask).size == 2:
            big_mask = np.kron(mask, np.ones((10, 10)))
            ax.contour(
                big_mask,
                colors=["k"],
                extent=extent,
                linewidths=[0.75],
                corner_mask=False,
                antialiased=False,
                levels=[0.5],
            )
        time_lims = [extent[0], extent[1]]
        if ylim is None:
            ylim = [extent[2], extent[3]]

    ax.set_xlim(time_lims[0], time_lims[-1])
    ax.set_ylim(ylim)

    if (draw_mask or draw_contour) and mask is not None:
        if mask.all():
            t_end = ", all points masked)"
        else:
            fraction = 1 - (np.float64(mask.sum()) / np.float64(mask.size))
            t_end = f", {fraction * 100:0.3g}% of points masked)"
    else:
        t_end = ")"

    return im, t_end


@fill_doc
def _make_combine_callable(
    combine,
    *,
    axis=1,
    valid=("mean", "median", "std", "gfp"),
    ch_type=None,
    keepdims=False,
):
    """Convert None or string values of ``combine`` into callables.

    Params
    ------
    combine : None | str | callable
        If callable, the callable must accept one positional input (a numpy array) and
        return an array with one fewer dimensions (the missing dimension's position is
        given by ``axis``).
    axis : int
        Axis of data array across which to combine. May vary depending on data
        context; e.g., if data are time-domain sensor traces or TFRs, continuous
        or epoched, etc.
    valid : tuple
        Valid string values for built-in combine methods
        (may vary for, e.g., combining TFRs versus time-domain signals).
    ch_type : str
        Channel type. Affects whether "gfp" is allowed as a synonym for "rms".
    keepdims : bool
        Whether to retain the singleton dimension after collapsing across it.
    """
    kwargs = dict(axis=axis, keepdims=keepdims)
    if combine is None:
        combine = _identity_function if keepdims else partial(np.squeeze, axis=axis)
    elif isinstance(combine, str):
        combine_dict = {
            key: partial(getattr(np, key), **kwargs)
            for key in valid
            if getattr(np, key, None) is not None
        }
        # marginal median that is safe for complex values:
        if "median" in valid:
            combine_dict["median"] = partial(_median_complex, axis=axis)

        # RMS and GFP; if GFP requested for MEG channels, will use RMS anyway
        def _rms(data):
            return np.sqrt((data**2).mean(**kwargs))

        def _gfp(data):
            return data.std(axis=axis, ddof=0)

        # make them play nice with _set_title_multiple_electrodes()
        _rms.__name__ = "RMS"
        _gfp.__name__ = "GFP"
        if "rms" in valid:
            combine_dict["rms"] = _rms
        if "gfp" in valid and ch_type == "eeg":
            combine_dict["gfp"] = _gfp
        elif "gfp" in valid:
            combine_dict["gfp"] = _rms
        try:
            combine = combine_dict[combine]
        except KeyError:
            raise ValueError(
                f'"combine" must be None, a callable, or one of "{", ".join(valid)}"; '
                f'got {combine}'
            )
    return combine


def _convert_psds(
    psds, dB, estimate, scaling, unit, ch_names=None, first_dim="channel"
):
    """Convert PSDs to dB (if necessary) and appropriate units."""
    _check_option("first_dim", first_dim, ["channel", "epoch"])
    where = np.where(psds.min(1) <= 0)[0]
    if len(where) > 0:
        # Construct a helpful error message, depending on whether the first dimension of
        # `psds` corresponds to channels or epochs.
        if dB:
            bad_value = "Infinite"
        else:
            bad_value = "Zero"

        if first_dim == "channel":
            bads = ", ".join(ch_names[ii] for ii in where)
        else:
            bads = ", ".join(str(ii) for ii in where)

        msg = f"{bad_value} value in PSD for {first_dim}{_pl(where)} {bads}."
        if first_dim == "channel":
            msg += "\nThese channels might be dead."
        warn(msg, UserWarning)

    _check_option("estimate", estimate, ("power", "amplitude"))
    if estimate == "amplitude":
        np.sqrt(psds, out=psds)
        psds *= scaling
        ylabel = rf"$\mathrm{{{unit}/\sqrt{{Hz}}}}$"
        coef = 20
    else:
        psds *= scaling * scaling
        if "/" in unit:
            unit = f"({unit})"
        ylabel = rf"$\mathrm{{{unit}²/Hz}}$"
        coef = 10
    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
        psds *= coef
        ylabel = r"$\mathrm{dB}\ $" + ylabel
    ylabel = "Power (" + ylabel if estimate == "power" else "Amplitude (" + ylabel
    ylabel += ")"

    return ylabel


def _plot_psd(
    inst,
    fig,
    freqs,
    psd_list,
    picks_list,
    titles_list,
    units_list,
    scalings_list,
    ax_list,
    make_label,
    color,
    area_mode,
    area_alpha,
    dB,
    estimate,
    average,
    spatial_colors,
    xscale,
    line_alpha,
    sphere,
    xlabels_list,
):
    # helper function for Spectrum.plot()
    from matplotlib.ticker import ScalarFormatter

    from ..stats import _ci
    from .evoked import _plot_lines

    for key, ls in zip(["lowpass", "highpass", "line_freq"], ["--", "--", "-."]):
        if inst.info[key] is not None:
            for ax in ax_list:
                ax.axvline(
                    inst.info[key],
                    color="k",
                    linestyle=ls,
                    alpha=0.25,
                    linewidth=2,
                    zorder=2,
                )
    if line_alpha is None:
        line_alpha = 1.0 if average else 0.75
    line_alpha = float(line_alpha)
    ylabels = list()
    for ii, (psd, picks, title, ax, scalings, units) in enumerate(
        zip(psd_list, picks_list, titles_list, ax_list, scalings_list, units_list)
    ):
        ylabel = _convert_psds(
            psd, dB, estimate, scalings, units, [inst.ch_names[pi] for pi in picks]
        )
        ylabels.append(ylabel)
        del ylabel

        if average:
            # mean across channels
            psd_mean = np.mean(psd, axis=0)
            if area_mode in ("sd", "std"):
                # std across channels
                psd_std = np.std(psd, axis=0)
                hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
            elif area_mode == "range":
                hyp_limits = (np.min(psd, axis=0), np.max(psd, axis=0))
            elif area_mode is None:
                hyp_limits = None
            else:  # area_mode is float
                hyp_limits = _ci(psd, ci=area_mode)

            ax.plot(freqs, psd_mean, color=color, alpha=line_alpha, linewidth=0.5)
            if hyp_limits is not None:
                ax.fill_between(
                    freqs,
                    hyp_limits[0],
                    y2=hyp_limits[1],
                    facecolor=color,
                    alpha=area_alpha,
                )

    if not average:
        picks = np.concatenate(picks_list)
        info = pick_info(inst.info, sel=picks, copy=True)
        bad_ch_idx = [info["ch_names"].index(ch) for ch in info["bads"]]
        types = np.array(info.get_channel_types())
        ch_types_used = list()
        for this_type in _VALID_CHANNEL_TYPES:
            if this_type in types:
                ch_types_used.append(this_type)
        assert len(ch_types_used) == len(ax_list)
        unit = ""
        units = {t: yl for t, yl in zip(ch_types_used, ylabels)}
        titles = {c: t for c, t in zip(ch_types_used, titles_list)}
        # here we overwrite `picks` because of how _plot_lines works;
        # we already have the data, ch_types, etc in sync.
        psd_array = np.concatenate(psd_list)
        picks = np.arange(len(psd_array))
        if not spatial_colors:
            spatial_colors = color
        _plot_lines(
            psd_array,
            info,
            picks,
            fig,
            ax_list,
            spatial_colors,
            unit,
            units=units,
            scalings=None,
            hline=None,
            gfp=False,
            types=types,
            zorder="std",
            xlim=(freqs[0], freqs[-1]),
            ylim=None,
            times=freqs,
            bad_ch_idx=bad_ch_idx,
            titles=titles,
            ch_types_used=ch_types_used,
            selectable=True,
            psd=True,
            line_alpha=line_alpha,
            nave=None,
            time_unit="ms",
            sphere=sphere,
            highlight=None,
        )

    for ii, (ax, xlabel) in enumerate(zip(ax_list, xlabels_list)):
        ax.grid(True, linestyle=":")
        if xscale == "log":
            ax.set(xscale="log")
            ax.set(xlim=[freqs[1] if freqs[0] == 0 else freqs[0], freqs[-1]])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
        else:  # xscale == 'linear'
            ax.set(xlim=(freqs[0], freqs[-1]))
        if make_label:
            ax.set(ylabel=ylabels[ii], title=titles_list[ii])
            if xlabel:
                ax.set_xlabel("Frequency (Hz)")

    if make_label:
        fig.align_ylabels(axs=ax_list)
    return fig


def _format_units_psd(unit, latex=False, power=True, dB=False):
    """Format PSD measurement units nicely."""
    unit = f"({unit})" if "/" in unit else unit
    if power:
        denom = "Hz"
        exp = r"^{2}" if latex else "²"
    else:
        denom = r"\sqrt{Hz}" if latex else "√(Hz)"
        exp = ""
    pre, post = (r"$\mathrm{", r"}$") if latex else ("", "")
    db = " (dB)" if dB else ""
    return f"{pre}{unit}{exp}/{denom}{post}{db}"


def _prepare_sensor_names(names, show_names):
    """Apply callable to sensor names (if provided)."""
    if callable(show_names):
        names = [show_names(name) for name in names]
    elif not show_names:
        names = None
    return names


def _trim_ticks(ticks, _min, _max):
    """Remove ticks that are more extreme than the given limits."""
    if np.isclose(_min, _max):
        keep_idx = 0  # ensure we always keep at least one tick
    else:
        keep_idx = np.where(np.logical_and(ticks >= _min, ticks <= _max))
    return np.atleast_1d(ticks[keep_idx])


def _set_window_title(fig, title):
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)


def _shorten_path_from_middle(fpath, max_len=60, replacement="..."):
    """Truncate a path from the middle by omitting complete path elements."""
    from os.path import sep

    if len(fpath) > max_len:
        pathlist = fpath.split(sep)
        # indices starting from middle, alternating sides, omitting final elem:
        # range(8) → 3, 4, 2, 5, 1, 6; range(7) → 2, 3, 1, 4, 0, 5
        ixs_to_trunc = list(
            zip(
                range(len(pathlist) // 2 - 1, -1, -1),
                range(len(pathlist) // 2, len(pathlist) - 1),
            )
        )
        ixs_to_trunc = np.array(ixs_to_trunc).flatten()
        for ix in ixs_to_trunc:
            pathlist[ix] = replacement
            truncs = (np.array(pathlist) == replacement).nonzero()[0]
            newpath = sep.join(pathlist[: truncs[0]] + pathlist[truncs[-1] :])
            if len(newpath) < max_len:
                break
        return newpath
    return fpath


def centers_to_edges(*arrays):
    """Convert center points to edges.

    Parameters
    ----------
    *arrays : list of ndarray
        Each input array should be 1D monotonically increasing,
        and will be cast to float.

    Returns
    -------
    arrays : list of ndarray
        Given each input of shape (N,), the output will have shape (N+1,).

    Examples
    --------
    >>> x = [0., 0.1, 0.2, 0.3]
    >>> y = [20, 30, 40]
    >>> centers_to_edges(x, y)  # doctest: +SKIP
    [array([-0.05, 0.05, 0.15, 0.25, 0.35]), array([15., 25., 35., 45.])]
    """
    out = list()
    for ai, arr in enumerate(arrays):
        arr = np.asarray(arr, dtype=float)
        _check_option(f"arrays[{ai}].ndim", arr.ndim, (1,))
        if len(arr) > 1:
            arr_diff = np.diff(arr) / 2.0
        else:
            arr_diff = [abs(arr[0]) * 0.001] if arr[0] != 0 else [0.001]
        out.append(
            np.concatenate(
                [[arr[0] - arr_diff[0]], arr[:-1] + arr_diff, [arr[-1] + arr_diff[-1]]]
            )
        )
    return out


def _figure_agg(**kwargs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(**kwargs)
    FigureCanvasAgg(fig)
    return fig


def _ndarray_to_fig(img, dpi=100):
    """Convert to MPL figure, adapted from matplotlib.image.imsave."""
    figsize = np.array(img.shape[:2][::-1]) / dpi
    fig = _figure_agg(dpi=dpi, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
    ax.imshow(img)
    return fig


def _save_ndarray_img(fname, img):
    """Save an image to disk."""
    from PIL import Image

    Image.fromarray(img).save(fname)


def concatenate_images(images, axis=0, bgcolor="black", centered=True, n_channels=3):
    """Concatenate a list of images.

    Parameters
    ----------
    images : list of ndarray
        The list of images to concatenate.
    axis : 0 or 1
        The images are concatenated horizontally if 0 and vertically otherwise.
        The default orientation is horizontal.
    bgcolor : str | list
        The color of the background. The name of the color is accepted
        (e.g 'red') or a list of RGB values between 0 and 1. Defaults to
        'black'.
    centered : bool
        If True, the images are centered. Defaults to True.
    n_channels : int
        Number of color channels. Can be 3 or 4. The default value is 3.

    Returns
    -------
    img : ndarray
        The concatenated image.
    """
    n_channels = _ensure_int(n_channels, "n_channels")
    axis = _ensure_int(axis)
    _check_option("axis", axis, (0, 1))
    _check_option("n_channels", n_channels, (3, 4))
    alpha = True if n_channels == 4 else False
    bgcolor = _to_rgb(bgcolor, name="bgcolor", alpha=alpha)
    bgcolor = np.asarray(bgcolor) * 255
    funcs = [np.sum, np.max]
    ret_shape = np.asarray(
        [
            funcs[axis]([image.shape[0] for image in images]),
            funcs[1 - axis]([image.shape[1] for image in images]),
        ]
    )
    ret = np.zeros((ret_shape[0], ret_shape[1], n_channels), dtype=np.uint8)
    ret[:, :, :] = bgcolor
    ptr = np.array([0, 0])
    sec = np.array([0 == axis, 1 == axis]).astype(int)
    for image in images:
        shape = image.shape[:-1]
        dec = ptr.copy()
        dec += ((ret_shape - shape) // 2) * (1 - sec) if centered else 0
        ret[dec[0] : dec[0] + shape[0], dec[1] : dec[1] + shape[1], :] = image
        ptr += shape * sec
    return ret


def _generate_default_filename(ext=".png"):
    now = datetime.now()
    dt_string = now.strftime("_%Y-%m-%d_%H-%M-%S")
    return "MNE" + dt_string + ext


def _handle_precompute(precompute):
    _validate_type(precompute, (bool, str, None), "precompute")
    if precompute is None:
        precompute = get_config("MNE_BROWSER_PRECOMPUTE", "auto").lower()
        _check_option(
            "MNE_BROWSER_PRECOMPUTE",
            precompute,
            ("true", "false", "auto"),
            extra="when precompute=None is used",
        )
        precompute = dict(true=True, false=False, auto="auto")[precompute]
    return precompute


def _set_3d_axes_equal(ax):
    """Make axes of 3D plot have equal scale on all dimensions.

    This way spheres appear as actual spheres, cubes as cubes, etc.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        A matplotlib 3d axis to use.
    """
    ranges = tuple(
        np.abs(np.diff(getattr(ax, f"get_{d}lim")())).item() for d in ("x", "y", "z")
    )
    ax.set_box_aspect(ranges)


def _check_type_projs(projs):
    _validate_type(projs, (list, tuple, Projection), "projs")
    if isinstance(projs, Projection):
        projs = [projs]
    for pi, p in enumerate(projs):
        _validate_type(p, Projection, f"projs[{pi}]")
    return projs


def _get_cmap(colormap, lut=None):
    from matplotlib import colors, rcParams

    try:
        from matplotlib import colormaps
    except Exception:
        from matplotlib.cm import get_cmap
    else:

        def get_cmap(cmap):
            return colormaps[cmap]

    if colormap is None:
        colormap = rcParams["image.cmap"]
    if isinstance(colormap, str) and colormap in ("mne", "mne_analyze"):
        colormap = mne_analyze_colormap([0, 1, 2], format="matplotlib")
    elif not isinstance(colormap, colors.Colormap):
        colormap = get_cmap(colormap)
    if lut is not None:
        colormap = colormap.resampled(lut)
    return colormap


def _get_plot_ch_type(inst, ch_type, allow_ref_meg=False):
    """Choose a single channel type (usually for plotting).

    Usually used in plotting to plot a single datatype, e.g. look for mags,
    then grads, then ... to plot.
    """
    if ch_type is None:
        allowed_types = list(_DATA_CH_TYPES_SPLIT)
        allowed_types += ["ref_meg"] if allow_ref_meg else []
        has_types = inst.get_channel_types(unique=True)
        for type_ in allowed_types:
            if type_ in has_types:
                ch_type = type_
                break
        else:
            raise RuntimeError(
                f"No plottable channel types found. Allowed types are: {allowed_types}"
            )
    return ch_type
