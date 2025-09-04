"""Functions to plot on circle as for connectivity."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial
from itertools import cycle
from types import SimpleNamespace

import numpy as np

from ..utils import _validate_type
from .utils import _get_cmap, plt_show


def circular_layout(
    node_names,
    node_order,
    start_pos=90,
    start_between=True,
    group_boundaries=None,
    group_sep=10,
):
    """Create layout arranging nodes on a circle.

    Parameters
    ----------
    node_names : list of str
        Node names.
    node_order : list of str
        List with node names defining the order in which the nodes are
        arranged. Must have the elements as node_names but the order can be
        different. The nodes are arranged clockwise starting at "start_pos"
        degrees.
    start_pos : float
        Angle in degrees that defines where the first node is plotted.
    start_between : bool
        If True, the layout starts with the position between the nodes. This is
        the same as adding "180. / len(node_names)" to start_pos.
    group_boundaries : None | array-like
        List of of boundaries between groups at which point a "group_sep" will
        be inserted. E.g. "[0, len(node_names) / 2]" will create two groups.
    group_sep : float
        Group separation angle in degrees. See "group_boundaries".

    Returns
    -------
    node_angles : array, shape=(n_node_names,)
        Node angles in degrees.
    """
    n_nodes = len(node_names)

    if len(node_order) != n_nodes:
        raise ValueError("node_order has to be the same length as node_names")

    if group_boundaries is not None:
        boundaries = np.array(group_boundaries, dtype=np.int64)
        if np.any(boundaries >= n_nodes) or np.any(boundaries < 0):
            raise ValueError('"group_boundaries" has to be between 0 and n_nodes - 1.')
        if len(boundaries) > 1 and np.any(np.diff(boundaries) <= 0):
            raise ValueError('"group_boundaries" must have non-decreasing values.')
        n_group_sep = len(group_boundaries)
    else:
        n_group_sep = 0
        boundaries = None

    # convert it to a list with indices
    node_order = [node_order.index(name) for name in node_names]
    node_order = np.array(node_order)
    if len(np.unique(node_order)) != n_nodes:
        raise ValueError("node_order has repeated entries")

    node_sep = (360.0 - n_group_sep * group_sep) / n_nodes

    if start_between:
        start_pos += node_sep / 2

        if boundaries is not None and boundaries[0] == 0:
            # special case when a group separator is at the start
            start_pos += group_sep / 2
            boundaries = boundaries[1:] if n_group_sep > 1 else None

    node_angles = np.ones(n_nodes, dtype=np.float64) * node_sep
    node_angles[0] = start_pos
    if boundaries is not None:
        node_angles[boundaries] += group_sep

    node_angles = np.cumsum(node_angles)[node_order]

    return node_angles


def _plot_connectivity_circle_onpick(
    event,
    fig=None,
    ax=None,
    indices=None,
    n_nodes=0,
    node_angles=None,
    ylim=(9, 10),
):
    """Isolate connections around a single node when user left clicks a node.

    On right click, resets all connections.
    """
    if event.inaxes != ax:
        return

    if event.button == 1:  # left click
        # click must be near node radius
        if not ylim[0] <= event.ydata <= ylim[1]:
            return

        # all angles in range [0, 2*pi]
        node_angles = node_angles % (np.pi * 2)
        node = np.argmin(np.abs(event.xdata - node_angles))

        patches = event.inaxes.patches
        for ii, (x, y) in enumerate(zip(indices[0], indices[1])):
            patches[ii].set_visible(node in [x, y])
        fig.canvas.draw()
    elif event.button == 3:  # right click
        patches = event.inaxes.patches
        for ii in range(np.size(indices, axis=1)):
            patches[ii].set_visible(True)
        fig.canvas.draw()


def _plot_connectivity_circle(
    con,
    node_names,
    indices=None,
    n_lines=None,
    node_angles=None,
    node_width=None,
    node_height=None,
    node_colors=None,
    facecolor="black",
    textcolor="white",
    node_edgecolor="black",
    linewidth=1.5,
    colormap="hot",
    vmin=None,
    vmax=None,
    colorbar=True,
    title=None,
    colorbar_size=None,
    colorbar_pos=None,
    fontsize_title=12,
    fontsize_names=8,
    fontsize_colorbar=8,
    padding=6.0,
    ax=None,
    interactive=True,
    node_linewidth=2.0,
    show=True,
):
    import matplotlib.patches as m_patches
    import matplotlib.path as m_path
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes

    _validate_type(ax, (None, PolarAxes))

    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError("node_angles has to be the same length as node_names")
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        # widths correspond to the minimum angle between two nodes
        dist_mat = node_angles[None, :] - node_angles[:, None]
        dist_mat[np.diag_indices(n_nodes)] = 1e9
        node_width = np.min(np.abs(dist_mat))
    else:
        node_width = node_width * np.pi / 180

    if node_height is None:
        node_height = 1.0

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        try:
            spectral = plt.cm.spectral
        except AttributeError:
            spectral = plt.cm.Spectral
        node_colors = [spectral(i / float(n_nodes)) for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError("indices has to be provided if con.ndim == 1")
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError("con has to be 1D or a square matrix")
        # we use the lower-triangular part
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError("con has to be 1D or a square matrix")

    # get the colormap
    colormap = _get_cmap(colormap)

    # Use a polar axes
    if ax is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor, layout="constrained")
        ax = fig.add_subplot(polar=True)
    else:
        fig = ax.figure
    ax.set_facecolor(facecolor)

    # No ticks, we'll put our own
    ax.set_xticks([])
    ax.set_yticks([])

    # Set y axes limit, add additional space if requested
    ax.set_ylim(0, 10 + padding)

    # Remove the black axes border which may obscure the labels
    ax.spines["polar"].set_visible(False)

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.0

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    del con_abs
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int64)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initialize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= (nodes_n_con[start] - nodes_n_con_seen[start]) / float(
            nodes_n_con[start]
        )
        end_noise[i] *= (nodes_n_con[end] - nodes_n_con_seen[end]) / float(
            nodes_n_con[end]
        )

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [
            m_path.Path.MOVETO,
            m_path.Path.CURVE4,
            m_path.Path.CURVE4,
            m_path.Path.LINETO,
        ]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(
            path, fill=False, edgecolor=color, linewidth=linewidth, alpha=1.0
        )
        ax.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * node_height
    bars = ax.bar(
        node_angles,
        height,
        width=node_width,
        bottom=9,
        edgecolor=node_edgecolor,
        lw=node_linewidth,
        facecolor=".9",
        align="center",
    )

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = "left"
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = "right"

        ax.text(
            angle_rad,
            9.4 + node_height,
            name,
            size=fontsize_names,
            rotation=angle_deg,
            rotation_mode="anchor",
            horizontalalignment=ha,
            verticalalignment="center",
            color=textcolor,
        )

    if title is not None:
        ax.set_title(title, color=textcolor, fontsize=fontsize_title)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin, vmax))
        sm.set_array(np.linspace(vmin, vmax))
        colorbar_kwargs = dict()
        if colorbar_size is not None:
            colorbar_kwargs.update(shrink=colorbar_size)
        if colorbar_pos is not None:
            colorbar_kwargs.update(anchor=colorbar_pos)
        cb = fig.colorbar(sm, ax=ax, **colorbar_kwargs)
        cb_yticks = plt.getp(cb.ax.axes, "yticklabels")
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)
        fig.mne = SimpleNamespace(colorbar=cb)

    # Add callback for interaction
    if interactive:
        callback = partial(
            _plot_connectivity_circle_onpick,
            fig=fig,
            ax=ax,
            indices=indices,
            n_nodes=n_nodes,
            node_angles=node_angles,
        )

        fig.canvas.mpl_connect("button_press_event", callback)

    plt_show(show)
    return fig, ax


def plot_channel_labels_circle(labels, colors=None, picks=None, **kwargs):
    """Plot labels for each channel in a circle plot.

    .. note:: This primarily makes sense for sEEG channels where each
              channel can be assigned an anatomical label as the electrode
              passes through various brain areas.

    Parameters
    ----------
    labels : dict
        Lists of labels (values) associated with each channel (keys).
    colors : dict
        The color (value) for each label (key).
    picks : list | tuple
        The channels to consider.
    **kwargs : kwargs
        Keyword arguments for
        :func:`mne_connectivity.viz.plot_connectivity_circle`.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure handle.
    axes : instance of matplotlib.projections.polar.PolarAxes
        The subplot handle.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _validate_type(labels, dict, "labels")
    _validate_type(colors, (dict, None), "colors")
    _validate_type(picks, (list, tuple, None), "picks")
    if picks is not None:
        labels = {k: v for k, v in labels.items() if k in picks}
    ch_names = list(labels.keys())
    all_labels = list(set([label for val in labels.values() for label in val]))
    n_labels = len(all_labels)
    if colors is not None:
        for label in all_labels:
            if label not in colors:
                raise ValueError(f"No color provided for {label} in `colors`")
        # update all_labels, there may be unconnected labels in colors
        all_labels = list(colors.keys())
        n_labels = len(all_labels)
        # make colormap
        label_colors = [colors[label] for label in all_labels]
        node_colors = ["black"] * len(ch_names) + label_colors
        label_cmap = LinearSegmentedColormap.from_list(
            "label_cmap", label_colors, N=len(label_colors)
        )
    else:
        node_colors = None

    node_names = ch_names + all_labels
    con = np.zeros((len(node_names), len(node_names))) * np.nan
    for idx, ch_name in enumerate(ch_names):
        for label in labels[ch_name]:
            node_idx = node_names.index(label)
            label_color = all_labels.index(label) / n_labels
            con[idx, node_idx] = con[node_idx, idx] = label_color  # symmetric
    # plot
    node_order = ch_names + all_labels[::-1]
    node_angles = circular_layout(
        node_names, node_order, start_pos=90, group_boundaries=[0, len(ch_names)]
    )
    # provide defaults but don't overwrite
    if "node_angles" not in kwargs:
        kwargs.update(node_angles=node_angles)
    if "colorbar" not in kwargs:
        kwargs.update(colorbar=False)
    if "node_colors" not in kwargs:
        kwargs.update(node_colors=node_colors)
    if "vmin" not in kwargs:
        kwargs.update(vmin=0)
    if "vmax" not in kwargs:
        kwargs.update(vmax=1)
    if "colormap" not in kwargs:
        kwargs.update(colormap=label_cmap)
    return _plot_connectivity_circle(con, node_names, **kwargs)
