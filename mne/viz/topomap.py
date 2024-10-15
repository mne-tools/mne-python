"""Functions to plot M/EEG data e.g. topographies."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
import itertools
import warnings
from functools import partial
from numbers import Integral

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.sparse import csr_array
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import pdist, squareform

from .._fiff.meas_info import Info, _simplify_info
from .._fiff.pick import (
    _MEG_CH_TYPES_SPLIT,
    _pick_data_channels,
    _picks_by_type,
    _picks_to_idx,
    pick_channels,
    pick_info,
    pick_types,
)
from ..baseline import rescale
from ..defaults import (
    _BORDER_DEFAULT,
    _EXTRAPOLATE_DEFAULT,
    _INTERPOLATION_DEFAULT,
    _handle_default,
)
from ..transforms import apply_trans, invert_transform
from ..utils import (
    _check_option,
    _check_sphere,
    _clean_names,
    _is_numeric,
    _time_mask,
    _validate_type,
    check_version,
    fill_doc,
    legacy,
    logger,
    verbose,
    warn,
)
from ..utils.spectrum import _split_psd_kwargs
from .ui_events import TimeChange, publish, subscribe
from .utils import (
    DraggableColorbar,
    _check_delayed_ssp,
    _check_time_unit,
    _check_type_projs,
    _draw_proj_checkbox,
    _format_units_psd,
    _get_cmap,
    _get_plot_ch_type,
    _prepare_sensor_names,
    _prepare_trellis,
    _process_times,
    _set_3d_axes_equal,
    _setup_cmap,
    _setup_vmin_vmax,
    _validate_if_list_of_axes,
    figure_nobar,
    plot_sensors,
    plt_show,
)

_fnirs_types = ("hbo", "hbr", "fnirs_cw_amplitude", "fnirs_od")


# 3.8+ uses a single Collection artist rather than .collections
# https://github.com/matplotlib/matplotlib/pull/25247
def _cont_collections(cont):
    return (cont,) if check_version("matplotlib", "3.8") else tuple(cont.collections)


def _adjust_meg_sphere(sphere, info, ch_type):
    sphere = _check_sphere(sphere, info)
    assert ch_type is not None
    if ch_type in ("mag", "grad", "planar1", "planar2"):
        # move sphere X/Y (head coords) to device X/Y space
        if info["dev_head_t"] is not None:
            head_dev_t = invert_transform(info["dev_head_t"])
            sphere[:3] = apply_trans(head_dev_t, sphere[:3])
            # Set the sphere Z=0 because all this really affects is flattening.
            # We could make the head size change as a function of depth in
            # the helmet like:
            #
            #     sphere[2] /= -5
            #
            # but let's just assume some orthographic rather than parallel
            # projection for explicitness / simplicity.
            sphere[2] = 0.0
        clip_origin = (0.0, 0.0)
    else:
        clip_origin = sphere[:2].copy()
    return sphere, clip_origin


def _prepare_topomap_plot(inst, ch_type, sphere=None):
    """Prepare topo plot."""
    from ..channels.layout import _find_topomap_coords, _pair_grad_sensors, find_layout

    info = copy.deepcopy(inst if isinstance(inst, Info) else inst.info)
    sphere, clip_origin = _adjust_meg_sphere(sphere, info, ch_type)

    clean_ch_names = _clean_names(info["ch_names"])
    for ii, this_ch in enumerate(info["chs"]):
        this_ch["ch_name"] = clean_ch_names[ii]
    for comp in info["comps"]:
        comp["data"]["col_names"] = _clean_names(comp["data"]["col_names"])
    info._update_redundant()
    info["bads"] = _clean_names(info["bads"])
    info._check_consistency()

    # special case for merging grad channels
    layout = find_layout(info)
    if (
        ch_type == "grad"
        and layout is not None
        and (
            layout.kind.startswith("Vectorview")
            or layout.kind.startswith("Neuromag_122")
        )
    ):
        picks, _ = _pair_grad_sensors(info, layout)
        pos = _find_topomap_coords(info, picks[::2], sphere=sphere)
        merge_channels = True
    elif ch_type in _fnirs_types:
        # fNIRS data commonly has overlapping channels, so deal with separately
        picks, pos, merge_channels, overlapping_channels = _average_fnirs_overlaps(
            info, ch_type, sphere
        )
    else:
        merge_channels = False
        if ch_type == "eeg":
            picks = pick_types(info, meg=False, eeg=True, ref_meg=False, exclude="bads")
        elif ch_type == "csd":
            picks = pick_types(info, meg=False, csd=True, ref_meg=False, exclude="bads")
        elif ch_type == "dbs":
            picks = pick_types(info, meg=False, dbs=True, ref_meg=False, exclude="bads")
        elif ch_type == "seeg":
            picks = pick_types(
                info, meg=False, seeg=True, ref_meg=False, exclude="bads"
            )
        else:
            picks = pick_types(info, meg=ch_type, ref_meg=False, exclude="bads")

        if len(picks) == 0:
            raise ValueError(f"No channels of type {ch_type!r}")

        pos = _find_topomap_coords(info, picks, sphere=sphere)

    ch_names = [info["ch_names"][k] for k in picks]
    if ch_type in _fnirs_types:
        # Remove the chroma label type for cleaner labeling.
        ch_names = [k[:-4] for k in ch_names]

    if merge_channels:
        if ch_type == "grad":
            # change names so that vectorview combined grads appear as MEG014x
            # instead of MEG0142 or MEG0143 which are the 2 planar grads.
            ch_names = [ch_names[k][:-1] + "x" for k in range(0, len(ch_names), 2)]
        else:
            assert ch_type in _fnirs_types
            # Modify the nirs channel names to indicate they are to be merged
            # New names will have the form  S1_D1xS2_D2
            # More than two channels can overlap and be merged
            for set_ in overlapping_channels:
                idx = ch_names.index(set_[0][:-4])
                new_name = "x".join(s[:-4] for s in set_)
                ch_names[idx] = new_name

    pos = np.array(pos)[:, :2]  # 2D plot, otherwise interpolation bugs
    return picks, pos, merge_channels, ch_names, ch_type, sphere, clip_origin


def _average_fnirs_overlaps(info, ch_type, sphere):
    from ..channels.layout import _find_topomap_coords

    picks = pick_types(info, meg=False, ref_meg=False, fnirs=ch_type, exclude="bads")
    chs = [info["chs"][i] for i in picks]
    locs3d = np.array([ch["loc"][:3] for ch in chs])
    dist = pdist(locs3d)

    # Store the sets of channels to be merged
    overlapping_channels = list()
    # Channels to be excluded from picks, as will be removed after merging
    channels_to_exclude = list()

    if len(locs3d) > 1 and np.min(dist) < 1e-10:
        overlapping_mask = np.triu(squareform(dist < 1e-10))
        for chan_idx in range(overlapping_mask.shape[0]):
            already_overlapped = list(
                itertools.chain.from_iterable(overlapping_channels)
            )
            if overlapping_mask[chan_idx].any() and (
                chs[chan_idx]["ch_name"] not in already_overlapped
            ):
                # Determine the set of channels to be combined. Ensure the
                # first listed channel is the one to be replaced with merge
                overlapping_set = [
                    chs[i]["ch_name"] for i in np.where(overlapping_mask[chan_idx])[0]
                ]
                overlapping_set = np.insert(
                    overlapping_set, 0, (chs[chan_idx]["ch_name"])
                )
                overlapping_channels.append(overlapping_set)
                channels_to_exclude.append(overlapping_set[1:])

        exclude = list(itertools.chain.from_iterable(channels_to_exclude))
        [exclude.append(bad) for bad in info["bads"]]
        picks = pick_types(
            info, meg=False, ref_meg=False, fnirs=ch_type, exclude=exclude
        )
        pos = _find_topomap_coords(info, picks, sphere=sphere)
        picks = pick_types(info, meg=False, ref_meg=False, fnirs=ch_type)
        # Overload the merge_channels variable as this is returned to calling
        # function and indicates that merging of data is required
        merge_channels = overlapping_channels

    else:
        picks = pick_types(
            info, meg=False, ref_meg=False, fnirs=ch_type, exclude="bads"
        )
        merge_channels = False
        pos = _find_topomap_coords(info, picks, sphere=sphere)

    return picks, pos, merge_channels, overlapping_channels


def _plot_update_evoked_topomap(params, bools):
    """Update topomaps."""
    from ..channels.layout import _merge_ch_data

    projs = [
        proj for ii, proj in enumerate(params["projs"]) if ii in np.where(bools)[0]
    ]

    params["proj_bools"] = bools
    new_evoked = params["evoked"].copy()
    with new_evoked.info._unlock():
        new_evoked.info["projs"] = []
    new_evoked.add_proj(projs)
    new_evoked.apply_proj()

    data = new_evoked.data[:, params["time_idx"]] * params["scale"]
    if params["merge_channels"]:
        data, _ = _merge_ch_data(data, "grad", [])

    interp = params["interp"]
    new_contours = list()
    use_contours = params["contours_"]
    if not len(use_contours):
        use_contours = [None] * len(params["axes"])
    assert len(use_contours) == len(params["images"])
    assert len(params["axes"]) == len(params["images"])
    assert len(data.T) == len(params["images"])
    for cont, ax, im, d in zip(use_contours, params["axes"], params["images"], data.T):
        Zi = interp.set_values(d)()
        im.set_data(Zi)
        if cont is None:
            continue
        # must be removed and re-added
        cont_collections = _cont_collections(cont)
        for col in cont_collections:
            col.remove()
        col = cont_collections[0]
        lw = col.get_linewidth()
        visible = col.get_visible()
        patch_ = col.get_clip_path()
        color = col.get_edgecolors()
        cont = ax.contour(
            interp.Xi, interp.Yi, Zi, params["contours"], colors=color, linewidths=lw
        )
        cont_collections = _cont_collections(cont)
        for col in cont_collections:
            col.set_visible(visible)
            col.set_clip_path(patch_)
        new_contours.append(cont)
    params["contours_"] = new_contours

    params["fig"].canvas.draw()


def _add_colorbar(
    ax,
    im,
    cmap,
    *,
    title=None,
    format_=None,
    kind=None,
    ch_type=None,
):
    """Add a colorbar to an axis."""
    cbar = ax.figure.colorbar(im, format=format_, shrink=0.6)
    if cmap is not None and cmap[1]:
        ax.CB = DraggableColorbar(cbar, im, kind, ch_type)
    cax = cbar.ax
    if title is not None:
        cax.set_title(title, y=1.05, fontsize=10)
    return cbar, cax


def _eliminate_zeros(proj):
    """Remove grad or mag data if only contains 0s (gh 5641)."""
    GRAD_ENDING = ("2", "3")
    MAG_ENDING = "1"

    proj = copy.deepcopy(proj)
    proj["data"]["data"] = np.atleast_2d(proj["data"]["data"])

    for ending in (GRAD_ENDING, MAG_ENDING):
        names = proj["data"]["col_names"]
        idx = [i for i, name in enumerate(names) if name.endswith(ending)]

        # if all 0, remove the 0s an their labels
        if not proj["data"]["data"][0][idx].any():
            new_col_names = np.delete(np.array(names), idx).tolist()
            new_data = np.delete(np.array(proj["data"]["data"][0]), idx)
            proj["data"]["col_names"] = new_col_names
            proj["data"]["data"] = np.array([new_data])

    proj["data"]["ncol"] = len(proj["data"]["col_names"])
    return proj


@fill_doc
def plot_projs_topomap(
    projs,
    info,
    *,
    sensors=True,
    show_names=False,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    size=1,
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=False,
    cbar_fmt="%3.1f",
    units=None,
    axes=None,
    show=True,
):
    """Plot topographic maps of SSP projections.

    Parameters
    ----------
    projs : list of Projection
        The projections.
    %(info_not_none)s Must be associated with the channels in the projectors.

        .. versionchanged:: 0.20
            The positional argument ``layout`` was replaced by ``info``.
    %(sensors_topomap)s
    %(show_names_topomap)s

        .. versionadded:: 1.2
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionadded:: 0.20

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap_proj)s
    %(cnorm)s

        .. versionadded:: 1.2
    %(colorbar_topomap)s
    %(cbar_fmt_topomap)s

        .. versionadded:: 1.2
    %(units_topomap)s

        .. versionadded:: 1.2
    %(axes_plot_projs_topomap)s
    %(show)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure with a topomap subplot for each projector.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    fig = _plot_projs_topomap(
        projs,
        info,
        sensors=sensors,
        show_names=show_names,
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
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
    plt_show(show)
    return fig


def _plot_projs_topomap(
    projs,
    info,
    sensors=True,
    show_names=False,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    size=1,
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=False,
    cbar_fmt="%3.1f",
    units=None,
    axes=None,
):
    import matplotlib.pyplot as plt

    from ..channels.layout import _merge_ch_data

    sphere = _check_sphere(sphere, info)
    projs = _check_type_projs(projs)
    _validate_type(info, "info", "info")

    # Preprocess projs to deal with joint MEG projectors. If we duplicate these and
    # split into mag and grad, they should work as expected
    info_names = _clean_names(info["ch_names"], remove_whitespace=True)
    use_projs = list()
    for proj in projs:
        proj = _eliminate_zeros(proj)  # gh 5641, makes a copy
        proj["data"]["col_names"] = _clean_names(
            proj["data"]["col_names"],
            remove_whitespace=True,
        )
        picks = pick_channels(info_names, proj["data"]["col_names"], ordered=True)
        proj_types = info.get_channel_types(picks)
        unique_types = sorted(set(proj_types))
        for type_ in unique_types:
            proj_picks = np.where([proj_type == type_ for proj_type in proj_types])[0]
            use_projs.append(copy.deepcopy(proj))
            use_projs[-1]["data"]["data"] = proj["data"]["data"][:, proj_picks]
            use_projs[-1]["data"]["col_names"] = [
                proj["data"]["col_names"][pick] for pick in proj_picks
            ]
    projs = use_projs

    datas, poss, spheres, outliness, ch_typess = [], [], [], [], []
    for proj in projs:
        # get ch_names, ch_types, data
        data = proj["data"]["data"].ravel()
        picks = pick_channels(info_names, proj["data"]["col_names"], ordered=True)
        use_info = pick_info(info, picks)
        these_ch_types = use_info.get_channel_types(unique=True)
        assert len(these_ch_types) == 1  # should be guaranteed above
        ch_type = these_ch_types[0]
        (
            data_picks,
            pos,
            merge_channels,
            names,
            _,
            this_sphere,
            clip_origin,
        ) = _prepare_topomap_plot(use_info, ch_type, sphere=sphere)
        these_outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)
        data = data[data_picks]
        if merge_channels:
            data, _ = _merge_ch_data(data, "grad", [])
            data = data.ravel()

        # populate containers
        datas.append(data)
        poss.append(pos)
        spheres.append(this_sphere)
        outliness.append(these_outlines)
        ch_typess.append(ch_type)
        del data, pos, this_sphere, these_outlines, ch_type
    del sphere

    # setup axes
    n_projs = len(projs)
    if axes is None:
        fig, axes, ncols, nrows = _prepare_trellis(
            n_projs, ncols="auto", nrows="auto", size=size, sharex=True, sharey=True
        )
    elif isinstance(axes, plt.Axes):
        axes = [axes]
    _validate_if_list_of_axes(axes, n_projs)

    # handle vmin/vmax
    vlims = [None for _ in range(len(datas))]
    if vlim == "joint":
        for _ch_type in set(ch_typess):
            idx = np.where(np.isin(ch_typess, _ch_type))[0]
            these_data = np.concatenate(np.array(datas, dtype=object)[idx])
            norm = all(these_data >= 0)
            _vl = _setup_vmin_vmax(these_data, vmin=None, vmax=None, norm=norm)
            for _idx in idx:
                vlims[_idx] = _vl
        # make sure we got a vlim for all projs
        assert all([vl is not None for vl in vlims])
    else:
        vlims = [vlim] * len(datas)

    # plot
    for proj, ax, _data, _pos, _vlim, _sphere, _outlines, _ch_type in zip(
        projs, axes, datas, poss, vlims, spheres, outliness, ch_typess
    ):
        # ch_names
        names = [info["ch_names"][k] for k in _picks_to_idx(info, _ch_type)]
        names = _prepare_sensor_names(names, show_names)
        # title
        title = proj["desc"]
        title = "\n".join(title[ii : ii + 22] for ii in range(0, len(title), 22))
        ax.set_title(title, fontsize=10)
        # plot
        im, _ = plot_topomap(
            _data,
            _pos[:, :2],
            vlim=_vlim,
            cmap=cmap,
            sensors=sensors,
            names=names,
            res=res,
            axes=ax,
            outlines=_outlines,
            contours=contours,
            cnorm=cnorm,
            image_interp=image_interp,
            show=False,
            extrapolate=extrapolate,
            sphere=_sphere,
            border=border,
            ch_type=_ch_type,
        )

        if colorbar:
            _add_colorbar(
                ax,
                im,
                cmap,
                title=units,
                format_=cbar_fmt,
                kind="projs_topomap",
                ch_type=_ch_type,
            )

    return ax.get_figure()


def _make_head_outlines(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ("head", None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        nose_x = np.array([-dx, 0, dx]) * radius + x
        nose_y = np.array([dy, 1.15, dy]) * radius + y
        ear_x = np.array(
            [0.497, 0.510, 0.518, 0.5299, 0.5419, 0.54, 0.547, 0.532, 0.510, 0.489]
        ) * (radius * 2)
        ear_y = (
            np.array(
                [
                    0.0555,
                    0.0775,
                    0.0783,
                    0.0746,
                    0.0555,
                    -0.0055,
                    -0.0932,
                    -0.1313,
                    -0.1384,
                    -0.1199,
                ]
            )
            * (radius * 2)
            + y
        )

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(
                head=(head_x, head_y),
                nose=(nose_x, nose_y),
                ear_left=(-ear_x + x, ear_y),
                ear_right=(ear_x + x, ear_y),
            )
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict["mask_pos"] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict["clip_radius"] = (clip_radius,) * 2
        outlines_dict["clip_origin"] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if "mask_pos" not in outlines:
            raise ValueError("You must specify the coordinates of the image mask.")
    else:
        raise ValueError("Invalid value for `outlines`.")

    return outlines


def _draw_outlines(ax, outlines):
    """Draw the outlines for a topomap."""
    from matplotlib import rcParams

    outlines_ = {k: v for k, v in outlines.items() if k not in ["patch"]}
    for key, (x_coord, y_coord) in outlines_.items():
        if "mask" in key or key in ("clip_radius", "clip_origin"):
            continue
        ax.plot(
            x_coord,
            y_coord,
            color=rcParams["axes.edgecolor"],
            linewidth=1,
            clip_on=False,
        )
    return outlines_


def _get_extra_points(pos, extrapolate, origin, radii):
    """Get coordinates of additional interpolation points."""
    radii = np.array(radii, float)
    assert radii.shape == (2,)
    x, y = origin
    # auto should be gone by now
    _check_option("extrapolate", extrapolate, ("head", "box", "local"))

    # the old method of placement - large box
    mask_pos = None
    if extrapolate == "box":
        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs
        eidx = np.array(
            list(itertools.product(*([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1])))
        )
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        outer_pts = extremes[eidx, pidx]
        return outer_pts, mask_pos, Delaunay(np.concatenate((pos, outer_pts)))

    # check if positions are colinear:
    diffs = np.diff(pos, axis=0)
    with np.errstate(divide="ignore"):
        slopes = diffs[:, 1] / diffs[:, 0]
    colinear = (slopes == slopes[0]).all() or np.isinf(slopes).all()

    # compute median inter-electrode distance
    if colinear or pos.shape[0] < 4:
        dim = 1 if diffs[:, 1].sum() > diffs[:, 0].sum() else 0
        sorting = np.argsort(pos[:, dim])
        pos_sorted = pos[sorting, :]
        diffs = np.diff(pos_sorted, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        distance = np.median(distances)
    else:
        tri = Delaunay(pos, incremental=True)
        idx1, idx2, idx3 = tri.simplices.T
        distances = np.concatenate(
            [
                np.linalg.norm(pos[i1, :] - pos[i2, :], axis=1)
                for i1, i2 in zip([idx1, idx2], [idx2, idx3])
            ]
        )
        distance = np.median(distances)

    if extrapolate == "local":
        if colinear or pos.shape[0] < 4:
            # special case for colinear points and when there is too
            # little points for Delaunay (needs at least 3)
            edge_points = sorting[[0, -1]]
            line_len = np.diff(pos[edge_points, :], axis=0)
            unit_vec = line_len / np.linalg.norm(line_len) * distance
            unit_vec_par = unit_vec[:, ::-1] * [[-1, 1]]

            edge_pos = pos[edge_points, :] + np.concatenate(
                [-unit_vec, unit_vec], axis=0
            )
            new_pos = np.concatenate(
                [pos + unit_vec_par, pos - unit_vec_par, edge_pos], axis=0
            )

            if pos.shape[0] == 3:
                # there may be some new_pos points that are too close
                # to the original points
                new_pos_diff = pos[..., np.newaxis] - new_pos.T[np.newaxis, :]
                new_pos_diff = np.linalg.norm(new_pos_diff, axis=1)
                good_extra = (new_pos_diff > 0.5 * distance).all(axis=0)
                new_pos = new_pos[good_extra]

            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, new_pos, tri

        # get the convex hull of data points from triangulation
        hull_pos = pos[tri.convex_hull]

        # extend the convex hull limits outwards a bit
        channels_center = pos.mean(axis=0)
        radial_dir = hull_pos - channels_center
        unit_radial_dir = radial_dir / np.linalg.norm(
            radial_dir, axis=-1, keepdims=True
        )
        hull_extended = hull_pos + unit_radial_dir * distance
        mask_pos = hull_pos + unit_radial_dir * distance * 0.5
        hull_diff = np.diff(hull_pos, axis=1)[:, 0]
        hull_distances = np.linalg.norm(hull_diff, axis=-1)
        del channels_center

        # Construct a mask
        mask_pos = np.unique(mask_pos.reshape(-1, 2), axis=0)
        mask_center = np.mean(mask_pos, axis=0)
        mask_pos -= mask_center
        mask_pos = mask_pos[np.argsort(np.arctan2(mask_pos[:, 1], mask_pos[:, 0]))]
        mask_pos += mask_center

        # add points along hull edges so that the distance between points
        # is around that of average distance between channels
        add_points = list()
        eps = np.finfo("float").eps
        n_times_dist = np.round(0.25 * hull_distances / distance).astype("int")
        for n in range(2, n_times_dist.max() + 1):
            mask = n_times_dist == n
            mult = np.arange(1 / n, 1 - eps, 1 / n)[:, np.newaxis, np.newaxis]
            steps = hull_diff[mask][np.newaxis, ...] * mult
            add_points.append(
                (hull_extended[mask, 0][np.newaxis, ...] + steps).reshape((-1, 2))
            )

        # remove duplicates from hull_extended
        hull_extended = np.unique(hull_extended.reshape((-1, 2)), axis=0)
        new_pos = np.concatenate([hull_extended] + add_points)
    else:
        assert extrapolate == "head"
        # return points on the head circle
        angle = np.arcsin(min(distance / np.mean(radii), 1))
        n_pnts = max(12, int(np.round(2 * np.pi / angle)))
        points_l = np.linspace(0, 2 * np.pi, n_pnts, endpoint=False)
        use_radii = radii * 1.1 + distance
        points_x = np.cos(points_l) * use_radii[0] + x
        points_y = np.sin(points_l) * use_radii[1] + y
        new_pos = np.stack([points_x, points_y], axis=1)
        if colinear or pos.shape[0] == 3:
            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, mask_pos, tri
    tri.add_points(new_pos)
    return new_pos, mask_pos, tri


class _GridData:
    """Unstructured (x,y) data interpolator.

    This class allows optimized interpolation by computing parameters
    for a fixed set of true points, and allowing the values at those points
    to be set independently.
    """

    def __init__(self, pos, image_interp, extrapolate, origin, radii, border):
        # in principle this works in N dimensions, not just 2
        assert pos.ndim == 2 and pos.shape[1] == 2, pos.shape
        _validate_type(border, ("numeric", str), "border")

        # check that border, if string, is correct
        if isinstance(border, str):
            _check_option("border", border, ("mean",), extra="when a string")

        # Adding points outside the extremes helps the interpolators
        outer_pts, mask_pts, tri = _get_extra_points(pos, extrapolate, origin, radii)
        self.n_extra = outer_pts.shape[0]
        self.mask_pts = mask_pts
        self.border = border
        self.tri = tri
        self.interp = {
            "cubic": CloughTocher2DInterpolator,
            "nearest": NearestNDInterpolator,  # used only for anim
            "linear": LinearNDInterpolator,
        }[image_interp]

    def set_values(self, v):
        """Set the values at interpolation points."""
        # Rbf with thin-plate is what we used to use, but it's slower and
        # looks about the same:
        #
        #     zi = Rbf(x, y, v, function='multiquadric', smooth=0)(xi, yi)
        #
        # Eventually we could also do set_values with this class if we want,
        # see scipy/interpolate/rbf.py, especially the self.nodes one-liner.
        if isinstance(self.border, str):
            # we've already checked that border = 'mean'
            n_points = v.shape[0]
            v_extra = np.zeros(self.n_extra)
            indices, indptr = self.tri.vertex_neighbor_vertices
            rng = range(n_points, n_points + self.n_extra)
            used = np.zeros(len(rng), bool)
            for idx, extra_idx in enumerate(rng):
                ngb = indptr[indices[extra_idx] : indices[extra_idx + 1]]
                ngb = ngb[ngb < n_points]
                if len(ngb) > 0:
                    used[idx] = True
                    v_extra[idx] = v[ngb].mean()
            if not used.all() and used.any():
                # Eventually we might want to use the value of the nearest
                # point or something, but this case should hopefully be
                # rare so for now just use the average value of all extras
                v_extra[~used] = np.mean(v_extra[used])
        else:
            v_extra = np.full(self.n_extra, self.border, dtype=float)

        v = np.concatenate((v, v_extra))
        self.interpolator = self.interp(self.tri, v)
        return self

    def set_locations(self, Xi, Yi):
        """Set locations for easier (delayed) calling."""
        self.Xi = Xi
        self.Yi = Yi
        return self

    def __call__(self, *args):
        """Evaluate the interpolator."""
        if len(args) == 0:
            args = [self.Xi, self.Yi]
        return self.interpolator(*args)


def _topomap_plot_sensors(pos_x, pos_y, sensors, ax):
    """Plot sensors."""
    if sensors is True:
        ax.scatter(
            pos_x,
            pos_y,
            s=0.25,
            marker="o",
            edgecolor=["k"] * len(pos_x),
            facecolor="none",
        )
    else:
        ax.plot(pos_x, pos_y, sensors)


def _get_pos_outlines(info, picks, sphere, to_sphere=True):
    from ..channels.layout import _find_topomap_coords

    picks = _picks_to_idx(info, picks, "all", exclude=(), allow_empty=False)
    ch_type = _get_plot_ch_type(pick_info(_simplify_info(info), picks), None)
    orig_sphere = sphere
    sphere, clip_origin = _adjust_meg_sphere(sphere, info, ch_type)
    logger.debug(
        "Generating pos outlines with sphere "
        f"{sphere} from {orig_sphere} for {ch_type}"
    )
    pos = _find_topomap_coords(
        info, picks, ignore_overlap=True, to_sphere=to_sphere, sphere=sphere
    )
    outlines = _make_head_outlines(sphere, pos, "head", clip_origin)
    return pos, outlines


@fill_doc
def plot_topomap(
    data,
    pos,
    *,
    ch_type="eeg",
    sensors=True,
    names=None,
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
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    axes=None,
    show=True,
    onselect=None,
):
    """Plot a topographic map as image.

    Parameters
    ----------
    data : array, shape (n_chan,)
        The data values to plot.
    %(pos_topomap)s
        If an :class:`~mne.Info` object it must contain only one channel type
        and exactly ``len(data)`` channels; the x/y coordinates will
        be inferred from the montage in the :class:`~mne.Info` object.
    %(ch_type_topomap)s

        .. versionadded:: 0.21
    %(sensors_topomap)s
    %(names_topomap)s
    %(mask_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionadded:: 0.18

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap)s

        .. versionadded:: 1.2
    %(cnorm)s

        .. versionadded:: 0.24
    %(axes_plot_topomap)s

        .. versionchanged:: 1.2
           If ``axes=None``, a new :class:`~matplotlib.figure.Figure` is
           created instead of plotting into the current axes.
    %(show)s
    onselect : callable | None
        A function to be called when the user selects a set of channels by
        click-dragging (uses a matplotlib
        :class:`~matplotlib.widgets.RectangleSelector`). If ``None``
        interactive channel selection is disabled. Defaults to ``None``.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if axes is None:
        _, axes = plt.subplots(figsize=(size, size), layout="constrained")
    sphere = _check_sphere(sphere, pos if isinstance(pos, Info) else None)
    _validate_type(cnorm, (Normalize, None), "cnorm")
    if cnorm is not None and (vlim[0] is not None or vlim[1] is not None):
        warn(
            f"Provided cnorm implicitly defines vmin={cnorm.vmin} and "
            f"vmax={cnorm.vmax}; ignoring additional vlim/vmin/vmax params."
        )
    return _plot_topomap(
        data,
        pos,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap=cmap,
        sensors=sensors,
        res=res,
        axes=axes,
        names=names,
        mask=mask,
        mask_params=mask_params,
        outlines=outlines,
        contours=contours,
        image_interp=image_interp,
        show=show,
        onselect=onselect,
        extrapolate=extrapolate,
        sphere=sphere,
        border=border,
        ch_type=ch_type,
        cnorm=cnorm,
    )[:2]


def _setup_interp(pos, res, image_interp, extrapolate, outlines, border):
    if image_interp not in ("cubic", "linear", "nearest"):
        raise RuntimeError(
            "`image_interp` must be `cubic`, `linear` or `nearest`, got "
            f"{image_interp}. Previously, `image_interp` controlled "
            "the matplotlib image interpolation after an original cubic "
            "interpolation but this was changed to control the first "
            "interpolation instead for simplicity. To change the "
            "matplotlib image interpolation, use "
            "`im.set_interpolation(...)"
        )
    logger.debug(
        f"Interpolation mode {image_interp}, "
        f"extrapolation mode {extrapolate} to {border}"
    )
    xlim = (
        np.inf,
        -np.inf,
    )
    ylim = (
        np.inf,
        -np.inf,
    )
    mask_ = np.c_[outlines["mask_pos"]]
    clip_radius = outlines["clip_radius"]
    clip_origin = outlines.get("clip_origin", (0.0, 0.0))
    xmin, xmax = (
        np.min(np.r_[xlim[0], mask_[:, 0], clip_origin[0] - clip_radius[0]]),
        np.max(np.r_[xlim[1], mask_[:, 0], clip_origin[0] + clip_radius[0]]),
    )
    ymin, ymax = (
        np.min(np.r_[ylim[0], mask_[:, 1], clip_origin[1] - clip_radius[1]]),
        np.max(np.r_[ylim[1], mask_[:, 1], clip_origin[1] + clip_radius[1]]),
    )
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = _GridData(pos, image_interp, extrapolate, clip_origin, clip_radius, border)
    extent = (xmin, xmax, ymin, ymax)
    return extent, Xi, Yi, interp


_VORONOI_CIRCLE_RES = 100


def _voronoi_topomap(data, pos, outlines, ax, cmap, norm, extent, res):
    """Make a Voronoi diagram on a topomap."""
    # we need an image axis object so first empty image to plot over
    im = ax.imshow(
        np.zeros((res, res)) * np.nan,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        extent=extent,
        norm=norm,
    )
    rx, ry = outlines["clip_radius"]
    cx, cy = outlines.get("clip_origin", (0.0, 0.0))
    # add points on the circle to make boundaries, expand out to
    # ensure regions extend to the edge of the topomap
    vor = Voronoi(
        np.concatenate(
            [
                pos,
                [
                    (
                        rx * 1.5 * np.cos(2 * np.pi / _VORONOI_CIRCLE_RES * t),
                        ry * 1.5 * np.sin(2 * np.pi / _VORONOI_CIRCLE_RES * t),
                    )
                    for t in range(_VORONOI_CIRCLE_RES)
                ],
            ]
        )
    )
    for point_idx, region_idx in enumerate(vor.point_region[:-_VORONOI_CIRCLE_RES]):
        if -1 in vor.regions[region_idx]:
            continue
        polygon = list()
        for i in vor.regions[region_idx]:
            x, y = vor.vertices[i]
            if (x - cx) ** 2 / rx**2 + (y - cy) ** 2 / ry**2 < 1:
                polygon.append((x, y))
            else:
                x *= rx / np.linalg.norm(vor.vertices[i])
                y *= ry / np.linalg.norm(vor.vertices[i])
                polygon.append((x, y))
        ax.fill(*zip(*polygon), color=cmap(norm(data[point_idx])))
    return im


def _get_patch(outlines, extrapolate, interp, ax):
    from matplotlib import patches

    clip_radius = outlines["clip_radius"]
    clip_origin = outlines.get("clip_origin", (0.0, 0.0))
    _use_default_outlines = any(k.startswith("head") for k in outlines)
    patch_ = None
    if "patch" in outlines:
        patch_ = outlines["patch"]
        patch_ = patch_() if callable(patch_) else patch_
        patch_.set_clip_on(False)
        ax.add_patch(patch_)
        ax.set_transform(ax.transAxes)
        ax.set_clip_path(patch_)
    if _use_default_outlines:
        if extrapolate == "local":
            patch_ = patches.Polygon(
                interp.mask_pts, clip_on=True, transform=ax.transData
            )
        else:
            patch_ = patches.Ellipse(
                clip_origin,
                2 * clip_radius[0],
                2 * clip_radius[1],
                clip_on=True,
                transform=ax.transData,
            )
    return patch_


def _plot_topomap(
    data,
    pos,
    axes,
    *,
    ch_type="eeg",
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    cmap=None,
    vmin=None,
    vmax=None,
    cnorm=None,
    show=True,
    onselect=None,
):
    from matplotlib.colors import Normalize
    from matplotlib.widgets import RectangleSelector

    from ..channels.layout import (
        _find_topomap_coords,
        _merge_ch_data,
        _pair_grad_sensors,
    )

    data = np.asarray(data)
    logger.debug(f"Plotting topomap for {ch_type} data shape {data.shape}")

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos, exclude=())  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = pos.get_channel_types(picks=None, unique=True)
        info_help = "Pick Info with e.g. mne.pick_info and mne.channel_indices_by_type."
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " + info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError(
                f"Number of channels in the Info object ({len(pos['chs'])}) and the "
                f"data array ({data.shape[0]}) do not match." + info_help
            )
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ("planar", "grad")):
            # deal with grad pairs
            picks = _pair_grad_sensors(pos, topomap_coords=False)
            pos = _find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = _merge_ch_data(data[picks], ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)

    extrapolate = _check_extrapolate(extrapolate, ch_type)
    if data.ndim > 1:
        raise ValueError(
            f"Data needs to be array of shape (n_sensors,); got shape {data.shape}."
        )

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = (
        "Electrode positions should be specified as a 2D array with "
        "shape (n_channels, 2). Each row in this matrix contains the "
        "(x, y) position of an electrode."
    )
    if pos.ndim != 2:
        error = (
            f"{pos.ndim}D array supplied as electrode positions, where a 2D array was "
            "expected"
        )
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = (
            "The supplied electrode positions matrix contains 3 columns. "
            "Are you trying to specify XYZ coordinates? Perhaps the "
            "mne.channels.create_eeg_layout function is useful for you."
        )
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]

    if len(data) != len(pos):
        raise ValueError(
            "Data and pos need to be of same length. Got data of "
            f"length {len(data)}, pos of length { len(pos)}"
        )

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = "Reds" if norm else "RdBu_r"
    cmap = _get_cmap(cmap)

    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    assert isinstance(outlines, dict)

    _prepare_topomap(pos, axes)

    mask_params = _handle_default("mask_params", mask_params)

    # find mask limits and setup interpolation
    extent, Xi, Yi, interp = _setup_interp(
        pos, res, image_interp, extrapolate, outlines, border
    )
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = _get_patch(outlines, extrapolate, interp, axes)

    # get colormap normalization
    if cnorm is None:
        cnorm = Normalize(vmin=vmin, vmax=vmax)

    # plot interpolated map
    if image_interp == "nearest":  # plot over with Voronoi, more accurate
        im = _voronoi_topomap(
            data,
            pos=pos,
            outlines=outlines,
            ax=axes,
            cmap=cmap,
            norm=cnorm,
            extent=extent,
            res=res,
        )
    else:
        im = axes.imshow(
            Zi,
            cmap=cmap,
            origin="lower",
            aspect="equal",
            extent=extent,
            interpolation="bilinear",
            norm=cnorm,
        )

    # gh-1432 had a workaround for no contours here, but we'll remove it
    # because mpl has probably fixed it
    linewidth = mask_params["markeredgewidth"]
    cont = True
    if isinstance(contours, np.ndarray | list):
        pass
    elif contours == 0 or ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        cont = None  # can't make contours for constant-valued functions
    if cont:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            cont = axes.contour(
                Xi, Yi, Zi, contours, colors="k", linewidths=linewidth / 2.0
            )

    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in _cont_collections(cont):
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    mask = mask.astype(bool, copy=False) if mask is not None else None
    if sensors is not False and mask is None:
        _topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=axes)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=axes)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(axes, outlines)

    if names is not None:
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (_pos, _name) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            axes.text(
                _pos[0],
                _pos[1],
                _name,
                horizontalalignment="center",
                verticalalignment="center",
                size="x-small",
            )

    if onselect is not None:
        lim = axes.dataLim
        x0, y0, width, height = lim.x0, lim.y0, lim.width, lim.height
        axes.RS = RectangleSelector(axes, onselect=onselect)
        axes.set(xlim=[x0, x0 + width], ylim=[y0, y0 + height])
    plt_show(show)
    return im, cont, interp


def _plot_ica_topomap(
    ica,
    idx=0,
    ch_type=None,
    res=64,
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
    colorbar=False,
    title=None,
    show=True,
    outlines="head",
    contours=6,
    image_interp=_INTERPOLATION_DEFAULT,
    axes=None,
    sensors=True,
    allow_ref_meg=False,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    sphere=None,
    border=_BORDER_DEFAULT,
):
    """Plot single ica map to axes."""
    from matplotlib.axes import Axes

    from ..channels.layout import _merge_ch_data

    if ica.info is None:
        raise RuntimeError(
            "The ICA's measurement info is missing. Please "
            "fit the ICA or add the corresponding info object."
        )
    sphere = _check_sphere(sphere, ica.info)
    if not isinstance(axes, Axes):
        raise ValueError(
            "axis has to be an instance of matplotlib Axes, "
            f"got {type(axes)} instead."
        )
    ch_type = _get_plot_ch_type(ica, ch_type, allow_ref_meg=ica.allow_ref_meg)
    if ch_type == "ref_meg":
        logger.info("Cannot produce topographies for MEG reference channels.")
        return

    data = ica.get_components()[:, idx]
    (
        data_picks,
        pos,
        merge_channels,
        names,
        _,
        sphere,
        clip_origin,
    ) = _prepare_topomap_plot(ica, ch_type, sphere=sphere)
    data = data[data_picks]
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

    if merge_channels:
        data, names = _merge_ch_data(data, ch_type, names)

    topo_title = ica._ica_names[idx]
    if len(set(ica.get_channel_types())) > 1:
        topo_title += f" ({ch_type})"
    axes.set_title(topo_title, fontsize=12)
    vlim = _setup_vmin_vmax(data, vmin, vmax)
    im = plot_topomap(
        data.ravel(),
        pos,
        vlim=vlim,
        res=res,
        axes=axes,
        cmap=cmap,
        outlines=outlines,
        contours=contours,
        sensors=sensors,
        image_interp=image_interp,
        show=show,
        extrapolate=extrapolate,
        sphere=sphere,
        border=border,
        ch_type=ch_type,
    )[0]
    if colorbar:
        cbar, cax = _add_colorbar(
            axes,
            im,
            cmap,
            title="AU",
            format_="%3.2f",
            kind="ica_topomap",
            ch_type=ch_type,
        )
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks(vlim)
    _hide_frame(axes)


@verbose
def plot_ica_components(
    ica,
    picks=None,
    ch_type=None,
    *,
    inst=None,
    plot_std=True,
    reject="auto",
    sensors=True,
    show_names=False,
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
    colorbar=False,
    cbar_fmt="%3.2f",
    axes=None,
    title=None,
    nrows="auto",
    ncols="auto",
    show=True,
    image_args=None,
    psd_args=None,
    verbose=None,
):
    """Project mixing matrix on interpolated sensor topography.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    %(picks_ica)s
    %(ch_type_topomap)s
    inst : Raw | Epochs | None
        To be able to see component properties after clicking on component
        topomap you need to pass relevant data - instances of Raw or Epochs
        (for example the data that ICA was trained on). This takes effect
        only when running matplotlib in interactive mode.
    plot_std : bool | float
        Whether to plot standard deviation in ERP/ERF and spectrum plots.
        Defaults to True, which plots one standard deviation above/below.
        If set to float allows to control how many standard deviations are
        plotted. For example 2.5 will plot 2.5 standard deviation above/below.
    reject : ``'auto'`` | dict | None
        Allows to specify rejection parameters used to drop epochs
        (or segments if continuous signal is passed as inst).
        If None, no rejection is applied. The default is 'auto',
        which applies the rejection parameters used when fitting
        the ICA object.
    %(sensors_topomap)s
    %(show_names_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionadded:: 1.3
    %(border_topomap)s

        .. versionadded:: 1.3
    %(res_topomap)s
    %(size_topomap)s

        .. versionadded:: 1.3
    %(cmap_topomap)s
    %(vlim_plot_topomap)s

        .. versionadded:: 1.3
    %(cnorm)s

        .. versionadded:: 1.3
    %(colorbar_topomap)s
    %(cbar_fmt_topomap)s
    axes : Axes | array of Axes | None
        The subplot(s) to plot to. Either a single Axes or an iterable of Axes
        if more than one subplot is needed. The number of subplots must match
        the number of selected components. If None, new figures will be created
        with the number of subplots per figure controlled by ``nrows`` and
        ``ncols``.
    title : str | None
        The title of the generated figure. If ``None`` (default) and
        ``axes=None``, a default title of "ICA Components" will be used.
    %(nrows_ncols_ica_components)s

        .. versionadded:: 1.3
    %(show)s
    image_args : dict | None
        Dictionary of arguments to pass to :func:`~mne.viz.plot_epochs_image`
        in interactive mode. Ignored if ``inst`` is not supplied. If ``None``,
        nothing is passed. Defaults to ``None``.
    psd_args : dict | None
        Dictionary of arguments to pass to :meth:`~mne.Epochs.compute_psd` in
        interactive  mode. Ignored if ``inst`` is not supplied. If ``None``,
        nothing is passed. Defaults to ``None``.
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure | list of matplotlib.figure.Figure
        The figure object(s).

    Notes
    -----
    When run in interactive mode, ``plot_ica_components`` allows to reject
    components by clicking on their title label. The state of each component
    is indicated by its label color (gray: rejected; black: retained). It is
    also possible to open component properties by clicking on the component
    topomap (this option is only available when the ``inst`` argument is
    supplied).
    """  # noqa E501
    from matplotlib.pyplot import Axes

    from ..channels.layout import _merge_ch_data
    from ..epochs import BaseEpochs
    from ..io import BaseRaw

    if ica.info is None:
        raise RuntimeError(
            "The ICA's measurement info is missing. Please "
            "fit the ICA or add the corresponding info object."
        )

    # for backward compat, nrow='auto' ncol='auto' should yield 4 rows 5 cols
    # and create multiple figures if more than 20 components requested
    if nrows == "auto" and ncols == "auto":
        ncols = 5
        max_subplots = 20
    elif nrows == "auto" or ncols == "auto":
        # user provided incomplete row/col spec; put all in one figure
        max_subplots = ica.n_components_
    else:
        max_subplots = nrows * ncols

    # handle ch_type=None
    ch_type = _get_plot_ch_type(ica, ch_type)

    figs = []
    if picks is None:
        cut_points = range(max_subplots, ica.n_components_, max_subplots)
        pick_groups = np.split(range(ica.n_components_), cut_points)
    else:
        pick_groups = [_picks_to_idx(ica.n_components_, picks, picks_on="components")]

    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    for k, picks in enumerate(pick_groups):
        try:  # either an iterable, 1D numpy array or others
            _axes = axes[k * max_subplots : (k + 1) * max_subplots]
        except TypeError:  # None or Axes
            _axes = axes

        (
            data_picks,
            pos,
            merge_channels,
            names,
            ch_type,
            sphere,
            clip_origin,
        ) = _prepare_topomap_plot(ica, ch_type, sphere=sphere)

        cmap = _setup_cmap(cmap, n_axes=len(picks))
        names = _prepare_sensor_names(names, show_names)
        outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

        data = np.dot(
            ica.mixing_matrix_[:, picks].T, ica.pca_components_[: ica.n_components_]
        )
        data = np.atleast_2d(data)
        data = data[:, data_picks]

        if title is None:
            title = "ICA components"
        user_passed_axes = _axes is not None
        if not user_passed_axes:
            fig, _axes, _, _ = _prepare_trellis(len(data), ncols=ncols, nrows=nrows)
            fig.suptitle(title)
        else:
            _axes = [_axes] if isinstance(_axes, Axes) else _axes
            fig = _axes[0].get_figure()

        subplot_titles = list()
        for ii, data_, ax in zip(picks, data, _axes):
            kwargs = dict(color="gray") if ii in ica.exclude else dict()
            comp_title = ica._ica_names[ii]
            if len(set(ica.get_channel_types())) > 1:
                comp_title += f" ({ch_type})"
            subplot_titles.append(ax.set_title(comp_title, fontsize=12, **kwargs))
            if merge_channels:
                data_, names_ = _merge_ch_data(data_, ch_type, copy.copy(names))
            # ↓↓↓ NOTE: we intentionally use the default norm=False here, so that
            # ↓↓↓ we get vlims that are symmetric-about-zero, even if the data for
            # ↓↓↓ a given component happens to be one-sided.
            _vlim = _setup_vmin_vmax(data_, *vlim)
            im = plot_topomap(
                data_.flatten(),
                pos,
                ch_type=ch_type,
                sensors=sensors,
                names=names,
                contours=contours,
                outlines=outlines,
                sphere=sphere,
                image_interp=image_interp,
                extrapolate=extrapolate,
                border=border,
                res=res,
                size=size,
                cmap=cmap[0],
                vlim=_vlim,
                cnorm=cnorm,
                axes=ax,
                show=False,
            )[0]

            im.axes.set_label(ica._ica_names[ii])
            if colorbar:
                cbar, cax = _add_colorbar(
                    ax,
                    im,
                    cmap,
                    title="AU",
                    format_=cbar_fmt,
                    kind="ica_comp_topomap",
                    ch_type=ch_type,
                )
                cbar.ax.tick_params(labelsize=12)
                cbar.set_ticks(_vlim)
            _hide_frame(ax)
        del pos
        fig.canvas.draw()

        # add title selection interactivity
        def onclick_title(event, ica=ica, titles=subplot_titles, fig=fig):
            # check if any title was pressed
            title_pressed = None
            for title in titles:
                if title.contains(event)[0]:
                    title_pressed = title
                    break
            # title was pressed -> identify the IC
            if title_pressed is not None:
                label = title_pressed.get_text()
                ic = int(label.split(" ")[0][-3:])
                # add or remove IC from exclude depending on current state
                if ic in ica.exclude:
                    ica.exclude.remove(ic)
                    title_pressed.set_color("k")
                else:
                    ica.exclude.append(ic)
                    title_pressed.set_color("gray")
                fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick_title)

        # add plot_properties interactivity only if inst was passed
        if isinstance(inst, BaseRaw | BaseEpochs):
            topomap_args = dict(
                sensors=sensors,
                contours=contours,
                outlines=outlines,
                sphere=sphere,
                image_interp=image_interp,
                extrapolate=extrapolate,
                border=border,
                res=res,
                cmap=cmap[0],
                vmin=vlim[0],
                vmax=vlim[1],
            )

            def onclick_topo(event, ica=ica, inst=inst):
                # check which component to plot
                if event.inaxes is not None:
                    label = event.inaxes.get_label()
                    if label.startswith("ICA"):
                        ic = int(label.split(" ")[0][-3:])
                        ica.plot_properties(
                            inst,
                            picks=ic,
                            show=True,
                            plot_std=plot_std,
                            topomap_args=topomap_args,
                            image_args=image_args,
                            psd_args=psd_args,
                            reject=reject,
                        )

            fig.canvas.mpl_connect("button_press_event", onclick_topo)
        figs.append(fig)

    plt_show(show)
    return figs[0] if len(figs) == 1 else figs


@fill_doc
def plot_tfr_topomap(
    tfr,
    tmin=None,
    tmax=None,
    fmin=0.0,
    fmax=np.inf,
    *,
    ch_type=None,
    baseline=None,
    mode="mean",
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
    size=2,
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=True,
    cbar_fmt="%1.1e",
    units=None,
    axes=None,
    show=True,
):
    """Plot topographic maps of specific time-frequency intervals of TFR data.

    Parameters
    ----------
    tfr : AverageTFR
        The AverageTFR object.
    %(tmin_tmax_psd)s
    %(fmin_fmax_psd)s
    %(ch_type_topomap_psd)s
    baseline : tuple or list of length 2
        The time interval to apply rescaling / baseline correction. If None do
        not apply it. If baseline is (a, b) the interval is between "a (s)" and
        "b (s)". If a is None the beginning of the data is used and if b is
        None then b is set to the end of the interval. If baseline is equal to
        (None, None) the whole time interval is used.
    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio' | None
        Perform baseline correction by

          - subtracting the mean baseline power ('mean')
          - dividing by the mean baseline power ('ratio')
          - dividing by the mean baseline power and taking the log ('logratio')
          - subtracting the mean baseline power followed by dividing by the
            mean baseline power ('percent')
          - subtracting the mean baseline power and dividing by the standard
            deviation of the baseline power ('zscore')
          - dividing by the mean baseline power, taking the log, and dividing
            by the standard deviation of the baseline power ('zlogratio')

        If None no baseline correction is applied.
    %(sensors_topomap)s
    %(show_names_topomap)s
    %(mask_evoked_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap)s

        .. versionadded:: 1.2
    %(cnorm)s

        .. versionadded:: 1.2
    %(colorbar_topomap)s
    %(cbar_fmt_topomap)s
    %(units_topomap)s
    %(axes_plot_topomap)s
    %(show)s

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the topography.
    """  # noqa: E501
    import matplotlib.pyplot as plt

    from ..channels.layout import _merge_ch_data

    ch_type = _get_plot_ch_type(tfr, ch_type)

    picks, pos, merge_channels, names, _, sphere, clip_origin = _prepare_topomap_plot(
        tfr, ch_type, sphere=sphere
    )
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)
    data = tfr.data[picks, :, :]

    # merging grads before rescaling makes ERDs visible
    if merge_channels:
        data, names = _merge_ch_data(data, ch_type, names, method="mean")

    data = rescale(data, tfr.times, baseline, mode, copy=True)

    if np.iscomplexobj(data):
        data = np.sqrt((data * data.conj()).real)

    # crop time
    itmin, itmax = None, None
    idx = np.where(_time_mask(tfr.times, tmin, tmax))[0]
    if tmin is not None:
        itmin = idx[0]
    if tmax is not None:
        itmax = idx[-1] + 1
    # crop freqs
    ifmin, ifmax = None, None
    idx = np.where(_time_mask(tfr.freqs, fmin, fmax))[0]
    ifmin = idx[0]
    ifmax = idx[-1] + 1

    data = data[:, ifmin:ifmax, itmin:itmax]
    data = data.mean(axis=(1, 2))[:, np.newaxis]
    norm = False if np.min(data) < 0 else True
    vlim = _setup_vmin_vmax(data, *vlim, norm)
    cmap = _setup_cmap(cmap, norm=norm)

    axes = (
        plt.subplots(figsize=(size, size), layout="constrained")[1]
        if axes is None
        else axes
    )
    fig = axes.figure

    _hide_frame(axes)

    locator = None
    if not isinstance(contours, list | np.ndarray):
        locator, contours = _set_contour_locator(*vlim, contours)

    fig_wrapper = list()
    selection_callback = partial(
        _onselect,
        tfr=tfr,
        pos=pos,
        ch_type=ch_type,
        itmin=itmin,
        itmax=itmax,
        ifmin=ifmin,
        ifmax=ifmax,
        cmap=cmap[0],
        fig=fig_wrapper,
    )

    if not isinstance(contours, list | np.ndarray):
        _, contours = _set_contour_locator(*vlim, contours)

    names = _prepare_sensor_names(names, show_names)

    im, _ = plot_topomap(
        data[:, 0],
        pos,
        ch_type=ch_type,
        sensors=sensors,
        names=names,
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
        cmap=cmap[0],
        vlim=vlim,
        cnorm=cnorm,
        axes=axes,
        show=False,
        onselect=selection_callback,
    )

    if colorbar:
        from matplotlib import ticker

        units = _handle_default("units", units)["misc"]
        cbar, cax = _add_colorbar(
            axes,
            im,
            cmap,
            title=units,
            format_=cbar_fmt,
            kind="tfr_topomap",
            ch_type=ch_type,
        )
        if locator is None:
            locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=12)

    plt_show(show)
    return fig


@fill_doc
def plot_evoked_topomap(
    evoked,
    times="auto",
    *,
    average=None,
    ch_type=None,
    scalings=None,
    proj=False,
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
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=True,
    cbar_fmt="%3.1f",
    units=None,
    axes=None,
    time_unit="s",
    time_format=None,
    nrows=1,
    ncols="auto",
    show=True,
):
    """Plot topographic maps of specific time points of evoked data.

    Parameters
    ----------
    evoked : Evoked
        The Evoked object.
    times : float | array of float | "auto" | "peaks" | "interactive"
        The time point(s) to plot. If "auto", the number of ``axes`` determines
        the amount of time point(s). If ``axes`` is also None, at most 10
        topographies will be shown with a regular time spacing between the
        first and last time instant. If "peaks", finds time points
        automatically by checking for local maxima in global field power. If
        "interactive", the time can be set interactively at run-time by using a
        slider.
    %(average_plot_evoked_topomap)s
    %(ch_type_topomap)s
    %(scalings_topomap)s
    %(proj_plot)s
    %(sensors_topomap)s
    %(show_names_topomap)s
    %(mask_evoked_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionadded:: 0.18

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap_psd)s

        .. versionadded:: 1.2
    %(cnorm)s

        .. versionadded:: 1.2
    %(colorbar_topomap)s
    %(cbar_fmt_topomap)s
    %(units_topomap_evoked)s
    %(axes_evoked_plot_topomap)s
    time_unit : str
        The units for the time axis, can be "ms" or "s" (default).

        .. versionadded:: 0.16
    time_format : str | None
        String format for topomap values. Defaults (None) to "%%01d ms" if
        ``time_unit='ms'``, "%%0.3f s" if ``time_unit='s'``, and
        "%%g" otherwise. Can be an empty string to omit the time label.
    %(nrows_ncols_topomap)s Ignored when times == 'interactive'.

        .. versionadded:: 0.20
    %(show)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
       The figure.

    Notes
    -----
    When existing ``axes`` are provided and ``colorbar=True``, note that the
    colorbar scale will only accurately reflect topomaps that are generated in
    the same call as the colorbar. Note also that the colorbar will not be
    resized automatically when ``axes`` are provided; use Matplotlib's
    :meth:`axes.set_position() <matplotlib.axes.Axes.set_position>` method or
    :ref:`gridspec <matplotlib:arranging_axes>` interface to adjust the colorbar
    size yourself.

    When ``time=="interactive"``, the figure will publish and subscribe to the
    following UI events:

    * :class:`~mne.viz.ui_events.TimeChange` whenever a new time is selected.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import Slider

    from ..channels.layout import _merge_ch_data
    from ..evoked import Evoked

    _validate_type(evoked, Evoked, "evoked")
    _validate_type(colorbar, bool, "colorbar")
    evoked = evoked.copy()  # make a copy, since we'll be picking
    ch_type = _get_plot_ch_type(evoked, ch_type)
    # time units / formatting
    time_unit, _ = _check_time_unit(time_unit, evoked.times)
    scaling_time = 1.0 if time_unit == "s" else 1e3
    _validate_type(time_format, (None, str), "time_format")
    if time_format is None:
        time_format = "%0.3f s" if time_unit == "s" else "%01d ms"
    del time_unit
    # mask_params defaults
    mask_params = _handle_default("mask_params", mask_params)
    mask_params["markersize"] *= size / 2.0
    mask_params["markeredgewidth"] *= size / 2.0
    # setup various parameters, and prepare outlines
    (
        picks,
        pos,
        merge_channels,
        names,
        ch_type,
        sphere,
        clip_origin,
    ) = _prepare_topomap_plot(evoked, ch_type, sphere=sphere)
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)
    # check interactive
    axes_given = axes is not None
    interactive = isinstance(times, str) and times == "interactive"
    if interactive and axes_given:
        raise ValueError("User-provided axes not allowed when times='interactive'.")
    # units, scalings
    key = "grad" if ch_type.startswith("planar") else ch_type
    default_scaling = _handle_default("scalings", None)[key]
    scaling = _handle_default("scalings", scalings)[key]
    # if non-default scaling, fall back to "AU" if unit wasn't given by user
    key = "misc" if scaling != default_scaling else key
    unit = _handle_default("units", units)[key]
    # ch_names (required for NIRS)
    ch_names = names
    names = _prepare_sensor_names(names, show_names)
    # apply projections before picking. NOTE: the `if proj is True`
    # anti-pattern is needed here to exclude proj='interactive'
    _check_option("proj", proj, (True, False, "interactive", "reconstruct"))
    if proj is True and not evoked.proj:
        evoked.apply_proj()
    elif proj == "reconstruct":
        evoked._reconstruct_proj()
    data = evoked.data

    # remove compensation matrices (safe: only plotting & already made copy)
    with evoked.info._unlock():
        evoked.info["comps"] = []
    evoked = evoked._pick_drop_channels(picks, verbose=False)
    # determine which times to plot
    if isinstance(axes, plt.Axes):
        axes = [axes]
    n_peaks = len(axes) - int(colorbar) if axes_given else None
    times = _process_times(evoked, times, n_peaks)
    n_times = len(times)
    space = 1 / (2.0 * evoked.info["sfreq"])
    if max(times) > max(evoked.times) + space or min(times) < min(evoked.times) - space:
        raise ValueError(
            f"Times should be between {evoked.times[0]:0.3} and "
            f"{evoked.times[-1]:0.3}."
        )
    # create axes
    want_axes = n_times + int(colorbar)
    if interactive:
        height_ratios = [5, 1]
        nrows = 2
        ncols = n_times
        width = size * want_axes
        height = size + max(0, 0.1 * (4 - size))
        fig = figure_nobar(figsize=(width * 1.5, height * 1.5))
        gs = GridSpec(nrows, ncols, height_ratios=height_ratios, figure=fig)
        axes = []
        for ax_idx in range(n_times):
            axes.append(plt.subplot(gs[0, ax_idx]))
    elif axes is None:
        fig, axes, ncols, nrows = _prepare_trellis(
            n_times, ncols=ncols, nrows=nrows, size=size
        )
    else:
        nrows, ncols = None, None  # Deactivate ncols when axes were passed
        fig = axes[0].get_figure()
        # check: enough space for colorbar?
        if len(axes) != want_axes:
            cbar_err = " plus one for the colorbar" if colorbar else ""
            raise RuntimeError(
                f"You must provide {want_axes} axes (one for "
                f"each time{cbar_err}), got {len(axes)}."
            )
    del want_axes
    # find first index that's >= (to rounding error) to each time point
    time_idx = [
        np.where(
            _time_mask(evoked.times, tmin=t, tmax=None, sfreq=evoked.info["sfreq"])
        )[0][0]
        for t in times
    ]
    # do averaging if requested
    avg_err = (
        '"average" must be `None`, a positive number of seconds, or '
        "an array-like object of the previous"
    )

    averaged_times = []
    if average is None:
        average = np.array([None] * n_times)
        data = data[np.ix_(picks, time_idx)]
    else:
        if _is_numeric(average):
            average = np.array([average] * n_times)
        elif np.array(average).ndim == 0:
            # It should be an array-like object
            raise TypeError(f"{avg_err}; got type: {type(average)}.")
        else:
            average = np.array(average)

        if len(average) != n_times:
            raise ValueError(
                f"You requested to plot topographic maps for {n_times} time "
                f"points, but provided {len(average)} periods for "
                f"averaging. The number of time points and averaging periods "
                f"must be equal."
            )
        data_ = np.zeros((len(picks), len(time_idx)))

        for average_idx, (this_average, this_time, this_time_idx) in enumerate(
            zip(average, evoked.times[time_idx], time_idx)
        ):
            if (_is_numeric(this_average) and this_average <= 0) or (
                not _is_numeric(this_average) and this_average is not None
            ):
                if len(average) == 1:
                    msg = f"{avg_err}; got {this_average}"
                else:
                    msg = f"{avg_err}; got {this_average} in {average}"
                raise ValueError(msg)

            if this_average is None:
                data_[:, average_idx] = data[picks][:, this_time_idx]
                averaged_times.append([this_time])
            else:
                tmin_ = this_time - this_average / 2
                tmax_ = this_time + this_average / 2
                time_mask = (tmin_ < evoked.times) & (evoked.times < tmax_)
                data_[:, average_idx] = data[picks][:, time_mask].mean(-1)
                averaged_times.append(evoked.times[time_mask])
        data = data_

    # apply scalings and merge channels
    data *= scaling
    if merge_channels:
        data, ch_names = _merge_ch_data(data, ch_type, ch_names)
        if ch_type in _fnirs_types:
            merge_channels = False
    # apply mask if requested
    if mask is not None:
        mask = mask.astype(bool, copy=False)
        if ch_type == "grad":
            mask_ = (
                mask[np.ix_(picks[::2], time_idx)] | mask[np.ix_(picks[1::2], time_idx)]
            )
        else:  # mag, eeg, planar1, planar2
            mask_ = mask[np.ix_(picks, time_idx)]
    # set up colormap
    _vlim = [
        _setup_vmin_vmax(data[:, i], *vlim, norm=merge_channels) for i in range(n_times)
    ]
    _vlim = (np.min(_vlim), np.max(_vlim))
    cmap = _setup_cmap(cmap, n_axes=n_times, norm=_vlim[0] >= 0)
    # set up contours
    if not isinstance(contours, list | np.ndarray):
        _, contours = _set_contour_locator(*_vlim, contours)
    # prepare for main loop over times
    kwargs = dict(
        sensors=sensors,
        res=res,
        names=names,
        cmap=cmap[0],
        cnorm=cnorm,
        mask_params=mask_params,
        outlines=outlines,
        contours=contours,
        image_interp=image_interp,
        show=False,
        extrapolate=extrapolate,
        sphere=sphere,
        border=border,
        ch_type=ch_type,
    )
    images, contours_ = [], []
    # loop over times
    for average_idx, (time, this_average) in enumerate(zip(times, average)):
        tp, cn, interp = _plot_topomap(
            data[:, average_idx],
            pos,
            axes=axes[average_idx],
            mask=mask_[:, average_idx] if mask is not None else None,
            vmin=_vlim[0],
            vmax=_vlim[1],
            **kwargs,
        )

        images.append(tp)
        if cn is not None:
            contours_.append(cn)
        if time_format != "":
            if this_average is None:
                axes_title = time_format % (time * scaling_time)
            else:
                tmin_ = averaged_times[average_idx][0]
                tmax_ = averaged_times[average_idx][-1]
                from_time = time_format % (tmin_ * scaling_time)
                from_time = from_time.split(" ")[0]  # Remove unit
                to_time = time_format % (tmax_ * scaling_time)
                axes_title = f"{from_time} – {to_time}"
                del from_time, to_time, tmin_, tmax_
            axes[average_idx].set_title(axes_title)

    if interactive:
        # Add a slider to the figure and start publishing and subscribing to time_change
        # events.
        kwargs.update(vlim=_vlim)
        axes.append(fig.add_subplot(gs[1]))
        slider = Slider(
            axes[-1],
            "Time",
            evoked.times[0],
            evoked.times[-1],
            valinit=times[0],
            valfmt="%1.2fs",
        )
        slider.vline.remove()  # remove initial point indicator
        func = _merge_ch_data if merge_channels else lambda x: x

        def _slider_changed(val):
            publish(fig, TimeChange(time=val))

        slider.on_changed(_slider_changed)
        ts = np.tile(evoked.times, len(evoked.data)).reshape(evoked.data.shape)
        axes[-1].plot(ts, evoked.data, color="k")
        axes[-1].slider = slider

        subscribe(
            fig,
            "time_change",
            partial(
                _on_time_change,
                fig=fig,
                data=evoked.data,
                times=evoked.times,
                pos=pos,
                scaling=scaling,
                func=func,
                time_format=time_format,
                scaling_time=scaling_time,
                slider=slider,
                kwargs=kwargs,
            ),
        )
        subscribe(
            fig,
            "colormap_range",
            partial(_on_colormap_range, kwargs=kwargs),
        )

    if colorbar:
        if nrows is None or ncols is None:
            # axes were given by the user, so don't resize the colorbar
            cax = axes[-1]
        else:  # use the default behavior
            cax = None

        cbar = fig.colorbar(images[-1], ax=axes, cax=cax, format=cbar_fmt, shrink=0.6)
        if unit is not None:
            cbar.ax.set_title(unit)
        if cn is not None:
            cbar.set_ticks(contours)
        cbar.ax.tick_params(labelsize=7)
        if cmap[1]:
            for im in images:
                im.axes.CB = DraggableColorbar(
                    cbar, im, kind="evoked_topomap", ch_type=ch_type
                )

    if proj == "interactive":
        _check_delayed_ssp(evoked)
        params = dict(
            evoked=evoked,
            fig=fig,
            projs=evoked.info["projs"],
            picks=picks,
            images=images,
            contours_=contours_,
            pos=pos,
            time_idx=time_idx,
            res=res,
            plot_update_proj_callback=_plot_update_evoked_topomap,
            merge_channels=merge_channels,
            scale=scaling,
            axes=axes[: len(axes) - bool(interactive)],
            contours=contours,
            interp=interp,
            extrapolate=extrapolate,
        )
        _draw_proj_checkbox(None, params)
        # This is mostly for testing purposes, but it's also consistent with
        # raw.plot, so maybe not a bad thing in principle either
        from mne.viz._figure import BrowserParams

        fig.mne = BrowserParams(proj_checkboxes=params["proj_checks"])

    plt_show(show, block=False)
    if axes_given:
        fig.canvas.draw()
    return fig


def _resize_cbar(cax, n_fig_axes, size=1):
    """Resize colorbar."""
    cpos = cax.get_position()
    if size <= 1:
        cpos.x0 = 1 - (0.7 + 0.1 / size) / n_fig_axes
    cpos.x1 = cpos.x0 + 0.1 / n_fig_axes
    cpos.y0 = 0.2
    cpos.y1 = 0.7
    cax.set_position(cpos)


def _on_time_change(
    event,
    fig,
    data,
    times,
    pos,
    scaling,
    func,
    time_format,
    scaling_time,
    slider,
    kwargs,
):
    """Handle updating topomap to show a new time."""
    idx = np.argmin(np.abs(times - event.time))
    data = func(data[:, idx]).ravel() * scaling
    ax = fig.axes[0]
    ax.clear()
    im, _ = plot_topomap(data, pos, axes=ax, **kwargs)
    if hasattr(ax, "CB"):
        ax.CB.mappable = im
        _resize_cbar(ax.CB.cbar.ax, 2)
    if time_format is not None:
        ax.set_title(time_format % (event.time * scaling_time))
    # Updating the slider will generate a new time_change event. To prevent an
    # infinite loop, only update the slider if the time has actually changed.
    if event.time != slider.val:
        slider.set_val(event.time)
    ax.figure.canvas.draw_idle()


def _on_colormap_range(event, kwargs):
    """Handle updating colormap range."""
    logger.debug(f"Updating colormap range to {event.fmin}, {event.fmax}")
    kwargs.update(vlim=(event.fmin, event.fmax), cmap=event.cmap)


def _plot_topomap_multi_cbar(
    data,
    pos,
    ax,
    *,
    vlim,
    title,
    unit,
    cmap,
    outlines,
    colorbar,
    cbar_fmt,
    sphere,
    ch_type,
    sensors,
    names,
    mask,
    mask_params,
    contours,
    image_interp,
    extrapolate,
    border,
    res,
    size,
    cnorm,
):
    _hide_frame(ax)
    _vlim = (
        np.min(data) if vlim[0] is None else vlim[0],
        np.max(data) if vlim[1] is None else vlim[1],
    )
    # this definition of "norm" allows non-diverging colormap for cases
    # where min & vmax are both negative (e.g., when they are power in dB)
    signs = np.sign(_vlim)
    norm = len(set(signs)) == 1 or np.any(signs == 0)

    _cmap = _setup_cmap(cmap, norm=norm)
    if title is not None:
        ax.set_title(title, fontsize=10)
    im, _ = plot_topomap(
        data,
        pos,
        ch_type=ch_type,
        sensors=sensors,
        names=names,
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
        cmap=_cmap[0],
        vlim=_vlim,
        cnorm=cnorm,
        axes=ax,
        show=False,
        onselect=None,
    )

    if colorbar:
        cbar, cax = _add_colorbar(ax, im, cmap, title=None, format_=cbar_fmt)
        cbar.set_ticks(_vlim)
        if unit is not None:
            cbar.ax.set_ylabel(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)


@legacy(alt="Epochs.compute_psd().plot_topomap()")
@verbose
def plot_epochs_psd_topomap(
    epochs,
    bands=None,
    tmin=None,
    tmax=None,
    proj=False,
    *,
    bandwidth=None,
    adaptive=False,
    low_bias=True,
    normalization="length",
    ch_type=None,
    normalize=False,
    agg_fun=None,
    dB=False,
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=0,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    size=1,
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=True,
    cbar_fmt="auto",
    units=None,
    axes=None,
    show=True,
    n_jobs=None,
    verbose=None,
):
    """Plot the topomap of the power spectral density across epochs.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object.
    %(bands_psd_topo)s
    %(tmin_tmax_psd)s
    %(proj_psd)s
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4 Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    %(normalization)s
    %(ch_type_topomap_psd)s
    %(normalize_psd_topo)s
    %(agg_fun_psd_topo)s
    %(dB_plot_topomap)s
    %(sensors_topomap)s
    %(names_topomap)s
    %(mask_evoked_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap_psd)s

        .. versionadded:: 0.21
    %(cnorm)s

        .. versionadded:: 1.2
    %(colorbar_topomap)s
    %(cbar_fmt_topomap_psd)s
    %(units_topomap)s
    %(axes_spectrum_plot_topomap)s
    %(show)s
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure showing one scalp topography per frequency band.
    """
    from ..channels import rename_channels
    from ..time_frequency import Spectrum

    init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot_topomap)
    spectrum = epochs.compute_psd(**init_kw)
    plot_kw.setdefault("show_names", False)
    if names is not None:
        rename_channels(
            spectrum.info, dict(zip(spectrum.ch_names, names)), verbose=verbose
        )
        plot_kw["show_names"] = True
    return spectrum.plot_topomap(**plot_kw)


@fill_doc
def plot_psds_topomap(
    psds,
    freqs,
    pos,
    *,
    bands=None,
    ch_type="eeg",
    normalize=False,
    agg_fun=None,
    dB=True,
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=0,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    size=1,
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    colorbar=True,
    cbar_fmt="auto",
    unit=None,
    axes=None,
    show=True,
):
    """Plot spatial maps of PSDs.

    Parameters
    ----------
    psds : array of float, shape (n_channels, n_freqs)
        Power spectral densities.
    freqs : array of float, shape (n_freqs,)
        Frequencies used to compute psds.
    %(pos_topomap_psd)s
    %(bands_psd_topo)s
    %(ch_type_topomap)s
    %(normalize_psd_topo)s
    %(agg_fun_psd_topo)s
    %(dB_plot_topomap)s
    %(sensors_topomap)s
    %(names_topomap)s
    %(mask_evoked_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap_psd)s

        .. versionadded:: 0.21
    %(cnorm)s

        .. versionadded:: 1.2
    %(colorbar_topomap)s
    %(cbar_fmt_topomap_psd)s
    unit : str | None
        Measurement unit to be displayed with the colorbar. If ``None``, no
        unit is displayed (only "power" or "dB" as appropriate).
    %(axes_spectrum_plot_topomap)s
    %(show)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure with a topomap subplot for each band.
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    # handle some defaults
    sphere = _check_sphere(sphere)
    if cbar_fmt == "auto":
        cbar_fmt = "%0.1f" if dB else "%0.3f"
    # make sure `bands` is a dict
    if bands is None:
        bands = {
            "Delta (0-4 Hz)": (0, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha (8-12 Hz)": (8, 12),
            "Beta (12-30 Hz)": (12, 30),
            "Gamma (30-45 Hz)": (30, 45),
        }
    elif not hasattr(bands, "keys"):
        # convert legacy list-of-tuple input to a dict
        bands = {band[-1]: band[:-1] for band in bands}
        logger.info(
            "converting legacy list-of-tuples input to a dict for the "
            "`bands` parameter"
        )
    # upconvert single freqs to band upper/lower edges as needed
    bin_spacing = np.diff(freqs)[0]
    bin_edges = np.array([0, bin_spacing]) - bin_spacing / 2
    for band, _edges in bands.items():
        if not hasattr(_edges, "__len__"):
            _edges = (_edges,)
        if len(_edges) == 1:
            bands[band] = tuple(bin_edges + freqs[np.argmin(np.abs(freqs - _edges[0]))])
    # normalize data (if requested)
    if normalize:
        psds /= psds.sum(axis=-1, keepdims=True)
        assert np.allclose(psds.sum(axis=-1), 1.0)
    # aggregate within bands
    if agg_fun is None:
        agg_fun = np.sum if normalize else np.mean
    freq_masks = list()
    for band, (fmin, fmax) in bands.items():
        _mask = (fmin < freqs) & (freqs < fmax)
        # make sure no bands are empty
        if _mask.sum() == 0:
            raise RuntimeError(f'No frequencies in band "{band}" ({fmin}, {fmax})')
        freq_masks.append(_mask)
    band_data = [agg_fun(psds[:, _mask], axis=1) for _mask in freq_masks]
    if dB and not normalize:
        band_data = [10 * np.log10(_d) for _d in band_data]
    # handle vmin/vmax
    joint_vlim = vlim == "joint"
    if joint_vlim:
        vlim = (np.array(band_data).min(), np.array(band_data).max())
    # unit label
    if unit is None:
        unit = "dB" if dB and not normalize else "power"
    else:
        _dB = dB and not normalize
        unit = _format_units_psd(unit, dB=_dB)
    # set up figure / axes
    n_axes = len(bands)
    user_passed_axes = axes is not None
    if user_passed_axes:
        if isinstance(axes, Axes):
            axes = [axes]
        _validate_if_list_of_axes(axes, n_axes)
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(
            1, n_axes, figsize=(2 * n_axes, 1.5), layout="constrained"
        )
        if n_axes == 1:
            axes = [axes]
    # loop over subplots/frequency bands
    for ax, _mask, _data, (title, (fmin, fmax)) in zip(
        axes, freq_masks, band_data, bands.items()
    ):
        plot_colorbar = False if not colorbar else (not joint_vlim) or ax == axes[-1]
        _plot_topomap_multi_cbar(
            _data,
            pos,
            ax,
            title=title,
            vlim=vlim,
            cmap=cmap,
            outlines=outlines,
            colorbar=plot_colorbar,
            unit=unit,
            cbar_fmt=cbar_fmt,
            sphere=sphere,
            ch_type=ch_type,
            sensors=sensors,
            names=names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cnorm=cnorm,
        )

    if not user_passed_axes:
        fig.canvas.draw()
        plt_show(show)
    return fig


@fill_doc
def plot_layout(layout, picks=None, show_axes=False, show=True):
    """Plot the sensor positions.

    Parameters
    ----------
    layout : None | Layout
        Layout instance specifying sensor positions.
    %(picks_layout)s
    show_axes : bool
            Show layout axes if True. Defaults to False.
    show : bool
        Show figure if True. Defaults to True.

    Returns
    -------
    fig : instance of Figure
        Figure containing the sensor topography.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(
        figsize=(max(plt.rcParams["figure.figsize"]),) * 2, layout="constrained"
    )
    ax = fig.add_subplot(111)
    ax.set(xticks=[], yticks=[], aspect="equal")
    outlines = dict(border=([0, 1, 1, 0, 0], [0, 0, 1, 1, 0]))
    _draw_outlines(ax, outlines)
    layout = layout.copy().pick(picks)
    for ii, (p, ch_id) in enumerate(zip(layout.pos, layout.names)):
        center_pos = np.array((p[0] + p[2] / 2.0, p[1] + p[3] / 2.0))
        ax.annotate(
            ch_id,
            xy=center_pos,
            horizontalalignment="center",
            verticalalignment="center",
            size="x-small",
        )
        if show_axes:
            x1, x2, y1, y2 = p[0], p[0] + p[2], p[1], p[1] + p[3]
            ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color="k")
    ax.axis("off")
    plt_show(show)
    return fig


def _onselect(
    eclick,
    erelease,
    tfr,
    pos,
    ch_type,
    itmin,
    itmax,
    ifmin,
    ifmax,
    cmap,
    fig,
    layout=None,
):
    """Handle drawing average tfr over channels called from topomap."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    from ..channels.layout import _pair_grad_sensors

    ax = eclick.inaxes
    xmin = min(eclick.xdata, erelease.xdata)
    xmax = max(eclick.xdata, erelease.xdata)
    ymin = min(eclick.ydata, erelease.ydata)
    ymax = max(eclick.ydata, erelease.ydata)
    indices = (
        (pos[:, 0] < xmax)
        & (pos[:, 0] > xmin)
        & (pos[:, 1] < ymax)
        & (pos[:, 1] > ymin)
    )
    colors = ["r" if ii else "k" for ii in indices]
    indices = np.where(indices)[0]
    for collection in ax.collections:
        if isinstance(collection, PathCollection):  # this is our "scatter"
            collection.set_color(colors)
    ax.figure.canvas.draw()
    if len(indices) == 0:
        return
    data = tfr.data
    if ch_type == "mag":
        picks = pick_types(tfr.info, meg=ch_type, ref_meg=False)
        data = np.mean(data[indices, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[picks[x]] for x in indices]
    elif ch_type == "grad":
        grads = _pair_grad_sensors(tfr.info, layout=layout, topomap_coords=False)
        idxs = list()
        for idx in indices:
            idxs.append(grads[idx * 2])
            idxs.append(grads[idx * 2 + 1])  # pair of grads
        data = np.mean(data[idxs, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[x] for x in idxs]
    elif ch_type == "eeg":
        picks = pick_types(tfr.info, meg=False, eeg=True, ref_meg=False)
        data = np.mean(data[indices, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[picks[x]] for x in indices]
    logger.info("Averaging TFR over channels " + str(chs))
    if len(fig) == 0:
        fig.append(figure_nobar())
    if not plt.fignum_exists(fig[0].number):
        fig[0] = figure_nobar()
    ax = fig[0].add_subplot(111)
    itmax = len(tfr.times) - 1 if itmax is None else min(itmax, len(tfr.times) - 1)
    ifmax = len(tfr.freqs) - 1 if ifmax is None else min(ifmax, len(tfr.freqs) - 1)
    if itmin is None:
        itmin = 0
    if ifmin is None:
        ifmin = 0
    extent = (
        tfr.times[itmin] * 1e3,
        tfr.times[itmax] * 1e3,
        tfr.freqs[ifmin],
        tfr.freqs[ifmax],
    )

    title = f"Average over {len(chs)} {ch_type} channels."
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    img = ax.imshow(data, extent=extent, aspect="auto", origin="lower", cmap=cmap)
    if len(fig[0].get_axes()) < 2:
        fig[0].get_axes()[1].cbar = fig[0].colorbar(mappable=img)
    else:
        fig[0].get_axes()[1].cbar.on_mappable_changed(mappable=img)
    fig[0].canvas.draw()
    plt.figure(fig[0].number)
    plt_show(True)


def _prepare_topomap(pos, ax, check_nonzero=True):
    """Prepare the topomap axis and check positions.

    Hides axis frame and check that position information is present.
    """
    _hide_frame(ax)
    if check_nonzero and not pos.any():
        raise RuntimeError(
            "No position information found, cannot compute geometries for topomap."
        )


def _hide_frame(ax):
    """Hide axis frame for topomaps."""
    ax.get_yticks()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)


def _check_extrapolate(extrapolate, ch_type):
    _check_option("extrapolate", extrapolate, ("box", "local", "head", "auto"))
    if extrapolate == "auto":
        extrapolate = "local" if ch_type in _MEG_CH_TYPES_SPLIT else "head"
    return extrapolate


@verbose
def _init_anim(
    ax,
    ax_line,
    ax_cbar,
    params,
    merge_channels,
    sphere,
    ch_type,
    image_interp,
    extrapolate,
    verbose,
):
    """Initialize animated topomap."""
    logger.info("Initializing animation...")
    data = params["data"]
    items = list()
    vmin = params["vmin"] if "vmin" in params else None
    vmax = params["vmax"] if "vmax" in params else None
    if params["butterfly"]:
        all_times = params["all_times"]
        for idx in range(len(data)):
            ax_line.plot(all_times, data[idx], color="k", lw=1)
        vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)
        ax_line.set(
            yticks=np.around(np.linspace(vmin, vmax, 5), -1), xlim=all_times[[0, -1]]
        )
        params["line"] = ax_line.axvline(all_times[0], color="r")
        items.append(params["line"])
    if merge_channels:
        from mne.channels.layout import _merge_ch_data

        data, _ = _merge_ch_data(data, "grad", [])
    norm = True if np.min(data) > 0 else False
    cmap = "Reds" if norm else "RdBu_r"

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)

    outlines = _make_head_outlines(sphere, params["pos"], "head", params["clip_origin"])

    _hide_frame(ax)
    extent, Xi, Yi, interp = _setup_interp(
        pos=params["pos"],
        res=64,
        image_interp=image_interp,
        extrapolate=extrapolate,
        outlines=outlines,
        border=0,
    )

    patch_ = _get_patch(outlines, extrapolate, interp, ax)

    params["Zis"] = list()
    for frame in params["frames"]:
        params["Zis"].append(interp.set_values(data[:, frame])(Xi, Yi))
    Zi = params["Zis"][0]
    zi_min = np.nanmin(params["Zis"])
    zi_max = np.nanmax(params["Zis"])
    cont_lims = np.linspace(zi_min, zi_max, 7, endpoint=False)[1:]
    params.update(
        {
            "vmin": vmin,
            "vmax": vmax,
            "Xi": Xi,
            "Yi": Yi,
            "Zi": Zi,
            "extent": extent,
            "cmap": cmap,
            "cont_lims": cont_lims,
        }
    )
    # plot map and contour
    im = ax.imshow(
        Zi,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        extent=extent,
        interpolation="bilinear",
    )
    ax.autoscale(enable=True, tight=True)
    ax.figure.colorbar(im, cax=ax_cbar)
    cont = ax.contour(Xi, Yi, Zi, levels=cont_lims, colors="k", linewidths=1)

    im.set_clip_path(patch_)
    text = ax.text(0.55, 0.95, "", transform=ax.transAxes, va="center", ha="right")
    params["text"] = text
    items.append(im)
    items.append(text)
    cont_collections = _cont_collections(cont)
    for col in cont_collections:
        col.set_clip_path(patch_)

    outlines_ = _draw_outlines(ax, outlines)

    params.update({"patch": patch_, "outlines": outlines_})
    return tuple(items) + cont_collections


def _animate(frame, ax, ax_line, params):
    """Update animated topomap."""
    if params["pause"]:
        frame = params["frame"]
    time_idx = params["frames"][frame]

    if params["time_unit"] == "ms":
        title = f"{params['times'][frame] * 1e3:6.0f} ms"
    else:
        title = f"{params['times'][frame]:6.3f} s"
    if params["blit"]:
        text = params["text"]
    else:
        ax.cla()  # Clear old contours.
        text = ax.text(0.45, 1.15, "", transform=ax.transAxes)
        for k, (x, y) in params["outlines"].items():
            if "mask" in k:
                continue
            ax.plot(x, y, color="k", linewidth=1, clip_on=False)

    _hide_frame(ax)
    text.set_text(title)

    vmin = params["vmin"]
    vmax = params["vmax"]
    Xi = params["Xi"]
    Yi = params["Yi"]
    Zi = params["Zis"][frame]
    extent = params["extent"]
    cmap = params["cmap"]
    patch = params["patch"]

    im = ax.imshow(
        Zi,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        extent=extent,
        interpolation="bilinear",
    )
    cont_lims = params["cont_lims"]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        cont = ax.contour(Xi, Yi, Zi, levels=cont_lims, colors="k", linewidths=1)

    im.set_clip_path(patch)
    cont_collections = _cont_collections(cont)
    for col in cont_collections:
        col.set_clip_path(patch)

    items = [im, text]
    if params["butterfly"]:
        all_times = params["all_times"]
        line = params["line"]
        line.remove()
        ylim = ax_line.get_ylim()
        params["line"] = ax_line.axvline(all_times[time_idx], color="r")
        ax_line.set_ylim(ylim)
        items.append(params["line"])
    params["frame"] = frame
    return tuple(items) + cont_collections


def _pause_anim(event, params):
    """Pause or continue the animation on mouse click."""
    params["pause"] = not params["pause"]


def _key_press(event, params):
    """Handle key presses for the animation."""
    if event.key == "left":
        params["pause"] = True
        params["frame"] = max(params["frame"] - 1, 0)
    elif event.key == "right":
        params["pause"] = True
        params["frame"] = min(params["frame"] + 1, len(params["frames"]) - 1)


def _topomap_animation(
    evoked,
    ch_type,
    times,
    frame_rate,
    butterfly,
    blit,
    show,
    time_unit,
    sphere,
    image_interp,
    extrapolate,
    *,
    vmin,
    vmax,
    verbose=None,
):
    """Make animation of evoked data as topomap timeseries.

    See mne.evoked.Evoked.animate_topomap.
    """
    from matplotlib import animation
    from matplotlib import pyplot as plt

    if ch_type is None:
        ch_type = _get_plot_ch_type(evoked, ch_type)

    time_unit, _ = _check_time_unit(time_unit, evoked.times)
    if times is None:
        times = np.linspace(evoked.times[0], evoked.times[-1], 10)
    times = np.array(times)

    if times.ndim != 1:
        raise ValueError(f"times must be 1D, got {times.ndim} dimensions")
    if max(times) > evoked.times[-1] or min(times) < evoked.times[0]:
        raise ValueError("All times must be inside the evoked time series.")
    frames = [np.abs(evoked.times - time).argmin() for time in times]

    picks, pos, merge_channels, _, ch_type, sphere, clip_origin = _prepare_topomap_plot(
        evoked, ch_type, sphere=sphere
    )
    data = evoked.data[picks, :]
    data *= _handle_default("scalings")[ch_type]

    norm = np.min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)

    fig = plt.figure(figsize=(6, 5), layout="constrained")
    shape = (8, 12)
    colspan = shape[1] - 1
    rowspan = shape[0] - bool(butterfly)
    ax = plt.subplot2grid(shape, (0, 0), rowspan=rowspan, colspan=colspan)
    if butterfly:
        ax_line = plt.subplot2grid(shape, (rowspan, 0), colspan=colspan)
    else:
        ax_line = None
    if isinstance(frames, Integral):
        frames = np.linspace(0, len(evoked.times) - 1, frames).astype(int)
    ax_cbar = plt.subplot2grid(shape, (0, colspan), rowspan=rowspan)
    ax_cbar.set_title(_handle_default("units")[ch_type], fontsize=10)
    extrapolate = _check_extrapolate(extrapolate, ch_type)

    params = dict(
        data=data,
        pos=pos,
        all_times=evoked.times,
        frame=0,
        frames=frames,
        butterfly=butterfly,
        blit=blit,
        pause=False,
        times=times,
        time_unit=time_unit,
        clip_origin=clip_origin,
        vmin=vmin,
        vmax=vmax,
    )
    init_func = partial(
        _init_anim,
        ax=ax,
        ax_cbar=ax_cbar,
        ax_line=ax_line,
        params=params,
        merge_channels=merge_channels,
        sphere=sphere,
        ch_type=ch_type,
        image_interp=image_interp,
        extrapolate=extrapolate,
        verbose=verbose,
    )
    animate_func = partial(_animate, ax=ax, ax_line=ax_line, params=params)
    pause_func = partial(_pause_anim, params=params)
    fig.canvas.mpl_connect("button_press_event", pause_func)
    key_press_func = partial(_key_press, params=params)
    fig.canvas.mpl_connect("key_press_event", key_press_func)
    if frame_rate is None:
        frame_rate = evoked.info["sfreq"] / 10.0
    interval = 1000 / frame_rate  # interval is in ms
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        init_func=init_func,
        frames=len(frames),
        interval=interval,
        blit=blit,
    )
    fig.mne_animation = anim  # to make sure anim is not garbage collected
    plt_show(show, block=False)
    if "line" in params:
        # Finally remove the vertical line so it does not appear in saved fig.
        params["line"].remove()

    return fig, anim


def _set_contour_locator(vmin, vmax, contours):
    """Set correct contour levels."""
    locator = None
    if isinstance(contours, Integral) and contours > 0:
        from matplotlib import ticker

        # nbins = ticks - 1, since 2 of the ticks are vmin and vmax, the
        # correct number of bins is equal to contours + 1.
        locator = ticker.MaxNLocator(nbins=contours + 1)
        contours = locator.tick_values(vmin, vmax)
    return locator, contours


def _plot_corrmap(
    data,
    subjs,
    indices,
    ch_type,
    ica,
    label,
    *,
    show,
    outlines,
    cmap,
    contours,
    sensors=False,
    template=False,
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    show_names=False,
):
    """Customize ica.plot_components for corrmap."""
    from ..channels.layout import _merge_ch_data

    if not template:
        title = "Detected components"
        if label is not None:
            title += " of type " + label
    else:
        title = "Supplied template"

    picks = list(range(len(data)))

    p = 20
    if len(picks) > p:  # plot components by sets of 20
        n_components = len(picks)
        figs = [
            _plot_corrmap(
                data[k : k + p],
                subjs[k : k + p],
                indices[k : k + p],
                ch_type,
                ica,
                label,
                show=show,
                outlines=outlines,
                cmap=cmap,
                contours=contours,
                sensors=sensors,
                image_interp=image_interp,
                extrapolate=extrapolate,
                border=border,
                show_names=show_names,
            )
            for k in range(0, n_components, p)
        ]
        return figs
    elif np.isscalar(picks):
        picks = [picks]

    (
        data_picks,
        pos,
        merge_channels,
        names,
        _,
        sphere,
        clip_origin,
    ) = _prepare_topomap_plot(ica, ch_type, sphere=sphere)
    names = _prepare_sensor_names(names, show_names)
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

    data = np.atleast_2d(data)
    data = data[:, data_picks]

    # prepare data for iteration
    fig, axes, _, _ = _prepare_trellis(len(picks), ncols=5)
    fig.suptitle(title)

    for ii, data_, ax, subject, idx in zip(picks, data, axes, subjs, indices):
        if template:
            ttl = f"Subj. {subject}, {ica._ica_names[idx]}"
            ax.set_title(ttl, fontsize=12)
        else:
            ax.set_title(f"Subj. {subject}")
        if merge_channels:
            data_, _ = _merge_ch_data(data_, ch_type, [])
        _vlim = _setup_vmin_vmax(data_, None, None)
        plot_topomap(
            data_.flatten(),
            pos,
            vlim=_vlim,
            names=names,
            res=64,
            axes=ax,
            cmap=cmap,
            outlines=outlines,
            contours=contours,
            show=False,
            sensors=sensors,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
        )
        _hide_frame(ax)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _trigradient(x, y, z):
    """Take gradients of z on a mesh."""
    from matplotlib.tri import CubicTriInterpolator, Triangulation

    with warnings.catch_warnings():  # catch matplotlib warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        tri = Triangulation(x, y)
        tci = CubicTriInterpolator(tri, z)
        dx, dy = tci.gradient(tri.x, tri.y)
    return dx, dy


@fill_doc
def plot_arrowmap(
    data,
    info_from,
    info_to=None,
    scale=3e-10,
    vlim=(None, None),
    cnorm=None,
    cmap=None,
    sensors=True,
    res=64,
    axes=None,
    show_names=False,
    mask=None,
    mask_params=None,
    outlines="head",
    contours=6,
    image_interp=_INTERPOLATION_DEFAULT,
    show=True,
    onselect=None,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    sphere=None,
):
    """Plot arrow map.

    Compute arrowmaps, based upon the Hosaka-Cohen transformation
    :footcite:`CohenHosaka1976`, these arrows represents an estimation of the
    current flow underneath the MEG sensors. They are a poor man's MNE.

    Since planar gradiometers takes gradients along latitude and longitude,
    they need to be projected to the flattened manifold span by magnetometer
    or radial gradiometers before taking the gradients in the 2D Cartesian
    coordinate system for visualization on the 2D topoplot. You can use the
    ``info_from`` and ``info_to`` parameters to interpolate from
    gradiometer data to magnetometer data.

    Parameters
    ----------
    data : array, shape (n_channels,)
        The data values to plot.
    info_from : instance of Info
        The measurement info from data to interpolate from.
    info_to : instance of Info | None
        The measurement info to interpolate to. If None, it is assumed
        to be the same as info_from.
    scale : float, default 3e-10
        To scale the arrows.
    %(vlim_plot_topomap)s

        .. versionadded:: 1.2
    %(cnorm)s

        .. versionadded:: 1.2
    %(cmap_topomap_simple)s
    %(sensors_topomap)s
    %(res_topomap)s
    %(axes_plot_topomap)s
    %(show_names_topomap)s
        If ``True``, a list of names must be provided (see ``names`` keyword).
    %(mask_topomap)s
    %(mask_params_topomap)s
    %(outlines_topomap)s
    %(contours_topomap)s
    %(image_interp_topomap)s
    %(show)s
    onselect : callable | None
        Handle for a function that is called when the user selects a set of
        channels by rectangle selection (matplotlib ``RectangleSelector``). If
        None interactive selection is disabled. Defaults to None.
    %(extrapolate_topomap)s

        .. versionadded:: 0.18

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(sphere_topomap_auto)s

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure of the plot.

    Notes
    -----
    .. versionadded:: 0.17

    References
    ----------
    .. footbibliography::
    """
    from matplotlib import pyplot as plt

    from ..forward import _map_meg_or_eeg_channels

    sphere = _check_sphere(sphere, info_from)
    ch_type = _picks_by_type(info_from)

    if len(ch_type) > 1:
        raise ValueError(
            "Multiple channel types are not supported."
            "All channels must either be of type 'grad' "
            "or 'mag'."
        )
    else:
        ch_type = ch_type[0][0]

    if ch_type not in ("mag", "grad"):
        raise ValueError(
            f"Channel type '{ch_type}' not supported. Supported channel "
            "types are 'mag' and 'grad'."
        )

    if info_to is None and ch_type == "mag":
        info_to = info_from
    else:
        ch_type = _picks_by_type(info_to)
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types are not supported.")
        else:
            ch_type = ch_type[0][0]

        if ch_type != "mag":
            raise ValueError(f"only 'mag' channel type is supported. Got {ch_type}")

    if info_to is not info_from:
        info_to = pick_info(info_to, pick_types(info_to, meg=True))
        info_from = pick_info(info_from, pick_types(info_from, meg=True))
        # XXX should probably support the "origin" argument
        mapping = _map_meg_or_eeg_channels(
            info_from, info_to, origin=(0.0, 0.0, 0.04), mode="accurate"
        )
        data = np.dot(mapping, data)

    _, pos, _, _, _, sphere, clip_origin = _prepare_topomap_plot(
        info_to, "mag", sphere=sphere
    )
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)
    if axes is None:
        fig, axes = plt.subplots(layout="constrained")
    else:
        fig = axes.figure
    plot_topomap(
        data,
        pos,
        axes=axes,
        vlim=vlim,
        cmap=cmap,
        cnorm=cnorm,
        sensors=sensors,
        res=res,
        mask=mask,
        mask_params=mask_params,
        outlines=outlines,
        contours=contours,
        image_interp=image_interp,
        show=False,
        onselect=onselect,
        extrapolate=extrapolate,
        sphere=sphere,
        ch_type=ch_type,
    )
    x, y = tuple(pos.T)
    dx, dy = _trigradient(x, y, data)
    dxx = dy.data
    dyy = -dx.data
    axes.quiver(x, y, dxx, dyy, scale=scale, color="k", lw=1)
    plt_show(show)

    return fig


@fill_doc
def plot_bridged_electrodes(
    info, bridged_idx, ed_matrix, title=None, topomap_args=None
):
    """Topoplot electrode distance matrix with bridged electrodes connected.

    Parameters
    ----------
    %(info_not_none)s
    bridged_idx : list of tuple
        The indices of channels marked as bridged with each bridged
        pair stored as a tuple.
        Can be generated via
        :func:`mne.preprocessing.compute_bridged_electrodes`.
    ed_matrix : array of float, shape (n_channels, n_channels)
        The electrical distance matrix for each pair of EEG electrodes.
        Can be generated via
        :func:`mne.preprocessing.compute_bridged_electrodes`.
    title : str
        A title to add to the plot.
    topomap_args : dict | None
        Arguments to pass to :func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The topoplot figure handle.

    See Also
    --------
    mne.preprocessing.compute_bridged_electrodes
    """
    import matplotlib.pyplot as plt

    from ..channels.layout import _find_topomap_coords

    if topomap_args is None:
        topomap_args = dict()
    else:
        topomap_args = topomap_args.copy()  # don't change original
    picks = pick_types(info, eeg=True)
    topomap_args.setdefault("image_interp", "nearest")
    topomap_args.setdefault("cmap", "summer_r")
    topomap_args.setdefault("names", pick_info(info, picks).ch_names)
    topomap_args.setdefault("contours", False)
    sphere = topomap_args.get("sphere", _check_sphere(None))
    if "axes" not in topomap_args:
        fig, ax = plt.subplots(layout="constrained")
        topomap_args["axes"] = ax
    else:
        fig = None
    # handle colorbar here instead of in plot_topomap
    colorbar = topomap_args.pop("colorbar", True)
    if ed_matrix.shape[1:] != (picks.size, picks.size):
        raise RuntimeError(
            f"Expected {(ed_matrix.shape[0], picks.size, picks.size)} "
            f"shaped `ed_matrix`, got {ed_matrix.shape}"
        )
    # fill in lower triangular
    ed_matrix = ed_matrix.copy()
    tril_idx = np.tril_indices(picks.size)
    for epo_idx in range(ed_matrix.shape[0]):
        ed_matrix[epo_idx][tril_idx] = ed_matrix[epo_idx].T[tril_idx]
    elec_dists = np.median(np.nanmin(ed_matrix, axis=1), axis=0)

    im, cn = plot_topomap(elec_dists, pick_info(info, picks), **topomap_args)
    fig = im.figure if fig is None else fig
    # add bridged connections
    for idx0, idx1 in bridged_idx:
        pos = _find_topomap_coords(info, [idx0, idx1], sphere=sphere)
        im.axes.plot([pos[0, 0], pos[1, 0]], [pos[0, 1], pos[1, 1]], color="r")
    if title is not None:
        im.axes.set_title(title)
    if colorbar:
        cax = fig.colorbar(im, shrink=0.6)
        cax.set_label(r"Electrical Distance ($\mu$$V^2$)")
    return fig


def plot_ch_adjacency(info, adjacency, ch_names, kind="2d", edit=False):
    """Plot channel adjacency.

    Parameters
    ----------
    info : instance of Info
        Info object with channel locations.
    adjacency : array
        Array of channels x channels shape. Defines which channels are adjacent
        to each other. Note that if you edit adjacencies
        (via ``edit=True``), this array will be modified in place.
    ch_names : list of str
        Names of successive channels in the ``adjacency`` matrix.
    kind : str
        How to plot the adjacency. Can be either ``'3d'`` or ``'2d'``.
    edit : bool
        Whether to allow interactive editing of the adjacency matrix via
        clicking respective channel pairs. Once clicked, the channel is
        "activated" and turns green. Clicking on another channel adds or
        removes adjacency relation between the activated and newly clicked
        channel (depending on whether the channels are already adjacent or
        not); the newly clicked channel now becomes activated. Clicking on
        an activated channel deactivates it. Editing is currently only
        supported for ``kind='2d'``.

    Returns
    -------
    fig : Figure
        The :class:`~matplotlib.figure.Figure` instance where the channel
        adjacency is plotted.

    See Also
    --------
    mne.channels.get_builtin_ch_adjacencies
    mne.channels.read_ch_adjacency
    mne.channels.find_ch_adjacency

    Notes
    -----
    .. versionadded:: 1.1
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _validate_type(info, Info, "info")
    _validate_type(adjacency, (np.ndarray, csr_array), "adjacency")
    has_sparse = isinstance(adjacency, csr_array)

    if edit and kind == "3d":
        raise ValueError("Editing a 3d adjacency plot is not supported.")

    # select relevant channels
    sel = pick_channels(info.ch_names, ch_names, ordered=True)
    info = pick_info(info, sel)

    # make sure adjacency is correct size wrt to inst:
    n_channels = len(info.ch_names)
    if adjacency.shape[0] != n_channels:
        raise ValueError(
            "``adjacency`` must have the same number of rows "
            "as the number of channels in ``info``. Found "
            f"{adjacency.shape[0]} channels for ``adjacency`` and"
            f" {n_channels} for ``inst``."
        )

    if kind == "3d":
        with plt.rc_context({"toolbar": "None"}):
            fig = plot_sensors(info, kind=kind, show=False)
        _set_3d_axes_equal(fig.axes[0])
    elif kind == "2d":
        with plt.rc_context({"toolbar": "None"}):
            fig = plot_sensors(info, kind="topomap", show=False)
        fig.axes[0].axis("equal")

    path_collection = fig.axes[0].findobj(mpl.collections.PathCollection)
    path_collection[0].set_linewidths(0.0)

    if kind == "2d":
        path_collection[0].set_alpha(0.7)
        pos = path_collection[0].get_offsets()

        # make sure nodes are on top
        path_collection[0].set_zorder(10)

        # scale node size with number of connections
        n_connections = [np.sum(adjacency[[i]]) - 1 for i in range(adjacency.shape[0])]
        node_size = [max(x, 3) ** 2.5 for x in n_connections]
        path_collection[0].set_sizes(node_size)
    else:
        # plotting channel positions via mne.viz.plot_sensors(info) and using
        # the coordinates from info['chs'][ch_idx]['loc][:3] gives different
        # positions. Also .get_offsets gives 2d projections even for 3d points
        # so we use the private _offsets3d property...
        pos = path_collection[0]._offsets3d
        pos = np.stack([pos[0].data, pos[1].data, pos[2]], axis=1)

    ax = fig.axes[0]
    lines = dict()
    n_channels = adjacency.shape[0]
    for ch_idx in range(n_channels):
        # make sure we don't repeat channels
        row = adjacency[[ch_idx], ch_idx + 1 :]
        if has_sparse:
            ch_neighbours = row.nonzero()[1]
        else:
            ch_neighbours = np.where(row)[0]

        if len(ch_neighbours) == 0:
            continue

        ch_neighbours += ch_idx + 1

        for ngb_idx in ch_neighbours:
            this_pos = pos[[ch_idx, ngb_idx], :]
            ch_pair = tuple([ch_idx, ngb_idx])
            lines[ch_pair] = ax.plot(*this_pos.T, color=(0.55, 0.55, 0.55), lw=0.75)[0]

    if edit:
        # allow interactivity in 2d plots
        highlighted = dict()
        this_onpick = partial(
            _onpick_ch_adjacency,
            axes=ax,
            positions=pos,
            highlighted=highlighted,
            line_dict=lines,
            adjacency=adjacency,
            node_size=node_size,
            path_collection=path_collection,
        )
        fig.canvas.mpl_connect("pick_event", this_onpick)

    return fig


def _onpick_ch_adjacency(
    event,
    axes=None,
    positions=None,
    highlighted=None,
    line_dict=None,
    adjacency=None,
    node_size=None,
    path_collection=None,
):
    """Handle interactivity in plot_ch_adjacency."""
    node_ind = event.ind[0]

    if node_ind in highlighted:
        # de-select node, change its color back to normal
        highlighted[node_ind].remove()
        del highlighted[node_ind]
        axes.figure.canvas.draw()
    else:
        # new node selected
        if len(highlighted) == 0:
            # no highlighted nodes yet
            size = max(node_size[node_ind] * 2, 100)
            # add current node
            dots = axes.scatter(
                *positions[node_ind, :].T, color="tab:green", s=size, zorder=15
            )
            highlighted[node_ind] = dots
            axes.figure.canvas.draw()  # make sure it renders
        else:
            # one previously highlighted - add or remove line
            key = list(highlighted.keys())[0]
            both_nodes = [key, node_ind]
            both_nodes.sort()
            both_nodes = tuple(both_nodes)

            if both_nodes in line_dict.keys():
                # remove line
                n_conn_change = -1
                line_dict[both_nodes].remove()
                # remove line_dict entry
                del line_dict[both_nodes]

                # clear adjacency matrix entry
                _set_adjacency(adjacency, both_nodes, False)
            else:
                # add line
                n_conn_change = +1
                selected_pos = positions[both_nodes, :]
                line = axes.plot(*selected_pos.T, color="tab:green")[0]
                # add line to line_dict
                line_dict[both_nodes] = line

                # modify adjacency matrix
                _set_adjacency(adjacency, both_nodes, True)

            # de-highlight previous
            highlighted[key].remove()
            del highlighted[key]

            # update node sizes
            n_connections = [
                np.sum(adjacency[[idx]]) - 1 + n_conn_change for idx in both_nodes
            ]
            for idx, n_conn in zip(both_nodes, n_connections):
                node_size[idx] = max(n_conn, 3) ** 2.5
            path_collection[0].set_sizes(node_size)

            # highlight new node
            size = max(node_size[node_ind] * 2, 100)
            dots = axes.scatter(
                *positions[node_ind, :].T, color="tab:green", s=size, zorder=15
            )
            highlighted[node_ind] = dots
            axes.figure.canvas.draw()


def _set_adjacency(adjacency, both_nodes, value):
    """Set adjacency for given node pair, caching errors for sparse arrays."""
    import warnings

    with warnings.catch_warnings(record=True):
        adjacency[both_nodes, both_nodes[::-1]] = value


@fill_doc
def plot_regression_weights(
    model,
    *,
    ch_type=None,
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
    cmap=None,
    vlim=(None, None),
    cnorm=None,
    axes=None,
    colorbar=True,
    cbar_fmt="%1.1e",
    title=None,
    show=True,
):
    """Plot the regression weights of a fitted EOGRegression model.

    Parameters
    ----------
    model : EOGRegression
        The fitted EOGRegression model whose weights will be plotted.
    %(ch_type_topomap)s
    %(sensors_topomap)s
    %(show_names_topomap)s
    %(mask_topomap)s
    %(mask_params_topomap)s
    %(contours_topomap)s
    %(outlines_topomap)s
    %(sphere_topomap_auto)s
    %(image_interp_topomap)s
    %(extrapolate_topomap)s

        .. versionchanged:: 0.21

           - The default was changed to ``'local'`` for MEG sensors.
           - ``'local'`` was changed to use a convex hull mask
           - ``'head'`` was changed to extrapolate out to the clipping circle.
    %(border_topomap)s

        .. versionadded:: 0.20
    %(res_topomap)s
    %(size_topomap)s
    %(cmap_topomap)s
    %(vlim_plot_topomap)s
    %(cnorm)s
    %(axes_evoked_plot_topomap)s
    %(colorbar_topomap)s
    %(cbar_fmt_topomap)s
    %(title_none)s
    %(show)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure with a topomap subplot for each channel type.

    Notes
    -----
    .. versionadded:: 1.2
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from ..channels.layout import _merge_ch_data

    sphere = _check_sphere(sphere)
    if ch_type is None:
        ch_types = model.info_.get_channel_types(unique=True, only_data_chs=True)
    else:
        ch_types = [ch_type]
    del ch_type

    nrows = model.coef_.shape[1]
    ncols = len(ch_types)

    axes_was_none = axes is None
    if axes_was_none:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=(ncols * 2, nrows * 1.5 + 1),
            layout="constrained",
        )
        axes = axes.T.ravel()
    else:
        if isinstance(axes, matplotlib.axes.Axes):
            axes = [axes]
        fig = axes[0].get_figure()
    if len(axes) != nrows * ncols:
        raise ValueError(
            f"axes must be a list of {nrows * ncols} axes, got "
            f"length {len(axes)} ({axes})."
        )
    axes = iter(axes)

    data_picks = _picks_to_idx(model.info_, model.picks, exclude=model.exclude)
    data_info = pick_info(model.info_, data_picks)
    artifact_ch_names = [
        model.info_["chs"][idx]["ch_name"]
        for idx in _picks_to_idx(model.info_, model.picks_artifact)
    ]

    for ch_type in ch_types:
        (
            data_picks,
            pos,
            merge_channels,
            names,
            ch_type,
            sphere,
            clip_origin,
        ) = _prepare_topomap_plot(data_info, ch_type=ch_type, sphere=sphere)
        outlines = _make_head_outlines(
            sphere, pos, outlines=outlines, clip_origin=clip_origin
        )
        coef = model.coef_[data_picks]
        for data, ch_name in zip(coef.T, artifact_ch_names):
            if merge_channels:
                data, names = _merge_ch_data(data, ch_type, names)
            ax = next(axes)
            names = _prepare_sensor_names(data_info.ch_names, show_names)

            _plot_topomap_multi_cbar(
                data,
                pos,
                ax,
                title=f"{ch_type}/{ch_name}",
                vlim=vlim,
                cmap=cmap,
                outlines=outlines,
                colorbar=colorbar,
                unit="",
                cbar_fmt=cbar_fmt,
                sphere=sphere,
                ch_type=ch_type,
                sensors=sensors,
                names=names,
                mask=mask,
                mask_params=mask_params,
                contours=contours,
                image_interp=image_interp,
                extrapolate=extrapolate,
                border=border,
                res=res,
                size=size,
                cnorm=cnorm,
            )
    if axes_was_none:
        fig.suptitle(title)
    plt_show(show)
    return fig
