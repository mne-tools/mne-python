"""Functions to make 3D plots with M/EEG data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations  # only needed for Python ≤ 3.9

import os
import os.path as op
import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .._fiff.constants import FIFF
from .._fiff.meas_info import Info, create_info, read_fiducials
from .._fiff.pick import (
    _FNIRS_CH_TYPES_SPLIT,
    _MEG_CH_TYPES_SPLIT,
    channel_type,
    pick_info,
    pick_types,
)
from .._fiff.tag import _loc_to_coil_trans
from .._freesurfer import (
    _check_mri,
    _get_head_surface,
    _get_skull_surface,
    _read_mri_info,
    read_freesurfer_lut,
)
from ..defaults import DEFAULTS
from ..fixes import _crop_colorbar, _get_img_fdata
from ..surface import (
    _CheckInside,
    _DistanceQuery,
    _project_onto_surface,
    _read_mri_surface,
    _reorder_ccw,
    get_meg_helmet_surf,
)
from ..transforms import (
    Transform,
    _angle_between_quats,
    _angle_dist_between_rigid,
    _ensure_trans,
    _find_trans,
    _frame_to_str,
    _get_trans,
    _get_transforms_to_coord_frame,
    _print_coord_trans,
    apply_trans,
    combine_transforms,
    read_ras_mni_t,
    rot_to_quat,
    rotation,
    transform_surface_to,
)
from ..utils import (
    _check_option,
    _check_subject,
    _ensure_int,
    _import_nibabel,
    _pl,
    _to_rgb,
    _validate_type,
    check_version,
    fill_doc,
    get_config,
    get_subjects_dir,
    logger,
    verbose,
    warn,
)
from ._dipole import _check_concat_dipoles, _plot_dipole_3d, _plot_dipole_mri_outlines
from .evoked_field import EvokedField
from .utils import (
    _check_time_unit,
    _get_cmap,
    _get_color_list,
    figure_nobar,
    plt_show,
)

verbose_dec = verbose
FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_RPA)


# XXX: to unify with digitization
def _fiducial_coords(points, coord_frame=None):
    """Generate 3x3 array of fiducial coordinates."""
    points = points or []  # None -> list
    if coord_frame is not None:
        points = [p for p in points if p["coord_frame"] == coord_frame]
    points_ = {p["ident"]: p for p in points if p["kind"] == FIFF.FIFFV_POINT_CARDINAL}
    if points_:
        return np.array([points_[i]["r"] for i in FIDUCIAL_ORDER])
    else:
        # XXX eventually this should probably live in montage.py
        if coord_frame is None or coord_frame == FIFF.FIFFV_COORD_HEAD:
            # Try converting CTF HPI coils to fiducials
            out = np.empty((3, 3))
            out.fill(np.nan)
            for p in points:
                if p["kind"] == FIFF.FIFFV_POINT_HPI:
                    if np.isclose(p["r"][1:], 0, atol=1e-6).all():
                        out[0 if p["r"][0] < 0 else 2] = p["r"]
                    elif np.isclose(p["r"][::2], 0, atol=1e-6).all():
                        out[1] = p["r"]
            if np.isfinite(out).all():
                return out
        return np.array([])


@fill_doc
def plot_head_positions(
    pos,
    mode="traces",
    cmap="viridis",
    direction="z",
    *,
    show=True,
    destination=None,
    info=None,
    color="k",
    axes=None,
    totals=False,
):
    """Plot head positions.

    Parameters
    ----------
    pos : ndarray, shape (n_pos, 10) | list of ndarray
        The head position data. Can also be a list to treat as a
        concatenation of runs.
    mode : str
        Can be 'traces' (default) to show position and quaternion traces,
        or 'field' to show the position as a vector field over time.
    cmap : colormap
        Colormap to use for the trace plot, default is "viridis".
    direction : str
        Can be any combination of "x", "y", or "z" (default: "z") to show
        directional axes in "field" mode.
    show : bool
        Show figure if True. Defaults to True.
    destination : path-like | array-like, shape (3,) | instance of Transform | None
        The destination location for the head. See
        :func:`mne.preprocessing.maxwell_filter` for details.

        .. versionadded:: 0.16
    %(info)s If provided, will be used to show the destination position when
        ``destination is None``, and for showing the MEG sensors.

        .. versionadded:: 0.16
    color : color object
        The color to use for lines in ``mode == 'traces'`` and quiver
        arrows in ``mode == 'field'``.

        .. versionadded:: 0.16
    axes : array-like, shape (3, 2) or (4, 2)
        The matplotlib axes to use.

        .. versionadded:: 0.16
        .. versionchanged:: 1.8
           Added support for making use of this argument when ``mode="field"``.
    totals : bool
        If True and in traces mode, show the total distance and angle in a fourth row.

        .. versionadded:: 1.9

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    from ..chpi import head_pos_to_trans_rot_t
    from ..preprocessing.maxwell import _check_destination

    _check_option("mode", mode, ["traces", "field"])
    _validate_type(totals, bool, "totals")
    dest_info = dict(dev_head_t=None) if info is None else info
    destination = _check_destination(destination, dest_info, head_frame=True)
    if destination is not None:
        destination = _ensure_trans(destination, "head", "meg")  # probably inv
        destination = destination["trans"]

    if not isinstance(pos, list | tuple):
        pos = [pos]
    pos = list(pos)  # make our own mutable copy
    for ii, p in enumerate(pos):
        _validate_type(p, np.ndarray, f"pos[{ii}]")
        p = np.array(p, float)
        if p.ndim != 2 or p.shape[1] != 10:
            raise ValueError(
                "pos (or each entry in pos if a list) must be "
                f"dimension (N, 10), got {p.shape}"
            )
        if ii > 0:  # concatenation
            p[:, 0] += pos[ii - 1][-1, 0] - p[0, 0]
        pos[ii] = p
    borders = np.cumsum([len(pp) for pp in pos])
    pos = np.concatenate(pos, axis=0)
    trans, rot, t = head_pos_to_trans_rot_t(pos)  # also ensures pos is okay
    # trans, rot, and t are for dev_head_t, but what we really want
    # is head_dev_t (i.e., where the head origin is in device coords)
    use_trans = (
        np.einsum("ijk,ik->ij", rot[:, :3, :3].transpose([0, 2, 1]), -trans) * 1000
    )
    use_rot = rot.transpose([0, 2, 1])
    use_quats = -pos[:, 1:4]  # inverse (like doing rot.T)
    surf = rrs = lims = None
    if info is not None:
        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())
        if len(meg_picks) > 0:
            rrs = 1000 * np.array(
                [info["chs"][pick]["loc"][:3] for pick in meg_picks], float
            )
            if mode == "traces":
                lims = np.array((rrs.min(0), rrs.max(0))).T
            else:  # mode == 'field'
                surf = get_meg_helmet_surf(info)
                transform_surface_to(surf, "meg", info["dev_head_t"], copy=False)
                surf["rr"] *= 1000.0
    helmet_color = DEFAULTS["coreg"]["helmet_color"]
    if mode == "traces":
        want_shape = (3 + int(totals), 2)
        if axes is None:
            _, axes = plt.subplots(*want_shape, sharex=True, layout="constrained")
        else:
            axes = np.array(axes)
        _check_option("axes.shape", axes.shape, (want_shape,))
        fig = axes[0, 0].figure
        labels = [["x (mm)", "y (mm)", "z (mm)"], ["$q_1$", "$q_2$", "$q_3$"]]
        if totals:
            labels[0].append("dist (mm)")
            labels[1].append("angle (°)")
        for ii, (quat, coord) in enumerate(zip(use_quats.T, use_trans.T)):
            axes[ii, 0].plot(t, coord, color, lw=1.0, zorder=3)
            axes[ii, 0].set(ylabel=labels[0][ii], xlim=t[[0, -1]])
            axes[ii, 1].plot(t, quat, color, lw=1.0, zorder=3)
            axes[ii, 1].set(ylabel=labels[1][ii], xlim=t[[0, -1]])
            for b in borders[:-1]:
                for jj in range(2):
                    axes[ii, jj].axvline(t[b], color="r")
        if totals:
            vals = [
                np.linalg.norm(use_trans, axis=-1),
                np.rad2deg(_angle_between_quats(use_quats)),
            ]
            ii = -1
            for ci, val in enumerate(vals):
                axes[ii, ci].plot(t, val, color, lw=1.0, zorder=3)
                axes[ii, ci].set(ylabel=labels[ci][ii], xlim=t[[0, -1]])
        titles = ["Position", "Rotation"]
        for ci, title in enumerate(titles):
            axes[0, ci].set(title=title)
            axes[-1, ci].set(xlabel="Time (s)")
        if rrs is not None:
            pos_bads = np.any(
                [
                    (use_trans[:, ii] <= lims[ii, 0])
                    | (use_trans[:, ii] >= lims[ii, 1])
                    for ii in range(3)
                ],
                axis=0,
            )
            for ii in range(3):
                oidx = list(range(ii)) + list(range(ii + 1, 3))
                # knowing it will generally be spherical, we can approximate
                # how far away we are along the axis line by taking the
                # point to the left and right with the smallest distance
                dists = cdist(rrs[:, oidx], use_trans[:, oidx])
                left = rrs[:, [ii]] < use_trans[:, ii]
                left_dists_all = dists.copy()
                left_dists_all[~left] = np.inf
                # Don't show negative Z direction
                if ii != 2 and np.isfinite(left_dists_all).any():
                    idx = np.argmin(left_dists_all, axis=0)
                    left_dists = rrs[idx, ii]
                    bads = (
                        ~np.isfinite(left_dists_all[idx, np.arange(len(idx))])
                        | pos_bads
                    )
                    left_dists[bads] = np.nan
                    axes[ii, 0].plot(
                        t, left_dists, color=helmet_color, ls="-", lw=0.5, zorder=2
                    )
                else:
                    axes[ii, 0].axhline(
                        lims[ii][0], color=helmet_color, ls="-", lw=0.5, zorder=2
                    )
                right_dists_all = dists
                right_dists_all[left] = np.inf
                if np.isfinite(right_dists_all).any():
                    idx = np.argmin(right_dists_all, axis=0)
                    right_dists = rrs[idx, ii]
                    bads = (
                        ~np.isfinite(right_dists_all[idx, np.arange(len(idx))])
                        | pos_bads
                    )
                    right_dists[bads] = np.nan
                    axes[ii, 0].plot(
                        t, right_dists, color=helmet_color, ls="-", lw=0.5, zorder=2
                    )
                else:
                    axes[ii, 0].axhline(
                        lims[ii][1], color=helmet_color, ls="-", lw=0.5, zorder=2
                    )

        for ii in range(3):
            axes[ii, 1].set(ylim=[-1, 1])

        if destination is not None:
            vals = np.array(
                [1000 * destination[:3, 3], rot_to_quat(destination[:3, :3])]
            ).T.ravel()
            for ax, val in zip(axes[:3].ravel(), vals):
                ax.axhline(val, color="r", ls=":", zorder=2, lw=1.0)
            if totals:
                dest_ang, dest_dist = _angle_dist_between_rigid(
                    destination,
                    angle_units="deg",
                    distance_units="mm",
                )
                axes[-1, 0].axhline(dest_dist, color="r", ls=":", zorder=2, lw=1.0)
                axes[-1, 1].axhline(dest_ang, color="r", ls=":", zorder=2, lw=1.0)

    else:  # mode == 'field':
        from matplotlib.colors import Normalize
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, analysis:ignore
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        _validate_type(axes, (Axes3D, None), "ax", extra="when mode='field'")
        if axes is None:
            _, ax = plt.subplots(
                1, subplot_kw=dict(projection="3d"), layout="constrained"
            )
        else:
            ax = axes
        fig = ax.get_figure()
        del axes

        # First plot the trajectory as a colormap:
        # http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        pts = use_trans[:, np.newaxis]
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = Normalize(t[0], t[-2])
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        # now plot the head directions as a quiver
        dir_idx = dict(x=0, y=1, z=2)
        kwargs = dict(pivot="tail")
        for d, length in zip(direction, [5.0, 2.5, 1.0]):
            use_dir = use_rot[:, :, dir_idx[d]]
            # draws stems, then heads
            array = np.concatenate((t, np.repeat(t, 2)))
            ax.quiver(
                use_trans[:, 0],
                use_trans[:, 1],
                use_trans[:, 2],
                use_dir[:, 0],
                use_dir[:, 1],
                use_dir[:, 2],
                norm=norm,
                cmap=cmap,
                array=array,
                length=length,
                **kwargs,
            )
            if destination is not None:
                ax.quiver(
                    destination[0, 3],
                    destination[1, 3],
                    destination[2, 3],
                    destination[dir_idx[d], 0],
                    destination[dir_idx[d], 1],
                    destination[dir_idx[d], 2],
                    color=color,
                    length=length,
                    **kwargs,
                )
        mins = use_trans.min(0)
        maxs = use_trans.max(0)
        if surf is not None:
            ax.plot_trisurf(
                *surf["rr"].T,
                triangles=surf["tris"],
                color=helmet_color,
                alpha=0.1,
                shade=False,
            )
            ax.scatter(*rrs.T, s=1, color=helmet_color)
            mins = np.minimum(mins, rrs.min(0))
            maxs = np.maximum(maxs, rrs.max(0))
        scale = (maxs - mins).max() / 2.0
        xlim, ylim, zlim = (maxs + mins)[:, np.newaxis] / 2.0 + [-scale, scale]
        ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=xlim, ylim=ylim, zlim=zlim)
        _set_aspect_equal(ax)
        ax.view_init(30, 45)
    plt_show(show)
    return fig


def _set_aspect_equal(ax):
    # XXX recent MPL throws an error for 3D axis aspect setting, not much
    # we can do about it at this point
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass


@verbose
def plot_evoked_field(
    evoked,
    surf_maps,
    time=None,
    time_label="t = %0.0f ms",
    n_jobs=None,
    fig=None,
    vmax=None,
    n_contours=21,
    *,
    show_density=True,
    alpha=None,
    interpolation="nearest",
    interaction="terrain",
    time_viewer="auto",
    verbose=None,
):
    """Plot MEG/EEG fields on head surface and helmet in 3D.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked object.
    surf_maps : list
        The surface mapping information obtained with make_field_map.
    time : float | None
        The time point at which the field map shall be displayed. If None,
        the average peak latency (across sensor types) is used.
    time_label : str | None
        How to print info about the time instant visualized.
    %(n_jobs)s
    fig : Figure3D | mne.viz.Brain | None
        If None (default), a new figure will be created, otherwise it will
        plot into the given figure.

        .. versionadded:: 0.20
        .. versionadded:: 1.4
            ``fig`` can also be a ``Brain`` figure.
    vmax : float | dict | None
        Maximum intensity. Can be a dictionary with two entries ``"eeg"`` and ``"meg"``
        to specify separate values for EEG and MEG fields respectively. Can be
        ``None`` to use the maximum value of the data.

        .. versionadded:: 0.21
        .. versionadded:: 1.4
            ``vmax`` can be a dictionary to specify separate values for EEG and
            MEG fields.
    n_contours : int
        The number of contours.

        .. versionadded:: 0.21
    show_density : bool
        Whether to draw the field density as an overlay on top of the helmet/head
        surface. Defaults to ``True``.

        .. versionadded:: 1.6
    alpha : float | dict | None
        Opacity of the meshes (between 0 and 1). Can be a dictionary with two
        entries ``"eeg"`` and ``"meg"`` to specify separate values for EEG and
        MEG fields respectively. Can be ``None`` to use 1.0 when a single field
        map is shown, or ``dict(eeg=1.0, meg=0.5)`` when both field maps are shown.

        .. versionadded:: 1.4
    %(interpolation_brain_time)s

        .. versionadded:: 1.6
    %(interaction_scene)s
        Defaults to ``'terrain'``.

        .. versionadded:: 1.1
    time_viewer : bool | str
        Display time viewer GUI. Can also be ``"auto"``, which will mean
        ``True`` if there is more than one time point and ``False`` otherwise.

        .. versionadded:: 1.6
    %(verbose)s

    Returns
    -------
    fig : Figure3D | mne.viz.EvokedField
        Without the time viewer active, the figure is returned. With the time
        viewer active, an object is returned that can be used to control
        different aspects of the figure.
    """
    ef = EvokedField(
        evoked,
        surf_maps,
        time=time,
        time_label=time_label,
        n_jobs=n_jobs,
        fig=fig,
        vmax=vmax,
        n_contours=n_contours,
        alpha=alpha,
        show_density=show_density,
        interpolation=interpolation,
        interaction=interaction,
        time_viewer=time_viewer,
        verbose=verbose,
    )
    if ef.time_viewer:
        return ef
    else:
        return ef._renderer.scene()


@verbose
def plot_alignment(
    info=None,
    trans=None,
    subject=None,
    subjects_dir=None,
    surfaces="auto",
    coord_frame="auto",
    meg=None,
    eeg="original",
    fwd=None,
    dig=False,
    ecog=True,
    src=None,
    mri_fiducials=False,
    bem=None,
    seeg=True,
    fnirs=True,
    show_axes=False,
    dbs=True,
    fig=None,
    interaction="terrain",
    sensor_colors=None,
    *,
    sensor_scales=None,
    verbose=None,
):
    """Plot head, sensor, and source space alignment in 3D.

    Parameters
    ----------
    %(info)s If None (default), no sensor information will be shown.
    %(trans)s "auto" will load trans from the FreeSurfer directory
        specified by ``subject`` and ``subjects_dir`` parameters.

        .. versionchanged:: 0.19
            Support for 'fsaverage' argument.
    %(subject)s Can be omitted if ``src`` is provided.
    %(subjects_dir)s
    surfaces : str | list | dict
        Surfaces to plot. Supported values:

        * scalp: one of 'head', 'outer_skin' (alias for 'head'),
          'head-dense', or 'seghead' (alias for 'head-dense')
        * skull: 'outer_skull', 'inner_skull', 'brain' (alias for
          'inner_skull')
        * brain: one of 'pial', 'white', 'inflated', or 'brain'
          (alias for 'pial').

        Can be dict to specify alpha values for each surface. Use None
        to specify default value. Specified values must be between 0 and 1.
        for example::

            surfaces=dict(brain=0.4, outer_skull=0.6, head=None)

        Defaults to 'auto', which will look for a head surface and plot
        it if found.

        .. note:: For single layer BEMs it is recommended to use ``'brain'``.
    coord_frame : 'auto' | 'head' | 'meg' | 'mri'
        The coordinate frame to use. If ``'auto'`` (default), chooses ``'mri'``
        if ``trans`` was passed, and ``'head'`` otherwise.

        .. versionchanged:: 1.0
           Defaults to ``'auto'``.
    %(meg)s
    %(eeg)s
    %(fwd)s
    dig : bool | 'fiducials'
        If True, plot the digitization points; 'fiducials' to plot fiducial
        points only.
    %(ecog)s
    src : instance of SourceSpaces | None
        If not None, also plot the source space points.
    mri_fiducials : bool | str | path-like
        Plot MRI fiducials (default False). If ``True``, look for a file with
        the canonical name (``bem/{subject}-fiducials.fif``). If ``str``,
        it can be ``'estimated'`` to use :func:`mne.coreg.get_mni_fiducials`,
        otherwise it should provide the full path to the fiducials file.

        .. versionadded:: 0.22
           Support for ``'estimated'``.
    bem : list of dict | instance of ConductorModel | None
        Can be either the BEM surfaces (list of dict), a BEM solution or a
        sphere model. If None, we first try loading
        ``'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'``, and then look
        for ``'$SUBJECT*$SOURCE.fif'`` in the same directory. For
        ``'outer_skin'``, the subjects bem and bem/flash folders are searched.
        Defaults to None.
    %(seeg)s
    %(fnirs)s
        .. versionadded:: 0.20
    show_axes : bool
        If True (default False), coordinate frame axis indicators will be
        shown:

        * head in pink.
        * MRI in gray (if ``trans is not None``).
        * MEG in blue (if MEG sensors are present).

        .. versionadded:: 0.16
    %(dbs)s
    fig : Figure3D | None
        PyVista scene in which to plot the alignment.
        If ``None``, creates a new 600x600 pixel figure with black background.

        .. versionadded:: 0.16
    %(interaction_scene)s

        .. versionadded:: 0.16
        .. versionchanged:: 1.0
           Defaults to ``'terrain'``.
    %(sensor_colors)s

        .. versionchanged:: 1.6
            Support for passing a ``dict`` was added.
    %(sensor_scales)s

        .. versionadded:: 1.9
    %(verbose)s

    Returns
    -------
    fig : instance of Figure3D
        The figure.

    See Also
    --------
    mne.viz.plot_bem

    Notes
    -----
    This function serves the purpose of checking the validity of the many
    different steps of source reconstruction:

    - Transform matrix (keywords ``trans``, ``meg`` and ``mri_fiducials``),
    - BEM surfaces (keywords ``bem`` and ``surfaces``),
    - sphere conductor model (keywords ``bem`` and ``surfaces``) and
    - source space (keywords ``surfaces`` and ``src``).

    .. versionadded:: 0.15
    """
    # Update the backend
    from ..bem import ConductorModel, _bem_find_surface, _ensure_bem_surfaces
    from ..source_space._source_space import _ensure_src
    from .backends.renderer import _get_renderer

    meg, eeg, fnirs, warn_meg, sensor_alpha = _handle_sensor_types(meg, eeg, fnirs)
    _check_option("interaction", interaction, ["trackball", "terrain"])

    info = create_info(1, 1000.0, "misc") if info is None else info
    _validate_type(info, "info")

    # Handle surfaces:
    if surfaces == "auto" and trans is None:
        surfaces = list()  # if no `trans` can't plot mri surfaces
    if isinstance(surfaces, str):
        surfaces = [surfaces]
    if isinstance(surfaces, dict):
        user_alpha = surfaces.copy()
        for key, val in user_alpha.items():
            _validate_type(key, "str", f"surfaces key {repr(key)}")
            _validate_type(val, (None, "numeric"), f"surfaces[{repr(key)}]")
            if val is not None:
                user_alpha[key] = float(val)
                if not 0 <= user_alpha[key] <= 1:
                    raise ValueError(
                        f"surfaces[{repr(key)}] ({val}) must be between 0 and 1"
                    )
    else:
        user_alpha = {}
    surfaces = list(surfaces)
    for si, s in enumerate(surfaces):
        _validate_type(s, "str", f"surfaces[{si}]")

    bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel, None))
    assert isinstance(bem, ConductorModel) or bem is None

    _check_option("coord_frame", coord_frame, ["head", "meg", "mri", "auto"])
    if coord_frame == "auto":
        coord_frame = "head" if trans is None else "mri"

    if src is not None:
        src = _ensure_src(src)
        src_subject = src._subject
        subject = src_subject if subject is None else subject
        if src_subject is not None and subject != src_subject:
            raise ValueError(
                f'subject ("{subject}") did not match the '
                f'subject name in src ("{src_subject}")'
            )
    # configure transforms
    if isinstance(trans, str) and trans == "auto":
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        trans = _find_trans(subject, subjects_dir)
    trans, trans_type = _get_trans(trans, fro="head", to="mri")

    picks = pick_types(
        info,
        meg=("sensors" in meg),
        ref_meg=("ref" in meg),
        eeg=(len(eeg) > 0),
        ecog=ecog,
        seeg=seeg,
        dbs=dbs,
        fnirs=(len(fnirs) > 0),
    )
    if trans_type == "identity":
        # Some stuff is natively in head coords, others in MRI coords
        msg = (
            "A head<->mri transformation matrix (trans) is required "
            f"to plot {{}} in {coord_frame} coordinates, "
            "`trans=None` is not allowed"
        )
        if fwd is not None:
            fwd_frame = _frame_to_str[fwd["coord_frame"]]
            if fwd_frame != coord_frame:
                raise ValueError(
                    msg.format(f"a {fwd_frame}-coordinate forward solution")
                )
        if src is not None:
            src_frame = _frame_to_str[src[0]["coord_frame"]]
            if src_frame != coord_frame:
                raise ValueError(msg.format(f"a {src_frame}-coordinate source space"))
        if mri_fiducials is not False and coord_frame != "mri":
            raise ValueError(msg.format("mri fiducials"))
        # only enforce needing `trans` if there are channels in "head"/"device"
        if picks.size and coord_frame == "mri":
            raise ValueError(msg.format("sensors"))
        # if only plotting sphere model no trans needed
        if bem is not None:
            if not bem["is_sphere"]:
                if coord_frame != "mri":
                    raise ValueError(msg.format("a BEM"))
            elif surfaces not in (["brain"], []):  # can only plot these
                raise ValueError(msg.format(", ".join(surfaces) + " surfaces"))
        elif len(surfaces) > 0 and coord_frame != "mri":
            raise ValueError(msg.format(", ".join(surfaces) + " surfaces"))
        trans = Transform("head", "mri")  # not used so just use identity
    # get transforms
    head_mri_t = _get_trans(trans, "head", "mri")[0]
    to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

    # Surfaces:
    # both the head and helmet will be in MRI coordinates after this
    surfs = dict()

    # Brain surface:
    brain = sorted(set(surfaces) & set(["brain", "pial", "white", "inflated"]))
    if len(brain) > 1:
        raise ValueError(f"Only one brain surface can be plotted, got {brain}.")
    brain = brain[0] if brain else False
    if brain is not False:
        surfaces.pop(surfaces.index(brain))
        if bem is not None and bem["is_sphere"] and brain == "brain":
            surfs["lh"] = _bem_find_surface(bem, "brain")
        else:
            brain = "pial" if brain == "brain" else brain
            subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
            for hemi in ["lh", "rh"]:
                brain_fname = subjects_dir / subject / "surf" / f"{hemi}.{brain}"
                if not brain_fname.is_file():
                    raise RuntimeError(
                        f"No brain surface found for subject {subject}, "
                        f"expected {brain_fname} to exist"
                    )
                surfs[hemi] = _read_mri_surface(brain_fname)
            subjects_dir = str(subjects_dir)

    # Head surface:
    head_keys = ("auto", "head", "outer_skin", "head-dense", "seghead")
    head = [s for s in surfaces if s in head_keys]
    if len(head) > 1:
        raise ValueError(f"Can only supply one head-like surface name, got {head}")
    head = head[0] if head else False
    if head is not False:
        surfaces.pop(surfaces.index(head))
    elif "projected" in eeg:
        raise ValueError(
            "A head surface is required to project EEG, "
            '"head", "outer_skin", "head-dense" or "seghead" '
            'must be in surfaces or surfaces must be "auto"'
        )

    # Skull surface:
    skulls = [s for s in surfaces if s in ("outer_skull", "inner_skull")]
    for skull_name in skulls:
        surfaces.pop(surfaces.index(skull_name))
        skull = _get_skull_surface(
            skull_name.split("_")[0], subject, subjects_dir, bem=bem
        )
        skull["name"] = skull_name  # set name for alpha
        surfs[skull_name] = skull

    # we've looked through all of them, raise if some remain
    if len(surfaces) > 0:
        raise ValueError(f"Unknown surface type{_pl(surfaces)}: {surfaces}")

    # set colors and alphas
    defaults = DEFAULTS["coreg"]
    no_deep = not (dbs or seeg) or pick_types(info, dbs=True, seeg=True).size == 0
    max_alpha = 1.0 if no_deep else 0.75
    hemi_val = 0.5
    if src is None or (brain and any(s["type"] == "surf" for s in src)):
        hemi_val = max_alpha
    alpha_range = np.linspace(max_alpha / 2.0, 0, 5)[: len(skulls) + 1]
    if src is None and brain is False and len(skulls) == 0 and not show_axes:
        head_alpha = max_alpha
    else:
        head_alpha = alpha_range[0]
    alphas = dict(lh=hemi_val, rh=hemi_val)
    colors = dict(lh=(0.5,) * 3, rh=(0.5,) * 3)
    for idx, name in enumerate(skulls):
        alphas[name] = alpha_range[idx + 1]
        colors[name] = (0.95 - idx * 0.2, 0.85, 0.95 - idx * 0.2)
    if brain is not False and brain in user_alpha:
        alphas["lh"] = alphas["rh"] = user_alpha.pop(brain)
    # replace default alphas with specified user_alpha
    for k, v in user_alpha.items():
        if v is not None:
            alphas[k] = v
        if k in head_keys and v is not None:
            head_alpha = v
    fid_colors = tuple(defaults[f"{key}_color"] for key in ("lpa", "nasion", "rpa"))

    # initialize figure
    renderer = _get_renderer(
        fig,
        name=f"Sensor alignment: {subject}",
        bgcolor=(0.5, 0.5, 0.5),
        size=(800, 800),
    )
    renderer.set_interaction(interaction)

    # plot head
    _, _, head_surf = _plot_head_surface(
        renderer,
        head,
        subject,
        subjects_dir,
        bem,
        coord_frame,
        to_cf_t,
        alpha=head_alpha,
    )

    # plot helmet
    if "helmet" in meg and pick_types(info, meg=True).size > 0:
        _, _, src_surf = _plot_helmet(
            renderer,
            info,
            to_cf_t,
            head_mri_t,
            coord_frame,
            alpha=sensor_alpha["meg_helmet"],
        )

    # plot surfaces
    if brain and "lh" not in surfs:  # one layer sphere
        assert bem["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        center = bem["r0"].copy()
        center = apply_trans(to_cf_t["head"], center)
        renderer.sphere(center, scale=0.01, color=colors["lh"], opacity=alphas["lh"])
    if show_axes:
        _plot_axes(renderer, info, to_cf_t, head_mri_t)

    # plot points
    _check_option("dig", dig, (True, False, "fiducials"))
    if dig:
        if dig is True:
            _plot_hpi_coils(renderer, info, to_cf_t)
            _plot_head_shape_points(renderer, info, to_cf_t)
        _plot_head_fiducials(renderer, info, to_cf_t, fid_colors)

    if mri_fiducials:
        _plot_mri_fiducials(
            renderer, mri_fiducials, subjects_dir, subject, to_cf_t, fid_colors
        )

    for key, surf in surfs.items():
        # Surfs can sometimes be in head coords (e.g., if coming from sphere)
        assert isinstance(surf, dict), f"{key}: {type(surf)}"
        surf = transform_surface_to(
            surf, coord_frame, [to_cf_t["mri"], to_cf_t["head"]], copy=True
        )
        renderer.surface(
            surface=surf,
            color=colors[key],
            opacity=alphas[key],
            backface_culling=(key != "helmet"),
        )

    # plot sensors (NB snapshot_brain_montage relies on the last thing being
    # plotted being the sensors, so we need to do this after the surfaces)
    if picks.size > 0:
        _plot_sensors_3d(
            renderer,
            info,
            to_cf_t,
            picks,
            meg,
            eeg,
            fnirs,
            warn_meg,
            head_surf,
            "m",
            sensor_alpha=sensor_alpha,
            sensor_colors=sensor_colors,
            sensor_scales=sensor_scales,
        )

    if src is not None:
        atlas_ids, colors = read_freesurfer_lut()
        for ss in src:
            src_rr = ss["rr"][ss["inuse"].astype(bool)]
            src_nn = ss["nn"][ss["inuse"].astype(bool)]

            # update coordinate frame
            src_trans = to_cf_t[_frame_to_str[src[0]["coord_frame"]]]
            src_rr = apply_trans(src_trans, src_rr)
            src_nn = apply_trans(src_trans, src_nn, move=False)

            # volume sources
            if ss["type"] == "vol":
                seg_name = ss.get("seg_name", None)
                if seg_name is not None and seg_name in colors:
                    color = colors[seg_name][:3]
                    color = tuple(i / 256.0 for i in color)
                else:
                    color = (1.0, 1.0, 0.0)

            # surface and discrete sources
            else:
                color = (1.0, 1.0, 0.0)

            if len(src_rr) > 0:
                renderer.quiver3d(
                    x=src_rr[:, 0],
                    y=src_rr[:, 1],
                    z=src_rr[:, 2],
                    u=src_nn[:, 0],
                    v=src_nn[:, 1],
                    w=src_nn[:, 2],
                    color=color,
                    mode="cylinder",
                    scale=3e-3,
                    opacity=0.75,
                    glyph_height=0.25,
                    glyph_center=(0.0, 0.0, 0.0),
                    glyph_resolution=20,
                    backface_culling=True,
                )

    if fwd is not None:
        _plot_forward(renderer, fwd, to_cf_t[_frame_to_str[fwd["coord_frame"]]])

    renderer.set_camera(
        azimuth=90, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0)
    )
    renderer.show()
    return renderer.scene()


def _handle_sensor_types(meg, eeg, fnirs):
    """Handle plotting inputs for sensors types."""
    if eeg is True:
        eeg = ["original"]
    elif eeg is False:
        eeg = list()

    warn_meg = meg is not None  # only warn if the value is explicit
    if meg is True:
        meg = ["helmet", "sensors", "ref"]
    elif meg is None:
        meg = ["helmet", "sensors"]
    elif meg is False:
        meg = list()

    if fnirs is True:
        fnirs = ["pairs"]
    elif fnirs is False:
        fnirs = list()

    if isinstance(meg, str):
        meg = [meg]
    if isinstance(eeg, str):
        eeg = [eeg]
    if isinstance(fnirs, str):
        fnirs = [fnirs]

    alpha_map = dict(
        meg=dict(sensors="meg", helmet="meg_helmet", ref="ref_meg"),
        eeg=dict(original="eeg", projected="eeg_projected"),
        fnirs=dict(channels="fnirs", pairs="fnirs_pairs"),
    )
    sensor_alpha = {
        key: dict(meg_helmet=0.25, meg=0.25).get(key, 0.8)
        for ch_dict in alpha_map.values()
        for key in ch_dict.values()
    }
    for kind, var in zip(("eeg", "meg", "fnirs"), (eeg, meg, fnirs)):
        _validate_type(var, (list, tuple, dict), f"{kind}")
        for ix, x in enumerate(var):
            which = f"{kind} key {ix}" if isinstance(var, dict) else f"{kind}[{ix}]"
            _validate_type(x, str, which)
            if isinstance(var, dict) and x in alpha_map[kind]:
                alpha = var[x]
                _validate_type(alpha, "numeric", f"{kind}[{ix}]")
                if not 0 <= alpha <= 1:
                    raise ValueError(
                        f"{kind}[{ix}] alpha value must be between 0 and 1, got {alpha}"
                    )
                sensor_alpha[alpha_map[kind][x]] = alpha
    meg, eeg, fnirs = tuple(meg), tuple(eeg), tuple(fnirs)
    for xi, x in enumerate(meg):
        _check_option(f"meg[{xi}]", x, ("helmet", "sensors", "ref"))
    for xi, x in enumerate(eeg):
        _check_option(f"eeg[{xi}]", x, ("original", "projected"))
    for xi, x in enumerate(fnirs):
        _check_option(f"fnirs[{xi}]", x, ("channels", "pairs", "sources", "detectors"))
    # Add these for our True-only options, too -- eventually should support dict.
    sensor_alpha.update(
        seeg=0.8,
        ecog=0.8,
        source=sensor_alpha["fnirs"],
        detector=sensor_alpha["fnirs"],
    )
    return meg, eeg, fnirs, warn_meg, sensor_alpha


@verbose
def _ch_pos_in_coord_frame(info, to_cf_t, warn_meg=True, verbose=None):
    """Transform positions from head/device/mri to a coordinate frame."""
    from ..forward import _create_meg_coils
    from ..forward._make_forward import _read_coil_defs

    chs = dict(ch_pos=dict(), sources=dict(), detectors=dict())
    unknown_chs = list()  # prepare for chs with unknown coordinate frame
    type_counts = dict()
    coilset = _read_coil_defs(verbose=False)
    for idx in range(info["nchan"]):
        ch_type = channel_type(info, idx)
        if ch_type in type_counts:
            type_counts[ch_type] += 1
        else:
            type_counts[ch_type] = 1
        type_slices = dict(ch_pos=slice(0, 3))
        if ch_type in _FNIRS_CH_TYPES_SPLIT:
            # add sensors and detectors too for fNIRS
            type_slices.update(sources=slice(3, 6), detectors=slice(6, 9))
        for type_name, type_slice in type_slices.items():
            if ch_type in _MEG_CH_TYPES_SPLIT + ("ref_meg",):
                coil_trans = _loc_to_coil_trans(info["chs"][idx]["loc"])
                # Here we prefer accurate geometry in case we need to
                # ConvexHull the coil, we want true 3D geometry (and not, for
                # example, a straight line / 1D geometry)
                this_coil = [info["chs"][idx]]
                try:
                    coil = _create_meg_coils(
                        this_coil, acc="accurate", coilset=coilset
                    )[0]
                except RuntimeError:  # we don't have an accurate one
                    coil = _create_meg_coils(this_coil, acc="normal", coilset=coilset)[
                        0
                    ]
                # store verts as ch_coord
                ch_coord, triangles = _sensor_shape(coil)
                ch_coord = apply_trans(coil_trans, ch_coord)
                if len(ch_coord) == 0 and warn_meg:
                    warn(f"MEG sensor {info.ch_names[idx]} not found.")
            else:
                ch_coord = info["chs"][idx]["loc"][type_slice]
            ch_coord_frame = info["chs"][idx]["coord_frame"]
            if ch_coord_frame not in (
                FIFF.FIFFV_COORD_UNKNOWN,
                FIFF.FIFFV_COORD_DEVICE,
                FIFF.FIFFV_COORD_HEAD,
                FIFF.FIFFV_COORD_MRI,
            ):
                raise RuntimeError(
                    f"Channel {info.ch_names[idx]} has coordinate frame "
                    f'{ch_coord_frame}, must be "meg", "head" or "mri".'
                )
            # set unknown as head first
            if ch_coord_frame == FIFF.FIFFV_COORD_UNKNOWN:
                unknown_chs.append(info.ch_names[idx])
                ch_coord_frame = FIFF.FIFFV_COORD_HEAD
            ch_coord = apply_trans(to_cf_t[_frame_to_str[ch_coord_frame]], ch_coord)
            if ch_type in _MEG_CH_TYPES_SPLIT + ("ref_meg",):
                chs[type_name][info.ch_names[idx]] = (ch_coord, triangles)
            else:
                chs[type_name][info.ch_names[idx]] = ch_coord
    if unknown_chs:
        warn(
            f'Got coordinate frame "unknown" for {unknown_chs}, assuming '
            '"head" coordinates.'
        )
    logger.info(
        "Channel types::\t"
        + ", ".join([f"{ch_type}: {count}" for ch_type, count in type_counts.items()])
    )
    return chs["ch_pos"], chs["sources"], chs["detectors"]


def _plot_head_surface(
    renderer, head, subject, subjects_dir, bem, coord_frame, to_cf_t, alpha, color=None
):
    """Render a head surface in a 3D scene."""
    color = DEFAULTS["coreg"]["head_color"] if color is None else color
    actor = None
    src_surf = dst_surf = None
    if head is not False:
        src_surf = _get_head_surface(head, subject, subjects_dir, bem=bem)
        src_surf = transform_surface_to(
            src_surf, coord_frame, [to_cf_t["mri"], to_cf_t["head"]], copy=True
        )
        actor, dst_surf = renderer.surface(
            surface=src_surf, color=color, opacity=alpha, backface_culling=False
        )
    return actor, dst_surf, src_surf


def _plot_helmet(
    renderer,
    info,
    to_cf_t,
    head_mri_t,
    coord_frame,
    *,
    alpha=0.25,
    scale=1.0,
):
    color = DEFAULTS["coreg"]["helmet_color"]
    src_surf = get_meg_helmet_surf(info, head_mri_t)
    assert src_surf["coord_frame"] == FIFF.FIFFV_COORD_MRI
    if to_cf_t is not None:
        src_surf = transform_surface_to(
            src_surf, coord_frame, [to_cf_t["mri"], to_cf_t["head"]], copy=True
        )
    actor, dst_surf = renderer.surface(
        surface=src_surf, color=color, opacity=alpha, backface_culling=False
    )
    return actor, dst_surf, src_surf


def _plot_axes(renderer, info, to_cf_t, head_mri_t):
    """Render different axes a 3D scene."""
    axes = [(to_cf_t["head"], (0.9, 0.3, 0.3))]  # always show head
    if not np.allclose(head_mri_t["trans"], np.eye(4)):  # Show MRI
        axes.append((to_cf_t["mri"], (0.6, 0.6, 0.6)))
    if pick_types(info, meg=True).size > 0:  # Show MEG
        axes.append((to_cf_t["meg"], (0.0, 0.6, 0.6)))
    actors = list()
    for ax in axes:
        x, y, z = np.tile(ax[0]["trans"][:3, 3], 3).reshape((3, 3)).T
        u, v, w = ax[0]["trans"][:3, :3]
        actor, _ = renderer.sphere(
            center=np.column_stack((x[0], y[0], z[0])), color=ax[1], scale=3e-3
        )
        actors.append(actor)
        actor, _ = renderer.quiver3d(
            x=x,
            y=y,
            z=z,
            u=u,
            v=v,
            w=w,
            mode="arrow",
            scale=2e-2,
            color=ax[1],
            scale_mode="scalar",
            resolution=20,
            scalars=[0.33, 0.66, 1.0],
        )
        actors.append(actor)
    return actors


def _plot_head_fiducials(renderer, info, to_cf_t, fid_colors):
    defaults = DEFAULTS["coreg"]
    car_loc = _fiducial_coords(info["dig"], FIFF.FIFFV_COORD_HEAD)
    car_loc = apply_trans(to_cf_t["head"], car_loc)
    if len(car_loc) == 0:
        warn("Digitization points not found. Cannot plot digitization.")
    actors = list()
    for color, data in zip(fid_colors, car_loc):
        actor, _ = renderer.sphere(
            center=data,
            color=color,
            scale=defaults["dig_fid_scale"],
            opacity=defaults["dig_fid_opacity"],
            backface_culling=True,
        )
        actors.append(actor)
    return actors


def _plot_mri_fiducials(
    renderer, mri_fiducials, subjects_dir, subject, to_cf_t, fid_colors
):
    from ..coreg import get_mni_fiducials

    defaults = DEFAULTS["coreg"]
    if mri_fiducials is True:
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        if subject is None:
            raise ValueError(
                "Subject needs to be specified to "
                "automatically find the fiducials file."
            )
        mri_fiducials = subjects_dir / subject / "bem" / (subject + "-fiducials.fif")
    if isinstance(mri_fiducials, str) and mri_fiducials == "estimated":
        mri_fiducials = get_mni_fiducials(subject, subjects_dir)
    elif isinstance(mri_fiducials, str | Path | os.PathLike):
        mri_fiducials, cf = read_fiducials(mri_fiducials)
        if cf != FIFF.FIFFV_COORD_MRI:
            raise ValueError("Fiducials are not in MRI space")
    if isinstance(mri_fiducials, np.ndarray):
        fid_loc = mri_fiducials
    else:
        fid_loc = _fiducial_coords(mri_fiducials, FIFF.FIFFV_COORD_MRI)
    fid_loc = apply_trans(to_cf_t["mri"], fid_loc)
    transform = np.eye(4)
    transform[:3, :3] = to_cf_t["mri"]["trans"][:3, :3] * defaults["mri_fid_scale"]
    # rotate around Z axis 45 deg first
    transform = transform @ rotation(0, 0, np.pi / 4)
    actors = list()
    for color, data in zip(fid_colors, fid_loc):
        actor, _ = renderer.quiver3d(
            x=data[0],
            y=data[1],
            z=data[2],
            u=1.0,
            v=0.0,
            w=0.0,
            color=color,
            mode="oct",
            scale=1.0,
            opacity=defaults["mri_fid_opacity"],
            backface_culling=True,
            solid_transform=transform,
        )
        actors.append(actor)
    return actors


def _plot_hpi_coils(
    renderer,
    info,
    to_cf_t,
    opacity=0.5,
    scale=None,
    orient_glyphs=False,
    scale_by_distance=False,
    surf=None,
    check_inside=None,
    nearest=None,
):
    defaults = DEFAULTS["coreg"]
    scale = defaults["hpi_scale"] if scale is None else scale
    hpi_loc = np.array(
        [
            d["r"]
            for d in (info["dig"] or [])
            if (
                d["kind"] == FIFF.FIFFV_POINT_HPI
                and d["coord_frame"] == FIFF.FIFFV_COORD_HEAD
            )
        ]
    )
    hpi_loc = apply_trans(to_cf_t["head"], hpi_loc)
    actor, _ = _plot_glyphs(
        renderer=renderer,
        loc=hpi_loc,
        color=defaults["hpi_color"],
        scale=scale,
        opacity=opacity,
        orient_glyphs=orient_glyphs,
        scale_by_distance=scale_by_distance,
        surf=surf,
        backface_culling=True,
        check_inside=check_inside,
        nearest=nearest,
    )
    return actor


def _get_nearest(nearest, check_inside, project_to_trans, proj_rr):
    idx = nearest.query(proj_rr)[1]
    proj_pts = apply_trans(project_to_trans, nearest.data[idx])
    proj_nn = apply_trans(project_to_trans, check_inside.surf["nn"][idx], move=False)
    return proj_pts, proj_nn


def _orient_glyphs(
    pts,
    surf,
    project_to_surface=False,
    mark_inside=False,
    check_inside=None,
    nearest=None,
):
    if check_inside is None:
        check_inside = _CheckInside(surf, mode="pyvista")
    if nearest is None:
        nearest = _DistanceQuery(surf["rr"])
    project_to_trans = np.eye(4)
    inv_trans = np.linalg.inv(project_to_trans)
    proj_rr = apply_trans(inv_trans, pts)
    proj_pts, proj_nn = _get_nearest(nearest, check_inside, project_to_trans, proj_rr)
    vec = pts - proj_pts  # point to the surface
    nn = proj_nn
    scalars = np.ones(len(pts))
    if mark_inside and not project_to_surface:
        scalars[:] = ~check_inside(proj_rr)
    dist = np.linalg.norm(vec, axis=-1, keepdims=True)
    vectors = (250 * dist + 1) * nn
    return scalars, vectors, proj_pts


def _plot_glyphs(
    renderer,
    loc,
    color,
    scale,
    opacity=1,
    mode="cylinder",
    orient_glyphs=False,
    scale_by_distance=False,
    project_points=False,
    mark_inside=False,
    surf=None,
    backface_culling=False,
    check_inside=None,
    nearest=None,
):
    from matplotlib.colors import ListedColormap, to_rgba

    _validate_type(mark_inside, bool, "mark_inside")
    if surf is not None and len(loc) > 0:
        defaults = DEFAULTS["coreg"]
        scalars, vectors, proj_pts = _orient_glyphs(
            loc, surf, project_points, mark_inside, check_inside, nearest
        )
        if mark_inside:
            colormap = ListedColormap([to_rgba("darkslategray"), to_rgba(color)])
            color = None
            clim = [0, 1]
        else:
            scalars = None
            colormap = None
            clim = None
        mode = "cylinder" if orient_glyphs else "sphere"
        scale_mode = "vector" if scale_by_distance else "none"
        x, y, z = proj_pts.T if project_points else loc.T
        u, v, w = vectors.T
        return renderer.quiver3d(
            x,
            y,
            z,
            u,
            v,
            w,
            color=color,
            scale=scale,
            mode=mode,
            glyph_height=defaults["eegp_height"],
            glyph_center=(0.0, -defaults["eegp_height"], 0),
            resolution=16,
            glyph_resolution=16,
            glyph_radius=None,
            opacity=opacity,
            scale_mode=scale_mode,
            scalars=scalars,
            colormap=colormap,
            clim=clim,
        )
    else:
        return renderer.sphere(
            center=loc,
            color=color,
            scale=scale,
            opacity=opacity,
            backface_culling=backface_culling,
        )


@verbose
def _plot_head_shape_points(
    renderer,
    info,
    to_cf_t,
    opacity=0.25,
    orient_glyphs=False,
    scale_by_distance=False,
    mark_inside=False,
    surf=None,
    mask=None,
    check_inside=None,
    nearest=None,
    verbose=False,
):
    defaults = DEFAULTS["coreg"]
    ext_loc = np.array(
        [
            d["r"]
            for d in (info["dig"] or [])
            if (
                d["kind"] == FIFF.FIFFV_POINT_EXTRA
                and d["coord_frame"] == FIFF.FIFFV_COORD_HEAD
            )
        ]
    )
    ext_loc = apply_trans(to_cf_t["head"], ext_loc)
    ext_loc = ext_loc[mask] if mask is not None else ext_loc
    actor, _ = _plot_glyphs(
        renderer=renderer,
        loc=ext_loc,
        color=defaults["extra_color"],
        scale=defaults["extra_scale"],
        opacity=opacity,
        orient_glyphs=orient_glyphs,
        scale_by_distance=scale_by_distance,
        mark_inside=mark_inside,
        surf=surf,
        backface_culling=True,
        check_inside=check_inside,
        nearest=nearest,
    )
    return actor


def _plot_forward(renderer, fwd, fwd_trans, fwd_scale=1, scale=1.5e-3, alpha=1):
    from ..forward import Forward

    _validate_type(fwd, [Forward])
    n_dipoles = fwd["source_rr"].shape[0]
    fwd_rr = fwd["source_rr"]
    if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
        fwd_nn = fwd["source_nn"].reshape(-1, 1, 3)
    else:
        fwd_nn = fwd["source_nn"].reshape(-1, 3, 3)
    # update coordinate frame
    fwd_rr = apply_trans(fwd_trans, fwd_rr) * fwd_scale
    fwd_nn = apply_trans(fwd_trans, fwd_nn, move=False)
    red = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.0)
    blue = (0.0, 0.0, 1.0)
    actors = list()
    for ori, color in zip(range(fwd_nn.shape[1]), (red, green, blue)):
        actor, _ = renderer.quiver3d(
            *fwd_rr.T,
            *fwd_nn[:, ori].T,
            color=color,
            mode="arrow",
            scale_mode="scalar",
            scalars=np.ones(n_dipoles),
            scale=scale,
            opacity=alpha,
        )
        actors.append(actor)
    return actors


def _plot_sensors_3d(
    renderer,
    info,
    to_cf_t,
    picks,
    meg,
    eeg,
    fnirs,
    warn_meg,
    head_surf,
    units,
    sensor_alpha,
    orient_glyphs=False,
    scale_by_distance=False,
    project_points=False,
    surf=None,
    check_inside=None,
    nearest=None,
    sensor_colors=None,
    sensor_scales=None,
):
    """Render sensors in a 3D scene."""
    from matplotlib.colors import to_rgba_array

    defaults = DEFAULTS["coreg"]
    ch_pos, sources, detectors = _ch_pos_in_coord_frame(
        pick_info(info, picks), to_cf_t=to_cf_t, warn_meg=warn_meg
    )

    actors = defaultdict(lambda: list())
    locs = defaultdict(lambda: list())
    unit_scalar = 1 if units == "m" else 1e3
    for ch_name, ch_coord in ch_pos.items():
        ch_type = channel_type(info, info.ch_names.index(ch_name))
        # for default picking
        if ch_type in _FNIRS_CH_TYPES_SPLIT:
            ch_type = "fnirs"
        elif ch_type in _MEG_CH_TYPES_SPLIT:
            ch_type = "meg"
        # only plot sensor locations if channels/original in selection
        plot_sensors = True
        if ch_type == "fnirs":
            if not fnirs or "channels" not in fnirs:
                plot_sensors = False
        elif ch_type == "eeg":
            if not eeg or "original" not in eeg:
                plot_sensors = False
        elif ch_type == "meg":
            if not meg or "sensors" not in meg:
                plot_sensors = False
        # plot sensors
        if isinstance(ch_coord, tuple):  # is meg, plot coil
            ch_coord = dict(rr=ch_coord[0] * unit_scalar, tris=ch_coord[1])
        if plot_sensors:
            locs[ch_type].append(ch_coord)
        if ch_name in sources and "sources" in fnirs:
            locs["source"].append(sources[ch_name])
        if ch_name in detectors and "detectors" in fnirs:
            locs["detector"].append(detectors[ch_name])
        # Plot these now
        if ch_name in sources and ch_name in detectors and "pairs" in fnirs:
            actor, _ = renderer.tube(  # array of origin and dest points
                origin=sources[ch_name][np.newaxis] * unit_scalar,
                destination=detectors[ch_name][np.newaxis] * unit_scalar,
                radius=0.001 * unit_scalar,
                opacity=sensor_alpha["fnirs_pairs"],
            )
            actors[ch_type].append(actor)
            del ch_type

    # now actually plot the sensors
    extra = ""
    types = (dict, None)
    if len(locs) == 0:
        return
    elif len(locs) == 1:
        # Upsample from array-like to dict when there is one channel type
        extra = "(or array-like since only one sensor type is plotted)"
        if sensor_colors is not None and not isinstance(sensor_colors, dict):
            sensor_colors = {
                list(locs)[0]: to_rgba_array(sensor_colors),
            }
        if sensor_scales is not None and not isinstance(sensor_scales, dict):
            sensor_scales = {
                list(locs)[0]: sensor_scales,
            }
    else:
        extra = f"when more than one channel type ({list(locs)}) is plotted"
    _validate_type(sensor_colors, types, "sensor_colors", extra=extra)
    _validate_type(sensor_scales, types, "sensor_scales", extra=extra)
    del extra, types
    if sensor_colors is None:
        sensor_colors = dict()
    if sensor_scales is None:
        sensor_scales = dict()
    assert isinstance(sensor_colors, dict)
    assert isinstance(sensor_scales, dict)
    for ch_type, sens_loc in locs.items():
        logger.debug(f"Drawing {ch_type} sensors ({len(sens_loc)})")
        assert len(sens_loc)  # should be guaranteed above
        colors = to_rgba_array(sensor_colors.get(ch_type, defaults[ch_type + "_color"]))
        scales = np.atleast_1d(
            sensor_scales.get(ch_type, defaults[ch_type + "_scale"] * unit_scalar)
        )
        _check_option(
            f"len(sensor_colors[{repr(ch_type)}])",
            colors.shape[0],
            (len(sens_loc), 1),
        )
        _check_option(
            f"len(sensor_scales[{repr(ch_type)}])",
            scales.shape[0],
            (len(sens_loc), 1),
        )
        # Check that the scale is numerical
        assert np.issubdtype(scales.dtype, np.number), (
            f"scales for {ch_type} must contain only numerical values, "
            f"got {scales} instead."
        )

        this_alpha = sensor_alpha[ch_type]
        if isinstance(sens_loc[0], dict):  # meg coil
            if len(colors) == 1:
                colors = [colors[0]] * len(sens_loc)
            for surface, color in zip(sens_loc, colors):
                actor, _ = renderer.surface(
                    surface=surface,
                    color=color[:3],
                    opacity=this_alpha * color[3],
                    backface_culling=False,  # visible from all sides
                )
                actors[ch_type].append(actor)
        else:
            sens_loc = np.array(sens_loc, float)
            mask = ~np.isnan(sens_loc).any(axis=1)
            if len(colors) == 1 and len(scales) == 1:
                # Single color mode (one actor)
                actor, _ = _plot_glyphs(
                    renderer=renderer,
                    loc=sens_loc[mask] * unit_scalar,
                    color=colors[0, :3],
                    scale=scales[0],
                    opacity=this_alpha * colors[0, 3],
                    orient_glyphs=orient_glyphs,
                    scale_by_distance=scale_by_distance,
                    project_points=project_points,
                    surf=surf,
                    check_inside=check_inside,
                    nearest=nearest,
                )
                actors[ch_type].append(actor)
            elif len(colors) == len(sens_loc) and len(scales) == 1:
                # Multi-color single scale mode (multiple actors)
                for loc, color, usable in zip(sens_loc, colors, mask):
                    if not usable:
                        continue
                    actor, _ = _plot_glyphs(
                        renderer=renderer,
                        loc=loc * unit_scalar,
                        color=color[:3],
                        scale=scales[0],
                        opacity=this_alpha * color[3],
                        orient_glyphs=orient_glyphs,
                        scale_by_distance=scale_by_distance,
                        project_points=project_points,
                        surf=surf,
                        check_inside=check_inside,
                        nearest=nearest,
                    )
                    actors[ch_type].append(actor)
            elif len(colors) == 1 and len(scales) == len(sens_loc):
                # Multi-scale single color mode (multiple actors)
                for loc, scale, usable in zip(sens_loc, scales, mask):
                    if not usable:
                        continue
                    actor, _ = _plot_glyphs(
                        renderer=renderer,
                        loc=loc * unit_scalar,
                        color=colors[0, :3],
                        scale=scale,
                        opacity=this_alpha * colors[0, 3],
                        orient_glyphs=orient_glyphs,
                        scale_by_distance=scale_by_distance,
                        project_points=project_points,
                        surf=surf,
                        check_inside=check_inside,
                        nearest=nearest,
                    )
                    actors[ch_type].append(actor)
            else:
                # Multi-color multi-scale mode (multiple actors)
                for loc, color, scale, usable in zip(sens_loc, colors, scales, mask):
                    if not usable:
                        continue
                    actor, _ = _plot_glyphs(
                        renderer=renderer,
                        loc=loc * unit_scalar,
                        color=color[:3],
                        scale=scale,
                        opacity=this_alpha * color[3],
                        orient_glyphs=orient_glyphs,
                        scale_by_distance=scale_by_distance,
                        project_points=project_points,
                        surf=surf,
                        check_inside=check_inside,
                        nearest=nearest,
                    )
                    actors[ch_type].append(actor)
        if ch_type == "eeg" and "projected" in eeg:
            logger.info("Projecting sensors to the head surface")
            eegp_loc, eegp_nn = _project_onto_surface(
                sens_loc[mask], head_surf, project_rrs=True, return_nn=True
            )[2:4]
            eegp_loc *= unit_scalar
            actor, _ = renderer.quiver3d(
                x=eegp_loc[:, 0],
                y=eegp_loc[:, 1],
                z=eegp_loc[:, 2],
                u=eegp_nn[:, 0],
                v=eegp_nn[:, 1],
                w=eegp_nn[:, 2],
                color=defaults["eegp_color"],
                mode="cylinder",
                scale=defaults["eegp_scale"] * unit_scalar,
                opacity=sensor_alpha["eeg_projected"],
                glyph_height=defaults["eegp_height"],
                glyph_center=(0.0, -defaults["eegp_height"] / 2.0, 0),
                glyph_resolution=20,
                backface_culling=True,
            )
            actors["eeg"].append(actor)
    actors = dict(actors)  # get rid of defaultdict

    return actors


def _make_tris_fan(n_vert):
    """Make tris given a number of vertices of a circle-like obj."""
    tris = np.zeros((n_vert - 2, 3), int)
    tris[:, 2] = np.arange(2, n_vert)
    tris[:, 1] = tris[:, 2] - 1
    return tris


def _sensor_shape(coil):
    """Get the sensor shape vertices."""
    try:
        from scipy.spatial import QhullError
    except ImportError:  # scipy < 1.8
        from scipy.spatial.qhull import QhullError
    id_ = coil["type"] & 0xFFFF
    z_value = 0
    # Square figure eight
    if id_ in (
        FIFF.FIFFV_COIL_NM_122,
        FIFF.FIFFV_COIL_VV_PLANAR_W,
        FIFF.FIFFV_COIL_VV_PLANAR_T1,
        FIFF.FIFFV_COIL_VV_PLANAR_T2,
    ):
        # wound by right hand rule such that +x side is "up" (+z)
        long_side = coil["size"]  # length of long side (meters)
        offset = 0.0025  # offset of the center portion of planar grad coil
        rrs = np.array(
            [
                [offset, -long_side / 2.0],
                [long_side / 2.0, -long_side / 2.0],
                [long_side / 2.0, long_side / 2.0],
                [offset, long_side / 2.0],
                [-offset, -long_side / 2.0],
                [-long_side / 2.0, -long_side / 2.0],
                [-long_side / 2.0, long_side / 2.0],
                [-offset, long_side / 2.0],
            ]
        )
        tris = np.concatenate(
            (_make_tris_fan(4), _make_tris_fan(4)[:, ::-1] + 4), axis=0
        )
        # Offset for visibility (using heuristic for sanely named Neuromag coils)
        z_value = 0.001 * (1 + coil["chname"].endswith("2"))
    # Square
    elif id_ in (
        FIFF.FIFFV_COIL_POINT_MAGNETOMETER,
        FIFF.FIFFV_COIL_VV_MAG_T1,
        FIFF.FIFFV_COIL_VV_MAG_T2,
        FIFF.FIFFV_COIL_VV_MAG_T3,
        FIFF.FIFFV_COIL_KIT_REF_MAG,
    ):
        # square magnetometer (potentially point-type)
        size = 0.001 if id_ == 2000 else (coil["size"] / 2.0)
        rrs = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]) * size
        tris = _make_tris_fan(4)
    # Circle
    elif id_ in (
        FIFF.FIFFV_COIL_MAGNES_MAG,
        FIFF.FIFFV_COIL_MAGNES_REF_MAG,
        FIFF.FIFFV_COIL_CTF_REF_MAG,
        FIFF.FIFFV_COIL_BABY_MAG,
        FIFF.FIFFV_COIL_BABY_REF_MAG,
        FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG,
    ):
        n_pts = 15  # number of points for circle
        circle = np.exp(2j * np.pi * np.arange(n_pts) / float(n_pts))
        circle = np.concatenate(([0.0], circle))
        circle *= coil["size"] / 2.0  # radius of coil
        rrs = np.array([circle.real, circle.imag]).T
        tris = _make_tris_fan(n_pts + 1)
    # Circle
    elif id_ in (
        FIFF.FIFFV_COIL_MAGNES_GRAD,
        FIFF.FIFFV_COIL_CTF_GRAD,
        FIFF.FIFFV_COIL_CTF_REF_GRAD,
        FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD,
        FIFF.FIFFV_COIL_MAGNES_REF_GRAD,
        FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD,
        FIFF.FIFFV_COIL_KIT_GRAD,
        FIFF.FIFFV_COIL_BABY_GRAD,
        FIFF.FIFFV_COIL_ARTEMIS123_GRAD,
        FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD,
    ):
        # round coil 1st order (off-diagonal) gradiometer
        baseline = coil["base"] if id_ in (5004, 4005) else 0.0
        n_pts = 16  # number of points for circle
        # This time, go all the way around circle to close it fully
        circle = np.exp(2j * np.pi * np.arange(-1, n_pts) / float(n_pts - 1))
        circle[0] = 0  # center pt for triangulation
        circle *= coil["size"] / 2.0
        rrs = np.array(
            [  # first, second coil
                np.concatenate(
                    [circle.real + baseline / 2.0, circle.real - baseline / 2.0]
                ),
                np.concatenate([circle.imag, -circle.imag]),
            ]
        ).T
        tris = np.concatenate(
            [_make_tris_fan(n_pts + 1), _make_tris_fan(n_pts + 1) + n_pts + 1]
        )
    else:
        # 3D convex hull (will fail for 2D geometry)
        rrs = coil["rmag_orig"].copy()
        try:
            tris = _reorder_ccw(rrs, ConvexHull(rrs).simplices)
        except QhullError:  # 2D geometry likely
            logger.debug("Falling back to planar geometry")
            u, _, _ = np.linalg.svd(rrs.T, full_matrices=False)
            u[:, 2] = 0
            rr_rot = rrs @ u
            tris = Delaunay(rr_rot[:, :2]).simplices
            tris = np.concatenate((tris, tris[:, ::-1]))
        z_value = None

    # Go from (x,y) -> (x,y,z)
    if z_value is not None:
        rrs = np.pad(rrs, ((0, 0), (0, 1)), mode="constant", constant_values=z_value)
    assert rrs.ndim == 2 and rrs.shape[1] == 3
    return rrs, tris


def _process_clim(clim, colormap, transparent, data=0.0, allow_pos_lims=True):
    """Convert colormap/clim options to dict.

    This fills in any "auto" entries properly such that round-trip
    calling gives the same results.
    """
    # Based on type of limits specified, get cmap control points
    from matplotlib.colors import Colormap

    _validate_type(colormap, (str, Colormap), "colormap")
    data = np.asarray(data)
    if isinstance(colormap, str):
        if colormap == "auto":
            if clim == "auto":
                if allow_pos_lims and (data < 0).any():
                    colormap = "mne"
                else:
                    colormap = "hot"
            else:
                if "lims" in clim:
                    colormap = "hot"
                else:  # 'pos_lims' in clim
                    colormap = "mne"
        colormap = _get_cmap(colormap)
    assert isinstance(colormap, Colormap)
    diverging_maps = [
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    ]
    diverging_maps += [d + "_r" for d in diverging_maps]
    diverging_maps += ["mne", "mne_analyze"]
    if clim == "auto":
        # this is merely a heuristic!
        if allow_pos_lims and colormap.name in diverging_maps:
            key = "pos_lims"
        else:
            key = "lims"
        clim = {"kind": "percent", key: [96, 97.5, 99.95]}
    if not isinstance(clim, dict):
        raise ValueError(f'"clim" must be "auto" or dict, got {clim}')

    if ("lims" in clim) + ("pos_lims" in clim) != 1:
        raise ValueError(
            f"Exactly one of lims and pos_lims must be specified in clim, got {clim}"
        )
    if "pos_lims" in clim and not allow_pos_lims:
        raise ValueError('Cannot use "pos_lims" for clim, use "lims" instead')
    diverging = "pos_lims" in clim
    ctrl_pts = np.array(clim["pos_lims" if diverging else "lims"], float)
    ctrl_pts = np.array(ctrl_pts, float)
    if ctrl_pts.shape != (3,):
        raise ValueError(f"clim has shape {ctrl_pts.shape}, it must be (3,)")
    if (np.diff(ctrl_pts) < 0).any():
        raise ValueError(
            f"colormap limits must be monotonically increasing, got {ctrl_pts}"
        )
    clim_kind = clim.get("kind", "percent")
    _check_option("clim['kind']", clim_kind, ["value", "values", "percent"])
    if clim_kind == "percent":
        perc_data = np.abs(data) if diverging else data
        ctrl_pts = np.percentile(perc_data, ctrl_pts)
        logger.info(f"Using control points {ctrl_pts}")
    assert len(ctrl_pts) == 3
    clim = dict(kind="value")
    clim["pos_lims" if diverging else "lims"] = ctrl_pts
    mapdata = dict(clim=clim, colormap=colormap, transparent=transparent)
    return mapdata


def _separate_map(mapdata):
    """Help plotters that cannot handle limit equality."""
    diverging = "pos_lims" in mapdata["clim"]
    key = "pos_lims" if diverging else "lims"
    ctrl_pts = np.array(mapdata["clim"][key])
    assert ctrl_pts.shape == (3,)
    if len(set(ctrl_pts)) == 1:  # three points match
        if ctrl_pts[0] == 0:  # all are zero
            warn("All data were zero")
            ctrl_pts = np.arange(3, dtype=float)
        else:
            ctrl_pts *= [0.0, 0.5, 1]  # all nonzero pts == max
    elif len(set(ctrl_pts)) == 2:  # two points match
        # if points one and two are identical, add a tiny bit to the
        # control point two; if points two and three are identical,
        # subtract a tiny bit from point two.
        bump = 1e-5 if ctrl_pts[0] == ctrl_pts[1] else -1e-5
        ctrl_pts[1] = ctrl_pts[0] + bump * (ctrl_pts[2] - ctrl_pts[0])
    mapdata["clim"][key] = ctrl_pts


def _linearize_map(mapdata):
    from matplotlib.colors import ListedColormap

    diverging = "pos_lims" in mapdata["clim"]
    scale_pts = mapdata["clim"]["pos_lims" if diverging else "lims"]
    if diverging:
        lims = [-scale_pts[2], scale_pts[2]]
        ctrl_norm = (
            np.concatenate(
                [-scale_pts[::-1] / scale_pts[2], [0], scale_pts / scale_pts[2]]
            )
            / 2
            + 0.5
        )
        linear_norm = [0, 0.25, 0.5, 0.5, 0.5, 0.75, 1]
        trans_norm = [1, 1, 0, 0, 0, 1, 1]
    else:
        lims = [scale_pts[0], scale_pts[2]]
        range_ = scale_pts[2] - scale_pts[0]
        mid = (scale_pts[1] - scale_pts[0]) / range_ if range_ > 0 else 0.5
        ctrl_norm = [0, mid, 1]
        linear_norm = [0, 0.5, 1]
        trans_norm = [0, 1, 1]
    # do the piecewise linear transformation
    interp_to = np.linspace(0, 1, 256)
    colormap = np.array(
        mapdata["colormap"](np.interp(interp_to, ctrl_norm, linear_norm))
    )
    if mapdata["transparent"]:
        colormap[:, 3] = np.interp(interp_to, ctrl_norm, trans_norm)
    lims = np.array([lims[0], np.mean(lims), lims[1]])
    colormap = ListedColormap(colormap)
    return colormap, lims


def _get_map_ticks(mapdata):
    diverging = "pos_lims" in mapdata["clim"]
    ticks = mapdata["clim"]["pos_lims" if diverging else "lims"]
    delta = 1e-2 * (ticks[2] - ticks[0])
    if ticks[1] <= ticks[0] + delta:  # Only two worth showing
        ticks = ticks[::2]
    if ticks[1] <= ticks[0] + delta:  # Actually only one
        ticks = ticks[::2]
    if diverging:
        idx = int(ticks[0] == 0)
        ticks = list(-np.array(ticks[idx:])[::-1]) + [0] + list(ticks[idx:])
    return np.array(ticks)


def _handle_time(time_label, time_unit, times):
    """Handle time label string and units."""
    _validate_type(time_label, (None, str, "callable"), "time_label")
    if time_label == "auto":
        if times is not None and len(times) > 1:
            if time_unit == "s":
                time_label = "time=%0.3fs"
            elif time_unit == "ms":
                time_label = "time=%0.1fms"
        else:
            time_label = None
    # convert to callable
    if isinstance(time_label, str):
        time_label_fmt = time_label

        def time_label(x):
            try:
                return time_label_fmt % x
            except Exception:
                return time_label  # in case it's static

    assert time_label is None or callable(time_label)
    if times is not None:
        _, times = _check_time_unit(time_unit, times)
    return time_label, times


def _key_pressed_slider(event, params):
    """Handle key presses for time_viewer slider."""
    step = 1
    if event.key.startswith("ctrl"):
        step = 5
        event.key = event.key.split("+")[-1]
    if event.key not in ["left", "right"]:
        return
    time_viewer = event.canvas.figure
    value = time_viewer.slider.val
    times = params["stc"].times
    if params["time_unit"] == "ms":
        times = times * 1000.0
    time_idx = np.argmin(np.abs(times - value))
    if event.key == "left":
        time_idx = np.max((0, time_idx - step))
    elif event.key == "right":
        time_idx = np.min((len(times) - 1, time_idx + step))
    this_time = times[time_idx]
    time_viewer.slider.set_val(this_time)


def _smooth_plot(this_time, params, *, draw=True):
    """Smooth source estimate data and plot with mpl."""
    from ..morph import _hemi_morph

    ax = params["ax"]
    stc = params["stc"]
    ax.clear()
    times = stc.times
    scaler = 1000.0 if params["time_unit"] == "ms" else 1.0
    if this_time is None:
        time_idx = 0
    else:
        time_idx = np.argmin(np.abs(times - this_time / scaler))

    if params["hemi_idx"] == 0:
        data = stc.data[: len(stc.vertices[0]), time_idx : time_idx + 1]
    else:
        data = stc.data[len(stc.vertices[0]) :, time_idx : time_idx + 1]

    morph = _hemi_morph(
        params["tris"],
        params["inuse"],
        params["vertices"],
        params["smoothing_steps"],
        maps=None,
        warn=True,
    )
    array_plot = morph @ data

    range_ = params["scale_pts"][2] - params["scale_pts"][0]
    colors = (array_plot - params["scale_pts"][0]) / range_

    faces = params["faces"]
    greymap = params["greymap"]
    cmap = params["cmap"]
    polyc = ax.plot_trisurf(
        *params["coords"].T, triangles=faces, antialiased=False, vmin=0, vmax=1
    )
    color_ave = np.mean(colors[faces], axis=1).flatten()
    curv_ave = np.mean(params["curv"][faces], axis=1).flatten()
    colors = cmap(color_ave)
    # alpha blend
    colors[:, :3] *= colors[:, [3]]
    colors[:, :3] += greymap(curv_ave)[:, :3] * (1.0 - colors[:, [3]])
    colors[:, 3] = 1.0
    polyc.set_facecolor(colors)
    if params["time_label"] is not None:
        ax.set_title(
            params["time_label"](
                times[time_idx] * scaler,
            ),
            color="w",
        )
    _set_aspect_equal(ax)
    ax.axis("off")
    ax.set(xlim=[-80, 80], ylim=(-80, 80), zlim=[-80, 80])
    if draw:
        ax.figure.canvas.draw()


def _plot_mpl_stc(
    stc,
    subject=None,
    surface="inflated",
    hemi="lh",
    colormap="auto",
    time_label="auto",
    smoothing_steps=10,
    subjects_dir=None,
    views="lat",
    clim="auto",
    figure=None,
    initial_time=None,
    time_unit="s",
    background="black",
    spacing="oct6",
    time_viewer=False,
    colorbar=True,
    transparent=True,
):
    """Plot source estimate using mpl."""
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d import Axes3D

    from ..morph import _get_subject_sphere_tris
    from ..source_space._source_space import _check_spacing, _create_surf_spacing

    _check_option("hemi", hemi, ("lh", "rh"), extra="when using matplotlib")
    lh_kwargs = {
        "lat": {"elev": 0, "azim": 180},
        "med": {"elev": 0, "azim": 0},
        "ros": {"elev": 0, "azim": 90},
        "cau": {"elev": 0, "azim": -90},
        "dor": {"elev": 90, "azim": -90},
        "ven": {"elev": -90, "azim": -90},
        "fro": {"elev": 0, "azim": 106.739},
        "par": {"elev": 30, "azim": -120},
    }
    rh_kwargs = {
        "lat": {"elev": 0, "azim": 0},
        "med": {"elev": 0, "azim": 180},
        "ros": {"elev": 0, "azim": 90},
        "cau": {"elev": 0, "azim": -90},
        "dor": {"elev": 90, "azim": -90},
        "ven": {"elev": -90, "azim": -90},
        "fro": {"elev": 16.739, "azim": 60},
        "par": {"elev": 30, "azim": -60},
    }
    time_viewer = False if time_viewer == "auto" else time_viewer
    kwargs = dict(lh=lh_kwargs, rh=rh_kwargs)
    views = "lat" if views == "auto" else views
    _check_option("views", views, sorted(lh_kwargs.keys()))
    mapdata = _process_clim(clim, colormap, transparent, stc.data)
    _separate_map(mapdata)
    colormap, scale_pts = _linearize_map(mapdata)
    del transparent, mapdata

    time_label, times = _handle_time(time_label, time_unit, stc.times)
    # don't use constrained layout because Axes3D does not play well with it
    fig = plt.figure(figsize=(6, 6), layout=None) if figure is None else figure
    try:
        ax = Axes3D(fig, auto_add_to_figure=False)
    except Exception:  # old mpl
        ax = Axes3D(fig)
    else:
        fig.add_axes(ax)
    hemi_idx = 0 if hemi == "lh" else 1
    surf = subjects_dir / subject / "surf" / f"{hemi}.{surface}"
    if spacing == "all":
        coords, faces = nib.freesurfer.read_geometry(surf)
        inuse = slice(None)
    else:
        stype, sval, ico_surf, src_type_str = _check_spacing(spacing)
        surf = _create_surf_spacing(surf, hemi, subject, stype, ico_surf, subjects_dir)
        inuse = surf["vertno"]
        faces = surf["use_tris"]
        coords = surf["rr"][inuse]
        shape = faces.shape
        faces = rankdata(faces, "dense").reshape(shape) - 1
        faces = np.round(faces).astype(int)  # should really be int-like anyway
    del surf
    vertices = stc.vertices[hemi_idx]
    n_verts = len(vertices)
    tris = _get_subject_sphere_tris(subject, subjects_dir)[hemi_idx]
    cmap = _get_cmap(colormap)
    greymap = _get_cmap("Greys")

    curv = nib.freesurfer.read_morph_data(
        subjects_dir / subject / "surf" / f"{hemi}.curv"
    )[inuse]
    curv = np.clip(np.array(curv > 0, np.int64), 0.33, 0.66)
    params = dict(
        ax=ax,
        stc=stc,
        coords=coords,
        faces=faces,
        hemi_idx=hemi_idx,
        vertices=vertices,
        tris=tris,
        smoothing_steps=smoothing_steps,
        n_verts=n_verts,
        inuse=inuse,
        cmap=cmap,
        curv=curv,
        scale_pts=scale_pts,
        greymap=greymap,
        time_label=time_label,
        time_unit=time_unit,
    )
    _smooth_plot(initial_time, params, draw=False)

    ax.view_init(**kwargs[hemi][views])

    try:
        ax.set_facecolor(background)
    except AttributeError:
        ax.set_axis_bgcolor(background)

    if time_viewer:
        time_viewer = figure_nobar(figsize=(4.5, 0.25))
        fig.time_viewer = time_viewer
        ax_time = plt.axes()
        if initial_time is None:
            initial_time = 0
        slider = Slider(
            ax=ax_time,
            label="Time",
            valmin=times[0],
            valmax=times[-1],
            valinit=initial_time,
        )
        time_viewer.slider = slider
        callback_slider = partial(_smooth_plot, params=params)
        slider.on_changed(callback_slider)
        callback_key = partial(_key_pressed_slider, params=params)
        time_viewer.canvas.mpl_connect("key_press_event", callback_key)

    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)

    # add colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(scale_pts[0], scale_pts[2])
    )
    cax = inset_axes(ax, width="80%", height="5%", loc=8, borderpad=3.0)
    plt.setp(plt.getp(cax, "xticklabels"), color="w")
    sm.set_array(np.linspace(scale_pts[0], scale_pts[2], 256))
    if colorbar:
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb_yticks = plt.getp(cax, "yticklabels")
        plt.setp(cb_yticks, color="w")
        cax.tick_params(labelsize=16)
        cb.ax.set_facecolor("0.5")
        cax.set(xlim=(scale_pts[0], scale_pts[2]))
    plt_show(True)
    return fig


def link_brains(brains, time=True, camera=False, colorbar=True, picking=False):
    """Plot multiple SourceEstimate objects with PyVista.

    Parameters
    ----------
    brains : list, tuple or np.ndarray
        The collection of brains to plot.
    time : bool
        If True, link the time controllers. Defaults to True.
    camera : bool
        If True, link the camera controls. Defaults to False.
    colorbar : bool
        If True, link the colorbar controllers. Defaults to True.
    picking : bool
        If True, link the vertices picked with the mouse. Defaults to False.
    """
    from .backends.renderer import _get_3d_backend

    if _get_3d_backend() != "pyvistaqt":
        raise NotImplementedError(
            f"Expected 3d backend is pyvistaqt but {_get_3d_backend()} was given."
        )
    from ._brain import Brain, _LinkViewer

    if not isinstance(brains, Iterable):
        brains = [brains]
    if len(brains) == 0:
        raise ValueError("The collection of brains is empty.")
    for brain in brains:
        if not isinstance(brain, Brain):
            raise TypeError(f"Expected type is Brain but {type(brain)} was given.")
        # enable time viewer if necessary
        brain.setup_time_viewer()
    subjects = [brain._subject for brain in brains]
    if subjects.count(subjects[0]) != len(subjects):
        raise RuntimeError("Cannot link brains from different subjects.")

    # link brains properties
    _LinkViewer(
        brains=brains,
        time=time,
        camera=camera,
        colorbar=colorbar,
        picking=picking,
    )


def _check_volume(stc, src, surface, backend_name):
    from ..source_estimate import _BaseMixedSourceEstimate, _BaseSurfaceSourceEstimate
    from ..source_space import SourceSpaces

    if isinstance(stc, _BaseSurfaceSourceEstimate):
        return False
    else:
        _validate_type(
            src,
            SourceSpaces,
            "src",
            "src when stc is a mixed or volume source estimate",
        )
        if isinstance(stc, _BaseMixedSourceEstimate):
            # When showing subvolumes, surfaces that preserve geometry must
            # be used (i.e., no inflated)
            _check_option(
                "surface",
                surface,
                ("white", "pial"),
                extra="when plotting a mixed source estimate",
            )
        return True


@verbose
def plot_source_estimates(
    stc,
    subject=None,
    surface="inflated",
    hemi="lh",
    colormap="auto",
    time_label="auto",
    smoothing_steps=10,
    transparent=True,
    alpha=1.0,
    time_viewer="auto",
    subjects_dir=None,
    figure=None,
    views="auto",
    colorbar=True,
    clim="auto",
    cortex="classic",
    size=800,
    background="black",
    foreground=None,
    initial_time=None,
    time_unit="s",
    backend="auto",
    spacing="oct6",
    title=None,
    show_traces="auto",
    src=None,
    volume_options=1.0,
    view_layout="vertical",
    add_data_kwargs=None,
    brain_kwargs=None,
    verbose=None,
):
    """Plot SourceEstimate.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates to plot.
    %(subject_none)s
        If ``None``, ``stc.subject`` will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str
        Hemisphere id (ie ``'lh'``, ``'rh'``, ``'both'``, or ``'split'``). In
        the case of ``'both'``, both hemispheres are shown in the same window.
        In the case of ``'split'`` hemispheres are displayed side-by-side
        in different viewing panes.
    %(colormap)s
        The default ('auto') uses ``'hot'`` for one-sided data and
        'mne' for two-sided data.
    %(time_label)s
    smoothing_steps : int
        The amount of smoothing.
    %(transparent)s
    alpha : float
        Alpha value to apply globally to the overlay. Has no effect with mpl
        backend.
    time_viewer : bool | str
        Display time viewer GUI. Can also be 'auto', which will mean True
        for the PyVista backend and False otherwise.

        .. versionchanged:: 0.20.0
           "auto" mode added.
    %(subjects_dir)s
    figure : instance of Figure3D | instance of matplotlib.figure.Figure | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the PyVista
        figure by it's id or create a new figure with the given id. If an
        instance of matplotlib figure, mpl backend is used for plotting.
    %(views)s

        When plotting a standard SourceEstimate (not volume, mixed, or vector)
        and using the PyVista backend, ``views='flat'`` is also supported to
        plot cortex as a flatmap.

        Using multiple views (list) is not supported by the matplotlib backend.

        .. versionchanged:: 0.21.0
           Support for flatmaps.
    colorbar : bool
        If True, display colorbar on scene.
    %(clim)s
    cortex : str | tuple
        Specifies how binarized curvature values are rendered.
        Either the name of a preset Brain cortex colorscheme (one of
        ``'classic'``, ``'bone'``, ``'low_contrast'``, or ``'high_contrast'``),
        or the name of a colormap, or a tuple with values
        ``(colormap, min, max, reverse)`` to fully specify the curvature
        colors. Has no effect with the matplotlib backend.
    size : float or tuple of float
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
        Has no effect with mpl backend.
    background : matplotlib color
        Color of the background of the display window.
    foreground : matplotlib color | None
        Color of the foreground of the display window. Has no effect with mpl
        backend. None will choose white or black based on the background color.
    initial_time : float | None
        The time to display on the plot initially. ``None`` to display the
        first time sample (default).
    time_unit : ``'s'`` | ``'ms'``
        Whether time is represented in seconds ("s", default) or
        milliseconds ("ms").
    backend : ``'auto'`` | ``'pyvistaqt'`` | ``'matplotlib'``
        Which backend to use. If ``'auto'`` (default), tries to plot with
        pyvistaqt, but resorts to matplotlib if no 3d backend is available.

        .. versionadded:: 0.15.0
    spacing : str
        Only affects the matplotlib backend.
        The spacing to use for the source space. Can be ``'ico#'`` for a
        recursively subdivided icosahedron, ``'oct#'`` for a recursively
        subdivided octahedron, or ``'all'`` for all points. In general, you can
        speed up the plotting by selecting a sparser source space.
        Defaults  to 'oct6'.

        .. versionadded:: 0.15.0
    title : str | None
        Title for the figure. If None, the subject name will be used.

        .. versionadded:: 0.17.0
    %(show_traces)s
    %(src_volume_options)s
    %(view_layout)s
    %(add_data_kwargs)s
    %(brain_kwargs)s
    %(verbose)s

    Returns
    -------
    figure : instance of mne.viz.Brain | matplotlib.figure.Figure
        An instance of :class:`mne.viz.Brain` or matplotlib figure.

    Notes
    -----
    Flatmaps are available by default for ``fsaverage`` but not for other
    subjects reconstructed by FreeSurfer. We recommend using
    :func:`mne.compute_source_morph` to morph source estimates to ``fsaverage``
    for flatmap plotting. If you want to construct your own flatmap for a given
    subject, these links might help:

    - https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOccipitalFlattenedPatch
    - https://openwetware.org/wiki/Beauchamp:FreeSurfer
    """  # noqa: E501
    from ..source_estimate import _BaseSourceEstimate, _check_stc_src
    from .backends.renderer import _get_3d_backend, use_3d_backend

    _check_stc_src(stc, src)
    _validate_type(stc, _BaseSourceEstimate, "stc", "source estimate")
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir, raise_error=True)
    subject = _check_subject(stc.subject, subject)
    _check_option("backend", backend, ["auto", "matplotlib", "pyvistaqt", "notebook"])
    plot_mpl = backend == "matplotlib"
    if not plot_mpl:
        if backend == "auto":
            try:
                backend = _get_3d_backend()
            except (ImportError, ModuleNotFoundError):
                warn("No 3D backend found. Resorting to matplotlib 3d.")
                plot_mpl = True
    kwargs = dict(
        subject=subject,
        surface=surface,
        hemi=hemi,
        colormap=colormap,
        time_label=time_label,
        smoothing_steps=smoothing_steps,
        subjects_dir=subjects_dir,
        views=views,
        clim=clim,
        figure=figure,
        initial_time=initial_time,
        time_unit=time_unit,
        background=background,
        time_viewer=time_viewer,
        colorbar=colorbar,
        transparent=transparent,
    )
    if plot_mpl:
        return _plot_mpl_stc(stc, spacing=spacing, **kwargs)
    else:
        with use_3d_backend(backend):
            return _plot_stc(
                stc,
                overlay_alpha=alpha,
                brain_alpha=alpha,
                vector_alpha=alpha,
                cortex=cortex,
                foreground=foreground,
                size=size,
                scale_factor=None,
                show_traces=show_traces,
                src=src,
                volume_options=volume_options,
                view_layout=view_layout,
                add_data_kwargs=add_data_kwargs,
                brain_kwargs=brain_kwargs,
                **kwargs,
            )


def _plot_stc(
    stc,
    subject,
    surface,
    hemi,
    colormap,
    time_label,
    smoothing_steps,
    subjects_dir,
    views,
    clim,
    figure,
    initial_time,
    time_unit,
    background,
    time_viewer,
    colorbar,
    transparent,
    brain_alpha,
    overlay_alpha,
    vector_alpha,
    cortex,
    foreground,
    size,
    scale_factor,
    show_traces,
    src,
    volume_options,
    view_layout,
    add_data_kwargs,
    brain_kwargs,
):
    from ..source_estimate import _BaseVolSourceEstimate
    from .backends.renderer import _get_3d_backend, get_brain_class

    vec = stc._data_ndim == 3
    subjects_dir = str(get_subjects_dir(subjects_dir=subjects_dir, raise_error=True))
    subject = _check_subject(stc.subject, subject)

    backend = _get_3d_backend()
    del _get_3d_backend
    Brain = get_brain_class()
    views = _check_views(surface, views, hemi, stc, backend)
    _check_option("hemi", hemi, ["lh", "rh", "split", "both"])
    _check_option("view_layout", view_layout, ("vertical", "horizontal"))
    time_label, times = _handle_time(time_label, time_unit, stc.times)
    show_traces, time_viewer = _check_st_tv(show_traces, time_viewer, times)

    # convert control points to locations in colormap
    use = stc.magnitude().data if vec else stc.data
    mapdata = _process_clim(clim, colormap, transparent, use, allow_pos_lims=not vec)

    volume = _check_volume(stc, src, surface, backend)

    # XXX we should not need to do this for PyVista, the plotter should be
    # smart enough to do this separation in the cmap-to-ctab conversion
    _separate_map(mapdata)
    colormap = mapdata["colormap"]
    diverging = "pos_lims" in mapdata["clim"]
    scale_pts = mapdata["clim"]["pos_lims" if diverging else "lims"]
    transparent = mapdata["transparent"]
    del mapdata

    if hemi in ["both", "split"]:
        hemis = ["lh", "rh"]
    else:
        hemis = [hemi]

    if overlay_alpha is None:
        overlay_alpha = brain_alpha
    if overlay_alpha == 0:
        smoothing_steps = 1  # Disable smoothing to save time.

    title = subject if len(hemis) > 1 else f"{subject} - {hemis[0]}"
    kwargs = {
        "subject": subject,
        "hemi": hemi,
        "surf": surface,
        "title": title,
        "cortex": cortex,
        "size": size,
        "background": background,
        "foreground": foreground,
        "figure": figure,
        "subjects_dir": subjects_dir,
        "views": views,
        "alpha": brain_alpha,
    }
    if brain_kwargs is not None:
        kwargs.update(brain_kwargs)
    kwargs["show"] = False
    kwargs["view_layout"] = view_layout
    with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(**kwargs)
    del kwargs

    if scale_factor is None:
        # Configure the glyphs scale directly
        width = np.mean(
            [
                np.ptp(brain.geo[hemi].coords[:, 1])
                for hemi in hemis
                if hemi in brain.geo
            ]
        )
        scale_factor = 0.025 * width / scale_pts[-1]

    if transparent is None:
        transparent = True
    center = 0.0 if diverging else None
    kwargs = {
        "array": stc,
        "colormap": colormap,
        "smoothing_steps": smoothing_steps,
        "time": times,
        "time_label": time_label,
        "alpha": overlay_alpha,
        "colorbar": colorbar,
        "vector_alpha": vector_alpha,
        "scale_factor": scale_factor,
        "initial_time": initial_time,
        "transparent": transparent,
        "center": center,
        "fmin": scale_pts[0],
        "fmid": scale_pts[1],
        "fmax": scale_pts[2],
        "clim": clim,
        "src": src,
        "volume_options": volume_options,
        "verbose": None,
    }
    if add_data_kwargs is not None:
        kwargs.update(add_data_kwargs)
    for hemi in hemis:
        if isinstance(stc, _BaseVolSourceEstimate):  # no surf data
            break
        vertices = stc.vertices[0 if hemi == "lh" else 1]
        if len(vertices) == 0:  # no surf data for the given hemi
            continue  # no data
        use_kwargs = kwargs.copy()
        use_kwargs.update(hemi=hemi)
        with warnings.catch_warnings(record=True):  # traits warnings
            brain.add_data(**use_kwargs)

    if volume:
        use_kwargs = kwargs.copy()
        use_kwargs.update(hemi="vol")
        brain.add_data(**use_kwargs)
    del kwargs

    if time_viewer:
        brain.setup_time_viewer(time_viewer=time_viewer, show_traces=show_traces)
    else:
        brain.show()

    return brain


def _check_st_tv(show_traces, time_viewer, times):
    # time_viewer and show_traces
    _check_option("time_viewer", time_viewer, (True, False, "auto"))
    _validate_type(show_traces, (str, bool, "numeric"), "show_traces")
    if isinstance(show_traces, str):
        _check_option(
            "show_traces",
            show_traces,
            ("auto", "separate", "vertex", "label"),
            extra="when a string",
        )
    if time_viewer == "auto":
        time_viewer = True
    if show_traces == "auto":
        show_traces = time_viewer and times is not None and len(times) > 1
    if show_traces and not time_viewer:
        raise ValueError("show_traces cannot be used when time_viewer=False")
    return show_traces, time_viewer


def _glass_brain_crosshairs(params, x, y, z):
    for ax, a, b in (
        (params["ax_y"], x, z),
        (params["ax_x"], y, z),
        (params["ax_z"], x, y),
    ):
        ax.axvline(a, color="0.75")
        ax.axhline(b, color="0.75")


def _cut_coords_to_ijk(cut_coords, img):
    ijk = apply_trans(np.linalg.inv(img.affine), cut_coords)
    ijk = np.round(ijk).astype(int)
    logger.debug(f"{cut_coords} -> {ijk}")
    np.clip(ijk, 0, np.array(img.shape[:3]) - 1, out=ijk)
    return ijk


def _ijk_to_cut_coords(ijk, img):
    return apply_trans(img.affine, ijk)


def _load_subject_mri(mri, stc, subject, subjects_dir, name):
    import nibabel as nib
    from nibabel.spatialimages import SpatialImage

    _validate_type(mri, ("path-like", SpatialImage), name)
    if isinstance(mri, str):
        subject = _check_subject(stc.subject, subject)
        mri = nib.load(_check_mri(mri, subject, subjects_dir))
    return mri


_AX_NAME = dict(x="X (sagittal)", y="Y (coronal)", z="Z (axial)")


def _click_to_cut_coords(event, params):
    """Get voxel coordinates from mouse click."""
    import nibabel as nib

    if event.inaxes is params["ax_x"]:
        ax = "x"
        x = params["ax_z"].lines[0].get_xdata()[0]
        y, z = event.xdata, event.ydata
    elif event.inaxes is params["ax_y"]:
        ax = "y"
        y = params["ax_x"].lines[0].get_xdata()[0]
        x, z = event.xdata, event.ydata
    elif event.inaxes is params["ax_z"]:
        ax = "z"
        x, y = event.xdata, event.ydata
        z = params["ax_x"].lines[1].get_ydata()[0]
    else:
        logger.debug("    Click outside axes")
        return None
    cut_coords = np.array((x, y, z))
    logger.debug("")

    if params["mode"] == "glass_brain":  # find idx for MIP
        # Figure out what XYZ in world coordinates is in our voxel data
        codes = "".join(nib.aff2axcodes(params["img_idx"].affine))
        assert len(codes) == 3
        # We don't care about directionality, just which is which dim
        codes = codes.replace("L", "R").replace("P", "A").replace("I", "S")
        idx = codes.index(dict(x="R", y="A", z="S")[ax])
        img_data = np.abs(_get_img_fdata(params["img_idx"]))
        ijk = _cut_coords_to_ijk(cut_coords, params["img_idx"])
        if idx == 0:
            ijk[0] = np.argmax(img_data[:, ijk[1], ijk[2]])
            logger.debug(f"    MIP: i = {ijk[0]:d} idx")
        elif idx == 1:
            ijk[1] = np.argmax(img_data[ijk[0], :, ijk[2]])
            logger.debug(f"    MIP: j = {ijk[1]:d} idx")
        else:
            ijk[2] = np.argmax(img_data[ijk[0], ijk[1], :])
            logger.debug(f"    MIP: k = {ijk[2]} idx")
        cut_coords = _ijk_to_cut_coords(ijk, params["img_idx"])

    logger.debug(f"    Cut coords for {_AX_NAME[ax]}: {_str_ras(cut_coords)}")
    return cut_coords


def _str_ras(xyz):
    x, y, z = xyz
    return f"({x:0.1f}, {y:0.1f}, {z:0.1f}) mm"


def _str_vox(ijk):
    i, j, k = ijk
    return f"[{i:d}, {j:d}, {k:d}] vox"


def _press(event, params):
    """Manage keypress on the plot."""
    pos = params["lx"].get_xdata()
    idx = params["stc"].time_as_index(pos)[0]
    if event.key == "left":
        idx = max(0, idx - 2)
    elif event.key == "shift+left":
        idx = max(0, idx - 10)
    elif event.key == "right":
        idx = min(params["stc"].shape[1] - 1, idx + 2)
    elif event.key == "shift+right":
        idx = min(params["stc"].shape[1] - 1, idx + 10)
    _update_timeslice(idx, params)
    params["fig"].canvas.draw()


def _update_timeslice(idx, params):
    from nilearn.image import index_img

    params["lx"].set_xdata([idx / params["stc"].sfreq + params["stc"].tmin])
    ax_x, ax_y, ax_z = params["ax_x"], params["ax_y"], params["ax_z"]
    # Crosshairs are the first thing plotted in stat_map, and the last
    # in glass_brain
    idxs = [0, 0, 1] if params["mode"] == "stat_map" else [-2, -2, -1]
    cut_coords = (
        ax_y.lines[idxs[0]].get_xdata()[0],
        ax_x.lines[idxs[1]].get_xdata()[0],
        ax_x.lines[idxs[2]].get_ydata()[0],
    )
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()
    params.update({"img_idx": index_img(params["img"], idx)})
    params.update({"title": f"Activation (t={params['stc'].times[idx]:.3f} s.)"})
    _plot_and_correct(params=params, cut_coords=cut_coords)


def _update_vertlabel(loc_idx, params):
    params["vert_legend"].get_texts()[0].set_text(f"{params['vertices'][loc_idx]}")


@verbose_dec
def _onclick(event, params, verbose=None):
    """Manage clicks on the plot."""
    ax_x, ax_y, ax_z = params["ax_x"], params["ax_y"], params["ax_z"]
    if event.inaxes is params["ax_time"]:
        idx = params["stc"].time_as_index(event.xdata, use_rounding=True)[0]
        _update_timeslice(idx, params)

    cut_coords = _click_to_cut_coords(event, params)
    if cut_coords is None:
        return  # not in any axes

    ax_x.clear()
    ax_y.clear()
    ax_z.clear()
    _plot_and_correct(params=params, cut_coords=cut_coords)
    loc_idx = _cut_coords_to_idx(cut_coords, params["dist_to_verts"])
    ydata = params["stc"].data[loc_idx]
    if loc_idx is not None:
        params["ax_time"].lines[0].set_ydata(ydata)
    else:
        params["ax_time"].lines[0].set_ydata([0.0])
    _update_vertlabel(loc_idx, params)
    params["fig"].canvas.draw()


def _cut_coords_to_idx(cut_coords, dist_to_verts):
    """Convert voxel coordinates to index in stc.data."""
    logger.debug(f"    Starting coords: {cut_coords}")
    cut_coords = list(cut_coords)
    (dist,), (loc_idx,) = dist_to_verts.query([cut_coords])
    logger.debug(f"Mapped {cut_coords=} to vertices[{loc_idx}] {dist:0.1f} mm away")
    return loc_idx


def _plot_and_correct(*, params, cut_coords):
    # black_bg = True is needed because of some matplotlib
    # peculiarity. See: https://stackoverflow.com/a/34730204
    # Otherwise, event.inaxes does not work for ax_x and ax_z
    from nilearn.plotting import plot_glass_brain, plot_stat_map

    mode = params["mode"]
    nil_func = dict(stat_map=plot_stat_map, glass_brain=plot_glass_brain)[mode]
    plot_kwargs = dict(
        threshold=None,
        axes=params["axes"],
        resampling_interpolation="nearest",
        vmax=params["vmax"],
        figure=params["fig"],
        colorbar=params["colorbar"],
        bg_img=params["bg_img"],
        cmap=params["colormap"],
        black_bg=True,
        symmetric_cbar=True,
        title="",
    )
    params["axes"].clear()
    if params.get("fig_anat") is not None and plot_kwargs["colorbar"]:
        params["fig_anat"]._cbar.ax.clear()
    with warnings.catch_warnings(record=True):  # nilearn bug; ax recreated
        warnings.simplefilter("ignore", DeprecationWarning)
        params["fig_anat"] = nil_func(
            params["img_idx"], cut_coords=cut_coords, **plot_kwargs
        )
    params["fig_anat"]._cbar.outline.set_visible(False)
    for key in "xyz":
        params.update({"ax_" + key: params["fig_anat"].axes[key].ax})
    # Fix nilearn bug w/cbar background being white
    if plot_kwargs["colorbar"]:
        params["fig_anat"]._cbar.ax.set_facecolor("0.5")
        # adjust one-sided colorbars
        if not params["diverging"]:
            _crop_colorbar(params["fig_anat"]._cbar, *params["scale_pts"][[0, -1]])
        params["fig_anat"]._cbar.set_ticks(params["cbar_ticks"])
    if params["mode"] == "glass_brain":
        _glass_brain_crosshairs(params, *cut_coords)


@verbose
def plot_volume_source_estimates(
    stc,
    src,
    subject=None,
    subjects_dir=None,
    mode="stat_map",
    bg_img="T1.mgz",
    colorbar=True,
    colormap="auto",
    clim="auto",
    transparent=None,
    show=True,
    initial_time=None,
    initial_pos=None,
    verbose=None,
):
    """Plot Nutmeg style volumetric source estimates using nilearn.

    Parameters
    ----------
    stc : VectorSourceEstimate
        The vector source estimate to plot.
    src : instance of SourceSpaces | instance of SourceMorph
        The source space. Can also be a SourceMorph to morph the STC to
        a new subject (see Examples).

        .. versionchanged:: 0.18
           Support for :class:`~nibabel.spatialimages.SpatialImage`.
    %(subject_none)s
        If ``None``, ``stc.subject`` will be used.
    %(subjects_dir)s
    mode : ``'stat_map'`` | ``'glass_brain'``
        The plotting mode to use. For ``'glass_brain'``, activation absolute values are
        displayed after being transformed to a standard MNI brain.
    bg_img : instance of SpatialImage | str
        The background image used in the nilearn plotting function.
        Can also be a string to use the ``bg_img`` file in the subject's
        MRI directory (default is ``'T1.mgz'``).
        Not used in "glass brain" plotting.
    colorbar : bool
        If True, display a colorbar on the right of the plots.
    %(colormap)s
    %(clim)s
    %(transparent)s
    show : bool
        Show figures if True. Defaults to True.
    initial_time : float | None
        The initial time to plot. Can be None (default) to use the time point
        with the maximal absolute value activation across all voxels
        or the ``initial_pos`` voxel (if ``initial_pos is None`` or not,
        respectively).

        .. versionadded:: 0.19
    initial_pos : ndarray, shape (3,) | None
        The initial position to use (in m). Can be None (default) to use the
        voxel with the maximum absolute value activation across all time points
        or at ``initial_time`` (if ``initial_time is None`` or not,
        respectively).

        .. versionadded:: 0.19
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        The figure.

    Notes
    -----
    Click on any of the anatomical slices to explore the time series.
    Clicking on any time point will bring up the corresponding anatomical map.

    The left and right arrow keys can be used to navigate in time.
    To move in time by larger steps, use shift+left and shift+right.

    In ``'glass_brain'`` mode, values are transformed to the standard MNI
    brain using the FreeSurfer Talairach transformation
    ``$SUBJECTS_DIR/$SUBJECT/mri/transforms/talairach.xfm``.

    .. versionadded:: 0.17

    .. versionchanged:: 0.19
       MRI volumes are automatically transformed to MNI space in
       ``'glass_brain'`` mode.

    Examples
    --------
    Passing a :class:`mne.SourceMorph` as the ``src``
    parameter can be useful for plotting in a different subject's space
    (here, a ``'sample'`` STC in ``'fsaverage'``'s space)::

    >>> morph = mne.compute_source_morph(src_sample, subject_to='fsaverage')  # doctest: +SKIP
    >>> fig = stc_vol_sample.plot(morph)  # doctest: +SKIP
    """  # noqa: E501
    import nibabel as nib
    from matplotlib import colors
    from matplotlib import pyplot as plt

    from ..morph import SourceMorph
    from ..source_estimate import VolSourceEstimate
    from ..source_space._source_space import _ensure_src

    if not check_version("nilearn", "0.4"):
        raise RuntimeError("This function requires nilearn >= 0.4")

    from nilearn.image import index_img

    _check_option("mode", mode, ("stat_map", "glass_brain"))
    _validate_type(stc, VolSourceEstimate, "stc")
    if isinstance(src, SourceMorph):
        img = src.apply(stc, "nifti1", mri_resolution=False, mri_space=False)
        stc = src.apply(stc, mri_resolution=False, mri_space=False)
        kind, src_subject = "morph.subject_to", src.subject_to
    else:
        src = _ensure_src(src, kind="volume", extra=" or SourceMorph")
        img = stc.as_volume(src, mri_resolution=False)
        kind, src_subject = "src subject", src._subject
    del src
    _print_coord_trans(
        Transform("mri_voxel", "ras", img.affine),
        prefix="Image affine ",
        units="mm",
        level="debug",
    )
    subject = _check_subject(src_subject, subject, first_kind=kind)

    if mode == "glass_brain":
        subject = _check_subject(stc.subject, subject)
        ras_mni_t = read_ras_mni_t(subject, subjects_dir)
        if not np.allclose(ras_mni_t["trans"], np.eye(4)):
            _print_coord_trans(ras_mni_t, prefix="Transforming subject ", units="mm")
            logger.info("")
            # To get from voxel coords to world coords (i.e., define affine)
            # we would apply img.affine, then also apply ras_mni_t, which
            # transforms from the subject's RAS to MNI RAS. So we left-multiply
            # these.
            img = nib.Nifti1Image(img.dataobj, np.dot(ras_mni_t["trans"], img.affine))
        bg_img = None  # not used
    else:  # stat_map
        if bg_img is None:
            bg_img = "T1.mgz"
        bg_img = _load_subject_mri(bg_img, stc, subject, subjects_dir, "bg_img")

    params = dict(
        stc=stc,
        mode=mode,
        img=img,
        bg_img=bg_img,
        colorbar=colorbar,
    )
    vertices = np.hstack(stc.vertices)
    stc_ijk = np.array(np.unravel_index(vertices, img.shape[:3], order="F")).T
    assert stc_ijk.shape == (vertices.size, 3)
    params["dist_to_verts"] = _DistanceQuery(apply_trans(img.affine, stc_ijk))
    params["vertices"] = vertices
    del kind, stc_ijk

    if initial_time is None:
        time_sl = slice(0, None)
    else:
        initial_time = float(initial_time)
        logger.info(f"Fixing initial time: {initial_time} s")
        initial_time = np.argmin(np.abs(stc.times - initial_time))
        time_sl = slice(initial_time, initial_time + 1)
    if initial_pos is None:  # find max pos and (maybe) time
        loc_idx, time_idx = np.unravel_index(
            np.abs(stc.data[:, time_sl]).argmax(), stc.data[:, time_sl].shape
        )
        time_idx += time_sl.start
    else:  # position specified
        initial_pos = np.array(initial_pos, float)
        if initial_pos.shape != (3,):
            raise ValueError(
                "initial_pos must be float ndarray with shape "
                f"(3,), got shape {initial_pos.shape}"
            )
        initial_pos *= 1000
        logger.info(f"Fixing initial position: {initial_pos.tolist()} mm")
        loc_idx = _cut_coords_to_idx(initial_pos, params["dist_to_verts"])
        if initial_time is not None:  # time also specified
            time_idx = time_sl.start
        else:  # find the max
            time_idx = np.argmax(np.abs(stc.data[loc_idx]))
    img_idx = params["img_idx"] = index_img(img, time_idx)
    assert img_idx.shape == img.shape[:3]
    del initial_time, initial_pos
    ijk = np.unravel_index(vertices[loc_idx], img.shape[:3], order="F")
    cut_coords = _ijk_to_cut_coords(ijk, img_idx)
    np.testing.assert_allclose(_cut_coords_to_ijk(cut_coords, img_idx), ijk)
    logger.info(
        f"Showing: t = {stc.times[time_idx]:0.3f} s, "
        f"{_str_ras(cut_coords)}, "
        f"{_str_vox(ijk)}, "
        f"{vertices[loc_idx]:d} vertex"
    )
    del ijk

    # Plot initial figure
    fig, (axes, ax_time) = plt.subplots(2, layout="constrained")
    axes.set(xticks=[], yticks=[])
    marker = "o" if len(stc.times) == 1 else None
    ydata = stc.data[loc_idx]
    h = ax_time.plot(stc.times, ydata, color="k", marker=marker)[0]
    if len(stc.times) > 1:
        ax_time.set(xlim=stc.times[[0, -1]])
    ax_time.set(xlabel="Time (s)", ylabel="Activation")
    params["vert_legend"] = ax_time.legend([h], [""], title="Vertex")
    _update_vertlabel(loc_idx, params)
    lx = ax_time.axvline(stc.times[time_idx], color="g")
    params.update(fig=fig, ax_time=ax_time, lx=lx, axes=axes)

    allow_pos_lims = mode != "glass_brain"
    mapdata = _process_clim(clim, colormap, transparent, stc.data, allow_pos_lims)
    _separate_map(mapdata)
    diverging = "pos_lims" in mapdata["clim"]
    ticks = _get_map_ticks(mapdata)
    params.update(cbar_ticks=ticks, diverging=diverging)
    colormap, scale_pts = _linearize_map(mapdata)
    del mapdata

    ylim = [min((scale_pts[0], ydata.min())), max((scale_pts[-1], ydata.max()))]
    ylim = np.array(ylim) + np.array([-1, 1]) * 0.05 * np.diff(ylim)[0]
    dup_neg = False
    if stc.data.min() < 0:
        ax_time.axhline(0.0, color="0.5", ls="-", lw=0.5, zorder=2)
        dup_neg = not diverging  # glass brain with signed data
    yticks = list(ticks)
    if dup_neg:
        yticks += [0] + list(-np.array(ticks))
    yticks = np.unique(yticks)
    ax_time.set(yticks=yticks)
    ax_time.set(ylim=ylim)
    del yticks

    if not diverging:  # set eq above iff one-sided
        # there is a bug in nilearn where this messes w/transparency
        # Need to double the colormap
        if (scale_pts < 0).any():
            # XXX We should fix this, but it's hard to get nilearn to
            # use arbitrary bounds :(
            # Should get them to support non-mirrored colorbars, or
            # at least a proper `vmin` for one-sided things.
            # Hopefully this is a sufficiently rare use case!
            raise ValueError(
                "Negative colormap limits for sequential "
                'control points clim["lims"] not supported '
                "currently, consider shifting or flipping the "
                "sign of your data for visualization purposes"
            )
        # due to nilearn plotting weirdness, extend this to go
        # -scale_pts[2]->scale_pts[2] instead of scale_pts[0]->scale_pts[2]
        colormap = _get_cmap(colormap)
        colormap = colormap(
            np.interp(np.linspace(-1, 1, 256), scale_pts / scale_pts[2], [0, 0.5, 1])
        )
        colormap = colors.ListedColormap(colormap)
    params.update(vmax=scale_pts[-1], scale_pts=scale_pts, colormap=colormap)

    _plot_and_correct(params=params, cut_coords=cut_coords)

    plt_show(show)
    fig.canvas.mpl_connect(
        "button_press_event", partial(_onclick, params=params, verbose=verbose)
    )
    fig.canvas.mpl_connect("key_press_event", partial(_press, params=params))

    return fig


def _check_views(surf, views, hemi, stc=None, backend=None):
    from ..source_estimate import SourceEstimate
    from ._brain.view import views_dicts

    _validate_type(views, (list, tuple, str), "views")
    views = [views] if isinstance(views, str) else list(views)
    if surf == "flat":
        _check_option("views", views, (["auto"], ["flat"]))
        views = ["flat"]
    elif len(views) == 1 and views[0] == "auto":
        views = ["lateral"]
    if views == ["flat"]:
        if stc is not None:
            _validate_type(
                stc, SourceEstimate, "stc", "SourceEstimate when a flatmap is used"
            )
        if backend is not None:
            if backend not in ("pyvistaqt", "notebook"):
                raise RuntimeError(
                    "The PyVista 3D backend must be used to plot a flatmap"
                )
    if (views == ["flat"]) ^ (surf == "flat"):  # exactly only one of the two
        raise ValueError(
            'surface="flat" must be used with views="flat", got '
            f"surface={repr(surf)} and views={repr(views)}"
        )
    _check_option("hemi", hemi, ("split", "both", "lh", "rh", "vol", None))
    use_hemi = "lh" if hemi == "split" or hemi is None else hemi
    for vi, v in enumerate(views):
        _check_option(f"views[{vi}]", v, sorted(views_dicts[use_hemi]))
    return views


@verbose
def plot_vector_source_estimates(
    stc,
    subject=None,
    hemi="lh",
    colormap="hot",
    time_label="auto",
    smoothing_steps=10,
    transparent=None,
    brain_alpha=0.4,
    overlay_alpha=None,
    vector_alpha=1.0,
    scale_factor=None,
    time_viewer="auto",
    subjects_dir=None,
    figure=None,
    views="lateral",
    colorbar=True,
    clim="auto",
    cortex="classic",
    size=800,
    background="black",
    foreground=None,
    initial_time=None,
    time_unit="s",
    show_traces="auto",
    src=None,
    volume_options=1.0,
    view_layout="vertical",
    add_data_kwargs=None,
    brain_kwargs=None,
    verbose=None,
):
    """Plot VectorSourceEstimate with PyVista.

    A "glass brain" is drawn and all dipoles defined in the source estimate
    are shown using arrows, depicting the direction and magnitude of the
    current moment at the dipole. Additionally, an overlay is plotted on top of
    the cortex with the magnitude of the current.

    Parameters
    ----------
    stc : VectorSourceEstimate | MixedVectorSourceEstimate
        The vector source estimate to plot.
    %(subject_none)s
        If ``None``, ``stc.subject`` will be used.
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display.
    %(colormap)s
        This should be a sequential colormap.
    %(time_label)s
    smoothing_steps : int
        The amount of smoothing.
    %(transparent)s
    brain_alpha : float
        Alpha value to apply globally to the surface meshes. Defaults to 0.4.
    overlay_alpha : float
        Alpha value to apply globally to the overlay. Defaults to
        ``brain_alpha``.
    vector_alpha : float
        Alpha value to apply globally to the vector glyphs. Defaults to 1.
    scale_factor : float | None
        Scaling factor for the vector glyphs. By default, an attempt is made to
        automatically determine a sane value.
    time_viewer : bool | str
        Display time viewer GUI. Can be "auto", which is True for the PyVista
        backend and False otherwise.

        .. versionchanged:: 0.20
           Added "auto" option and default.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    figure : instance of Figure3D | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the PyVista
        figure by it's id or create a new figure with the given id.
    %(views)s
    colorbar : bool
        If True, display colorbar on scene.
    %(clim_onesided)s
    cortex : str or tuple
        Specifies how binarized curvature values are rendered.
        either the name of a preset Brain cortex colorscheme (one of
        'classic', 'bone', 'low_contrast', or 'high_contrast'), or the
        name of a colormap, or a tuple with values (colormap, min,
        max, reverse) to fully specify the curvature colors.
    size : float or tuple of float
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : matplotlib color
        Color of the background of the display window.
    foreground : matplotlib color | None
        Color of the foreground of the display window.
        None will choose black or white based on the background color.
    initial_time : float | None
        The time to display on the plot initially. ``None`` to display the
        first time sample (default).
    time_unit : 's' | 'ms'
        Whether time is represented in seconds ("s", default) or
        milliseconds ("ms").
    %(show_traces)s
    %(src_volume_options)s
    %(view_layout)s
    %(add_data_kwargs)s
    %(brain_kwargs)s
    %(verbose)s

    Returns
    -------
    brain : mne.viz.Brain
        A instance of :class:`mne.viz.Brain`.

    Notes
    -----
    .. versionadded:: 0.15

    If the current magnitude overlay is not desired, set ``overlay_alpha=0``
    and ``smoothing_steps=1``.
    """
    from ..source_estimate import _BaseVectorSourceEstimate

    _validate_type(stc, _BaseVectorSourceEstimate, "stc", "vector source estimate")
    return _plot_stc(
        stc,
        subject=subject,
        surface="white",
        hemi=hemi,
        colormap=colormap,
        time_label=time_label,
        smoothing_steps=smoothing_steps,
        subjects_dir=subjects_dir,
        views=views,
        clim=clim,
        figure=figure,
        initial_time=initial_time,
        time_unit=time_unit,
        background=background,
        time_viewer=time_viewer,
        colorbar=colorbar,
        transparent=transparent,
        brain_alpha=brain_alpha,
        overlay_alpha=overlay_alpha,
        vector_alpha=vector_alpha,
        cortex=cortex,
        foreground=foreground,
        size=size,
        scale_factor=scale_factor,
        show_traces=show_traces,
        src=src,
        volume_options=volume_options,
        view_layout=view_layout,
        add_data_kwargs=add_data_kwargs,
        brain_kwargs=brain_kwargs,
    )


@verbose
def plot_sparse_source_estimates(
    src,
    stcs,
    colors=None,
    linewidth=2,
    fontsize=18,
    bgcolor=(0.05, 0, 0.1),
    opacity=0.2,
    brain_color=(0.7,) * 3,
    show=True,
    high_resolution=False,
    fig_name=None,
    fig_number=None,
    labels=None,
    modes=("cone", "sphere"),
    scale_factors=(1, 0.6),
    verbose=None,
    **kwargs,
):
    """Plot source estimates obtained with sparse solver.

    Active dipoles are represented in a "Glass" brain.
    If the same source is active in multiple source estimates it is
    displayed with a sphere otherwise with a cone in 3D.

    Parameters
    ----------
    src : dict
        The source space.
    stcs : instance of SourceEstimate or list of instances of SourceEstimate
        The source estimates.
    colors : list
        List of colors.
    linewidth : int
        Line width in 2D plot.
    fontsize : int
        Font size.
    bgcolor : tuple of length 3
        Background color in 3D.
    opacity : float in [0, 1]
        Opacity of brain mesh.
    brain_color : tuple of length 3
        Brain color.
    show : bool
        Show figures if True.
    high_resolution : bool
        If True, plot on the original (non-downsampled) cortical mesh.
    fig_name : str
        PyVista figure name.
    fig_number : int
        Matplotlib figure number.
    labels : ndarray or list of ndarray
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    modes : list
        Should be a list, with each entry being ``'cone'`` or ``'sphere'``
        to specify how the dipoles should be shown.
        The pivot for the glyphs in ``'cone'`` mode is always the tail
        whereas the pivot in ``'sphere'`` mode is the center.
    scale_factors : list
        List of floating point scale factors for the markers.
    %(verbose)s
    **kwargs : kwargs
        Keyword arguments to pass to renderer.mesh.

    Returns
    -------
    surface : instance of Figure3D
        The 3D figure containing the triangular mesh surface.
    """
    import matplotlib.pyplot as plt

    # Update the backend
    from .backends.renderer import _get_renderer

    linestyles = [
        ("solid", "solid"),  # noqa: E241
        ("dashed", "dashed"),  # noqa: E241
        ("dotted", "dotted"),  # noqa: E241
        ("dashdot", "dashdot"),  # noqa: E241
        ("loosely dotted", (0, (1, 10))),  # noqa: E241
        ("dotted", (0, (1, 1))),  # noqa: E241
        ("densely dotted", (0, (1, 1))),  # noqa: E241
        ("loosely dashed", (0, (5, 10))),  # noqa: E241
        ("dashed", (0, (5, 5))),  # noqa: E241
        ("densely dashed", (0, (5, 1))),  # noqa: E241
        ("loosely dashdotted", (0, (3, 10, 1, 10))),  # noqa: E241
        ("dashdotted", (0, (3, 5, 1, 5))),  # noqa: E241
        ("densely dashdotted", (0, (3, 1, 1, 1))),  # noqa: E241
        ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),  # noqa: E241
        ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),  # noqa: E241
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),  # noqa: E241
    ]

    known_modes = ["cone", "sphere"]
    if not isinstance(modes, list | tuple) or not all(
        mode in known_modes for mode in modes
    ):
        raise ValueError('mode must be a list containing only "cone" or "sphere"')
    if not isinstance(stcs, list):
        stcs = [stcs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    if colors is None:
        colors = _get_color_list()

    linestyles = cycle(linestyles)
    linestyles = [next(linestyles)[1] for _ in range(len(stcs))]

    # Show 3D
    lh_points = src[0]["rr"]
    rh_points = src[1]["rr"]
    points = np.r_[lh_points, rh_points]

    lh_normals = src[0]["nn"]
    rh_normals = src[1]["nn"]
    normals = np.r_[lh_normals, rh_normals]

    if high_resolution:
        use_lh_faces = src[0]["tris"]
        use_rh_faces = src[1]["tris"]
    else:
        use_lh_faces = src[0]["use_tris"]
        use_rh_faces = src[1]["use_tris"]

    use_faces = np.r_[use_lh_faces, lh_points.shape[0] + use_rh_faces]

    points *= 170

    vertnos = [np.r_[stc.lh_vertno, lh_points.shape[0] + stc.rh_vertno] for stc in stcs]
    unique_vertnos = np.unique(np.concatenate(vertnos).ravel())

    renderer = _get_renderer(bgcolor=bgcolor, size=(600, 600), name=fig_name)
    renderer.mesh(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        triangles=use_faces,
        color=brain_color,
        opacity=opacity,
        backface_culling=True,
        normals=normals,
        **kwargs,
    )

    # Show time courses
    fig = plt.figure(fig_number, layout="constrained")
    fig.clf()
    ax = fig.add_subplot(111)

    colors = cycle(colors)

    logger.info(f"Total number of active sources: {unique_vertnos}")

    if labels is not None:
        colors = [
            next(colors) for _ in range(np.unique(np.concatenate(labels).ravel()).size)
        ]

    for idx, v in enumerate(unique_vertnos):
        # get indices of stcs it belongs to
        ind = [k for k, vertno in enumerate(vertnos) if v in vertno]
        is_common = len(ind) > 1

        if labels is None:
            c = next(colors)
        else:
            # if vertex is in different stcs than take label from first one
            c = colors[labels[ind[0]][vertnos[ind[0]] == v]]

        mode = modes[1] if is_common else modes[0]
        scale_factor = scale_factors[1] if is_common else scale_factors[0]

        if isinstance(scale_factor, np.ndarray | list | tuple) and len(
            unique_vertnos
        ) == len(scale_factor):
            scale_factor = scale_factor[idx]

        x, y, z = points[v]
        nx, ny, nz = normals[v]
        renderer.quiver3d(
            x=x,
            y=y,
            z=z,
            u=nx,
            v=ny,
            w=nz,
            color=_to_rgb(c),
            mode=mode,
            scale=scale_factor,
        )

        for k in ind:
            vertno = vertnos[k]
            mask = vertno == v
            assert np.sum(mask) == 1
            linestyle = linestyles[k]
            ax.plot(
                1e3 * stcs[k].times,
                1e9 * stcs[k].data[mask].ravel(),
                c=c,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    ax.set_xlabel("Time (ms)", fontsize=fontsize)
    ax.set_ylabel("Source amplitude (nAm)", fontsize=fontsize)

    if fig_name is not None:
        ax.set_title(fig_name)
    plt_show(show)

    renderer.show()
    renderer.set_camera(distance="auto", focalpoint="auto")
    return renderer.scene()


@verbose
def plot_dipole_locations(
    dipoles,
    trans=None,
    subject=None,
    subjects_dir=None,
    mode="orthoview",
    coord_frame="mri",
    idx="gof",
    show_all=True,
    ax=None,
    block=False,
    show=True,
    scale=None,
    color=None,
    *,
    highlight_color="r",
    fig=None,
    title=None,
    head_source="seghead",
    surf="pial",
    width=None,
    verbose=None,
):
    """Plot dipole locations.

    If mode is set to 'arrow' or 'sphere', only the location of the first
    time point of each dipole is shown else use the show_all parameter.

    Parameters
    ----------
    dipoles : list of instances of Dipole | Dipole
        The dipoles to plot.
    trans : dict | None
        The mri to head trans.
        Can be None with mode set to '3d'.
    subject : str | None
        The FreeSurfer subject name (will be used to set the FreeSurfer
        environment variable ``SUBJECT``).
        Can be ``None`` with mode set to ``'3d'``.
    %(subjects_dir)s
    mode : str
        Can be:

        ``'arrow'`` or ``'sphere'``
            Plot in 3D mode using PyVista with the given glyph type.
        ``'orthoview'``
            Plot in matplotlib ``Axes3D`` using matplotlib with MRI slices
            shown on the sides of a cube, with the dipole(s) shown as arrows
            extending outward from a dot (i.e., the arrows pivot on the tail).
        ``'outlines'``
            Plot in matplotlib ``Axes`` using a quiver of arrows for the
            dipoles in three axes (axial, coronal, and sagittal views),
            with the arrow pivoting in the middle of the arrow.

        .. versionchanged:: 1.1
           Added support for ``'outlines'``.
    coord_frame : str
        Coordinate frame to use: 'head' or 'mri'. Can also be 'mri_rotated'
        when mode equals ``'outlines'``. Defaults to 'mri'.

        .. versionadded:: 0.14.0
        .. versionchanged:: 1.1
           Added support for ``'mri_rotated'``.
    idx : int | 'gof' | 'amplitude'
        Index of the initially plotted dipole. Can also be 'gof' to plot the
        dipole with highest goodness of fit value or 'amplitude' to plot the
        dipole with the highest amplitude. The dipoles can also be browsed
        through using up/down arrow keys or mouse scroll. Defaults to 'gof'.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    show_all : bool
        Whether to always plot all the dipoles. If ``True`` (default), the
        active dipole is plotted as a red dot and its location determines the
        shown MRI slices. The non-active dipoles are plotted as small blue
        dots. If ``False``, only the active dipole is plotted.
        Only used if ``mode='orthoview'``.

        .. versionadded:: 0.14.0
    ax : instance of matplotlib Axes3D | list of matplotlib Axes | None
        Axes to plot into. If None (default), axes will be created.
        If mode equals ``'orthoview'``, must be a single ``Axes3D``.
        If mode equals ``'outlines'``, must be a list of three ``Axes``.

        .. versionadded:: 0.14.0
    block : bool
        Whether to halt program execution until the figure is closed. Defaults
        to False.
        Only used if mode equals 'orthoview'.

        .. versionadded:: 0.14.0
    show : bool
        Show figure if True. Defaults to True.
        Only used if mode equals 'orthoview'.
    scale : float
        The scale (size in meters) of the dipoles if ``mode`` is not
        ``'orthoview'``. The default is 0.03 when mode is ``'outlines'`` and
        0.005 otherwise.
    color : tuple
        The color of the dipoles.
        The default (None) will use ``'y'`` if mode is ``'orthoview'`` and
        ``show_all`` is True, else 'r'. Can also be a list of colors to use
        when mode is ``'outlines'``.

        .. versionchanged:: 0.19.0
           Color is now passed in orthoview mode.
    highlight_color : color
        The highlight color. Only used in orthoview mode with
        ``show_all=True``.

        .. versionadded:: 0.19.0
    fig : instance of Figure3D | None
        3D figure in which to plot the alignment.
        If ``None``, creates a new 600x600 pixel figure with black background.
        Only used when mode is ``'arrow'`` or ``'sphere'``.

        .. versionadded:: 0.19.0
    title : str | None
        The title of the figure if ``mode='orthoview'`` (ignored for all other
        modes). If ``None``, dipole number and its properties (amplitude,
        orientation etc.) will be shown. Defaults to ``None``.

        .. versionadded:: 0.21.0
    %(head_source)s
        Only used when mode equals ``'outlines'``.

        .. versionadded:: 1.1
    surf : str | None
        Brain surface to show outlines for, can be ``'white'``, ``'pial'``, or
        ``None``. Only used when mode is ``'outlines'``.

        .. versionadded:: 1.1
    width : float | None
        Width of the matplotlib quiver arrow, see
        :meth:`matplotlib:matplotlib.axes.Axes.quiver`. If None (default),
        when mode is ``'outlines'`` 0.015 will be used, and when mode is
        ``'orthoview'`` the matplotlib default is used.
    %(verbose)s

    Returns
    -------
    fig : instance of Figure3D or matplotlib.figure.Figure
        The PyVista figure or matplotlib Figure.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    _validate_type(mode, str, "mode")
    _validate_type(coord_frame, str, "coord_frame")
    _check_option("mode", mode, ("orthoview", "outlines", "arrow", "sphere"))
    if mode in ("orthoview", "outlines"):
        subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=True))
    kwargs = dict(
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        coord_frame=coord_frame,
        ax=ax,
        block=block,
        show=show,
        color=color,
        title=title,
        width=width,
    )
    dipoles = _check_concat_dipoles(dipoles)
    if mode == "orthoview":
        fig = _plot_dipole_mri_orthoview(
            dipoles,
            idx=idx,
            show_all=show_all,
            highlight_color=highlight_color,
            **kwargs,
        )
    elif mode == "outlines":
        fig = _plot_dipole_mri_outlines(
            dipoles, head_source=head_source, surf=surf, scale=scale, **kwargs
        )
    else:
        assert mode in ("arrow", "sphere"), mode
        fig = _plot_dipole_3d(
            dipoles,
            trans=trans,
            coord_frame=coord_frame,
            color=color,
            fig=fig,
            scale=scale,
            mode=mode,
        )

    return fig


def snapshot_brain_montage(fig, montage, hide_sensors=True):
    """Take a snapshot of a PyVista Scene and project channels onto 2d coords.

    Note that this will take the raw values for 3d coordinates of each channel,
    without applying any transforms. If brain images are flipped up/dn upon
    using `~matplotlib.pyplot.imshow`, check your matplotlib backend as this
    behavior changes.

    Parameters
    ----------
    fig : instance of Figure3D
        The figure on which you've plotted electrodes using
        :func:`mne.viz.plot_alignment`.
    montage : instance of DigMontage or Info | dict
        The digital montage for the electrodes plotted in the scene. If
        :class:`~mne.Info`, channel positions will be pulled from the ``loc``
        field of ``chs``. dict should have ch:xyz mappings.
    hide_sensors : bool
        Whether to remove the spheres in the scene before taking a snapshot.
        The sensors will always be shown in the final figure. If you want an
        image of just the brain, use :class:`mne.viz.Brain` instead.

    Returns
    -------
    xy : array, shape (n_channels, 2)
        The 2d location of each channel on the image of the current scene view.
    im : array, shape (m, n, 3)
        The screenshot of the current scene view.
    """
    from ..channels import DigMontage

    # Update the backend
    from .backends.renderer import _get_renderer

    if fig is None:
        raise ValueError("The figure must have a scene")
    if isinstance(montage, DigMontage):
        chs = montage._get_ch_pos()
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
    elif isinstance(montage, Info):
        xyz = [ich["loc"][:3] for ich in montage["chs"]]
        ch_names = [ich["ch_name"] for ich in montage["chs"]]
    elif isinstance(montage, dict):
        if not all(len(ii) == 3 for ii in montage.values()):
            raise ValueError("All electrode positions must be length 3")
        ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in montage.items()])
    else:
        raise TypeError(
            "montage must be an instance of `DigMontage`, `Info`, or `dict`"
        )

    # initialize figure
    renderer = _get_renderer(fig, show=True)

    xyz = np.vstack(xyz)
    proj = renderer.project(xyz=xyz, ch_names=ch_names)
    if hide_sensors is True:
        proj.visible(False)

    im = renderer.screenshot()
    proj.visible(True)
    return proj.xy, im


def _plot_dipole_mri_orthoview(
    dipole,
    trans,
    subject,
    subjects_dir=None,
    coord_frame="head",
    idx="gof",
    show_all=True,
    ax=None,
    block=False,
    show=True,
    color=None,
    highlight_color="r",
    title=None,
    width=None,
):
    """Plot dipoles on top of MRI slices in 3-D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    _import_nibabel("plotting MRI slices")

    _check_option("coord_frame", coord_frame, ["head", "mri"])

    if idx == "gof":
        idx = np.argmax(dipole.gof)
    elif idx == "amplitude":
        idx = np.argmax(np.abs(dipole.amplitude))
    else:
        idx = _ensure_int(idx, "idx", 'an int or one of ["gof", "amplitude"]')

    vox, ori, pos, data = _get_dipole_loc(
        dipole, trans, subject, subjects_dir, coord_frame
    )

    dims = len(data)  # Symmetric size assumed.
    dd = dims // 2
    if ax is None:
        fig, ax = plt.subplots(
            1, subplot_kw=dict(projection="3d"), layout="constrained"
        )
    else:
        _validate_type(ax, Axes3D, "ax", "Axes3D", extra='when mode is "orthoview"')
        fig = ax.get_figure()

    gridx, gridy = np.meshgrid(
        np.linspace(-dd, dd, dims), np.linspace(-dd, dd, dims), indexing="ij"
    )
    params = {
        "ax": ax,
        "data": data,
        "idx": idx,
        "dipole": dipole,
        "vox": vox,
        "gridx": gridx,
        "gridy": gridy,
        "ori": ori,
        "coord_frame": coord_frame,
        "show_all": show_all,
        "pos": pos,
        "color": color,
        "highlight_color": highlight_color,
        "title": title,
        "width": width,
    }
    _plot_dipole(**params)
    ax.view_init(elev=30, azim=-140)

    callback_func = partial(_dipole_changed, params=params)
    fig.canvas.mpl_connect("scroll_event", callback_func)
    fig.canvas.mpl_connect("key_press_event", callback_func)

    plt_show(show, block=block)
    return fig


RAS_AFFINE = np.eye(4)
RAS_AFFINE[:3, 3] = [-128] * 3
RAS_SHAPE = (256, 256, 256)


def _get_dipole_loc(dipole, trans, subject, subjects_dir, coord_frame):
    """Get the dipole locations and orientations."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    _check_option("coord_frame", coord_frame, ["head", "mri"])

    subjects_dir = str(get_subjects_dir(subjects_dir=subjects_dir, raise_error=True))
    t1_fname = op.join(subjects_dir, subject, "mri", "T1.mgz")
    t1 = nib.load(t1_fname)
    # Do everything in mm here to make life slightly easier
    vox_ras_t, _, mri_ras_t, _, _ = _read_mri_info(t1_fname, units="mm")
    head_mri_t = _get_trans(trans, fro="head", to="mri")[0].copy()
    head_mri_t["trans"][:3, 3] *= 1000  # m→mm
    del trans
    pos = dipole.pos * 1e3  # m→mm
    ori = dipole.ori
    # Figure out how to always resample to an identity, 256x256x256 RAS:
    #
    # 1. Resample to head or MRI surface RAS (the conditional), but also
    # 2. Resample to what will work for the standard 1mm** RAS_AFFINE (resamp)
    #
    # We could do this with two resample_from_to calls, but it's cleaner,
    # faster, and we get fewer boundary artifacts if we do it in one shot.
    # So first olve usamp s.t. ``upsamp @ vox_ras_t == RAS_AFFINE`` (2):
    upsamp = np.linalg.solve(vox_ras_t["trans"].T, RAS_AFFINE.T).T
    # Now figure out how we would resample from RAS to head or MRI coords:
    if coord_frame == "head":
        dest_ras_t = combine_transforms(head_mri_t, mri_ras_t, "head", "ras")["trans"]
    else:
        pos = apply_trans(head_mri_t, pos)
        ori = apply_trans(head_mri_t, dipole.ori, move=False)
        dest_ras_t = mri_ras_t["trans"]
    # The order here is wacky because we need `resample_from_to` to operate
    # in a reverse order
    affine = np.dot(np.dot(dest_ras_t, upsamp), vox_ras_t["trans"])
    t1 = resample_from_to(t1, (RAS_SHAPE, affine), order=0)
    # Now we could do:
    #
    #    t1 = SpatialImage(t1.dataobj, RAS_AFFINE)
    #
    # And t1 would be in our destination (mri or head) space. But we don't
    # need to construct the image -- let's just get our voxel coords and data:
    vox = apply_trans(np.linalg.inv(RAS_AFFINE), pos)
    t1_data = _get_img_fdata(t1)
    return vox, ori, pos, t1_data


def _plot_dipole(
    ax,
    data,
    vox,
    idx,
    dipole,
    gridx,
    gridy,
    ori,
    coord_frame,
    show_all,
    pos,
    color,
    highlight_color,
    title,
    width,
):
    """Plot dipoles."""
    import matplotlib.pyplot as plt

    xidx, yidx, zidx = np.round(vox[idx]).astype(int)
    xslice = data[xidx]
    yslice = data[:, yidx]
    zslice = data[:, :, zidx]

    ori = ori[idx]
    if color is None:
        color = "y" if show_all else "r"
    color = np.array(_to_rgb(color, alpha=True))
    highlight_color = np.array(
        _to_rgb(highlight_color, name="highlight_color", alpha=True)
    )
    if show_all:
        colors = np.repeat(color[np.newaxis], len(vox), axis=0)
        colors[idx] = highlight_color
        size = np.repeat(5, len(vox))
        size[idx] = 20
        visible = np.arange(len(vox))
    else:
        colors = color
        size = 20
        visible = idx

    offset = np.min(gridx)
    xyz = pos
    ax.scatter(
        xs=xyz[visible, 0],
        ys=xyz[visible, 1],
        zs=xyz[visible, 2],
        zorder=2,
        s=size,
        facecolor=colors,
    )
    xx = np.linspace(offset, xyz[idx, 0], xidx)
    yy = np.linspace(offset, xyz[idx, 1], yidx)
    zz = np.linspace(offset, xyz[idx, 2], zidx)
    ax.plot(
        xx,
        np.repeat(xyz[idx, 1], len(xx)),
        zs=xyz[idx, 2],
        zorder=1,
        linestyle="-",
        color=highlight_color,
    )
    ax.plot(
        np.repeat(xyz[idx, 0], len(yy)),
        yy,
        zs=xyz[idx, 2],
        zorder=1,
        linestyle="-",
        color=highlight_color,
    )
    ax.plot(
        np.repeat(xyz[idx, 0], len(zz)),
        np.repeat(xyz[idx, 1], len(zz)),
        zs=zz,
        zorder=1,
        linestyle="-",
        color=highlight_color,
    )
    q_kwargs = dict(length=50, color=highlight_color, pivot="tail")
    if width is not None:
        q_kwargs["width"] = width
    ax.quiver(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], ori[0], ori[1], ori[2], **q_kwargs)
    dims = np.array([(len(data) / -2.0), (len(data) / 2.0)])
    ax.set(xlim=-dims, ylim=-dims, zlim=dims)

    # Plot slices
    ax.contourf(
        xslice, gridx, gridy, offset=offset, zdir="x", cmap="gray", zorder=0, alpha=0.5
    )
    ax.contourf(
        gridx, yslice, gridy, offset=offset, zdir="y", cmap="gray", zorder=0, alpha=0.5
    )
    ax.contourf(
        gridx, gridy, zslice, offset=offset, zdir="z", cmap="gray", zorder=0, alpha=0.5
    )

    # Plot orientations
    args = np.array([list(xyz[idx]) + list(ori)] * 3)
    for ii in range(3):
        args[ii, [ii, ii + 3]] = [offset + 0.5, 0]  # half a mm inward  (z ord)
    ax.quiver(*args.T, alpha=0.75, **q_kwargs)

    # These are the only two options
    coord_frame_name = "Head" if coord_frame == "head" else "MRI"

    if title is None:
        title = (
            f"Dipole #{idx + 1} / {len(dipole.times)} @ {dipole.times[idx]:.3f}s, "
            f"GOF: {dipole.gof[idx]:.1f}%, {dipole.amplitude[idx] * 1e9:.1f}nAm\n"
            f"{coord_frame_name}: {_str_ras(xyz[idx])}"
        )

    ax.get_figure().suptitle(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.draw()


def _dipole_changed(event, params):
    """Handle dipole plotter scroll/key event."""
    if event.key is not None:
        if event.key == "up":
            params["idx"] += 1
        elif event.key == "down":
            params["idx"] -= 1
        else:  # some other key
            return
    elif event.step > 0:  # scroll event
        params["idx"] += 1
    else:
        params["idx"] -= 1
    params["idx"] = min(max(0, params["idx"]), len(params["dipole"].pos) - 1)
    params["ax"].clear()
    _plot_dipole(**params)


@fill_doc
def plot_brain_colorbar(
    ax,
    clim,
    colormap="auto",
    transparent=True,
    orientation="vertical",
    label="Activation",
    bgcolor="0.5",
):
    """Plot a colorbar that corresponds to a brain activation map.

    Parameters
    ----------
    ax : instance of Axes
        The Axes to plot into.
    %(clim)s
    %(colormap)s
    %(transparent)s
    orientation : str
        Orientation of the colorbar, can be "vertical" or "horizontal".
    label : str
        The colorbar label.
    bgcolor : color
        The color behind the colorbar (for alpha blending).

    Returns
    -------
    cbar : instance of ColorbarBase
        The colorbar.

    Notes
    -----
    .. versionadded:: 0.19
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    mapdata = _process_clim(clim, colormap, transparent)
    ticks = _get_map_ticks(mapdata)
    colormap, lims = _linearize_map(mapdata)
    del mapdata
    norm = Normalize(vmin=lims[0], vmax=lims[2])
    cbar = ColorbarBase(
        ax, cmap=colormap, norm=norm, ticks=ticks, label=label, orientation=orientation
    )
    # make the colorbar background match the brain color
    cbar.ax.set(facecolor=bgcolor)
    # remove the colorbar frame except for the line containing the ticks
    cbar.outline.set_visible(False)
    cbar.ax.set_frame_on(True)
    for key in ("left", "top", "bottom" if orientation == "vertical" else "right"):
        ax.spines[key].set_visible(False)
    return cbar


@dataclass()
class _3d_Options:
    antialias: bool | None
    depth_peeling: bool | None
    smooth_shading: bool | None
    multi_samples: int | None


_3d_options = _3d_Options(
    antialias=None,
    depth_peeling=None,
    smooth_shading=None,
    multi_samples=None,
)
_3d_default = _3d_Options(
    antialias="true",
    depth_peeling="true",
    smooth_shading="true",
    multi_samples="4",
)


def set_3d_options(
    antialias=None, depth_peeling=None, smooth_shading=None, *, multi_samples=None
):
    """Set 3D rendering options.

    Parameters
    ----------
    antialias : bool | None
        If bool, whether to enable or disable full-screen anti-aliasing.
        False is useful when renderers have problems (such as software
        MESA renderers). If None, use the default setting. This option
        can also be controlled using an environment variable, e.g.,
        ``MNE_3D_OPTION_ANTIALIAS=false``.
    depth_peeling : bool | None
        If bool, whether to enable or disable accurate transparency.
        False is useful when renderers have problems (for instance
        while X forwarding on remote servers). If None, use the default
        setting. This option can also be controlled using an environment
        variable, e.g., ``MNE_3D_OPTION_DEPTH_PEELING=false``.
    smooth_shading : bool | None
        If bool, whether to enable or disable smooth color transitions
        between polygons. False is useful on certain configurations
        where this type of shading is not supported or for performance
        reasons. This option can also be controlled using an environment
        variable, e.g., ``MNE_3D_OPTION_SMOOTH_SHADING=false``.
    multi_samples : int
        Number of multi-samples. Should be 1 for MESA for volumetric rendering
        to work properly.

        .. versionadded:: 1.1

    Notes
    -----
    .. versionadded:: 0.21.0
    """
    if antialias is not None:
        _3d_options.antialias = bool(antialias)
    if depth_peeling is not None:
        _3d_options.depth_peeling = bool(depth_peeling)
    if smooth_shading is not None:
        _3d_options.smooth_shading = bool(smooth_shading)
    if multi_samples is not None:
        _3d_options.multi_samples = int(multi_samples)


def _get_3d_option(key):
    _validate_type(key, "str", "key")
    opt = getattr(_3d_options, key)
    if opt is None:  # parse get_config (and defaults)
        default_value = getattr(_3d_default, key)
        opt = get_config(f"MNE_3D_OPTION_{key.upper()}", default_value)
        if key == "multi_samples":
            opt = int(opt)
        else:
            opt = opt.lower() == "true"
    return opt
