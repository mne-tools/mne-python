"""Dipole viz specific functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op

import numpy as np
from scipy.spatial import ConvexHull

from .._freesurfer import _estimate_talxfm_rigid, _get_head_surface
from ..surface import read_surface
from ..transforms import _get_trans, apply_trans, invert_transform
from ..utils import _check_option, _validate_type, get_subjects_dir
from .utils import _validate_if_list_of_axes, plt_show


def _check_concat_dipoles(dipole):
    from ..dipole import Dipole, _concatenate_dipoles

    if not isinstance(dipole, Dipole):
        dipole = _concatenate_dipoles(dipole)
    return dipole


def _plot_dipole_mri_outlines(
    dipoles,
    *,
    subject,
    trans,
    ax,
    subjects_dir,
    color,
    scale,
    coord_frame,
    show,
    block,
    head_source,
    title,
    surf,
    width,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Circle

    extra = 'when mode is "outlines"'
    trans = _get_trans(trans, fro="head", to="mri")[0]
    _check_option(
        "coord_frame", coord_frame, ["head", "mri", "mri_rotated"], extra=extra
    )
    _validate_type(surf, (str, None), "surf")
    _check_option("surf", surf, ("white", "pial", None))
    if ax is None:
        _, ax = plt.subplots(1, 3, figsize=(7, 2.5), squeeze=True, layout="constrained")
    _validate_if_list_of_axes(ax, 3, name="ax")
    dipoles = _check_concat_dipoles(dipoles)
    color = "r" if color is None else color
    scale = 0.03 if scale is None else scale
    width = 0.015 if width is None else width
    fig = ax[0].figure
    surfs = dict()
    hemis = ("lh", "rh")
    if surf is not None:
        for hemi in hemis:
            surfs[hemi] = read_surface(
                op.join(subjects_dir, subject, "surf", f"{hemi}.{surf}"),
                return_dict=True,
            )[2]
            surfs[hemi]["rr"] /= 1000.0
    subjects_dir = get_subjects_dir(subjects_dir)
    if subjects_dir is not None:
        subjects_dir = str(subjects_dir)
    surfs["head"] = _get_head_surface(head_source, subject, subjects_dir)
    del head_source
    mri_trans = head_trans = np.eye(4)
    if coord_frame in ("mri", "mri_rotated"):
        head_trans = trans["trans"]
        if coord_frame == "mri_rotated":
            rot = _estimate_talxfm_rigid(subject, subjects_dir)
            rot[:3, 3] = 0.0
            head_trans = rot @ head_trans
            mri_trans = rot @ mri_trans
    else:
        assert coord_frame == "head"
        mri_trans = invert_transform(trans)["trans"]
    for s in surfs.values():
        s["rr"] = 1000 * apply_trans(mri_trans, s["rr"])
    del mri_trans
    levels = dict()
    if surf is not None:
        use_rr = np.concatenate([surfs[key]["rr"] for key in hemis])
    else:
        use_rr = surfs["head"]["rr"]
    views = [("Axial", "XY"), ("Coronal", "XZ"), ("Sagittal", "YZ")]
    # axial: 25% up the Z axis
    axial = float(np.percentile(use_rr[:, 2], 20.0))
    coronal = float(np.percentile(use_rr[:, 1], 55.0))
    for key in hemis + ("head",):
        levels[key] = dict(Axial=axial, Coronal=coronal)
    if surf is not None:
        levels["rh"]["Sagittal"] = float(np.percentile(surfs["rh"]["rr"][:, 0], 50))
    levels["head"]["Sagittal"] = 0.0
    for ax_, (name, coords) in zip(ax, views):
        idx = list(map(dict(X=0, Y=1, Z=2).get, coords))
        miss = np.setdiff1d(np.arange(3), idx)[0]
        pos = 1000 * apply_trans(head_trans, dipoles.pos)
        ori = 1000 * apply_trans(head_trans, dipoles.ori, move=False)
        lims = dict()
        for ii, char in enumerate(coords):
            lim = surfs["head"]["rr"][:, idx[ii]]
            lim = np.array([lim.min(), lim.max()])
            lims[char] = lim
        ax_.quiver(
            pos[:, idx[0]],
            pos[:, idx[1]],
            scale * ori[:, idx[0]],
            scale * ori[:, idx[1]],
            color=color,
            pivot="middle",
            zorder=5,
            scale_units="xy",
            angles="xy",
            scale=1.0,
            width=width,
            minshaft=0.5,
            headwidth=2.5,
            headlength=2.5,
            headaxislength=2,
        )
        coll = PatchCollection(
            [
                Circle((x, y), radius=scale * 1000 * width * 6)
                for x, y in zip(pos[:, idx[0]], pos[:, idx[1]])
            ],
            linewidths=0.0,
            facecolors=color,
            zorder=6,
        )
        for key, surf in surfs.items():
            try:
                level = levels[key][name]
            except KeyError:
                continue
            if key != "head":
                rrs = surf["rr"][:, idx]
                tris = ConvexHull(rrs).simplices
                segments = LineCollection(
                    rrs[:, [0, 1]][tris],
                    linewidths=1,
                    linestyles="-",
                    colors="k",
                    zorder=3,
                    alpha=0.25,
                )
                ax_.add_collection(segments)
            ax_.tricontour(
                surf["rr"][:, idx[0]],
                surf["rr"][:, idx[1]],
                surf["tris"],
                surf["rr"][:, miss],
                levels=[level],
                colors="k",
                linewidths=1.0,
                linestyles=["-"],
                zorder=4,
                alpha=0.5,
            )
            # TODO: this breaks the PatchCollection in MPL
            # for coll in h.collections:
            #     coll.set_clip_on(False)
        ax_.add_collection(coll)
        ax_.set(
            title=name,
            xlim=lims[coords[0]],
            ylim=lims[coords[1]],
            xlabel=coords[0] + " (mm)",
            ylabel=coords[1] + " (mm)",
        )
        for spine in ax_.spines.values():
            spine.set_visible(False)
        ax_.grid(True, ls=":", zorder=2)
        ax_.set_aspect("equal")

    if title is not None:
        fig.suptitle(title)
    plt_show(show, block=block)

    return fig


def _plot_dipole_3d(dipoles, *, coord_frame, color, fig, trans, scale, mode):
    from .backends.renderer import _get_renderer

    _check_option("coord_frame", coord_frame, ("head", "mri"))
    color = "r" if color is None else color
    scale = 0.005 if scale is None else scale
    renderer = _get_renderer(fig=fig, size=(600, 600))
    pos = dipoles.pos
    ori = dipoles.ori
    if coord_frame != "head":
        trans = _get_trans(trans, fro="head", to=coord_frame)[0]
        pos = apply_trans(trans, pos)
        ori = apply_trans(trans, ori)

    renderer.sphere(center=pos, color=color, scale=scale)
    if mode == "arrow":
        x, y, z = pos.T
        u, v, w = ori.T
        renderer.quiver3d(x, y, z, u, v, w, scale=3 * scale, color=color, mode="arrow")
    renderer.show()
    fig = renderer.scene()
    return fig
