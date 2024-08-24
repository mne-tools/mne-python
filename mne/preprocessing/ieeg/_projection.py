# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..._fiff.pick import _picks_to_idx
from ...channels import make_dig_montage
from ...surface import (
    _compute_nearest,
    _read_mri_surface,
    _read_patch,
    fast_cross_3d,
    read_surface,
)
from ...transforms import _cart_to_sph, _ensure_trans, apply_trans, invert_transform
from ...utils import _ensure_int, _validate_type, get_subjects_dir, verbose


@verbose
def project_sensors_onto_brain(
    info,
    trans,
    subject,
    subjects_dir=None,
    picks=None,
    n_neighbors=10,
    copy=True,
    verbose=None,
):
    """Project sensors onto the brain surface.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    %(subject)s
    %(subjects_dir)s
    %(picks_base)s only ``ecog`` channels.
    n_neighbors : int
        The number of neighbors to use to compute the normal vectors
        for the projection. Must be 2 or greater. More neighbors makes
        a normal vector with greater averaging which preserves the grid
        structure. Fewer neighbors has less averaging which better
        preserves contours in the grid.
    copy : bool
        If ``True``, return a new instance of ``info``, if ``False``
        ``info`` is modified in place.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s

    Notes
    -----
    This is useful in ECoG analysis for compensating for "brain shift"
    or shrinking of the brain away from the skull due to changes
    in pressure during the craniotomy.

    To use the brain surface, a BEM model must be created e.g. using
    :ref:`mne watershed_bem` using the T1 or :ref:`mne flash_bem`
    using a FLASH scan.
    """
    n_neighbors = _ensure_int(n_neighbors, "n_neighbors")
    _validate_type(copy, bool, "copy")
    if copy:
        info = info.copy()
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be 2 or greater, got {n_neighbors}")
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    try:
        surf = _read_mri_surface(subjects_dir / subject / "bem" / "brain.surf")
    except FileNotFoundError as err:
        raise RuntimeError(
            f"{err}\n\nThe brain surface requires generating "
            "a BEM using `mne flash_bem` (if you have "
            "the FLASH scan) or `mne watershed_bem` (to "
            "use the T1)"
        ) from None
    # get channel locations
    picks_idx = _picks_to_idx(info, "ecog" if picks is None else picks)
    locs = np.array([info["chs"][idx]["loc"][:3] for idx in picks_idx])
    trans = _ensure_trans(trans, "head", "mri")
    locs = apply_trans(trans, locs)
    # compute distances for nearest neighbors
    dists = squareform(pdist(locs))
    # find angles for brain surface and points
    angles = _cart_to_sph(locs)
    surf_angles = _cart_to_sph(surf["rr"])
    # initialize projected locs
    proj_locs = np.zeros(locs.shape) * np.nan
    for i, loc in enumerate(locs):
        neighbor_pts = locs[np.argsort(dists[i])[: n_neighbors + 1]]
        pt1, pt2, pt3 = map(np.array, zip(*combinations(neighbor_pts, 3)))
        normals = fast_cross_3d(pt1 - pt2, pt1 - pt3)
        normals[normals @ loc < 0] *= -1
        normal = np.mean(normals, axis=0)
        normal /= np.linalg.norm(normal)
        # find the correct orientation brain surface point nearest the line
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        use_rr = surf["rr"][
            abs(surf_angles[:, 1:] - angles[i, 1:]).sum(axis=1) < np.pi / 4
        ]
        surf_dists = np.linalg.norm(
            fast_cross_3d(use_rr - loc, use_rr - loc + normal), axis=1
        )
        proj_locs[i] = use_rr[np.argmin(surf_dists)]
    # back to the "head" coordinate frame for storing in ``raw``
    proj_locs = apply_trans(invert_transform(trans), proj_locs)
    montage = info.get_montage()
    montage_kwargs = (
        montage.get_positions() if montage else dict(ch_pos=dict(), coord_frame="head")
    )
    for idx, loc in zip(picks_idx, proj_locs):
        # surface RAS-> head and mm->m
        montage_kwargs["ch_pos"][info.ch_names[idx]] = loc
    info.set_montage(make_dig_montage(**montage_kwargs))
    return info


@verbose
def _project_sensors_onto_inflated(
    info,
    trans,
    subject,
    subjects_dir=None,
    picks=None,
    max_dist=0.004,
    flat=False,
    verbose=None,
):
    """Project sensors onto the brain surface.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    %(subject)s
    %(subjects_dir)s
    %(picks_base)s only ``seeg`` channels.
    %(max_dist_ieeg)s
    flat : bool
        Whether to project the sensors onto the flat map of the
        inflated brain instead of the normal inflated brain.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s

    Notes
    -----
    This is useful in sEEG analysis for visualization
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    surf_data = dict(lh=dict(), rh=dict())
    x_dir = np.array([1.0, 0.0, 0.0])
    surfs = ("pial", "inflated")
    if flat:
        surfs += ("cortex.patch.flat",)
    for hemi in ("lh", "rh"):
        for surf in surfs:
            for img in ("", ".T1", ".T2", ""):
                surf_fname = subjects_dir / subject / "surf" / f"{hemi}.{surf}"
                if surf_fname.is_file():
                    break
            if surf.split(".")[-1] == "flat":
                surf = "flat"
                coords, faces, orig_faces = _read_patch(surf_fname)
                # rotate 90 degrees to get to a more standard orientation
                # where X determines the distance between the hemis
                coords = coords[:, [1, 0, 2]]
                coords[:, 1] *= -1
            else:
                coords, faces = read_surface(surf_fname)
            if surf in ("inflated", "flat"):
                x_ = coords @ x_dir
                coords -= np.max(x_) * x_dir if hemi == "lh" else np.min(x_) * x_dir
            surf_data[hemi][surf] = (coords / 1000, faces)  # mm -> m
    # get channel locations
    picks_idx = _picks_to_idx(info, "seeg" if picks is None else picks)
    locs = np.array([info["chs"][idx]["loc"][:3] for idx in picks_idx])
    trans = _ensure_trans(trans, "head", "mri")
    locs = apply_trans(trans, locs)
    # initialize projected locs
    proj_locs = np.zeros(locs.shape) * np.nan
    surf = "flat" if flat else "inflated"
    for hemi in ("lh", "rh"):
        hemi_picks = np.where(locs[:, 0] <= 0 if hemi == "lh" else locs[:, 0] > 0)[0]
        # compute distances to pial vertices
        nearest, dists = _compute_nearest(
            surf_data[hemi]["pial"][0], locs[hemi_picks], return_dists=True
        )
        mask = dists / 1000 < max_dist
        proj_locs[hemi_picks[mask]] = surf_data[hemi][surf][0][nearest[mask]]
    # back to the "head" coordinate frame for storing in ``raw``
    proj_locs = apply_trans(invert_transform(trans), proj_locs)
    montage = info.get_montage()
    montage_kwargs = (
        montage.get_positions() if montage else dict(ch_pos=dict(), coord_frame="head")
    )
    for idx, loc in zip(picks_idx, proj_locs):
        montage_kwargs["ch_pos"][info.ch_names[idx]] = loc
    info.set_montage(make_dig_montage(**montage_kwargs))
    return info
