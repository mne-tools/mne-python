# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

from os import path as op

from itertools import combinations
import numpy as np

from ...io.pick import _picks_to_idx
from ...surface import _read_mri_surface
from ...transforms import (apply_trans, invert_transform, _cart_to_sph,
                           _ensure_trans)
from ...utils import verbose, get_subjects_dir, _validate_type, _ensure_int


@verbose
def project_sensors_onto_brain(info, trans, subject, subjects_dir=None,
                               picks=None, n_neighbors=10, copy=True,
                               verbose=None):
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
    from scipy.spatial.distance import pdist, squareform
    n_neighbors = _ensure_int(n_neighbors, 'n_neighbors')
    _validate_type(copy, bool, 'copy')
    if copy:
        info = info.copy()
    if n_neighbors < 2:
        raise ValueError(
            f'n_neighbors must be 2 or greater, got {n_neighbors}')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    try:
        surf = _read_mri_surface(op.join(
            subjects_dir, subject, 'bem', 'brain.surf'))
    except FileNotFoundError as err:
        raise RuntimeError(f'{err}\n\nThe brain surface requires generating '
                           'a BEM using `mne flash_bem` (if you have '
                           'the FLASH scan) or `mne watershed_bem` (to '
                           'use the T1)') from None
    # get channel locations
    picks_idx = _picks_to_idx(info, 'ecog' if picks is None else picks)
    locs = np.array([info['chs'][idx]['loc'][:3] for idx in picks_idx])
    trans = _ensure_trans(trans, 'head', 'mri')
    locs = apply_trans(trans, locs)
    # compute distances for nearest neighbors
    dists = squareform(pdist(locs))
    # find angles for brain surface and points
    angles = _cart_to_sph(locs)
    surf_angles = _cart_to_sph(surf['rr'])
    # initialize projected locs
    proj_locs = np.zeros(locs.shape) * np.nan
    for i, loc in enumerate(locs):
        neighbor_pts = locs[np.argsort(dists[i])[:n_neighbors + 1]]
        normals = list()
        for pt1, pt2, pt3 in combinations(neighbor_pts, 3):
            n = np.cross(pt1 - pt2, pt1 - pt3)
            # flip direction if needed
            normals.append(n if np.dot(n, loc) >= 0 else -n)
        normal = np.mean(normals, axis=0)
        normal /= np.linalg.norm(normal)
        # find the correct orientation brain surface point nearest the line
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        use_rr = surf['rr'][abs(
            surf_angles[:, 1:] - angles[i, 1:]).sum(axis=1) < np.pi / 4]
        surf_dists = [np.linalg.norm(np.cross(
            surf_pt - loc, surf_pt - loc + normal)) for surf_pt in use_rr]
        proj_locs[i] = use_rr[np.argmin(surf_dists)]
    # back to the "head" coordinate frame for storing in ``raw``
    proj_locs = apply_trans(invert_transform(trans), proj_locs)
    for idx, loc in zip(picks_idx, proj_locs):
        info['chs'][idx]['loc'][:3] = loc
    return info
