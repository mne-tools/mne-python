# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Many of the computations in this code were derived from Matti Hämäläinen's
# C code.

import json
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache, partial
from glob import glob
from os import path as op
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_array, csr_array
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

from ._fiff.constants import FIFF
from ._fiff.pick import pick_types
from .fixes import bincount, jit, prange
from .parallel import parallel_func
from .transforms import (
    Transform,
    _angle_between_quats,
    _cart_to_sph,
    _fit_matched_points,
    _get_trans,
    _MatchedDisplacementFieldInterpolator,
    _pol_to_cart,
    apply_trans,
    transform_surface_to,
)
from .utils import (
    _check_fname,
    _check_freesurfer_home,
    _check_option,
    _ensure_int,
    _hashable_ndarray,
    _import_nibabel,
    _pl,
    _soft_import,
    _TempDir,
    _validate_type,
    fill_doc,
    get_subjects_dir,
    logger,
    run_subprocess,
    verbose,
    warn,
)

_helmet_path = Path(__file__).parent / "data" / "helmets"


###############################################################################
# AUTOMATED SURFACE FINDING


@verbose
def get_head_surf(
    subject, source=("bem", "head"), subjects_dir=None, on_defects="raise", verbose=None
):
    """Load the subject head surface.

    Parameters
    ----------
    subject : str
        Subject name.
    source : str | list of str
        Type to load. Common choices would be ``'bem'`` or ``'head'``. We first
        try loading ``'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'``, and
        then look for ``'$SUBJECT*$SOURCE.fif'`` in the same directory by going
        through all files matching the pattern. The head surface will be read
        from the first file containing a head surface. Can also be a list
        to try multiple strings.
    subjects_dir : path-like | None
        Path to the ``SUBJECTS_DIR``. If None, the path is obtained by using
        the environment variable ``SUBJECTS_DIR``.
    %(on_defects)s

        .. versionadded:: 1.0
    %(verbose)s

    Returns
    -------
    surf : dict
        The head surface.
    """
    return _get_head_surface(
        subject=subject, source=source, subjects_dir=subjects_dir, on_defects=on_defects
    )


# TODO this should be refactored with mne._freesurfer._get_head_surface
def _get_head_surface(subject, source, subjects_dir, on_defects, raise_error=True):
    """Load the subject head surface."""
    from .bem import read_bem_surfaces

    # Load the head surface from the BEM
    subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=True))
    _validate_type(subject, str, "subject")
    # use realpath to allow for linked surfaces (c.f. MNE manual 196-197)
    if isinstance(source, str):
        source = [source]
    surf = None
    for this_source in source:
        this_head = op.realpath(
            op.join(subjects_dir, subject, "bem", f"{subject}-{this_source}.fif")
        )
        if op.exists(this_head):
            surf = read_bem_surfaces(
                this_head,
                True,
                FIFF.FIFFV_BEM_SURF_ID_HEAD,
                on_defects=on_defects,
                verbose=False,
            )
        else:
            # let's do a more sophisticated search
            path = op.join(subjects_dir, subject, "bem")
            if not op.isdir(path):
                raise OSError(f'Subject bem directory "{path}" does not exist.')
            files = sorted(glob(op.join(path, f"{subject}*{this_source}.fif")))
            for this_head in files:
                try:
                    surf = read_bem_surfaces(
                        this_head,
                        True,
                        FIFF.FIFFV_BEM_SURF_ID_HEAD,
                        on_defects=on_defects,
                        verbose=False,
                    )
                except ValueError:
                    pass
                else:
                    break
        if surf is not None:
            break

    if surf is None:
        if raise_error:
            raise OSError(
                f'No file matching "{subject}*{this_source}" and containing a head '
                "surface found."
            )
        else:
            return surf
    logger.info(f"Using surface from {this_head}.")
    return surf


# New helmets can be written for example with:
#
# import os.path as op
# import mne
# from mne.io.constants import FIFF
# surf = mne.read_surface('kernel.obj', return_dict=True)[-1]
# surf['rr'] *= 1000  # needs to be in mm
# mne.surface.complete_surface_info(surf, copy=False, do_neighbor_tri=False)
# surf['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
# surfs = mne.bem._surfaces_to_bem(
#     [surf], ids=[FIFF.FIFFV_MNE_SURF_MEG_HELMET], sigmas=[1.],
#     incomplete='ignore')
# del surfs[0]['sigma']
# bem_fname = op.join(op.dirname(mne.__file__), 'data', 'helmets',
#                     'kernel.fif.gz')
# mne.write_bem_surfaces(bem_fname, surfs, overwrite=True)


@verbose
def get_meg_helmet_surf(info, trans=None, *, upsampling=1, verbose=None):
    """Load the MEG helmet associated with the MEG sensors.

    Parameters
    ----------
    %(info_not_none)s
    trans : dict
        The head<->MRI transformation, usually obtained using
        read_trans(). Can be None, in which case the surface will
        be in head coordinates instead of MRI coordinates.
    %(helmet_upsampling)s
    %(verbose)s

    Returns
    -------
    surf : dict
        The MEG helmet as a surface.

    Notes
    -----
    A built-in helmet is loaded if possible. If not, a helmet surface
    will be approximated based on the sensor locations.
    """
    from .bem import _fit_sphere, read_bem_surfaces
    from .channels.channels import _get_meg_system

    _validate_type(upsampling, "int", "upsampling")

    system, have_helmet = _get_meg_system(info)
    incomplete = False
    if have_helmet:
        logger.info(f"Getting helmet for system {system}")
        fname = _helmet_path / f"{system}.fif.gz"
        surf = read_bem_surfaces(
            fname, False, FIFF.FIFFV_MNE_SURF_MEG_HELMET, verbose=False
        )
        surf = _scale_helmet_to_sensors(system, surf, info)
    else:
        rr = np.array(
            [
                info["chs"][pick]["loc"][:3]
                for pick in pick_types(info, meg=True, ref_meg=False, exclude=())
            ]
        )
        logger.info(
            "Getting helmet for system %s (derived from %d MEG channel locations)",
            system,
            len(rr),
        )
        hull = ConvexHull(rr)
        rr = rr[np.unique(hull.simplices)]
        R, center = _fit_sphere(rr)
        sph = _cart_to_sph(rr - center)[:, 1:]
        # add a point at the front of the helmet (where the face should be):
        # 90 deg az and maximal el (down from Z/up axis)
        front_sph = [[np.pi / 2.0, sph[:, 1].max()]]
        sph = np.concatenate((sph, front_sph))
        xy = _pol_to_cart(sph[:, ::-1])
        tris = Delaunay(xy).simplices
        # remove the frontal point we added from the simplices
        tris = tris[(tris != len(sph) - 1).all(-1)]
        tris = _reorder_ccw(rr, tris)
        surf = dict(rr=rr, tris=tris)
        incomplete = True
    if upsampling > 1:
        # Use VTK (could also use Butterfly but Loop is smoother)
        pv = _soft_import("pyvista", "upsample a mesh")
        factor = 4 ** (upsampling - 1)
        rr, tris = surf["rr"], surf["tris"]
        logger.info(
            f"Upsampling from {len(rr)} to {len(rr) * factor} vertices ({upsampling=})"
        )
        tris = np.c_[np.full(len(tris), 3), tris]
        mesh = pv.PolyData(rr, tris)
        mesh = mesh.subdivide(upsampling - 1, subfilter="linear")
        rr, tris = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        tris = _reorder_ccw(rr, tris)
        surf = dict(rr=rr, tris=tris)
        incomplete = True
    if incomplete:
        complete_surface_info(surf, copy=False, verbose=False)

    # Ignore what the file says, it's in device coords and we want MRI coords
    surf["coord_frame"] = FIFF.FIFFV_COORD_DEVICE
    dev_head_t = info["dev_head_t"]
    if dev_head_t is None:
        dev_head_t = Transform("meg", "head")
    transform_surface_to(surf, "head", dev_head_t)
    if trans is not None:
        transform_surface_to(surf, "mri", trans)
    return surf


def _scale_helmet_to_sensors(system, surf, info):
    fname = _helmet_path / f"{system}_ch_pos.txt"
    if not fname.is_file():
        return surf
    with open(fname) as fid:
        ch_pos_from = json.load(fid)
    # find correspondence
    fro, to = list(), list()
    for key, f_ in ch_pos_from.items():
        t_ = [ch["loc"][:3] for ch in info["chs"] if ch["ch_name"].startswith(key)]
        if not len(t_):
            continue
        fro.append(f_)
        to.append(np.mean(t_, axis=0))
    if len(fro) < 4:
        logger.info(
            "Using untransformed helmet, not enough sensors found to deform to match "
            f"acquisition based on sensor positions (got {len(fro)}, need at least 4)"
        )
        return surf
    fro = np.array(fro, float)
    to = np.array(to, float)
    delta = np.ptp(surf["rr"], axis=0) * 0.1  # 10% beyond bounds
    extrema = np.array([surf["rr"].min(0) - delta, surf["rr"].max(0) + delta])
    interp = _MatchedDisplacementFieldInterpolator(fro, to, extrema=extrema)
    new_rr = interp(surf["rr"])
    try:
        quat, sc = _fit_matched_points(surf["rr"], new_rr)
    except np.linalg.LinAlgError as exc:
        logger.info(
            f"Using untransformed helmet, deformation using {len(fro)} points "
            f"failed ({exc})"
        )
        return surf
    rot = np.rad2deg(_angle_between_quats(quat[:3]))
    tr = 1000 * np.linalg.norm(quat[3:])
    logger.info(
        f"    Deforming CAD helmet to match {len(fro)} acquisition sensor positions:"
    )
    logger.info(f"    1. Affine: {rot:0.1f}°, {tr:0.1f} mm, {sc:0.2f}× scale")
    deltas = interp._last_deltas * 1000
    mu, mx = np.mean(deltas), np.max(deltas)
    logger.info(f"    2. Nonlinear displacement: mean={mu:0.1f}, max={mx:0.1f} mm")
    surf["rr"] = new_rr
    complete_surface_info(surf, copy=False, verbose=False)
    return surf


def _reorder_ccw(rrs, tris):
    """Reorder tris of a convex hull to be wound counter-clockwise."""
    # This ensures that rendering with front-/back-face culling works properly
    com = np.mean(rrs, axis=0)
    rr_tris = rrs[tris]
    dirs = np.sign(
        (
            np.cross(rr_tris[:, 1] - rr_tris[:, 0], rr_tris[:, 2] - rr_tris[:, 0])
            * (rr_tris[:, 0] - com)
        ).sum(-1)
    ).astype(int)
    return np.array([t[::d] for d, t in zip(dirs, tris)])


###############################################################################
# EFFICIENCY UTILITIES


def fast_cross_3d(x, y):
    """Compute cross product between list of 3D vectors.

    Much faster than np.cross() when the number of cross products
    becomes large (>= 500). This is because np.cross() methods become
    less memory efficient at this stage.

    Parameters
    ----------
    x : array
        Input array 1, shape (..., 3).
    y : array
        Input array 2, shape (..., 3).

    Returns
    -------
    z : array, shape (..., 3)
        Cross product of x and y along the last dimension.

    Notes
    -----
    x and y must broadcast against each other.
    """
    assert x.ndim >= 1
    assert y.ndim >= 1
    assert x.shape[-1] == 3
    assert y.shape[-1] == 3
    if max(x.size, y.size) >= 500:
        out = np.empty(np.broadcast(x, y).shape)
        _jit_cross(out, x, y)
        return out
    else:
        return np.cross(x, y)


@jit()
def _jit_cross(out, x, y):
    out[..., 0] = x[..., 1] * y[..., 2]
    out[..., 0] -= x[..., 2] * y[..., 1]
    out[..., 1] = x[..., 2] * y[..., 0]
    out[..., 1] -= x[..., 0] * y[..., 2]
    out[..., 2] = x[..., 0] * y[..., 1]
    out[..., 2] -= x[..., 1] * y[..., 0]


@jit()
def _fast_cross_nd_sum(a, b, c):
    """Fast cross and sum."""
    return (
        (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0]
        + (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1]
        + (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2]
    )


@jit()
def _accumulate_normals(tris, tri_nn, npts):
    """Efficiently accumulate triangle normals."""
    # this code replaces the following, but is faster (vectorized):
    #
    # this['nn'] = np.zeros((this['np'], 3))
    # for p in xrange(this['ntri']):
    #     verts = this['tris'][p]
    #     this['nn'][verts, :] += this['tri_nn'][p, :]
    #
    nn = np.zeros((npts, 3))
    for vi in range(3):
        verts = tris[:, vi]
        for idx in range(3):  # x, y, z
            nn[:, idx] += bincount(verts, weights=tri_nn[:, idx], minlength=npts)
    return nn


def _triangle_neighbors(tris, npts):
    """Efficiently compute vertex neighboring triangles."""
    # this code replaces the following, but is faster (vectorized):
    # neighbor_tri = [list() for _ in range(npts)]
    # for ti, tri in enumerate(tris):
    #     for t in tri:
    #         neighbor_tri[t].append(ti)
    rows = tris.ravel()
    cols = np.repeat(np.arange(len(tris)), 3)
    data = np.ones(len(cols))
    csr = coo_array((data, (rows, cols)), shape=(npts, len(tris))).tocsr()
    neighbor_tri = [
        csr.indices[start:stop] for start, stop in zip(csr.indptr[:-1], csr.indptr[1:])
    ]
    assert len(neighbor_tri) == npts
    return neighbor_tri


@jit()
def _triangle_coords(r, best, r1, nn, r12, r13, a, b, c):  # pragma: no cover
    """Get coordinates of a vertex projected to a triangle."""
    r1 = r1[best]
    tri_nn = nn[best]
    r12 = r12[best]
    r13 = r13[best]
    a = a[best]
    b = b[best]
    c = c[best]
    rr = r - r1
    z = np.sum(rr * tri_nn)
    v1 = np.sum(rr * r12)
    v2 = np.sum(rr * r13)
    det = a * b - c * c
    x = (b * v1 - c * v2) / det
    y = (a * v2 - c * v1) / det
    return x, y, z


def _project_onto_surface(
    rrs, surf, project_rrs=False, return_nn=False, method="accurate"
):
    """Project points onto (scalp) surface."""
    if method == "accurate":
        surf_geom = _get_tri_supp_geom(surf)
        pt_tris = np.empty((0,), int)
        pt_lens = np.zeros(len(rrs) + 1, int)
        out = _find_nearest_tri_pts(rrs, pt_tris, pt_lens, reproject=True, **surf_geom)
        if project_rrs:  #
            out += (np.einsum("ij,ijk->ik", out[0], surf["rr"][surf["tris"][out[1]]]),)
        if return_nn:
            out += (surf_geom["nn"][out[1]],)
    else:  # nearest neighbor
        assert project_rrs
        idx = _compute_nearest(surf["rr"], rrs)
        out = (None, None, surf["rr"][idx])
        if return_nn:
            surf_geom = _get_tri_supp_geom(surf)
            nn = _accumulate_normals(
                surf["tris"].astype(int), surf_geom["nn"], len(surf["rr"])
            )
            nn = nn[idx]
            _normalize_vectors(nn)
            out += (nn,)
    return out


def _normal_orth(nn):
    """Compute orthogonal basis given a normal."""
    assert nn.shape[-1:] == (3,)
    prod = np.einsum("...i,...j->...ij", nn, nn)
    _, u = np.linalg.eigh(np.eye(3) - prod)
    u = u[..., ::-1]
    #  Make sure that ez is in the direction of nn
    signs = np.sign(np.matmul(nn[..., np.newaxis, :], u[..., -1:]))
    signs[signs == 0] = 1
    u *= signs
    return u.swapaxes(-1, -2)


@verbose
def complete_surface_info(
    surf, do_neighbor_vert=False, copy=True, do_neighbor_tri=True, *, verbose=None
):
    """Complete surface information.

    Parameters
    ----------
    surf : dict
        The surface.
    do_neighbor_vert : bool
        If True (default False), add neighbor vertex information.
    copy : bool
        If True (default), make a copy. If False, operate in-place.
    do_neighbor_tri : bool
        If True (default), compute triangle neighbors.
    %(verbose)s

    Returns
    -------
    surf : dict
        The transformed surface.
    """
    if copy:
        surf = deepcopy(surf)
    # based on mne_source_space_add_geometry_info() in mne_add_geometry_info.c

    #   Main triangulation [mne_add_triangle_data()]
    surf["ntri"] = surf.get("ntri", len(surf["tris"]))
    surf["np"] = surf.get("np", len(surf["rr"]))
    surf["tri_area"] = np.zeros(surf["ntri"])
    r1 = surf["rr"][surf["tris"][:, 0], :]
    r2 = surf["rr"][surf["tris"][:, 1], :]
    r3 = surf["rr"][surf["tris"][:, 2], :]
    surf["tri_cent"] = (r1 + r2 + r3) / 3.0
    surf["tri_nn"] = fast_cross_3d((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    surf["tri_area"] = _normalize_vectors(surf["tri_nn"]) / 2.0
    zidx = np.where(surf["tri_area"] == 0)[0]
    if len(zidx) > 0:
        logger.info(f"    Warning: zero size triangles: {zidx}")

    #    Find neighboring triangles, accumulate vertex normals, normalize
    logger.info("    Triangle neighbors and vertex normals...")
    surf["nn"] = _accumulate_normals(
        surf["tris"].astype(int), surf["tri_nn"], surf["np"]
    )
    _normalize_vectors(surf["nn"])

    #   Check for topological defects
    if do_neighbor_tri:
        surf["neighbor_tri"] = _triangle_neighbors(surf["tris"], surf["np"])
        zero, fewer = list(), list()
        for ni, n in enumerate(surf["neighbor_tri"]):
            if len(n) < 3:
                if len(n) == 0:
                    zero.append(ni)
                else:
                    fewer.append(ni)
                    surf["neighbor_tri"][ni] = np.array([], int)
        if len(zero) > 0:
            logger.info(
                "    Vertices do not have any neighboring triangles: "
                f"[{', '.join(str(z) for z in zero)}]"
            )
        if len(fewer) > 0:
            fewer = ", ".join(str(f) for f in fewer)
            logger.info(
                "    Vertices have fewer than three neighboring triangles, removing "
                f"neighbors: [{fewer}]"
            )

    #   Determine the neighboring vertices and fix errors
    if do_neighbor_vert is True:
        logger.info("    Vertex neighbors...")
        surf["neighbor_vert"] = [
            _get_surf_neighbors(surf, k) for k in range(surf["np"])
        ]

    return surf


def _get_surf_neighbors(surf, k):
    """Calculate the surface neighbors based on triangulation."""
    verts = set()
    for v in surf["tris"][surf["neighbor_tri"][k]].flat:
        verts.add(v)
    verts.remove(k)
    verts = np.array(sorted(verts))
    assert np.all(verts < surf["np"])
    nneighbors = len(verts)
    nneigh_max = len(surf["neighbor_tri"][k])
    if nneighbors > nneigh_max:
        raise RuntimeError(f"Too many neighbors for vertex {k}.")
    elif nneighbors != nneigh_max:
        logger.info(
            "    Incorrect number of distinct neighbors for vertex"
            " %d (%d instead of %d) [fixed].",
            k,
            nneighbors,
            nneigh_max,
        )
    return verts


def _normalize_vectors(rr):
    """Normalize surface vertices."""
    size = np.linalg.norm(rr, axis=1)
    mask = size > 0
    rr[mask] /= size[mask, np.newaxis]  # operate in-place
    return size


class _CDist:
    """Wrapper for cdist that uses a Tree-like pattern."""

    def __init__(self, xhs):
        self._xhs = xhs

    def query(self, rr):
        nearest = list()
        dists = list()
        for r in rr:
            d = cdist(r[np.newaxis, :], self._xhs)
            idx = np.argmin(d)
            nearest.append(idx)
            dists.append(d[0, idx])
        return np.array(dists), np.array(nearest)


def _compute_nearest(xhs, rr, method="BallTree", return_dists=False):
    """Find nearest neighbors.

    Parameters
    ----------
    xhs : array, shape=(n_samples, n_dim)
        Points of data set.
    rr : array, shape=(n_query, n_dim)
        Points to find nearest neighbors for.
    method : str
        The query method. If scikit-learn and scipy<1.0 are installed,
        it will fall back to the slow brute-force search.
    return_dists : bool
        If True, return associated distances.

    Returns
    -------
    nearest : array, shape=(n_query,)
        Index of nearest neighbor in xhs for every point in rr.
    distances : array, shape=(n_query,)
        The distances. Only returned if return_dists is True.
    """
    if xhs.size == 0 or rr.size == 0:
        if return_dists:
            return np.array([], int), np.array([])
        return np.array([], int)
    tree = _DistanceQuery(xhs, method=method)
    out = tree.query(rr)
    return out[::-1] if return_dists else out[1]


def _safe_query(rr, func, reduce=False, **kwargs):
    if len(rr) == 0:
        return np.array([]), np.array([], int)
    out = func(rr)
    out = [out[0][:, 0], out[1][:, 0]] if reduce else out
    return out


class _DistanceQuery:
    """Wrapper for fast distance queries."""

    def __init__(self, xhs, method="BallTree"):
        assert method in ("BallTree", "KDTree", "cdist")

        # Fastest for our problems: balltree
        if method == "BallTree":
            try:
                from sklearn.neighbors import BallTree
            except ImportError:
                logger.info(
                    "Nearest-neighbor searches will be significantly "
                    "faster if scikit-learn is installed."
                )
                method = "KDTree"
            else:
                self.query = partial(
                    _safe_query,
                    func=BallTree(xhs).query,
                    reduce=True,
                    return_distance=True,
                )

        # Then KDTree
        if method == "KDTree":
            from scipy.spatial import KDTree

            self.query = KDTree(xhs).query

        # Then the worst: cdist
        if method == "cdist":
            self.query = _CDist(xhs).query

        self.data = xhs


@verbose
def _points_outside_surface(rr, surf, n_jobs=None, verbose=None):
    """Check whether points are outside a surface.

    Parameters
    ----------
    rr : ndarray
        Nx3 array of points to check.
    surf : dict
        Surface with entries "rr" and "tris".

    Returns
    -------
    outside : ndarray
        1D logical array of size N for which points are outside the surface.
    """
    rr = np.atleast_2d(rr)
    assert rr.shape[1] == 3
    parallel, p_fun, n_jobs = parallel_func(_get_solids, n_jobs)
    tot_angles = parallel(
        p_fun(surf["rr"][tris], rr) for tris in np.array_split(surf["tris"], n_jobs)
    )
    return np.abs(np.sum(tot_angles, axis=0) / (2 * np.pi) - 1.0) > 1e-5


def _surface_to_polydata(surf):
    import pyvista as pv

    vertices = np.array(surf["rr"])
    if "tris" not in surf:
        return pv.PolyData(vertices)
    else:
        triangles = np.array(surf["tris"])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        return pv.PolyData(vertices, triangles)


def _polydata_to_surface(pd, normals=True):
    from pyvista import PolyData

    if not isinstance(pd, PolyData):
        pd = PolyData(pd)
    out = dict(rr=pd.points, tris=pd.faces.reshape(-1, 4)[:, 1:])
    if normals:
        out["nn"] = pd.point_normals
    return out


class _CheckInside:
    """Efficiently check if points are inside a surface."""

    @verbose
    def __init__(self, surf, *, mode="old", verbose=None):
        assert mode in ("pyvista", "old")
        self.mode = mode
        t0 = time.time()
        self.surf = surf
        if self.mode == "pyvista":
            self._init_pyvista()
        else:
            self._init_old()
        logger.debug(
            f"Setting up {mode} interior check for {len(self.surf['rr'])} "
            f"points took {(time.time() - t0) * 1000:0.1f} ms"
        )

    def _init_old(self):
        self.inner_r = None
        self.cm = self.surf["rr"].mean(0)
        # We could use Delaunay or ConvexHull here, Delaunay is slightly slower
        # to construct but faster to evaluate
        # See https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl  # noqa
        self.del_tri = Delaunay(self.surf["rr"])
        if self.del_tri.find_simplex(self.cm) >= 0:
            # Immediately cull some points from the checks
            dists = np.linalg.norm(self.surf["rr"] - self.cm, axis=-1)
            self.inner_r = dists.min()
            self.outer_r = dists.max()

    def _init_pyvista(self):
        if not isinstance(self.surf, dict):
            self.pdata = self.surf
            self.surf = _polydata_to_surface(self.pdata)
        else:
            self.pdata = _surface_to_polydata(self.surf).clean()

    @verbose
    def __call__(self, rr, n_jobs=None, verbose=None):
        n_orig = len(rr)
        logger.info(
            f"Checking surface interior status for {n_orig} point{_pl(n_orig, ' ')}..."
        )
        t0 = time.time()
        if self.mode == "pyvista":
            inside = self._call_pyvista(rr)
        else:
            inside = self._call_old(rr, n_jobs)
        n = inside.sum()
        logger.info(f"    Total {n}/{n_orig} point{_pl(n, ' ')} inside the surface")
        logger.info(f"Interior check completed in {(time.time() - t0) * 1000:0.1f} ms")
        return inside

    def _call_pyvista(self, rr):
        pdata = _surface_to_polydata(dict(rr=rr))
        out = pdata.select_enclosed_points(self.pdata, check_surface=False)
        return out["SelectedPoints"].astype(bool)

    def _call_old(self, rr, n_jobs):
        n_orig = len(rr)
        prec = int(np.ceil(np.log10(max(n_orig, 10))))
        inside = np.ones(n_orig, bool)  # innocent until proven guilty
        idx = np.arange(n_orig)
        # Limit to indices that can plausibly be outside the surf
        # but are not definitely outside it
        if self.inner_r is not None:
            dists = np.linalg.norm(rr - self.cm, axis=-1)
            in_mask = dists < self.inner_r
            n = (in_mask).sum()
            n_pad = str(n).rjust(prec)
            logger.info(
                f"    Found {n_pad}/{n_orig} point{_pl(n, ' ')} "
                f"inside  an interior sphere of radius "
                f"{1000 * self.inner_r:6.1f} mm"
            )
            out_mask = dists > self.outer_r
            inside[out_mask] = False
            n = (out_mask).sum()
            n_pad = str(n).rjust(prec)
            logger.info(
                f"    Found {n_pad}/{n_orig} point{_pl(n, ' ')} "
                f"outside an exterior sphere of radius "
                f"{1000 * self.outer_r:6.1f} mm"
            )
            mask = (~in_mask) & (~out_mask)  # not definitely inside or outside
            idx = idx[mask]
            rr = rr[mask]

        # Use qhull as our first pass (*much* faster than our check)
        del_outside = self.del_tri.find_simplex(rr) < 0
        n = sum(del_outside)
        inside[idx[del_outside]] = False
        idx = idx[~del_outside]
        rr = rr[~del_outside]
        n_pad = str(n).rjust(prec)
        check_pad = str(len(del_outside)).rjust(prec)
        logger.info(
            f"    Found {n_pad}/{check_pad} point{_pl(n, ' ')} outside using "
            "surface Qhull"
        )

        # use our more accurate check
        solid_outside = _points_outside_surface(rr, self.surf, n_jobs)
        n = np.sum(solid_outside)
        n_pad = str(n).rjust(prec)
        check_pad = str(len(solid_outside)).rjust(prec)
        logger.info(
            f"    Found {n_pad}/{check_pad} point{_pl(n, ' ')} outside using "
            "solid angles"
        )
        inside[idx[solid_outside]] = False
        return inside


###############################################################################
# Handle freesurfer


def _fread3(fobj):
    """Read 3 bytes and adjust."""
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3).astype(np.int64)
    return (b1 << 16) + (b2 << 8) + b3


def read_curvature(filepath, binary=True):
    """Load in curvature values from the ?h.curv file.

    Parameters
    ----------
    filepath : path-like
        Input path to the ``.curv`` file.
    binary : bool
        Specify if the output array is to hold binary values. Defaults to True.

    Returns
    -------
    curv : array of shape (n_vertices,)
        The curvature values loaded from the user given file.
    """
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            _fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
    if binary:
        return 1 - np.array(curv != 0, np.int64)
    else:
        return curv


@verbose
def read_surface(
    fname, read_metadata=False, return_dict=False, file_format="auto", verbose=None
):
    """Load a Freesurfer surface mesh in triangular format.

    Parameters
    ----------
    fname : path-like
        The name of the file containing the surface.
    read_metadata : bool
        Read metadata as key-value pairs. Only works when reading a FreeSurfer
        surface file. For .obj files this dictionary will be empty.

        Valid keys:

            * 'head' : array of int
            * 'valid' : str
            * 'filename' : str
            * 'volume' : array of int, shape (3,)
            * 'voxelsize' : array of float, shape (3,)
            * 'xras' : array of float, shape (3,)
            * 'yras' : array of float, shape (3,)
            * 'zras' : array of float, shape (3,)
            * 'cras' : array of float, shape (3,)

        .. versionadded:: 0.13.0

    return_dict : bool
        If True, a dictionary with surface parameters is returned.
    file_format : 'auto' | 'freesurfer' | 'obj'
        File format to use. Can be 'freesurfer' to read a FreeSurfer surface
        file, or 'obj' to read a Wavefront .obj file (common format for
        importing in other software), or 'auto' to attempt to infer from the
        file name. Defaults to 'auto'.

        .. versionadded:: 0.21.0
    %(verbose)s

    Returns
    -------
    rr : array, shape=(n_vertices, 3)
        Coordinate points.
    tris : int array, shape=(n_faces, 3)
        Triangulation (each line contains indices for three points which
        together form a face).
    volume_info : dict-like
        If read_metadata is true, key-value pairs found in the geometry file.
    surf : dict
        The surface parameters. Only returned if ``return_dict`` is True.

    See Also
    --------
    write_surface
    read_tri
    """
    fname = _check_fname(fname, "read", True)
    _check_option("file_format", file_format, ["auto", "freesurfer", "obj"])

    if file_format == "auto":
        if fname.suffix.lower() == ".obj":
            file_format = "obj"
        else:
            file_format = "freesurfer"

    if file_format == "freesurfer":
        _import_nibabel("read surface geometry")
        from nibabel.freesurfer import read_geometry

        ret = read_geometry(fname, read_metadata=read_metadata)
    elif file_format == "obj":
        ret = _read_wavefront_obj(fname)
        if read_metadata:
            ret += (dict(),)

    if return_dict:
        ret += (_rr_tris_dict(ret[0], ret[1]),)
    return ret


def _rr_tris_dict(rr, tris):
    return dict(rr=rr, tris=tris, ntri=len(tris), use_tris=tris, np=len(rr))


def _read_mri_surface(fname):
    surf = read_surface(fname, return_dict=True)[2]
    surf["rr"] /= 1000.0
    surf.update(coord_frame=FIFF.FIFFV_COORD_MRI)
    return surf


def _read_wavefront_obj(fname):
    """Read a surface form a Wavefront .obj file.

    Parameters
    ----------
    fname : str
        Name of the .obj file to read.

    Returns
    -------
    coords : ndarray, shape (n_points, 3)
        The XYZ coordinates of each vertex.
    faces : ndarray, shape (n_faces, 3)
        For each face of the mesh, the integer indices of the vertices that
        make up the face.
    """
    coords = []
    faces = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            split = line.split()
            if split[0] == "v":  # vertex
                coords.append([float(item) for item in split[1:]])
            elif split[0] == "f":  # face
                dat = [int(item.split("/")[0]) for item in split[1:]]
                if len(dat) != 3:
                    raise RuntimeError("Only triangle faces allowed.")
                # In .obj files, indexing starts at 1
                faces.append([d - 1 for d in dat])
    return np.array(coords), np.array(faces)


def _read_patch(fname):
    """Load a FreeSurfer binary patch file.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    rrs : ndarray, shape (n_vertices, 3)
        The points.
    tris : ndarray, shape (n_tris, 3)
        The patches. Not all vertices will be present.
    """
    # This is adapted from PySurfer PR #269, Bruce Fischl's read_patch.m,
    # and PyCortex (BSD)
    patch = dict()
    with open(fname) as fid:
        ver = np.fromfile(fid, dtype=">i4", count=1).item()
        if ver != -1:
            raise RuntimeError(f"incorrect version # {ver} (not -1) found")
        npts = np.fromfile(fid, dtype=">i4", count=1).item()
        dtype = np.dtype([("vertno", ">i4"), ("x", ">f"), ("y", ">f"), ("z", ">f")])
        recs = np.fromfile(fid, dtype=dtype, count=npts)
    # numpy to dict
    patch = {key: recs[key] for key in dtype.fields.keys()}
    patch["vertno"] -= 1

    # read surrogate surface
    rrs, tris = read_surface(
        op.join(op.dirname(fname), op.basename(fname)[:3] + "sphere")
    )
    orig_tris = tris
    is_vert = patch["vertno"] > 0  # negative are edges, ignored for now
    verts = patch["vertno"][is_vert]

    # eliminate invalid tris and zero out unused rrs
    mask = np.zeros((len(rrs),), dtype=bool)
    mask[verts] = True
    rrs[~mask] = 0.0
    tris = tris[mask[tris].all(1)]
    for ii, key in enumerate(["x", "y", "z"]):
        rrs[verts, ii] = patch[key][is_vert]
    return rrs, tris, orig_tris


##############################################################################
# SURFACE CREATION


def _get_ico_surface(grade, patch_stats=False):
    """Return an icosahedral surface of the desired grade."""
    # always use verbose=False since users don't need to know we're pulling
    # these from a file
    from .bem import read_bem_surfaces

    ico_file_name = op.join(op.dirname(__file__), "data", "icos.fif.gz")
    ico = read_bem_surfaces(
        ico_file_name, patch_stats, s_id=9000 + grade, verbose=False
    )
    return ico


def _tessellate_sphere_surf(level, rad=1.0):
    """Return a surface structure instead of the details."""
    rr, tris = _tessellate_sphere(level)
    npt = len(rr)  # called "npt" instead of "np" because of numpy...
    ntri = len(tris)
    nn = rr.copy()
    rr *= rad
    s = dict(
        rr=rr,
        np=npt,
        tris=tris,
        use_tris=tris,
        ntri=ntri,
        nuse=npt,
        nn=nn,
        inuse=np.ones(npt, int),
    )
    return s


def _norm_midpt(ai, bi, rr):
    """Get normalized midpoint."""
    c = rr[ai]
    c += rr[bi]
    _normalize_vectors(c)
    return c


def _tessellate_sphere(mylevel):
    """Create a tessellation of a unit sphere."""
    # Vertices of a unit octahedron
    rr = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],  # xplus, xminus
            [0, 1, 0],
            [0, -1, 0],  # yplus, yminus
            [0, 0, 1],
            [0, 0, -1],
        ],
        float,
    )  # zplus, zminus
    tris = np.array(
        [
            [0, 4, 2],
            [2, 4, 1],
            [1, 4, 3],
            [3, 4, 0],
            [0, 2, 5],
            [2, 1, 5],
            [1, 3, 5],
            [3, 0, 5],
        ],
        int,
    )

    # A unit octahedron
    if mylevel < 1:
        raise ValueError("oct subdivision must be >= 1")

    # Reverse order of points in each triangle
    # for counter-clockwise ordering
    tris = tris[:, [2, 1, 0]]

    # Subdivide each starting triangle (mylevel - 1) times
    for _ in range(1, mylevel):
        r"""
        Subdivide each triangle in the old approximation and normalize
        the new points thus generated to lie on the surface of the unit
        sphere.

        Each input triangle with vertices labelled [0,1,2] as shown
        below will be turned into four new triangles:

                             Make new points
                             a = (0+2)/2
                             b = (0+1)/2
                             c = (1+2)/2
                 1
                /\           Normalize a, b, c
               /  \
             b/____\c        Construct new triangles
             /\    /\        [0,b,a]
            /  \  /  \       [b,1,c]
           /____\/____\      [a,b,c]
          0     a      2     [a,c,2]

        """
        # use new method: first make new points (rr)
        a = _norm_midpt(tris[:, 0], tris[:, 2], rr)
        b = _norm_midpt(tris[:, 0], tris[:, 1], rr)
        c = _norm_midpt(tris[:, 1], tris[:, 2], rr)
        lims = np.cumsum([len(rr), len(a), len(b), len(c)])
        aidx = np.arange(lims[0], lims[1])
        bidx = np.arange(lims[1], lims[2])
        cidx = np.arange(lims[2], lims[3])
        rr = np.concatenate((rr, a, b, c))

        # now that we have our points, make new triangle definitions
        tris = np.array(
            (
                np.c_[tris[:, 0], bidx, aidx],
                np.c_[bidx, tris[:, 1], cidx],
                np.c_[aidx, bidx, cidx],
                np.c_[aidx, cidx, tris[:, 2]],
            ),
            int,
        ).swapaxes(0, 1)
        tris = np.reshape(tris, (np.prod(tris.shape[:2]), 3))

    # Copy the resulting approximation into standard table
    rr_orig = rr
    rr = np.empty_like(rr)
    nnode = 0
    for k, tri in enumerate(tris):
        for j in range(3):
            coord = rr_orig[tri[j]]
            # this is faster than cdist (no need for sqrt)
            similarity = np.dot(rr[:nnode], coord)
            idx = np.where(similarity > 0.99999)[0]
            if len(idx) > 0:
                tris[k, j] = idx[0]
            else:
                rr[nnode] = coord
                tris[k, j] = nnode
                nnode += 1
    rr = rr[:nnode].copy()
    return rr, tris


def _create_surf_spacing(surf, hemi, subject, stype, ico_surf, subjects_dir):
    """Load a surf and use the subdivided icosahedron to get points."""
    # Based on load_source_space_surf_spacing() in load_source_space.c
    surf = read_surface(surf, return_dict=True)[-1]
    do_neighbor_vert = stype == "spacing"
    complete_surface_info(surf, do_neighbor_vert, copy=False)
    if stype == "all":
        surf["inuse"] = np.ones(surf["np"], int)
        surf["use_tris"] = None
    elif stype == "spacing":
        _decimate_surface_spacing(surf, ico_surf)
        surf["use_tris"] = None
        del surf["neighbor_vert"]
    else:  # ico or oct
        # ## from mne_ico_downsample.c ## #
        surf_name = subjects_dir / subject / "surf" / f"{hemi}.sphere"
        logger.info(f"Loading geometry from {surf_name}...")
        from_surf = read_surface(surf_name, return_dict=True)[-1]
        _normalize_vectors(from_surf["rr"])
        if from_surf["np"] != surf["np"]:
            raise RuntimeError(
                "Mismatch between number of surface vertices, "
                "possible parcellation error?"
            )
        _normalize_vectors(ico_surf["rr"])

        # Make the maps
        mmap = _compute_nearest(from_surf["rr"], ico_surf["rr"])
        nmap = len(mmap)
        surf["inuse"] = np.zeros(surf["np"], int)
        for k in range(nmap):
            if surf["inuse"][mmap[k]]:
                # Try the nearest neighbors
                neigh = _get_surf_neighbors(surf, mmap[k])
                was = mmap[k]
                inds = np.where(np.logical_not(surf["inuse"][neigh]))[0]
                if len(inds) == 0:
                    raise RuntimeError(
                        f"Could not find neighbor for vertex {k} / {nmap}."
                    )
                else:
                    mmap[k] = neigh[inds[-1]]
                logger.info(
                    "    Source space vertex moved from %d to %d "
                    "because of double occupation",
                    was,
                    mmap[k],
                )
            elif mmap[k] < 0 or mmap[k] > surf["np"]:
                raise RuntimeError(
                    f"Map number out of range ({mmap[k]}), this is probably due to "
                    "inconsistent surfaces. Parts of the FreeSurfer reconstruction "
                    "need to be redone."
                )
            surf["inuse"][mmap[k]] = True

        logger.info("Setting up the triangulation for the decimated surface...")
        surf["use_tris"] = np.array([mmap[ist] for ist in ico_surf["tris"]], np.int32)
    if surf["use_tris"] is not None:
        surf["nuse_tri"] = len(surf["use_tris"])
    else:
        surf["nuse_tri"] = 0
    surf["nuse"] = np.sum(surf["inuse"])
    surf["vertno"] = np.where(surf["inuse"])[0]

    # set some final params
    sizes = _normalize_vectors(surf["nn"])
    surf["inuse"][sizes <= 0] = False
    surf["nuse"] = np.sum(surf["inuse"])
    surf["subject_his_id"] = subject
    return surf


def _decimate_surface_spacing(surf, spacing):
    assert isinstance(spacing, int)
    assert spacing > 0
    logger.info("    Decimating...")
    d = np.full(surf["np"], 10000, int)

    # A mysterious algorithm follows
    for k in range(surf["np"]):
        neigh = surf["neighbor_vert"][k]
        d[k] = min(np.min(d[neigh]) + 1, d[k])
        if d[k] >= spacing:
            d[k] = 0
        d[neigh] = np.minimum(d[neigh], d[k] + 1)

    if spacing == 2.0:
        for k in range(surf["np"] - 1, -1, -1):
            for n in surf["neighbor_vert"][k]:
                d[k] = min(d[k], d[n] + 1)
                d[n] = min(d[n], d[k] + 1)
        for k in range(surf["np"]):
            if d[k] > 0:
                neigh = surf["neighbor_vert"][k]
                n = np.sum(d[neigh] == 0)
                if n <= 2:
                    d[k] = 0
                d[neigh] = np.minimum(d[neigh], d[k] + 1)

    surf["inuse"] = np.zeros(surf["np"], int)
    surf["inuse"][d == 0] = 1
    return surf


@verbose
def write_surface(
    fname,
    coords,
    faces,
    create_stamp="",
    volume_info=None,
    file_format="auto",
    overwrite=False,
    *,
    verbose=None,
):
    """Write a triangular Freesurfer surface mesh.

    Accepts the same data format as is returned by read_surface().

    Parameters
    ----------
    fname : path-like
        File to write.
    coords : array, shape=(n_vertices, 3)
        Coordinate points.
    faces : int array, shape=(n_faces, 3)
        Triangulation (each line contains indices for three points which
        together form a face).
    create_stamp : str
        Comment that is written to the beginning of the file. Can not contain
        line breaks.
    volume_info : dict-like or None
        Key-value pairs to encode at the end of the file.
        Valid keys:

            * 'head' : array of int
            * 'valid' : str
            * 'filename' : str
            * 'volume' : array of int, shape (3,)
            * 'voxelsize' : array of float, shape (3,)
            * 'xras' : array of float, shape (3,)
            * 'yras' : array of float, shape (3,)
            * 'zras' : array of float, shape (3,)
            * 'cras' : array of float, shape (3,)

        .. versionadded:: 0.13.0
    file_format : 'auto' | 'freesurfer' | 'obj'
        File format to use. Can be 'freesurfer' to write a FreeSurfer surface
        file, or 'obj' to write a Wavefront .obj file (common format for
        importing in other software), or 'auto' to attempt to infer from the
        file name. Defaults to 'auto'.

        .. versionadded:: 0.21.0
    %(overwrite)s
    %(verbose)s

    See Also
    --------
    read_surface
    read_tri
    """
    fname = _check_fname(fname, overwrite=overwrite)
    _check_option("file_format", file_format, ["auto", "freesurfer", "obj"])

    if file_format == "auto":
        if fname.suffix.lower() == ".obj":
            file_format = "obj"
        else:
            file_format = "freesurfer"

    if file_format == "freesurfer":
        _import_nibabel("write surface geometry")
        from nibabel.freesurfer import write_geometry

        write_geometry(
            fname, coords, faces, create_stamp=create_stamp, volume_info=volume_info
        )
    else:
        assert file_format == "obj"
        with open(fname, "w") as fid:
            for line in create_stamp.splitlines():
                fid.write(f"# {line}\n")
            for v in coords:
                fid.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in faces:
                fid.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


###############################################################################
# Decimation


def _decimate_surface_vtk(points, triangles, n_triangles):
    """Aux function."""
    try:
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkFiltersCore import vtkQuadricDecimation
    except ImportError:
        raise ValueError("This function requires the VTK package to be installed")
    if triangles.max() > len(points) - 1:
        raise ValueError(
            "The triangles refer to undefined points. Please check your mesh."
        )
    src = vtkPolyData()
    vtkpoints = vtkPoints()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        vtkpoints.SetData(numpy_to_vtk(points.astype(np.float64)))
    src.SetPoints(vtkpoints)
    vtkcells = vtkCellArray()
    triangles_ = np.pad(triangles, ((0, 0), (1, 0)), "constant", constant_values=3)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        idarr = numpy_to_vtkIdTypeArray(triangles_.ravel().astype(np.int64))
    vtkcells.SetCells(triangles.shape[0], idarr)
    src.SetPolys(vtkcells)
    # vtkDecimatePro was not very good, even with SplittingOff and
    # PreserveTopologyOn
    decimate = vtkQuadricDecimation()
    decimate.VolumePreservationOn()
    decimate.SetInputData(src)
    reduction = 1 - (float(n_triangles) / len(triangles))
    decimate.SetTargetReduction(reduction)
    decimate.Update()

    out = _polydata_to_surface(decimate.GetOutput(), normals=False)
    return out["rr"], out["tris"]


def _decimate_surface_sphere(rr, tris, n_triangles):
    _check_freesurfer_home()
    map_ = {}
    ico_levels = [20, 80, 320, 1280, 5120, 20480]
    map_.update({n_tri: ("ico", ii) for ii, n_tri in enumerate(ico_levels)})
    oct_levels = 2 ** (2 * np.arange(7) + 3)
    map_.update({n_tri: ("oct", ii) for ii, n_tri in enumerate(oct_levels, 1)})
    _check_option(
        "n_triangles", n_triangles, sorted(map_), extra=' when method="sphere"'
    )
    func_map = dict(ico=_get_ico_surface, oct=_tessellate_sphere_surf)
    kind, level = map_[n_triangles]
    logger.info(f"Decimating using Freesurfer spherical {kind}{level} downsampling")
    ico_surf = func_map[kind](level)
    assert len(ico_surf["tris"]) == n_triangles
    tempdir = _TempDir()
    orig = op.join(tempdir, "lh.temp")
    write_surface(orig, rr, tris)
    logger.info("    Extracting main mesh component ...")
    run_subprocess(["mris_extract_main_component", orig, orig], verbose="error")
    logger.info("    Smoothing ...")
    smooth = orig + ".smooth"
    run_subprocess(["mris_smooth", "-nw", orig, smooth], verbose="error")
    logger.info("    Inflating ...")
    inflated = orig + ".inflated"
    run_subprocess(["mris_inflate", "-no-save-sulc", smooth, inflated], verbose="error")
    logger.info("    Sphere ...")
    qsphere = orig + ".qsphere"
    run_subprocess(["mris_sphere", "-q", inflated, qsphere], verbose="error")
    sphere_rr, _ = read_surface(qsphere)
    norms = np.linalg.norm(sphere_rr, axis=1, keepdims=True)
    sphere_rr /= norms
    idx = _compute_nearest(sphere_rr, ico_surf["rr"], method="KDTree")
    n_dup = len(idx) - len(np.unique(idx))
    if n_dup:
        raise RuntimeError(
            f"Could not reduce to {n_triangles} triangles using ico, "
            f"{n_dup}/{len(idx)} vertices were duplicates."
        )
    logger.info("[done]")
    return rr[idx], ico_surf["tris"]


@verbose
def decimate_surface(points, triangles, n_triangles, method="quadric", *, verbose=None):
    """Decimate surface data.

    Parameters
    ----------
    points : ndarray
        The surface to be decimated, a 3 x number of points array.
    triangles : ndarray
        The surface to be decimated, a 3 x number of triangles array.
    n_triangles : int
        The desired number of triangles.
    method : str
        Can be "quadric" or "sphere". "sphere" will inflate the surface to a
        sphere using Freesurfer and downsample to an icosahedral or
        octahedral mesh.

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    points : ndarray
        The decimated points.
    triangles : ndarray
        The decimated triangles.

    Notes
    -----
    **"quadric" mode**

    This requires VTK. If an odd target number was requested,
    the ``'decimation'`` algorithm used results in the
    next even number of triangles. For example a reduction request
    to 30001 triangles may result in 30000 triangles.

    **"sphere" mode**

    This requires Freesurfer to be installed and available in the
    environment. The destination number of triangles must be one of
    ``[20, 80, 320, 1280, 5120, 20480]`` for ico (0-5) downsampling or one of
    ``[8, 32, 128, 512, 2048, 8192, 32768]`` for oct (1-7) downsampling.

    This mode is slower, but could be more suitable for decimating meshes for
    BEM creation (recommended ``n_triangles=5120``) due to better topological
    property preservation.
    """
    n_triangles = _ensure_int(n_triangles)
    method_map = dict(quadric=_decimate_surface_vtk, sphere=_decimate_surface_sphere)
    _check_option("method", method, sorted(method_map))
    if n_triangles > len(triangles):
        raise ValueError(
            f"Requested n_triangles ({n_triangles}) exceeds number of "
            f"original triangles ({len(triangles)})"
        )
    return method_map[method](points, triangles, n_triangles)


###############################################################################
# Geometry


@jit()
def _get_tri_dist(p, q, p0, q0, a, b, c, dist):  # pragma: no cover
    """Get the distance to a triangle edge."""
    p1 = p - p0
    q1 = q - q0
    out = p1 * p1 * a
    out += q1 * q1 * b
    out += p1 * q1 * c
    out += dist * dist
    return np.sqrt(out)


def _get_tri_supp_geom(surf):
    """Create supplementary geometry information using tris and rrs."""
    r1 = surf["rr"][surf["tris"][:, 0], :]
    r12 = surf["rr"][surf["tris"][:, 1], :] - r1
    r13 = surf["rr"][surf["tris"][:, 2], :] - r1
    r1213 = np.ascontiguousarray(np.array([r12, r13]).swapaxes(0, 1))
    a = np.einsum("ij,ij->i", r12, r12)
    b = np.einsum("ij,ij->i", r13, r13)
    c = np.einsum("ij,ij->i", r12, r13)
    mat = np.ascontiguousarray(np.rollaxis(np.array([[b, -c], [-c, a]]), 2))
    norm = a * b - c * c
    norm[norm == 0] = 1.0  # avoid divide by zero
    mat /= norm[:, np.newaxis, np.newaxis]
    nn = fast_cross_3d(r12, r13)
    _normalize_vectors(nn)
    return dict(r1=r1, r12=r12, r13=r13, r1213=r1213, a=a, b=b, c=c, mat=mat, nn=nn)


@jit(parallel=True)
def _find_nearest_tri_pts(
    rrs,
    pt_triss,
    pt_lens,
    a,
    b,
    c,
    nn,
    r1,
    r12,
    r13,
    r1213,
    mat,
    run_all=True,
    reproject=False,
):  # pragma: no cover
    """Find nearest point mapping to a set of triangles.

    If run_all is False, if the point lies within a triangle, it stops.
    If run_all is True, edges of other triangles are checked in case
    those (somehow) are closer.
    """
    # The following dense code is equivalent to the following:
    #   rr = r1[pt_tris] - to_pts[ii]
    #   v1s = np.sum(rr * r12[pt_tris], axis=1)
    #   v2s = np.sum(rr * r13[pt_tris], axis=1)
    #   aas = a[pt_tris]
    #   bbs = b[pt_tris]
    #   ccs = c[pt_tris]
    #   dets = aas * bbs - ccs * ccs
    #   pp = (bbs * v1s - ccs * v2s) / dets
    #   qq = (aas * v2s - ccs * v1s) / dets
    #   pqs = np.array(pp, qq)

    weights = np.empty((len(rrs), 3))
    tri_idx = np.empty(len(rrs), np.int64)
    for ri in prange(len(rrs)):
        rr = np.reshape(rrs[ri], (1, 3))
        start, stop = pt_lens[ri : ri + 2]
        if start == stop == 0:  # use all
            drs = rr - r1
            tri_nn = nn
            mats = mat
            r1213s = r1213
            reindex = False
        else:
            pt_tris = pt_triss[start:stop]
            drs = rr - r1[pt_tris]
            tri_nn = nn[pt_tris]
            mats = mat[pt_tris]
            r1213s = r1213[pt_tris]
            reindex = True
        use = np.ones(len(drs), np.int64)
        pqs = np.empty((len(drs), 2))
        dists = np.empty(len(drs))
        dist = np.inf
        # make life easier for numba var typing
        p, q, pt = np.float64(0.0), np.float64(1.0), np.int64(0)
        found = False
        for ii in range(len(drs)):
            pqs[ii] = np.dot(mats[ii], np.dot(r1213s[ii], drs[ii]))
            dists[ii] = np.dot(drs[ii], tri_nn[ii])
            pp, qq = pqs[ii]
            if pp >= 0 and qq >= 0 and pp <= 1 and qq <= 1 and pp + qq < 1:
                found = True
                use[ii] = False
                if np.abs(dists[ii]) < np.abs(dist):
                    p, q, pt, dist = pp, qq, ii, dists[ii]
        # re-reference back to original numbers
        if found and reindex:
            pt = pt_tris[pt]

        if not found or run_all:
            # don't include ones that we might have found before
            # these are the ones that we want to check the sides of
            s = np.where(use)[0]
            # Tough: must investigate the sides
            if reindex:
                use_pt_tris = pt_tris[s].astype(np.int64)
            else:
                use_pt_tris = s.astype(np.int64)
            pp, qq, ptt, distt = _nearest_tri_edge(
                use_pt_tris, pqs[s], dists[s], a, b, c
            )
            if np.abs(distt) < np.abs(dist):
                p, q, pt, dist = pp, qq, ptt, distt
        w = (1 - p - q, p, q)
        if reproject:
            # Calculate a linear interpolation between the vertex values to
            # get coords of pt projected onto closest triangle
            coords = _triangle_coords(rr[0], pt, r1, nn, r12, r13, a, b, c)
            w = (1.0 - coords[0] - coords[1], coords[0], coords[1])
        weights[ri] = w
        tri_idx[ri] = pt
    return weights, tri_idx


@jit()
def _nearest_tri_edge(pt_tris, pqs, dist, a, b, c):  # pragma: no cover
    """Get nearest location from a point to the edge of a set of triangles."""
    # We might do something intelligent here. However, for now
    # it is ok to do it in the hard way
    aa = a[pt_tris]
    bb = b[pt_tris]
    cc = c[pt_tris]
    pp = pqs[:, 0]
    qq = pqs[:, 1]
    # Find the nearest point from a triangle:
    #   Side 1 -> 2
    p0 = np.minimum(np.maximum(pp + 0.5 * (qq * cc) / aa, 0.0), 1.0)
    q0 = np.zeros_like(p0)
    #   Side 2 -> 3
    t1 = 0.5 * ((2.0 * aa - cc) * (1.0 - pp) + (2.0 * bb - cc) * qq) / (aa + bb - cc)
    t1 = np.minimum(np.maximum(t1, 0.0), 1.0)
    p1 = 1.0 - t1
    q1 = t1
    #   Side 1 -> 3
    q2 = np.minimum(np.maximum(qq + 0.5 * (pp * cc) / bb, 0.0), 1.0)
    p2 = np.zeros_like(q2)

    # figure out which one had the lowest distance
    dist0 = _get_tri_dist(pp, qq, p0, q0, aa, bb, cc, dist)
    dist1 = _get_tri_dist(pp, qq, p1, q1, aa, bb, cc, dist)
    dist2 = _get_tri_dist(pp, qq, p2, q2, aa, bb, cc, dist)
    pp = np.concatenate((p0, p1, p2))
    qq = np.concatenate((q0, q1, q2))
    dists = np.concatenate((dist0, dist1, dist2))
    ii = np.argmin(np.abs(dists))
    p, q, pt, dist = pp[ii], qq[ii], pt_tris[ii % len(pt_tris)], dists[ii]
    return p, q, pt, dist


def mesh_edges(tris):
    """Return sparse matrix with edges as an adjacency matrix.

    Parameters
    ----------
    tris : array of shape [n_triangles x 3]
        The triangles.

    Returns
    -------
    edges : scipy.sparse.spmatrix
        The adjacency matrix.
    """
    tris = _hashable_ndarray(tris)
    return _mesh_edges(tris=tris)


@lru_cache(maxsize=10)
def _mesh_edges(tris=None):
    if np.max(tris) > len(np.unique(tris)):
        raise ValueError("Cannot compute adjacency on a selection of triangles.")

    npoints = np.max(tris) + 1
    ones_ntris = np.ones(3 * len(tris))

    a, b, c = tris.T
    x = np.concatenate((a, b, c))
    y = np.concatenate((b, c, a))
    edges = coo_array((ones_ntris, (x, y)), shape=(npoints, npoints))
    edges = edges.tocsr()
    edges = edges + edges.T
    return edges


def mesh_dist(tris, vert):
    """Compute adjacency matrix weighted by distances.

    It generates an adjacency matrix where the entries are the distances
    between neighboring vertices.

    Parameters
    ----------
    tris : array (n_tris x 3)
        Mesh triangulation.
    vert : array (n_vert x 3)
        Vertex locations.

    Returns
    -------
    dist_matrix : scipy.sparse.csr_array
        Sparse matrix with distances between adjacent vertices.
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.linalg.norm(vert[edges.row, :] - vert[edges.col, :], axis=1)
    dist_matrix = csr_array((dist, (edges.row, edges.col)), shape=edges.shape)
    return dist_matrix


@verbose
def read_tri(fname_in, swap=False, verbose=None):
    """Read triangle definitions from an ascii file.

    Parameters
    ----------
    fname_in : path-like
        Path to surface ASCII file (ending with ``'.tri'``).
    swap : bool
        Assume the ASCII file vertex ordering is clockwise instead of
        counterclockwise.
    %(verbose)s

    Returns
    -------
    rr : array, shape=(n_vertices, 3)
        Coordinate points.
    tris : int array, shape=(n_faces, 3)
        Triangulation (each line contains indices for three points which
        together form a face).

    See Also
    --------
    read_surface
    write_surface

    Notes
    -----
    .. versionadded:: 0.13.0
    """
    with open(fname_in) as fid:
        lines = fid.readlines()
    n_nodes = int(lines[0])
    n_tris = int(lines[n_nodes + 1])
    n_items = len(lines[1].split())
    if n_items in [3, 6, 14, 17]:
        inds = range(3)
    elif n_items in [4, 7]:
        inds = range(1, 4)
    else:
        raise OSError("Unrecognized format of data.")
    rr = np.array(
        [
            np.array([float(v) for v in line.split()])[inds]
            for line in lines[1 : n_nodes + 1]
        ]
    )
    tris = np.array(
        [
            np.array([int(v) for v in line.split()])[inds]
            for line in lines[n_nodes + 2 : n_nodes + 2 + n_tris]
        ]
    )
    if swap:
        tris[:, [2, 1]] = tris[:, [1, 2]]
    tris -= 1
    logger.info(
        f"Loaded surface from {fname_in} with {n_nodes} nodes and {n_tris} triangles."
    )
    if n_items in [3, 4]:
        logger.info("Node normals were not included in the source file.")
    else:
        warn("Node normals were not read.")
    return (rr, tris)


@jit()
def _get_solids(tri_rrs, fros):
    """Compute _sum_solids_div total angle in chunks."""
    # NOTE: This incorporates the division by 4PI that used to be separate
    tot_angle = np.zeros(len(fros))
    for ti in range(len(tri_rrs)):
        tri_rr = tri_rrs[ti]
        v1 = fros - tri_rr[0]
        v2 = fros - tri_rr[1]
        v3 = fros - tri_rr[2]
        v4 = np.empty((v1.shape[0], 3))
        _jit_cross(v4, v1, v2)
        triple = np.sum(v4 * v3, axis=1)
        l1 = np.sqrt(np.sum(v1 * v1, axis=1))
        l2 = np.sqrt(np.sum(v2 * v2, axis=1))
        l3 = np.sqrt(np.sum(v3 * v3, axis=1))
        s = (
            l1 * l2 * l3
            + np.sum(v1 * v2, axis=1) * l3
            + np.sum(v1 * v3, axis=1) * l2
            + np.sum(v2 * v3, axis=1) * l1
        )
        tot_angle -= np.arctan2(triple, s)
    return tot_angle


def _complete_sphere_surf(sphere, idx, level, complete=True):
    """Convert sphere conductor model to surface."""
    rad = sphere["layers"][idx]["rad"]
    r0 = sphere["r0"]
    surf = _tessellate_sphere_surf(level, rad=rad)
    surf["rr"] += r0
    if complete:
        complete_surface_info(surf, copy=False)
    surf["coord_frame"] = sphere["coord_frame"]
    return surf


@verbose
def dig_mri_distances(
    info,
    trans,
    subject,
    subjects_dir=None,
    dig_kinds="auto",
    exclude_frontal=False,
    on_defects="raise",
    verbose=None,
):
    """Compute distances between head shape points and the scalp surface.

    This function is useful to check that coregistration is correct.
    Unless outliers are present in the head shape points,
    one can assume an average distance around 2-3 mm.

    Parameters
    ----------
    %(info_not_none)s Must contain the head shape points in ``info['dig']``.
    trans : str | instance of Transform
        The head<->MRI transform. If str is passed it is the
        path to file on disk.
    subject : str
        The name of the subject.
    subjects_dir : str | None
        Directory containing subjects data. If None use
        the Freesurfer SUBJECTS_DIR environment variable.
    %(dig_kinds)s
    %(exclude_frontal)s
        Default is False.
    %(on_defects)s

        .. versionadded:: 1.0
    %(verbose)s

    Returns
    -------
    dists : array, shape (n_points,)
        The distances.

    See Also
    --------
    mne.bem.get_fitting_dig

    Notes
    -----
    .. versionadded:: 0.19
    """
    from .bem import get_fitting_dig

    pts = get_head_surf(
        subject,
        ("head-dense", "head", "bem"),
        subjects_dir=subjects_dir,
        on_defects=on_defects,
    )["rr"]
    trans = _get_trans(trans, fro="mri", to="head")[0]
    pts = apply_trans(trans, pts)
    info_dig = get_fitting_dig(info, dig_kinds, exclude_frontal=exclude_frontal)
    dists = _compute_nearest(pts, info_dig, return_dists=True)[1]
    return dists


def _mesh_borders(tris, mask):
    assert isinstance(mask, np.ndarray) and mask.ndim == 1
    edges = mesh_edges(tris)
    edges = edges.tocoo()
    border_edges = mask[edges.row] != mask[edges.col]
    return np.unique(edges.row[border_edges])


def _marching_cubes(image, level, smooth=0, fill_hole_size=None, use_flying_edges=True):
    """Compute marching cubes on a 3D image."""
    # vtkDiscreteMarchingCubes would be another option, but it merges
    # values at boundaries which is not what we want
    # https://kitware.github.io/vtk-examples/site/Cxx/Medical/GenerateModelsFromLabels/  # noqa: E501
    # Also vtkDiscreteFlyingEdges3D should be faster.
    # If we ever want not-discrete (continuous/float) marching cubes,
    # we should probably use vtkFlyingEdges3D rather than vtkMarchingCubes.
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes, vtkImageData
    from vtkmodules.vtkFiltersCore import vtkThreshold
    from vtkmodules.vtkFiltersGeneral import (
        vtkDiscreteFlyingEdges3D,
        vtkDiscreteMarchingCubes,
    )
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    if image.ndim != 3:
        raise ValueError(f"3D data must be supplied, got {image.shape}")

    level = np.array(level)
    if level.ndim != 1 or level.size == 0 or level.dtype.kind not in "ui":
        raise TypeError(
            "level must be non-empty numeric or 1D array-like of int, "
            f"got {level.ndim}D array-like of {level.dtype} with "
            f"{level.size} elements"
        )

    # vtkImageData indexes as slice, row, col (Z, Y, X):
    # https://discourse.vtk.org/t/very-confused-about-imdata-matrix-index-order/6608/2
    # We can accomplish this by raveling with order='F' later, so we might as
    # well make a copy with Fortran order now.
    # We also use double as passing integer types directly can be problematic!
    image = np.array(image, dtype=float, order="F")
    image_shape = image.shape

    # fill holes
    if fill_hole_size is not None:
        for val in level:
            bin_image = image == val
            mask = image == 0  # don't go into other areas
            bin_image = binary_dilation(bin_image, iterations=fill_hole_size, mask=mask)
            image[bin_image] = val

    data_vtk = numpy_to_vtk(image.ravel(order="F"), deep=False)

    mc = vtkDiscreteFlyingEdges3D() if use_flying_edges else vtkDiscreteMarchingCubes()
    # create image
    imdata = vtkImageData()
    imdata.SetDimensions(image_shape)
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(data_vtk)

    # compute marching cubes on smoothed data
    mc.SetNumberOfContours(len(level))
    for li, lev in enumerate(level):
        mc.SetValue(li, lev)
    mc.SetInputData(imdata)
    mc.Update()
    mc = _vtk_smooth(mc.GetOutput(), smooth)

    # get verts and triangles
    selector = vtkThreshold()
    selector.SetInputData(mc)
    dsa = vtkDataSetAttributes()
    selector.SetInputArrayToProcess(
        0,
        0,
        0,
        imdata.FIELD_ASSOCIATION_POINTS
        if use_flying_edges
        else imdata.FIELD_ASSOCIATION_CELLS,
        dsa.SCALARS,
    )
    geometry = vtkGeometryFilter()
    geometry.SetInputConnection(selector.GetOutputPort())

    out = list()
    for val in level:
        try:
            selector.SetLowerThreshold
        except AttributeError:
            selector.ThresholdBetween(val, val)
        else:
            # default SetThresholdFunction is between, so:
            selector.SetLowerThreshold(val)
            selector.SetUpperThreshold(val)
        geometry.Update()
        polydata = geometry.GetOutput()
        rr = vtk_to_numpy(polydata.GetPoints().GetData())
        tris = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        rr = np.ascontiguousarray(rr)
        tris = np.ascontiguousarray(tris)
        out.append((rr, tris))
    return out


def _vtk_smooth(pd, smooth):
    _validate_type(smooth, "numeric", smooth)
    smooth = float(smooth)
    if not 0 <= smooth < 1:
        raise ValueError(
            "smoothing factor must be between 0 (inclusive) and "
            f"1 (exclusive), got {smooth}"
        )
    if smooth == 0:
        return pd
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter

    logger.info(f"    Smoothing by a factor of {smooth}")
    return_ndarray = False
    if isinstance(pd, dict):
        pd = _surface_to_polydata(pd)
        return_ndarray = True
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pd)
    smoother.SetNumberOfIterations(100)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(1 - smooth)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()
    smoother.Update()
    out = smoother.GetOutput()
    if return_ndarray:
        out = _polydata_to_surface(out, normals=False)
    return out


_VOXELS_MAX = 1000  # define constant to avoid runtime issues


@fill_doc
def get_montage_volume_labels(montage, subject, subjects_dir=None, aseg="auto", dist=2):
    """Get regions of interest near channels from a Freesurfer parcellation.

    .. note:: This is applicable for channels inside the brain
              (intracranial electrodes).

    Parameters
    ----------
    %(montage)s
    %(subject)s
    %(subjects_dir)s
    %(aseg)s
    dist : float
        The distance in mm to use for identifying regions of interest.

    Returns
    -------
    labels : dict
        The regions of interest labels within ``dist`` of each channel.
    colors : dict
        The Freesurfer lookup table colors for the labels.
    """
    from ._freesurfer import _get_aseg, read_freesurfer_lut
    from .channels import DigMontage

    _validate_type(montage, DigMontage, "montage")
    _validate_type(dist, (int, float), "dist")

    if dist < 0 or dist > 10:
        raise ValueError("`dist` must be between 0 and 10")

    aseg, aseg_data = _get_aseg(aseg, subject, subjects_dir)

    # read freesurfer lookup table
    lut, fs_colors = read_freesurfer_lut()
    label_lut = {v: k for k, v in lut.items()}

    # assert that all the values in the aseg are in the labels
    assert all([idx in label_lut for idx in np.unique(aseg_data)])

    # get transform to surface RAS for distance units instead of voxels
    vox2ras_tkr = aseg.header.get_vox2ras_tkr()

    ch_dict = montage.get_positions()
    if ch_dict["coord_frame"] != "mri":
        raise RuntimeError(
            "Coordinate frame not supported, expected "
            '"mri", got ' + str(ch_dict["coord_frame"])
        )
    ch_coords = np.array(list(ch_dict["ch_pos"].values()))

    # convert to freesurfer voxel space
    ch_coords = apply_trans(
        np.linalg.inv(aseg.header.get_vox2ras_tkr()), ch_coords * 1000
    )
    labels = OrderedDict()
    for ch_name, ch_coord in zip(montage.ch_names, ch_coords):
        if np.isnan(ch_coord).any():
            labels[ch_name] = list()
        else:
            voxels = _voxel_neighbors(
                ch_coord,
                aseg_data,
                dist=dist,
                vox2ras_tkr=vox2ras_tkr,
                voxels_max=_VOXELS_MAX,
            )
            label_idxs = set([aseg_data[tuple(voxel)].astype(int) for voxel in voxels])
            labels[ch_name] = [label_lut[idx] for idx in label_idxs]

    all_labels = set([label for val in labels.values() for label in val])
    colors = {label: tuple(fs_colors[label][:3] / 255) + (1.0,) for label in all_labels}
    return labels, colors


def _get_neighbors(loc, image, voxels, thresh, dist_params):
    """Find all the neighbors above a threshold near a voxel."""
    neighbors = set()
    for axis in range(len(loc)):
        for i in (-1, 1):
            next_loc = np.array(loc)
            next_loc[axis] += i
            if thresh is not None:
                assert dist_params is None
                # must be above thresh, monotonically decreasing from
                # the peak and not already found
                next_loc = tuple(next_loc)
                if (
                    image[next_loc] > thresh
                    and image[next_loc] <= image[loc]
                    and next_loc not in voxels
                ):
                    neighbors.add(next_loc)
            else:
                assert thresh is None
                dist, seed_fs_ras, vox2ras_tkr = dist_params
                next_loc_fs_ras = apply_trans(vox2ras_tkr, next_loc + 0.5)
                if np.linalg.norm(seed_fs_ras - next_loc_fs_ras) <= dist:
                    neighbors.add(tuple(next_loc))
    return neighbors


def _voxel_neighbors(
    seed,
    image,
    thresh=None,
    max_peak_dist=1,
    use_relative=True,
    dist=None,
    vox2ras_tkr=None,
    voxels_max=100,
):
    """Find voxels above a threshold contiguous with a seed location.

    Parameters
    ----------
    seed : tuple | ndarray
        The location in image coordinated to seed the algorithm.
    image : ndarray
        The image to search.
    thresh : float
        The threshold to use as a cutoff for what qualifies as a neighbor.
        Will be relative to the peak if ``use_relative`` or absolute if not.
    max_peak_dist : int
        The maximum number of voxels to search for the peak near
        the seed location.
    use_relative : bool
        If ``True``, the threshold will be relative to the peak, if
        ``False``, the threshold will be absolute.
    dist : float
        The distance in mm to include surrounding voxels.
    vox2ras_tkr : ndarray
        The voxel to surface RAS affine. Must not be None if ``dist``
        if not None.
    voxels_max : int
        The maximum size of the output ``voxels``.

    Returns
    -------
    voxels : set
        The set of locations including the ``seed`` voxel and
        surrounding that meet the criteria.

    .. note:: Either ``dist`` or ``thesh`` may be used but not both.
              When ``thresh`` is used, first a peak nearby the seed
              location is found and then voxels are only included if they
              decrease monotonically from the peak. When ``dist`` is used,
              only voxels within ``dist`` mm of the seed are included.
    """
    seed = np.array(seed).round().astype(int)
    assert ((dist is not None) + (thresh is not None)) == 1
    if thresh is not None:
        dist_params = None
        check_grid = image[
            tuple([slice(idx - max_peak_dist, idx + max_peak_dist + 1) for idx in seed])
        ]
        peak = (
            np.array(np.unravel_index(np.argmax(check_grid), check_grid.shape))
            - max_peak_dist
            + seed
        )
        voxels = neighbors = set([tuple(peak)])
        if use_relative:
            thresh *= image[tuple(peak)]
    else:
        assert vox2ras_tkr is not None
        seed_fs_ras = apply_trans(vox2ras_tkr, seed + 0.5)  # center of voxel
        dist_params = (dist, seed_fs_ras, vox2ras_tkr)
        voxels = neighbors = set([tuple(seed)])
    while neighbors and len(voxels) <= voxels_max:
        next_neighbors = set()
        for next_loc in neighbors:
            voxel_neighbors = _get_neighbors(
                next_loc, image, voxels, thresh, dist_params
            )
            # prevent looping back to already visited voxels
            voxel_neighbors = voxel_neighbors.difference(voxels)
            # add voxels not already visited to search next
            next_neighbors = next_neighbors.union(voxel_neighbors)
            # add new voxels that match the criteria to the overall set
            voxels = voxels.union(voxel_neighbors)
            if len(voxels) > voxels_max:
                break
        neighbors = next_neighbors  # start again checking all new neighbors
    return voxels
