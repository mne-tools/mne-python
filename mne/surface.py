# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

# Many of the computations in this code were derived from Matti Hämäläinen's
# C code.

from copy import deepcopy
from distutils.version import LooseVersion
from glob import glob
from functools import partial
import os
from os import path as op
import warnings
from struct import pack

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye as speye

from .io.constants import FIFF
from .io.open import fiff_open
from .io.pick import pick_types
from .io.tree import dir_tree_find
from .io.tag import find_tag
from .io.write import (write_int, start_file, end_block, start_block, end_file,
                       write_string, write_float_sparse_rcs)
from .channels.channels import _get_meg_system
from .parallel import parallel_func
from .transforms import (transform_surface_to, _pol_to_cart, _cart_to_sph,
                         _get_trans, apply_trans, Transform)
from .utils import (logger, verbose, get_subjects_dir, warn, _check_fname,
                    _check_option, _ensure_int, _TempDir, run_subprocess,
                    _check_freesurfer_home)
from .fixes import (_serialize_volume_info, _get_read_geometry, einsum, jit,
                    prange, bincount)


###############################################################################
# AUTOMATED SURFACE FINDING

@verbose
def get_head_surf(subject, source=('bem', 'head'), subjects_dir=None,
                  verbose=None):
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
    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
    %(verbose)s

    Returns
    -------
    surf : dict
        The head surface.
    """
    return _get_head_surface(subject=subject, source=source,
                             subjects_dir=subjects_dir)


def _get_head_surface(subject, source, subjects_dir, raise_error=True):
    """Load the subject head surface."""
    from .bem import read_bem_surfaces
    # Load the head surface from the BEM
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if not isinstance(subject, str):
        raise TypeError('subject must be a string, not %s.' % (type(subject,)))
    # use realpath to allow for linked surfaces (c.f. MNE manual 196-197)
    if isinstance(source, str):
        source = [source]
    surf = None
    for this_source in source:
        this_head = op.realpath(op.join(subjects_dir, subject, 'bem',
                                        '%s-%s.fif' % (subject, this_source)))
        if op.exists(this_head):
            surf = read_bem_surfaces(this_head, True,
                                     FIFF.FIFFV_BEM_SURF_ID_HEAD,
                                     verbose=False)
        else:
            # let's do a more sophisticated search
            path = op.join(subjects_dir, subject, 'bem')
            if not op.isdir(path):
                raise IOError('Subject bem directory "%s" does not exist.'
                              % path)
            files = sorted(glob(op.join(path, '%s*%s.fif'
                                        % (subject, this_source))))
            for this_head in files:
                try:
                    surf = read_bem_surfaces(this_head, True,
                                             FIFF.FIFFV_BEM_SURF_ID_HEAD,
                                             verbose=False)
                except ValueError:
                    pass
                else:
                    break
        if surf is not None:
            break

    if surf is None:
        if raise_error:
            raise IOError('No file matching "%s*%s" and containing a head '
                          'surface found.' % (subject, this_source))
        else:
            return surf
    logger.info('Using surface from %s.' % this_head)
    return surf


@verbose
def get_meg_helmet_surf(info, trans=None, verbose=None):
    """Load the MEG helmet associated with the MEG sensors.

    Parameters
    ----------
    info : instance of Info
        Measurement info.
    trans : dict
        The head<->MRI transformation, usually obtained using
        read_trans(). Can be None, in which case the surface will
        be in head coordinates instead of MRI coordinates.
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
    from scipy.spatial import ConvexHull, Delaunay
    from .bem import read_bem_surfaces, _fit_sphere
    system, have_helmet = _get_meg_system(info)
    if have_helmet:
        logger.info('Getting helmet for system %s' % system)
        fname = op.join(op.split(__file__)[0], 'data', 'helmets',
                        system + '.fif.gz')
        surf = read_bem_surfaces(fname, False, FIFF.FIFFV_MNE_SURF_MEG_HELMET,
                                 verbose=False)
    else:
        rr = np.array([info['chs'][pick]['loc'][:3]
                       for pick in pick_types(info, meg=True, ref_meg=False,
                                              exclude=())])
        logger.info('Getting helmet for system %s (derived from %d MEG '
                    'channel locations)' % (system, len(rr)))
        hull = ConvexHull(rr)
        rr = rr[np.unique(hull.simplices)]
        R, center = _fit_sphere(rr, disp=False)
        sph = _cart_to_sph(rr - center)[:, 1:]
        # add a point at the front of the helmet (where the face should be):
        # 90 deg az and maximal el (down from Z/up axis)
        front_sph = [[np.pi / 2., sph[:, 1].max()]]
        sph = np.concatenate((sph, front_sph))
        xy = _pol_to_cart(sph[:, ::-1])
        tris = Delaunay(xy).simplices
        # remove the frontal point we added from the simplices
        tris = tris[(tris != len(sph) - 1).all(-1)]
        tris = _reorder_ccw(rr, tris)

        surf = dict(rr=rr, tris=tris)
        complete_surface_info(surf, copy=False, verbose=False)

    # Ignore what the file says, it's in device coords and we want MRI coords
    surf['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    dev_head_t = info['dev_head_t']
    if dev_head_t is None:
        dev_head_t = Transform('meg', 'head')
    transform_surface_to(surf, 'head', dev_head_t)
    if trans is not None:
        transform_surface_to(surf, 'mri', trans)
    return surf


def _reorder_ccw(rrs, tris):
    """Reorder tris of a convex hull to be wound counter-clockwise."""
    # This ensures that rendering with front-/back-face culling works properly
    com = np.mean(rrs, axis=0)
    rr_tris = rrs[tris]
    dirs = np.sign((np.cross(rr_tris[:, 1] - rr_tris[:, 0],
                             rr_tris[:, 2] - rr_tris[:, 0]) *
                    (rr_tris[:, 0] - com)).sum(-1)).astype(int)
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
    return ((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0] +
            (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1] +
            (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2])


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
            nn[:, idx] += bincount(verts, weights=tri_nn[:, idx],
                                   minlength=npts)
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
    csr = coo_matrix((data, (rows, cols)), shape=(npts, len(tris))).tocsr()
    neighbor_tri = [csr.indices[start:stop]
                    for start, stop in zip(csr.indptr[:-1], csr.indptr[1:])]
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


def _project_onto_surface(rrs, surf, project_rrs=False, return_nn=False,
                          method='accurate'):
    """Project points onto (scalp) surface."""
    if method == 'accurate':
        surf_geom = _get_tri_supp_geom(surf)
        pt_tris = np.empty((0,), int)
        pt_lens = np.zeros(len(rrs) + 1, int)
        out = _find_nearest_tri_pts(rrs, pt_tris, pt_lens,
                                    reproject=True, **surf_geom)
        if project_rrs:  #
            out += (einsum('ij,ijk->ik', out[0],
                           surf['rr'][surf['tris'][out[1]]]),)
        if return_nn:
            out += (surf_geom['nn'][out[1]],)
    else:  # nearest neighbor
        assert project_rrs
        idx = _compute_nearest(surf['rr'], rrs)
        out = (None, None, surf['rr'][idx])
        if return_nn:
            nn = _accumulate_normals(surf['tris'].astype(int), surf_geom['nn'],
                                     len(surf['rr']))
            out += (nn[idx],)
    return out


def _normal_orth(nn):
    """Compute orthogonal basis given a normal."""
    assert nn.shape[-1:] == (3,)
    prod = np.einsum('...i,...j->...ij', nn, nn)
    _, u = np.linalg.eigh(np.eye(3) - prod)
    u = u[..., ::-1]
    #  Make sure that ez is in the direction of nn
    signs = np.sign(np.matmul(nn[..., np.newaxis, :], u[..., -1:]))
    signs[signs == 0] = 1
    u *= signs
    return u.swapaxes(-1, -2)


@verbose
def complete_surface_info(surf, do_neighbor_vert=False, copy=True,
                          verbose=None):
    """Complete surface information.

    Parameters
    ----------
    surf : dict
        The surface.
    do_neighbor_vert : bool
        If True, add neighbor vertex information.
    copy : bool
        If True (default), make a copy. If False, operate in-place.
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
    surf['ntri'] = surf.get('ntri', len(surf['tris']))
    surf['np'] = surf.get('np', len(surf['rr']))
    surf['tri_area'] = np.zeros(surf['ntri'])
    r1 = surf['rr'][surf['tris'][:, 0], :]
    r2 = surf['rr'][surf['tris'][:, 1], :]
    r3 = surf['rr'][surf['tris'][:, 2], :]
    surf['tri_cent'] = (r1 + r2 + r3) / 3.0
    surf['tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    surf['tri_area'] = _normalize_vectors(surf['tri_nn']) / 2.0
    zidx = np.where(surf['tri_area'] == 0)[0]
    if len(zidx) > 0:
        logger.info('    Warning: zero size triangles: %s' % zidx)

    #    Find neighboring triangles, accumulate vertex normals, normalize
    logger.info('    Triangle neighbors and vertex normals...')
    surf['neighbor_tri'] = _triangle_neighbors(surf['tris'], surf['np'])
    surf['nn'] = _accumulate_normals(surf['tris'].astype(int),
                                     surf['tri_nn'], surf['np'])
    _normalize_vectors(surf['nn'])

    #   Check for topological defects
    zero, fewer = list(), list()
    for ni, n in enumerate(surf['neighbor_tri']):
        if len(n) < 3:
            if len(n) == 0:
                zero.append(ni)
            else:
                fewer.append(ni)
                surf['neighbor_tri'][ni] = np.array([], int)
    if len(zero) > 0:
        logger.info('    Vertices do not have any neighboring '
                    'triangles: [%s]' % ', '.join(str(z) for z in zero))
    if len(fewer) > 0:
        logger.info('    Vertices have fewer than three neighboring '
                    'triangles, removing neighbors: [%s]'
                    % ', '.join(str(f) for f in fewer))

    #   Determine the neighboring vertices and fix errors
    if do_neighbor_vert is True:
        logger.info('    Vertex neighbors...')
        surf['neighbor_vert'] = [_get_surf_neighbors(surf, k)
                                 for k in range(surf['np'])]

    return surf


def _get_surf_neighbors(surf, k):
    """Calculate the surface neighbors based on triangulation."""
    verts = set()
    for v in surf['tris'][surf['neighbor_tri'][k]].flat:
        verts.add(v)
    verts.remove(k)
    verts = np.array(sorted(verts))
    assert np.all(verts < surf['np'])
    nneighbors = len(verts)
    nneigh_max = len(surf['neighbor_tri'][k])
    if nneighbors > nneigh_max:
        raise RuntimeError('Too many neighbors for vertex %d' % k)
    elif nneighbors != nneigh_max:
        logger.info('    Incorrect number of distinct neighbors for vertex'
                    ' %d (%d instead of %d) [fixed].' % (k, nneighbors,
                                                         nneigh_max))
    return verts


def _normalize_vectors(rr):
    """Normalize surface vertices."""
    size = np.linalg.norm(rr, axis=1)
    mask = (size > 0)
    rr[mask] /= size[mask, np.newaxis]  # operate in-place
    return size


class _CDist(object):
    """Wrapper for cdist that uses a Tree-like pattern."""

    def __init__(self, xhs):
        self._xhs = xhs

    def query(self, rr):
        from scipy.spatial.distance import cdist
        nearest = list()
        dists = list()
        for r in rr:
            d = cdist(r[np.newaxis, :], self._xhs)
            idx = np.argmin(d)
            nearest.append(idx)
            dists.append(d[0, idx])
        return np.array(dists), np.array(nearest)


def _compute_nearest(xhs, rr, method='BallTree', return_dists=False):
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


class _DistanceQuery(object):
    """Wrapper for fast distance queries."""

    def __init__(self, xhs, method='BallTree', allow_kdtree=False):
        assert method in ('BallTree', 'cKDTree', 'cdist')

        # Fastest for our problems: balltree
        if method == 'BallTree':
            try:
                from sklearn.neighbors import BallTree
            except ImportError:
                logger.info('Nearest-neighbor searches will be significantly '
                            'faster if scikit-learn is installed.')
                method = 'cKDTree'
            else:
                self.query = partial(_safe_query, func=BallTree(xhs).query,
                                     reduce=True, return_distance=True)

        # Then cKDTree
        if method == 'cKDTree':
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                method = 'cdist'
            else:
                self.query = cKDTree(xhs).query

        # KDTree is really only faster for huge (~100k) sets,
        # (e.g., with leafsize=2048), and it's slower for small (~5k)
        # sets. We can add it later if we think it will help.

        # Then the worst: cdist
        if method == 'cdist':
            self.query = _CDist(xhs).query

        self.data = xhs


@verbose
def _points_outside_surface(rr, surf, n_jobs=1, verbose=None):
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
    assert n_jobs > 0
    parallel, p_fun, _ = parallel_func(_get_solids, n_jobs)
    tot_angles = parallel(p_fun(surf['rr'][tris], rr)
                          for tris in np.array_split(surf['tris'], n_jobs))
    return np.abs(np.sum(tot_angles, axis=0) / (2 * np.pi) - 1.0) > 1e-5


class _CheckInside(object):
    """Efficiently check if points are inside a surface."""

    def __init__(self, surf):
        from scipy.spatial import Delaunay
        self.surf = surf
        self.inner_r = None
        self.cm = surf['rr'].mean(0)
        if not _points_outside_surface(
                self.cm[np.newaxis], surf)[0]:  # actually inside
            # Immediately cull some points from the checks
            self.inner_r = np.linalg.norm(surf['rr'] - self.cm, axis=-1).min()
        # We could use Delaunay or ConvexHull here, Delaunay is slightly slower
        # to construct but faster to evaluate
        # See https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl  # noqa
        self.del_tri = Delaunay(surf['rr'])

    @verbose
    def __call__(self, rr, n_jobs=1, verbose=None):
        inside = np.ones(len(rr), bool)  # innocent until proven guilty
        idx = np.arange(len(rr))

        # Limit to indices that can plausibly be outside the surf
        if self.inner_r is not None:
            mask = np.linalg.norm(rr - self.cm, axis=-1) >= self.inner_r
            idx = idx[mask]
            rr = rr[mask]
            logger.info('    Skipping interior check for %d sources that fit '
                        'inside a sphere of radius %6.1f mm'
                        % ((~mask).sum(), self.inner_r * 1000))

        # Use qhull as our first pass (*much* faster than our check)
        del_outside = self.del_tri.find_simplex(rr) < 0
        omit_outside = sum(del_outside)
        inside[idx[del_outside]] = False
        idx = idx[~del_outside]
        rr = rr[~del_outside]
        logger.info('    Skipping solid angle check for %d points using Qhull'
                    % (omit_outside,))

        # use our more accurate check
        solid_outside = _points_outside_surface(rr, self.surf, n_jobs)
        omit_outside += np.sum(solid_outside)
        inside[idx[solid_outside]] = False
        return inside


###############################################################################
# Handle freesurfer

def _fread3(fobj):
    """Read 3 bytes and adjust."""
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3


def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object."""
    b1, b2, b3 = np.fromfile(fobj, ">u1",
                             3 * n).reshape(-1, 3).astype(np.int64).T
    return (b1 << 16) + (b2 << 8) + b3


def read_curvature(filepath, binary=True):
    """Load in curvature values from the ?h.curv file.

    Parameters
    ----------
    filepath : str
        Input path to the .curv file.
    binary : bool
        Specify if the output array is to hold binary values. Defaults to True.

    Returns
    -------
    curv : array, shape=(n_vertices,)
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
def read_surface(fname, read_metadata=False, return_dict=False,
                 file_format='auto', verbose=None):
    """Load a Freesurfer surface mesh in triangular format.

    Parameters
    ----------
    fname : str
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
    fname = _check_fname(fname, 'read', True)
    _check_option('file_format', file_format, ['auto', 'freesurfer', 'obj'])

    if file_format == 'auto':
        _, ext = op.splitext(fname)
        if ext.lower() == '.obj':
            file_format = 'obj'
        else:
            file_format = 'freesurfer'

    if file_format == 'freesurfer':
        ret = _get_read_geometry()(fname, read_metadata=read_metadata)
    elif file_format == 'obj':
        ret = _read_wavefront_obj(fname)
        if read_metadata:
            ret += (dict(),)

    if return_dict:
        ret += (dict(rr=ret[0], tris=ret[1], ntri=len(ret[1]), use_tris=ret[1],
                     np=len(ret[0])),)
    return ret


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
                    raise RuntimeError('Only triangle faces allowed.')
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
    with open(fname, 'r') as fid:
        ver = np.fromfile(fid, dtype='>i4', count=1)[0]
        if ver != -1:
            raise RuntimeError(f'incorrect version # {ver} (not -1) found')
        npts = np.fromfile(fid, dtype='>i4', count=1)[0]
        dtype = np.dtype(
            [('vertno', '>i4'), ('x', '>f'), ('y', '>f'), ('z', '>f')])
        recs = np.fromfile(fid, dtype=dtype, count=npts)
    # numpy to dict
    patch = {key: recs[key] for key in dtype.fields.keys()}
    patch['vertno'] -= 1

    # read surrogate surface
    rrs, tris = read_surface(
        op.join(op.dirname(fname), op.basename(fname)[:3] + 'sphere'))
    orig_tris = tris
    is_vert = patch['vertno'] > 0  # negative are edges, ignored for now
    verts = patch['vertno'][is_vert]

    # eliminate invalid tris and zero out unused rrs
    mask = np.zeros((len(rrs),), dtype=bool)
    mask[verts] = True
    rrs[~mask] = 0.
    tris = tris[mask[tris].all(1)]
    for ii, key in enumerate(['x', 'y', 'z']):
        rrs[verts, ii] = patch[key][is_vert]
    return rrs, tris, orig_tris


##############################################################################
# SURFACE CREATION

def _get_ico_surface(grade, patch_stats=False):
    """Return an icosahedral surface of the desired grade."""
    # always use verbose=False since users don't need to know we're pulling
    # these from a file
    from .bem import read_bem_surfaces
    ico_file_name = op.join(op.dirname(__file__), 'data',
                            'icos.fif.gz')
    ico = read_bem_surfaces(ico_file_name, patch_stats, s_id=9000 + grade,
                            verbose=False)
    return ico


def _tessellate_sphere_surf(level, rad=1.0):
    """Return a surface structure instead of the details."""
    rr, tris = _tessellate_sphere(level)
    npt = len(rr)  # called "npt" instead of "np" because of numpy...
    ntri = len(tris)
    nn = rr.copy()
    rr *= rad
    s = dict(rr=rr, np=npt, tris=tris, use_tris=tris, ntri=ntri, nuse=npt,
             nn=nn, inuse=np.ones(npt, int))
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
    rr = np.array([[1, 0, 0], [-1, 0, 0],  # xplus, xminus
                   [0, 1, 0], [0, -1, 0],  # yplus, yminus
                   [0, 0, 1], [0, 0, -1]], float)  # zplus, zminus
    tris = np.array([[0, 4, 2], [2, 4, 1], [1, 4, 3], [3, 4, 0],
                     [0, 2, 5], [2, 1, 5], [1, 3, 5], [3, 0, 5]], int)

    # A unit octahedron
    if mylevel < 1:
        raise ValueError('oct subdivision must be >= 1')

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
        tris = np.array((np.c_[tris[:, 0], bidx, aidx],
                         np.c_[bidx, tris[:, 1], cidx],
                         np.c_[aidx, bidx, cidx],
                         np.c_[aidx, cidx, tris[:, 2]]), int).swapaxes(0, 1)
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
    do_neighbor_vert = (stype == 'spacing')
    complete_surface_info(surf, do_neighbor_vert, copy=False)
    if stype == 'all':
        surf['inuse'] = np.ones(surf['np'], int)
        surf['use_tris'] = None
    elif stype == 'spacing':
        _decimate_surface_spacing(surf, ico_surf)
        surf['use_tris'] = None
        del surf['neighbor_vert']
    else:  # ico or oct
        # ## from mne_ico_downsample.c ## #
        surf_name = op.join(subjects_dir, subject, 'surf', hemi + '.sphere')
        logger.info('Loading geometry from %s...' % surf_name)
        from_surf = read_surface(surf_name, return_dict=True)[-1]
        _normalize_vectors(from_surf['rr'])
        if from_surf['np'] != surf['np']:
            raise RuntimeError('Mismatch between number of surface vertices, '
                               'possible parcellation error?')
        _normalize_vectors(ico_surf['rr'])

        # Make the maps
        mmap = _compute_nearest(from_surf['rr'], ico_surf['rr'])
        nmap = len(mmap)
        surf['inuse'] = np.zeros(surf['np'], int)
        for k in range(nmap):
            if surf['inuse'][mmap[k]]:
                # Try the nearest neighbors
                neigh = _get_surf_neighbors(surf, mmap[k])
                was = mmap[k]
                inds = np.where(np.logical_not(surf['inuse'][neigh]))[0]
                if len(inds) == 0:
                    raise RuntimeError('Could not find neighbor for vertex '
                                       '%d / %d' % (k, nmap))
                else:
                    mmap[k] = neigh[inds[-1]]
                logger.info('    Source space vertex moved from %d to %d '
                            'because of double occupation', was, mmap[k])
            elif mmap[k] < 0 or mmap[k] > surf['np']:
                raise RuntimeError('Map number out of range (%d), this is '
                                   'probably due to inconsistent surfaces. '
                                   'Parts of the FreeSurfer reconstruction '
                                   'need to be redone.' % mmap[k])
            surf['inuse'][mmap[k]] = True

        logger.info('Setting up the triangulation for the decimated '
                    'surface...')
        surf['use_tris'] = np.array([mmap[ist] for ist in ico_surf['tris']],
                                    np.int32)
    if surf['use_tris'] is not None:
        surf['nuse_tri'] = len(surf['use_tris'])
    else:
        surf['nuse_tri'] = 0
    surf['nuse'] = np.sum(surf['inuse'])
    surf['vertno'] = np.where(surf['inuse'])[0]

    # set some final params
    sizes = _normalize_vectors(surf['nn'])
    surf['inuse'][sizes <= 0] = False
    surf['nuse'] = np.sum(surf['inuse'])
    surf['subject_his_id'] = subject
    return surf


def _decimate_surface_spacing(surf, spacing):
    assert isinstance(spacing, int)
    assert spacing > 0
    logger.info('    Decimating...')
    d = np.full(surf['np'], 10000, int)

    # A mysterious algorithm follows
    for k in range(surf['np']):
        neigh = surf['neighbor_vert'][k]
        d[k] = min(np.min(d[neigh]) + 1, d[k])
        if d[k] >= spacing:
            d[k] = 0
        d[neigh] = np.minimum(d[neigh], d[k] + 1)

    if spacing == 2.0:
        for k in range(surf['np'] - 1, -1, -1):
            for n in surf['neighbor_vert'][k]:
                d[k] = min(d[k], d[n] + 1)
                d[n] = min(d[n], d[k] + 1)
        for k in range(surf['np']):
            if d[k] > 0:
                neigh = surf['neighbor_vert'][k]
                n = np.sum(d[neigh] == 0)
                if n <= 2:
                    d[k] = 0
                d[neigh] = np.minimum(d[neigh], d[k] + 1)

    surf['inuse'] = np.zeros(surf['np'], int)
    surf['inuse'][d == 0] = 1
    return surf


def write_surface(fname, coords, faces, create_stamp='', volume_info=None,
                  file_format='auto', overwrite=False):
    """Write a triangular Freesurfer surface mesh.

    Accepts the same data format as is returned by read_surface().

    Parameters
    ----------
    fname : str
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
    overwrite : bool
        If True, overwrite the file if it exists.

    See Also
    --------
    read_surface
    read_tri
    """
    fname = _check_fname(fname, overwrite=overwrite)
    _check_option('file_format', file_format, ['auto', 'freesurfer', 'obj'])

    if file_format == 'auto':
        _, ext = op.splitext(fname)
        if ext.lower() == '.obj':
            file_format = 'obj'
        else:
            file_format = 'freesurfer'

    if file_format == 'freesurfer':
        try:
            import nibabel as nib
            has_nibabel = LooseVersion(nib.__version__) > LooseVersion('2.1.0')
        except ImportError:
            has_nibabel = False
        if has_nibabel:
            nib.freesurfer.io.write_geometry(fname, coords, faces,
                                             create_stamp=create_stamp,
                                             volume_info=volume_info)
            return
        if len(create_stamp.splitlines()) > 1:
            raise ValueError("create_stamp can only contain one line")

        with open(fname, 'wb') as fid:
            fid.write(pack('>3B', 255, 255, 254))
            strs = ['%s\n' % create_stamp, '\n']
            strs = [s.encode('utf-8') for s in strs]
            fid.writelines(strs)
            vnum = len(coords)
            fnum = len(faces)
            fid.write(pack('>2i', vnum, fnum))
            fid.write(np.array(coords, dtype='>f4').tobytes())
            fid.write(np.array(faces, dtype='>i4').tobytes())

            # Add volume info, if given
            if volume_info is not None and len(volume_info) > 0:
                fid.write(_serialize_volume_info(volume_info))

    elif file_format == 'obj':
        with open(fname, 'w') as fid:
            for line in create_stamp.splitlines():
                fid.write(f'# {line}\n')
            for v in coords:
                fid.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for f in faces:
                fid.write(f'f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n')


###############################################################################
# Decimation

def _decimate_surface_vtk(points, triangles, n_triangles):
    """Aux function."""
    try:
        from vtk.util.numpy_support import \
            numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtk.numpy_interface.dataset_adapter import WrapDataObject
        from vtk import \
            vtkPolyData, vtkQuadricDecimation, vtkPoints, vtkCellArray
    except ImportError:
        raise ValueError('This function requires the VTK package to be '
                         'installed')
    if triangles.max() > len(points) - 1:
        raise ValueError('The triangles refer to undefined points. '
                         'Please check your mesh.')
    src = vtkPolyData()
    vtkpoints = vtkPoints()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
        vtkpoints.SetData(numpy_to_vtk(points.astype(np.float64)))
    src.SetPoints(vtkpoints)
    vtkcells = vtkCellArray()
    triangles_ = np.pad(
        triangles, ((0, 0), (1, 0)), 'constant', constant_values=3)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
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
    out = WrapDataObject(decimate.GetOutput())
    rrs = out.Points
    tris = out.Polygons.reshape(-1, 4)[:, 1:]
    return rrs, tris


def _decimate_surface_sphere(rr, tris, n_triangles):
    _check_freesurfer_home()
    map_ = {}
    ico_levels = [20, 80, 320, 1280, 5120, 20480]
    map_.update({n_tri: ('ico', ii) for ii, n_tri in enumerate(ico_levels)})
    oct_levels = 2 ** (2 * np.arange(7) + 3)
    map_.update({n_tri: ('oct', ii) for ii, n_tri in enumerate(oct_levels, 1)})
    _check_option('n_triangles', n_triangles, sorted(map_),
                  extra=' when method="sphere"')
    func_map = dict(ico=_get_ico_surface, oct=_tessellate_sphere_surf)
    kind, level = map_[n_triangles]
    logger.info('Decimating using Freesurfer spherical %s%s downsampling'
                % (kind, level))
    ico_surf = func_map[kind](level)
    assert len(ico_surf['tris']) == n_triangles
    tempdir = _TempDir()
    orig = op.join(tempdir, 'lh.temp')
    write_surface(orig, rr, tris)
    logger.info('    Extracting main mesh component ...')
    run_subprocess(
        ['mris_extract_main_component', orig, orig],
        verbose='error')
    logger.info('    Smoothing ...')
    smooth = orig + '.smooth'
    run_subprocess(
        ['mris_smooth', '-nw', orig, smooth],
        verbose='error')
    logger.info('    Inflating ...')
    inflated = orig + '.inflated'
    run_subprocess(
        ['mris_inflate', '-no-save-sulc', smooth, inflated],
        verbose='error')
    logger.info('    Sphere ...')
    qsphere = orig + '.qsphere'
    run_subprocess(
        ['mris_sphere', '-q', inflated, qsphere], verbose='error')
    sphere_rr, _ = read_surface(qsphere)
    norms = np.linalg.norm(sphere_rr, axis=1, keepdims=True)
    sphere_rr /= norms
    idx = _compute_nearest(sphere_rr, ico_surf['rr'], method='cKDTree')
    n_dup = len(idx) - len(np.unique(idx))
    if n_dup:
        raise RuntimeError('Could not reduce to %d triangles using ico, '
                           '%d/%d vertices were duplicates'
                           % (n_triangles, n_dup, len(idx)))
    logger.info('[done]')
    return rr[idx], ico_surf['tris']


@verbose
def decimate_surface(points, triangles, n_triangles, method='quadric',
                     verbose=None):
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
    method_map = dict(quadric=_decimate_surface_vtk,
                      sphere=_decimate_surface_sphere)
    _check_option('method', method, sorted(method_map))
    if n_triangles > len(triangles):
        raise ValueError('Requested n_triangles (%s) exceeds number of '
                         'original triangles (%s)'
                         % (n_triangles, len(triangles)))
    return method_map[method](points, triangles, n_triangles)


###############################################################################
# Morph maps

# XXX this morphing related code should probably be moved to morph.py

@verbose
def read_morph_map(subject_from, subject_to, subjects_dir=None, xhemi=False,
                   verbose=None):
    """Read morph map.

    Morph maps can be generated with mne_make_morph_maps. If one isn't
    available, it will be generated automatically and saved to the
    ``subjects_dir/morph_maps`` directory.

    Parameters
    ----------
    subject_from : str
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : str
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    subjects_dir : str
        Path to SUBJECTS_DIR is not set in the environment.
    xhemi : bool
        Morph across hemisphere. Currently only implemented for
        ``subject_to == subject_from``. See notes of
        :func:`mne.compute_source_morph`.
    %(verbose)s

    Returns
    -------
    left_map, right_map : ~scipy.sparse.csr_matrix
        The morph maps for the 2 hemispheres.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # First check for morph-map dir existence
    mmap_dir = op.join(subjects_dir, 'morph-maps')
    if not op.isdir(mmap_dir):
        try:
            os.mkdir(mmap_dir)
        except Exception:
            warn('Could not find or make morph map directory "%s"' % mmap_dir)

    # filename components
    if xhemi:
        if subject_to != subject_from:
            raise NotImplementedError(
                "Morph-maps between hemispheres are currently only "
                "implemented for subject_to == subject_from")
        map_name_temp = '%s-%s-xhemi'
        log_msg = 'Creating morph map %s -> %s xhemi'
    else:
        map_name_temp = '%s-%s'
        log_msg = 'Creating morph map %s -> %s'

    map_names = [map_name_temp % (subject_from, subject_to),
                 map_name_temp % (subject_to, subject_from)]

    # find existing file
    for map_name in map_names:
        fname = op.join(mmap_dir, '%s-morph.fif' % map_name)
        if op.exists(fname):
            return _read_morph_map(fname, subject_from, subject_to)
    # if file does not exist, make it
    logger.info('Morph map "%s" does not exist, creating it and saving it to '
                'disk' % fname)
    logger.info(log_msg % (subject_from, subject_to))
    mmap_1 = _make_morph_map(subject_from, subject_to, subjects_dir, xhemi)
    if subject_to == subject_from:
        mmap_2 = None
    else:
        logger.info(log_msg % (subject_to, subject_from))
        mmap_2 = _make_morph_map(subject_to, subject_from, subjects_dir,
                                 xhemi)
    _write_morph_map(fname, subject_from, subject_to, mmap_1, mmap_2)
    return mmap_1


def _read_morph_map(fname, subject_from, subject_to):
    """Read a morph map from disk."""
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Locate all maps
        maps = dir_tree_find(tree, FIFF.FIFFB_MNE_MORPH_MAP)
        if len(maps) == 0:
            raise ValueError('Morphing map data not found')

        # Find the correct ones
        left_map = None
        right_map = None
        for m in maps:
            tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_FROM)
            if tag.data == subject_from:
                tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_TO)
                if tag.data == subject_to:
                    #  Names match: which hemishere is this?
                    tag = find_tag(fid, m, FIFF.FIFF_MNE_HEMI)
                    if tag.data == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
                        tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                        left_map = tag.data
                        logger.info('    Left-hemisphere map read.')
                    elif tag.data == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
                        tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                        right_map = tag.data
                        logger.info('    Right-hemisphere map read.')

    if left_map is None or right_map is None:
        raise ValueError('Could not find both hemispheres in %s' % fname)

    return left_map, right_map


def _write_morph_map(fname, subject_from, subject_to, mmap_1, mmap_2):
    """Write a morph map to disk."""
    try:
        fid = start_file(fname)
    except Exception as exp:
        warn('Could not write morph-map file "%s" (error: %s)'
             % (fname, exp))
        return

    assert len(mmap_1) == 2
    hemis = [FIFF.FIFFV_MNE_SURF_LEFT_HEMI, FIFF.FIFFV_MNE_SURF_RIGHT_HEMI]
    for m, hemi in zip(mmap_1, hemis):
        start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_from)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_to)
        write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
        end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
    # don't write mmap_2 if it is identical (subject_to == subject_from)
    if mmap_2 is not None:
        assert len(mmap_2) == 2
        for m, hemi in zip(mmap_2, hemis):
            start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
            write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_to)
            write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_from)
            write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
            write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
            end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
    end_file(fid)


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
    r1 = surf['rr'][surf['tris'][:, 0], :]
    r12 = surf['rr'][surf['tris'][:, 1], :] - r1
    r13 = surf['rr'][surf['tris'][:, 2], :] - r1
    r1213 = np.ascontiguousarray(np.array([r12, r13]).swapaxes(0, 1))
    a = einsum('ij,ij->i', r12, r12)
    b = einsum('ij,ij->i', r13, r13)
    c = einsum('ij,ij->i', r12, r13)
    mat = np.ascontiguousarray(np.rollaxis(np.array([[b, -c], [-c, a]]), 2))
    norm = (a * b - c * c)
    norm[norm == 0] = 1.  # avoid divide by zero
    mat /= norm[:, np.newaxis, np.newaxis]
    nn = fast_cross_3d(r12, r13)
    _normalize_vectors(nn)
    return dict(r1=r1, r12=r12, r13=r13, r1213=r1213,
                a=a, b=b, c=c, mat=mat, nn=nn)


def _make_morph_map(subject_from, subject_to, subjects_dir, xhemi):
    """Construct morph map from one subject to another.

    Note that this is close, but not exactly like the C version.
    For example, parts are more accurate due to double precision,
    so expect some small morph-map differences!

    Note: This seems easily parallelizable, but the overhead
    of pickling all the data structures makes it less efficient
    than just running on a single core :(
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    if xhemi:
        reg = '%s.sphere.left_right'
        hemis = (('lh', 'rh'), ('rh', 'lh'))
    else:
        reg = '%s.sphere.reg'
        hemis = (('lh', 'lh'), ('rh', 'rh'))

    return [_make_morph_map_hemi(subject_from, subject_to, subjects_dir,
                                 reg % hemi_from, reg % hemi_to)
            for hemi_from, hemi_to in hemis]


def _make_morph_map_hemi(subject_from, subject_to, subjects_dir, reg_from,
                         reg_to):
    """Construct morph map for one hemisphere."""
    # add speedy short-circuit for self-maps
    if subject_from == subject_to and reg_from == reg_to:
        fname = op.join(subjects_dir, subject_from, 'surf', reg_from)
        n_pts = len(read_surface(fname, verbose=False)[0])
        return speye(n_pts, n_pts, format='csr')

    # load surfaces and normalize points to be on unit sphere
    fname = op.join(subjects_dir, subject_from, 'surf', reg_from)
    from_rr, from_tri = read_surface(fname, verbose=False)
    fname = op.join(subjects_dir, subject_to, 'surf', reg_to)
    to_rr = read_surface(fname, verbose=False)[0]
    _normalize_vectors(from_rr)
    _normalize_vectors(to_rr)

    # from surface: get nearest neighbors, find triangles for each vertex
    nn_pts_idx = _compute_nearest(from_rr, to_rr, method='cKDTree')
    from_pt_tris = _triangle_neighbors(from_tri, len(from_rr))
    from_pt_tris = [from_pt_tris[pt_idx].astype(int) for pt_idx in nn_pts_idx]
    from_pt_lens = np.cumsum([0] + [len(x) for x in from_pt_tris])
    from_pt_tris = np.concatenate(from_pt_tris)
    assert from_pt_tris.ndim == 1
    assert from_pt_lens[-1] == len(from_pt_tris)

    # find triangle in which point lies and assoc. weights
    tri_inds = []
    weights = []
    tri_geom = _get_tri_supp_geom(dict(rr=from_rr, tris=from_tri))
    weights, tri_inds = _find_nearest_tri_pts(
        to_rr, from_pt_tris, from_pt_lens, run_all=False, reproject=False,
        **tri_geom)

    nn_idx = from_tri[tri_inds]
    weights = np.array(weights)

    row_ind = np.repeat(np.arange(len(to_rr)), 3)
    this_map = csr_matrix((weights.ravel(), (row_ind, nn_idx.ravel())),
                          shape=(len(to_rr), len(from_rr)))
    return this_map


@jit(parallel=True)
def _find_nearest_tri_pts(rrs, pt_triss, pt_lens,
                          a, b, c, nn, r1, r12, r13, r1213, mat,
                          run_all=True, reproject=False):  # pragma: no cover
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
        start, stop = pt_lens[ri:ri + 2]
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
        p, q, pt = np.float64(0.), np.float64(1.), np.int64(0)
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
                use_pt_tris, rr[0], pqs[s], dists[s], a, b, c)
            if np.abs(distt) < np.abs(dist):
                p, q, pt, dist = pp, qq, ptt, distt
        w = (1 - p - q, p, q)
        if reproject:
            # Calculate a linear interpolation between the vertex values to
            # get coords of pt projected onto closest triangle
            coords = _triangle_coords(rr[0], pt, r1, nn, r12, r13, a, b, c)
            w = (1. - coords[0] - coords[1], coords[0], coords[1])
        weights[ri] = w
        tri_idx[ri] = pt
    return weights, tri_idx


@jit()
def _nearest_tri_edge(pt_tris, to_pt, pqs, dist, a, b, c):  # pragma: no cover
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
    t1 = (0.5 * ((2.0 * aa - cc) * (1.0 - pp) +
                 (2.0 * bb - cc) * qq) / (aa + bb - cc))
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
    edges : sparse matrix
        The adjacency matrix.
    """
    if np.max(tris) > len(np.unique(tris)):
        raise ValueError(
            'Cannot compute adjacency on a selection of triangles.')

    npoints = np.max(tris) + 1
    ones_ntris = np.ones(3 * len(tris))

    a, b, c = tris.T
    x = np.concatenate((a, b, c))
    y = np.concatenate((b, c, a))
    edges = coo_matrix((ones_ntris, (x, y)), shape=(npoints, npoints))
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
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices.
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.linalg.norm(vert[edges.row, :] - vert[edges.col, :], axis=1)
    dist_matrix = csr_matrix((dist, (edges.row, edges.col)), shape=edges.shape)
    return dist_matrix


@verbose
def read_tri(fname_in, swap=False, verbose=None):
    """Read triangle definitions from an ascii file.

    Parameters
    ----------
    fname_in : str
        Path to surface ASCII file (ending with '.tri').
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
    with open(fname_in, "r") as fid:
        lines = fid.readlines()
    n_nodes = int(lines[0])
    n_tris = int(lines[n_nodes + 1])
    n_items = len(lines[1].split())
    if n_items in [3, 6, 14, 17]:
        inds = range(3)
    elif n_items in [4, 7]:
        inds = range(1, 4)
    else:
        raise IOError('Unrecognized format of data.')
    rr = np.array([np.array([float(v) for v in line.split()])[inds]
                   for line in lines[1:n_nodes + 1]])
    tris = np.array([np.array([int(v) for v in line.split()])[inds]
                     for line in lines[n_nodes + 2:n_nodes + 2 + n_tris]])
    if swap:
        tris[:, [2, 1]] = tris[:, [1, 2]]
    tris -= 1
    logger.info('Loaded surface from %s with %s nodes and %s triangles.' %
                (fname_in, n_nodes, n_tris))
    if n_items in [3, 4]:
        logger.info('Node normals were not included in the source file.')
    else:
        warn('Node normals were not read.')
    return (rr, tris)


@jit()
def _get_solids(tri_rrs, fros):
    """Compute _sum_solids_div total angle in chunks."""
    # NOTE: This incorporates the division by 4PI that used to be separate
    tot_angle = np.zeros((len(fros)))
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
        s = (l1 * l2 * l3 +
             np.sum(v1 * v2, axis=1) * l3 +
             np.sum(v1 * v3, axis=1) * l2 +
             np.sum(v2 * v3, axis=1) * l1)
        tot_angle -= np.arctan2(triple, s)
    return tot_angle


def _complete_sphere_surf(sphere, idx, level, complete=True):
    """Convert sphere conductor model to surface."""
    rad = sphere['layers'][idx]['rad']
    r0 = sphere['r0']
    surf = _tessellate_sphere_surf(level, rad=rad)
    surf['rr'] += r0
    if complete:
        complete_surface_info(surf, copy=False)
    surf['coord_frame'] = sphere['coord_frame']
    return surf


@verbose
def dig_mri_distances(info, trans, subject, subjects_dir=None,
                      dig_kinds='auto', exclude_frontal=False, verbose=None):
    """Compute distances between head shape points and the scalp surface.

    This function is useful to check that coregistration is correct.
    Unless outliers are present in the head shape points,
    one can assume an average distance around 2-3 mm.

    Parameters
    ----------
    info : instance of Info
        The measurement info that contains the head shape
        points in ``info['dig']``.
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
    pts = get_head_surf(subject, ('head-dense', 'head', 'bem'),
                        subjects_dir=subjects_dir)['rr']
    trans = _get_trans(trans, fro="mri", to="head")[0]
    pts = apply_trans(trans, pts)
    info_dig = get_fitting_dig(
        info, dig_kinds, exclude_frontal=exclude_frontal)
    dists = _compute_nearest(pts, info_dig, return_dists=True)[1]
    return dists
