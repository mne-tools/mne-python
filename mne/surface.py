# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from distutils.version import LooseVersion
from glob import glob
from functools import partial
import os
from os import path as op
import sys
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
from .transforms import transform_surface_to, _pol_to_cart, _cart_to_sph
from .utils import logger, verbose, get_subjects_dir, warn
from .fixes import _serialize_volume_info, _get_read_geometry, einsum


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
        Type to load. Common choices would be `'bem'` or `'head'`. We first
        try loading `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and
        then look for `'$SUBJECT*$SOURCE.fif'` in the same directory by going
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
    from .bem import read_bem_surfaces
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
        rr = rr[np.unique(ConvexHull(rr).simplices)]
        com = rr.mean(axis=0)
        xy = _pol_to_cart(_cart_to_sph(rr - com)[:, 1:][:, ::-1])
        tris = _reorder_ccw(rr, Delaunay(xy).simplices)

        surf = dict(rr=rr, tris=tris)
        complete_surface_info(surf, copy=False, verbose=False)

    # Ignore what the file says, it's in device coords and we want MRI coords
    surf['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    transform_surface_to(surf, 'head', info['dev_head_t'])
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
    becomes large (>500). This is because np.cross() methods become
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
    if max(x.size, y.size) >= 1500:
        a = x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1]
        b = x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2]
        c = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
        # Once we bump to NumPy 1.10, np.stack simplifies this
        return np.concatenate([
            a[..., np.newaxis], b[..., np.newaxis], c[..., np.newaxis]], -1)
    else:
        return np.cross(x, y)


def _fast_cross_nd_sum(a, b, c):
    """Fast cross and sum."""
    return ((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0] +
            (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1] +
            (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2])


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
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(verts, weights=tri_nn[:, idx],
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


def _triangle_coords(r, geom, best):
    """Get coordinates of a vertex projected to a triangle."""
    r1 = geom['r1'][best]
    tri_nn = geom['nn'][best]
    r12 = geom['r12'][best]
    r13 = geom['r13'][best]
    a = geom['a'][best]
    b = geom['b'][best]
    c = geom['c'][best]
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
    surf_geom = _get_tri_supp_geom(surf)
    coords = np.empty((len(rrs), 3))
    tri_idx = np.empty((len(rrs),), int)
    if method == 'accurate':
        for ri, rr in enumerate(rrs):
            # Get index of closest tri on scalp BEM to electrode position
            tri_idx[ri] = _find_nearest_tri_pt(rr, surf_geom)[2]
            # Calculate a linear interpolation between the vertex values to
            # get coords of pt projected onto closest triangle
            coords[ri] = _triangle_coords(rr, surf_geom, tri_idx[ri])
        weights = np.array([1. - coords[:, 0] - coords[:, 1], coords[:, 0],
                            coords[:, 1]])
        out = (weights, tri_idx)
        if project_rrs:  #
            out += (einsum('ij,jik->jk', weights,
                           surf['rr'][surf['tris'][tri_idx]]),)
        if return_nn:
            out += (surf_geom['nn'][tri_idx],)
    else:  # nearest neighbor
        assert project_rrs
        idx = _compute_nearest(surf['rr'], rrs)
        out = (None, None, surf['rr'][idx])
        if return_nn:
            nn = _accumulate_normals(surf['tris'], surf_geom['nn'],
                                     len(surf['rr']))
            out += (nn[idx],)
    return out


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
    surf['nn'] = _accumulate_normals(surf['tris'], surf['tri_nn'], surf['np'])
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


###############################################################################
# Handle freesurfer

def _fread3(fobj):
    """Read 3 bytes and adjust."""
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3


def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object."""
    b1, b2, b3 = np.fromfile(fobj, ">u1",
                             3 * n).reshape(-1, 3).astype(np.int).T
    return (b1 << 16) + (b2 << 8) + b3


def read_curvature(filepath):
    """Load in curavature values from the ?h.curv file."""
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            _fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
        bin_curv = 1 - np.array(curv != 0, np.int)
    return bin_curv


@verbose
def read_surface(fname, read_metadata=False, return_dict=False, verbose=None):
    """Load a Freesurfer surface mesh in triangular format.

    Parameters
    ----------
    fname : str
        The name of the file containing the surface.
    read_metadata : bool
        Read metadata as key-value pairs.
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
    ret = _get_read_geometry()(fname, read_metadata=read_metadata)
    if return_dict:
        ret += (dict(rr=ret[0], tris=ret[1], ntri=len(ret[1]), use_tris=ret[1],
                     np=len(ret[0])),)
    return ret


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
        raise ValueError('# of levels must be >= 1')

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


def write_surface(fname, coords, faces, create_stamp='', volume_info=None):
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

    See Also
    --------
    read_surface
    read_tri
    """
    try:
        import nibabel as nib
        has_nibabel = True
    except ImportError:
        has_nibabel = False
    if has_nibabel and LooseVersion(nib.__version__) > LooseVersion('2.1.0'):
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
        fid.write(np.array(coords, dtype='>f4').tostring())
        fid.write(np.array(faces, dtype='>i4').tostring())

        # Add volume info, if given
        if volume_info is not None and len(volume_info) > 0:
            fid.write(_serialize_volume_info(volume_info))


###############################################################################
# Decimation

def _decimate_surface(points, triangles, reduction):
    """Aux function."""
    if 'DISPLAY' not in os.environ and sys.platform != 'win32':
        os.environ['ETS_TOOLKIT'] = 'null'
    try:
        from tvtk.api import tvtk
        from tvtk.common import configure_input
    except ImportError:
        raise ValueError('This function requires the TVTK package to be '
                         'installed')
    if triangles.max() > len(points) - 1:
        raise ValueError('The triangles refer to undefined points. '
                         'Please check your mesh.')
    src = tvtk.PolyData(points=points, polys=triangles)
    decimate = tvtk.QuadricDecimation(target_reduction=reduction)
    configure_input(decimate, src)
    decimate.update()
    out = decimate.output
    tris = out.polys.to_array()
    # n-tuples + interleaved n-next -- reshape trick
    return out.points.to_array(), tris.reshape(tris.size // 4, 4)[:, 1:]


def decimate_surface(points, triangles, n_triangles):
    """Decimate surface data.

    .. note:: Requires TVTK to be installed for this to function.

    .. note:: If an if an odd target number was requested,
              the ``'decimation'`` algorithm used results in the
              next even number of triangles. For example a reduction request
              to 30001 triangles may result in 30000 triangles.

    Parameters
    ----------
    points : ndarray
        The surface to be decimated, a 3 x number of points array.
    triangles : ndarray
        The surface to be decimated, a 3 x number of triangles array.
    n_triangles : int
        The desired number of triangles.

    Returns
    -------
    points : ndarray
        The decimated points.
    triangles : ndarray
        The decimated triangles.
    """
    reduction = 1 - (float(n_triangles) / len(triangles))
    return _decimate_surface(points, triangles, reduction)


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
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    subjects_dir : string
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
    warn('Morph map "%s" does not exist, creating it and saving it to '
         'disk (this may take a few minutes)' % fname)
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


def _get_tri_dist(p, q, p0, q0, a, b, c, dist):
    """Get the distance to a triangle edge."""
    p1 = p - p0
    q1 = q - q0
    out = p1 * p1 * a
    out += q1 * q1 * b
    out += p1 * q1 * c
    out += dist * dist
    return np.sqrt(out, out=out)


def _get_tri_supp_geom(surf):
    """Create supplementary geometry information using tris and rrs."""
    r1 = surf['rr'][surf['tris'][:, 0], :]
    r12 = surf['rr'][surf['tris'][:, 1], :] - r1
    r13 = surf['rr'][surf['tris'][:, 2], :] - r1
    r1213 = np.array([r12, r13]).swapaxes(0, 1)
    a = einsum('ij,ij->i', r12, r12)
    b = einsum('ij,ij->i', r13, r13)
    c = einsum('ij,ij->i', r12, r13)
    mat = np.rollaxis(np.array([[b, -c], [-c, a]]), 2)
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
    nn_pts_idx = _compute_nearest(from_rr, to_rr)
    from_pt_tris = _triangle_neighbors(from_tri, len(from_rr))
    from_pt_tris = [from_pt_tris[pt_idx] for pt_idx in nn_pts_idx]

    # find triangle in which point lies and assoc. weights
    tri_inds = []
    weights = []
    tri_geom = _get_tri_supp_geom(dict(rr=from_rr, tris=from_tri))
    for pt_tris, to_pt in zip(from_pt_tris, to_rr):
        p, q, idx, dist = _find_nearest_tri_pt(to_pt, tri_geom, pt_tris,
                                               run_all=False)
        tri_inds.append(idx)
        weights.append([1. - (p + q), p, q])

    nn_idx = from_tri[tri_inds]
    weights = np.array(weights)

    row_ind = np.repeat(np.arange(len(to_rr)), 3)
    this_map = csr_matrix((weights.ravel(), (row_ind, nn_idx.ravel())),
                          shape=(len(to_rr), len(from_rr)))
    return this_map


def _find_nearest_tri_pt(rr, tri_geom, pt_tris=None, run_all=True):
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

    # This einsum is equivalent to doing:
    # pqs = np.array([np.dot(x, y) for x, y in zip(r1213, r1-to_pt)])
    if pt_tris is None:  # use all points
        pt_tris = slice(len(tri_geom['r1']))
    rrs = rr - tri_geom['r1'][pt_tris]
    tri_nn = tri_geom['nn'][pt_tris]
    vect = einsum('ijk,ik->ij', tri_geom['r1213'][pt_tris], rrs)
    mats = tri_geom['mat'][pt_tris]
    # This einsum is equivalent to doing:
    # pqs = np.array([np.dot(m, v) for m, v in zip(mats, vect)]).T
    pqs = einsum('ijk,ik->ji', mats, vect)
    found = False
    dists = np.sum(rrs * tri_nn, axis=1)

    # There can be multiple (sadness), find closest
    idx = np.where(np.all(pqs >= 0., axis=0))[0]
    idx = idx[np.where(np.all(pqs[:, idx] <= 1., axis=0))[0]]
    idx = idx[np.where(np.sum(pqs[:, idx], axis=0) < 1.)[0]]
    dist = np.inf
    if len(idx) > 0:
        found = True
        pt = idx[np.argmin(np.abs(dists[idx]))]
        p, q = pqs[:, pt]
        dist = dists[pt]
        # re-reference back to original numbers
        if not isinstance(pt_tris, slice):
            pt = pt_tris[pt]

    if found is False or run_all is True:
        # don't include ones that we might have found before
        # these are the ones that we want to check thesides of
        s = np.setdiff1d(np.arange(dists.shape[0]), idx)
        # Tough: must investigate the sides
        use_pt_tris = s if isinstance(pt_tris, slice) else pt_tris[s]
        pp, qq, ptt, distt = _nearest_tri_edge(use_pt_tris, rr, pqs[:, s],
                                               dists[s], tri_geom)
        if np.abs(distt) < np.abs(dist):
            p, q, pt, dist = pp, qq, ptt, distt
    return p, q, pt, dist


def _nearest_tri_edge(pt_tris, to_pt, pqs, dist, tri_geom):
    """Get nearest location from a point to the edge of a set of triangles."""
    # We might do something intelligent here. However, for now
    # it is ok to do it in the hard way
    aa = tri_geom['a'][pt_tris]
    bb = tri_geom['b'][pt_tris]
    cc = tri_geom['c'][pt_tris]
    pp = pqs[0]
    qq = pqs[1]
    # Find the nearest point from a triangle:
    #   Side 1 -> 2
    p0 = np.minimum(np.maximum(pp + 0.5 * (qq * cc) / aa,
                               0.0), 1.0)
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
    pp = np.r_[p0, p1, p2]
    qq = np.r_[q0, q1, q2]
    dists = np.r_[dist0, dist1, dist2]
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
        raise ValueError('Cannot compute connectivity on a selection of '
                         'triangles.')

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
        Mesh triangulation
    vert : array (n_vert x 3)
        Vertex locations

    Returns
    -------
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
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

    Notes
    -----
    .. versionadded:: 0.13.0

    See Also
    --------
    read_surface
    write_surface
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
    rr = np.array([np.array([float(v) for v in l.split()])[inds]
                   for l in lines[1:n_nodes + 1]])
    tris = np.array([np.array([int(v) for v in l.split()])[inds]
                     for l in lines[n_nodes + 2:n_nodes + 2 + n_tris]])
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


def _get_solids(tri_rrs, fros):
    """Compute _sum_solids_div total angle in chunks."""
    # NOTE: This incorporates the division by 4PI that used to be separate
    # for tri_rr in tri_rrs:
    #     v1 = fros - tri_rr[0]
    #     v2 = fros - tri_rr[1]
    #     v3 = fros - tri_rr[2]
    #     triple = np.sum(fast_cross_3d(v1, v2) * v3, axis=1)
    #     l1 = np.sqrt(np.sum(v1 * v1, axis=1))
    #     l2 = np.sqrt(np.sum(v2 * v2, axis=1))
    #     l3 = np.sqrt(np.sum(v3 * v3, axis=1))
    #     s = (l1 * l2 * l3 +
    #          np.sum(v1 * v2, axis=1) * l3 +
    #          np.sum(v1 * v3, axis=1) * l2 +
    #          np.sum(v2 * v3, axis=1) * l1)
    #     tot_angle -= np.arctan2(triple, s)

    # This is the vectorized version, but with a slicing heuristic to
    # prevent memory explosion
    tot_angle = np.zeros((len(fros)))
    slices = np.r_[np.arange(0, len(fros), 100), [len(fros)]]
    for i1, i2 in zip(slices[:-1], slices[1:]):
        # shape (3 verts, n_tri, n_fro, 3 X/Y/Z)
        vs = (fros[np.newaxis, np.newaxis, i1:i2] -
              tri_rrs.transpose([1, 0, 2])[:, :, np.newaxis])
        triples = _fast_cross_nd_sum(vs[0], vs[1], vs[2])
        ls = np.linalg.norm(vs, axis=3)
        ss = np.prod(ls, axis=0)
        ss += einsum('ijk,ijk,ij->ij', vs[0], vs[1], ls[2])
        ss += einsum('ijk,ijk,ij->ij', vs[0], vs[2], ls[1])
        ss += einsum('ijk,ijk,ij->ij', vs[1], vs[2], ls[0])
        tot_angle[i1:i2] = -np.sum(np.arctan2(triples, ss), axis=0)
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
