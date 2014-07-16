# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from .externals.six import string_types
import os
from os import path as op
import sys
from struct import pack
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from fnmatch import fnmatch

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tree import dir_tree_find
from .io.tag import find_tag
from .io.write import (write_int, write_float, write_float_matrix,
                       write_int_matrix, start_file, end_block,
                       start_block, end_file, write_string,
                       write_float_sparse_rcs)
from .channels import _get_meg_system
from .transforms import transform_surface_to
from .utils import logger, verbose, get_subjects_dir


##############################################################################
# BEM

@verbose
def read_bem_surfaces(fname, add_geom=False, s_id=None, verbose=None):
    """Read the BEM surfaces from a FIF file

    Parameters
    ----------
    fname : string
        The name of the file containing the surfaces.
    add_geom : bool, optional (default False)
        If True add geometry information to the surfaces.
    s_id : int | None
        If int, only read and return the surface with the given s_id.
        An error will be raised if it doesn't exist. If None, all
        surfaces are read and returned.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    surf: list | dict
        A list of dictionaries that each contain a surface. If s_id
        is not None, only the requested surface will be returned.
    """
    #
    #   Default coordinate frame
    #
    coord_frame = FIFF.FIFFV_COORD_MRI
    #
    #   Open the file, create directory
    #
    fid, tree, _ = fiff_open(fname)
    #
    #   Find BEM
    #
    bem = dir_tree_find(tree, FIFF.FIFFB_BEM)
    if bem is None:
        fid.close()
        raise ValueError('BEM data not found')

    bem = bem[0]
    #
    #   Locate all surfaces
    #
    bemsurf = dir_tree_find(bem, FIFF.FIFFB_BEM_SURF)
    if bemsurf is None:
        fid.close()
        raise ValueError('BEM surface data not found')

    logger.info('    %d BEM surfaces found' % len(bemsurf))
    #
    #   Coordinate frame possibly at the top level
    #
    tag = find_tag(fid, bem, FIFF.FIFF_BEM_COORD_FRAME)
    if tag is not None:
        coord_frame = tag.data
    #
    #   Read all surfaces
    #
    if s_id is not None:
        surfs = [_read_bem_surface(fid, bsurf, coord_frame, s_id)
                 for bsurf in bemsurf]
        surfs = [s for s in surfs if s is not None]
        if not len(surfs) == 1:
            raise ValueError('surface with id %d not found' % s_id)
        fid.close()
        return surfs[0]

    surf = []
    for bsurf in bemsurf:
        logger.info('    Reading a surface...')
        this = _read_bem_surface(fid, bsurf, coord_frame)
        logger.info('[done]')
        if add_geom:
            _complete_surface_info(this)
        surf.append(this)

    logger.info('    %d BEM surfaces read' % len(surf))

    fid.close()

    return surf


def _read_bem_surface(fid, this, def_coord_frame, s_id=None):
    """Read one bem surface
    """
    res = dict()
    #
    #   Read all the interesting stuff
    #
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_ID)

    if tag is None:
        res['id'] = FIFF.FIFFV_BEM_SURF_ID_UNKNOWN
    else:
        res['id'] = int(tag.data)

    if s_id is not None:
        if res['id'] != s_id:
            return None

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SIGMA)
    if tag is None:
        res['sigma'] = 1.0
    else:
        res['sigma'] = float(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NNODE)
    if tag is None:
        fid.close()
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NTRI)
    if tag is None:
        fid.close()
        raise ValueError('Number of triangles not found')
    else:
        res['ntri'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_BEM_COORD_FRAME)
        if tag is None:
            res['coord_frame'] = def_coord_frame
        else:
            res['coord_frame'] = tag.data
    else:
        res['coord_frame'] = tag.data
    #
    #   Vertices, normals, and triangles
    #
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NODES)
    if tag is None:
        fid.close()
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float)  # XXX : double because of mayavi bug
    if res['rr'].shape[0] != res['np']:
        fid.close()
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        tag = tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NORMALS)
    if tag is None:
        res['nn'] = []
    else:
        res['nn'] = tag.data
        if res['nn'].shape[0] != res['np']:
            fid.close()
            raise ValueError('Vertex normal information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_TRIANGLES)
    if tag is None:
        fid.close()
        raise ValueError('Triangulation not found')

    res['tris'] = tag.data - 1  # index start at 0 in Python
    if res['tris'].shape[0] != res['ntri']:
        fid.close()
        raise ValueError('Triangulation information is incorrect')

    return res


@verbose
def read_bem_solution(fname, verbose=None):
    """Read the BEM solution from a file

    Parameters
    ----------
    fname : string
        The file containing the BEM solution.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    bem : dict
        The BEM solution.
    """
    logger.info('Loading surfaces...')
    bem_surfs = read_bem_surfaces(fname, add_geom=True, verbose=False)
    if len(bem_surfs) == 3:
        logger.info('Three-layer model surfaces loaded.')
        needed = np.array([FIFF.FIFFV_BEM_SURF_ID_HEAD,
                           FIFF.FIFFV_BEM_SURF_ID_SKULL,
                           FIFF.FIFFV_BEM_SURF_ID_BRAIN])
        if not all([x['id'] in needed for x in bem_surfs]):
            raise RuntimeError('Could not find necessary BEM surfaces')
        # reorder surfaces as necessary (shouldn't need to?)
        reorder = [None] * 3
        for x in bem_surfs:
            reorder[np.where(x['id'] == needed)[0][0]] = x
        bem_surfs = reorder
    elif len(bem_surfs) == 1:
        if not bem_surfs[0]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN:
            raise RuntimeError('BEM Surfaces not found')
        logger.info('Homogeneous model surface loaded.')

    # convert from surfaces to solution
    bem = dict(surfs=bem_surfs)
    logger.info('\nLoading the solution matrix...\n')
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find the BEM data
        nodes = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if len(nodes) == 0:
            raise RuntimeError('No BEM data in %s' % fname)
        bem_node = nodes[0]

        # Approximation method
        tag = find_tag(f, bem_node, FIFF.FIFF_BEM_APPROX)
        method = tag.data[0]
        if method == FIFF.FIFFV_BEM_APPROX_CONST:
            method = 'constant collocation'
        elif method == FIFF.FIFFV_BEM_APPROX_LINEAR:
            method = 'linear collocation'
        else:
            raise RuntimeError('Cannot handle BEM approximation method : %d'
                               % method)

        tag = find_tag(fid, bem_node, FIFF.FIFF_BEM_POT_SOLUTION)
        dims = tag.data.shape
        if len(dims) != 2:
            raise RuntimeError('Expected a two-dimensional solution matrix '
                               'instead of a %d dimensional one' % dims[0])

        dim = 0
        for surf in bem['surfs']:
            if method == 'linear collocation':
                dim += surf['np']
            else:
                dim += surf['ntri']

        if dims[0] != dim or dims[1] != dim:
            raise RuntimeError('Expected a %d x %d solution matrix instead of '
                               'a %d x %d one' % (dim, dim, dims[1], dims[0]))
        sol = tag.data
        nsol = dims[0]

    # Gamma factors and multipliers
    bem['sigma'] = np.array([surf['sigma'] for surf in bem['surfs']])
    # Dirty trick for the zero conductivity outside
    sigma = np.r_[0.0, bem['sigma']]
    bem['source_mult'] = 2.0 / (sigma[1:] + sigma[:-1])
    bem['field_mult'] = sigma[1:] - sigma[:-1]
    # make sure subsequent "zip"s work correctly
    assert len(bem['surfs']) == len(bem['field_mult'])
    bem['gamma'] = ((sigma[1:] - sigma[:-1])[np.newaxis, :] /
                    (sigma[1:] + sigma[:-1])[:, np.newaxis])
    bem['sol_name'] = fname
    bem['solution'] = sol
    bem['nsol'] = nsol
    bem['bem_method'] = method
    logger.info('Loaded %s BEM solution from %s', bem['bem_method'], fname)
    return bem


###############################################################################
# AUTOMATED SURFACE FINDING

def get_head_surf(subject, source='bem', subjects_dir=None):
    """Load the subject head surface

    Parameters
    ----------
    subject : str
        Subject name.
    source : str
        Type to load. Common choices would be `'bem'` or `'head'`. We first
        try loading `'$SUBJECTS_DIR/$SUBJECT/bem/$SUBJECT-$SOURCE.fif'`, and
        then look for `'$SUBJECT*$SOURCE.fif'` in the same directory.
    subjects_dir : str, or None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.

    Returns
    -------
    surf : dict
        The head surface.
    """
    # Load the head surface from the BEM
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    # use realpath to allow for linked surfaces (c.f. MNE manual 196-197)
    this_head = op.realpath(op.join(subjects_dir, subject, 'bem',
                                    '%s-%s.fif' % (subject, source)))
    if not op.isfile(this_head):
        # let's do a more sophisticated search
        this_head = None
        path = op.join(subjects_dir, subject, 'bem')
        if not op.isdir(path):
            raise IOError('Subject bem directory "%s" does not exist'
                          % path)
        files = os.listdir(path)
        for fname in files:
            if fnmatch(fname, '%s*%s.fif' % (subject, source)):
                this_head = op.join(path, fname)
                break
        if this_head is None:
            raise IOError('No file matching "%s*%s" found'
                          % (subject, source))
    surf = read_bem_surfaces(this_head, True,
                             FIFF.FIFFV_BEM_SURF_ID_HEAD)
    return surf


def get_meg_helmet_surf(info, trans=None):
    """Load the MEG helmet associated with the MEG sensors

    Parameters
    ----------
    info : instance of io.meas_info.Info
        Measurement info.
    trans : dict
        The head<->MRI transformation, usually obtained using
        read_trans(). Can be None, in which case the surface will
        be in head coordinates instead of MRI coordinates.

    Returns
    -------
    surf : dict
        The MEG helmet as a surface.
    """
    system = _get_meg_system(info)
    fname = op.join(op.split(__file__)[0], 'data', 'helmets',
                    system + '.fif.gz')
    surf = read_bem_surfaces(fname, False, FIFF.FIFFV_MNE_SURF_MEG_HELMET)

    # Ignore what the file says, it's in device coords and we want MRI coords
    surf['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    transform_surface_to(surf, 'head', info['dev_head_t'])
    if trans is not None:
        transform_surface_to(surf, 'mri', trans)
    return surf


###############################################################################
# EFFICIENCY UTILITIES

def fast_cross_3d(x, y):
    """Compute cross product between list of 3D vectors

    Much faster than np.cross() when the number of cross products
    becomes large (>500). This is because np.cross() methods become
    less memory efficient at this stage.

    Parameters
    ----------
    x : array
        Input array 1.
    y : array
        Input array 2.

    Returns
    -------
    z : array
        Cross product of x and y.

    Notes
    -----
    x and y must both be 2D row vectors. One must have length 1, or both
    lengths must match.
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == 3
    assert y.shape[1] == 3
    assert (x.shape[0] == 1 or y.shape[0] == 1) or x.shape[0] == y.shape[0]
    if max([x.shape[0], y.shape[0]]) >= 500:
        return np.c_[x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
                     x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
                     x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]]
    else:
        return np.cross(x, y)


def _accumulate_normals(tris, tri_nn, npts):
    """Efficiently accumulate triangle normals"""
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
    """Efficiently compute vertex neighboring triangles"""
    # this code replaces the following, but is faster (vectorized):
    #
    # this['neighbor_tri'] = [list() for _ in xrange(this['np'])]
    # for p in xrange(this['ntri']):
    #     verts = this['tris'][p]
    #     this['neighbor_tri'][verts[0]].append(p)
    #     this['neighbor_tri'][verts[1]].append(p)
    #     this['neighbor_tri'][verts[2]].append(p)
    # this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]
    #
    verts = tris.ravel()
    counts = np.bincount(verts, minlength=npts)
    reord = np.argsort(verts)
    tri_idx = np.unravel_index(reord, (len(tris), 3))[0]
    idx = np.cumsum(np.r_[0, counts])
    # the sort below slows it down a bit, but is needed for equivalence
    neighbor_tri = [np.sort(tri_idx[v1:v2])
                    for v1, v2 in zip(idx[:-1], idx[1:])]
    return neighbor_tri


def _triangle_coords(r, geom, best):
    """Get coordinates of a vertex projected to a triangle"""
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


def _complete_surface_info(this, do_neighbor_vert=False):
    """Complete surface info"""
    # based on mne_source_space_add_geometry_info() in mne_add_geometry_info.c

    #   Main triangulation [mne_add_triangle_data()]
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    size = np.sqrt(np.sum(this['tri_nn'] ** 2, axis=1))
    this['tri_area'] = size / 2.0
    zidx = np.where(size == 0)[0]
    for idx in zidx:
        logger.info('    Warning: zero size triangle # %s' % idx)
    size[zidx] = 1.0  # prevent ugly divide-by-zero
    this['tri_nn'] /= size[:, None]

    #    Find neighboring triangles, accumulate vertex normals, normalize
    logger.info('    Triangle neighbors and vertex normals...')
    this['neighbor_tri'] = _triangle_neighbors(this['tris'], this['np'])
    this['nn'] = _accumulate_normals(this['tris'], this['tri_nn'], this['np'])
    _normalize_vectors(this['nn'])

    #   Check for topological defects
    idx = np.where([len(n) == 0 for n in this['neighbor_tri']])[0]
    if len(idx) > 0:
        logger.info('    Vertices [%s] do not have any neighboring'
                    'triangles!' % ','.join([str(ii) for ii in idx]))
    idx = np.where([len(n) < 3 for n in this['neighbor_tri']])[0]
    if len(idx) > 0:
        logger.info('    Vertices [%s] have fewer than three neighboring '
                    'tris, omitted' % ','.join([str(ii) for ii in idx]))
    for k in idx:
        this['neighbor_tri'] = np.array([], int)

    #   Determine the neighboring vertices and fix errors
    if do_neighbor_vert is True:
        this['neighbor_vert'] = [_get_surf_neighbors(this, k)
                                 for k in range(this['np'])]

    return this


def _get_surf_neighbors(surf, k):
    """Calculate the surface neighbors based on triangulation"""
    verts = np.concatenate([surf['tris'][nt]
                            for nt in surf['neighbor_tri'][k]])
    verts = np.setdiff1d(verts, [k], assume_unique=False)
    if np.any(verts >= surf['np']):
        raise RuntimeError
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
    """Normalize surface vertices"""
    size = np.sqrt(np.sum(rr * rr, axis=1))
    size[size == 0] = 1.0  # avoid divide-by-zero
    rr /= size[:, np.newaxis]  # operate in-place


def _compute_nearest(xhs, rr, use_balltree=True, return_dists=False):
    """Find nearest neighbors

    Note: The rows in xhs and rr must all be unit-length vectors, otherwise
    the result will be incorrect.

    Parameters
    ----------
    xhs : array, shape=(n_samples, n_dim)
        Points of data set.
    rr : array, shape=(n_query, n_dim)
        Points to find nearest neighbors for.
    use_balltree : bool
        Use fast BallTree based search from scikit-learn. If scikit-learn
        is not installed it will fall back to the slow brute force search.

    Returns
    -------
    nearest : array, shape=(n_query,)
        Index of nearest neighbor in xhs for every point in rr.
    """
    if use_balltree:
        try:
            from sklearn.neighbors import BallTree
        except ImportError:
            logger.info('Nearest-neighbor searches will be significantly '
                        'faster if scikit-learn is installed.')
            use_balltree = False

    if use_balltree is True:
        ball_tree = BallTree(xhs)
        if return_dists:
            out = ball_tree.query(rr, k=1, return_distance=True)
            return out[1][:, 0], out[0][:, 0]
        else:
            nearest = ball_tree.query(rr, k=1, return_distance=False)[:, 0]
            return nearest
    else:
        if return_dists:
            nearest = list()
            dists = list()
            for r in rr:
                d = cdist(r[np.newaxis, :], xhs)
                idx = np.argmin(d)
                nearest.append(idx)
                dists.append(d[0, idx])
            return (np.array(nearest), np.array(dists))
        else:
            nearest = np.array([np.argmin(cdist(r[np.newaxis, :], xhs))
                                for r in rr])
            return nearest


###############################################################################
# Handle freesurfer

def _fread3(fobj):
    """Docstring"""
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
def read_surface(fname, verbose=None):
    """Load a Freesurfer surface mesh in triangular format

    Parameters
    ----------
    fname : str
        The name of the file containing the surface.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    rr : array, shape=(n_vertices, 3)
        Coordinate points.
    tris : int array, shape=(n_faces, 3)
        Triangulation (each line contains indexes for three points which
        together form a face).
    """
    TRIANGLE_MAGIC = 16777214
    QUAD_MAGIC = 16777215
    NEW_QUAD_MAGIC = 16777213
    with open(fname, "rb", buffering=0) as fobj:  # buffering=0 for np bug
        magic = _fread3(fobj)
        if (magic == QUAD_MAGIC) or (magic == NEW_QUAD_MAGIC):  # Quad file or new quad
            create_stamp = ''
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            if magic == QUAD_MAGIC:
                coords = np.fromfile(fobj, ">i2", nvert * 3).astype(np.float) / 100.
            else:
                coords = np.fromfile(fobj, ">f4", nvert * 3).astype(np.float)

            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            #
            #   Face splitting follows
            #
            faces = np.zeros((2 * nquad, 3), dtype=np.int)
            nface = 0
            for quad in quads:
                if (quad[0] % 2) == 0:
                    faces[nface] = quad[0], quad[1], quad[3]
                    nface += 1
                    faces[nface] = quad[2], quad[3], quad[1]
                    nface += 1
                else:
                    faces[nface] = quad[0], quad[1], quad[2]
                    nface += 1
                    faces[nface] = quad[0], quad[2], quad[3]
                    nface += 1

        elif magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline()
            _ = fobj.readline()  # analysis:ignore
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            #raise RuntimeError
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)
        else:
            raise ValueError("%s does not appear to be a Freesurfer surface"
                             % fname)
        logger.info('Triangle file: %s nvert = %s ntri = %s'
                    % (create_stamp.strip(), len(coords), len(faces)))

    coords = coords.astype(np.float)  # XXX: due to mayavi bug on mac 32bits
    return coords, faces


@verbose
def _read_surface_geom(fname, add_geom=True, norm_rr=False, verbose=None):
    """Load the surface as dict, optionally add the geometry information"""
    # based on mne_load_surface_geom() in mne_surface_io.c
    if isinstance(fname, string_types):
        rr, tris = read_surface(fname)  # mne_read_triangle_file()
        nvert = len(rr)
        ntri = len(tris)
        s = dict(rr=rr, tris=tris, use_tris=tris, ntri=ntri,
                 np=nvert)
    elif isinstance(fname, dict):
        s = fname
    else:
        raise RuntimeError('fname cannot be understood as str or dict')
    if add_geom is True:
        s = _complete_surface_info(s)
    if norm_rr is True:
        _normalize_vectors(s['rr'])
    return s


##############################################################################
# SURFACE CREATION

def _get_ico_surface(grade):
    """Return an icosahedral surface of the desired grade"""
    # always use verbose=False since users don't need to know we're pulling
    # these from a file
    ico_file_name = op.join(op.dirname(__file__), 'data',
                            'icos.fif.gz')
    ico = read_bem_surfaces(ico_file_name, s_id=9000 + grade, verbose=False)
    return ico


def _tessellate_sphere_surf(level, rad=1.0):
    """Return a surface structure instead of the details"""
    rr, tris = _tessellate_sphere(level)
    npt = len(rr)  # called "npt" instead of "np" because of numpy...
    ntri = len(tris)
    nn = rr.copy()
    rr *= rad
    s = dict(rr=rr, np=npt, tris=tris, use_tris=tris, ntri=ntri, nuse=np,
             nn=nn, inuse=np.ones(npt, int))
    return s


def _norm_midpt(ai, bi, rr):
    a = np.array([rr[aii] for aii in ai])
    b = np.array([rr[bii] for bii in bi])
    c = (a + b) / 2.
    return c / np.sqrt(np.sum(c ** 2, 1))[:, np.newaxis]


def _tessellate_sphere(mylevel):
    """Create a tessellation of a unit sphere"""
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
        """
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
             /\    /\	       [0,b,a]
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


def _create_surf_spacing(surf, hemi, subject, stype, sval, ico_surf,
                         subjects_dir):
    """Load a surf and use the subdivided icosahedron to get points"""
    # Based on load_source_space_surf_spacing() in load_source_space.c
    surf = _read_surface_geom(surf)

    if stype in ['ico', 'oct']:
        ### from mne_ico_downsample.c ###
        surf_name = op.join(subjects_dir, subject, 'surf', hemi + '.sphere')
        logger.info('Loading geometry from %s...' % surf_name)
        from_surf = _read_surface_geom(surf_name, norm_rr=True, add_geom=False)
        if not len(from_surf['rr']) == surf['np']:
            raise RuntimeError('Mismatch between number of surface vertices, '
                               'possible parcellation error?')
        _normalize_vectors(ico_surf['rr'])

        # Make the maps
        logger.info('Mapping %s %s -> %s (%d) ...'
                    % (hemi, subject, stype, sval))
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
    else:  # use_all is True
        surf['inuse'] = np.ones(surf['np'], int)
        surf['use_tris'] = None
    if surf['use_tris'] is not None:
        surf['nuse_tri'] = len(surf['use_tris'])
    else:
        surf['nuse_tri'] = 0
    surf['nuse'] = np.sum(surf['inuse'])
    surf['vertno'] = np.where(surf['inuse'])[0]

    # set some final params
    inds = np.arange(surf['np'])
    sizes = np.sqrt(np.sum(surf['nn'] ** 2, axis=1))
    surf['nn'][inds] = surf['nn'][inds] / sizes[:, np.newaxis]
    surf['inuse'][sizes <= 0] = False
    surf['nuse'] = np.sum(surf['inuse'])
    surf['subject_his_id'] = subject
    return surf


def write_surface(fname, coords, faces, create_stamp=''):
    """Write a triangular Freesurfer surface mesh

    Accepts the same data format as is returned by read_surface().

    Parameters
    ----------
    fname : str
        File to write.
    coords : array, shape=(n_vertices, 3)
        Coordinate points.
    faces : int array, shape=(n_faces, 3)
        Triangulation (each line contains indexes for three points which
        together form a face).
    create_stamp : str
        Comment that is written to the beginning of the file. Can not contain
        line breaks.
    """
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


###############################################################################
# Write

def write_bem_surface(fname, surf):
    """Write one bem surface

    Parameters
    ----------
    fname : string
        File to write
    surf : dict
        A surface structured as obtained with read_bem_surfaces
    """

    # Create the file and save the essentials
    fid = start_file(fname)

    start_block(fid, FIFF.FIFFB_BEM)
    start_block(fid, FIFF.FIFFB_BEM_SURF)

    write_int(fid, FIFF.FIFF_BEM_SURF_ID, surf['id'])
    write_float(fid, FIFF.FIFF_BEM_SIGMA, surf['sigma'])
    write_int(fid, FIFF.FIFF_BEM_SURF_NNODE, surf['np'])
    write_int(fid, FIFF.FIFF_BEM_SURF_NTRI, surf['ntri'])
    write_int(fid, FIFF.FIFF_BEM_COORD_FRAME, surf['coord_frame'])
    write_float_matrix(fid, FIFF.FIFF_BEM_SURF_NODES, surf['rr'])

    if 'nn' in surf and surf['nn'] is not None and len(surf['nn']) > 0:
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, surf['nn'])

    # index start at 0 in Python
    write_int_matrix(fid, FIFF.FIFF_BEM_SURF_TRIANGLES, surf['tris'] + 1)

    end_block(fid, FIFF.FIFFB_BEM_SURF)
    end_block(fid, FIFF.FIFFB_BEM)

    end_file(fid)


def _decimate_surface(points, triangles, reduction):
    """Aux function"""
    if 'DISPLAY' not in os.environ and sys.platform != 'win32':
        os.environ['ETS_TOOLKIT'] = 'null'
    try:
        from tvtk.api import tvtk
    except ImportError:
        raise ValueError('This function requires the TVTK package to be '
                         'installed')
    if triangles.max() > len(points) - 1:
        raise ValueError('The triangles refer to undefined points. '
                         'Please check your mesh.')
    src = tvtk.PolyData(points=points, polys=triangles)
    decimate = tvtk.QuadricDecimation(input=src, target_reduction=reduction)
    decimate.update()
    out = decimate.output
    tris = out.polys.to_array()
    # n-tuples + interleaved n-next -- reshape trick
    return out.points.to_array(), tris.reshape(tris.size / 4, 4)[:, 1:]


def decimate_surface(points, triangles, n_triangles):
    """ Decimate surface data

    Note. Requires TVTK to be installed for this to function.

    Note. If an if an odd target number was requested,
    the ``quadric decimation`` algorithm used results in the
    next even number of triangles. For example a reduction request to 30001
    triangles will result in 30000 triangles.

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

@verbose
def read_morph_map(subject_from, subject_to, subjects_dir=None,
                   verbose=None):
    """Read morph map

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    left_map, right_map : sparse matrix
        The morph maps for the 2 hemispheres.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # First check for morph-map dir existence
    mmap_dir = op.join(subjects_dir, 'morph-maps')
    if not op.isdir(mmap_dir):
        try:
            os.mkdir(mmap_dir)
        except:
            logger.warning('Could not find or make morph map directory "%s"'
                           % mmap_dir)

    # Does the file exist
    fname = op.join(mmap_dir, '%s-%s-morph.fif' % (subject_from, subject_to))
    if not op.exists(fname):
        fname = op.join(mmap_dir, '%s-%s-morph.fif'
                        % (subject_to, subject_from))
        if not op.exists(fname):
            logger.warning('Morph map "%s" does not exist, '
                           'creating it and saving it to disk (this may take '
                           'a few minutes)' % fname)
            logger.info('Creating morph map %s -> %s'
                        % (subject_from, subject_to))
            mmap_1 = _make_morph_map(subject_from, subject_to, subjects_dir)
            logger.info('Creating morph map %s -> %s'
                        % (subject_to, subject_from))
            mmap_2 = _make_morph_map(subject_to, subject_from, subjects_dir)
            try:
                _write_morph_map(fname, subject_from, subject_to,
                                 mmap_1, mmap_2)
            except Exception as exp:
                logger.warning('Could not write morph-map file "%s" '
                               '(error: %s)' % (fname, exp))
            return mmap_1

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

    if left_map is None:
        raise ValueError('Left hemisphere map not found in %s' % fname)

    if right_map is None:
        raise ValueError('Left hemisphere map not found in %s' % fname)

    return left_map, right_map


def _write_morph_map(fname, subject_from, subject_to, mmap_1, mmap_2):
    """Write a morph map to disk"""
    fid = start_file(fname)
    assert len(mmap_1) == 2
    assert len(mmap_2) == 2
    hemis = [FIFF.FIFFV_MNE_SURF_LEFT_HEMI, FIFF.FIFFV_MNE_SURF_RIGHT_HEMI]
    for m, hemi in zip(mmap_1, hemis):
        start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_from)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_to)
        write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
        end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
    for m, hemi in zip(mmap_2, hemis):
        start_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_FROM, subject_to)
        write_string(fid, FIFF.FIFF_MNE_MORPH_MAP_TO, subject_from)
        write_int(fid, FIFF.FIFF_MNE_HEMI, hemi)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_MORPH_MAP, m)
        end_block(fid, FIFF.FIFFB_MNE_MORPH_MAP)
    end_file(fid)


def _get_tri_dist(p, q, p0, q0, a, b, c, dist):
    """Auxiliary function for getting the distance to a triangle edge"""
    return np.sqrt((p - p0) * (p - p0) * a +
                   (q - q0) * (q - q0) * b +
                   (p - p0) * (q - q0) * c +
                   dist * dist)


def _get_tri_supp_geom(tris, rr):
    """Create supplementary geometry information using tris and rrs"""
    r1 = rr[tris[:, 0], :]
    r12 = rr[tris[:, 1], :] - r1
    r13 = rr[tris[:, 2], :] - r1
    r1213 = np.array([r12, r13]).swapaxes(0, 1)
    a = np.sum(r12 * r12, axis=1)
    b = np.sum(r13 * r13, axis=1)
    c = np.sum(r12 * r13, axis=1)
    mat = np.rollaxis(np.array([[b, -c], [-c, a]]), 2)
    mat /= (a * b - c * c)[:, np.newaxis, np.newaxis]
    nn = fast_cross_3d(r12, r13)
    _normalize_vectors(nn)
    return dict(r1=r1, r12=r12, r13=r13, r1213=r1213,
                a=a, b=b, c=c, mat=mat, nn=nn)


@verbose
def _make_morph_map(subject_from, subject_to, subjects_dir=None):
    """Construct morph map from one subject to another

    Note that this is close, but not exactly like the C version.
    For example, parts are more accurate due to double precision,
    so expect some small morph-map differences!

    Note: This seems easily parallelizable, but the overhead
    of pickling all the data structures makes it less efficient
    than just running on a single core :(
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    morph_maps = list()

    # add speedy short-circuit for self-maps
    if subject_from == subject_to:
        for hemi in ['lh', 'rh']:
            fname = op.join(subjects_dir, subject_from, 'surf',
                            '%s.sphere.reg' % hemi)
            from_pts = read_surface(fname, verbose=False)[0]
            n_pts = len(from_pts)
            morph_maps.append(sparse.eye(n_pts, n_pts, format='csr'))
        return morph_maps

    for hemi in ['lh', 'rh']:
        # load surfaces and normalize points to be on unit sphere
        fname = op.join(subjects_dir, subject_from, 'surf',
                        '%s.sphere.reg' % hemi)
        from_pts, from_tris = read_surface(fname, verbose=False)
        n_from_pts = len(from_pts)
        _normalize_vectors(from_pts)
        tri_geom = _get_tri_supp_geom(from_tris, from_pts)

        fname = op.join(subjects_dir, subject_to, 'surf',
                        '%s.sphere.reg' % hemi)
        to_pts = read_surface(fname, verbose=False)[0]
        n_to_pts = len(to_pts)
        _normalize_vectors(to_pts)

        # from surface: get nearest neighbors, find triangles for each vertex
        nn_pts_idx = _compute_nearest(from_pts, to_pts)
        from_pt_tris = _triangle_neighbors(from_tris, len(from_pts))
        from_pt_tris = [from_pt_tris[pt_idx] for pt_idx in nn_pts_idx]

        # find triangle in which point lies and assoc. weights
        nn_tri_inds = []
        nn_tris_weights = []
        for pt_tris, to_pt in zip(from_pt_tris, to_pts):
            p, q, idx, dist = _find_nearest_tri_pt(pt_tris, to_pt, tri_geom)
            nn_tri_inds.append(idx)
            nn_tris_weights.extend([1. - (p + q), p, q])

        nn_tris = from_tris[nn_tri_inds]
        row_ind = np.repeat(np.arange(n_to_pts), 3)
        this_map = sparse.csr_matrix((nn_tris_weights,
                                     (row_ind, nn_tris.ravel())),
                                     shape=(n_to_pts, n_from_pts))
        morph_maps.append(this_map)

    return morph_maps


def _find_nearest_tri_pt(pt_tris, to_pt, tri_geom, run_all=False):
    """Find nearest point mapping to a set of triangles

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
    r1 = tri_geom['r1'][pt_tris]
    rrs = to_pt - r1
    tri_nn = tri_geom['nn'][pt_tris]
    vect = np.einsum('ijk,ik->ij', tri_geom['r1213'][pt_tris], rrs)
    mats = tri_geom['mat'][pt_tris]
    # This einsum is equivalent to doing:
    # pqs = np.array([np.dot(m, v) for m, v in zip(mats, vect)]).T
    pqs = np.einsum('ijk,ik->ji', mats, vect)
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
        pt = pt_tris[pt]

    if found is False or run_all is True:
        # don't include ones that we might have found before
        s = np.setdiff1d(np.arange(len(pt_tris)), idx)  # ones to check sides
        # Tough: must investigate the sides
        pp, qq, ptt, distt = _nearest_tri_edge(pt_tris[s], to_pt, pqs[:, s],
                                               dists[s], tri_geom)
        if np.abs(distt) < np.abs(dist):
            p, q, pt, dist = pp, qq, ptt, distt
    return p, q, pt, dist


def _nearest_tri_edge(pt_tris, to_pt, pqs, dist, tri_geom):
    """Get nearest location from a point to the edge of a set of triangles"""
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
    t1 = (0.5 * ((2.0 * aa - cc) * (1.0 - pp)
                 + (2.0 * bb - cc) * qq) / (aa + bb - cc))
    t1 = np.minimum(np.maximum(t1, 0.0), 1.0)
    p1 = 1.0 - t1
    q1 = t1
    #   Side 1 -> 3
    q2 = np.minimum(np.maximum(qq + 0.5 * (pp * cc)
                               / bb, 0.0), 1.0)
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
