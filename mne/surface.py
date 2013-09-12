# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os
from os import path as op
import sys
from struct import pack
import numpy as np

from .fiff.constants import FIFF
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .fiff.tag import find_tag
from .fiff.write import write_int, write_float, write_float_matrix, \
                        write_int_matrix, start_file, end_block, \
                        start_block, end_file
from .utils import logger, verbose
from scipy.spatial.distance import cdist

#
#   These fiff definitions are not needed elsewhere
#
FIFFB_BEM = 310  # BEM data
FIFFB_BEM_SURF = 311  # One of the surfaces
FIFF_BEM_SURF_ID = 3101  # int    surface number
FIFF_BEM_SURF_NAME = 3102  # string surface name
FIFF_BEM_SURF_NNODE = 3103  # int    number of nodes on a surface
FIFF_BEM_SURF_NTRI = 3104  # int     number of triangles on a surface
FIFF_BEM_SURF_NODES = 3105  # float  surface nodes (nnode,3)
FIFF_BEM_SURF_TRIANGLES = 3106  # int    surface triangles (ntri,3)
FIFF_BEM_SURF_NORMALS = 3107  # float  surface node normal unit vectors
FIFF_BEM_COORD_FRAME = 3112  # The coordinate frame of the mode
FIFF_BEM_SIGMA = 3113  # Conductivity of a compartment


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
    bem = dir_tree_find(tree, FIFFB_BEM)
    if bem is None:
        fid.close()
        raise ValueError('BEM data not found')

    bem = bem[0]
    #
    #   Locate all surfaces
    #
    bemsurf = dir_tree_find(bem, FIFFB_BEM_SURF)
    if bemsurf is None:
        fid.close()
        raise ValueError('BEM surface data not found')

    logger.info('    %d BEM surfaces found' % len(bemsurf))
    #
    #   Coordinate frame possibly at the top level
    #
    tag = find_tag(fid, bem, FIFF_BEM_COORD_FRAME)
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
    tag = find_tag(fid, this, FIFF_BEM_SURF_ID)

    if tag is None:
        res['id'] = FIFF.FIFFV_BEM_SURF_ID_UNKNOWN
    else:
        res['id'] = int(tag.data)

    if s_id is not None:
        if res['id'] != s_id:
            return None

    tag = find_tag(fid, this, FIFF_BEM_SIGMA)
    if tag is None:
        res['sigma'] = 1.0
    else:
        res['sigma'] = float(tag.data)

    tag = find_tag(fid, this, FIFF_BEM_SURF_NNODE)
    if tag is None:
        fid.close()
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF_BEM_SURF_NTRI)
    if tag is None:
        fid.close()
        raise ValueError('Number of triangles not found')
    else:
        res['ntri'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        tag = find_tag(fid, this, FIFF_BEM_COORD_FRAME)
        if tag is None:
            res['coord_frame'] = def_coord_frame
        else:
            res['coord_frame'] = tag.data
    else:
        res['coord_frame'] = tag.data
    #
    #   Vertices, normals, and triangles
    #
    tag = find_tag(fid, this, FIFF_BEM_SURF_NODES)
    if tag is None:
        fid.close()
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float)  # XXX : double because of mayavi bug
    if res['rr'].shape[0] != res['np']:
        fid.close()
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        res['nn'] = []
    else:
        res['nn'] = tag.data
        if res['nn'].shape[0] != res['np']:
            fid.close()
            raise ValueError('Vertex normal information is incorrect')

    tag = find_tag(fid, this, FIFF_BEM_SURF_TRIANGLES)
    if tag is None:
        fid.close()
        raise ValueError('Triangulation not found')

    res['tris'] = tag.data - 1  # index start at 0 in Python
    if res['tris'].shape[0] != res['ntri']:
        fid.close()
        raise ValueError('Triangulation information is incorrect')

    return res


def _complete_surface_info(this):
    """Complete surface info"""
    # based on mne_source_space_add_geometry_info() in mne_add_geometry_info.c

    #   Main triangulation [mne_add_triangle_data()]
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = np.cross((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    size = np.sqrt(np.sum(this['tri_nn'] ** 2, axis=1))
    this['tri_area'] = size / 2.0
    zidx = np.where(size == 0)[0]
    for idx in zidx:
        logger.info('    Warning: zero size triangle # %s' % idx)
    size[zidx] = 1.0  # prevent ugly divide-by-zero
    this['tri_nn'] /= size[:, None]

    #    Find neighboring triangles and accumulate vertex normals
    this['nn'] = np.zeros((this['np'], 3))
    # as we don't know the number of neighbors, use lists (faster to append)
    this['neighbor_tri'] = [list() for _ in xrange(this['np'])]
    logger.info('    Triangle normals and neighboring triangles...')
    for p in xrange(this['ntri']):
        # vertex normals
        verts = this['tris'][p]
        this['nn'][verts, :] += this['tri_nn'][p, :]

        # Add to the list of neighbors
        this['neighbor_tri'][verts[0]].append(p)
        this['neighbor_tri'][verts[1]].append(p)
        this['neighbor_tri'][verts[2]].append(p)

    # convert the neighbor lists to arrays
    this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]

    #   Normalize the lengths of the vertex normals
    size = np.sqrt(np.sum(this['nn'] ** 2, axis=1))
    size[size == 0] = 1  # prevent ugly divide-by-zero
    this['nn'] /= size[:, None]

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

    return this


def _get_surf_neighbors(this, k):
    verts = np.concatenate([this['tris'][nt]
                            for nt in this['neighbor_tri'][k]])
    verts = np.setdiff1d(verts, [k], assume_unique=False)
    if np.any(verts >= this['np']):
        raise RuntimeError
    nneighbors = len(verts)
    nneigh_max = len(this['neighbor_tri'][k])
    if nneighbors > nneigh_max:
        raise RuntimeError('Too many neighbors for vertex %d' % k)
    elif nneighbors != nneigh_max:
        logger.info('    Incorrect number of distinct neighbors for vertex'
                    ' %d (%d instead of %d) [fixed].' % (k, nneighbors,
                                                         nneigh_max))


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


def read_surface(fname):
    """Load a Freesurfer surface mesh in triangular format

    Parameters
    ----------
    fname : str
        The name of the file containing the surface.

    Returns
    -------
    coords : array, shape=(n_vertices, 3)
        Coordinate points.
    faces : int array, shape=(n_faces, 3)
        Triangulation (each line contains indexes for three points which
        together form a face).
    """
    with open(fname, "rb") as fobj:
        magic = _fread3(fobj)
        if (magic == 16777215) or (magic == 16777213):  # Quad file or new quad
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            coords = np.fromfile(fobj, ">i2", nvert * 3).astype(np.float)
            coords = coords.reshape(-1, 3) / 100.0
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

        elif magic == 16777214:  # Triangle file
            create_stamp = fobj.readline()
            _ = fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)
        else:
            raise ValueError("%s does not appear to be a Freesurfer surface"
                             % fname)
        logger.info('Triangle file: %s nvert = %s ntri = %s'
                    % (create_stamp.strip(), len(coords), len(faces)))

    coords = coords.astype(np.float)  # XXX: due to mayavi bug on mac 32bits
    return coords, faces


def _read_surface_geom(fname):
    """Load the surface and add the geometry information"""
    # based on mne_load_surface_geom() in mne_surface_io.c
    coords, tris = read_surface(fname)  # mne_read_triangle_file()
    nvert = len(coords)
    ntri = len(tris)
    s = dict(rr=coords, tris=tris, use_tris=tris, ntri=ntri,
             np=nvert)
    s = _complete_surface_info(s)
    return s


def _get_ico_surface(grade):
    """Return an icosahedral surface of the desired grade"""
    ico_file_name = os.path.join(os.path.dirname(__file__), 'data',
                                 'icos.fif.gz')
    ico = read_bem_surfaces(ico_file_name, s_id=9000 + grade)
    return ico


def _normalize_vertices(s):
    """Normalize surface vertices"""
    size = np.sqrt(np.sum(s['rr'] * s['rr'], axis=1))
    size[size == 0] = 1.0  # avoid divide-by-zero
    s['rr'] /= size[:, np.newaxis]


def _get_ico_oct_map(subject, hemi, ico, oct, use_reg, subjects_dir):
    """Get mapping to the nodes of an icos-/oct-ahedron"""
    surf_name = hemi + ('.sphere.reg' if use_reg is True else '.sphere')
    surf_name = op.join(subjects_dir, subject, 'surf', surf_name)
    logger.info('Loading geometry from %s...' % surf_name)
    from_surf = _read_surface_geom(surf_name)
    _normalize_vertices(from_surf)
    if oct is not None:
        to_surf = _tessellate_sphere_surf(oct)
    else:
        to_surf = _get_ico_surface(ico)
    _normalize_vertices(to_surf)

    # Make the maps
    if ico is not None:
        logger.info('Mapping %s %s -> ico (%d) ...', hemi, subject, ico)
    else:
        logger.info('Mapping %s %s -> oct (%d) ...', hemi, subject, oct)
    from_to_map = _get_nearest(to_surf['rr'], from_surf['rr'])
    return from_to_map


def _get_nearest(to, fro):
    """For each point on 'to', find closest on 'fro'"""
    # triage based on sklearn having ball_tree presence
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm='ball_tree').fit(fro)
        from_to_map = nbrs.kneighbors(to)[1].ravel()
    except:
        from_to_map = np.array([np.argmin(d) for d in cdist(to, fro)])
    return from_to_map


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
    for level in range(1, mylevel):
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
    ntri = len(tris)
    nodes = np.zeros((3 * ntri, 3))  # over-allocate for safety
    corners = np.zeros((ntri, 3), int)
    nnode = 0
    for k, tri in enumerate(tris):
        coords = np.array([rr[t] for t in tri])
        for j in range(3):
            dists = cdist(coords[j][np.newaxis, :], nodes[:nnode])[0]
            idx = np.where(dists < 1e-4)[0]
            if len(idx) > 0:
                corners[k, j] = idx[0]
            else:
                nodes[nnode] = coords[j]
                corners[k, j] = nnode
                nnode += 1
    nodes = nodes[:nnode].copy()
    return nodes, corners


def _create_surf_spacing(surf, hemi, subject, ico, oct, spacing, subjects_dir):
    """Load a surf and use the subdivided icosahedron to get points"""
    # Based on load_source_space_surf_spacing() in load_source_space.c
    surf = _read_surface_geom(surf)

    if ico is not None or oct is not None:
        ### from mne_ico_downsample.c ###
        if ico is not None:
            logger.info('Doing the icosahedral vertex picking...')
            ico_surf = _get_ico_surface(ico)
        else:
            logger.info('Doing the octahedral vertex picking...')
            ico_surf = _tessellate_sphere_surf(oct)
        mmap = _get_ico_oct_map(subject, hemi, ico, oct, False, subjects_dir)
        nmap = len(mmap)
        surf['inuse'] = np.zeros(surf['np'], int)
        for k in xrange(nmap):
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
                    'surface')
        surf['nuse_tri'] = ico_surf['ntri']
        surf['use_tris'] = ico_surf['tris'].astype(np.int32)
        for k in xrange(surf['nuse_tri']):
            surf['use_tris'][k] = mmap[surf['use_tris'][k]]

    elif spacing is not None:
        ### from mne_make_source_space/decimate.c ###
        # This is based on MRISubsampleDist in FreeSurfer
        logger.info('    Decimating...')
        d = np.empty(surf['np'], int)
        d.fill(10000)

        # construct all neighbor verts (SLOW!)
        neighbor_vert = [_get_surf_neighbors(surf, k)
                         for k in xrange(surf['np'])]
        # A mysterious algorithm follows (quoth Matti)
        for k in xrange(surf['np']):
            neigh = neighbor_vert[k]
            d[k] = np.min(d[neigh] + 1)
            d[k] = 0 if d[k] >= spacing else d[k]
            d[neigh] = np.minimum(d[k] + 1, d[neigh])

        for k in xrange(surf['np'] - 1, -1, -1):
            neigh = neighbor_vert[k]
            for p in xrange(len(neigh)):
                d[k] = np.minimum(d[neigh[p]] + 1, d[k])
                d[neigh[p]] = np.minimum(d[k] + 1, d[neigh[p]])

        if spacing == 2.0:
            for k in xrange(surf['np']):
                if d[k] > 0:
                    neigh = neighbor_vert[k]
                    n = np.sum(d[neigh] == 0)
                    if n <= 2:
                        d[k] = 0
                    d[neigh] = np.minimum(d[neigh], d[k] + 1)

        logger.info("[done]")
        surf['inuse'] = (d == 0).astype(int)

    else:
        surf['inuse'] = np.ones(surf['np'], int)
    surf['nuse'] = np.sum(surf['inuse'])
    surf['vertno'] = np.where(surf['inuse'])[0]

    # set some final params
    inds = np.arange(surf['np'])
    sizes = np.sqrt(np.sum(surf['nn'] ** 2, axis=1))
    surf['nn'][inds] = surf['nn'][inds] / sizes[:, np.newaxis]
    surf['inuse'][sizes <= 0] = False
    surf['nuse'] = np.sum(surf['inuse'])
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

    with open(fname, 'w') as fid:
        fid.write(pack('>3B', 255, 255, 254))
        fid.writelines(('%s\n' % create_stamp, '\n'))
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

    start_block(fid, FIFFB_BEM)
    start_block(fid, FIFFB_BEM_SURF)

    write_int(fid, FIFF_BEM_SURF_ID, surf['id'])
    write_float(fid, FIFF_BEM_SIGMA, surf['sigma'])
    write_int(fid, FIFF_BEM_SURF_NNODE, surf['np'])
    write_int(fid, FIFF_BEM_SURF_NTRI, surf['ntri'])
    write_int(fid, FIFF_BEM_COORD_FRAME, surf['coord_frame'])
    write_float_matrix(fid, FIFF_BEM_SURF_NODES, surf['rr'])

    if 'nn' in surf and surf['nn'] is not None and len(surf['nn']) > 0:
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, surf['nn'])

    # index start at 0 in Python
    write_int_matrix(fid, FIFF_BEM_SURF_TRIANGLES, surf['tris'] + 1)

    end_block(fid, FIFFB_BEM_SURF)
    end_block(fid, FIFFB_BEM)

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
