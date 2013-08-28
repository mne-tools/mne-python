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


@verbose
def _complete_surface_info(this, verbose=None):
    """Complete surface info"""
    #
    #   Main triangulation
    #
    logger.info('    Completing triangulation info...')
    logger.info('triangle normals...')
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = np.cross((r2 - r1), (r3 - r1))
    #
    #   Triangle normals and areas
    #
    size = np.sqrt(np.sum(this['tri_nn'] ** 2, axis=1))
    this['tri_area'] = size / 2.0
    this['tri_nn'] /= size[:, None]
    #
    #   Accumulate the vertex normals
    #
    logger.info('vertex normals...')
    this['nn'] = np.zeros((this['np'], 3))
    for p in range(this['ntri']):
        this['nn'][this['tris'][p, :], :] += this['tri_nn'][p, :]
    #
    #   Compute the lengths of the vertex normals and scale
    #
    logger.info('normalize...')
    this['nn'] /= np.sqrt(np.sum(this['nn'] ** 2, axis=1))[:, None]

    logger.info('[done]')
    # XXX TODO: Add neighbor checking for source space generation
    return this


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

    coords = coords.astype(np.float)  # XXX: due to mayavi bug on mac 32bits
    return coords, faces


def _read_surface_geom(fname, check_neighbors=True):
    """Load the surface and add the geometry information"""
    coords, tris = read_surface(fname)
    s = dict(rr=coords, tris=tris, itris=tris, ntri=len(tris), np=len(coords))
    # XXX TODO: add check_neighbors support
    s = _complete_surface_info(s)
    s['nuse'] = s['np']
    s['inuse'] = np.ones(s['np'], int)
    s['vertno'] = np.arange(s['np'], dtype=int)
    return s


def _get_ico_surface(grade):
    """Return an icosahedral surface of the desired grade"""
    ico_file_name = os.path.join(os.path.dirname(__file__), 'data',
                                 'icos.fif.gz')
    ico = read_bem_surfaces(ico_file_name, s_id=9000 + grade)
    return ico


def _normalize_vertices(s):
    """Normalize surface vertices"""
    s['rr'] /= np.sqrt(np.sum(s['rr'] * s['rr'], axis=1))[:, np.newaxis]


def _get_ico_map(subject, hemi, ico, oct, use_reg, subjects_dir):
    """Get mapping to the nodes of an icosahedron"""
    surf_name = hemi + ('.sphere.reg' if use_reg is True else '.sphere')
    surf_name = op.join(subjects_dir, subject, 'surf', surf_name)
    logger.info('Loading geometry from %s...' % surf_name)
    from_surf = _read_surface_geom(surf_name, True)
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
    from_to_map = _get_nearest(to_surf, from_surf)
    logger.info('[done]')
    return from_to_map


def _get_nearest(to, fro):
    """For each point on (spherical) 'to', find closest on 'fro'"""
    from_to_map = np.zeros(to['np'], int)
    # Get a set of points for the hierarchical search
    nodes = _tessellate_sphere(5)
    nnode = len(nodes)

    max_cos = np.max(np.sqrt(np.sum(nodes[0] * nodes, axis=1)))
    search_cos = np.cos(1.2 * np.acos(max_cos))
    search = [dict(r=nodes[k].copy(), verts=None, nvert=0)
              for k in range(nnode)]
    temp = np.zeros(fro['np'], int)
    for k in range(nnode):
        this_node = search[k]
        ntemp = 0
        for p in range(fro['np']):
            this_cos = np.dot(fro['rr'][p], this_node['r'])
            if this_cos > search_cos:
                temp[ntemp] = p
                ntemp += 1
        if ntemp > 0:
            this_node['nvert'] = ntemp
            this_node['verts'] = temp[:ntemp].copy()

    # Do a hierarchical search
    for k in range(to['np']):
        r = to['rr'][k]
        # Perform stage1 search first
        vals = [np.dot(r, s['r']) for s in search]
        max_stage1 = np.argmax(vals)
        max_cos = vals[max_stage1]

        # Then look at the viable nodes
        from_to_map[k] = 0
        ntemp = search[max_stage1]['nvert']
        verts = search[max_stage1]['verts']
        vals = [np.dot(r, fro['rr'][verts[p]]) for p in range(ntemp)]
        from_to_map[k] = np.argmax(vals)

    return from_to_map


def _tessellate_sphere_surf(level, rad=1.0):
    """Return a surface structure instead of the details"""
    rr, npt, itris, ntri = _tessellate_sphere(level)
    nn = rr.copy()
    rr *= rad
    s = dict(rr=rr, np=npt, tris=itris, itris=itris, ntri=ntri, nuse=npt,
             nn=nn, inuse=np.ones(npt, int))
    s = _complete_surface_info(s)
    return s


"""
typedef struct {
  point     pt[3];	/* Vertices of triangle */
} triangle;

typedef struct {
  int       npoly;	/* # of triangles in object */
  triangle *poly;	/* Triangles */
} object;

  static triangle octahedron[] = {
    { { XPLUS, ZPLUS, YPLUS }},
    { { YPLUS, ZPLUS, XMIN  }},
    { { XMIN , ZPLUS, YMIN  }},
    { { YMIN , ZPLUS, XPLUS }},
    { { XPLUS, YPLUS, ZMIN  }},
    { { YPLUS, XMIN , ZMIN  }},
    { { XMIN , YMIN , ZMIN  }},
    { { YMIN , XPLUS, ZMIN  }}};
  /*
   * A unit octahedron
   */
  static object oct = {
    sizeof(octahedron) / sizeof(octahedron[0]),
    &octahedron[0]
  };
"""


def _norm_midpt(a, b):
    assert a.shape[1] == 3
    assert b.shape[1] == 3
    c = (a + b) / 2.
    return c / np.sqrt(np.sum(c ** 2, 1))[:, np.newaxis]


def _tessellate_sphere(mylevel):
    """Create a tessellation of a unit sphere"""
    XPLUS = [1, 0, 0]
    XMIN = [-1, 0, 0]
    YPLUS = [0, 1, 0]
    YMIN = [0, -1, 0]
    ZPLUS = [0, 0, 1]
    ZMIN = [0, 0, -1]
    # Vertices of a unit octahedron
    octahedron = np.array([[XPLUS, ZPLUS, YPLUS],
                           [YPLUS, ZPLUS, XMIN],
                           [XMIN, ZPLUS, YMIN],
                           [YMIN, ZPLUS, XPLUS],
                           [XPLUS, YPLUS, ZMIN],
                           [YPLUS, XMIN, ZMIN],
                           [XMIN, YMIN, ZMIN],
                           [YMIN, XPLUS, ZMIN]])

    # A unit octahedron
    if mylevel < 1:
        raise ValueError('# of levels must be >= 1')

    # Reverse order of points in each triangle
    # for counter-clockwise ordering
    octahedron = octahedron[:, [2, 1, 0], :]
    old_object = octahedron

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
        new_object = np.zeros((len(old_object), 4, 3, 3))
        a = _norm_midpt(old_object[:, 0], old_object[:, 2])
        b = _norm_midpt(old_object[:, 0], old_object[:, 1])
        c = _norm_midpt(old_object[:, 1], old_object[:, 1])
        new_object[:, 0] = np.array([old_object[:, 0], b, a]).swapaxes(0, 1)
        new_object[:, 1] = np.array([b, old_object[:, 1], c]).swapaxes(0, 1)
        new_object[:, 2] = np.array([a, b, c]).swapaxes(0, 1)
        new_object[:, 3] = np.array([a, c, old_object[:, 2]]).swapaxes(0, 1)
        new_object = np.reshape(new_object, (len(old_object) * 4, 3, 3))
        old_object = np.ascontiguousarray(new_object)

    # Copy the resulting approximation into standard table
    ntri = len(old_object)
    nodes = np.zeros((3 * ntri, 3))
    corners = np.zeros((ntri, 3), int)
    nnode = 0
    for k in range(ntri):
        tri = old_object[k]
        for j in range(3):
            dists = np.sqrt(np.sum((tri[j] - nodes[:nnode]) ** 2, 1))
            idx = np.where(dists < 1e-4)[0]
            if len(idx) > 0:
                corners[k, j] = idx[0]
            else:
                nodes[nnode] = tri[j]
                corners[k, j] = nnode
                nnode += 1

    return nodes, nnode, corners, ntri


def _create_surf_spacing(surf, hemi, subject, ico, oct, spacing, subjects_dir):
    if hemi == 'lh':
        s_id = FIFF.FIFFV_MNE_SURF_LEFT_HEMI
    else:
        s_id = FIFF.FIFFV_MNE_SURF_LEFT_HEMI
    surf = read_surface(surf)
    surf = dict(verts=surf[0], tris=surf[1], id=s_id)

    if ico is not None or oct is not None:
        if ico is not None:
            logger.info('Doing the octahedral vertex picking...')
            ico_surf = _get_ico_surface(ico)
        else:
            logger.info('Doing the icosahedral vertex picking...')
            ico_surf = _tessellate_sphere_surf(oct)
        mmap = _get_ico_map(subject, hemi, ico, oct, False, subjects_dir)
        nmap = len(mmap)
        surf['inuse'].fill(False)
        for k in range(nmap):
            if surf['inuse'][mmap[k]]:
                # Try the nearest neighbors
                neigh = surf['neighbor_vert'][map[k]]
                nneigh = surf['nneighbor_vert'][map[k]]
                was = mmap[k]
                inds = np.where(np.logical_not(surf['inuse'][neigh]))[0]
                if len(inds) == 0:
                    raise RuntimeError('Could not find neighbor')
                else:
                    mmap[k] = neigh[inds[-1]]
                logger.info('Source space vertex moved from %d to %d '
                            'because of double occupation', was, mmap[k])
            elif mmap[k] < 0 or map[k] > surf['np']:
                raise RuntimeError('Map number out of range (%d), this is '
                                   'probably due to inconsistent surfaces. '
                                   'Parts of the FreeSurfer reconstruction '
                                   'need to be redone.' % mmap[k])
            surf['inuse'][mmap[k]] = True
        surf['nuse'] = nmap

        logger.info('Setting up the triangulation for the decimated surface')
        surf['nuse_tri'] = ico_surf['ntri']
        surf['use_itris'] = ico_surf['itris']
        ico_surf['itris'] = None
        for k in range(surf['nuse_tri']):
            surf['use_itris'][k] = mmap[surf['use_itris'][k]]

    elif spacing is not None:
        # This is based on MRISubsampleDist in FreeSurfer
        logger.info('Decimating...')
        d = np.empty(surf['np'], int)
        d.fill(10000)

        # A mysterious algorithm follows
        for k in range(surf['np']):
            neigh = surf['neighbor_vert'][k]
            d[k] = np.min(d[neigh] + 1)
            d[k] = 0 if d[k] >= spacing else d[k]
            d[neigh] = np.minimum(d[k] + 1, d[neigh])

        for k in range(surf['np'] - 1, -1, -1):
            nneigh = surf['nneighbor_vert'][k]
            neigh = surf['neighbor_vert'][k]
            for p in range(nneigh):
                d[k] = np.minimum(d[neigh[p]] + 1, d[k])
                d[neigh[p]] = np.minimum(d[k] + 1, d[neigh[p]])

        if spacing == 2.0:
            for k in range(surf['np']):
                if d[k] > 0:
                    nneigh = surf['nneighbor_vert'][k]
                    neigh = surf['neighbor_vert'][k]
                    n = np.sum(d[neigh] == 0)
                    if n <= 2:
                        d[k] = 0
                    d[neigh] = np.minimum(d[neigh], d[k] + 1)

        logger.info("[done]")
        surf['inuse'] = (d == 0)
        surf['nuse'] = np.sum(surf['inuse'])

    else:
        surf['inuse'].fill(True)
        surf['nuse'] = surf['np']

    # set some final params
    inds = np.arange(surf['np'])
    sizes = np.sqrt(np.sum(surf['nn'] ** 2, axis=1))
    surf['nn'][inds] = surf['nn'][inds] / sizes
    surf['inuse'][sizes <= 0] = False
    surf['nuse'] = np.sum(surf['inuse'])
    surf['subject'] = subject


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
