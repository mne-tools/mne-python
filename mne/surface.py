# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from struct import pack

import logging
logger = logging.getLogger('mne')

from .fiff.constants import FIFF
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .fiff.tag import find_tag
from .fiff.write import write_int, write_float, write_float_matrix, \
                        write_int_matrix, start_file, end_block, \
                        start_block, end_file
from . import verbose

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
