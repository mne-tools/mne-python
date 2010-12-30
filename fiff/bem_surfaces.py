import numpy as np

from .constants import FIFF
from .open import fiff_open
from .tree import dir_tree_find
from .tag import find_tag
from scipy import linalg

#
#   These fiff definitions are not needed elsewhere
#
FIFFB_BEM               = 310    # BEM data
FIFFB_BEM_SURF          = 311    # One of the surfaces
#
FIFF_BEM_SURF_ID        = 3101   # int    surface number
FIFF_BEM_SURF_NAME      = 3102   # string surface name
FIFF_BEM_SURF_NNODE	    = 3103   # int    # of nodes on a surface
FIFF_BEM_SURF_NTRI	    = 3104   # int    # number of triangles on a surface
FIFF_BEM_SURF_NODES     = 3105   # float  surface nodes (nnode,3)
FIFF_BEM_SURF_TRIANGLES = 3106   # int    surface triangles (ntri,3)
FIFF_BEM_SURF_NORMALS   = 3107   # float  surface node normal unit vectors (nnode,3)
FIFF_BEM_COORD_FRAME    = 3112   # The coordinate frame of the mode
FIFF_BEM_SIGMA          = 3113   # Conductivity of a compartment


def read_bem_surfaces(fname, add_geom=False):
    """
    #
    # [surf] = mne_read_bem_surfaces(fname, add_geom)
    #
    # Reads source spaces from a fif file
    #
    # fname       - The name of the file or an open file id
    # add_geom    - Add geometry information to the surfaces
    #
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
        raise ValueError, 'BEM data not found'

    bem = bem[0]
    #
    #   Locate all surfaces
    #
    bemsurf = dir_tree_find(bem, FIFFB_BEM_SURF)
    if bemsurf is None:
        fid.close()
        raise ValueError, 'BEM surface data not found'

    print '\t%d BEM surfaces found' % len(bemsurf)
    #
    #   Coordinate frame possibly at the top level
    #
    tag = find_tag(fid, bem, FIFF_BEM_COORD_FRAME)
    if tag is not None:
        coord_frame = tag.data
    #
    #   Read all surfaces
    #
    surf = []
    for bsurf in bemsurf:
        print '\tReading a surface...'
        this = read_bem_surface(fid, bsurf, coord_frame)
        print '[done]'
        if add_geom:
            complete_surface_info(this)
        surf.append(this)

    print '\t%d BEM surfaces read' % len(surf)

    fid.close()

    return surf


def read_bem_surface(fid, this, def_coord_frame):
    """
    """
    res = dict()
    #
    #   Read all the interesting stuff
    #
    tag = find_tag(fid, this, FIFF_BEM_SURF_ID)
    if tag is None:
        res['id'] = FIFF.FIFFV_BEM_SURF_ID_UNKNOWN
    else:
        res['id'] = tag.data

    tag = find_tag(fid, this, FIFF_BEM_SIGMA)
    if tag is None:
        res['sigma'] = 1.0
    else:
        res['sigma'] = tag.data

    tag = find_tag(fid, this, FIFF_BEM_SURF_NNODE)
    if tag is None:
        fid.close()
        raise ValueError, 'Number of vertices not found'

    res = dict()
    res['np'] = tag.data

    tag = find_tag(fid, this,FIFF_BEM_SURF_NTRI)
    if tag is None:
        fid.close()
        raise ValueError, 'Number of triangles not found'
    else:
        res['ntri'] = tag.data

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
        raise ValueError, 'Vertex data not found'

    res['rr'] = tag.data.astype(np.float) # XXX : double because of mayavi bug
    if res['rr'].shape[0] != res['np']:
        fid.close()
        raise ValueError, 'Vertex information is incorrect'

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        res['nn'] = []
    else:
        res['nn'] = tag.data
        if res['nn'].shape[0] != res['np']:
            fid.close()
            raise ValueError, 'Vertex normal information is incorrect'

    tag = find_tag(fid, this, FIFF_BEM_SURF_TRIANGLES)
    if tag is None:
        fid.close()
        raise ValueError, 'Triangulation not found'

    res['tris'] = tag.data - 1 # index start at 0 in Python
    if res['tris'].shape[0] != res['ntri']:
        fid.close()
        raise ValueError, 'Triangulation information is incorrect'

    return res


def complete_surface_info(this):
    """ XXX : should be factorize with complete_source_space_info
    """
    #
    #   Main triangulation
    #
    print '\tCompleting triangulation info...'
    print 'triangle normals...'
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:,0],:]
    r2 = this['rr'][this['tris'][:,1],:]
    r3 = this['rr'][this['tris'][:,2],:]
    this['tri_cent'] = (r1+r2+r3) /3.0
    this['tri_nn'] = np.cross((r2-r1), (r3-r1))
    #
    #   Triangle normals and areas
    #
    for p in range(this['ntri']):
        size = linalg.norm(this['tri_nn'][p,:])
        this['tri_area'][p] = size / 2.0
    if size > 0.0:
        this['tri_nn'][p,:] = this['tri_nn'][p,:] / size
    #
    #   Accumulate the vertex normals
    #
    print 'vertex normals...'
    this['nn'] = np.zeros((this['np'], 3))
    for p in range(this['ntri']):
        this['nn'][this['tris'][p,:],:] = this['nn'][this['tris'][p,:],:] \
                              + np.kron(np.ones((3, 1)), this['tri_nn'][p,:])
    #
    #   Compute the lengths of the vertex normals and scale
    #
    print 'normalize...'
    for p in range(this['np']):
        size = linalg.norm(this['nn'][p,:])
        if size > 0:
            this['nn'][p,:] = this['nn'][p,:] / size

    print '[done]\n'
    return this
