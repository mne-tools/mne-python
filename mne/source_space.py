# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from scipy import sparse

import logging
logger = logging.getLogger('mne')

from .label import _aslabel
from .fiff.constants import FIFF
from .fiff.tree import dir_tree_find
from .fiff.tag import find_tag, read_tag
from .fiff.open import fiff_open
from .fiff.write import start_block, end_block, write_int, \
                        write_float_sparse_rcs, write_string, \
                        write_float_matrix, write_int_matrix, \
                        write_coord_trans, start_file, end_file, write_id
from .surface import read_surface
from .utils import get_subjects_dir
from . import verbose


class SourceSpaces(list):
    """Represent a list of source space

    Currently implemented as a list of dictionaries containing the source
    space information

    Parameters
    ----------
    source_spaces : list
        A list of dictionaries containing the source space information.
    info : dict
        Dictionary with information about the creation of the source space
        file. Has keys 'working_dir' and 'command_line'.

    Attributes
    ----------
    info : dict
        Dictionary with information about the creation of the source space
        file. Has keys 'working_dir' and 'command_line'.
    """
    def __init__(self, source_spaces, info=None):
        super(SourceSpaces, self).__init__(source_spaces)
        if info is None:
            self.info = dict()
        else:
            self.info = dict(info)

    def __repr__(self):
        ss_repr = []
        for ss in self:
            ss_type = ss['type']
            if ss_type == 'vol':
                r = "'vol', shape=%s, n_used=%i" % (repr(ss['shape']), ss['nuse'])
            elif ss_type == 'surf':
                r = "'surf', n_vertices=%i, n_used=%i" % (ss['np'], ss['nuse'])
            else:
                r = "%r" % ss_type
            ss_repr.append('<%s>' % r)
        ss_repr = ', '.join(ss_repr)
        return "<SourceSpaces: [{ss}]>".format(ss=ss_repr)

    def save(self, fname):
        """Save the source spaces to a fif file

        Parameters
        ----------
        fname : str
            File to write.
        """
        write_source_spaces(fname, self)


def patch_info(nearest):
    """Patch information in a source space

    Generate the patch information from the 'nearest' vector in
    a source space. For vertex in the source space it provides
    the list of neighboring vertices in the high resolution
    triangulation.

    Parameters
    ----------
    nearest : array
        For each vertex gives the index of its closest neighbor.

    Returns
    -------
    pinfo : list
        List of neighboring vertices
    """
    if nearest is None:
        pinfo = None
        return pinfo

    indn = np.argsort(nearest)
    nearest_sorted = nearest[indn]

    steps = np.where(nearest_sorted[1:] != nearest_sorted[:-1])[0] + 1
    starti = np.r_[[0], steps]
    stopi = np.r_[steps, [len(nearest)]]

    pinfo = list()
    for start, stop in zip(starti, stopi):
        pinfo.append(np.sort(indn[start:stop]))

    return pinfo


@verbose
def read_source_spaces_from_tree(fid, tree, add_geom=False, verbose=None):
    """Read the source spaces from a FIF file

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    tree : dict
        The FIF tree structure if source is a file id.
    add_geom : bool, optional (default False)
        Add geometry information to the surfaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : SourceSpaces
        The source spaces.
    """
    #   Find all source spaces
    spaces = dir_tree_find(tree, FIFF.FIFFB_MNE_SOURCE_SPACE)
    if len(spaces) == 0:
        raise ValueError('No source spaces found')

    src = list()
    for s in spaces:
        logger.info('    Reading a source space...')
        this = _read_one_source_space(fid, s)
        logger.info('    [done]')
        if add_geom:
            complete_source_space_info(this)

        src.append(this)

    src = SourceSpaces(src)
    logger.info('    %d source spaces read' % len(spaces))

    return src


@verbose
def read_source_spaces(fname, add_geom=False, verbose=None):
    """Read the source spaces from a FIF file

    Parameters
    ----------
    fname : str
        The name of the file.
    add_geom : bool, optional (default False)
        Add geometry information to the surfaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : SourceSpaces
        The source spaces.
    """
    fid, tree, _ = fiff_open(fname)
    src = read_source_spaces_from_tree(fid, tree, add_geom=add_geom,
                                       verbose=verbose)
    src.info['fname'] = fname

    node = dir_tree_find(tree, FIFF.FIFFB_MNE_ENV)
    if node:
        node = node[0]
        for p in range(node['nent']):
            kind = node['directory'][p].kind
            pos = node['directory'][p].pos
            tag = read_tag(fid, pos)
            if kind == FIFF.FIFF_MNE_ENV_WORKING_DIR:
                src.info['working_dir'] = tag.data
            elif kind == FIFF.FIFF_MNE_ENV_COMMAND_LINE:
                src.info['command_line'] = tag.data

    return src


@verbose
def _read_one_source_space(fid, this, verbose=None):
    """Read one source space
    """
    FIFF_BEM_SURF_NTRI = 3104
    FIFF_BEM_SURF_TRIANGLES = 3106

    res = dict()

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_ID)
    if tag is None:
        res['id'] = int(FIFF.FIFFV_MNE_SURF_UNKNOWN)
    else:
        res['id'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE)
    if tag is None:
        raise ValueError('Unknown source space type')
    else:
        src_type = int(tag.data)
        if src_type == 1:
            res['type'] = 'surf'
        elif src_type == 2:
            res['type'] = 'vol'
        else:
            raise ValueError('Unknown source space type (%d)' % src_type)

    if res['type'] == 'vol':

        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS)
        if tag is not None:
            res['shape'] = tuple(tag.data)

        tag = find_tag(fid, this, FIFF.FIFF_COORD_TRANS)
        if tag is not None:
            res['src_mri_t'] = tag.data

        parent_mri = dir_tree_find(this, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        if len(parent_mri) == 0:
            raise ValueError('Can not find parent MRI location')

        mri = parent_mri[0]
        for d in mri['directory']:
            if d.kind == FIFF.FIFF_COORD_TRANS:
                tag = read_tag(fid, d.pos)
                trans = tag.data
                if trans['from'] == FIFF.FIFFV_MNE_COORD_MRI_VOXEL:
                    res['vox_mri_t'] = tag.data
                if trans['to'] == FIFF.FIFFV_MNE_COORD_RAS:
                    res['mri_ras_t'] = tag.data

        tag = find_tag(fid, mri, FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR)
        if tag is not None:
            res['interpolator'] = tag.data
        else:
            logger.info("Interpolation matrix for MRI not found.")

        tag = find_tag(fid, mri, FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE)
        if tag is not None:
            res['mri_file'] = tag.data

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_WIDTH)
        if tag is not None:
            res['mri_width'] = int(tag.data)

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_HEIGHT)
        if tag is not None:
            res['mri_height'] = int(tag.data)

        tag = find_tag(fid, mri, FIFF.FIFF_MRI_DEPTH)
        if tag is not None:
            res['mri_depth'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF_BEM_SURF_NTRI)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI)
        if tag is None:
            res['ntri'] = 0
        else:
            res['ntri'] = int(tag.data)
    else:
        res['ntri'] = tag.data

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        raise ValueError('Coordinate frame information not found')

    res['coord_frame'] = tag.data

    #   Vertices, normals, and triangles
    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS)
    if tag is None:
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float)  # double precision for mayavi
    if res['rr'].shape[0] != res['np']:
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        raise ValueError('Vertex normals not found')

    res['nn'] = tag.data
    if res['nn'].shape[0] != res['np']:
        raise ValueError('Vertex normal information is incorrect')

    if res['ntri'] > 0:
        tag = find_tag(fid, this, FIFF_BEM_SURF_TRIANGLES)
        if tag is None:
            tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES)
            if tag is None:
                raise ValueError('Triangulation not found')
            else:
                res['tris'] = tag.data - 1  # index start at 0 in Python
        else:
            res['tris'] = tag.data - 1  # index start at 0 in Python

        if res['tris'].shape[0] != res['ntri']:
            raise ValueError('Triangulation information is incorrect')
    else:
        res['tris'] = None

    #   Which vertices are active
    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE)
    if tag is None:
        res['nuse'] = 0
        res['inuse'] = np.zeros(res['nuse'], dtype=np.int)
        res['vertno'] = None
    else:
        res['nuse'] = int(tag.data)
        tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION)
        if tag is None:
            raise ValueError('Source selection information missing')

        res['inuse'] = tag.data.astype(np.int).T
        if len(res['inuse']) != res['np']:
            raise ValueError('Incorrect number of entries in source space '
                             'selection')

        res['vertno'] = np.where(res['inuse'])[0]

    #   Use triangulation
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES)
    if tag1 is None or tag2 is None:
        res['nuse_tri'] = 0
        res['use_tris'] = None
    else:
        res['nuse_tri'] = tag1.data
        res['use_tris'] = tag2.data - 1  # index start at 0 in Python

    #   Patch-related information
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST)

    if tag1 is None or tag2 is None:
        res['nearest'] = None
        res['nearest_dist'] = None
    else:
        res['nearest'] = tag1.data
        res['nearest_dist'] = tag2.data.T

    res['pinfo'] = patch_info(res['nearest'])
    if (res['pinfo'] is not None):
        logger.info('Patch information added...')

    #   Distances
    tag1 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_DIST)
    tag2 = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT)
    if tag1 is None or tag2 is None:
        res['dist'] = None
        res['dist_limit'] = None
    else:
        res['dist'] = tag1.data
        res['dist_limit'] = tag2.data
        #   Add the upper triangle
        res['dist'] = res['dist'] + res['dist'].T
    if (res['dist'] is not None):
        logger.info('Distance information added...')

    tag = find_tag(fid, this, FIFF.FIFF_SUBJ_HIS_ID)
    if tag is not None:
        res['subject_his_id'] = tag.data

    return res


@verbose
def complete_source_space_info(this, verbose=None):
    """Add more info on surface
    """
    #   Main triangulation
    logger.info('    Completing triangulation info...')
    this['tri_area'] = np.zeros(this['ntri'])
    r1 = this['rr'][this['tris'][:, 0], :]
    r2 = this['rr'][this['tris'][:, 1], :]
    r3 = this['rr'][this['tris'][:, 2], :]
    this['tri_cent'] = (r1 + r2 + r3) / 3.0
    this['tri_nn'] = np.cross((r2 - r1), (r3 - r1))
    size = np.sqrt(np.sum(this['tri_nn'] ** 2, axis=1))
    this['tri_area'] = size / 2.0
    this['tri_nn'] /= size[:, None]
    logger.info('[done]')

    #   Selected triangles
    logger.info('    Completing selection triangulation info...')
    if this['nuse_tri'] > 0:
        r1 = this['rr'][this['use_tris'][:, 0], :]
        r2 = this['rr'][this['use_tris'][:, 1], :]
        r3 = this['rr'][this['use_tris'][:, 2], :]
        this['use_tri_cent'] = (r1 + r2 + r3) / 3.0
        this['use_tri_nn'] = np.cross((r2 - r1), (r3 - r1))
        this['use_tri_area'] = np.sqrt(np.sum(this['use_tri_nn'] ** 2, axis=1)
                                       ) / 2.0
    logger.info('[done]')


def find_source_space_hemi(src):
    """Return the hemisphere id for a source space

    Parameters
    ----------
    src : dict
        The source space to investigate

    Returns
    -------
    hemi : int
        Deduced hemisphere id
    """
    xave = src['rr'][:, 0].sum()

    if xave < 0:
        hemi = int(FIFF.FIFFV_MNE_SURF_LEFT_HEMI)
    else:
        hemi = int(FIFF.FIFFV_MNE_SURF_RIGHT_HEMI)

    return hemi


def label_src_vertno_sel(label, src):
    """ Find vertex numbers and indices from label

    Parameters
    ----------
    label : Label
        Source space label
    src : dict
        Source space

    Returns
    -------
    vertno : list of length 2
        Vertex numbers for lh and rh
    src_sel : array of int (len(idx) = len(vertno[0]) + len(vertno[1]))
        Indices of the selected vertices in sourse space
    """
    label = _aslabel(label)

    if src[0]['type'] != 'surf':
        return Exception('Label are only supported with surface source spaces')

    vertno = [src[0]['vertno'], src[1]['vertno']]

    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(vertno[0], label.vertices)
        src_sel = np.searchsorted(vertno[0], vertno_sel)
        vertno[0] = vertno_sel
        vertno[1] = np.array([])
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(vertno[1], label.vertices)
        src_sel = np.searchsorted(vertno[1], vertno_sel) + len(vertno[0])
        vertno[0] = np.array([])
        vertno[1] = vertno_sel
    elif label.hemi == 'both':
        vertno_sel_lh = np.intersect1d(vertno[0], label.lh.vertices)
        src_sel_lh = np.searchsorted(vertno[0], vertno_sel_lh)
        vertno_sel_rh = np.intersect1d(vertno[1], label.rh.vertices)
        src_sel_rh = np.searchsorted(vertno[1], vertno_sel_rh) + len(vertno[0])
        src_sel = np.hstack((src_sel_lh, src_sel_rh))
        vertno = [vertno_sel_lh, vertno_sel_rh]
    else:
        raise Exception("Unknown hemisphere type")

    return vertno, src_sel


def _get_vertno(src):
    return [s['vertno'] for s in src]


###############################################################################
# Write routines

@verbose
def write_source_spaces_to_fid(fid, src, verbose=None):
    """Write the source spaces to a FIF file

    Parameters
    ----------
    fid : file descriptor
        An open file descriptor.
    src : list
        The list of source spaces.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    for s in src:
        logger.info('    Write a source space...')
        start_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        _write_one_source_space(fid, s, verbose)
        end_block(fid, FIFF.FIFFB_MNE_SOURCE_SPACE)
        logger.info('    [done]')
    logger.info('    %d source spaces written' % len(src))


@verbose
def write_source_spaces(fname, src, verbose=None):
    """Write source spaces to a file

    Parameters
    ----------
    fname : str
        File to write.
    src : SourceSpaces
        The source spaces (as returned by read_source_spaces).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MNE)

    if src.info:
        start_block(fid, FIFF.FIFFB_MNE_ENV)

        write_id(fid, FIFF.FIFF_BLOCK_ID)

        data = src.info.get('working_dir', None)
        if data:
            write_string(fid, FIFF.FIFF_MNE_ENV_WORKING_DIR, data)
        data = src.info.get('command_line', None)
        if data:
            write_string(fid, FIFF.FIFF_MNE_ENV_COMMAND_LINE, data)

        end_block(fid, FIFF.FIFFB_MNE_ENV)

    write_source_spaces_to_fid(fid, src, verbose)

    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)


def _write_one_source_space(fid, this, verbose=None):
    """Write one source space"""
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_ID, this['id'])
    if this['type'] == 'surf':
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, 1)
    elif this['type'] == 'vol':
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, 2)
    else:
        raise ValueError('Unknown source space type (%d)' % this['type'])

    data = this.get('subject_his_id', None)
    if data:
        write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, data)

    if this['type'] == 'vol':

        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS, this['shape'])
        write_coord_trans(fid, this['src_mri_t'])

        start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        write_coord_trans(fid, this['vox_mri_t'])

        write_coord_trans(fid, this['mri_ras_t'])

        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR,
                            this['interpolator'])

        if 'mri_file' in this and this['mri_file'] is not None:
            write_string(fid, FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE,
                         this['mri_file'])

        write_int(fid, FIFF.FIFF_MRI_WIDTH, this['mri_width'])
        write_int(fid, FIFF.FIFF_MRI_HEIGHT, this['mri_height'])
        write_int(fid, FIFF.FIFF_MRI_DEPTH, this['mri_depth'])

        end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, this['np'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI, this['ntri'])
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, this['coord_frame'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS, this['rr'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, this['nn'])

    if this['ntri'] > 0:
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES,
                         this['tris'] + 1)

    #   Which vertices are active
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE, this['nuse'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION, this['inuse'])

    if this['type'] != 'vol' and this['use_tris'] is not None:
        #   Use triangulation
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI, this['nuse_tri'])
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES,
                         this['use_tris'] + 1)

    #   Patch-related information
    if this['nearest'] is not None:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST, this['nearest'])
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST,
                  this['nearest_dist'])

    #   Distances
    if this['dist'] is not None:
        # Save only lower triangle
        dists = this['dist'].copy()
        dists = sparse.triu(dists, format=dists.format)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST, dists)
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT,
                           this['dist_limit'])


@verbose
def vertex_to_mni(vertices, hemis, subject, subjects_dir=None, verbose=None):
    """Convert the array of vertices for a hemisphere to MNI coordinates

    Parameters
    ----------
    vertices : int, or list of int
        Vertex number(s) to convert
    hemis : int, or list of int
        Hemisphere(s) the vertices belong to
    subject : string
        Name of the subject to load surfaces from.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    coordinates : n_vertices x 3 array of float
        The MNI coordinates (in mm) of the vertices
    """

    if not isinstance(vertices, list) and not isinstance(vertices, np.ndarray):
        vertices = [vertices]

    if not isinstance(hemis, list) and not isinstance(hemis, np.ndarray):
        hemis = [hemis] * len(vertices)

    if not len(hemis) == len(vertices):
        raise ValueError('hemi and vertices must match in length')

    subjects_dir = get_subjects_dir(subjects_dir)

    surfs = [op.join(subjects_dir, subject, 'surf', '%s.white' % h)
             for h in ['lh', 'rh']]
    rr = [read_surface(s)[0] for s in surfs]

    # take point locations in RAS space and convert to MNI coordinates
    xfm = _freesurfer_read_talxfm(op.join(subjects_dir, subject, 'mri',
                                          'transforms', 'talairach.xfm'))
    data = np.array([np.concatenate((rr[h][v, :], [1]))
                     for h, v in zip(hemis, vertices)]).T
    return np.dot(xfm, data).T


@verbose
def _freesurfer_read_talxfm(fname, verbose=None):
    """Read MNI transform from FreeSurfer talairach.xfm file

    Adapted from freesurfer m-files.
    """

    fid = open(fname, 'r')

    logger.debug('Reading FreeSurfer talairach.xfm file:\n%s' % fname)

    # read lines until we get the string 'Linear_Transform', which precedes
    # the data transformation matrix
    got_it = False
    comp = 'Linear_Transform'
    for line in fid:
        if line[:len(comp)] == comp:
            # we have the right line, so don't read any more
            got_it = True
            break

    if got_it:
        xfm = list()
        # read the transformation matrix (3x4)
        for ii, line in enumerate(fid):
            digs = [float(s) for s in line.strip('\n;').split()]
            xfm.append(digs)
            if ii == 2:
                break
        # xfm.append([0., 0., 0., 1.])  # Don't bother appending this
        xfm = np.array(xfm)
        fid.close()
    else:
        fid.close()
        raise ValueError('failed to find \'Linear_Transform\' string in xfm '
                         'file:\n%s' % fname)

    return xfm
