# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import os
import os.path as op
from scipy import sparse, linalg

from .fiff.constants import FIFF
from .fiff.tree import dir_tree_find
from .fiff.tag import find_tag, read_tag
from .fiff.open import fiff_open
from .fiff.write import start_block, end_block, write_int, \
                        write_float_sparse_rcs, write_string, \
                        write_float_matrix, write_int_matrix, \
                        write_coord_trans, start_file, end_file, write_id
from .surface import read_surface, _create_surf_spacing, _get_vertex_map
from .utils import get_subjects_dir, run_subprocess, has_freesurfer, \
                   has_nibabel, logger, verbose


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
                r = ("'vol', shape=%s, n_used=%i"
                     % (repr(ss['shape']), ss['nuse']))
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


def _add_patch_info(s):
    """Patch information in a source space

    Generate the patch information from the 'nearest' vector in
    a source space. For vertex in the source space it provides
    the list of neighboring vertices in the high resolution
    triangulation.

    Parameters
    ----------
    s : dict
        The source space.
    """
    nearest = s['nearest']
    if nearest is None:
        s['pinfo'] = None
        s['patch_inds'] = None
        return

    logger.info('    Computing patch statistics...')

    indn = np.argsort(nearest)
    nearest_sorted = nearest[indn]

    steps = np.where(nearest_sorted[1:] != nearest_sorted[:-1])[0] + 1
    starti = np.r_[[0], steps]
    stopi = np.r_[steps, [len(nearest)]]

    pinfo = list()
    for start, stop in zip(starti, stopi):
        pinfo.append(np.sort(indn[start:stop]))
    s['pinfo'] = pinfo

    # compute patch indices of the in-use source space vertices
    patch_verts = nearest_sorted[steps - 1]
    s['patch_inds'] = np.searchsorted(patch_verts, s['vertno'])

    logger.info('    Patch information added...')


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
            _complete_source_space_info(this)

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
            # MNE 2.7.3 (and earlier) didn't store necessary information
            # about volume coordinate translations. Although there is a
            # FFIF_COORD_TRANS in the higher level of the FIFF file, this
            # doesn't contain all the info we need. Safer to return an
            # error unless a user really wants us to add backward compat.
            raise ValueError('Can not find parent MRI location. The volume '
                             'source space may have been made with an MNE '
                             'version that is too old (<= 2.7.3). Consider '
                             'updating and regenerating the inverse.')

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

    _add_patch_info(res)

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
        logger.info('    Distance information added...')

    tag = find_tag(fid, this, FIFF.FIFF_SUBJ_HIS_ID)
    if tag is not None:
        res['subject_his_id'] = tag.data

    return res


@verbose
def _complete_source_space_info(this, verbose=None):
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
    if this['type'] == 'surf':
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, 1)
    elif this['type'] == 'vol':
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, 2)
    else:
        raise ValueError('Unknown source space type (%d)' % this['type'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_ID, this['id'])

    data = this.get('subject_his_id', None)
    if data:
        write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, data)
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, this['coord_frame'])

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
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS, this['rr'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, this['nn'])

    #   Which vertices are active
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION, this['inuse'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE, this['nuse'])

    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI, this['ntri'])
    if this['ntri'] > 0:
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES,
                         this['tris'] + 1)

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
        # Save only upper triangular portion of the matrix
        dists = this['dist'].copy()
        dists = sparse.triu(dists, format=dists.format)
        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST, dists)
        write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT,
                           this['dist_limit'])


@verbose
def vertex_to_mni(vertices, hemis, subject, subjects_dir=None, mode=None,
                  verbose=None):
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
    mode : string | None
        Either 'nibabel' or 'freesurfer' for the software to use to
        obtain the transforms. If None, 'nibabel' is tried first, falling
        back to 'freesurfer' if it fails. Results should be equivalent with
        either option, but nibabel may be quicker (and more pythonic).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    coordinates : n_vertices x 3 array of float
        The MNI coordinates (in mm) of the vertices

    Notes
    -----
    This function requires either nibabel (in Python) or Freesurfer
    (with utility "mri_info") to be correctly installed.
    """
    if not has_freesurfer and not has_nibabel():
        raise RuntimeError('NiBabel (Python) or Freesurfer (Unix) must be '
                           'correctly installed and accessible from Python')

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
    xfm = _read_talxfm(subject, subjects_dir, mode)
    data = np.array([np.concatenate((rr[h][v, :], [1]))
                     for h, v in zip(hemis, vertices)]).T
    return np.dot(xfm, data)[:3, :].T.copy()


@verbose
def _read_talxfm(subject, subjects_dir, mode=None, verbose=None):
    """Read MNI transform from FreeSurfer talairach.xfm file

    Adapted from freesurfer m-files. Altered to deal with Norig
    and Torig correctly.
    """
    if mode is not None and not mode in ['nibabel', 'freesurfer']:
        raise ValueError('mode must be "nibabel" or "freesurfer"')
    fname = op.join(subjects_dir, subject, 'mri', 'transforms',
                    'talairach.xfm')
    with open(fname, 'r') as fid:
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
            xfm.append([0., 0., 0., 1.])
            xfm = np.array(xfm, dtype=float)
        else:
            raise ValueError('failed to find \'Linear_Transform\' string in '
                             'xfm file:\n%s' % fname)

    # now get Norig and Torig
    path = op.join(subjects_dir, subject, 'mri', 'orig.mgz')

    try:
        import nibabel as nib
        use_nibabel = True
    except ImportError:
        use_nibabel = False
        if mode == 'nibabel':
            raise ImportError('Tried to import nibabel but failed, try using '
                              'mode=None or mode=Freesurfer')

    # note that if mode == None, then we default to using nibabel
    if use_nibabel is True and mode == 'freesurfer':
        use_nibabel = False
    if use_nibabel:
        img = nib.load(path)
        hdr = img.get_header()
        n_orig = hdr.get_vox2ras()
        ds = np.array(hdr.get_zooms())
        ns = (np.array(hdr.get_data_shape()[:3]) * ds) / 2.0
        t_orig = np.array([[-ds[0], 0, 0, ns[0]],
                           [0, 0, ds[2], -ns[2]],
                           [0, -ds[1], 0, ns[1]],
                           [0, 0, 0, 1]], dtype=float)
        nt_orig = [n_orig, t_orig]
    else:
        nt_orig = list()
        for conv in ['--vox2ras', '--vox2ras-tkr']:
            stdout, stderr = run_subprocess(['mri_info', conv, path])
            stdout = np.fromstring(stdout, sep=' ').astype(float)
            if not stdout.size == 16:
                raise ValueError('Could not parse Freesurfer mri_info output')
            nt_orig.append(stdout.reshape(4, 4))
    xfm = np.dot(xfm, np.dot(nt_orig[0], linalg.inv(nt_orig[1])))
    return xfm


###############################################################################
# Creation and decimation

@verbose
def setup_source_space(subject, fname=True, spacing=None, ico=None, oct=None,
                       use_all=False, surface='white', overwrite=False,
                       morph=None, subjects_dir=None, verbose=None):
    """Setup a source space with decimation

    Parameters
    ----------
    subject : str
        Subject to process.
    fname : str | bool
        Filename to use. If True, a default name will be used. If False,
        the source space will not be saved (only returned).
    spacing : float | None
        The spacing to use (in mm). Should be None if ``ico``, ``oct``,
        or ``all`` are used. If all are None, a spacing of 7 will be used.
    ico : float | None
        Use a recursively subdivided icosahedron. Should be None if
        ``spacing``, ``oct``, or ``all`` are used.
    oct : float | None
        Use a recursively subdivided octahedron. Should be None if
        ``spacing``, ``ico``, or ``all`` are used.
    use_all : bool
        If True, include all vertices in the source space. Should be
        False if ``spacing``, ``ico``, or ``oct`` are used.
    surface : str
        The surface to use.
    overwrite: bool
        If True, overwrite file (if it exists).
    morph : str | None
        Morph source space to this subject. If None, uses subject
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : list
        The source space for each hemisphere.
    """
    cmd = ('setup_source_space(%s, fname=%s, spacing=%s, ico=%s, oct=%s, '
           'use_all=%s, surface=%s, overwrite=%s, morph=%s, subjects_dir=%s, '
           'verbose=%s)' % (subject, fname, spacing, ico, oct, use_all,
                            surface, overwrite, morph, subjects_dir, verbose))
    # check to make sure our parameters are good
    use_all = None if use_all is False else use_all
    opts = [ico, oct, spacing, use_all]
    n_chosen = sum([x is not None for x in opts])
    if n_chosen == 0:
        spacing = 7
        n_chosen = 1
    if not n_chosen == 1:
        raise ValueError('Exactly one of "ico", "oct", "spacing", and "all"'
                         'must be defined')
    if morph is None:
        morph = subject
        dest_subject = subject
    else:
        dest_subject = morph + '-' + subject
    subjects_dir = get_subjects_dir(subjects_dir)

    subj_dir = op.join(subjects_dir, subject)
    surf_dir = op.join(subj_dir, 'surf')
    bem_dir = op.join(subjects_dir, morph, 'bem')
    lh_surf = op.join(surf_dir, 'lh.' + surface)
    rh_surf = op.join(surf_dir, 'rh.' + surface)

    if not op.isdir(subj_dir):
        raise IOError('Could not find the MRI data directory %s' % subj_dir)

    if not op.isdir(surf_dir):
        raise IOError('Could not find the surface reconstruction directory '
                      '%s' % surf_dir)
    if not op.isdir(bem_dir):
        raise IOError('Could not create the model directory %s' % bem_dir)
    for surf, hemi in zip([lh_surf, rh_surf], ['LH', 'RH']):
        if not op.isfile(lh_surf):
            raise IOError('Could not find the %s surface %s' % (hemi, surf))

    if fname is True:
        if ico is not None:
            extra = 'ico-%s' % ico
        elif oct is not None:
            extra = 'oct-%s' % oct
        elif spacing is not None:
            extra = str(spacing)
        else:
            extra = 'all'
        fname = op.join(bem_dir, '%s-%s-src.fif' % (dest_subject, extra))
    if fname is not False and op.isfile(fname) and overwrite is False:
        raise IOError('file "%s" exists, use overwrite=True if you want '
                      'to overwrite the file' % fname)

    logger.info('Setting up the source space with the following parameters:\n')
    logger.info('SUBJECTS_DIR = %s' % subjects_dir)
    logger.info('Subject      = %s' % subject)
    if morph != subject:
        logger.info('Morph        = %s' % morph)
    logger.info('Surface      = %s' % surf)
    if ico is not None:
        logger.info('Icosahedron subdivision grade %s\n' % ico)
    elif oct is not None:
        logger.info('Octahedron subdivision grade %s\n' % oct)
    elif spacing is not None:
        logger.info('Grid spacing = %s mm\n' % spacing)
    else:
        logger.info('Include all vertices\n')

    # Create the fif file
    if fname is not None:
        if fname is False:
            logger.info('>>> 1. Creating the source space...\n')
        else:
            logger.info('>>> 1. Creating the source space file %s...' % fname)

    # mne_make_source_space ...
    src = []
    if morph != subject:
        morph_maps = read_morph_map(subject, morph)
    else:
        morph_maps = [[], []]
    # actually make the source spaces
    for hemi, surf, mmap in zip(['lh', 'rh'], [lh_surf, rh_surf], morph_maps):
        logger.info('Loading %s...' % surf)
        s = _create_surf_spacing(surf, hemi, subject, ico, oct, spacing,
                                 subjects_dir)
        logger.info('loaded %s %d/%d selected to source space'
                    % (op.split(surf)[1], s['nuse'], s['np']))
        if morph != subject:
            ms = _create_surf_spacing(surf, hemi, morph, None, None, None,
                                      subjects_dir)
            logger.info('loaded %s of %s %d/%d selected to source space',
                        surf, morph, ms['nuse'], ms['np'])
            best = _get_vertex_map(s, ms, mmap)
            ms['nuse'] = s['nuse']
            ms['inuse'].fill(False)
            ms['vertno'] = best[s['vertno']]
            ms['inuse'][ms['vertno']] = True

            # Possibly add the source space triangulation information
            if s['nuse'] == s['np'] and s['use_itris'] is None:
                s['nuse_tri '] = s['ntri']
                s['use_tris'] = s['tris'].copy()

            ms['nuse_tri'] = s['nuse_tri']
            if s['use_tris'] is not None:
                ms['use_tris'] = best[s['use_tris']]
            src.append(ms)
        else:
            src.append(s)

        logger.info('')  # newline after both subject types are run

    # Fill in source space info
    hemi_ids = [FIFF.FIFFV_MNE_SURF_LEFT_HEMI, FIFF.FIFFV_MNE_SURF_RIGHT_HEMI]
    for s, s_id in zip(src, hemi_ids):
        # Add missing fields
        s.update(dict(dist=None, dist_limit=None, nearest=None, type='surf',
                      nearest_dist=None, pinfo=None, patch_inds=None, id=s_id,
                      coord_frame=np.array((FIFF.FIFFV_COORD_MRI,), np.int32)))
        s['rr'] /= 1000.0
        del s['tri_area']
        del s['tri_cent']
        del s['tri_nn']
        del s['neighbor_tri']

    # upconvert to object format from lists
    src = SourceSpaces(src, dict(working_dir=os.getcwd(), command_line=cmd))

    # write out if requested, then return the data
    if fname is not False:
        write_source_spaces(fname, src)
        logger.info('Wrote %s' % fname)
    logger.info('You are now one step closer to computing the gain matrix')
    return src


@verbose
def read_morph_map(subject_from, subject_to, subjects_dir=None,
                   verbose=None):
    """Read morph map generated with mne_make_morph_maps

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

    subjects_dir = get_subjects_dir(subjects_dir)

    # Does the file exist
    name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_from,
                                              subject_to)
    if not os.path.exists(name):
        name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_to,
                                                  subject_from)
        if not os.path.exists(name):
            raise ValueError('The requested morph map does not exist\n' +
                             'Perhaps you need to run the MNE tool:\n' +
                             '  mne_make_morph_maps --from %s --to %s'
                             % (subject_from, subject_to))

    fid, tree, _ = fiff_open(name)

    # Locate all maps
    maps = dir_tree_find(tree, FIFF.FIFFB_MNE_MORPH_MAP)
    if len(maps) == 0:
        fid.close()
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

    fid.close()
    if left_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    if right_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    return left_map, right_map
