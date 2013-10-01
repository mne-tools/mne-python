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
from .surface import read_surface, _create_surf_spacing, _get_ico_surface, \
                     _tessellate_sphere_surf, read_bem_surfaces, \
                     _read_surface_geom, _normalize_vectors, \
                     _complete_surface_info, _compute_nearest, \
                     fast_cross_3d
from .utils import get_subjects_dir, run_subprocess, has_freesurfer, \
                   has_nibabel, logger, verbose
from .fixes import in1d
from .transforms import invert_transform, apply_trans, _print_coord_trans, \
                        combine_transforms
if has_nibabel():
    import nibabel as nib


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
        if src_type == FIFF.FIFFV_MNE_SPACE_SURFACE:
            res['type'] = 'surf'
        elif src_type == FIFF.FIFFV_MNE_SPACE_VOLUME:
            res['type'] = 'vol'
        elif src_type == FIFF.FIFFV_MNE_SPACE_DISCRETE:
            res['type'] = 'discrete'
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
    this['tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
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
        this['use_tri_nn'] = fast_cross_3d((r2 - r1), (r3 - r1))
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
def _write_source_spaces_to_fid(fid, src, verbose=None):
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

    _write_source_spaces_to_fid(fid, src, verbose)

    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)


def _write_one_source_space(fid, this, verbose=None):
    """Write one source space"""
    if this['type'] == 'surf':
        src_type = FIFF.FIFFV_MNE_SPACE_SURFACE
    elif this['type'] == 'vol':
        src_type = FIFF.FIFFV_MNE_SPACE_VOLUME
    elif this['type'] == 'discrete':
        src_type = FIFF.FIFFV_MNE_SPACE_DISCRETE
    else:
        raise ValueError('Unknown source space type (%s)' % this['type'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TYPE, src_type)
    if this['id'] >= 0:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_ID, this['id'])

    data = this.get('subject_his_id', None)
    if data:
        write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, data)
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, this['coord_frame'])

    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, this['np'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_POINTS, this['rr'])
    write_float_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS, this['nn'])

    #   Which vertices are active
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION, this['inuse'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE, this['nuse'])

    if this['ntri'] > 0:
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NTRI, this['ntri'])
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES,
                         this['tris'] + 1)

    if this['type'] != 'vol' and this['use_tris'] is not None:
        #   Use triangulation
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI, this['nuse_tri'])
        write_int_matrix(fid, FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES,
                         this['use_tris'] + 1)

    if this['type'] == 'vol':
        neighbor_vert = this.get('neighbor_vert', None)
        if neighbor_vert is not None:
            nneighbors = np.array([len(n) for n in neighbor_vert])
            neighbors = np.concatenate(neighbor_vert)
            write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NNEIGHBORS, nneighbors)
            write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NEIGHBORS, neighbors)

        write_coord_trans(fid, this['src_mri_t'])

        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS, this['shape'])

        start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        write_coord_trans(fid, this['mri_ras_t'])
        write_coord_trans(fid, this['vox_mri_t'])

        mri_volume_name = this.get('mri_volume_name', None)
        if mri_volume_name is not None:
            write_string(fid, FIFF.FIFF_MNE_FILE_NAME, mri_volume_name)

        write_float_sparse_rcs(fid, FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR,
                               this['interpolator'])

        if 'mri_file' in this and this['mri_file'] is not None:
            write_string(fid, FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE,
                         this['mri_file'])

        write_int(fid, FIFF.FIFF_MRI_WIDTH, this['mri_width'])
        write_int(fid, FIFF.FIFF_MRI_HEIGHT, this['mri_height'])
        write_int(fid, FIFF.FIFF_MRI_DEPTH, this['mri_depth'])

        end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

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

    if has_nibabel():
        use_nibabel = True
    else:
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
def setup_source_space(subject, fname=True, spacing='oct6', surface='white',
                       overwrite=False, subjects_dir=None, verbose=None):
    """Setup a source space with subsampling

    Parameters
    ----------
    subject : str
        Subject to process.
    fname : str | None | bool
        Filename to use. If True, a default name will be used. If None,
        the source space will not be saved (only returned).
    spacing : str
        The spacing to use. Can be ``'ico#'`` for a recursively subdivided
        icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
        or ``'all'`` for all points.
    surface : str
        The surface to use.
    overwrite: bool
        If True, overwrite output file (if it exists).
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : list
        The source space for each hemisphere.
    """
    cmd = ('setup_source_space(%s, fname=%s, spacing=%s, surface=%s, '
           'overwrite=%s, subjects_dir=%s, verbose=%s)'
           % (subject, fname, spacing, surface, overwrite,
              subjects_dir, verbose))
    # check to make sure our parameters are good, parse 'spacing'
    space_err = ('"spacing" must be a string with values '
                 '"ico#", "oct#", or "all", and "ico" and "oct"'
                 'numbers must be integers')
    if not isinstance(spacing, basestring) or len(spacing) < 3:
        raise ValueError(space_err)
    if spacing == 'all':
        stype = 'all'
        sval = ''
    elif spacing[:3] == 'ico':
        stype = 'ico'
        sval = spacing[3:]
    elif spacing[:3] == 'oct':
        stype = 'oct'
        sval = spacing[3:]
    else:
        raise ValueError(space_err)
    try:
        if stype in ['ico', 'oct']:
            sval = int(sval)
        elif stype == 'spacing':  # spacing
            sval = float(sval)
    except:
        raise ValueError(space_err)
    subjects_dir = get_subjects_dir(subjects_dir)
    surfs = [op.join(subjects_dir, subject, 'surf', hemi + surface)
             for hemi in ['lh.', 'rh.']]
    bem_dir = op.join(subjects_dir, subject, 'bem')

    for surf, hemi in zip(surfs, ['LH', 'RH']):
        if surf is not None and not op.isfile(surf):
            raise IOError('Could not find the %s surface %s'
                          % (hemi, surf))

    if not (fname is True or fname is None or isinstance(fname, basestring)):
        raise ValueError('"fname" must be a string, True, or None')
    if fname is True:
        extra = '%s-%s' % (stype, sval) if sval != '' else stype
        fname = op.join(bem_dir, '%s-%s-src.fif' % (subject, extra))
    if fname is not None and op.isfile(fname) and overwrite is False:
        raise IOError('file "%s" exists, use overwrite=True if you want '
                      'to overwrite the file' % fname)

    logger.info('Setting up the source space with the following parameters:\n')
    logger.info('SUBJECTS_DIR = %s' % subjects_dir)
    logger.info('Subject      = %s' % subject)
    logger.info('Surface      = %s' % surface)
    if stype == 'ico':
        src_type_str = 'ico = %s' % sval
        logger.info('Icosahedron subdivision grade %s\n' % sval)
    elif stype == 'oct':
        src_type_str = 'oct = %s' % sval
        logger.info('Octahedron subdivision grade %s\n' % sval)
    else:
        src_type_str = 'all'
        logger.info('Include all vertices\n')

    # Create the fif file
    if fname is not None:
        logger.info('>>> 1. Creating the source space file %s...' % fname)
    else:
        logger.info('>>> 1. Creating the source space...\n')

    # mne_make_source_space ... actually make the source spaces
    src = []

    # pre-load ico/oct surf (once) for speed, if necessary
    if stype in ['ico', 'oct']:
        ### from mne_ico_downsample.c ###
        if stype == 'ico':
            logger.info('Doing the icosahedral vertex picking...')
            ico_surf = _get_ico_surface(sval)
        else:
            logger.info('Doing the octahedral vertex picking...')
            ico_surf = _tessellate_sphere_surf(sval)
    else:
        ico_surf = None

    for hemi, surf in zip(['lh', 'rh'], surfs):
        logger.info('Loading %s...' % surf)
        s = _create_surf_spacing(surf, hemi, subject, stype, sval, ico_surf,
                                 subjects_dir)
        logger.info('loaded %s %d/%d selected to source space (%s)'
                    % (op.split(surf)[1], s['nuse'], s['np'], src_type_str))
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
    if fname is not None:
        write_source_spaces(fname, src)
        logger.info('Wrote %s' % fname)
    logger.info('You are now one step closer to computing the gain matrix')
    return src


@verbose
def setup_volume_source_space(subject, fname=None, pos=5.0, mri=None,
                              sphere=(0.0, 0.0, 0.0, 90.0), bem=None,
                              surface=None, mindist=5.0, exclude=0.0,
                              use_all=False, overwrite=False,
                              subjects_dir=None, verbose=None):
    """Setup a volume source space with grid spacing

    Parameters
    ----------
    subject : str
        Subject to process.
    fname : str | None
        Filename to use. If None, the source space will not be saved
        (only returned).
    pos : float | dict
        Positions to use for sources. If float, a grid will be constructed
        with the spacing given by `pos` in mm. If dict, pos['rr'] and
        pos['nn'] will be used as the source space locations (in meters)
        and normals, respectively.
    mri : str | None
        The filename of an MRI volume (mgh or mgz) to create the
        interpolation matrix over. Source estimates obtained in the
        volume source space can then be morphed onto the MRI volume
        using this interpolator. If pos is supplied, this can be None.
    sphere : array_like (length 4)
        Define spherical source space bounds using origin and radius given
        by (ox, oy, oz, rad) in mm. Only used if `bem` and `surface` are
        both None.
    bem : str | None
        Define source space bounds using a BEM file (specifically the inner
        skull surface).
    surface : str | dict | None
        Define source space bounds using a FreeSurfer surface file. Can
        also be a dictionary with entries `'rr'` and `'tris'`, such as
        those returned by `read_surface()`.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass
        of the bounding surface.
    overwrite: bool
        If True, overwrite output file (if it exists).
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    src : list
        The source space. Note that this list will have length 1 for
        compatibility reasons, as most functions expect source spaces
        to be provided as lists).
    """
    if bem is not None and surface is not None:
        raise ValueError('Only one of "bem" and "surface" should be '
                         'specified')
    if mri is not None:
        if not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if not has_nibabel():
            raise RuntimeError('nibabel is required to process mri data')

    sphere = np.asarray(sphere)
    if sphere.size != 4:
        raise ValueError('"sphere" must be array_like with 4 elements')

    # triage bounding argument
    if bem is not None:
        logger.info('BEM file              : %s', bem)
    elif surface is not None:
        if isinstance(surface, dict):
            if not all([key in surface for key in ['rr', 'tris']]):
                raise KeyError('surface, if dict, must have entries "rr" '
                               'and "tris"')
            # let's make sure we have geom info
            surface = _read_surface_geom(surface, verbose=False)
            surf_extra = 'dict()'
        elif isinstance(surface, basestring):
            if not op.isfile(surface):
                raise IOError('surface file "%s" not found' % surface)
            surf_extra = surface
        logger.info('Boundary surface file : %s', surf_extra)
    else:
        logger.info('Sphere                : origin at (%.1f %.1f %.1f) mm'
                    % (sphere[0], sphere[1], sphere[2]))
        logger.info('              radius  : %.1f mm' % sphere[3])

    # triage pos argument
    if isinstance(pos, dict):
        if not all([key in pos for key in ['rr', 'nn']]):
            raise KeyError('pos, if dict, must contain "rr" and "nn"')
        pos_extra = 'dict()'
    else:  # pos should be float-like
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            raise ValueError('pos must be a dict, or something that can be '
                             'cast to float()')
    if not isinstance(pos, float):
        logger.info('Source location file  : %s', pos_extra)
        logger.info('Assuming input in millimeters')
        logger.info('Assuming input in MRI coordinates')

    logger.info('Output file           : %s', fname)
    if isinstance(pos, float):
        logger.info('grid                  : %.1f mm' % pos)
        logger.info('mindist               : %.1f mm' % mindist)
        pos /= 1000.0
    if exclude > 0.0:
        logger.info('Exclude               : %.1f mm' % exclude)
    if mri is not None:
        logger.info('MRI volume            : %s' % mri)
    exclude /= 1000.0
    logger.info('')

    # Explicit list of points
    if not isinstance(pos, float):
        # Make the grid of sources
        sp = _make_discrete_source_space(pos)
    else:
        # Load the brain surface as a template
        if bem is not None:
            surf = read_bem_surfaces(bem, s_id=FIFF.FIFFV_BEM_SURF_ID_BRAIN,
                                     verbose=False)
            logger.info('Loaded inner skull from %s (%d nodes)'
                        % (bem, surf['np']))
        elif surface is not None:
            if isinstance(surf, basestring):
                surf = _read_surface_geom(surface)
            else:
                surf = surface
            logger.info('Loaded bounding surface from %s (%d nodes)'
                        % (surface, surf['np']))
        else:  # Load an icosahedron and use that as the surface
            logger.info('Setting up the sphere...')
            surf = _get_ico_surface(3)

            # Scale and shift
            _normalize_vectors(surf['rr'])
            surf['rr'] *= sphere[3] / 1000.0  # scale by radius
            surf['rr'] += sphere[:3] / 1000.0  # move by center
            _complete_surface_info(surf, True)
        # Make the grid of sources
        sp = _make_volume_source_space(surf, pos, exclude, mindist)

    # Compute an interpolation matrix to show data in an MRI volume
    if mri is not None:
        _add_interpolator(sp, mri)

    if 'vol_dims' in sp:
        del sp['vol_dims']

    # Save it
    sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                   dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                   nuse_tri=0, tris=None))
    sp = SourceSpaces([sp], dict(working_dir=os.getcwd(), command_line='None'))
    if fname is not None:
        write_source_spaces(fname, sp, verbose=False)
    return sp


def _make_voxel_ras_trans(move, ras, voxel_size):
    """Make a transformation for MRI voxel to MRI surface RAS"""
    assert voxel_size.ndim == 1
    assert voxel_size.size == 3
    rot = ras.T * voxel_size[np.newaxis, :]
    assert rot.ndim == 2
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    trans = np.c_[np.r_[rot, np.zeros((1, 3))], np.r_[move, 1.0]]
    t = {'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL, 'to': FIFF.FIFFV_COORD_MRI,
         'trans': trans}
    return t


def _make_discrete_source_space(pos):
    """Use a discrete set of source locs/oris to make src space

    Parameters
    ----------
    pos : dict
        Must have entries "rr" and "nn". Data should be in meters.

    Returns
    -------
    src : dict
        The source space.
    """
    # process points
    rr = pos['rr'].copy()
    nn = pos['nn'].copy()
    if not (rr.ndim == nn.ndim == 2 and nn.shape[0] == nn.shape[0] and
            rr.shape[1] == nn.shape[1]):
        raise RuntimeError('"rr" and "nn" must both be 2D arrays with '
                           'the same number of rows and 3 columns')
    npts = rr.shape[0]
    _normalize_vectors(nn)
    nz = np.sum(np.sum(nn * nn, axis=1) == 0)
    if nz != 0:
        raise RuntimeError('%d sources have zero length normal' % nz)
    logger.info('Positions (in meters) and orientations')
    logger.info('%d sources' % npts)

    # Ready to make the source space
    coord_frame = FIFF.FIFFV_COORD_MRI
    sp = dict(coord_frame=coord_frame, type='discrete', nuse=npts, np=npts,
              inuse=np.ones(npts, int), vertno=np.arange(npts), rr=rr, nn=nn,
              id=-1)
    return sp


def _make_volume_source_space(surf, grid, exclude, mindist):
    """Make a source space which covers the volume bounded by surf"""

    # Figure out the grid size
    mins = np.min(surf['rr'], axis=0)
    maxs = np.max(surf['rr'], axis=0)
    cm = np.mean(surf['rr'], axis=0)  # center of mass

    # Define the sphere which fits the surface
    maxdist = np.sqrt(np.max(np.sum((surf['rr'] - cm) ** 2, axis=1)))

    logger.info('Surface CM = (%6.1f %6.1f %6.1f) mm'
                % (1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    logger.info('Surface fits inside a sphere with radius %6.1f mm'
                % (1000 * maxdist))
    logger.info('Surface extent:')
    for c, mi, ma in zip('xyz', mins, maxs):
        logger.info('    %s = %6.1f ... %6.1f mm' % (c, 1000 * mi, 1000 * ma))
    maxn = np.zeros(3, int)
    minn = np.zeros(3, int)
    for c in range(3):
        if maxs[c] > 0:
            maxn[c] = np.floor(np.abs(maxs[c]) / grid) + 1
        else:
            maxn[c] = -np.floor(np.abs(maxs[c]) / grid) - 1
        if mins[c] > 0:
            minn[c] = np.floor(np.abs(mins[c]) / grid) + 1
        else:
            minn[c] = -np.floor(np.abs(mins[c]) / grid) - 1

    logger.info('Grid extent:')
    for c, mi, ma in zip('xyz', minn, maxn):
        logger.info('    %s = %6.1f ... %6.1f mm'
                    % (c, 1000 * mi * grid, 1000 * ma * grid))

    # Now make the initial grid
    ns = maxn - minn + 1
    npts = np.prod(ns)
    nrow = ns[0]
    ncol = ns[1]
    nplane = nrow * ncol
    sp = dict(np=npts, rr=np.zeros((npts, 3)), nn=np.zeros((npts, 3)),
              inuse=np.ones(npts, int), type='vol', nuse=npts,
              coord_frame=FIFF.FIFFV_COORD_MRI, id=-1, shape=ns)
    sp['nn'][:, 2] = 1.0  # Source orientation is immaterial

    x = np.arange(minn[0], maxn[0] + 1)[np.newaxis, np.newaxis, :]
    y = np.arange(minn[1], maxn[1] + 1)[np.newaxis, :, np.newaxis]
    z = np.arange(minn[2], maxn[2] + 1)[:, np.newaxis, np.newaxis]
    z = np.tile(z, (1, ns[1], ns[0])).ravel()
    y = np.tile(y, (ns[2], 1, ns[0])).ravel()
    x = np.tile(x, (ns[2], ns[1], 1)).ravel()
    k = np.arange(npts)
    sp['rr'] = np.c_[x, y, z] * grid
    neigh = np.empty((26, npts), int)
    neigh.fill(-1)

    # Figure out each neighborhood:
    # 6-neighborhood first
    idxs = [z > minn[2], x < maxn[0], y < maxn[1],
            x > minn[0], y > minn[1], z < maxn[2]]
    offsets = [-nplane, 1, nrow, -1, -nrow, nplane]
    for n, idx, offset in zip(neigh[:6], idxs, offsets):
        n[idx] = k[idx] + offset

    # Then the rest to complete the 26-neighborhood

    # First the plane below
    idx1 = z > minn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[6, idx2] = k[idx2] + 1 - nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[7, idx3] = k[idx3] + 1 + nrow - nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[8, idx2] = k[idx2] + nrow - nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[9, idx3] = k[idx3] - 1 + nrow - nplane
    neigh[10, idx2] = k[idx2] - 1 - nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[11, idx3] = k[idx3] - 1 - nrow - nplane

    idx2 = np.logical_and(idx1,  y > minn[1])
    neigh[12, idx2] = k[idx2] - nrow - nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[13, idx3] = k[idx3] + 1 - nrow - nplane

    # Then the same plane
    idx1 = np.logical_and(x < maxn[0], y < maxn[1])
    neigh[14, idx1] = k[idx1] + 1 + nrow

    idx1 = x > minn[0]
    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[15, idx2] = k[idx2] - 1 + nrow
    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[16, idx2] = k[idx2] - 1 - nrow

    idx1 = np.logical_and(y > minn[1], x < maxn[0])
    neigh[17, idx1] = k[idx1] + 1 - nrow - nplane

    # Finally one plane above
    idx1 = z < maxn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[18, idx2] = k[idx2] + 1 + nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[19, idx3] = k[idx3] + 1 + nrow + nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[20, idx2] = k[idx2] + nrow + nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[21, idx3] = k[idx3] - 1 + nrow + nplane
    neigh[22, idx2] = k[idx2] - 1 + nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[23, idx3] = k[idx3] - 1 - nrow + nplane

    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[24, idx2] = k[idx2] - nrow + nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[25, idx3] = k[idx3] + 1 - nrow + nplane

    logger.info('%d sources before omitting any.', sp['nuse'])

    # Exclude infeasible points
    dists = np.sqrt(np.sum((sp['rr'] - cm) ** 2, axis=1))
    bads = np.where(np.logical_or(dists < exclude, dists > maxdist))[0]
    sp['inuse'][bads] = False
    sp['nuse'] -= len(bads)
    logger.info('%d sources after omitting infeasible sources.', sp['nuse'])

    _filter_source_spaces(surf, mindist, None, [sp])
    logger.info('%d sources remaining after excluding the sources outside '
                'the surface and less than %6.1f mm inside.'
                % (sp['nuse'], mindist))

    # Omit unused vertices from the neighborhoods
    logger.info('Adjusting the neighborhood info...')
    # remove non source-space points
    log_inuse = sp['inuse'] > 0
    neigh[:, np.logical_not(log_inuse)] = -1
    # remove these points from neigh
    vertno = np.where(log_inuse)[0]
    sp['vertno'] = vertno
    old_shape = neigh.shape
    neigh = neigh.ravel()
    checks = np.where(neigh >= 0)[0]
    removes = np.logical_not(in1d(checks, vertno))
    neigh[checks[removes]] = -1
    neigh.shape = old_shape
    neigh = neigh.T
    # Thought we would need this, but C code keeps -1 vertices, so we will:
    #neigh = [n[n >= 0] for n in enumerate(neigh[vertno])]
    sp['neighbor_vert'] = neigh

    # Set up the volume data (needed for creating the interpolation matrix)
    r0 = minn * grid
    voxel_size = grid * np.ones(3)
    ras = np.eye(3)
    sp['src_mri_t'] = _make_voxel_ras_trans(r0, ras, voxel_size)
    sp['vol_dims'] = maxn - minn + 1
    sp['voxel_dims'] = voxel_size
    return sp


def _vol_vertex(width, height, jj, kk, pp):
    return jj + width * kk + pp * (width * height)


def _add_interpolator(s, mri_name):
    """Compute a sparse matrix to interpolate the data into an MRI volume"""
    # extract transformation information from mri
    logger.info('Reading %s...' % mri_name)
    mri_hdr = nib.load(mri_name).get_header()
    mri_width, mri_height, mri_depth = mri_hdr.get_data_shape()
    s.update(dict(mri_width=mri_width, mri_height=mri_height,
                  mri_depth=mri_depth))
    trans = mri_hdr.get_vox2ras_tkr()
    trans[:3, :] /= 1000.0
    s['vox_mri_t'] = {'trans': trans, 'from': FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                      'to': FIFF.FIFFV_COORD_MRI}  # ras_tkr
    trans = linalg.inv(np.dot(mri_hdr.get_vox2ras_tkr(),
                              mri_hdr.get_ras2vox()))
    trans[:3, 3] /= 1000.0
    s['mri_ras_t'] = {'trans': trans, 'from': FIFF.FIFFV_COORD_MRI,
                      'to': FIFF.FIFFV_MNE_COORD_RAS}  # ras

    _print_coord_trans(s['src_mri_t'], 'Source space : ')
    _print_coord_trans(s['vox_mri_t'], 'MRI volume : ')
    _print_coord_trans(s['mri_ras_t'], 'MRI volume : ')
    # Convert from destination to source volume coords
    combo_trans = combine_transforms(s['vox_mri_t'],
                                     invert_transform(s['src_mri_t']),
                                     FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                                     FIFF.FIFFV_MNE_COORD_MRI_VOXEL)
    combo_trans['trans'] = combo_trans['trans'].astype(np.float32)

    logger.info('Setting up interpolation...')
    js = np.arange(mri_width, dtype=np.float32)
    js = np.tile(js[np.newaxis, np.newaxis, :],
                 (mri_depth, mri_height, 1)).ravel()
    ks = np.arange(mri_height, dtype=np.float32)
    ks = np.tile(ks[np.newaxis, :, np.newaxis],
                 (mri_depth, 1, mri_width)).ravel()
    ps = np.arange(mri_depth, dtype=np.float32)
    ps = np.tile(ps[:, np.newaxis, np.newaxis],
                 (1, mri_height, mri_width)).ravel()

    r0 = apply_trans(combo_trans['trans'], np.c_[js, ks, ps])
    del js, ks, ps
    rn = np.floor(r0).astype(int)
    maxs = (s['vol_dims'] - 1)[np.newaxis, :]
    good = np.logical_and(np.all(rn >= 0, axis=1), np.all(rn < maxs, axis=1))
    rn = rn[good]
    r0 = r0[good]
    jj = rn[:, 0]
    kk = rn[:, 1]
    pp = rn[:, 2]
    vss = np.empty((8, len(jj)), int)
    width = s['vol_dims'][0]
    height = s['vol_dims'][1]
    vss[0, :] = _vol_vertex(width, height, jj, kk, pp)
    vss[1, :] = _vol_vertex(width, height, jj + 1, kk, pp)
    vss[2, :] = _vol_vertex(width, height, jj + 1, kk + 1, pp)
    vss[3, :] = _vol_vertex(width, height, jj, kk + 1, pp)
    vss[4, :] = _vol_vertex(width, height, jj, kk, pp + 1)
    vss[5, :] = _vol_vertex(width, height, jj + 1, kk, pp + 1)
    vss[6, :] = _vol_vertex(width, height, jj + 1, kk + 1, pp + 1)
    vss[7, :] = _vol_vertex(width, height, jj, kk + 1, pp + 1)
    del jj, kk, pp
    uses = np.any(s['inuse'][vss], axis=0)

    verts = vss[:, uses].ravel()  # vertex (col) numbers in csr matrix
    row_idx = np.tile(np.where(good)[0][uses], (8, 1)).ravel()

    # figure out weights for each vertex
    r0 = r0[uses]
    rn = rn[uses]
    xf = r0[:, 0] - rn[:, 0].astype(np.float32)
    yf = r0[:, 1] - rn[:, 1].astype(np.float32)
    zf = r0[:, 2] - rn[:, 2].astype(np.float32)
    omxf = 1.0 - xf
    omyf = 1.0 - yf
    omzf = 1.0 - zf
    weights = np.concatenate([omxf * omyf * omzf,  # correspond to rows of vss
                              xf * omyf * omzf,
                              xf * yf * omzf,
                              omxf * yf * omzf,
                              omxf * omyf * zf,
                              xf * omyf * zf,
                              xf * yf * zf,
                              omxf * yf * zf])
    del xf, yf, zf, omxf, omyf, omzf

    # Compose the sparse matrix
    ij = (row_idx, verts)
    nvox = mri_width * mri_height * mri_depth
    interp = sparse.csr_matrix((weights, ij), shape=(nvox, s['np']))
    s['interpolator'] = interp
    s['mri_volume_name'] = mri_name
    logger.info(' %d/%d nonzero values [done]' % (len(weights), nvox))


@verbose
def _filter_source_spaces(surf, limit, mri_head_t, src, verbose=None):
    """Remove all source space points closer than a given limit"""
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD and mri_head_t is None:
        raise RuntimeError('Source spaces are in head coordinates and no '
                           'coordinate transform was provided!')

    # How close are the source points to the surface?
    out_str = 'Source spaces are in '

    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        inv_trans = invert_transform(mri_head_t)
        out_str += 'head coordinates.'
    elif src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        out_str += 'MRI coordinates.'
    else:
        out_str += 'unknown (%d) coordinates.' % src[0]['coord_frame']
    logger.info(out_str)
    out_str = 'Checking that the sources are inside the bounding surface'
    if limit > 0.0:
        out_str += ' and at least %6.1f mm away' % (limit)
    logger.info(out_str + ' (will take a few...)')

    for s in src:
        vertno = np.where(s['inuse'])[0]  # can't trust s['vertno'] this deep
        # Convert all points here first to save time
        r1s = s['rr'][vertno]
        if s['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            r1s = apply_trans(inv_trans['trans'], r1s)

        # Check that the source is inside surface (often the inner skull)
        x = _sum_solids_div(r1s, surf)
        outside = np.abs(x - 1.0) > 1e-5
        omit_outside = np.sum(outside)

        # vectorized nearest using BallTree (or cdist)
        omit = 0
        if limit > 0.0:
            dists = _compute_nearest(surf['rr'], r1s, return_dists=True)[1]
            close = np.logical_and(dists < limit / 1000.0,
                                   np.logical_not(outside))
            omit = np.sum(close)
            outside = np.logical_or(outside, close)
        s['inuse'][vertno[outside]] = False
        s['nuse'] -= (omit + omit_outside)

        if omit_outside > 0:
            logger.info('%d source space points omitted because they are '
                        'outside the inner skull surface.' % omit_outside)
        if omit > 0:
            logger.info('%d source space points omitted because of the '
                        '%6.1f-mm distance limit.' % (omit, limit))
    logger.info('Thank you for waiting.')


def _sum_solids_div(fros, surf):
    """Compute sum of solid angles according to van Oosterom for all tris"""
    # NOTE: This incorporates the division by 4PI that used to be separate
    tot_angle = np.zeros((len(fros)))
    for tri in surf['tris']:
        v1 = fros - surf['rr'][tri[0]]
        v2 = fros - surf['rr'][tri[1]]
        v3 = fros - surf['rr'][tri[2]]
        triple = np.sum(fast_cross_3d(v1, v2) * v3, axis=1)
        l1 = np.sqrt(np.sum(v1 * v1, axis=1))
        l2 = np.sqrt(np.sum(v2 * v2, axis=1))
        l3 = np.sqrt(np.sum(v3 * v3, axis=1))
        s = (l1 * l2 * l3 +
             np.sum(v1 * v2, axis=1) * l3 +
             np.sum(v1 * v3, axis=1) * l2 +
             np.sum(v2 * v3, axis=1) * l1)
        tot_angle -= np.arctan2(triple, s)
    return tot_angle / (2 * np.pi)
