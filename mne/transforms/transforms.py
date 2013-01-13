# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import numpy as np
from numpy import sin, cos
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..fiff import FIFF
from ..fiff.open import fiff_open
from ..fiff.tag import read_tag, find_tag
from ..fiff.tree import dir_tree_find
from ..fiff.write import start_file, end_file, start_block, end_block, \
                   write_coord_trans, write_dig_point, write_int


def apply_trans(trans, pts):
    """Apply a transform matrix to an array of points

    Parameters
    ----------
    trans : array, shape = (4, 4)
        Transform matrix.
    pts : array, shape = (3,) | (n, 3)
        Array with coordinates for one or n points.

    Returns
    -------
    transformed_pts : shape = (3,) | (n, 3)
        Transformed point(s).
    """
    trans = np.asarray(trans)
    pts = np.asarray(pts)
    if pts.ndim == 1:
        pts = np.vstack((pts[:, None], [1]))
        pts = np.dot(trans, pts)
        pts = pts[:3, 0]
    else:
        pts = np.vstack((pts.T, np.ones(len(pts))))
        pts = np.dot(trans, pts)
        pts = pts[:3].T
    return pts


def rotation(x=0, y=0, z=0):
    """Create an array with a rotation matrix

    Parameters
    ----------
    x, y, z : scalar
        Rotation around the origin (in rad).

    Returns
    -------
    r : array, shape = (4, 4)
        The rotation matrix.
    """
    cos_x = cos(x)
    cos_y = cos(y)
    cos_z = cos(z)
    sin_x = sin(x)
    sin_y = sin(y)
    sin_z = sin(z)
    r = np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z,
                   sin_x * sin_z + cos_x * sin_y * cos_z, 0],
                  [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z,
                   - sin_x * cos_z + cos_x * sin_y * sin_z, 0],
                  [-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
                  [0, 0, 0, 1]], dtype=float)
    return r


def scaling(x=1, y=1, z=1):
    """Create an array with a scaling matrix

    Parameters
    ----------
    x, y, z : scalar
        Scaling factors.

    Returns
    -------
    s : array, shape = (4, 4)
        The scaling matrix.
    """
    s = np.array([[x, 0, 0, 0],
                  [0, y, 0, 0],
                  [0, 0, z, 0],
                  [0, 0, 0, 1]], dtype=float)
    return s


def translation(x=0, y=0, z=0):
    """Create an array with a translation matrix

    Parameters
    ----------
    x, y, z : scalar
        Translation parameters.

    Returns
    -------
    m : array, shape = (4, 4)
        The translation matrix.
    """
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m


def read_trans(fname):
    """Read a -trans.fif file

    Parameters
    ----------
    fname : str
        The name of the file.

    Returns
    -------
    info : dict
        The contents of the trans file.
    """
    info = {}
    fid, tree, _ = fiff_open(fname)
    block = dir_tree_find(tree, FIFF.FIFFB_MNE)[0]

    tag = find_tag(fid, block, FIFF.FIFF_COORD_TRANS)
    info.update(tag.data)

    isotrak = dir_tree_find(block, FIFF.FIFFB_ISOTRAK)
    isotrak = isotrak[0]

    tag = find_tag(fid, isotrak, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        coord_frame = 0
    else:
        coord_frame = int(tag.data)

    info['dig'] = dig = []
    for k in range(isotrak['nent']):
        kind = isotrak['directory'][k].kind
        pos = isotrak['directory'][k].pos
        if kind == FIFF.FIFF_DIG_POINT:
            tag = read_tag(fid, pos)
            tag.data['coord_frame'] = coord_frame
            dig.append(tag.data)

    fid.close()
    return info


def write_trans(fname, info):
    """Write a -trans.fif file

    Parameters
    ----------
    fname : str
        The name of the file.
    info : dict
        Trans file data, as returned by read_trans.
    """
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MNE)

    write_coord_trans(fid, info)

    dig = info['dig']
    if dig:
        start_block(fid, FIFF.FIFFB_ISOTRAK)

        coord_frames = set(d['coord_frame'] for d in dig)
        if len(coord_frames) > 1:
            raise ValueError("dig points in different coord_frames")
        coord_frame = coord_frames.pop()
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)

        for d in dig:
            write_dig_point(fid, d)
        end_block(fid, FIFF.FIFFB_ISOTRAK)

    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)


def invert_transform(trans):
    """Invert a transformation between coordinate systems
    """
    itrans = {'to': trans['from'], 'from': trans['to'],
              'trans': linalg.inv(trans['trans'])}
    return itrans


def transform_source_space_to(src, dest, trans):
    """Transform source space data to the desired coordinate system

    Parameters
    ----------
    src : dict
        Source space.
    dest : int
        Destination coordinate system (one of mne.fiff.FIFF.FIFFV_COORD_...).
    trans : dict
        Transformation.

    Returns
    -------
    res : dict
        Transformed source space. Data are modified in-place.
    """

    if src['coord_frame'] == dest:
        res = src
        return res

    if trans['to'] == src['coord_frame'] and trans['from'] == dest:
        trans = invert_transform(trans)
    elif trans['from'] != src['coord_frame'] or trans['to'] != dest:
        raise ValueError('Cannot transform the source space using this '
                         'coordinate transformation')

    t = trans['trans'][:3, :]
    res = src
    res['coord_frame'] = dest

    res['rr'] = np.dot(np.c_[res['rr'], np.ones((res['np'], 1))], t.T)
    res['nn'] = np.dot(np.c_[res['nn'], np.zeros((res['np'], 1))], t.T)
    return res


def transform_coordinates(filename, pos, orig, dest):
    """Transform coordinates between various MRI-related coordinate frames

    Parameters
    ----------
    filename: string
        Name of a fif file containing the coordinate transformations
        This file can be conveniently created with mne_collect_transforms
    pos: array of shape N x 3
        array of locations to transform (in meters)
    orig: 'meg' | 'mri'
        Coordinate frame of the above locations.
        'meg' is MEG head coordinates
        'mri' surface RAS coordinates
    dest: 'meg' | 'mri' | 'fs_tal' | 'mni_tal'
        Coordinate frame of the result.
        'mni_tal' is MNI Talairach
        'fs_tal' is FreeSurfer Talairach

    Returns
    -------
    trans_pos: array of shape N x 3
        The transformed locations

    Example
    -------
    transform_coordinates('all-trans.fif', np.eye(3), 'meg', 'fs_tal')
    transform_coordinates('all-trans.fif', np.eye(3), 'mri', 'mni_tal')
    """
    #   Read the fif file containing all necessary transformations
    fid, tree, directory = fiff_open(filename)

    coord_names = dict(mri=FIFF.FIFFV_COORD_MRI,
                       meg=FIFF.FIFFV_COORD_HEAD,
                       mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
                       fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL)

    orig = coord_names[orig]
    dest = coord_names[dest]

    T0 = T1 = T2 = T3plus = T3minus = None
    for d in directory:
        if d.kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, d.pos)
            trans = tag.data
            if (trans['from'] == FIFF.FIFFV_COORD_MRI and
                trans['to'] == FIFF.FIFFV_COORD_HEAD):
                T0 = invert_transform(trans)
            elif (trans['from'] == FIFF.FIFFV_COORD_MRI and
                  trans['to'] == FIFF.FIFFV_MNE_COORD_RAS):
                T1 = trans
            elif (trans['from'] == FIFF.FIFFV_MNE_COORD_RAS and
                  trans['to'] == FIFF.FIFFV_MNE_COORD_MNI_TAL):
                T2 = trans
            elif trans['from'] == FIFF.FIFFV_MNE_COORD_MNI_TAL:
                if trans['to'] == FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ:
                    T3plus = trans
                elif trans['to'] == FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ:
                    T3minus = trans
    fid.close()
    #
    #   Check we have everything we need
    #
    if ((orig == FIFF.FIFFV_COORD_HEAD and T0 is None) or (T1 is None)
            or (T2 is None) or (dest == FIFF.FIFFV_MNE_COORD_FS_TAL and
                                ((T3minus is None) or (T3minus is None)))):
        raise ValueError('All required coordinate transforms not found')

    #
    #   Go ahead and transform the data
    #
    if pos.shape[1] != 3:
        raise ValueError('Coordinates must be given in a N x 3 array')

    if dest == orig:
        trans_pos = pos.copy()
    else:
        n_points = pos.shape[0]
        pos = np.c_[pos, np.ones(n_points)].T
        if orig == FIFF.FIFFV_COORD_HEAD:
            pos = np.dot(T0['trans'], pos)
        elif orig != FIFF.FIFFV_COORD_MRI:
            raise ValueError('Input data must be in MEG head or surface RAS '
                             'coordinates')

        if dest == FIFF.FIFFV_COORD_HEAD:
            pos = np.dot(linalg.inv(T0['trans']), pos)
        elif dest != FIFF.FIFFV_COORD_MRI:
            pos = np.dot(np.dot(T2['trans'], T1['trans']), pos)
            if dest != FIFF.FIFFV_MNE_COORD_MNI_TAL:
                if dest == FIFF.FIFFV_MNE_COORD_FS_TAL:
                    for k in xrange(n_points):
                        if pos[2, k] > 0:
                            pos[:, k] = np.dot(T3plus['trans'], pos[:, k])
                        else:
                            pos[:, k] = np.dot(T3minus['trans'], pos[:, k])
                else:
                    raise ValueError('Illegal choice for the output '
                                     'coordinates')

        trans_pos = pos[:3, :].T

    return trans_pos


# @verbose
# def transform_meg_chs(chs, trans, verbose=None):
#     """
#     %
#     % [res, count] = fiff_transform_meg_chs(chs,trans)
#     %
#     % Move to another coordinate system in MEG channel channel info
#     % Count gives the number of channels transformed
#     %
#     % NOTE: Only the coil_trans field is modified by this routine, not
#     % loc which remains to reflect the original data read from the fif file
#     %
#     %
#
#     XXX
#     """
#
#     res = copy.deepcopy(chs)
#
#     count = 0
#     t = trans['trans']
#     for ch in res:
#         if (ch['kind'] == FIFF.FIFFV_MEG_CH
#                                     or ch['kind'] == FIFF.FIFFV_REF_MEG_CH):
#             if (ch['coord_frame'] == trans['from']
#                                             and ch['coil_trans'] is not None):
#                 ch['coil_trans'] = np.dot(t, ch['coil_trans'])
#                 ch['coord_frame'] = trans['to']
#                 count += 1
#
#     if count > 0:
#         logger.info('    %d MEG channel locations transformed' % count)
#
#     return res, count

# @verbose
# def transform_eeg_chs(chs, trans, verbose=None):
#     """
#     %
#     % [res, count] = fiff_transform_eeg_chs(chs,trans)
#     %
#     % Move to another coordinate system in EEG channel channel info
#     % Count gives the number of channels transformed
#     %
#     % NOTE: Only the eeg_loc field is modified by this routine, not
#     % loc which remains to reflect the original data read from the fif file
#     %
#
#     XXX
#     """
#     res = copy.deepcopy(chs)
#
#     count = 0
#     #
#     #   Output unaugmented vectors from the transformation
#     #
#     t = trans['trans'][:3,:]
#     for ch in res:
#         if ch['kind'] == FIFF.FIFFV_EEG_CH:
#             if (ch['coord_frame'] == trans['from']
#                                             and ch['eeg_loc'] is not None):
#                 #
#                 # Transform the augmented EEG location vectors
#                 #
#                 for p in range(ch['eeg_loc'].shape[1]):
#                     ch['eeg_loc'][:, p] = np.dot(t,
#                                                 np.r_[ch['eeg_loc'][:,p], 1])
#                 count += 1
#                 ch['coord_frame'] = trans['to']
#
#     if count > 0:
#         logger.info('    %d EEG electrode locations transformed\n' % count)
#
#     return res, count
