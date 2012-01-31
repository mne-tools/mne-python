import copy
import numpy as np
from scipy import linalg

from .fiff import FIFF
from .fiff.open import fiff_open
from .fiff.tag import read_tag


def invert_transform(trans):
    """Invert a transformation between coordinate systems
    """
    itrans = copy.deepcopy(trans)
    aux = itrans['from']
    itrans['from'] = itrans['to']
    itrans['to'] = aux
    itrans['trans'] = linalg.inv(itrans['trans'])
    return itrans


def transform_source_space_to(src, dest, trans):
    """Transform source space data to the desired coordinate system

    Parameters
    ----------
    src : dict
        Source space
    dest : dict
        destination coordinate system
    trans : dict
        Transformation

    Returns
    -------
    res : dict
        Transformed source space
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

    Return
    ------
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


# def transform_meg_chs(chs, trans):
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
#         print '    %d MEG channel locations transformed' % count
#
#     return res, count


# def transform_eeg_chs(chs, trans):
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
#         print '    %d EEG electrode locations transformed\n' % count
#
#     return res, count
