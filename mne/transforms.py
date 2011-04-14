import copy
import numpy as np
from scipy import linalg

# from .fiff import FIFF


def invert_transform(trans):
    """Invert a transformation between coordinate systems
    """
    itrans = copy.copy(trans)
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
#     res = copy.copy(chs)
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
#         print '\t%d MEG channel locations transformed' % count
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
#     res = copy.copy(chs)
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
#         print '\t%d EEG electrode locations transformed\n' % count
#
#     return res, count
