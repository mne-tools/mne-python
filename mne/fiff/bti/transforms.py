# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#
#          simplified bsd-3 license


from .. constants import BTI

import numpy as np

BTI.T_ROT_VV = ((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (1, 1, 1, 1))
BTI.T_IDENT = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 1, 1, 1))
BTI.T_ROT_IX = slice(0, 3), slice(0, 3)
BTI.T_TRANS_IX = slice(0, 3), slice(3, 4)
BTI.T_SCA_IX = slice(3, 4), slice(0, 4)


def bti_to_vv_trans(adjust=None, translation=(0.0, 0.02, 0.11)):
    """ Get the general Magnes3600WH to Neuromag coordinate transform

    Parameters
    ----------
    adjust : int | None
        Degrees to tilt x-axis for sensor frame misalignment.
        If None, no adjustment will be applied.
    translation : array-like
        The translation to place the origin of coordinate system
        to the center of the head.

    Returns
    -------
    m_nm_t : ndarray
        4 x 4 rotation, translation, scaling matrix.

    """
    flip_t = np.array(BTI.T_ROT_VV, np.float64)
    adjust_t = np.array(BTI.T_IDENT, np.float64)
    adjust = 0 if adjust is None else adjust
    deg = np.deg2rad(float(adjust))
    adjust_t[[1, 2], [1, 2]] = np.cos(deg)
    adjust_t[[1, 2], [2, 1]] = -np.sin(deg), np.sin(deg)
    m_nm_t = np.ones([4, 4])
    m_nm_t[BTI.T_ROT_IX] = np.dot(flip_t[BTI.T_ROT_IX],
                                  adjust_t[BTI.T_ROT_IX])
    m_nm_t[BTI.T_TRANS_IX] = np.matrix(translation).T

    return m_nm_t


def bti_to_vv_coil_trans(ch_t, bti_t, nm_t, nm_default_scale=True):
    """ transforms 4D coil position to fiff / Neuromag
    """
    ch_t = np.array(ch_t.split(', '), dtype=float).reshape([4, 4])

    nm_coil_trans = apply_trans(inverse_trans(ch_t, bti_t), nm_t)

    if nm_default_scale:
        nm_coil_trans[3, :3] = 0.

    return nm_coil_trans


def inverse_trans(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX,
               scal=BTI.T_SCA_IX):
    """ Undo a transform
    """
    x = x.copy()
    x[scal] *= t[scal]
    x[rot] = np.dot(t[rot].T, x[rot])
    x[trans] -= t[trans]
    x[trans] = np.dot(t[rot].T, x[trans])

    return x


def apply_trans(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX,
                scal=BTI.T_SCA_IX):
    """ Apply a transform
    """
    x = x.copy()
    x[rot] = np.dot(t[rot], x[rot])
    x[trans] = np.dot(t[rot], x[trans])
    x[trans] += t[trans]
    x[scal] *= t[scal]

    return x


def merge_trans(t1, t2):
    """ Merge two transforms
    """
    t = np.array(BTI.T_IDENT, dtype=np.float32)
    t[BTI.T_ROT_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_ROT_IX])
    t[BTI.T_TRANS_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_TRANS_IX])
    t[BTI.T_TRANS_IX] += t1[BTI.T_TRANS_IX]

    return t
