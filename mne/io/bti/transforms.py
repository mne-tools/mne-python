# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#
#          simplified BSD-3 license


import numpy as np
from .constants import BTI


def bti_identity_trans(dtype='>f8'):
    """ Get BTi identity transform

    Parameters
    ----------
    dtype : str | dtype object
        The data format of the transform

    Returns
    -------
    itrans : ndarray
        The 4 x 4 transformation matrix.
    """
    return np.array(BTI.T_IDENT, dtype=dtype)


def bti_to_vv_trans(adjust=None, translation=(0.0, 0.02, 0.11), dtype='>f8'):
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
    flip_t = np.array(BTI.T_ROT_VV, dtype=dtype)
    adjust_t = bti_identity_trans(dtype=dtype)
    adjust = 0 if adjust is None else adjust
    deg = np.deg2rad(np.float64(adjust))
    adjust_t[[1, 2], [1, 2]] = np.cos(deg)
    adjust_t[[1, 2], [2, 1]] = -np.sin(deg), np.sin(deg)
    m_nm_t = np.ones([4, 4], dtype=dtype)
    m_nm_t[BTI.T_ROT_IX] = np.dot(flip_t[BTI.T_ROT_IX],
                                  adjust_t[BTI.T_ROT_IX])
    m_nm_t[BTI.T_TRANS_IX] = np.matrix(translation, dtype=dtype).T

    return m_nm_t


def bti_to_vv_coil_trans(ch_t, bti_t, nm_t, nm_default_scale=True):
    """ transforms 4D coil position to fiff / Neuromag
    """
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


def merge_trans(t1, t2, dtype='>f8'):
    """ Merge two transforms
    """
    t = bti_identity_trans(dtype=dtype)
    t[BTI.T_ROT_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_ROT_IX])
    t[BTI.T_TRANS_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_TRANS_IX])
    t[BTI.T_TRANS_IX] += t1[BTI.T_TRANS_IX]

    return t
