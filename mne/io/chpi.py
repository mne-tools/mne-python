# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op

from .pick import pick_types
from .base import _BaseRaw
from ..utils import verbose
from ..externals.six import string_types


@verbose
def get_chpi_positions(raw, t_step=None, verbose=None):
    """Extract head positions

    Note that the raw instance must have CHPI channels recorded.

    Parameters
    ----------
    raw : instance of Raw | str
        Raw instance to extract the head positions from. Can also be a
        path to a Maxfilter log file (str).
    t_step : float | None
        Sampling interval to use when converting data. If None, it will
        be automatically determined. By default, a sampling interval of
        1 second is used if processing a raw data. If processing a
        Maxfilter log file, this must be None because the log file
        itself will determine the sampling interval.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    translation : array
        A 2-dimensional array of head position vectors (n_time x 3).
    rotation : array
        A 3-dimensional array of rotation matrices (n_time x 3 x 3).
    t : array
        The time points associated with each position (n_time).

    Notes
    -----
    The digitized HPI head frame y is related to the frame position X as:

        Y = np.dot(rotation, X) + translation

    Note that if a Maxfilter log file is being processed, the start time
    may not use the same reference point as the rest of mne-python (i.e.,
    it could be referenced relative to raw.first_samp or something else).
    """
    if isinstance(raw, _BaseRaw):
        # for simplicity, we'll sample at 1 sec intervals like maxfilter
        if t_step is None:
            t_step = 1.0
        if not np.isscalar(t_step):
            raise TypeError('t_step must be a scalar or None')
        picks = pick_types(raw.info, meg=False, ref_meg=False,
                           chpi=True, exclude=[])
        if len(picks) == 0:
            raise RuntimeError('raw file has no CHPI channels')
        time_idx = raw.time_as_index(np.arange(0, raw.times[-1], t_step))
        data = [raw[picks, ti] for ti in time_idx]
        t = np.array([d[1] for d in data])
        data = np.array([d[0][:, 0] for d in data])
        data = np.c_[t, data]
    else:
        if not isinstance(raw, string_types):
            raise TypeError('raw must be an instance of Raw or string')
        if not op.isfile(raw):
            raise IOError('File "%s" does not exist' % raw)
        if t_step is not None:
            raise ValueError('t_step must be None if processing a log')
        data = np.loadtxt(raw, skiprows=1)  # first line is header, skip it
    return _quats_to_trans_rot_t(data)


def _quats_to_trans_rot_t(quats):
    """Convert Maxfilter-formatted head position quaternions

    Parameters
    ----------
    quats : ndarray, shape (N, 10)
        Maxfilter-formatted quaternions.

    Returns
    -------
    translation : ndarray, shape (N, 3)
        Translations at each time point.
    rotation : ndarray, shape (N, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (N,)
        The time points.

    See Also
    --------
    calculate_chpi_positions, get_chpi_positions
    """
    t = quats[:, 0].copy()
    translation = quats[:, 4:7].copy()
    rotation = _quat_to_rot(quats[:, 1:4])
    return translation, rotation, t


def _quat_to_rot(q):
    """Helper to convert quaternions to rotations"""
    # z = a + bi + cj + dk
    # use max() here to be safe in case roundoff errs put us over
    a = np.sqrt(np.maximum(1. - np.sum(q[..., 0:3] ** 2, axis=-1), 0.))
    b, c, d = q[..., 0], q[..., 1], q[..., 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    rotation = np.array([(aa + bb - cc - dd, bc_2 - ad_2, bd_2 + ac_2),
                         (bc_2 + ad_2, aa + cc - bb - dd, cd_2 - ab_2),
                         (bd_2 - ac_2, cd_2 + ab_2, aa + dd - bb - cc),
                         ])
    rotation = np.rollaxis(np.rollaxis(rotation, 1, q.ndim + 1), 0, q.ndim)
    return rotation


def _rot_to_quat(rot):
    """Here we derive qw from qx, qy, qz"""
    qw_4 = np.sqrt(1 + rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]) * 2
    qx = (rot[..., 2, 1] - rot[..., 1, 2]) / qw_4
    qy = (rot[..., 0, 2] - rot[..., 2, 0]) / qw_4
    qz = (rot[..., 1, 0] - rot[..., 0, 1]) / qw_4
    return np.rollaxis(np.array((qx, qy, qz)), 0, rot.ndim - 1)
