"""Numba-accelerated helpers for :mod:`mne.transforms`.

Kept in its own module so that ``import mne.transforms`` does not import numba;
the public wrappers in ``mne.transforms`` import this lazily. Jitted functions must
live together here because numba can only call other jitted functions in nopython
mode (e.g. ``_fit_matched_points`` calls ``_quat_to_rot``).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ._numba import jit


@jit()
def _quat_to_rot(quat):
    """Convert a set of quaternions to rotations (see :func:`mne.transforms.quat_to_rot`)."""  # noqa: E501
    # z = a + bi + cj + dk
    b, c, d = quat[..., 0], quat[..., 1], quat[..., 2]
    bb, cc, dd = b * b, c * c, d * d
    # use max() here to be safe in case roundoff errs put us over
    aa = np.maximum(1.0 - bb - cc - dd, 0.0)
    a = np.sqrt(aa)
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    rotation = np.empty(quat.shape[:-1] + (3, 3))
    rotation[..., 0, 0] = aa + bb - cc - dd
    rotation[..., 0, 1] = bc_2 - ad_2
    rotation[..., 0, 2] = bd_2 + ac_2
    rotation[..., 1, 0] = bc_2 + ad_2
    rotation[..., 1, 1] = aa + cc - bb - dd
    rotation[..., 1, 2] = cd_2 - ab_2
    rotation[..., 2, 0] = bd_2 - ac_2
    rotation[..., 2, 1] = cd_2 + ab_2
    rotation[..., 2, 2] = aa + dd - bb - cc
    return rotation


@jit()
def _one_rot_to_quat(rot, *, tol=1e-3):
    """Convert a rotation matrix to quaternions."""
    # see e.g. http://www.euclideanspace.com/maths/geometry/rotations/
    #                 conversions/matrixToQuaternion/
    det = np.linalg.det(np.reshape(rot, (3, 3)))
    if np.abs(det - 1.0) > tol:
        raise ValueError("Matrix is not a pure rotation, got determinant != 1")
    t = 1.0 + rot[0] + rot[4] + rot[8]
    if t > np.finfo(rot.dtype).eps:
        s = np.sqrt(t) * 2.0
        # qw = 0.25 * s
        qx = (rot[7] - rot[5]) / s
        qy = (rot[2] - rot[6]) / s
        qz = (rot[3] - rot[1]) / s
    elif rot[0] > rot[4] and rot[0] > rot[8]:
        s = np.sqrt(1.0 + rot[0] - rot[4] - rot[8]) * 2.0
        # qw = (rot[7] - rot[5]) / s
        qx = 0.25 * s
        qy = (rot[1] + rot[3]) / s
        qz = (rot[2] + rot[6]) / s
    elif rot[4] > rot[8]:
        s = np.sqrt(1.0 - rot[0] + rot[4] - rot[8]) * 2
        # qw = (rot[2] - rot[6]) / s
        qx = (rot[1] + rot[3]) / s
        qy = 0.25 * s
        qz = (rot[5] + rot[7]) / s
    else:
        s = np.sqrt(1.0 - rot[0] - rot[4] + rot[8]) * 2.0
        # qw = (rot[3] - rot[1]) / s
        qx = (rot[2] + rot[6]) / s
        qy = (rot[5] + rot[7]) / s
        qz = 0.25 * s
    return np.array((qx, qy, qz))


@jit()
def _fit_matched_points(p, x, weights=None, scale=False):
    """Fit matched points using an analytical formula."""
    # Follow notation of P.J. Besl and N.D. McKay, A Method for
    # Registration of 3-D Shapes, IEEE Trans. Patt. Anal. Machine Intell., 14,
    # 239 - 255, 1992.
    #
    # The original method is actually by Horn, Closed-form solution of absolute
    # orientation using unit quaternions, J Opt. Soc. Amer. A vol 4 no 4
    # pp 629-642, Apr. 1987. This paper describes how weights can be
    # easily incorporated, and a uniform scale factor can be computed.
    #
    # Caution: This can be dangerous if there are 3 points, or 4 points in
    #          a symmetric layout, as the geometry can be explained
    #          equivalently under 180 degree rotations.
    #
    # Eventually this can be extended to also handle a uniform scale factor,
    # as well.
    assert p.shape == x.shape
    assert p.ndim == 2
    assert p.shape[1] == 3
    # (weighted) centroids
    weights_ = np.full((p.shape[0], 1), 1.0 / max(p.shape[0], 1))
    if weights is not None:
        weights_[:] = np.reshape(weights / weights.sum(), (weights.size, 1))
    mu_p = np.dot(weights_.T, p)[0]
    mu_x = np.dot(weights_.T, x)[0]
    dots = np.dot(p.T, weights_ * x)
    Sigma_px = dots - np.outer(mu_p, mu_x)  # eq 24
    # x and p should no longer be used
    A_ij = Sigma_px - Sigma_px.T
    Delta = np.array([A_ij[1, 2], A_ij[2, 0], A_ij[0, 1]])
    tr_Sigma_px = np.trace(Sigma_px)
    # "N" in Horn:
    Q = np.empty((4, 4))
    Q[0, 0] = tr_Sigma_px
    Q[0, 1:] = Delta
    Q[1:, 0] = Delta
    Q[1:, 1:] = Sigma_px + Sigma_px.T - tr_Sigma_px * np.eye(3)
    _, v = np.linalg.eigh(Q)  # sorted ascending
    quat = np.empty(6)
    quat[:3] = v[1:, -1]
    if v[0, -1] != 0:
        quat[:3] *= np.sign(v[0, -1])
    rot = _quat_to_rot(quat[:3])
    # scale factor is easy once we know the rotation
    if scale:  # p is "right" (from), x is "left" (to) in Horn 1987
        dev_x = x - mu_x
        dev_p = p - mu_p
        dev_x *= dev_x
        dev_p *= dev_p
        if weights is not None:
            dev_x *= weights_
            dev_p *= weights_
        s = np.sqrt(np.sum(dev_x) / np.sum(dev_p))
    else:
        s = 1.0
    # translation is easy once rotation and scale are known
    quat[3:] = mu_x - s * np.dot(rot, mu_p)
    return quat, s
