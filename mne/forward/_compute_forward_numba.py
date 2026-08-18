"""Numba-accelerated helpers for :mod:`mne.forward._compute_forward`.

Kept in its own module so that importing the forward machinery -- which happens via
``mne.chpi`` and ``mne.dipole`` -- does not import numba.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._numba import bincount, jit
from .._surface_numba import _jit_cross
from ._compute_forward import _MAG_FACTOR, _MIN_DIST_LIMIT


@jit()
def _do_lin_field_coeff(bem_rr, tris, tn, ta, rmags, cosmags, ws, bins):
    """Compute field coefficients (parallel-friendly).

    See section IV of Mosher et al., 1999 (specifically equation 35).

    Parameters
    ----------
    bem_rr : ndarray, shape (n_BEM_vertices, 3)
        Positions on one BEM surface in 3-space. 2562 BEM vertices for BEM with
        5120 triangles (ico-4)
    tris : ndarray, shape (n_BEM_vertices, 3)
        Vertex indices for each triangle (referring to bem_rr)
    tn : ndarray, shape (n_BEM_vertices, 3)
        Triangle unit normal vectors
    ta : ndarray, shape (n_BEM_vertices,)
        Triangle areas
    rmag : ndarray, shape (n_sensor_pts, 3)
        3D positions of MEG coil integration points (from coil['rmag'])
    cosmag : ndarray, shape (n_sensor_pts, 3)
        Direction of the MEG coil integration points (from coil['cosmag'])
    ws : ndarray, shape (n_sensor_pts,)
        Weights for MEG coil integration points
    bins : ndarray, shape (n_sensor_pts,)
        The sensor assignments for each rmag/cosmag/w.

    Returns
    -------
    coeff : ndarray, shape (n_MEG_sensors, n_BEM_vertices)
        Linear coefficients with effect of each BEM vertex on each sensor (?)
    """
    n_bem_rr = len(bem_rr)
    coeff = np.zeros((bins[-1] + 1, n_bem_rr))
    w_cosmags = ws.reshape(-1, 1) * cosmags
    # Store as (n_BEM_vertices, n_sensor_pts, 3): below we repeatedly index
    # a single BEM vertex out of this array, and this layout makes that a
    # contiguous slice along the leading axis instead of a strided gather
    # along a middle axis, which is ~2x faster in practice.
    diff = rmags.reshape(1, -1, 3) - bem_rr.reshape(n_bem_rr, 1, 3)
    den = np.sum(diff * diff, axis=-1)
    den *= np.sqrt(den)
    den *= 3
    for ti in range(len(tris)):
        tri, tri_nn, tri_area = tris[ti], tn[ti], ta[ti]
        # Accumulate the coefficients for each triangle node and add to the
        # corresponding coefficient matrix

        # Simple version (bem_lin_field_coeffs_simple)
        # The following is equivalent to:
        # tri_rr = bem_rr[tri]
        # for j, coil in enumerate(coils['coils']):
        #     x = func(coil['rmag'], coil['cosmag'],
        #              tri_rr, tri_nn, tri_area)
        #     res = np.sum(coil['w'][np.newaxis, :] * x, axis=1)
        #     coeff[j][tri + off] += mult * res
        for vi in range(3):
            idx = tri[vi]
            c = np.empty((diff.shape[1], 3))
            _jit_cross(c, diff[idx], tri_nn)
            c *= w_cosmags
            x = np.sum(c, axis=-1)
            x /= den[idx] / tri_area
            coeff[:, idx] += bincount(bins, weights=x, minlength=bins[-1] + 1)
    return coeff


@jit()
def _bem_inf_pots(mri_rr, bem_rr, mri_Q=None):
    """Compute the infinite medium potential in all 3 directions.

    Parameters
    ----------
    mri_rr : ndarray, shape (n_dipole_vertices, 3)
        Chunk of 3D dipole positions in MRI coordinates
    bem_rr: ndarray, shape (n_BEM_vertices, 3)
        3D vertex positions for one BEM surface
    mri_Q : ndarray, shape (3, 3)
        3x3 head -> MRI transform. I.e., head_mri_t.dot(np.eye(3))

    Returns
    -------
    ndarray : shape(n_dipole_vertices, 3, n_BEM_vertices)
    """
    # NOTE: the (μ_0 / (4π) factor has been moved to _prep_field_communication
    # Get position difference vector between BEM vertex and dipole
    diff = np.empty((len(mri_rr), 3, len(bem_rr)))
    for ri in range(mri_rr.shape[0]):
        rr = mri_rr[ri]
        this_diff = bem_rr - rr
        diff_norm = np.sum(this_diff * this_diff, axis=1)
        diff_norm *= np.sqrt(diff_norm)
        diff_norm[diff_norm == 0] = 1.0
        if mri_Q is not None:
            this_diff = np.dot(this_diff, mri_Q.T)
        this_diff /= diff_norm.reshape(-1, 1)
        diff[ri] = this_diff.T

    return diff


@jit()
def _bem_inf_fields(rr, rmag, cosmag):
    """Compute infinite-medium magnetic field at one MEG sensor.

    This operates on all dipoles in all 3 basis directions.

    Parameters
    ----------
    rr : ndarray, shape (n_source_points, 3)
        3D dipole source positions
    rmag : ndarray, shape (n_sensor points, 3)
        3D positions of 1 MEG coil's integration points (from coil['rmag'])
    cosmag : ndarray, shape (n_sensor_points, 3)
        Direction of 1 MEG coil's integration points (from coil['cosmag'])

    Returns
    -------
    ndarray, shape (n_dipoles, 3, n_integration_pts)
        Magnetic field from all dipoles at each MEG sensor integration point
    """
    # rr, rmag refactored according to Equation (19) in Mosher, 1999
    # Knowing that we're doing all directions, refactor above function:

    # rr, 3, rmag
    diff = rmag.T.reshape(1, 3, rmag.shape[0]) - rr.reshape(rr.shape[0], 3, 1)
    diff_norm = np.sum(diff * diff, axis=1)  # rr, rmag
    diff_norm *= np.sqrt(diff_norm)  # Get magnitude of distance cubed
    diff_norm_ = diff_norm.reshape(-1)
    diff_norm_[diff_norm_ == 0] = 1  # avoid nans

    # This is the result of cross-prod calcs with basis vectors,
    # as if we had taken (Q=np.eye(3)), then multiplied by cosmags
    # factor, and then summed across directions
    x = np.empty((rr.shape[0], 3, rmag.shape[0]))
    x[:, 0] = diff[:, 1] * cosmag[:, 2] - diff[:, 2] * cosmag[:, 1]
    x[:, 1] = diff[:, 2] * cosmag[:, 0] - diff[:, 0] * cosmag[:, 2]
    x[:, 2] = diff[:, 0] * cosmag[:, 1] - diff[:, 1] * cosmag[:, 0]
    diff_norm = diff_norm_.reshape((rr.shape[0], 1, rmag.shape[0]))
    x /= diff_norm
    # x.shape == (rr.shape[0], 3, rmag.shape[0])
    return x


@jit()
def _do_sphere_field(rrs, rmags, cosmags, ws, bins, r0):
    n_coils = bins[-1] + 1
    # Shift to the sphere model coordinates
    rrs = rrs - r0
    # this_poss and r don't depend on the dipole (ri), so compute once
    this_poss = rmags - r0
    r = np.sqrt(np.sum(this_poss * this_poss, axis=1))
    B = np.zeros((3 * len(rrs), n_coils))
    for ri in range(len(rrs)):
        rr = rrs[ri]
        # Check for a dipole at the origin
        if np.sqrt(np.dot(rr, rr)) <= 1e-10:
            continue
        # Vector from dipole to the field point
        a_vec = this_poss - rr
        a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
        rr0 = np.sum(this_poss * rr, axis=1)
        ar = (r * r) - rr0
        ar0 = ar / a
        F = a * (r * a + ar)
        gr = (a * a) / r + ar0 + 2.0 * (a + r)
        g0 = a + 2 * r + ar0
        # Compute the dot products needed
        re = np.sum(this_poss * cosmags, axis=1)
        r0e = np.sum(rr * cosmags, axis=1)
        g = (g0 * r0e - gr * re) / (F * F)
        good = (a > 0) | (r > 0) | ((a * r) + 1 > 1e-5)
        rr_ = rr.reshape(1, 3)
        v1 = np.empty((cosmags.shape[0], 3))
        _jit_cross(v1, rr_, cosmags)
        v2 = np.empty((cosmags.shape[0], 3))
        _jit_cross(v2, rr_, this_poss)
        xx = (good * ws).reshape(-1, 1) * (
            v1 / F.reshape(-1, 1) + v2 * g.reshape(-1, 1)
        )
        for jj in range(3):
            zz = bincount(bins, xx[:, jj], n_coils)
            B[3 * ri + jj, :] = zz
    B *= _MAG_FACTOR
    return B


@jit()
def _do_eeg_spherepot_coil(rrs, rmags, ws, bins, r0, rad, mu, lams):
    n_coils = bins[-1] + 1

    # Shift to the sphere model coordinates. This part (unlike the Berg-Scherg
    # equivalent-dipole loop below) does not depend on the dipole or fit
    # term, so it is computed only once rather than on every (ri, eq) pair.
    rrs = rrs - r0
    this_pos = rmags - r0
    r2 = np.sum(this_pos * this_pos, axis=1)
    r = np.sqrt(r2)

    B = np.zeros((3 * len(rrs), n_coils))
    for ri in range(len(rrs)):
        rr = rrs[ri]
        # Only process dipoles inside the innermost sphere
        if np.sqrt(np.dot(rr, rr)) >= rad:
            continue
        # fwd_eeg_spherepot_vec
        vval_one = np.zeros((len(rmags), 3))

        # Make a weighted sum over the equivalence parameters
        for eq in range(len(mu)):
            # Scale the dipole position
            rd = mu[eq] * rr
            rd2 = np.sum(rd * rd)
            rd2_inv = 1.0 / rd2

            # Vector from dipole to the field point
            a_vec = this_pos - rd

            # Compute the dot products needed
            a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
            a3 = 2.0 / (a * a * a)
            rrd = np.sum(this_pos * rd, axis=1)
            ra = r2 - rrd
            rda = rrd - rd2

            # The main ingredients
            F = a * (r * a + ra)
            c1 = a3 * rda + 1.0 / a - 1.0 / r
            c2 = a3 + (a + r) / (r * F)

            # Mix them together and scale by lambda/(rd*rd)
            m1 = c1 - c2 * rrd
            m2 = c2 * rd2

            vval_one += (
                lams[eq]
                * rd2_inv
                * (m1.reshape(-1, 1) * rd + m2.reshape(-1, 1) * this_pos)
            )

        # compute total result
        xx = vval_one * ws.reshape(-1, 1)
        for jj in range(3):
            B[3 * ri + jj] = bincount(bins, xx[:, jj], n_coils)
    # finishing by scaling by 1/(4*M_PI)
    B *= 0.25 / np.pi
    return B


@jit()
def _compute_mdfv(rrs, rmags, cosmags, ws, bins, too_close):
    """Compute an MEG forward solution for a set of magnetic dipoles."""
    # The code below is a more efficient version (~30x) of this:
    # for ri, rr in enumerate(rrs):
    #     for k in range(len(coils)):
    #         this_coil = coils[k]
    #         # Go through all points
    #         diff = this_coil['rmag'] - rr
    #         dist2 = np.sum(diff * diff, axis=1)[:, np.newaxis]
    #         dist = np.sqrt(dist2)
    #         if (dist < 1e-5).any():
    #             raise RuntimeError('Coil too close')
    #         dist5 = dist2 * dist2 * dist
    #         sum_ = (3 * diff * np.sum(diff * this_coil['cosmag'],
    #                                   axis=1)[:, np.newaxis] -
    #                 dist2 * this_coil['cosmag']) / dist5
    #         fwd[3*ri:3*ri+3, k] = 1e-7 * np.dot(this_coil['w'], sum_)
    fwd = np.zeros((3 * len(rrs), bins[-1] + 1))
    min_dist = np.inf
    ws2 = ws.reshape(-1, 1)
    for ri in range(len(rrs)):
        rr = rrs[ri]
        diff = rmags - rr
        dist2_ = np.sum(diff * diff, axis=1)
        dist2 = dist2_.reshape(-1, 1)
        dist = np.sqrt(dist2)
        min_dist = min(dist.min(), min_dist)
        if min_dist < _MIN_DIST_LIMIT and too_close == "raise":
            break
        t_ = np.sum(diff * cosmags, axis=1)
        t = t_.reshape(-1, 1)
        sum_ = ws2 * (3 * diff * t - dist2 * cosmags) / (dist2 * dist2 * dist)
        for ii in range(3):
            fwd[3 * ri + ii] = bincount(bins, sum_[:, ii], bins[-1] + 1)
    fwd *= _MAG_FACTOR
    return fwd, min_dist
