"""Numba-accelerated helpers for :mod:`mne.surface`.

Kept in its own module so that ``import mne.surface`` -- which is on the import path
of ``mne.channels`` and ``mne.io`` -- does not import numba. These have to live
together because numba can only call other jitted functions in nopython mode.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ._numba import bincount, has_numba, jit, prange


@jit()
def _fast_cross_nd_sum(a, b, c):
    """Fast cross and sum."""
    return (
        (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]) * c[..., 0]
        + (a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]) * c[..., 1]
        + (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) * c[..., 2]
    )


@jit()
def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit()
def _accumulate_normals(tris, tri_nn, npts):
    """Efficiently accumulate triangle normals."""
    # this code replaces the following, but is faster (vectorized):
    #
    # this['nn'] = np.zeros((this['np'], 3))
    # for p in xrange(this['ntri']):
    #     verts = this['tris'][p]
    #     this['nn'][verts, :] += this['tri_nn'][p, :]
    #
    nn = np.zeros((npts, 3))
    for vi in range(3):
        verts = tris[:, vi]
        for idx in range(3):  # x, y, z
            nn[:, idx] += bincount(verts, weights=tri_nn[:, idx], minlength=npts)
    return nn


@jit()
def _decimate_spacing(indptr, indices, spacing):  # pragma: no cover
    """Find the vertices to use for an integer source space spacing."""
    n_vert = len(indptr) - 1
    d = np.full(n_vert, 10000, np.int64)
    # A mysterious algorithm follows
    for k in range(n_vert):
        neigh = indices[indptr[k] : indptr[k + 1]]
        d[k] = min(np.min(d[neigh]) + 1, d[k])
        if d[k] >= spacing:
            d[k] = 0
        d[neigh] = np.minimum(d[neigh], d[k] + 1)
    if spacing == 2:
        for k in range(n_vert - 1, -1, -1):
            for n in indices[indptr[k] : indptr[k + 1]]:
                d[k] = min(d[k], d[n] + 1)
                d[n] = min(d[n], d[k] + 1)
        for k in range(n_vert):
            if d[k] > 0:
                neigh = indices[indptr[k] : indptr[k + 1]]
                if np.sum(d[neigh] == 0) <= 2:
                    d[k] = 0
                d[neigh] = np.minimum(d[neigh], d[k] + 1)
    return d == 0


@jit()
def _triangle_coords(r, best, r1, nn, r12, r13, a, b, c):  # pragma: no cover
    """Get coordinates of a vertex projected to a triangle."""
    r1 = r1[best]
    tri_nn = nn[best]
    r12 = r12[best]
    r13 = r13[best]
    a = a[best]
    b = b[best]
    c = c[best]
    rr = r - r1
    z = np.sum(rr * tri_nn)
    v1 = np.sum(rr * r12)
    v2 = np.sum(rr * r13)
    det = a * b - c * c
    x = (b * v1 - c * v2) / det
    y = (a * v2 - c * v1) / det
    return x, y, z


@jit()
def _get_tri_dist(p, q, p0, q0, a, b, c, dist):  # pragma: no cover
    """Get the distance to a triangle edge."""
    p1 = p - p0
    q1 = q - q0
    out = p1 * p1 * a
    out += q1 * q1 * b
    out += p1 * q1 * c
    out += dist * dist
    return np.sqrt(out)


@jit(parallel=True)
def _find_nearest_tri_pts(
    rrs,
    pt_triss,
    pt_lens,
    a,
    b,
    c,
    nn,
    r1,
    r12,
    r13,
    r1213,
    mat,
    run_all=True,
    reproject=False,
):  # pragma: no cover
    """Find nearest point mapping to a set of triangles.

    If run_all is False, if the point lies within a triangle, it stops.
    If run_all is True, edges of other triangles are checked in case
    those (somehow) are closer.
    """
    # The following dense code is equivalent to the following:
    #   rr = r1[pt_tris] - to_pts[ii]
    #   v1s = np.sum(rr * r12[pt_tris], axis=1)
    #   v2s = np.sum(rr * r13[pt_tris], axis=1)
    #   aas = a[pt_tris]
    #   bbs = b[pt_tris]
    #   ccs = c[pt_tris]
    #   dets = aas * bbs - ccs * ccs
    #   pp = (bbs * v1s - ccs * v2s) / dets
    #   qq = (aas * v2s - ccs * v1s) / dets
    #   pqs = np.array(pp, qq)

    weights = np.empty((len(rrs), 3))
    tri_idx = np.empty(len(rrs), np.int64)
    for ri in prange(len(rrs)):
        rr = np.reshape(rrs[ri], (1, 3))
        start, stop = pt_lens[ri : ri + 2]
        if start == stop == 0:  # use all
            drs = rr - r1
            tri_nn = nn
            mats = mat
            r1213s = r1213
            reindex = False
        else:
            pt_tris = pt_triss[start:stop]
            drs = rr - r1[pt_tris]
            tri_nn = nn[pt_tris]
            mats = mat[pt_tris]
            r1213s = r1213[pt_tris]
            reindex = True
        use = np.ones(len(drs), np.int64)
        pqs = np.empty((len(drs), 2))
        dists = np.empty(len(drs))
        dist = np.inf
        # make life easier for numba var typing
        p, q, pt = np.float64(0.0), np.float64(1.0), np.int64(0)
        found = False
        for ii in range(len(drs)):
            # Inlined as scalar arithmetic rather than np.dot on the tiny
            # (2, 3)/(2, 2)/(3,) arrays: much faster under numba, which
            # doesn't optimize away the overhead of these tiny matrix ops.
            dr0, dr1, dr2 = drs[ii, 0], drs[ii, 1], drs[ii, 2]
            v1 = (
                r1213s[ii, 0, 0] * dr0 + r1213s[ii, 0, 1] * dr1 + r1213s[ii, 0, 2] * dr2
            )
            v2 = (
                r1213s[ii, 1, 0] * dr0 + r1213s[ii, 1, 1] * dr1 + r1213s[ii, 1, 2] * dr2
            )
            pp = mats[ii, 0, 0] * v1 + mats[ii, 0, 1] * v2
            qq = mats[ii, 1, 0] * v1 + mats[ii, 1, 1] * v2
            pqs[ii, 0], pqs[ii, 1] = pp, qq
            dists[ii] = dr0 * tri_nn[ii, 0] + dr1 * tri_nn[ii, 1] + dr2 * tri_nn[ii, 2]
            if pp >= 0 and qq >= 0 and pp <= 1 and qq <= 1 and pp + qq < 1:
                found = True
                use[ii] = False
                if np.abs(dists[ii]) < np.abs(dist):
                    p, q, pt, dist = pp, qq, ii, dists[ii]
        # re-reference back to original numbers
        if found and reindex:
            pt = pt_tris[pt]

        if not found or run_all:
            # don't include ones that we might have found before
            # these are the ones that we want to check the sides of
            s = np.where(use)[0]
            # Tough: must investigate the sides
            if reindex:
                use_pt_tris = pt_tris[s].astype(np.int64)
            else:
                use_pt_tris = s.astype(np.int64)
            pp, qq, ptt, distt = _nearest_tri_edge(
                use_pt_tris, pqs[s], dists[s], a, b, c
            )
            if np.abs(distt) < np.abs(dist):
                p, q, pt, dist = pp, qq, ptt, distt
        w = (1 - p - q, p, q)
        if reproject:
            # Calculate a linear interpolation between the vertex values to
            # get coords of pt projected onto closest triangle
            coords = _triangle_coords(rr[0], pt, r1, nn, r12, r13, a, b, c)
            w = (1.0 - coords[0] - coords[1], coords[0], coords[1])
        weights[ri] = w
        tri_idx[ri] = pt
    return weights, tri_idx


@jit()
def _nearest_tri_edge(pt_tris, pqs, dist, a, b, c):  # pragma: no cover
    """Get nearest location from a point to the edge of a set of triangles."""
    # We might do something intelligent here. However, for now
    # it is ok to do it in the hard way
    aa = a[pt_tris]
    bb = b[pt_tris]
    cc = c[pt_tris]
    pp = pqs[:, 0]
    qq = pqs[:, 1]
    # Find the nearest point from a triangle:
    #   Side 1 -> 2
    p0 = np.minimum(np.maximum(pp + 0.5 * (qq * cc) / aa, 0.0), 1.0)
    q0 = np.zeros_like(p0)
    #   Side 2 -> 3
    t1 = 0.5 * ((2.0 * aa - cc) * (1.0 - pp) + (2.0 * bb - cc) * qq) / (aa + bb - cc)
    t1 = np.minimum(np.maximum(t1, 0.0), 1.0)
    p1 = 1.0 - t1
    q1 = t1
    #   Side 1 -> 3
    q2 = np.minimum(np.maximum(qq + 0.5 * (pp * cc) / bb, 0.0), 1.0)
    p2 = np.zeros_like(q2)

    # figure out which one had the lowest distance
    dist0 = _get_tri_dist(pp, qq, p0, q0, aa, bb, cc, dist)
    dist1 = _get_tri_dist(pp, qq, p1, q1, aa, bb, cc, dist)
    dist2 = _get_tri_dist(pp, qq, p2, q2, aa, bb, cc, dist)
    pp = np.concatenate((p0, p1, p2))
    qq = np.concatenate((q0, q1, q2))
    dists = np.concatenate((dist0, dist1, dist2))
    ii = np.argmin(np.abs(dists))
    p, q, pt, dist = pp[ii], qq[ii], pt_tris[ii % len(pt_tris)], dists[ii]
    return p, q, pt, dist


if has_numba:

    @jit(parallel=True)
    def _get_solids(tri_rrs, fros):
        """Compute _sum_solids_div total angle in chunks.

        Written as an explicit double loop (points outer, triangles inner)
        with scalar arithmetic rather than per-triangle array ops over all
        points (see ``_get_solids_numpy``): inside a numba nopython function,
        the array form allocates and walks several full (n_points, 3)-shaped
        temporaries per triangle, which is much slower than keeping a
        point's running total in a scalar register. This is ~5x faster than
        ``_get_solids_numpy`` but, lacking numba's JIT, ~5x slower if run
        as plain Python, hence the two separate implementations.
        """
        n_tris = len(tri_rrs)
        tot_angle = np.zeros(len(fros))
        for pi in prange(len(fros)):
            fx, fy, fz = fros[pi, 0], fros[pi, 1], fros[pi, 2]
            angle = 0.0
            for ti in range(n_tris):
                tri_rr = tri_rrs[ti]
                v1x, v1y, v1z = fx - tri_rr[0, 0], fy - tri_rr[0, 1], fz - tri_rr[0, 2]
                v2x, v2y, v2z = fx - tri_rr[1, 0], fy - tri_rr[1, 1], fz - tri_rr[1, 2]
                v3x, v3y, v3z = fx - tri_rr[2, 0], fy - tri_rr[2, 1], fz - tri_rr[2, 2]
                # v4 = cross(v1, v2); triple = dot(v4, v3)
                triple = (
                    (v1y * v2z - v1z * v2y) * v3x
                    + (v1z * v2x - v1x * v2z) * v3y
                    + (v1x * v2y - v1y * v2x) * v3z
                )
                l1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
                l2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
                l3 = np.sqrt(v3x * v3x + v3y * v3y + v3z * v3z)
                s = (
                    l1 * l2 * l3
                    + (v1x * v2x + v1y * v2y + v1z * v2z) * l3
                    + (v1x * v3x + v1y * v3y + v1z * v3z) * l2
                    + (v2x * v3x + v2y * v3y + v2z * v3z) * l1
                )
                angle -= np.arctan2(triple, s)
            tot_angle[pi] = angle
        return tot_angle

    @jit(parallel=True)
    def _lin_pot_coeff_sub(fros, tri_rrs, tri_nn, tri_area, tris, same, submat):
        """Subtract linear potential coefficients of all triangles from submat.

        Same math as ``bem._lin_pot_coeff``, but as a scalar loop (points outer so
        each thread owns rows of ``submat``) for the same reasons as ``_get_solids``.
        """
        edges = tri_rrs[:, np.array([1, 2, 0])] - tri_rrs
        sizes = np.sqrt(np.sum(edges * edges, axis=-1))
        # not in-place: numba's broadcasted /= gave NaNs here
        edges = edges / sizes.reshape(-1, 3, 1)
        for pi in prange(len(fros)):
            v = np.empty((3, 3))
            ll = np.empty(3)
            beta = np.empty(3)
            vec_omega = np.empty(3)
            for ti in range(len(tris)):
                tri = tris[ti]
                # No contribution from a triangle that this vertex belongs to
                if same and (tri[0] == pi or tri[1] == pi or tri[2] == pi):
                    continue
                for k in range(3):
                    for c in range(3):
                        v[k, c] = tri_rrs[ti, k, c] - fros[pi, c]
                    ll[k] = np.sqrt(_dot3(v[k], v[k]))
                triple = _fast_cross_nd_sum(v[0], v[1], v[2])
                ss = ll[0] * ll[1] * ll[2]
                for k in range(3):
                    ss += _dot3(v[(k + 1) % 3], v[(k + 2) % 3]) * ll[k]
                solid = np.arctan2(triple, ss)
                if abs(solid) < np.pi / 1e6:
                    continue
                for k in range(3):  # _calc_beta
                    k1 = (k + 1) % 3
                    num = ll[k] + _dot3(v[k], edges[ti, k])
                    den = ll[k1] + _dot3(v[k1], edges[ti, k])
                    beta[k] = np.log(num / den) / sizes[ti, k]
                for c in range(3):
                    vec_omega[c] = 0.0
                    for k in range(3):
                        vec_omega[c] += (beta[k - 1] - beta[k]) * v[k, c]
                area2 = 2.0 * tri_area[ti]
                for k in range(3):
                    k1, k2 = (k + 1) % 3, (k + 2) % 3
                    zdots = _fast_cross_nd_sum(v[k1], v[k2], tri_nn[ti])
                    omega = 2.0 * area2 * solid * zdots
                    for c in range(3):
                        omega -= triple * (v[k2, c] - v[k1, c]) * vec_omega[c]
                    submat[pi, tri[k]] += omega / (area2 * area2)

else:  # pragma: no cover
    from .bem import _lin_pot_coeff_sub_numpy
    from .surface import _get_solids_numpy

    _get_solids = _get_solids_numpy
    _lin_pot_coeff_sub = _lin_pot_coeff_sub_numpy
