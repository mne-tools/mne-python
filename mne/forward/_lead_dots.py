import numpy as np

from ..utils import logger


def _next_legen_der(n, x, p0, p01, p0d, p0dd):
    """Compute the next Legendre polynomial and its derivatives"""
    # only good for n > 1 !
    help_ = p0
    helpd = p0d
    p0 = ((2 * n - 1) * x * help_ - (n - 1) * p01) / n
    p0d = n * help_ + x * helpd
    p0dd = (n + 1) * helpd + x * p0dd
    p01 = help_
    return p0, p01, p0d, p0dd


def _next_legen(n, x, p0, p01):
    """Compute the next Legendre polynomials of the first kind"""
    # Only good for n > 1 !
    return ((2 * n - 1) * x * p0 - (n - 1) * p01) / n, p0


def _comp_sums(beta, volume_integral, ctheta):
    """Lead field dot products using Legendre polynomial (P_n) series"""
    # Compute the sums occuring in the evaluation.
    # Two point magnetometers on the xz plane are assumed.
    # The four sums are:
    #  * sums[0]    n(n+1)/(2n+1) beta^(n+1) P_n
    #  * sums[1]    n/(2n+1) beta^(n+1) P_n'
    #  * sums[2]    n/((2n+1)(n+1)) beta^(n+1) P_n'
    #  * sums[3]    n/((2n+1)(n+1)) beta^(n+1) P_n''
    nterms = 100  # max number to compute
    eps = 1e-10

    # do n=1 special case
    betan = beta
    p01 = 1.0
    p0 = ctheta
    p0d = 1.0
    p0dd = 0.0
    if volume_integral:
        multn = betan / 15.0
    else:
        multn = betan / 3.0
    mult = multn / 2.0
    s0 = 2.0 * multn * p0
    s1 = multn * p0d
    s2 = mult * p0d
    s3 = mult * p0dd
    # now do interative procedure
    for n in range(2, nterms + 1):
        betan = beta * betan
        if betan < eps:
            break
        p0, p01, p0d, p0dd = _next_legen_der(n, ctheta,
                                             p0, p01, p0d, p0dd)
        if volume_integral:
            multn = betan * n / ((2.0 * n + 1.0) * (2.0 * n + 3.0))
        else:
            multn = betan * n / (2.0 * n + 1.0)
        mult = multn / (n + 1.0)
        s0 += (n + 1.0) * multn * p0
        s1 += multn * p0d
        s2 += mult * p0d
        s3 += mult * p0dd
    sums = np.array([beta * s0, beta * s1, beta * s2, beta * s3])
    return sums


def _comp_sum(beta, ctheta):
    """Lead field dot products using Legendre polynomial (P_n) series"""
    # Compute the sum occuring in the evaluation.
    # The result is
    #   sums[0]    (2n+1)^2/n beta^n P_n
    nterms = 100
    eps = 1e-10

    # do n == 1 case first, since it is exceptional
    betan = beta
    p0 = ctheta
    p01 = 1.0
    s0 = p0 * betan * 9.0
    # now do the rest by recursion
    for n in range(2, nterms + 1):
        betan *= beta
        if betan < eps:
            break
        p0, p01 = _next_legen(n, ctheta, p0, p01)
        s0 += p0 * betan * (2.0 * n + 1.0) * (2.0 * n + 1.0) / n
    return s0


def _fast_sphere_dot_r0(r, rmags1, rmags2, cosmags1, cosmags2, ws1, ws2, r0,
                        volume_integral):
    """Lead field dot product computation for MEG in sphere model"""
    rmags1 = rmags1 - r0[np.newaxis, :]
    rmags2 = rmags2 - r0[np.newaxis, :]
    r1 = np.sqrt(np.sum(rmags1 * rmags1, axis=1))
    r2 = np.sqrt(np.sum(rmags2 * rmags2, axis=1))
    rr1 = rmags1 / r1[:, np.newaxis]
    rr2 = rmags2 / r2[:, np.newaxis]

    # expand axes for vectorized computation
    rr1 = rr1[:, np.newaxis, :]  # (n_rr1, n_rr2, n_coord) e.g. 4x4x3
    rr2 = rr2[np.newaxis, :, :]
    ct = np.sum(rr1 * rr2, axis=2)
    cosmags1 = cosmags1[:, np.newaxis, :]
    cosmags2 = cosmags2[np.newaxis, :, :]
    r1r2 = r1[:, np.newaxis] * r2[np.newaxis, :]

    # now we have to do every pairwise one
    beta = (r * r) / r1r2
    sums = np.array([[_comp_sums(bbb, volume_integral, ccc)
                      for bbb, ccc in zip(bb, cc)]
                     for bb, cc in zip(beta, ct)])

    # Accumulate the result, a little bit streamlined version
    n1c1 = np.sum(cosmags1 * rr1, axis=2)
    n1c2 = np.sum(cosmags1 * rr2, axis=2)
    n2c1 = np.sum(cosmags2 * rr1, axis=2)
    n2c2 = np.sum(cosmags2 * rr2, axis=2)
    n1n2 = np.sum(cosmags1 * cosmags2, axis=2)
    part1 = ct * n1c1 * n2c2
    part2 = n1c1 * n2c1 + n1c2 * n2c2

    result = (n1c1 * n2c2 * sums[:, :, 0] +
              (2.0 * part1 - part2) * sums[:, :, 1] +
              (n1n2 + part1 - part2) * sums[:, :, 2] +
              (n1c2 - ct * n1c1) * (n2c1 - ct * n2c2) * sums[:, :, 3])

    # Give it a finishing touch!
    const = 4e-14 * np.pi  # This is \mu_0^2/4\pi
    result *= (const / r1r2)
    if volume_integral:
        result *= r
    # new we add them all up with weights
    result = np.sum(ws1[:, np.newaxis] * ws2[np.newaxis, :] * result)
    return result


def _fast_eeg_sphere_dot_r0(r, rel1, rel2, w1, w2, r0):
    """Lead field dot product computation for EEG in the sphere model"""
    # This is a version that uses an explicit origin given in the call.
    rel1 = rel1 - r0
    rel2 = rel2 - r0
    r1 = np.sqrt(np.sum(rel1 * rel1, axis=1))
    r2 = np.sqrt(np.sum(rel2 * rel2, axis=1))
    rr1 = rel1 / r1[:, np.newaxis]
    rr2 = rel2 / r2[:, np.newaxis]

    # expand axes
    r1 = r1[:, np.newaxis]
    r2 = r2[np.newaxis, :]
    rr1 = rr1[:, np.newaxis, :]
    rr2 = rr2[np.newaxis, :, :]

    ct = np.sum(rr1 * rr2, axis=2)
    beta = (r * r) / (r1 * r2)
    sums = np.array([[_comp_sum(bbb, ccc) for bbb, ccc in zip(bb, cc)]
                     for bb, cc in zip(beta, ct)])
    # Give it a finishing touch!
    eeg_const = 1.0 / (4.0 * np.pi)
    result = eeg_const * sums / (r1 * r2)
    result = np.sum(w1[:, np.newaxis] * w2[np.newaxis, :] * result)
    return result


def _do_self_dots(intrad, volume, coils, r0, ctype):
    """Perform the lead field dot product integrations"""
    if ctype == 'eeg':
        intrad *= 0.7
    ncoil = len(coils)
    products = np.empty((ncoil, ncoil))
    for ci1, c1 in enumerate(coils):
        for ci2, c2 in enumerate(coils[:ci1 + 1]):
            if ctype == 'meg':
                res = _fast_sphere_dot_r0(intrad, c1['rmag'], c2['rmag'],
                                          c1['cosmag'], c2['cosmag'],
                                          c1['w'], c2['w'], r0, volume)
            else:  # 'eeg'
                res = _fast_eeg_sphere_dot_r0(intrad, c1['rmag'], c2['rmag'],
                                              c1['w'], c2['w'], r0)
            products[ci1, ci2] = res
            products[ci2, ci1] = res
    return products


def _do_surf_map_dots(intrad, volume, coils, surf, sel, r0, ctype):
    """Compute the map construction products"""
    virt_ref = False
    ncoil = len(coils)
    products = np.zeros((len(sel), ncoil))
    if ctype == 'eeg':
        intrad *= 0.7
        if virt_ref:
            virt_ref_products = np.empty(ncoil)
            vr = virt_ref[np.newaxis, :]
            this_w = np.array([1.])
            for ei, el in enumerate(coils):
                res = _fast_eeg_sphere_dot_r0(intrad, vr, el['rmag'],
                                              this_w, el['w'], r0)
                virt_ref_products[ei] = res
            logger.info('Virtual reference included')

    this_w = np.array([1.])
    for si, ss in enumerate(sel):
        rvirt = surf['rr'][ss][np.newaxis, :]
        this_nn = surf['nn'][ss][np.newaxis, :]
        if ctype == 'meg':
            for ci, coil in enumerate(coils):
                res = _fast_sphere_dot_r0(intrad, rvirt, coil['rmag'],
                                          this_nn, coil['cosmag'],
                                          this_w, coil['w'], r0, volume)
                products[si, ci] = res
        else:  # 'eeg'
            for ei, el in enumerate(coils):
                res = _fast_eeg_sphere_dot_r0(intrad, rvirt, el['rmag'],
                                              this_w, el['w'], r0)
                if virt_ref:
                    products[si, ei] = res - virt_ref_products[ei]
                else:
                    products[si, ei] = res
    return products


