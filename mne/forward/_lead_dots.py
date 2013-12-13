import numpy as np

from ..utils import logger


def _next_legen_der(n, x, p0, p01, p0d, p0dd):
    """Compute the next Legendre polynomial and its derivatives"""
    if n > 1:
        help_ = p0
        helpd = p0d
        p0 = ((2 * n - 1) * x * help_ - (n - 1) * p01) / n
        p0d = n * help_ + x * helpd
        p0dd = (n + 1) * helpd + x * p0dd
        p01 = help_
    elif n == 0:
        p0 = 1.0
        p0d = 0.0
        p0dd = 0.0
    elif n == 1:
        p01 = 1.0
        p0 = x
        p0d = 1.0
        p0dd = 0.0
    return p0, p01, p0d, p0dd


def _next_legen(n, x, p0, p01):
    """Compute the next Legendre polynomials of the first kind"""
    if n > 1:
        p0 = ((2 * n - 1) * x * p0 - (n - 1) * p01) / n
    elif n == 0:
        p0 = 1.0
    elif n == 1:
        p01 = 1.0
        p0 = x
    return p0, p01


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
    betan = 1
    s0 = s1 = s2 = s3 = 0.0
    p0 = p01 = p0d = p0dd = 0
    for n in range(1, nterms + 1):
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
        s0 = s0 + (n + 1.0) * multn * p0
        s1 = s1 + multn * p0d
        s2 = s2 + mult * p0d
        s3 = s3 + mult * p0dd
    sums = np.array([beta * s0, beta * s1, beta * s2, beta * s3])
    return sums


def _comp_sum(beta, ctheta):
    """Lead field dot products using Legendre polynomial (P_n) series"""
    # Compute the sum occuring in the evaluation.
    # The result is
    #   sums[0]    (2n+1)^2/n beta^n P_n
    nterms = 100
    eps = 1e-10
    betan = 1
    s0 = 0.0
    p0 = p01 = 0.0
    for n in range(1, nterms + 1):
        betan = beta * betan
        if betan < eps:
            break
        p0, p01 = _next_legen(n, ctheta, p0, p01)
        s0 = s0 + p0 * betan * (2.0 * n + 1.0) * (2.0 * n + 1.0) / n
    return s0


def _sphere_dot_r0(r, rmag1, rmag2, cosmag1, cosmag2, r0, volume_integral):
    """Lead field dot product computation for MEG in sphere model"""
    rmag1 = rmag1 - r0
    rmag2 = rmag2 - r0
    r1 = np.sqrt(np.sum(rmag1 * rmag1))
    r2 = np.sqrt(np.sum(rmag2 * rmag2))
    beta = (r * r) / (r1 * r2)
    rr1 = rmag1 / r1
    rr2 = rmag2 / r2
    ct = np.sum(rr1 * rr2)
    sums = _comp_sums(beta, volume_integral, ct)

    # Accumulate the result, a little bit streamlined version
    n1c1 = np.sum(cosmag1 * rr1)
    n1c2 = np.sum(cosmag1 * rr2)
    n2c1 = np.sum(cosmag2 * rr1)
    n2c2 = np.sum(cosmag2 * rr2)
    n1n2 = np.sum(cosmag1 * cosmag2)
    part1 = ct * n1c1 * n2c2
    part2 = n1c1 * n2c1 + n1c2 * n2c2

    result = (n1c1 * n2c2 * sums[0] +
              (2.0 * part1 - part2) * sums[1] +
              (n1n2 + part1 - part2) * sums[2] +
              (n1c2 - ct * n1c1) * (n2c1 - ct * n2c2) * sums[3])

    # Give it a finishing touch!
    const = 4e-14 * np.pi  # This is \mu_0^2/4\pi
    result *= (const / (r1 * r2))
    if volume_integral:
        result *= r
    return result


def _eeg_sphere_dot_r0(r, rel1, rel2, r0):
    """Lead field dot product computation for EEG in the sphere model"""
    # This is a version that uses an explicit origin given in the call.
    rel1 = rel1 - r0
    rel2 = rel2 - r0
    r1 = np.sqrt(np.sum(rel1 * rel1))
    r2 = np.sqrt(np.sum(rel2 * rel2))
    beta = (r * r) / (r1 * r2)

    rr1 = rel1 / r1
    rr2 = rel2 / r2
    ct = np.sum(rr1 * rr2)
    # Give it a finishing touch!
    eeg_const = 1.0 / (4.0 * np.pi)
    return eeg_const * _comp_sum(beta, ct) / (r1 * r2)


def _do_self_dots(intrad, volume, coils, r0):
    """Perform the lead field dot product integrations"""
    nmag = len(coils)
    products = np.zeros((nmag, nmag))
    for ci1, coil1 in enumerate(coils):
        for ci2, coil2 in enumerate(coils[:ci1 + 1]):
            res = 0.0
            # XXX should be made much more efficient here
            for n1 in range(coil1['np']):
                for n2 in range(coil2['np']):
                    dd = _sphere_dot_r0(intrad, coil1['rmag'][n1],
                                        coil2['rmag'][n2],
                                        coil1['cosmag'][n1],
                                        coil2['cosmag'][n2],
                                        r0, volume)
                    res += (coil1['w'][n1] * coil2['w'][n2] * dd)
            products[ci1, ci2] = res
            products[ci2, ci1] = res
    return products


def _do_surf_map_dots(intrad, volume, coils, surf, sel, r0):
    """Compute the map construction products"""
    nmag = len(coils)
    products = np.zeros((len(sel), nmag))
    for si, ss in enumerate(sel):
        rvirt = surf['rr'][ss]
        this_nn = surf['nn'][ss]
        # XXX should be made more efficient here
        for ci, coil in enumerate(coils):
            res = 0
            for n in range(coil['np']):
                dd = _sphere_dot_r0(intrad, rvirt, coil['rmag'][n],
                                    this_nn, coil['cosmag'][n],
                                    r0, volume)
                res += coil['w'][n] * dd
            products[si][ci] = res
    return products


def _do_eeg_self_dots(intrad, volume, els, *r0):
    """Perform the lead field dot product integrations"""
    products = np.zeros((len(els), len(els)))
    for ei1, el1 in enumerate(els):
        for ei2, el2 in enumerate(els[:ei1 + 1]):
            res = 0.0
            for n1 in range(el1['np']):
                for n2 in range(el2['np']):
                    dd = _eeg_sphere_dot_r0(intrad, el1['rmag'][n1],
                                            el2['rmag'][n2], r0)
                    res += el1['w'][n1] * el2['w'][n2] * dd
            products[ei1, ei2] = res
            products[ei2, ei1] = res
    return products


def _do_eeg_surf_map_dots(intrad, volume, els, surf, sel, virt_ref, r0):
    """Compute the map construction products"""
    nel = len(els)
    products = np.zeros((len(sel), nel))
    if virt_ref:
        virt_ref_products = np.empty(nel)
        for p, el in enumerate(els):
            res = 0
            for n in range(el['np']):
                dd = _eeg_sphere_dot_r0(intrad, virt_ref, el['rmag'][n], r0)
                res += el['w'][n] * dd
            virt_ref_products[p] = res
        logger.info('Virtual reference included')

    for si, ss in enumerate(sel):
        # Go over all electrodes
        rvirt = surf['rr'][ss]
        for ei, el in enumerate(els):
            res = 0
            for n in range(el['np']):
                dd = _eeg_sphere_dot_r0(intrad, rvirt, el['rmag'][n], r0)
                res += el['w'][n] * dd
            if virt_ref:
                products[si, p] = res - virt_ref_products[p]
            else:
                products[si, p] = res
    return products
