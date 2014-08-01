import os
from os import path as op

import numpy as np
from numpy.polynomial import legendre

from ..parallel import parallel_func
from ..utils import logger, _get_extra_data_path


##############################################################################
# FAST LEGENDRE (DERIVATIVE) POLYNOMIALS USING LOOKUP TABLE

def _next_legen_der(n, x, p0, p01, p0d, p0dd):
    """Compute the next Legendre polynomial and its derivatives"""
    # only good for n > 1 !
    help_ = p0
    helpd = p0d
    p0 = ((2 * n - 1) * x * help_ - (n - 1) * p01) / n
    p0d = n * help_ + x * helpd
    p0dd = (n + 1) * helpd + x * p0dd
    p01 = help_
    return p0, p0d, p0dd


def _get_legen(x, n_coeff=100):
    """Get Legendre polynomials expanded about x"""
    return legendre.legvander(x, n_coeff - 1)


def _get_legen_der(xx, n_coeff=100):
    """Get Legendre polynomial derivatives expanded about x"""
    coeffs = np.empty((len(xx), n_coeff, 3))
    for c, x in zip(coeffs, xx):
        p0s, p0ds, p0dds = c[:, 0], c[:, 1], c[:, 2]
        p0s[:2] = [1.0, x]
        p0ds[:2] = [0.0, 1.0]
        p0dds[:2] = [0.0, 0.0]
        for n in range(2, n_coeff):
            p0s[n], p0ds[n], p0dds[n] = _next_legen_der(n, x, p0s[n - 1],
                                            p0s[n - 2], p0ds[n - 1],
                                            p0dds[n - 1])
    return coeffs


def _get_legen_table(ch_type, volume_integral=False, n_coeff=100,
                     n_interp=20000, force_calc=False):
    """Return a (generated) LUT of Legendre (derivative) polynomial coeffs"""
    if n_interp % 2 != 0:
        raise RuntimeError('n_interp must be even')
    fname = op.join(_get_extra_data_path(), 'tables')
    if not op.isdir(fname):
        # Updated due to API chang (GH 1167)
        os.makedirs(fname)
    if ch_type == 'meg':
        fname = op.join(fname, 'legder_%s_%s.bin' % (n_coeff, n_interp))
        leg_fun = _get_legen_der
        extra_str = ' derivative'
        lut_shape = (n_interp + 1, n_coeff, 3)
    else:  # 'eeg'
        fname = op.join(fname, 'legval_%s_%s.bin' % (n_coeff, n_interp))
        leg_fun = _get_legen
        extra_str = ''
        lut_shape = (n_interp + 1, n_coeff)
    if not op.isfile(fname) or force_calc:
        n_out = (n_interp // 2)
        logger.info('Generating Legendre%s table...' % extra_str)
        x_interp = np.arange(-n_out, n_out + 1, dtype=np.float64) / n_out
        lut = leg_fun(x_interp, n_coeff).astype(np.float32)
        if not force_calc:
            with open(fname, 'wb') as fid:
                fid.write(lut.tostring())
    else:
        logger.info('Reading Legendre%s table...' % extra_str)
        with open(fname, 'rb', buffering=0) as fid:
            lut = np.fromfile(fid, np.float32)
    lut.shape = lut_shape

    # we need this for the integration step
    n_fact = np.arange(1, n_coeff, dtype=float)
    if ch_type == 'meg':
        n_facts = list()  # multn, then mult, then multn * (n + 1)
        if volume_integral:
            n_facts.append(n_fact / ((2.0 * n_fact + 1.0)
                                     * (2.0 * n_fact + 3.0)))
        else:
            n_facts.append(n_fact / (2.0 * n_fact + 1.0))
        n_facts.append(n_facts[0] / (n_fact + 1.0))
        n_facts.append(n_facts[0] * (n_fact + 1.0))
        # skip the first set of coefficients because they are not used
        lut = lut[:, 1:, [0, 1, 1, 2]]  # for multiplicative convenience later
        # reshape this for convenience, too
        n_facts = np.array(n_facts)[[2, 0, 1, 1], :].T
        n_facts = np.ascontiguousarray(n_facts)
        n_fact = n_facts
    else:  # 'eeg'
        n_fact = (2.0 * n_fact + 1.0) * (2.0 * n_fact + 1.0) / n_fact
        # skip the first set of coefficients because they are not used
        lut = lut[:, 1:].copy()
    return lut, n_fact


def _get_legen_lut_fast(x, lut):
    """Return Legendre coefficients for given x values in -1<=x<=1"""
    # map into table vals (works for both vals and deriv tables)
    n_interp = (lut.shape[0] - 1.0)
    # equiv to "(x + 1.0) / 2.0) * n_interp" but faster
    mm = x * (n_interp / 2.0) + 0.5 * n_interp
    # nearest-neighbor version (could be decent enough...)
    idx = np.round(mm).astype(int)
    vals = lut[idx]
    return vals


def _get_legen_lut_accurate(x, lut):
    """Return Legendre coefficients for given x values in -1<=x<=1"""
    # map into table vals (works for both vals and deriv tables)
    n_interp = (lut.shape[0] - 1.0)
    # equiv to "(x + 1.0) / 2.0) * n_interp" but faster
    mm = x * (n_interp / 2.0) + 0.5 * n_interp
    # slower, more accurate interpolation version
    mm = np.minimum(mm, n_interp - 0.0000000001)
    idx = np.floor(mm).astype(int)
    w2 = mm - idx
    w2.shape += tuple([1] * (lut.ndim - w2.ndim))  # expand to correct size
    vals = (1 - w2) * lut[idx] + w2 * lut[idx + 1]
    return vals


def _comp_sum_eeg(beta, ctheta, lut_fun, n_fact):
    """Lead field dot products using Legendre polynomial (P_n) series"""
    # Compute the sum occurring in the evaluation.
    # The result is
    #   sums[:]    (2n+1)^2/n beta^n P_n
    coeffs = lut_fun(ctheta)
    betans = np.cumprod(np.tile(beta[:, np.newaxis], (1, n_fact.shape[0])),
                        axis=1)
    s0 = np.dot(coeffs * betans, n_fact)  # == weighted sum across cols
    return s0


def _comp_sums_meg(beta, ctheta, lut_fun, n_fact, volume_integral):
    """Lead field dot products using Legendre polynomial (P_n) series"""
    # Compute the sums occurring in the evaluation.
    # Two point magnetometers on the xz plane are assumed.
    # The four sums are:
    #  * sums[:, 0]    n(n+1)/(2n+1) beta^(n+1) P_n
    #  * sums[:, 1]    n/(2n+1) beta^(n+1) P_n'
    #  * sums[:, 2]    n/((2n+1)(n+1)) beta^(n+1) P_n'
    #  * sums[:, 3]    n/((2n+1)(n+1)) beta^(n+1) P_n''
    coeffs = lut_fun(ctheta)
    beta = (np.cumprod(np.tile(beta[:, np.newaxis], (1, n_fact.shape[0])),
                       axis=1) * beta[:, np.newaxis])
    # This is equivalent, but slower:
    # sums = np.sum(beta[:, :, np.newaxis] * n_fact * coeffs, axis=1)
    # sums = np.rollaxis(sums, 2)
    sums = np.einsum('ij,jk,ijk->ki', beta, n_fact, coeffs)
    return sums


###############################################################################
# SPHERE DOTS

def _fast_sphere_dot_r0(r, rr1, rr2, lr1, lr2, cosmags1, cosmags2,
                        w1, w2, volume_integral, lut, n_fact, ch_type):
    """Lead field dot product computation for M/EEG in the sphere model"""
    ct = np.einsum('ik,jk->ij', rr1, rr2)  # outer product, sum over coords

    # expand axes
    rr1 = rr1[:, np.newaxis, :]  # (n_rr1, n_rr2, n_coord) e.g. 4x4x3
    rr2 = rr2[np.newaxis, :, :]
    lr1lr2 = lr1[:, np.newaxis] * lr2[np.newaxis, :]

    beta = (r * r) / lr1lr2
    if ch_type == 'meg':
        sums = _comp_sums_meg(beta.flatten(), ct.flatten(), lut, n_fact,
                              volume_integral)
        sums.shape = (4,) + beta.shape

        # Accumulate the result, a little bit streamlined version
        #cosmags1 = cosmags1[:, np.newaxis, :]
        #cosmags2 = cosmags2[np.newaxis, :, :]
        #n1c1 = np.sum(cosmags1 * rr1, axis=2)
        #n1c2 = np.sum(cosmags1 * rr2, axis=2)
        #n2c1 = np.sum(cosmags2 * rr1, axis=2)
        #n2c2 = np.sum(cosmags2 * rr2, axis=2)
        #n1n2 = np.sum(cosmags1 * cosmags2, axis=2)
        n1c1 = np.einsum('ik,ijk->ij', cosmags1, rr1)
        n1c2 = np.einsum('ik,ijk->ij', cosmags1, rr2)
        n2c1 = np.einsum('jk,ijk->ij', cosmags2, rr1)
        n2c2 = np.einsum('jk,ijk->ij', cosmags2, rr2)
        n1n2 = np.einsum('ik,jk->ij', cosmags1, cosmags2)
        part1 = ct * n1c1 * n2c2
        part2 = n1c1 * n2c1 + n1c2 * n2c2

        result = (n1c1 * n2c2 * sums[0] +
                  (2.0 * part1 - part2) * sums[1] +
                  (n1n2 + part1 - part2) * sums[2] +
                  (n1c2 - ct * n1c1) * (n2c1 - ct * n2c2) * sums[3])

        # Give it a finishing touch!
        const = 4e-14 * np.pi  # This is \mu_0^2/4\pi
        result *= (const / lr1lr2)
        if volume_integral:
            result *= r
    else:  # 'eeg'
        sums = _comp_sum_eeg(beta.flatten(), ct.flatten(), lut, n_fact)
        sums.shape = beta.shape

        # Give it a finishing touch!
        eeg_const = 1.0 / (4.0 * np.pi)
        result = eeg_const * sums / lr1lr2
    # new we add them all up with weights
    if w1 is None:  # operating on surface, treat independently
        #result = np.sum(w2[np.newaxis, :] * result, axis=1)
        result = np.dot(result, w2)
    else:
        #result = np.sum((w1[:, np.newaxis] * w2[np.newaxis, :]) * result)
        result = np.einsum('i,j,ij', w1, w2, result)
    return result


def _do_self_dots(intrad, volume, coils, r0, ch_type, lut, n_fact, n_jobs):
    """Perform the lead field dot product integrations"""
    if ch_type == 'eeg':
        intrad *= 0.7
    # convert to normalized distances from expansion center
    rmags = [coil['rmag'] - r0[np.newaxis, :] for coil in coils]
    rlens = [np.sqrt(np.sum(r * r, axis=1)) for r in rmags]
    rmags = [r / rl[:, np.newaxis] for r, rl in zip(rmags, rlens)]
    cosmags = [coil['cosmag'] for coil in coils]
    ws = [coil['w'] for coil in coils]
    parallel, p_fun, _ = parallel_func(_do_self_dots_subset, n_jobs)
    prods = parallel(p_fun(intrad, rmags, rlens, cosmags,
                           ws, volume, lut, n_fact, ch_type, idx)
                     for idx in np.array_split(np.arange(len(rmags)), n_jobs))
    products = np.sum(prods, axis=0)
    return products


def _do_self_dots_subset(intrad, rmags, rlens, cosmags, ws, volume, lut,
                         n_fact, ch_type, idx):
    """Helper for parallelization"""
    products = np.zeros((len(rmags), len(rmags)))
    for ci1 in idx:
        for ci2 in range(0, ci1 + 1):
            res = _fast_sphere_dot_r0(intrad, rmags[ci1], rmags[ci2],
                                      rlens[ci1], rlens[ci2],
                                      cosmags[ci1], cosmags[ci2],
                                      ws[ci1], ws[ci2], volume, lut,
                                      n_fact, ch_type)
            products[ci1, ci2] = res
            products[ci2, ci1] = res
    return products


def _do_surface_dots(intrad, volume, coils, surf, sel, r0, ch_type,
                     lut, n_fact, n_jobs):
    """Compute the map construction products"""
    virt_ref = False
    # convert to normalized distances from expansion center
    rmags = [coil['rmag'] - r0[np.newaxis, :] for coil in coils]
    rlens = [np.sqrt(np.sum(r * r, axis=1)) for r in rmags]
    rmags = [r / rl[:, np.newaxis] for r, rl in zip(rmags, rlens)]
    cosmags = [coil['cosmag'] for coil in coils]
    ws = [coil['w'] for coil in coils]
    rref = None
    refl = None
    if ch_type == 'eeg':
        intrad *= 0.7
        if virt_ref:
            rref = virt_ref[np.newaxis, :] - r0[np.newaxis, :]
            refl = np.sqrt(np.sum(rref * rref, axis=1))
            rref /= refl[:, np.newaxis]

    rsurf = surf['rr'][sel] - r0[np.newaxis, :]
    lsurf = np.sqrt(np.sum(rsurf * rsurf, axis=1))
    rsurf /= lsurf[:, np.newaxis]
    this_nn = surf['nn'][sel]

    parallel, p_fun, _ = parallel_func(_do_surface_dots_subset, n_jobs)
    prods = parallel(p_fun(intrad, rsurf, rmags, rref, refl, lsurf, rlens,
                           this_nn, cosmags, ws, volume, lut, n_fact, ch_type,
                           idx)
                     for idx in np.array_split(np.arange(len(rmags)), n_jobs))
    products = np.sum(prods, axis=0)
    return products


def _do_surface_dots_subset(intrad, rsurf, rmags, rref, refl, lsurf, rlens,
                            this_nn, cosmags, ws, volume, lut, n_fact, ch_type,
                            idx):
    """Helper for parallelization"""
    products = np.zeros((len(rsurf), len(rmags)))
    for ci in idx:
        res = _fast_sphere_dot_r0(intrad, rsurf, rmags[ci],
                                  lsurf, rlens[ci],
                                  this_nn, cosmags[ci],
                                  None, ws[ci], volume, lut,
                                  n_fact, ch_type)
        if rref is not None:
            vres = _fast_sphere_dot_r0(intrad, rref, rmags[ci],
                                       refl, rlens[ci],
                                       None, ws[ci], volume,
                                       lut, n_fact, ch_type)
            products[:, ci] = res - vres
        else:
            products[:, ci] = res
    return products
