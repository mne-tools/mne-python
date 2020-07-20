# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.

import os
import os.path as op

import numpy as np
from numpy.polynomial import legendre

from ..fixes import einsum
from ..parallel import parallel_func
from ..utils import logger, verbose, _get_extra_data_path, fill_doc


##############################################################################
# FAST LEGENDRE (DERIVATIVE) POLYNOMIALS USING LOOKUP TABLE

def _next_legen_der(n, x, p0, p01, p0d, p0dd):
    """Compute the next Legendre polynomial and its derivatives."""
    # only good for n > 1 !
    old_p0 = p0
    old_p0d = p0d
    p0 = ((2 * n - 1) * x * old_p0 - (n - 1) * p01) / n
    p0d = n * old_p0 + x * old_p0d
    p0dd = (n + 1) * old_p0d + x * p0dd
    return p0, p0d, p0dd


def _get_legen(x, n_coeff=100):
    """Get Legendre polynomials expanded about x."""
    return legendre.legvander(x, n_coeff - 1)


def _get_legen_der(xx, n_coeff=100):
    """Get Legendre polynomial derivatives expanded about x."""
    coeffs = np.empty((len(xx), n_coeff, 3))
    for c, x in zip(coeffs, xx):
        p0s, p0ds, p0dds = c[:, 0], c[:, 1], c[:, 2]
        p0s[:2] = [1.0, x]
        p0ds[:2] = [0.0, 1.0]
        p0dds[:2] = [0.0, 0.0]
        for n in range(2, n_coeff):
            p0s[n], p0ds[n], p0dds[n] = _next_legen_der(
                n, x, p0s[n - 1], p0s[n - 2], p0ds[n - 1], p0dds[n - 1])
    return coeffs


@verbose
def _get_legen_table(ch_type, volume_integral=False, n_coeff=100,
                     n_interp=20000, force_calc=False, verbose=None):
    """Return a (generated) LUT of Legendre (derivative) polynomial coeffs."""
    if n_interp % 2 != 0:
        raise RuntimeError('n_interp must be even')
    fname = op.join(_get_extra_data_path(), 'tables')
    if not op.isdir(fname):
        # Updated due to API change (GH 1167)
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
        logger.info('Generating Legendre%s table...' % extra_str)
        x_interp = np.linspace(-1, 1, n_interp + 1)
        lut = leg_fun(x_interp, n_coeff).astype(np.float32)
        if not force_calc:
            with open(fname, 'wb') as fid:
                fid.write(lut.tobytes())
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
            n_facts.append(n_fact / ((2.0 * n_fact + 1.0) *
                                     (2.0 * n_fact + 3.0)))
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


def _comp_sum_eeg(beta, ctheta, lut_fun, n_fact):
    """Lead field dot products using Legendre polynomial (P_n) series."""
    # Compute the sum occurring in the evaluation.
    # The result is
    #   sums[:]    (2n+1)^2/n beta^n P_n
    n_chunk = 50000000 // (8 * max(n_fact.shape) * 2)
    lims = np.concatenate([np.arange(0, beta.size, n_chunk), [beta.size]])
    s0 = np.empty(beta.shape)
    for start, stop in zip(lims[:-1], lims[1:]):
        coeffs = lut_fun(ctheta[start:stop])
        betans = np.tile(beta[start:stop][:, np.newaxis], (1, n_fact.shape[0]))
        np.cumprod(betans, axis=1, out=betans)  # run inplace
        coeffs *= betans
        s0[start:stop] = np.dot(coeffs, n_fact)  # == weighted sum across cols
    return s0


def _comp_sums_meg(beta, ctheta, lut_fun, n_fact, volume_integral):
    """Lead field dot products using Legendre polynomial (P_n) series.

    Parameters
    ----------
    beta : array, shape (n_points * n_points, 1)
        Coefficients of the integration.
    ctheta : array, shape (n_points * n_points, 1)
        Cosine of the angle between the sensor integration points.
    lut_fun : callable
        Look-up table for evaluating Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.
    volume_integral : bool
        If True, compute volume integral.

    Returns
    -------
    sums : array, shape (4, n_points * n_points)
        The results.
    """
    # Compute the sums occurring in the evaluation.
    # Two point magnetometers on the xz plane are assumed.
    # The four sums are:
    #  * sums[:, 0]    n(n+1)/(2n+1) beta^(n+1) P_n
    #  * sums[:, 1]    n/(2n+1) beta^(n+1) P_n'
    #  * sums[:, 2]    n/((2n+1)(n+1)) beta^(n+1) P_n'
    #  * sums[:, 3]    n/((2n+1)(n+1)) beta^(n+1) P_n''

    # This is equivalent, but slower:
    # sums = np.sum(bbeta[:, :, np.newaxis].T * n_fact * coeffs, axis=1)
    # sums = np.rollaxis(sums, 2)
    # or
    # sums = einsum('ji,jk,ijk->ki', bbeta, n_fact, lut_fun(ctheta)))
    sums = np.empty((n_fact.shape[1], len(beta)))
    # beta can be e.g. 3 million elements, which ends up using lots of memory
    # so we split up the computations into ~50 MB blocks
    n_chunk = 50000000 // (8 * max(n_fact.shape) * 2)
    lims = np.concatenate([np.arange(0, beta.size, n_chunk), [beta.size]])
    for start, stop in zip(lims[:-1], lims[1:]):
        bbeta = np.tile(beta[start:stop][np.newaxis], (n_fact.shape[0], 1))
        bbeta[0] *= beta[start:stop]
        np.cumprod(bbeta, axis=0, out=bbeta)  # run inplace
        einsum('ji,jk,ijk->ki', bbeta, n_fact, lut_fun(ctheta[start:stop]),
               out=sums[:, start:stop])
    return sums


###############################################################################
# SPHERE DOTS

_meg_const = 4e-14 * np.pi  # This is \mu_0^2/4\pi
_eeg_const = 1.0 / (4.0 * np.pi)


def _fast_sphere_dot_r0(r, rr1_orig, rr2s, lr1, lr2s, cosmags1, cosmags2s,
                        w1, w2s, volume_integral, lut, n_fact, ch_type):
    """Lead field dot product computation for M/EEG in the sphere model.

    Parameters
    ----------
    r : float
        The integration radius. It is used to calculate beta as:
        beta = (r * r) / (lr1 * lr2).
    rr1 : array, shape (n_points x 3)
        Normalized position vectors of integrations points in first sensor.
    rr2s : list
        Normalized position vector of integration points in second sensor.
    lr1 : array, shape (n_points x 1)
        Magnitude of position vector of integration points in first sensor.
    lr2s : list
        Magnitude of position vector of integration points in second sensor.
    cosmags1 : array, shape (n_points x 1)
        Direction of integration points in first sensor.
    cosmags2s : list
        Direction of integration points in second sensor.
    w1 : array, shape (n_points x 1) | None
        Weights of integration points in the first sensor.
    w2s : list
        Weights of integration points in the second sensor.
    volume_integral : bool
        If True, compute volume integral.
    lut : callable
        Look-up table for evaluating Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.
    ch_type : str
        The channel type. It can be 'meg' or 'eeg'.

    Returns
    -------
    result : float
        The integration sum.
    """
    if w1 is None:  # operating on surface, treat independently
        out_shape = (len(rr2s), len(rr1_orig))
        sum_axis = 1  # operate along second axis only at the end
    else:
        out_shape = (len(rr2s),)
        sum_axis = None  # operate on flattened array at the end
    out = np.empty(out_shape)
    rr2 = np.concatenate(rr2s)
    lr2 = np.concatenate(lr2s)
    cosmags2 = np.concatenate(cosmags2s)

    # outer product, sum over coords
    ct = einsum('ik,jk->ij', rr1_orig, rr2)
    np.clip(ct, -1, 1, ct)

    # expand axes
    rr1 = rr1_orig[:, np.newaxis, :]  # (n_rr1, n_rr2, n_coord) e.g. 4x4x3
    rr2 = rr2[np.newaxis, :, :]
    lr1lr2 = lr1[:, np.newaxis] * lr2[np.newaxis, :]

    beta = (r * r) / lr1lr2
    if ch_type == 'meg':
        sums = _comp_sums_meg(beta.flatten(), ct.flatten(), lut, n_fact,
                              volume_integral)
        sums.shape = (4,) + beta.shape

        # Accumulate the result, a little bit streamlined version
        # cosmags1 = cosmags1[:, np.newaxis, :]
        # cosmags2 = cosmags2[np.newaxis, :, :]
        # n1c1 = np.sum(cosmags1 * rr1, axis=2)
        # n1c2 = np.sum(cosmags1 * rr2, axis=2)
        # n2c1 = np.sum(cosmags2 * rr1, axis=2)
        # n2c2 = np.sum(cosmags2 * rr2, axis=2)
        # n1n2 = np.sum(cosmags1 * cosmags2, axis=2)
        n1c1 = einsum('ik,ijk->ij', cosmags1, rr1)
        n1c2 = einsum('ik,ijk->ij', cosmags1, rr2)
        n2c1 = einsum('jk,ijk->ij', cosmags2, rr1)
        n2c2 = einsum('jk,ijk->ij', cosmags2, rr2)
        n1n2 = einsum('ik,jk->ij', cosmags1, cosmags2)
        part1 = ct * n1c1 * n2c2
        part2 = n1c1 * n2c1 + n1c2 * n2c2

        result = (n1c1 * n2c2 * sums[0] +
                  (2.0 * part1 - part2) * sums[1] +
                  (n1n2 + part1 - part2) * sums[2] +
                  (n1c2 - ct * n1c1) * (n2c1 - ct * n2c2) * sums[3])

        # Give it a finishing touch!
        result *= (_meg_const / lr1lr2)
        if volume_integral:
            result *= r
    else:  # 'eeg'
        result = _comp_sum_eeg(beta.flatten(), ct.flatten(), lut, n_fact)
        result.shape = beta.shape
        # Give it a finishing touch!
        result *= _eeg_const
        result /= lr1lr2
        # now we add them all up with weights
    offset = 0
    result *= np.concatenate(w2s)
    if w1 is not None:
        result *= w1[:, np.newaxis]
    for ii, w2 in enumerate(w2s):
        out[ii] = np.sum(result[:, offset:offset + len(w2)], axis=sum_axis)
        offset += len(w2)
    return out


@fill_doc
def _do_self_dots(intrad, volume, coils, r0, ch_type, lut, n_fact, n_jobs):
    """Perform the lead field dot product integrations.

    Parameters
    ----------
    intrad : float
        The integration radius. It is used to calculate beta as:
        beta = (intrad * intrad) / (r1 * r2).
    volume : bool
        If True, perform volume integral.
    coils : list of dict
        The coils.
    r0 : array, shape (3 x 1)
        The origin of the sphere.
    ch_type : str
        The channel type. It can be 'meg' or 'eeg'.
    lut : callable
        Look-up table for evaluating Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.
    %(n_jobs)s

    Returns
    -------
    products : array, shape (n_coils, n_coils)
        The integration products.
    """
    if ch_type == 'eeg':
        intrad = intrad * 0.7
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
    """Parallelize."""
    # all possible combinations of two magnetometers
    products = np.zeros((len(rmags), len(rmags)))
    for ci1 in idx:
        ci2 = ci1 + 1
        res = _fast_sphere_dot_r0(
            intrad, rmags[ci1], rmags[:ci2], rlens[ci1], rlens[:ci2],
            cosmags[ci1], cosmags[:ci2], ws[ci1], ws[:ci2], volume, lut,
            n_fact, ch_type)
        products[ci1, :ci2] = res
        products[:ci2, ci1] = res
    return products


def _do_cross_dots(intrad, volume, coils1, coils2, r0, ch_type,
                   lut, n_fact):
    """Compute lead field dot product integrations between two coil sets.

    The code is a direct translation of MNE-C code found in
    `mne_map_data/lead_dots.c`.

    Parameters
    ----------
    intrad : float
        The integration radius. It is used to calculate beta as:
        beta = (intrad * intrad) / (r1 * r2).
    volume : bool
        If True, compute volume integral.
    coils1 : list of dict
        The original coils.
    coils2 : list of dict
        The coils to which data is being mapped.
    r0 : array, shape (3 x 1).
        The origin of the sphere.
    ch_type : str
        The channel type. It can be 'meg' or 'eeg'
    lut : callable
        Look-up table for evaluating Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.

    Returns
    -------
    products : array, shape (n_coils, n_coils)
        The integration products.
    """
    if ch_type == 'eeg':
        intrad = intrad * 0.7
    rmags1 = [coil['rmag'] - r0[np.newaxis, :] for coil in coils1]
    rmags2 = [coil['rmag'] - r0[np.newaxis, :] for coil in coils2]

    rlens1 = [np.sqrt(np.sum(r * r, axis=1)) for r in rmags1]
    rlens2 = [np.sqrt(np.sum(r * r, axis=1)) for r in rmags2]

    rmags1 = [r / rl[:, np.newaxis] for r, rl in zip(rmags1, rlens1)]
    rmags2 = [r / rl[:, np.newaxis] for r, rl in zip(rmags2, rlens2)]

    ws1 = [coil['w'] for coil in coils1]
    ws2 = [coil['w'] for coil in coils2]

    cosmags1 = [coil['cosmag'] for coil in coils1]
    cosmags2 = [coil['cosmag'] for coil in coils2]

    products = np.zeros((len(rmags1), len(rmags2)))
    for ci1 in range(len(coils1)):
        res = _fast_sphere_dot_r0(
            intrad, rmags1[ci1], rmags2, rlens1[ci1], rlens2, cosmags1[ci1],
            cosmags2, ws1[ci1], ws2, volume, lut, n_fact, ch_type)
        products[ci1, :] = res
    return products


@fill_doc
def _do_surface_dots(intrad, volume, coils, surf, sel, r0, ch_type,
                     lut, n_fact, n_jobs):
    """Compute the map construction products.

    Parameters
    ----------
    intrad : float
        The integration radius. It is used to calculate beta as:
        beta = (intrad * intrad) / (r1 * r2)
    volume : bool
        If True, compute a volume integral.
    coils : list of dict
        The coils.
    surf : dict
        The surface on which the field is interpolated.
    sel : array
        Indices of the surface vertices to select.
    r0 : array, shape (3 x 1)
        The origin of the sphere.
    ch_type : str
        The channel type. It can be 'meg' or 'eeg'.
    lut : callable
        Look-up table for Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.
    %(n_jobs)s

    Returns
    -------
    products : array, shape (n_coils, n_coils)
        The integration products.
    """
    # convert to normalized distances from expansion center
    rmags = [coil['rmag'] - r0[np.newaxis, :] for coil in coils]
    rlens = [np.sqrt(np.sum(r * r, axis=1)) for r in rmags]
    rmags = [r / rl[:, np.newaxis] for r, rl in zip(rmags, rlens)]
    cosmags = [coil['cosmag'] for coil in coils]
    ws = [coil['w'] for coil in coils]
    rref = None
    refl = None
    # virt_ref = False
    if ch_type == 'eeg':
        intrad = intrad * 0.7
        # The virtual ref code is untested and unused, so it is
        # commented out for now
        # if virt_ref:
        #     rref = virt_ref[np.newaxis, :] - r0[np.newaxis, :]
        #     refl = np.sqrt(np.sum(rref * rref, axis=1))
        #     rref /= refl[:, np.newaxis]

    rsurf = surf['rr'][sel] - r0[np.newaxis, :]
    lsurf = np.sqrt(np.sum(rsurf * rsurf, axis=1))
    rsurf /= lsurf[:, np.newaxis]
    this_nn = surf['nn'][sel]

    # loop over the coils
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
    """Parallelize.

    Parameters
    ----------
    refl : array | None
        If ch_type is 'eeg', the magnitude of position vector of the
        virtual reference (never used).
    lsurf : array
        Magnitude of position vector of the surface points.
    rlens : list of arrays of length n_coils
        Magnitude of position vector.
    this_nn : array, shape (n_vertices, 3)
        Surface normals.
    cosmags : list of array.
        Direction of the integration points in the coils.
    ws : list of array
        Integration weights of the coils.
    volume : bool
        If True, compute volume integral.
    lut : callable
        Look-up table for evaluating Legendre polynomials.
    n_fact : array
        Coefficients in the integration sum.
    ch_type : str
        'meg' or 'eeg'
    idx : array, shape (n_coils x 1)
        Index of coil.

    Returns
    -------
    products : array, shape (n_coils, n_coils)
        The integration products.
    """
    products = _fast_sphere_dot_r0(
        intrad, rsurf, rmags, lsurf, rlens, this_nn, cosmags, None, ws,
        volume, lut, n_fact, ch_type).T
    if rref is not None:
        raise NotImplementedError  # we don't ever use this, isn't tested
        # vres = _fast_sphere_dot_r0(
        #     intrad, rref, rmags, refl, rlens, this_nn, cosmags, None, ws,
        #     volume, lut, n_fact, ch_type)
        # products -= vres
    return products
