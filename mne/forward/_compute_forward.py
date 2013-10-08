# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from copy import deepcopy
from functools import partial

from ..surface import (fast_cross_3d, _find_nearest_tri_pt, _get_tri_supp_geom,
                       _triangle_coords)
from ..fiff.constants import FIFF
from ..transforms import apply_trans
from ..utils import logger
from ..parallel import parallel_func
from ..fiff.compensator import get_current_comp, make_compensator
from ..fiff.pick import pick_types


##############################################################################
# COIL DEFINITIONS

def _dup_coil_set(s, t):
    """Make a duplicate"""
    if t is not None and s['coord_frame'] != t['from']:
        raise RuntimeError('transformation frame does not match the coil set')
    res = deepcopy(s)
    if t is not None:
        res['coord_frame'] = t['to']
        for coil in res['coils']:
            coil['r0'] = apply_trans(t['trans'], coil['r0'])
            coil['ex'] = apply_trans(t['trans'], coil['ex'], False)
            coil['ey'] = apply_trans(t['trans'], coil['ey'], False)
            coil['ez'] = apply_trans(t['trans'], coil['ez'], False)
            coil['rmag'] = apply_trans(t['trans'], coil['rmag'])
            coil['cosmag'] = apply_trans(t['trans'], coil['cosmag'], False)
            coil['coord_frame'] = t['to']
    return res


def _bem_one_lin_field_coeff_simple(dest, normal, tri_rr, tri_nn, tri_area):
    """Simple version..."""
    out = np.zeros((3, len(dest)))
    for rr, o in zip(tri_rr, out):
        diff = dest - rr
        dl = np.sum(diff * diff, axis=1)
        x = fast_cross_3d(diff, tri_nn[np.newaxis, :])
        o[:] = tri_area * np.sum(x * normal, axis=1) / (3.0 * dl * np.sqrt(dl))
    return out


def _check_coil_frame(coils, bem):
    if coils['coord_frame'] != FIFF.FIFFV_COORD_MRI:
        if coils['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            # Make a transformed duplicate
            coils = _dup_coil_set(coils, bem['head_mri_t'])
        else:
            raise RuntimeError('Bad coil coordinate frame %d'
                               % coils['coord_frame'])
    return coils


def _bem_lin_field_coeff(bem, coils, n_jobs, method):
    """Compute the weighting factors to obtain the magnetic field"""
    #in the linear potential approximation
    coils = _check_coil_frame(coils, bem)

    # leaving this in in case we want to easily add in the future
    #if method != 'simple':  # in ['ferguson', 'urankar']:
    #    raise NotImplementedError
    #else:
    func = _bem_one_lin_field_coeff_simple

    # Process each of the surfaces
    rmags = np.concatenate([coil['rmag'] for coil in coils['coils']])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils['coils']])
    lims = np.cumsum(np.r_[0, [len(coil['rmag']) for coil in coils['coils']]])
    ws = np.concatenate([coil['w'] for coil in coils['coils']])

    # might as well parallelize over surfaces here
    parallel, p_fun, _ = parallel_func(_do_lin_field_coeff, n_jobs)
    coeff = parallel(p_fun(surf, mult, rmags, cosmags, ws, lims, func)
                     for surf, mult in zip(bem['surfs'], bem['field_mult']))
    # coeff checked against fwd_bem_lin_field_coeff, equiv to at least 7
    coeff = np.concatenate(coeff, axis=1)
    return coeff


def _do_lin_field_coeff(surf, mult, rmags, cosmags, ws, lims, func):
    coeff = np.zeros((len(lims) - 1, surf['np']))
    for tri, tri_nn, tri_area in zip(surf['tris'],
                                     surf['tri_nn'], surf['tri_area']):
        # Accumulate the coefficients for each triangle node
        # and add to the corresponding coefficient matrix
        tri_rr = surf['rr'][tri]

        # The following is equivalent to:
        #for j, coil in enumerate(coils['coils']):
        #    x = func(coil['rmag'], coil['cosmag'],
        #             tri_rr, tri_nn, tri_area)
        #    res = np.sum(coil['w'][np.newaxis, :] * x, axis=1)
        #    coeff[j][tri + off] += mult * res

        xx = func(rmags, cosmags, tri_rr, tri_nn, tri_area)
        yy = np.c_[np.zeros((3, 1)), np.cumsum(xx * ws, axis=1)]
        zz = mult * np.diff(yy[:, lims], axis=1)
        coeff[:, tri] += zz.T
    return coeff


def _bem_specify_coils(bem, coils, n_jobs):
    """Set up for computing the solution at a set of coils"""
    sol = _bem_lin_field_coeff(bem, coils, n_jobs, 'simple')
    sol = np.dot(sol, bem['solution'])
    coils['solution'] = sol


def _bem_specify_els(bem, els):
    """Set up for computing the solution at a set of electrodes"""
    sol = np.zeros((len(els['coils']), bem['nsol']))
    # Go through all coils
    scalp = bem['surfs'][0]
    scalp['geom'] = _get_tri_supp_geom(scalp['tris'], scalp['rr'])
    inds = np.arange(len(scalp['tris']))

    # In principle this could be parallelized, but pickling overhead is huge
    # (makes it slower than non-parallel)
    for k, el in enumerate(els['coils']):
        # Go through all 'integration points'
        el_r = apply_trans(bem['head_mri_t']['trans'], el['rmag'])
        for elw, r in zip(el['w'], el_r):
            best = _find_nearest_tri_pt(inds, r, scalp['geom'], True)[2]
            # Calculate a linear interpolation between the vertex values
            tri = scalp['tris'][best]
            x, y, z = _triangle_coords(r, scalp['geom'], best)
            w = elw * np.array([(1.0 - x - y), x, y])
            amt = np.dot(w, bem['solution'][tri])
            sol[k] += amt
    els['solution'] = sol


#############################################################################
# FORWARD COMPUTATION

def _make_ctf_comp_coils(comp, coils):
    """Call mne_make_ctf_comp using the information in the coil sets"""
    info = comp['set']['info']

    # adapted from mne_make_ctf_comp() from mne_ctf_comp.c
    logger.info('Setting up compensation data...')
    comp_num = get_current_comp(info)
    if comp_num is None or comp_num == 0:
        logger.info('    No compensation set. Nothing more to do.')
        comp['set']['current'] = None
        return

    # Need to meaningfully populate comp['set'] dict a.k.a. compset
    n_comp_ch = sum([c['kind'] == FIFF.FIFFV_MEG_CH for c in info['chs']])
    logger.info('    %d out of %d channels have the compensation set.'
                % (n_comp_ch, len(coils['coils'])))

    # Find the desired compensation data matrix
    comp['set']['current'] = make_compensator(info, 0, comp_num, True)
    logger.info('    Desired compensation data (%s) found.' % comp_num)
    logger.info('    All compensation channels found.')
    logger.info('    Preselector created.')
    logger.info('    Compensation data matrix created.')
    logger.info('    Postselector created.')


#def _bem_inf_pot(rd, Q, rp):
#    """The infinite medium potential in one direction"""
#    # NOTE: the (4.0 * np.pi) that was in the denominator has been moved!
#    diff = rp - rd
#    diff2 = np.sum(diff * diff, axis=1)
#    return np.sum(Q * diff, axis=1) / (diff2 * np.sqrt(diff2))


def _bem_inf_pots(rr, surf_rr, Q=None):
    """The infinite medium potential in all 3 directions"""
    # NOTE: the (4.0 * np.pi) that was in the denominator has been moved!
    diff = surf_rr.T[np.newaxis, :, :] - rr[:, :, np.newaxis]  # n_rr, 3, n_bem
    diff_norm = np.sum(diff * diff, axis=1)
    diff_norm *= np.sqrt(diff_norm)
    diff_norm[diff_norm == 0] = 1  # avoid nans
    if Q is None:  # save time when Q=np.eye(3) (e.g., MEG sensors)
        return diff / diff_norm[:, np.newaxis, :]
    else:  # get components in each direction (e.g., EEG sensors)
        return np.einsum('ijk,mj->imk', diff, Q) / diff_norm[:, np.newaxis, :]


# This function has been refactored to process all points simultaneously
#def _bem_inf_field(rd, Q, rp, d):
#    """Infinite-medium magnetic field"""
#    diff = rp - rd
#    diff2 = np.sum(diff * diff, axis=1)
#    x = fast_cross_3d(Q[np.newaxis, :], diff)
#    return np.sum(x * d, axis=1) / (diff2 * np.sqrt(diff2))


def _bem_inf_fields(rr, rp, c):
    """Infinite-medium magnetic field in all 3 basis directions"""
    # Knowing that we're doing all directions, the above can be refactored:
    diff = rp.T[np.newaxis, :, :] - rr[:, :, np.newaxis]
    diff_norm = np.sum(diff * diff, axis=1)
    diff_norm *= np.sqrt(diff_norm)
    diff_norm[diff_norm == 0] = 1  # avoid nans
    # This is the result of cross-prod calcs with basis vectors,
    # as if we had taken (Q=np.eye(3)), then multiplied by the cosmags (c)
    # factor, and then summed across directions
    x = np.array([diff[:, 1] * c[:, 2] - diff[:, 2] * c[:, 1],
                  diff[:, 2] * c[:, 0] - diff[:, 0] * c[:, 2],
                  diff[:, 0] * c[:, 1] - diff[:, 1] * c[:, 0]])
    return np.rollaxis(x / diff_norm, 1)


def _comp_field(rrs, coils, comp, n_jobs):
    """Calculate the compensated field"""
    # First compute the field in the primary set of coils
    res, dbg = comp['field'](rrs, coils, comp['client'], n_jobs)

    # Compensation needed?
    if comp['set']['current'] is not None:
        # Compute the field in the compensation coils
        work, _ = comp['field'](rrs, comp['comp_coils'], comp['client'],
                                n_jobs)  # XXX dbg
        # Combine solutions so we can do the compensation
        both = np.zeros((work.shape[0], res.shape[1] + work.shape[1]))
        picks = pick_types(comp['set']['info'], meg=True, ref_meg=False)
        both[:, picks] = res
        picks = pick_types(comp['set']['info'], meg=False, ref_meg=True)
        both[:, picks] = work
        res = np.dot(both, comp['set']['current'].T)
    return res, dbg


def _bem_pot_or_field(rr, coils, bem, n_jobs, ctype):
    """Calculate the magnetic field or electric potential

    The code is very similar between EEG and MEG potentials, so we'll
    combine them.
    """
    # multiply solution by "mults" here for simplicity
    mults = np.repeat(bem['source_mult'] / (4.0 * np.pi),
                      [len(s['rr']) for s in bem['surfs']])
    solution = coils['solution'] * mults[np.newaxis, :]
    dbg = dict(solution=coils['solution'], mults=mults)

    # The dipole location and orientation must be transformed
    mri_rr = apply_trans(bem['head_mri_t']['trans'], rr)
    mri_Q = apply_trans(bem['head_mri_t']['trans'], np.eye(3), False)
    dbg['rr'] = rr.copy()
    dbg['mri_rr'] = mri_rr.copy()
    dbg['mri_Q'] = mri_Q.copy()

    # Both MEG and EEG have the inifinite-medium potentials
    srr = np.concatenate([s['rr'] for s in bem['surfs']])
    dbg['srr'] = srr
    # This could be just vectorized, but eats too much memory, so instead we
    # reduce memory by chunking within _do_inf_pots and parallelize, too:
    parallel, p_fun, _ = parallel_func(_do_inf_pots, n_jobs)
    nas = np.array_split
    B = np.sum(parallel(p_fun(mri_rr, sr.copy(), mri_Q, sol.copy())
                        for sr, sol in zip(nas(srr, n_jobs),
                                           nas(solution.T, n_jobs))), axis=0)
    dbg['B0'] = B.copy()
    # The copy()s above should make it so the whole objects don't need to be
    # pickled...

    # Only MEG gets the primary current distribution
    if ctype.lower() == 'meg':
        # Primary current contribution (can be calc. in coil/dipole coords)
        parallel, p_fun, _ = parallel_func(_do_prim_curr, n_jobs)
        pcc = np.concatenate(parallel(p_fun(rr, c)
                                      for c in nas(coils['coils'], n_jobs)),
                             axis=1)
        B += pcc
        B *= 1e-7  # MAG_FACTOR from C code
    dbg['B'] = B.copy()
    return B, dbg


def _do_prim_curr(rr, coils):
    """Calculate primary currents in a set of coils"""
    out = np.empty((len(rr) * 3, len(coils)))
    for ci, c in enumerate(coils):
        out[:, ci] = np.sum(c['w'] * _bem_inf_fields(rr, c['rmag'],
                                                     c['cosmag']), 2).ravel()
    return out


def _do_inf_pots(rr, srr, mri_Q, sol):
    """Calculate infinite potentials using chunks"""
    # The following code is equivalent to this, but saves memory
    #v0s = _bem_inf_pots(rr, srr, mri_Q)  # n_rr x 3 x n_surf_rr
    #v0s.shape = (len(rr) * 3, v0s.shape[2])
    #B = np.dot(v0s, sol)

    # We chunk the source rr's in order to save memory
    bounds = np.r_[np.arange(0, len(rr), 1000), len(rr)]
    B = np.empty((len(rr) * 3, sol.shape[1]))
    for bi in xrange(len(bounds) - 1):
        v0s = _bem_inf_pots(rr[bounds[bi]:bounds[bi + 1]], srr, mri_Q)
        v0s.shape = (v0s.shape[0] * 3, v0s.shape[2])
        B[3 * bounds[bi]:3 * bounds[bi + 1]] = np.dot(v0s, sol)
    return B


def _compute_forward(src, coils, comp_coils, info, bem, ctype, n_jobs):
    """Compute the M/EEG forward solution"""
    if bem['bem_method'] != 'linear collocation':
        raise RuntimeError('only linear collocation supported')
    field = partial(_bem_pot_or_field, ctype=ctype)
    if ctype == 'meg':
        # Use the new compensated field computation
        # It works the same way independent of whether or not the compensation
        # is in effect

        # Compose a compensation data set
        comp = dict(set=dict(info=info), comp_coils=comp_coils, field=field,
                    vec_field=None, client=bem)
        _make_ctf_comp_coils(comp, coils)

        # Field computation matrices...
        logger.info('')
        logger.info('Composing the field computation matrix...')
        _bem_specify_coils(bem, coils, n_jobs)

        if comp['set']['current'] is not None:
            logger.info('Composing the field computation matrix '
                        '(compensation coils)...')
            _bem_specify_coils(bem, comp['comp_coils'], n_jobs)
        field = _comp_field
        client = comp
    elif ctype == 'eeg':
        _bem_specify_els(bem, coils)
        client = bem

    rrs = np.concatenate([s['rr'][s['vertno']] for s in src])
    logger.info('Computing %s at %d source locations '
                '(free orientations)...' % (ctype.upper(), len(rrs)))
    res, dbg = field(rrs, coils, client, n_jobs)
    return res, dbg
