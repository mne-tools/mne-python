# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from copy import deepcopy

from ..surface import (fast_cross_3d, _find_nearest_tri_pt, _get_tri_supp_geom,
                       _triangle_coords)
from ..io.constants import FIFF
from ..transforms import apply_trans
from ..utils import logger
from ..parallel import parallel_func
from ..io.compensator import get_current_comp, make_compensator
from ..io.pick import pick_types


##############################################################################
# COIL SPECIFICATION

def _dup_coil_set(coils, coord_frame, t):
    """Make a duplicate"""
    if t is not None and coord_frame != t['from']:
        raise RuntimeError('transformation frame does not match the coil set')
    coils = deepcopy(coils)
    if t is not None:
        coord_frame = t['to']
        for coil in coils:
            coil['r0'] = apply_trans(t['trans'], coil['r0'])
            coil['ex'] = apply_trans(t['trans'], coil['ex'], False)
            coil['ey'] = apply_trans(t['trans'], coil['ey'], False)
            coil['ez'] = apply_trans(t['trans'], coil['ez'], False)
            coil['rmag'] = apply_trans(t['trans'], coil['rmag'])
            coil['cosmag'] = apply_trans(t['trans'], coil['cosmag'], False)
            coil['coord_frame'] = t['to']
    return coils, coord_frame


def _check_coil_frame(coils, coord_frame, bem):
    """Check to make sure the coils are in the correct coordinate frame"""
    if coord_frame != FIFF.FIFFV_COORD_MRI:
        if coord_frame == FIFF.FIFFV_COORD_HEAD:
            # Make a transformed duplicate
            coils, coord_Frame = _dup_coil_set(coils, coord_frame,
                                               bem['head_mri_t'])
        else:
            raise RuntimeError('Bad coil coordinate frame %d' % coord_frame)
    return coils, coord_frame


def _bem_lin_field_coeffs_simple(dest, normal, tri_rr, tri_nn, tri_area):
    """Simple version..."""
    out = np.zeros((3, len(dest)))
    for rr, o in zip(tri_rr, out):
        diff = dest - rr
        dl = np.sum(diff * diff, axis=1)
        x = fast_cross_3d(diff, tri_nn[np.newaxis, :])
        o[:] = tri_area * np.sum(x * normal, axis=1) / (3.0 * dl * np.sqrt(dl))
    return out


def _lin_field_coeff(s, mult, rmags, cosmags, ws, counts, func, n_jobs):
    """Use the linear field approximation to get field coefficients"""
    parallel, p_fun, _ = parallel_func(_do_lin_field_coeff, n_jobs)
    nas = np.array_split
    coeffs = parallel(p_fun(s['rr'], t, tn, ta,
                            rmags, cosmags, ws, counts, func)
                      for t, tn, ta in zip(nas(s['tris'], n_jobs),
                                           nas(s['tri_nn'], n_jobs),
                                           nas(s['tri_area'], n_jobs)))
    return mult * np.sum(coeffs, axis=0)


def _do_lin_field_coeff(rr, t, tn, ta, rmags, cosmags, ws, counts, func):
    """Actually get field coefficients (parallel-friendly)"""
    coeff = np.zeros((len(counts), len(rr)))
    bins = np.repeat(np.arange(len(counts)), counts)
    for tri, tri_nn, tri_area in zip(t, tn, ta):
        # Accumulate the coefficients for each triangle node
        # and add to the corresponding coefficient matrix
        tri_rr = rr[tri]

        # The following is equivalent to:
        #for j, coil in enumerate(coils['coils']):
        #    x = func(coil['rmag'], coil['cosmag'],
        #             tri_rr, tri_nn, tri_area)
        #    res = np.sum(coil['w'][np.newaxis, :] * x, axis=1)
        #    coeff[j][tri + off] += mult * res

        xx = func(rmags, cosmags, tri_rr, tri_nn, tri_area)
        # only loops 3x (one per direction)
        zz = np.array([np.bincount(bins, weights=x * ws,
                                   minlength=len(counts)) for x in xx])
        coeff[:, tri] += zz.T
    return coeff


def _bem_specify_coils(bem, coils, coord_frame, n_jobs):
    """Set up for computing the solution at a set of coils"""
    # Compute the weighting factors to obtain the magnetic field
    # in the linear potential approximation
    coils, coord_frame = _check_coil_frame(coils, coord_frame, bem)

    # leaving this in in case we want to easily add in the future
    #if method != 'simple':  # in ['ferguson', 'urankar']:
    #    raise NotImplementedError
    #else:
    func = _bem_lin_field_coeffs_simple

    # Process each of the surfaces
    rmags = np.concatenate([coil['rmag'] for coil in coils])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils])
    counts = np.array([len(coil['rmag']) for coil in coils])
    ws = np.concatenate([coil['w'] for coil in coils])

    lens = np.cumsum(np.r_[0, [len(s['rr']) for s in bem['surfs']]])
    coeff = np.empty((len(counts), lens[-1]))
    for o1, o2, surf, mult in zip(lens[:-1], lens[1:],
                                  bem['surfs'], bem['field_mult']):
        coeff[:, o1:o2] = _lin_field_coeff(surf, mult, rmags, cosmags,
                                           ws, counts, func, n_jobs)
    # put through the bem
    sol = np.dot(coeff, bem['solution'])
    return sol


def _bem_specify_els(bem, els):
    """Set up for computing the solution at a set of electrodes"""
    sol = np.zeros((len(els), bem['solution'].shape[1]))
    # Go through all coils
    scalp = bem['surfs'][0]
    scalp['geom'] = _get_tri_supp_geom(scalp['tris'], scalp['rr'])
    inds = np.arange(len(scalp['tris']))

    # In principle this could be parallelized, but pickling overhead is huge
    # (makes it slower than non-parallel)
    for k, el in enumerate(els):
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
    return sol


#############################################################################
# FORWARD COMPUTATION

def _make_ctf_comp_coils(info, coils):
    """Get the correct compensator for CTF coils"""
    # adapted from mne_make_ctf_comp() from mne_ctf_comp.c
    logger.info('Setting up compensation data...')
    comp_num = get_current_comp(info)
    if comp_num is None or comp_num == 0:
        logger.info('    No compensation set. Nothing more to do.')
        return None

    # Need to meaningfully populate comp['set'] dict a.k.a. compset
    n_comp_ch = sum([c['kind'] == FIFF.FIFFV_MEG_CH for c in info['chs']])
    logger.info('    %d out of %d channels have the compensation set.'
                % (n_comp_ch, len(coils)))

    # Find the desired compensation data matrix
    compensator = make_compensator(info, 0, comp_num, True)
    logger.info('    Desired compensation data (%s) found.' % comp_num)
    logger.info('    All compensation channels found.')
    logger.info('    Preselector created.')
    logger.info('    Compensation data matrix created.')
    logger.info('    Postselector created.')
    return compensator


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


def _bem_pot_or_field(rr, mri_rr, mri_Q, mults, coils, solution, srr,
                      n_jobs, coil_type):
    """Calculate the magnetic field or electric potential

    The code is very similar between EEG and MEG potentials, so we'll
    combine them.
    """
    # multiply solution by "mults" here for simplicity
    # we can do this one in-place because it's not used elsewhere
    solution *= mults

    # Both MEG and EEG have the inifinite-medium potentials
    # This could be just vectorized, but eats too much memory, so instead we
    # reduce memory by chunking within _do_inf_pots and parallelize, too:
    parallel, p_fun, _ = parallel_func(_do_inf_pots, n_jobs)
    nas = np.array_split
    B = np.sum(parallel(p_fun(mri_rr, sr.copy(), mri_Q, sol.copy())
                        for sr, sol in zip(nas(srr, n_jobs),
                                           nas(solution.T, n_jobs))), axis=0)
    # The copy()s above should make it so the whole objects don't need to be
    # pickled...

    # Only MEG gets the primary current distribution
    if coil_type == 'meg':
        # Primary current contribution (can be calc. in coil/dipole coords)
        parallel, p_fun, _ = parallel_func(_do_prim_curr, n_jobs)
        pcc = np.concatenate(parallel(p_fun(rr, c)
                                      for c in nas(coils, n_jobs)), axis=1)
        B += pcc
        B *= 1e-7  # MAG_FACTOR from C code
    return B


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
    for bi in range(len(bounds) - 1):
        v0s = _bem_inf_pots(rr[bounds[bi]:bounds[bi + 1]], srr, mri_Q)
        v0s.shape = (v0s.shape[0] * 3, v0s.shape[2])
        B[3 * bounds[bi]:3 * bounds[bi + 1]] = np.dot(v0s, sol)
    return B


def _compute_forwards(src, bem, coils_list, cfs, ccoils_list, ccfs,
                      infos, coil_types, n_jobs):
    """Compute the MEG and EEG forward solutions"""
    if bem['bem_method'] != 'linear collocation':
        raise RuntimeError('only linear collocation supported')

    # Precompute some things that are used for both MEG and EEG
    rr = np.concatenate([s['rr'][s['vertno']] for s in src])
    mults = np.repeat(bem['source_mult'] / (4.0 * np.pi),
                      [len(s['rr']) for s in bem['surfs']])[np.newaxis, :]
    # The dipole location and orientation must be transformed
    mri_rr = apply_trans(bem['head_mri_t']['trans'], rr)
    mri_Q = apply_trans(bem['head_mri_t']['trans'], np.eye(3), False)
    srr = np.concatenate([s['rr'] for s in bem['surfs']])

    # Now, actually compute MEG and EEG solutions
    Bs = list()
    for coil_type, coils, cf, ccoils, ccf, info in zip(coil_types, coils_list,
                                                       cfs, ccoils_list, ccfs,
                                                       infos):
        if coils is None:  # nothing to do
            Bs.append(None)
        else:
            if coil_type == 'meg':
                # Compose a compensation data set if necessary
                compensator = _make_ctf_comp_coils(info, coils)

                # Field computation matrices...
                logger.info('')
                start = 'Composing the field computation matrix'
                logger.info(start + '...')
                solution = _bem_specify_coils(bem, coils, cf, n_jobs)
                if compensator is not None:
                    logger.info(start + ' (compensation coils)...')
                    csolution = _bem_specify_coils(bem, ccoils, ccf, n_jobs)

            elif coil_type == 'eeg':
                solution = _bem_specify_els(bem, coils)
                compensator = None

            # Do the actual calculation
            logger.info('Computing %s at %d source locations '
                        '(free orientations)...'
                        % (coil_type.upper(), len(rr)))
            # Note: this function modifies "solution" in-place
            B = _bem_pot_or_field(rr, mri_rr, mri_Q, mults, coils,
                                  solution, srr, n_jobs, coil_type)

            # Compensate if needed (only done for MEG systems w/compensation)
            if compensator is not None:
                # Compute the field in the compensation coils
                work = _bem_pot_or_field(rr, mri_rr, mri_Q, mults,
                                         ccoils, csolution, srr, n_jobs,
                                         coil_type)
                # Combine solutions so we can do the compensation
                both = np.zeros((work.shape[0], B.shape[1] + work.shape[1]))
                picks = pick_types(info, meg=True, ref_meg=False)
                both[:, picks] = B
                picks = pick_types(info, meg=False, ref_meg=True)
                both[:, picks] = work
                B = np.dot(both, compensator.T)
            Bs.append(B)

    return Bs
