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
from ..utils import logger, verbose
from ..parallel import parallel_func
from ..io.compensator import get_current_comp, make_compensator
from ..io.pick import pick_types


# #############################################################################
# COIL SPECIFICATION AND FIELD COMPUTATION MATRIX

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
            raise RuntimeError('Bad coil coordinate frame %s' % coord_frame)
    return coils, coord_frame


def _lin_field_coeff(s, mult, rmags, cosmags, ws, counts, n_jobs):
    """Use the linear field approximation to get field coefficients"""
    parallel, p_fun, _ = parallel_func(_do_lin_field_coeff, n_jobs)
    nas = np.array_split
    coeffs = parallel(p_fun(s['rr'], t, tn, ta,
                            rmags, cosmags, ws, counts)
                      for t, tn, ta in zip(nas(s['tris'], n_jobs),
                                           nas(s['tri_nn'], n_jobs),
                                           nas(s['tri_area'], n_jobs)))
    return mult * np.sum(coeffs, axis=0)


def _do_lin_field_coeff(rr, t, tn, ta, rmags, cosmags, ws, counts):
    """Actually get field coefficients (parallel-friendly)"""
    coeff = np.zeros((len(counts), len(rr)))
    bins = np.repeat(np.arange(len(counts)), counts)
    for tri, tri_nn, tri_area in zip(t, tn, ta):
        # Accumulate the coefficients for each triangle node
        # and add to the corresponding coefficient matrix
        tri_rr = rr[tri]

        # The following is equivalent to:
        # for j, coil in enumerate(coils['coils']):
        #     x = func(coil['rmag'], coil['cosmag'],
        #              tri_rr, tri_nn, tri_area)
        #     res = np.sum(coil['w'][np.newaxis, :] * x, axis=1)
        #     coeff[j][tri + off] += mult * res

        # Simple version (bem_lin_field_coeffs_simple)
        zz = []
        for trr in tri_rr:
            diff = rmags - trr
            dl = np.sum(diff * diff, axis=1)
            c = fast_cross_3d(diff, tri_nn[np.newaxis, :])
            x = tri_area * np.sum(c * cosmags, axis=1) / \
                (3.0 * dl * np.sqrt(dl))
            zz += [np.bincount(bins, weights=x * ws, minlength=len(counts))]
        coeff[:, tri] += np.array(zz).T
    return coeff


def _concatenate_coils(coils):
    """Helper to concatenate coil parameters"""
    rmags = np.concatenate([coil['rmag'] for coil in coils])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils])
    ws = np.concatenate([coil['w'] for coil in coils])
    counts = np.array([len(coil['rmag']) for coil in coils])
    return rmags, cosmags, ws, counts


def _bem_specify_coils(bem, coils, coord_frame, mults, n_jobs):
    """Set up for computing the solution at a set of coils"""
    # Compute the weighting factors to obtain the magnetic field
    # in the linear potential approximation
    coils, coord_frame = _check_coil_frame(coils, coord_frame, bem)

    # leaving this in in case we want to easily add in the future
    # if method != 'simple':  # in ['ferguson', 'urankar']:
    #     raise NotImplementedError

    # Process each of the surfaces
    rmags, cosmags, ws, counts = _concatenate_coils(coils)
    lens = np.cumsum(np.r_[0, [len(s['rr']) for s in bem['surfs']]])
    coeff = np.empty((len(counts), lens[-1]))
    for o1, o2, surf, mult in zip(lens[:-1], lens[1:],
                                  bem['surfs'], bem['field_mult']):
        coeff[:, o1:o2] = _lin_field_coeff(surf, mult, rmags, cosmags,
                                           ws, counts, n_jobs)
    # put through the bem
    sol = np.dot(coeff, bem['solution'])
    sol *= mults
    return sol


def _bem_specify_els(bem, els, mults):
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
    sol *= mults
    return sol


# #############################################################################
# COMPENSATION

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


# #############################################################################
# BEM COMPUTATION

_MAG_FACTOR = 1e-7  # μ_0 / (4π)

# def _bem_inf_pot(rd, Q, rp)*pi:
#     """The infinite medium potential in one direction"""
#     # NOTE: the (4.0 * np.pi) that was in the denominator has been moved!
#     diff = rp - rd
#     diff2 = np.sum(diff * diff, axis=1)
#     return np.sum(Q * diff, axis=1) / (diff2 * np.sqrt(diff2))


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
# def _bem_inf_field(rd, Q, rp, d):
#     """Infinite-medium magnetic field"""
#     diff = rp - rd
#     diff2 = np.sum(diff * diff, axis=1)
#     x = fast_cross_3d(Q[np.newaxis, :], diff)
#     return np.sum(x * d, axis=1) / (diff2 * np.sqrt(diff2))


def _bem_inf_fields(rr, rp, c):
    """Infinite-medium magnetic field in all 3 basis directions
    rr : n x 3 array
        dipole locations
    rp : n x 3 array?
        coil 'rmag'
    c : coil 'cosmag'
    """
    # Knowing that we're doing all directions, the above can be refactored:
    diff = rp.T[np.newaxis, :, :] - rr[:, :, np.newaxis]
    diff_norm = np.sum(diff * diff, axis=1)
    diff_norm *= np.sqrt(diff_norm)
    diff_norm[diff_norm == 0] = 1  # avoid nans
    # This is the result of cross-prod calcs with basis vectors,
    # as if we had taken (Q=np.eye(3)), then multiplied by the cosmags (c)
    # factor, and then summed across directions
    # Equation (19) in Mosher, 1999
    x = np.array([diff[:, 1] * c[:, 2] - diff[:, 2] * c[:, 1],
                  diff[:, 2] * c[:, 0] - diff[:, 0] * c[:, 2],
                  diff[:, 0] * c[:, 1] - diff[:, 1] * c[:, 0]])
    return np.rollaxis(x / diff_norm, 1)


def _bem_pot_or_field(rr, mri_rr, mri_Q, coils, solution, srr,
                      n_jobs, coil_type):
    """Calculate the magnetic field or electric potential

    The code is very similar between EEG and MEG potentials, so we'll
    combine them.

    This does the work of "fwd_comp_field" (which wraps to "fwd_bem_field") and
    "fwd_bem_pot_els" in MNE-C.
    """
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
        B *= _MAG_FACTOR
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
    # v0s = _bem_inf_pots(rr, srr, mri_Q)  # n_rr x 3 x n_surf_rr
    # v0s.shape = (len(rr) * 3, v0s.shape[2])
    # B = np.dot(v0s, sol)

    # We chunk the source rr's in order to save memory
    bounds = np.r_[np.arange(0, len(rr), 1000), len(rr)]
    B = np.empty((len(rr) * 3, sol.shape[1]))
    for bi in range(len(bounds) - 1):
        v0s = _bem_inf_pots(rr[bounds[bi]:bounds[bi + 1]], srr, mri_Q)
        v0s.shape = (v0s.shape[0] * 3, v0s.shape[2])
        B[3 * bounds[bi]:3 * bounds[bi + 1]] = np.dot(v0s, sol)
    return B


# #############################################################################
# SPHERE COMPUTATION

def _sphere_pot_or_field(rr, mri_rr, mri_Q, coils, sphere, srr,
                         n_jobs, coil_type):
    """Do potential or field for spherical model"""
    fun = _eeg_spherepot_coil if coil_type == 'eeg' else _sphere_field
    parallel, p_fun, _ = parallel_func(fun, n_jobs)
    B = np.concatenate(parallel(p_fun(r, coils, sphere)
                       for r in np.array_split(rr, n_jobs)))
    return B


def _sphere_field(rrs, coils, sphere):
    """This uses Jukka Sarvas' field computation

    Jukka Sarvas, "Basic mathematical and electromagnetic concepts of the
    biomagnetic inverse problem", Phys. Med. Biol. 1987, Vol. 32, 1, 11-22.

    The formulas have been manipulated for efficient computation
    by Matti Hamalainen, February 1990
    """
    rmags, cosmags, ws, counts = _concatenate_coils(coils)
    bins = np.repeat(np.arange(len(counts)), counts)

    # Shift to the sphere model coordinates
    rrs = rrs - sphere['r0']

    B = np.zeros((3 * len(rrs), len(coils)))
    for ri, rr in enumerate(rrs):
        # Check for a dipole at the origin
        if np.sqrt(np.dot(rr, rr)) <= 1e-10:
            continue
        this_poss = rmags - sphere['r0']

        # Vector from dipole to the field point
        a_vec = this_poss - rr
        a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
        r = np.sqrt(np.sum(this_poss * this_poss, axis=1))
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
        v1 = fast_cross_3d(rr[np.newaxis, :], cosmags)
        v2 = fast_cross_3d(rr[np.newaxis, :], this_poss)
        xx = ((good * ws)[:, np.newaxis] *
              (v1 / F[:, np.newaxis] + v2 * g[:, np.newaxis]))
        zz = np.array([np.bincount(bins, weights=x,
                                   minlength=len(counts)) for x in xx.T])
        B[3 * ri:3 * ri + 3, :] = zz
    B *= _MAG_FACTOR
    return B


def _eeg_spherepot_coil(rrs, coils, sphere):
    """Calculate the EEG in the sphere model"""
    rmags, cosmags, ws, counts = _concatenate_coils(coils)
    bins = np.repeat(np.arange(len(counts)), counts)

    # Shift to the sphere model coordinates
    rrs = rrs - sphere['r0']

    B = np.zeros((3 * len(rrs), len(coils)))
    for ri, rr in enumerate(rrs):
        # Only process dipoles inside the innermost sphere
        if np.sqrt(np.dot(rr, rr)) >= sphere['layers'][0]['rad']:
            continue
        # fwd_eeg_spherepot_vec
        vval_one = np.zeros((len(rmags), 3))

        # Make a weighted sum over the equivalence parameters
        for eq in range(sphere['nfit']):
            # Scale the dipole position
            rd = sphere['mu'][eq] * rr
            rd2 = np.sum(rd * rd)
            rd2_inv = 1.0 / rd2
            # Go over all electrodes
            this_pos = rmags - sphere['r0']

            # Scale location onto the surface of the sphere (not used)
            # if sphere['scale_pos']:
            #     pos_len = (sphere['layers'][-1]['rad'] /
            #                np.sqrt(np.sum(this_pos * this_pos, axis=1)))
            #     this_pos *= pos_len

            # Vector from dipole to the field point
            a_vec = this_pos - rd

            # Compute the dot products needed
            a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
            a3 = 2.0 / (a * a * a)
            r2 = np.sum(this_pos * this_pos, axis=1)
            r = np.sqrt(r2)
            rrd = np.sum(this_pos * rd, axis=1)
            ra = r2 - rrd
            rda = rrd - rd2

            # The main ingredients
            F = a * (r * a + ra)
            c1 = a3 * rda + 1.0 / a - 1.0 / r
            c2 = a3 + (a + r) / (r * F)

            # Mix them together and scale by lambda/(rd*rd)
            m1 = (c1 - c2 * rrd)
            m2 = c2 * rd2

            vval_one += (sphere['lambda'][eq] * rd2_inv *
                         (m1[:, np.newaxis] * rd +
                          m2[:, np.newaxis] * this_pos))

            # compute total result
            xx = vval_one * ws[:, np.newaxis]
            zz = np.array([np.bincount(bins, weights=x,
                                       minlength=len(counts)) for x in xx.T])
            B[3 * ri:3 * ri + 3, :] = zz
    # finishing by scaling by 1/(4*M_PI)
    B *= 0.25 / np.pi
    return B


# #############################################################################
# MAGNETIC DIPOLE (e.g. CHPI)

def _magnetic_dipole_field_vec(rrs, coils):
    """Compute an MEG forward solution for a set of magnetic dipoles"""
    fwd = np.empty((3 * len(rrs), len(coils)))
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

    fwd = np.empty((3 * len(rrs), len(coils)))
    rmags, cosmags, ws, counts = _concatenate_coils(coils)
    bins = np.repeat(np.arange(len(counts)), counts)
    for ri, rr in enumerate(rrs):
        diff = rmags - rr
        dist2 = np.sum(diff * diff, axis=1)[:, np.newaxis]
        dist = np.sqrt(dist2)
        if (dist < 1e-5).any():
            raise RuntimeError('Coil too close (dist = %g m)' % dist.min())
        sum_ = ws[:, np.newaxis] * (3 * diff * np.sum(diff * cosmags,
                                                      axis=1)[:, np.newaxis] -
                                    dist2 * cosmags) / (dist2 * dist2 * dist)
        for ii in range(3):
            fwd[3 * ri + ii] = np.bincount(bins, weights=sum_[:, ii],
                                           minlength=len(counts))
    fwd *= 1e-7
    return fwd


# #############################################################################
# MAIN TRIAGING FUNCTION

@verbose
def _prep_field_computation(rr, bem, fwd_data, n_jobs, verbose=None):
    """Precompute some things that are used for both MEG and EEG"""
    cf = FIFF.FIFFV_COORD_HEAD
    srr = mults = mri_Q = head_mri_t = None
    if not bem['is_sphere']:
        if bem['bem_method'] != 'linear collocation':
            raise RuntimeError('only linear collocation supported')
        mults = np.repeat(bem['source_mult'] / (4.0 * np.pi),
                          [len(s['rr']) for s in bem['surfs']])[np.newaxis, :]
        srr = np.concatenate([s['rr'] for s in bem['surfs']])

        # The dipole location and orientation must be transformed
        head_mri_t = bem['head_mri_t']
        mri_Q = apply_trans(bem['head_mri_t']['trans'], np.eye(3), False)

    if len(set(fwd_data['coil_types'])) != len(fwd_data['coil_types']):
        raise RuntimeError('Non-unique coil types found')
    compensators, solutions, csolutions = [], [], []
    for coil_type, coils, ccoils, info in zip(fwd_data['coil_types'],
                                              fwd_data['coils_list'],
                                              fwd_data['ccoils_list'],
                                              fwd_data['infos']):
        compensator = solution = csolution = None
        if len(coils) > 0:  # something to actually do
            if coil_type == 'meg':
                # Compose a compensation data set if necessary
                compensator = _make_ctf_comp_coils(info, coils)

            if not bem['is_sphere']:
                # multiply solution by "mults" here for simplicity
                if coil_type == 'meg':
                    # Field computation matrices for BEM
                    start = 'Composing the field computation matrix'
                    logger.info('\n' + start + '...')
                    solution = _bem_specify_coils(bem, coils, cf,
                                                  mults, n_jobs)
                    if compensator is not None:
                        logger.info(start + ' (compensation coils)...')
                        csolution = _bem_specify_coils(bem, ccoils, cf,
                                                       mults, n_jobs)
                else:
                    solution = _bem_specify_els(bem, coils, mults)
            else:
                solution = bem
                if coil_type == 'eeg':
                    logger.info('Using the equivalent source approach in the '
                                'homogeneous sphere for EEG')
        compensators.append(compensator)
        solutions.append(solution)
        csolutions.append(csolution)
    fun = _sphere_pot_or_field if bem['is_sphere'] else _bem_pot_or_field
    fwd_data.update(dict(srr=srr, mri_Q=mri_Q, head_mri_t=head_mri_t,
                         compensators=compensators, solutions=solutions,
                         csolutions=csolutions, fun=fun))


@verbose
def _compute_forwards_meeg(rr, fd, n_jobs, verbose=None):
    # Now, actually compute MEG and EEG solutions
    n_jobs = max(min(n_jobs, len(rr)), 1)
    Bs = list()
    # The dipole location and orientation must be transformed
    mri_rr = None
    if fd['head_mri_t'] is not None:
        mri_rr = apply_trans(fd['head_mri_t']['trans'], rr)
    mri_Q, srr, fun = fd['mri_Q'], fd['srr'], fd['fun']
    for ci in range(len(fd['coils_list'])):
        coils, ccoils = fd['coils_list'][ci], fd['ccoils_list'][ci]
        coil_type, compensator = fd['coil_types'][ci], fd['compensators'][ci]
        solution, csolution = fd['solutions'][ci], fd['csolutions'][ci]
        info = fd['infos'][ci]
        if len(coils) == 0:  # nothing to do
            Bs.append(np.zeros((3 * len(rr), 0)))
            continue

        # Do the actual calculation
        logger.info('Computing %s at %d source location%s '
                    '(free orientations)...'
                    % (coil_type.upper(), len(rr),
                       '' if len(rr) == 1 else 's'))
        B = fun(rr, mri_rr, mri_Q, coils, solution, srr,
                n_jobs, coil_type)

        # Compensate if needed (only done for MEG systems w/compensation)
        if compensator is not None:
            # Compute the field in the compensation coils
            work = fun(rr, mri_rr, mri_Q, ccoils, csolution, srr,
                       n_jobs, coil_type)
            # Combine solutions so we can do the compensation
            both = np.zeros((work.shape[0], B.shape[1] + work.shape[1]))
            picks = pick_types(info, meg=True, ref_meg=False)
            both[:, picks] = B
            picks = pick_types(info, meg=False, ref_meg=True)
            both[:, picks] = work
            B = np.dot(both, compensator.T)
        Bs.append(B)
    return Bs


@verbose
def _compute_forwards(rr, bem, coils_list, ccoils_list,
                      infos, coil_types, n_jobs, verbose=None):
    """Compute the MEG and EEG forward solutions

    This effectively combines compute_forward_meg and compute_forward_eeg
    from MNE-C.
    """
    # These are split into two steps to save (potentially) a lot of time
    # when e.g. dipole fitting
    fwd_data = dict(coils_list=coils_list, ccoils_list=ccoils_list,
                    infos=infos, coil_types=coil_types)
    _prep_field_computation(rr, bem, fwd_data, n_jobs)
    Bs = _compute_forwards_meeg(rr, fwd_data, n_jobs)
    return Bs
