# -*- coding: utf-8 -*-
# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#          Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD-3-Clause

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.
#
# Many of the idealized equations behind these calculations can be found in:
# 1) Realistic conductivity geometry model of the human head for interpretation
#        of neuromagnetic data. Hämäläinen and Sarvas, 1989. Specific to MNE
# 2) EEG and MEG: forward solutions for inverse methods. Mosher, Leahy, and
#        Lewis, 1999. Generalized discussion of forward solutions.

import numpy as np
from copy import deepcopy

from ..fixes import jit, bincount
from ..io.compensator import get_current_comp, make_compensator
from ..io.constants import FIFF, FWD
from ..io.pick import pick_types
from ..parallel import parallel_func
from ..surface import _project_onto_surface, _jit_cross
from ..transforms import apply_trans
from ..utils import logger, verbose, _pl, warn, fill_doc


# #############################################################################
# COIL SPECIFICATION AND FIELD COMPUTATION MATRIX

def _dup_coil_set(coils, coord_frame, t):
    """Make a duplicate."""
    if t is not None and coord_frame != t['from']:
        raise RuntimeError('transformation frame does not match the coil set')
    coils = deepcopy(coils)
    if t is not None:
        coord_frame = t['to']
        for coil in coils:
            for key in ('ex', 'ey', 'ez'):
                if key in coil:
                    coil[key] = apply_trans(t['trans'], coil[key], False)
            coil['r0'] = apply_trans(t['trans'], coil['r0'])
            coil['rmag'] = apply_trans(t['trans'], coil['rmag'])
            coil['cosmag'] = apply_trans(t['trans'], coil['cosmag'], False)
            coil['coord_frame'] = t['to']
    return coils, coord_frame


def _check_coil_frame(coils, coord_frame, bem):
    """Check to make sure the coils are in the correct coordinate frame."""
    if coord_frame != FIFF.FIFFV_COORD_MRI:
        if coord_frame == FIFF.FIFFV_COORD_HEAD:
            # Make a transformed duplicate
            coils, coord_Frame = _dup_coil_set(coils, coord_frame,
                                               bem['head_mri_t'])
        else:
            raise RuntimeError('Bad coil coordinate frame %s' % coord_frame)
    return coils, coord_frame


@fill_doc
def _lin_field_coeff(surf, mult, rmags, cosmags, ws, bins, n_jobs):
    """Parallel wrapper for _do_lin_field_coeff to compute linear coefficients.

    Parameters
    ----------
    surf : dict
        Dict containing information for one surface of the BEM
    mult : float
        Multiplier for particular BEM surface (Iso Skull Approach discussed in
        Mosher et al., 1999 and Hämäläinen and Sarvas, 1989 Section III?)
    rmag : ndarray, shape (n_integration_pts, 3)
        3D positions of MEG coil integration points (from coil['rmag'])
    cosmag : ndarray, shape (n_integration_pts, 3)
        Direction of the MEG coil integration points (from coil['cosmag'])
    ws : ndarray, shape (n_integration_pts,)
        Weights for MEG coil integration points
    bins : ndarray, shape (n_integration_points,)
        The sensor assignments for each rmag/cosmag/w.
    %(n_jobs)s

    Returns
    -------
    coeff : list
        Linear coefficients with lead fields for each BEM vertex on each sensor
        (?)
    """
    parallel, p_fun, _ = parallel_func(_do_lin_field_coeff, n_jobs)
    nas = np.array_split
    coeffs = parallel(p_fun(surf['rr'], t, tn, ta, rmags, cosmags, ws, bins)
                      for t, tn, ta in zip(nas(surf['tris'], n_jobs),
                                           nas(surf['tri_nn'], n_jobs),
                                           nas(surf['tri_area'], n_jobs)))
    return mult * np.sum(coeffs, axis=0)


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
    coeff = np.zeros((bins[-1] + 1, len(bem_rr)))
    w_cosmags = ws.reshape(-1, 1) * cosmags
    diff = rmags.reshape(rmags.shape[0], 1, rmags.shape[1]) - bem_rr
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

        c = np.empty((diff.shape[0], tri.shape[0], diff.shape[2]))
        _jit_cross(c, diff[:, tri], tri_nn)
        c *= w_cosmags.reshape(w_cosmags.shape[0], 1, w_cosmags.shape[1])
        for ti in range(3):
            x = np.sum(c[:, ti], axis=-1)
            x /= den[:, tri[ti]] / tri_area
            coeff[:, tri[ti]] += \
                bincount(bins, weights=x, minlength=bins[-1] + 1)
    return coeff


def _concatenate_coils(coils):
    """Concatenate MEG coil parameters."""
    rmags = np.concatenate([coil['rmag'] for coil in coils])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils])
    ws = np.concatenate([coil['w'] for coil in coils])
    n_int = np.array([len(coil['rmag']) for coil in coils])
    if n_int[-1] == 0:
        # We assume each sensor has at least one integration point,
        # which should be a safe assumption. But let's check it here, since
        # our code elsewhere relies on bins[-1] + 1 being the number of sensors
        raise RuntimeError('not supported')
    bins = np.repeat(np.arange(len(n_int)), n_int)
    return rmags, cosmags, ws, bins


@fill_doc
def _bem_specify_coils(bem, coils, coord_frame, mults, n_jobs):
    """Set up for computing the solution at a set of MEG coils.

    Parameters
    ----------
    bem : instance of ConductorModel
        BEM information
    coils : list of dict, len(n_MEG_sensors)
        MEG sensor information dicts
    coord_frame : int
        Class constant identifying coordinate frame
    mults : ndarray, shape (1, n_BEM_vertices)
        Multiplier for every vertex in BEM
    %(n_jobs)s

    Returns
    -------
    sol: ndarray, shape (n_MEG_sensors, n_BEM_vertices)
        MEG solution
    """
    # Make sure MEG coils are in MRI coordinate frame to match BEM coords
    coils, coord_frame = _check_coil_frame(coils, coord_frame, bem)

    # leaving this in in case we want to easily add in the future
    # if method != 'simple':  # in ['ferguson', 'urankar']:
    #     raise NotImplementedError

    # Compute the weighting factors to obtain the magnetic field in the linear
    # potential approximation

    # Process each of the surfaces
    rmags, cosmags, ws, bins = _triage_coils(coils)
    del coils
    lens = np.cumsum(np.r_[0, [len(s['rr']) for s in bem['surfs']]])
    sol = np.zeros((bins[-1] + 1, bem['solution'].shape[1]))

    lims = np.concatenate([np.arange(0, sol.shape[0], 100), [sol.shape[0]]])
    # Put through the bem (in channel-based chunks to save memory)
    for start, stop in zip(lims[:-1], lims[1:]):
        mask = np.logical_and(bins >= start, bins < stop)
        r, c, w, b = rmags[mask], cosmags[mask], ws[mask], bins[mask] - start
        # Compute coeffs for each surface, one at a time
        for o1, o2, surf, mult in zip(lens[:-1], lens[1:],
                                      bem['surfs'], bem['field_mult']):
            coeff = _lin_field_coeff(surf, mult, r, c, w, b, n_jobs)
            sol[start:stop] += np.dot(coeff, bem['solution'][o1:o2])
    sol *= mults
    return sol


def _bem_specify_els(bem, els, mults):
    """Set up for computing the solution at a set of EEG electrodes.

    Parameters
    ----------
    bem : instance of ConductorModel
        BEM information
    els : list of dict, len(n_EEG_sensors)
        List of EEG sensor information dicts
    mults: ndarray, shape (1, n_BEM_vertices)
        Multiplier for every vertex in BEM

    Returns
    -------
    sol : ndarray, shape (n_EEG_sensors, n_BEM_vertices)
        EEG solution
    """
    sol = np.zeros((len(els), bem['solution'].shape[1]))
    scalp = bem['surfs'][0]

    # Operate on all integration points for all electrodes (in MRI coords)
    rrs = np.concatenate([apply_trans(bem['head_mri_t']['trans'], el['rmag'])
                          for el in els], axis=0)
    ws = np.concatenate([el['w'] for el in els])
    tri_weights, tri_idx = _project_onto_surface(rrs, scalp)
    tri_weights *= ws[:, np.newaxis]
    weights = np.matmul(tri_weights[:, np.newaxis],
                        bem['solution'][scalp['tris'][tri_idx]])[:, 0]
    # there are way more vertices than electrodes generally, so let's iterate
    # over the electrodes
    edges = np.concatenate([[0], np.cumsum([len(el['w']) for el in els])])
    for ii, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        sol[ii] = weights[start:stop].sum(0)
    sol *= mults
    return sol


# #############################################################################
# COMPENSATION

def _make_ctf_comp_coils(info, coils):
    """Get the correct compensator for CTF coils."""
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

# def _bem_inf_pot(rd, Q, rp):
#     """The infinite medium potential in one direction. See Eq. (8) in
#     Mosher, 1999"""
#     NOTE: the (μ_0 / (4π) factor has been moved to _prep_field_communication
#     diff = rp - rd  # (Observation point position) - (Source position)
#     diff2 = np.sum(diff * diff, axis=1)  # Squared magnitude of diff
#     # (Dipole moment) dot (diff) / (magnitude ^ 3)
#     return np.sum(Q * diff, axis=1) / (diff2 * np.sqrt(diff2))


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
        diff_norm[diff_norm == 0] = 1.
        if mri_Q is not None:
            this_diff = np.dot(this_diff, mri_Q.T)
        this_diff /= diff_norm.reshape(-1, 1)
        diff[ri] = this_diff.T

    return diff

# This function has been refactored to process all points simultaneously
# def _bem_inf_field(rd, Q, rp, d):
# """Infinite-medium magnetic field. See (7) in Mosher, 1999"""
#     # Get vector from source to sensor integration point
#     diff = rp - rd
#     diff2 = np.sum(diff * diff, axis=1)  # Get magnitude of diff
#
#     # Compute cross product between diff and dipole to get magnetic field at
#     # integration point
#     x = fast_cross_3d(Q[np.newaxis, :], diff)
#
#     # Take magnetic field dotted by integration point normal to get magnetic
#     # field threading the current loop. Divide by R^3 (equivalently, R^2 * R)
#     return np.sum(x * d, axis=1) / (diff2 * np.sqrt(diff2))


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


@fill_doc
def _bem_pot_or_field(rr, mri_rr, mri_Q, coils, solution, bem_rr, n_jobs,
                      coil_type):
    """Calculate the magnetic field or electric potential forward solution.

    The code is very similar between EEG and MEG potentials, so combine them.
    This does the work of "fwd_comp_field" (which wraps to "fwd_bem_field")
    and "fwd_bem_pot_els" in MNE-C.

    Parameters
    ----------
    rr : ndarray, shape (n_dipoles, 3)
        3D dipole source positions
    mri_rr : ndarray, shape (n_dipoles, 3)
        3D source positions in MRI coordinates
    mri_Q :
        3x3 head -> MRI transform. I.e., head_mri_t.dot(np.eye(3))
    coils : list of dict, len(sensors)
        List of sensors where each element contains sensor specific information
    solution : ndarray, shape (n_sensors, n_BEM_rr)
        Comes from _bem_specify_coils
    bem_rr : ndarray, shape (n_BEM_vertices, 3)
        3D vertex positions for all surfaces in the BEM
    %(n_jobs)s
    coil_type : str
        'meg' or 'eeg'

    Returns
    -------
    B : ndarray, shape (n_dipoles * 3, n_sensors)
        Forward solution for a set of sensors
    """
    # Both MEG and EEG have the inifinite-medium potentials
    # This could be just vectorized, but eats too much memory, so instead we
    # reduce memory by chunking within _do_inf_pots and parallelize, too:
    parallel, p_fun, _ = parallel_func(_do_inf_pots, n_jobs)
    nas = np.array_split
    B = np.sum(parallel(p_fun(mri_rr, sr.copy(), np.ascontiguousarray(mri_Q),
                              np.array(sol))  # copy and contig
                        for sr, sol in zip(nas(bem_rr, n_jobs),
                                           nas(solution.T, n_jobs))), axis=0)
    # The copy()s above should make it so the whole objects don't need to be
    # pickled...

    # Only MEG coils are sensitive to the primary current distribution.
    if coil_type == 'meg':
        # Primary current contribution (can be calc. in coil/dipole coords)
        parallel, p_fun, _ = parallel_func(_do_prim_curr, n_jobs)
        pcc = np.concatenate(parallel(p_fun(r, coils)
                                      for r in nas(rr, n_jobs)), axis=0)
        B += pcc
        B *= _MAG_FACTOR
    return B


def _do_prim_curr(rr, coils):
    """Calculate primary currents in a set of MEG coils.

    See Mosher et al., 1999 Section II for discussion of primary vs. volume
    currents.

    Parameters
    ----------
    rr : ndarray, shape (n_dipoles, 3)
        3D dipole source positions in head coordinates
    coils : list of dict
        List of MEG coils where each element contains coil specific information

    Returns
    -------
    pc : ndarray, shape (n_sources, n_MEG_sensors)
        Primary current for set of MEG coils due to all sources
    """
    rmags, cosmags, ws, bins = _triage_coils(coils)
    n_coils = bins[-1] + 1
    del coils
    pc = np.empty((len(rr) * 3, n_coils))
    for start, stop in _rr_bounds(rr, chunk=1):
        pp = _bem_inf_fields(rr[start:stop], rmags, cosmags)
        pp *= ws
        pp.shape = (3 * (stop - start), -1)
        pc[3 * start:3 * stop] = [bincount(bins, this_pp, bins[-1] + 1)
                                  for this_pp in pp]
    return pc


def _rr_bounds(rr, chunk=200):
    # chunk data nicely
    bounds = np.concatenate([np.arange(0, len(rr), chunk), [len(rr)]])
    return zip(bounds[:-1], bounds[1:])


def _do_inf_pots(mri_rr, bem_rr, mri_Q, sol):
    """Calculate infinite potentials for MEG or EEG sensors using chunks.

    Parameters
    ----------
    mri_rr : ndarray, shape (n_dipoles, 3)
        3D dipole source positions in MRI coordinates
    bem_rr : ndarray, shape (n_BEM_vertices, 3)
        3D vertex positions for all surfaces in the BEM
    mri_Q :
        3x3 head -> MRI transform. I.e., head_mri_t.dot(np.eye(3))
    sol : ndarray, shape (n_sensors_subset, n_BEM_vertices_subset)
        Comes from _bem_specify_coils

    Returns
    -------
    B : ndarray, (n_dipoles * 3, n_sensors)
        Forward solution for sensors due to volume currents
    """
    # Doing work of 'fwd_bem_pot_calc' in MNE-C
    # The following code is equivalent to this, but saves memory
    # v0s = _bem_inf_pots(rr, bem_rr, Q)  # n_rr x 3 x n_bem_rr
    # v0s.shape = (len(rr) * 3, v0s.shape[2])
    # B = np.dot(v0s, sol)

    # We chunk the source mri_rr's in order to save memory
    B = np.empty((len(mri_rr) * 3, sol.shape[1]))
    for start, stop in _rr_bounds(mri_rr):
        # v0 in Hämäläinen et al., 1989 == v_inf in Mosher, et al., 1999
        v0s = _bem_inf_pots(mri_rr[start:stop], bem_rr, mri_Q)
        v0s = v0s.reshape(-1, v0s.shape[2])
        B[3 * start:3 * stop] = np.dot(v0s, sol)
    return B


# #############################################################################
# SPHERE COMPUTATION

def _sphere_pot_or_field(rr, mri_rr, mri_Q, coils, sphere, bem_rr,
                         n_jobs, coil_type):
    """Do potential or field for spherical model."""
    fun = _eeg_spherepot_coil if coil_type == 'eeg' else _sphere_field
    parallel, p_fun, _ = parallel_func(fun, n_jobs)
    B = np.concatenate(parallel(p_fun(r, coils, sphere)
                                for r in np.array_split(rr, n_jobs)))
    return B


def _sphere_field(rrs, coils, sphere):
    """Compute field for spherical model using Jukka Sarvas' field computation.

    Jukka Sarvas, "Basic mathematical and electromagnetic concepts of the
    biomagnetic inverse problem", Phys. Med. Biol. 1987, Vol. 32, 1, 11-22.

    The formulas have been manipulated for efficient computation
    by Matti Hämäläinen, February 1990
    """
    rmags, cosmags, ws, bins = _triage_coils(coils)
    return _do_sphere_field(rrs, rmags, cosmags, ws, bins, sphere['r0'])


@jit()
def _do_sphere_field(rrs, rmags, cosmags, ws, bins, r0):
    n_coils = bins[-1] + 1
    # Shift to the sphere model coordinates
    rrs = rrs - r0
    B = np.zeros((3 * len(rrs), n_coils))
    for ri in range(len(rrs)):
        rr = rrs[ri]
        # Check for a dipole at the origin
        if np.sqrt(np.dot(rr, rr)) <= 1e-10:
            continue
        this_poss = rmags - r0

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
        rr_ = rr.reshape(1, 3)
        v1 = np.empty((cosmags.shape[0], 3))
        _jit_cross(v1, rr_, cosmags)
        v2 = np.empty((cosmags.shape[0], 3))
        _jit_cross(v2, rr_, this_poss)
        xx = ((good * ws).reshape(-1, 1) *
              (v1 / F.reshape(-1, 1) + v2 * g.reshape(-1, 1)))
        for jj in range(3):
            zz = bincount(bins, xx[:, jj], n_coils)
            B[3 * ri + jj, :] = zz
    B *= _MAG_FACTOR
    return B


def _eeg_spherepot_coil(rrs, coils, sphere):
    """Calculate the EEG in the sphere model."""
    rmags, cosmags, ws, bins = _triage_coils(coils)
    n_coils = bins[-1] + 1
    del coils

    # Shift to the sphere model coordinates
    rrs = rrs - sphere['r0']

    B = np.zeros((3 * len(rrs), n_coils))
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
            zz = np.array([bincount(bins, x, bins[-1] + 1) for x in xx.T])
            B[3 * ri:3 * ri + 3, :] = zz
    # finishing by scaling by 1/(4*M_PI)
    B *= 0.25 / np.pi
    return B


def _triage_coils(coils):
    return coils if isinstance(coils, tuple) else _concatenate_coils(coils)


# #############################################################################
# MAGNETIC DIPOLE (e.g. CHPI)

_MIN_DIST_LIMIT = 1e-5


def _magnetic_dipole_field_vec(rrs, coils, too_close='raise'):
    rmags, cosmags, ws, bins = _triage_coils(coils)
    fwd, min_dist = _compute_mdfv(rrs, rmags, cosmags, ws, bins, too_close)
    if min_dist < _MIN_DIST_LIMIT:
        msg = 'Coil too close (dist = %g mm)' % (min_dist * 1000,)
        if too_close == 'raise':
            raise RuntimeError(msg)
        func = warn if too_close == 'warning' else logger.info
        func(msg)
    return fwd


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
        if min_dist < _MIN_DIST_LIMIT and too_close == 'raise':
            break
        t_ = np.sum(diff * cosmags, axis=1)
        t = t_.reshape(-1, 1)
        sum_ = ws2 * (3 * diff * t - dist2 * cosmags) / (dist2 * dist2 * dist)
        for ii in range(3):
            fwd[3 * ri + ii] = bincount(bins, sum_[:, ii], bins[-1] + 1)
    fwd *= _MAG_FACTOR
    return fwd, min_dist


# #############################################################################
# MAIN TRIAGING FUNCTION

@verbose
def _prep_field_computation(rr, bem, fwd_data, n_jobs, verbose=None):
    """Precompute and store some things that are used for both MEG and EEG.

    Calculation includes multiplication factors, coordinate transforms,
    compensations, and forward solutions. All are stored in modified fwd_data.

    Parameters
    ----------
    rr : ndarray, shape (n_dipoles, 3)
        3D dipole source positions in head coordinates
    bem : instance of ConductorModel
        Boundary Element Model information
    fwd_data : dict
        Dict containing sensor information. Gets updated here with BEM and
        sensor information for later forward calculations
    %(n_jobs)s
    %(verbose)s
    """
    bem_rr = mults = mri_Q = head_mri_t = None
    if not bem['is_sphere']:
        if bem['bem_method'] != FWD.BEM_LINEAR_COLL:
            raise RuntimeError('only linear collocation supported')
        # Store (and apply soon) μ_0/(4π) factor before source computations
        mults = np.repeat(bem['source_mult'] / (4.0 * np.pi),
                          [len(s['rr']) for s in bem['surfs']])[np.newaxis, :]
        # Get positions of BEM points for every surface
        bem_rr = np.concatenate([s['rr'] for s in bem['surfs']])

        # The dipole location and orientation must be transformed
        head_mri_t = bem['head_mri_t']
        mri_Q = bem['head_mri_t']['trans'][:3, :3].T

    # Compute solution and compensation for dif sensor types ('meg', 'eeg')
    if len(set(fwd_data['coil_types'])) != len(fwd_data['coil_types']):
        raise RuntimeError('Non-unique sensor types found')
    compensators, solutions, csolutions = [], [], []
    coils_list, ccoils_list = [], []
    for coil_type, coils, ccoils, info in zip(fwd_data['coil_types'],
                                              fwd_data['coils_list'],
                                              fwd_data['ccoils_list'],
                                              fwd_data['infos']):
        compensator = solution = csolution = None
        if len(coils) > 0:  # Only proceed if sensors exist
            if coil_type == 'meg':
                # Compose a compensation data set if necessary
                compensator = _make_ctf_comp_coils(info, coils)

            if not bem['is_sphere']:
                if coil_type == 'meg':
                    # MEG field computation matrices for BEM
                    start = 'Composing the field computation matrix'
                    logger.info('\n' + start + '...')
                    cf = FIFF.FIFFV_COORD_HEAD
                    # multiply solution by "mults" here for simplicity
                    solution = _bem_specify_coils(bem, coils, cf, mults,
                                                  n_jobs)
                    if compensator is not None:
                        logger.info(start + ' (compensation coils)...')
                        csolution = _bem_specify_coils(bem, ccoils, cf,
                                                       mults, n_jobs)
                else:
                    # Compute solution for EEG sensor
                    logger.info('Setting up for EEG...')
                    solution = _bem_specify_els(bem, coils, mults)
            else:
                solution = csolution = bem
                if coil_type == 'eeg':
                    logger.info('Using the equivalent source approach in the '
                                'homogeneous sphere for EEG')
            coils = _triage_coils(coils)
            if ccoils is not None and len(ccoils) > 0:
                ccoils = _triage_coils(ccoils)
        coils_list.append(coils)
        ccoils_list.append(ccoils)
        compensators.append(compensator)
        solutions.append(solution)
        csolutions.append(csolution)

    # Get appropriate forward physics function depending on sphere or BEM model
    fun = _sphere_pot_or_field if bem['is_sphere'] else _bem_pot_or_field

    # Update fwd_data with
    #    bem_rr (3D BEM vertex positions)
    #    mri_Q (3x3 Head->MRI coord transformation applied to identity matrix)
    #    head_mri_t (head->MRI coord transform dict)
    #    fun (_bem_pot_or_field if not 'sphere'; otherwise _sph_pot_or_field)
    #    solutions (len 2 list; [ndarray, shape (n_MEG_sens, n BEM vertices),
    #                            ndarray, shape (n_EEG_sens, n BEM vertices)]
    #    csolutions (compensation for solution)
    fwd_data.update(dict(bem_rr=bem_rr, mri_Q=mri_Q, head_mri_t=head_mri_t,
                         compensators=compensators, solutions=solutions,
                         csolutions=csolutions, fun=fun,
                         coils_list=coils_list, ccoils_list=ccoils_list))


@fill_doc
def _compute_forwards_meeg(rr, fd, n_jobs, silent=False):
    """Compute MEG and EEG forward solutions for all sensor types.

    Parameters
    ----------
    rr : ndarray, shape (n_dipoles, 3)
        3D dipole positions in head coordinates
    fd : dict
        Dict containing forward data after update in _prep_field_computation
    %(n_jobs)s
    silent : bool
        If True, don't emit logger.info.
        This saves time over ``verbose`` when this function is called a lot.

    Returns
    -------
    Bs : list
        Each element contains ndarray, shape (3 * n_dipoles, n_sensors) where
        n_sensors depends on which channel types are requested (MEG and/or EEG)
    """
    n_jobs = max(min(n_jobs, len(rr)), 1)
    Bs = list()
    # The dipole location and orientation must be transformed to mri coords
    mri_rr = None
    if fd['head_mri_t'] is not None:
        mri_rr = np.ascontiguousarray(
            apply_trans(fd['head_mri_t']['trans'], rr))
    mri_Q, bem_rr, fun = fd['mri_Q'], fd['bem_rr'], fd['fun']
    for ci in range(len(fd['coils_list'])):
        coils, ccoils = fd['coils_list'][ci], fd['ccoils_list'][ci]
        if len(coils) == 0:  # nothing to do
            Bs.append(np.zeros((3 * len(rr), 0)))
            continue

        coil_type, compensator = fd['coil_types'][ci], fd['compensators'][ci]
        solution, csolution = fd['solutions'][ci], fd['csolutions'][ci]
        info = fd['infos'][ci]

        # Do the actual forward calculation for a list MEG/EEG sensors
        if not silent:
            logger.info('Computing %s at %d source location%s '
                        '(free orientations)...'
                        % (coil_type.upper(), len(rr), _pl(rr)))
        # Calculate forward solution using spherical or BEM model
        B = fun(rr, mri_rr, mri_Q, coils, solution, bem_rr, n_jobs,
                coil_type)

        # Compensate if needed (only done for MEG systems w/compensation)
        if compensator is not None:
            # Compute the field in the compensation sensors
            work = fun(rr, mri_rr, mri_Q, ccoils, csolution, bem_rr,
                       n_jobs, coil_type)
            # Combine solutions so we can do the compensation
            both = np.zeros((work.shape[0], B.shape[1] + work.shape[1]))
            picks = pick_types(info, meg=True, ref_meg=False, exclude=[])
            both[:, picks] = B
            picks = pick_types(info, meg=False, ref_meg=True, exclude=[])
            both[:, picks] = work
            B = np.dot(both, compensator.T)
        Bs.append(B)
    return Bs


@verbose
def _compute_forwards(rr, bem, coils_list, ccoils_list, infos, coil_types,
                      n_jobs, verbose=None):
    """Compute the MEG and EEG forward solutions.

    This effectively combines compute_forward_meg and compute_forward_eeg
    from MNE-C.

    Parameters
    ----------
    rr : ndarray, shape (n_sources, 3)
        3D dipole in head coordinates
    bem : instance of ConductorModel
        Boundary Element Model information for all surfaces
    coils_list : list
        List of MEG and/or EEG sensor information dicts
    ccoils_list : list
        Optional list of MEG compensation information
    coil_types : list of str
        Sensor types. May contain 'meg' and/or 'eeg'
    %(n_jobs)s
    infos : list, len(2)
        infos[0] is MEG info, infos[1] is EEG info

    Returns
    -------
    Bs : list of ndarray
        Each element contains ndarray, shape (3 * n_dipoles, n_sensors) where
        n_sensors depends on which channel types are requested (MEG and/or EEG)
    """
    # Split calculation into two steps to save (potentially) a lot of time
    # when e.g. dipole fitting
    fwd_data = dict(coils_list=coils_list, ccoils_list=ccoils_list,
                    infos=infos, coil_types=coil_types)
    _prep_field_computation(rr, bem, fwd_data, n_jobs)
    Bs = _compute_forwards_meeg(rr, fwd_data, n_jobs)
    return Bs
