# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import sys
import os
import os.path as op
import shutil
import numpy as np
from scipy import linalg

from .fixes import partial
from .utils import (verbose, logger, run_subprocess, deprecated,
                    get_subjects_dir)
from .io.constants import FIFF
from .io.write import (start_file, start_block, write_float, write_int,
                       write_float_matrix, write_int_matrix, end_block,
                       end_file)
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.open import fiff_open
from .externals.six import string_types


# ############################################################################
# Compute BEM solution

# define VEC_DIFF(from,to,diff) {\
# (diff)[X] = (to)[X] - (from)[X];\

# The following approach is based on:
#
# de Munck JC: "A linear discretization of the volume conductor boundary
# integral equation using analytically integrated elements",
# IEEE Trans Biomed Eng. 1992 39(9) : 986 - 990
#


def _calc_beta(rk, rk_norm, rk1, rk1_norm):
    """These coefficients are used to calculate the magic vector omega"""
    rkk1 = rk1[0] - rk[0]
    size = np.sqrt(np.dot(rkk1, rkk1))
    rkk1 /= size
    num = rk_norm + np.dot(rk, rkk1)
    den = rk1_norm + np.dot(rk1, rkk1)
    res = np.log(num / den) / size
    return res


def _lin_pot_coeff(fros, tri_rr, tri_nn, tri_area):
    """The linear potential matrix element computations"""
    from .source_space import _fast_cross_nd_sum
    omega = np.zeros((len(fros), 3))

    # we replicate a little bit of the _get_solids code here for speed
    v1 = tri_rr[np.newaxis, 0, :] - fros
    v2 = tri_rr[np.newaxis, 1, :] - fros
    v3 = tri_rr[np.newaxis, 2, :] - fros
    triples = _fast_cross_nd_sum(v1, v2, v3)
    l1 = np.sqrt(np.sum(v1 * v1, axis=1))
    l2 = np.sqrt(np.sum(v2 * v2, axis=1))
    l3 = np.sqrt(np.sum(v3 * v3, axis=1))
    ss = (l1 * l2 * l3 +
          np.sum(v1 * v2, axis=1) * l3 +
          np.sum(v1 * v3, axis=1) * l2 +
          np.sum(v2 * v3, axis=1) * l1)
    solids = np.arctan2(triples, ss)

    # We *could* subselect the good points from v1, v2, v3, triples, solids,
    # l1, l2, and l3, but there are *very* few bad points. So instead we do
    # some unnecessary calculations, and then omit them from the final
    # solution. These three lines ensure we don't get invalid values in
    # _calc_beta.
    bad_mask = np.abs(solids) < np.pi / 1e6
    l1[bad_mask] = 1.
    l2[bad_mask] = 1.
    l3[bad_mask] = 1.

    # Calculate the magic vector vec_omega
    beta = [_calc_beta(v1, l1, v2, l2)[:, np.newaxis],
            _calc_beta(v2, l2, v3, l3)[:, np.newaxis],
            _calc_beta(v3, l3, v1, l1)[:, np.newaxis]]
    vec_omega = (beta[2] - beta[0]) * v1
    vec_omega += (beta[0] - beta[1]) * v2
    vec_omega += (beta[1] - beta[2]) * v3

    area2 = 2.0 * tri_area
    n2 = 1.0 / (area2 * area2)
    # leave omega = 0 otherwise
    # Put it all together...
    yys = [v1, v2, v3]
    idx = [0, 1, 2, 0, 2]
    for k in range(3):
        diff = yys[idx[k - 1]] - yys[idx[k + 1]]
        zdots = _fast_cross_nd_sum(yys[idx[k + 1]], yys[idx[k - 1]], tri_nn)
        omega[:, k] = -n2 * (area2 * zdots * 2. * solids -
                             triples * (diff * vec_omega).sum(axis=-1))
    # omit the bad points from the solution
    omega[bad_mask] = 0.
    return omega


def _correct_auto_elements(surf, mat):
    """Improve auto-element approximation..."""
    pi2 = 2.0 * np.pi
    tris_flat = surf['tris'].ravel()
    misses = pi2 - mat.sum(axis=1)
    for j, miss in enumerate(misses):
        # How much is missing?
        n_memb = len(surf['neighbor_tri'][j])
        # The node itself receives one half
        mat[j, j] = miss / 2.0
        # The rest is divided evenly among the member nodes...
        miss /= (4.0 * n_memb)
        members = np.where(j == tris_flat)[0]
        mods = members % 3
        offsets = np.array([[1, 2], [-1, 1], [-1, -2]])
        tri_1 = members + offsets[mods, 0]
        tri_2 = members + offsets[mods, 1]
        for t1, t2 in zip(tri_1, tri_2):
            mat[j, tris_flat[t1]] += miss
            mat[j, tris_flat[t2]] += miss
    return


def _fwd_bem_lin_pot_coeff(surfs):
    """Calculate the coefficients for linear collocation approach"""
    # taken from fwd_bem_linear_collocation.c
    nps = [surf['np'] for surf in surfs]
    np_tot = sum(nps)
    coeff = np.zeros((np_tot, np_tot))
    offsets = np.cumsum(np.concatenate(([0], nps)))
    for si_1, surf1 in enumerate(surfs):
        rr_ord = np.arange(nps[si_1])
        for si_2, surf2 in enumerate(surfs):
            logger.info("        %s (%d) -> %s (%d) ..." %
                        (_bem_explain_surface(surf1['id']), nps[si_1],
                         _bem_explain_surface(surf2['id']), nps[si_2]))
            tri_rr = surf2['rr'][surf2['tris']]
            tri_nn = surf2['tri_nn']
            tri_area = surf2['tri_area']
            submat = coeff[offsets[si_1]:offsets[si_1 + 1],
                           offsets[si_2]:offsets[si_2 + 1]]  # view
            for k in range(surf2['ntri']):
                tri = surf2['tris'][k]
                if si_1 == si_2:
                    skip_idx = ((rr_ord == tri[0]) |
                                (rr_ord == tri[1]) |
                                (rr_ord == tri[2]))
                else:
                    skip_idx = list()
                # No contribution from a triangle that
                # this vertex belongs to
                # if sidx1 == sidx2 and (tri == j).any():
                #     continue
                # Otherwise do the hard job
                coeffs = _lin_pot_coeff(surf1['rr'], tri_rr[k], tri_nn[k],
                                        tri_area[k])
                coeffs[skip_idx] = 0.
                submat[:, tri] -= coeffs
            if si_1 == si_2:
                _correct_auto_elements(surf1, submat)
    return coeff


def _fwd_bem_multi_solution(solids, gamma, nps):
    """Do multi surface solution

      * Invert I - solids/(2*M_PI)
      * Take deflation into account
      * The matrix is destroyed after inversion
      * This is the general multilayer case

    """
    pi2 = 1.0 / (2 * np.pi)
    n_tot = np.sum(nps)
    assert solids.shape == (n_tot, n_tot)
    nsurf = len(nps)
    defl = 1.0 / n_tot
    # Modify the matrix
    offsets = np.cumsum(np.concatenate(([0], nps)))
    for si_1 in range(nsurf):
        for si_2 in range(nsurf):
            mult = pi2 if gamma is None else pi2 * gamma[si_1, si_2]
            slice_j = slice(offsets[si_1], offsets[si_1 + 1])
            slice_k = slice(offsets[si_2], offsets[si_2 + 1])
            solids[slice_j, slice_k] = defl - solids[slice_j, slice_k] * mult
    solids += np.eye(n_tot)
    return linalg.inv(solids, overwrite_a=True)


def _fwd_bem_homog_solution(solids, nps):
    """Helper to make a homogeneous solution"""
    return _fwd_bem_multi_solution(solids, None, nps)


def _fwd_bem_ip_modify_solution(solution, ip_solution, ip_mult, n_tri):
    """Modify the solution according to the IP approach"""
    n_last = n_tri[-1]
    mult = (1.0 + ip_mult) / ip_mult

    logger.info('        Combining...')
    offsets = np.cumsum(np.concatenate(([0], n_tri)))
    for si in range(len(n_tri)):
        # Pick the correct submatrix (right column) and multiply
        sub = solution[offsets[si]:offsets[si + 1], np.sum(n_tri[:-1]):]
        # Multiply
        sub -= 2 * np.dot(sub, ip_solution)

    # The lower right corner is a special case
    sub[-n_last:, -n_last:] += mult * ip_solution

    # Final scaling
    logger.info('        Scaling...')
    solution *= ip_mult
    return


def _fwd_bem_linear_collocation_solution(m):
    """Compute the linear collocation potential solution"""
    # first, add surface geometries
    from .surface import _complete_surface_info
    for surf in m['surfs']:
        _complete_surface_info(surf, verbose=False)

    logger.info('Computing the linear collocation solution...')
    logger.info('    Matrix coefficients...')
    coeff = _fwd_bem_lin_pot_coeff(m['surfs'])
    m['nsol'] = len(coeff)
    logger.info("    Inverting the coefficient matrix...")
    nps = [surf['np'] for surf in m['surfs']]
    m['solution'] = _fwd_bem_multi_solution(coeff, m['gamma'], nps)
    if len(m['surfs']) == 3:
        ip_mult = m['sigma'][1] / m['sigma'][2]
        if ip_mult <= FIFF.FWD_BEM_IP_APPROACH_LIMIT:
            logger.info('IP approach required...')
            logger.info('    Matrix coefficients (homog)...')
            coeff = _fwd_bem_lin_pot_coeff([m['surfs'][-1]])
            logger.info('    Inverting the coefficient matrix (homog)...')
            ip_solution = _fwd_bem_homog_solution(coeff,
                                                  [m['surfs'][-1]['np']])
            logger.info('    Modify the original solution to incorporate '
                        'IP approach...')
            _fwd_bem_ip_modify_solution(m['solution'], ip_solution, ip_mult,
                                        nps)
    m['bem_method'] = FIFF.FWD_BEM_LINEAR_COLL
    logger.info("Solution ready.")


@verbose
def make_bem_solution(surfs, verbose=None):
    """Create a BEM solution using the linear collocation approach

    Parameters
    ----------
    surfs : list of dict
        The BEM surfaces to use.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    bem : dict
        The BEM solution.

    Notes
    -----
    .. versionadded:: 0.10.0

    See Also
    --------
    make_bem_model
    read_bem_surfaces
    write_bem_surfaces
    read_bem_solution
    write_bem_solution
    """
    logger.info('Approximation method : Linear collocation\n')
    if isinstance(surfs, string_types):
        # Load the surfaces
        logger.info('Loading surfaces...')
        surfs = read_bem_surfaces(surfs)
    bem = dict(surfs=surfs)
    _add_gamma_multipliers(bem)
    if len(bem['surfs']) == 3:
        logger.info('Three-layer model surfaces loaded.')
    elif len(bem['surfs']) == 1:
        logger.info('Homogeneous model surface loaded.')
    else:
        raise RuntimeError('Only 1- or 3-layer BEM computations supported')
    _fwd_bem_linear_collocation_solution(bem)
    logger.info('BEM geometry computations complete.')
    return bem


# ############################################################################
# Make BEM model

def _ico_downsample(surf, dest_grade):
    """Downsample the surface if isomorphic to a subdivided icosahedron"""
    from .surface import _get_ico_surface
    n_tri = surf['ntri']
    found = -1
    bad_msg = ("A surface with %d triangles cannot be isomorphic with a "
               "subdivided icosahedron." % surf['ntri'])
    if n_tri % 20 != 0:
        raise RuntimeError(bad_msg)
    n_tri = n_tri // 20
    found = int(round(np.log(n_tri) / np.log(4)))
    if n_tri != 4 ** found:
        raise RuntimeError(bad_msg)
    del n_tri

    if dest_grade > found:
        raise RuntimeError('For this surface, decimation grade should be %d '
                           'or less, not %s.' % (found, dest_grade))

    source = _get_ico_surface(found)
    dest = _get_ico_surface(dest_grade, patch_stats=True)
    del dest['tri_cent']
    del dest['tri_nn']
    del dest['neighbor_tri']
    del dest['tri_area']
    if not np.array_equal(source['tris'], surf['tris']):
        raise RuntimeError('The source surface has a matching number of '
                           'triangles but ordering is wrong')
    logger.info('Going from %dth to %dth subdivision of an icosahedron '
                '(n_tri: %d -> %d)' % (found, dest_grade, surf['ntri'],
                                       dest['ntri']))
    # Find the mapping
    dest['rr'] = surf['rr'][_get_ico_map(source, dest)]
    return dest


def _get_ico_map(fro, to):
    """Helper to get a mapping between ico surfaces"""
    from .surface import _compute_nearest
    nearest, dists = _compute_nearest(fro['rr'], to['rr'], return_dists=True)
    n_bads = (dists > 5e-3).sum()
    if n_bads > 0:
        raise RuntimeError('No matching vertex for %d destination vertices'
                           % (n_bads))
    return nearest


def _order_surfaces(surfs):
    """Reorder the surfaces"""
    if len(surfs) != 3:
        return surfs
    # we have three surfaces
    surf_order = [FIFF.FIFFV_BEM_SURF_ID_HEAD,
                  FIFF.FIFFV_BEM_SURF_ID_SKULL,
                  FIFF.FIFFV_BEM_SURF_ID_BRAIN]
    ids = np.array([surf['id'] for surf in surfs])
    if set(ids) != set(surf_order):
        raise RuntimeError('bad surface ids: %s' % ids)
    order = [np.where(ids == id_)[0][0] for id_ in surf_order]
    surfs = [surfs[idx] for idx in order]
    return surfs


def _assert_complete_surface(surf):
    """Check the sum of solid angles as seen from inside"""
    # from surface_checks.c
    from .source_space import _get_solids
    tot_angle = 0.
    # Center of mass....
    cm = surf['rr'].mean(axis=0)
    logger.info('%s CM is %6.2f %6.2f %6.2f mm' %
                (_surf_name[surf['id']],
                 1000 * cm[0], 1000 * cm[1], 1000 * cm[2]))
    tot_angle = _get_solids(surf['rr'][surf['tris']], cm[np.newaxis, :])[0]
    if np.abs(tot_angle / (2 * np.pi) - 1.0) > 1e-5:
        raise RuntimeError('Surface %s is not complete (sum of solid angles '
                           '= %g * 4*PI instead).' %
                           (_surf_name[surf['id']], tot_angle))


_surf_name = {
    FIFF.FIFFV_BEM_SURF_ID_HEAD: 'outer skin ',
    FIFF.FIFFV_BEM_SURF_ID_SKULL: 'outer skull',
    FIFF.FIFFV_BEM_SURF_ID_BRAIN: 'inner skull',
    FIFF.FIFFV_BEM_SURF_ID_UNKNOWN: 'unknown    ',
}


def _assert_inside(fro, to):
    """Helper to check one set of points is inside a surface"""
    # this is "is_inside" in surface_checks.c
    from .source_space import _get_solids
    tot_angle = _get_solids(to['rr'][to['tris']], fro['rr'])
    if (np.abs(tot_angle / (2 * np.pi) - 1.0) > 1e-5).any():
        raise RuntimeError('Surface %s is not completely inside surface %s'
                           % (_surf_name[fro['id']], _surf_name[to['id']]))


def _check_surfaces(surfs):
    """Check that the surfaces are complete and non-intersecting"""
    for surf in surfs:
        _assert_complete_surface(surf)
    # Then check the topology
    for surf_1, surf_2 in zip(surfs[:-1], surfs[1:]):
        logger.info('Checking that %s surface is inside %s surface...' %
                    (_surf_name[surf_2['id']], _surf_name[surf_1['id']]))
        _assert_inside(surf_2, surf_1)


def _check_surface_size(surf):
    """Check that the coordinate limits are reasonable"""
    sizes = surf['rr'].max(axis=0) - surf['rr'].min(axis=0)
    if (sizes < 0.05).any():
        raise RuntimeError('Dimensions of the surface %s seem too small '
                           '(%9.5f mm). Maybe the the unit of measure is '
                           'meters instead of mm' %
                           (_surf_name[surf['id']], 1000 * sizes.min()))


def _check_thicknesses(surfs):
    """How close are we?"""
    from .surface import _compute_nearest
    for surf_1, surf_2 in zip(surfs[:-1], surfs[1:]):
        min_dist = _compute_nearest(surf_1['rr'], surf_2['rr'],
                                    return_dists=True)[0]
        min_dist = min_dist.min()
        logger.info('Checking distance between %s and %s surfaces...' %
                    (_surf_name[surf_1['id']], _surf_name[surf_2['id']]))
        logger.info('Minimum distance between the %s and %s surfaces is '
                    'approximately %6.1f mm' %
                    (_surf_name[surf_1['id']], _surf_name[surf_2['id']],
                     1000 * min_dist))


def _surfaces_to_bem(fname_surfs, ids, sigmas, ico=None):
    """Convert surfaces to a BEM
    """
    from .surface import _read_surface_geom
    # equivalent of mne_surf2bem
    surfs = list()
    assert len(fname_surfs) in (1, 3)
    for fname in fname_surfs:
        surfs.append(_read_surface_geom(fname, patch_stats=False,
                                        verbose=False))
        surfs[-1]['rr'] /= 1000.
    # Downsampling if the surface is isomorphic with a subdivided icosahedron
    if ico is not None:
        for si, surf in enumerate(surfs):
            surfs[si] = _ico_downsample(surf, ico)
    for surf, id_ in zip(surfs, ids):
        surf['id'] = id_

    # Shifting surfaces is not implemented here

    # Order the surfaces for the benefit of the topology checks
    for surf, sigma in zip(surfs, sigmas):
        surf['sigma'] = sigma
    surfs = _order_surfaces(surfs)

    # Check topology as best we can
    _check_surfaces(surfs)
    for surf in surfs:
        _check_surface_size(surf)
    _check_thicknesses(surfs)
    logger.info('Surfaces passed the basic topology checks.')
    return surfs


@verbose
def make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                   subjects_dir=None, verbose=None):
    """Create a BEM model for a subject

    Parameters
    ----------
    subject : str
        The subject.
    ico : int | None
        The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280.
    conductivity : array of int, shape (3,) or (1,)
        The conductivities to use for each shell. Should be a single element
        for a one-layer model, or three elements for a three-layer model.
        Defaults to ``[0.3, 0.006, 0.3]``. The MNE-C default for a
        single-layer model would be ``[0.3]``.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    surfaces : list of dict
        The BEM surfaces.

    Notes
    -----
    .. versionadded:: 0.10.0

    See Also
    --------
    make_bem_solution
    make_sphere_model
    read_bem_surfaces
    write_bem_surfaces
    """
    conductivity = np.array(conductivity, float)
    if conductivity.ndim != 1 or conductivity.size not in (1, 3):
        raise ValueError('conductivity must be 1D array-like with 1 or 3 '
                         'elements')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subject_dir = op.join(subjects_dir, subject)
    bem_dir = op.join(subject_dir, 'bem')
    inner_skull = op.join(bem_dir, 'inner_skull.surf')
    outer_skull = op.join(bem_dir, 'outer_skull.surf')
    outer_skin = op.join(bem_dir, 'outer_skin.surf')
    surfaces = [inner_skull, outer_skull, outer_skin]
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    logger.info('Creating the BEM geometry...')
    if len(conductivity) == 1:
        surfaces = surfaces[:1]
        ids = ids[:1]
    surfaces = _surfaces_to_bem(surfaces, ids, conductivity, ico)
    logger.info('Complete.\n')
    return surfaces


# ############################################################################
# Compute EEG sphere model

def _fwd_eeg_get_multi_sphere_model_coeffs(m, n_terms):
    """Get the model depended weighting factor for n"""
    nlayer = len(m['layers'])
    if nlayer in (0, 1):
        return 1.

    # Initialize the arrays
    c1 = np.zeros(nlayer - 1)
    c2 = np.zeros(nlayer - 1)
    cr = np.zeros(nlayer - 1)
    cr_mult = np.zeros(nlayer - 1)
    for k in range(nlayer - 1):
        c1[k] = m['layers'][k]['sigma'] / m['layers'][k + 1]['sigma']
        c2[k] = c1[k] - 1.0
        cr_mult[k] = m['layers'][k]['rel_rad']
        cr[k] = cr_mult[k]
        cr_mult[k] *= cr_mult[k]

    coeffs = np.zeros(n_terms - 1)
    for n in range(1, n_terms):
        # Increment the radius coefficients
        for k in range(nlayer - 1):
            cr[k] *= cr_mult[k]

        # Multiply the matrices
        M = np.eye(2)
        n1 = n + 1.0
        for k in range(nlayer - 2, -1, -1):
            M = np.dot([[n + n1 * c1[k], n1 * c2[k] / cr[k]],
                        [n * c2[k] * cr[k], n1 + n * c1[k]]], M)
        num = n * (2.0 * n + 1.0) ** (nlayer - 1)
        coeffs[n - 1] = num / (n * M[1, 1] + n1 * M[1, 0])
    return coeffs


def _compose_linear_fitting_data(mu, u):
    # y is the data to be fitted (nterms-1 x 1)
    # M is the model matrix      (nterms-1 x nfit-1)
    for k in range(u['nterms'] - 1):
        k1 = k + 1
        mu1n = np.power(mu[0], k1)
        u['y'][k] = u['w'][k] * (u['fn'][k1] - mu1n * u['fn'][0])
        for p in range(u['nfit'] - 1):
            u['M'][k][p] = u['w'][k] * (np.power(mu[p + 1], k1) - mu1n)


def _compute_linear_parameters(mu, u):
    """Compute the best-fitting linear parameters"""
    _compose_linear_fitting_data(mu, u)
    uu, sing, vv = linalg.svd(u['M'], full_matrices=False)

    # Compute the residuals
    u['resi'] = u['y'].copy()

    vec = np.empty(u['nfit'] - 1)
    for p in range(u['nfit'] - 1):
        vec[p] = np.dot(uu[:, p], u['y'])
        for k in range(u['nterms'] - 1):
            u['resi'][k] -= uu[k, p] * vec[p]
        vec[p] = vec[p] / sing[p]

    lambda_ = np.zeros(u['nfit'])
    for p in range(u['nfit'] - 1):
        sum_ = 0.
        for q in range(u['nfit'] - 1):
            sum_ += vv[q, p] * vec[q]
        lambda_[p + 1] = sum_
    lambda_[0] = u['fn'][0] - np.sum(lambda_[1:])
    rv = np.dot(u['resi'], u['resi']) / np.dot(u['y'], u['y'])
    return rv, lambda_


def _one_step(mu, u):
    """Evaluate the residual sum of squares fit for one set of mu values"""
    if np.abs(mu).max() > 1.0:
        return 1.0

    # Compose the data for the linear fitting, compute SVD, then residuals
    _compose_linear_fitting_data(mu, u)
    u['uu'], u['sing'], u['vv'] = linalg.svd(u['M'])
    u['resi'][:] = u['y'][:]
    for p in range(u['nfit'] - 1):
        dot = np.dot(u['uu'][p], u['y'])
        for k in range(u['nterms'] - 1):
            u['resi'][k] = u['resi'][k] - u['uu'][p, k] * dot

    # Return their sum of squares
    return np.dot(u['resi'], u['resi'])


def _fwd_eeg_fit_berg_scherg(m, nterms, nfit):
    """Fit the Berg-Scherg equivalent spherical model dipole parameters"""
    from scipy.optimize import fmin_cobyla
    assert nfit >= 2
    u = dict(y=np.zeros(nterms - 1), resi=np.zeros(nterms - 1),
             nfit=nfit, nterms=nterms, M=np.zeros((nterms - 1, nfit - 1)))

    # (1) Calculate the coefficients of the true expansion
    u['fn'] = _fwd_eeg_get_multi_sphere_model_coeffs(m, nterms + 1)

    # (2) Calculate the weighting
    f = (min([layer['rad'] for layer in m['layers']]) /
         max([layer['rad'] for layer in m['layers']]))

    # correct weighting
    k = np.arange(1, nterms + 1)
    u['w'] = np.sqrt((2.0 * k + 1) * (3.0 * k + 1.0) /
                     k) * np.power(f, (k - 1.0))
    u['w'][-1] = 0

    # Do the nonlinear minimization, constraining mu to the interval [-1, +1]
    mu_0 = np.random.RandomState(0).rand(nfit) * f
    fun = partial(_one_step, u=u)
    max_ = 1. - 2e-4  # adjust for fmin_cobyla "catol" that not all scipy have
    cons = [(lambda x: max_ - np.abs(x[ii])) for ii in range(nfit)]
    mu = fmin_cobyla(fun, mu_0, cons, rhobeg=0.5, rhoend=5e-3, disp=0)

    # (6) Do the final step: calculation of the linear parameters
    rv, lambda_ = _compute_linear_parameters(mu, u)
    order = np.argsort(mu)[::-1]
    mu, lambda_ = mu[order], lambda_[order]  # sort: largest mu first

    m['mu'] = mu
    # This division takes into account the actual conductivities
    m['lambda'] = lambda_ / m['layers'][-1]['sigma']
    m['nfit'] = nfit
    return rv


@verbose
def make_sphere_model(r0=(0., 0., 0.04), head_radius=0.09, info=None,
                      relative_radii=(0.90, 0.92, 0.97, 1.0),
                      sigmas=(0.33, 1.0, 0.004, 0.33), verbose=None):
    """Create a spherical model for forward solution calculation

    Parameters
    ----------
    r0 : array-like | str
        Head center to use (in head coordinates). If 'auto', the head
        center will be calculated from the digitization points in info.
    head_radius : float | str | None
        If float, compute spherical shells for EEG using the given radius.
        If 'auto', estimate an approriate radius from the dig points in Info,
        If None, exclude shells.
    info : instance of mne.io.meas_info.Info | None
        Measurement info. Only needed if ``r0`` or ``head_radius`` are
        ``'auto'``.
    relative_radii : array-like
        Relative radii for the spherical shells.
    sigmas : array-like
        Sigma values for the spherical shells.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    sphere : dict
        A spherical BEM.

    Notes
    -----
    .. versionadded:: 0.9.0

    See Also
    --------
    make_bem_model
    make_bem_solution
    """
    for name in ('r0', 'head_radius'):
        param = locals()[name]
        if isinstance(param, string_types):
            if param != 'auto':
                raise ValueError('%s, if str, must be "auto" not "%s"'
                                 % (name, param))

    if (isinstance(r0, string_types) and r0 == 'auto') or \
       (isinstance(head_radius, string_types) and head_radius == 'auto'):
        if info is None:
            raise ValueError('Info must not be None for auto mode')
        head_radius_fit, r0_fit = fit_sphere_to_headshape(info)[:2]
        if isinstance(r0, string_types):
            r0 = r0_fit / 1000.
        if isinstance(head_radius, string_types):
            head_radius = head_radius_fit / 1000.
    sphere = dict(r0=np.array(r0), is_sphere=True,
                  coord_frame=FIFF.FIFFV_COORD_HEAD)
    sphere['layers'] = list()
    if head_radius is not None:
        # Eventually these could be configurable...
        relative_radii = np.array(relative_radii, float)
        sigmas = np.array(sigmas, float)
        order = np.argsort(relative_radii)
        relative_radii = relative_radii[order]
        sigmas = sigmas[order]
        layers = sphere['layers']
        for rel_rad, sig in zip(relative_radii, sigmas):
            # sort layers by (relative) radius, and scale radii
            layer = dict(rad=rel_rad, sigma=sig)
            layer['rel_rad'] = layer['rad'] = rel_rad
            layers.append(layer)

        # scale the radii
        R = layers[-1]['rad']
        rR = layers[-1]['rel_rad']
        for layer in layers:
            layer['rad'] /= R
            layer['rel_rad'] /= rR

        #
        # Setup the EEG sphere model calculations
        #

        # Scale the relative radii
        for k in range(len(relative_radii)):
            layers[k]['rad'] = (head_radius * layers[k]['rel_rad'])
        rv = _fwd_eeg_fit_berg_scherg(sphere, 200, 3)
        logger.info('\nEquiv. model fitting -> RV = %g %%' % (100 * rv))
        for k in range(3):
            logger.info('mu%d = %g    lambda%d = %g'
                        % (k + 1, sphere['mu'][k], k + 1,
                           layers[-1]['sigma'] * sphere['lambda'][k]))
        logger.info('Set up EEG sphere model with scalp radius %7.1f mm\n'
                    % (1000 * head_radius,))
    return sphere


# #############################################################################
# Helpers

@verbose
def fit_sphere_to_headshape(info, dig_kinds=(FIFF.FIFFV_POINT_EXTRA,),
                            verbose=None):
    """Fit a sphere to the headshape points to determine head center

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        Measurement info.
    dig_kinds : tuple of int
        Kind of digitization points to use in the fitting. These can be
        any kind defined in io.constants.FIFF:
            FIFFV_POINT_CARDINAL
            FIFFV_POINT_HPI
            FIFFV_POINT_EEG
            FIFFV_POINT_ECG
            FIFFV_POINT_EXTRA
        Defaults to (FIFFV_POINT_EXTRA,).

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    radius : float
        Sphere radius in mm.
    origin_head: ndarray, shape (3,)
        Head center in head coordinates (mm).
    origin_device: ndarray, shape (3,)
        Head center in device coordinates (mm).
    """
    # get head digization points of the specified kind
    hsp = [p['r'] for p in info['dig'] if p['kind'] in dig_kinds]

    # exclude some frontal points (nose etc.)
    hsp = [p for p in hsp if not (p[2] < 0 and p[1] > 0)]

    if len(hsp) == 0:
        raise ValueError('No head digitization points of the specified '
                         'kinds (%s) found.' % dig_kinds)

    hsp = 1e3 * np.array(hsp)

    radius, origin_head = _fit_sphere(hsp, disp=False)
    # compute origin in device coordinates
    trans = info['dev_head_t']
    if trans['from'] != FIFF.FIFFV_COORD_DEVICE \
            or trans['to'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('device to head transform not found')

    head_to_dev = linalg.inv(trans['trans'])
    origin_device = 1e3 * np.dot(head_to_dev,
                                 np.r_[1e-3 * origin_head, 1.0])[:3]

    logger.info('Fitted sphere radius:'.ljust(30) + '%0.1f mm' % radius)
    logger.info('Origin head coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(origin_head))
    logger.info('Origin device coordinates:'.ljust(30) +
                '%0.1f %0.1f %0.1f mm' % tuple(origin_device))

    return radius, origin_head, origin_device


def _fit_sphere(points, disp='auto'):
    """Aux function to fit points to a sphere"""
    from scipy.optimize import fmin_powell
    if isinstance(disp, string_types) and disp == 'auto':
        disp = True if logger.level <= 20 else False
    # initial guess for center and radius
    xradius = (np.max(points[:, 0]) - np.min(points[:, 0])) / 2.
    yradius = (np.max(points[:, 1]) - np.min(points[:, 1])) / 2.

    radius_init = (xradius + yradius) / 2.
    center_init = np.array([0.0, 0.0, np.max(points[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]

    def cost_fun(x, points):
        return np.sum((np.sqrt(np.sum((points - x[:3]) ** 2, axis=1)) -
                      x[3]) ** 2)

    x_opt = fmin_powell(cost_fun, x0, args=(points,), disp=disp)

    origin = x_opt[:3]
    radius = x_opt[3]
    return radius, origin


# ############################################################################
# Create BEM surfaces

@verbose
def make_watershed_bem(subject, subjects_dir=None, overwrite=False,
                       volume='T1', atlas=False, gcaatlas=False, preflood=None,
                       verbose=None):
    """
    Create BEM surfaces using the watershed algorithm included with FreeSurfer

    Parameters
    ----------
    subject : str
        Subject name (required)
    subjects_dir : str
        Directory containing subjects data. If None use
        the Freesurfer SUBJECTS_DIR environment variable.
    overwrite : bool
        Write over existing files
    volume : str
        Defaults to T1
    atlas : bool
        Specify the --atlas option for mri_watershed
    gcaatlas : bool
        Use the subcortical atlas
    preflood : int
        Change the preflood height
    verbose : bool, str or None
        If not None, override default verbose level

    .. versionadded:: 0.10
    """
    from .surface import read_surface
    env = os.environ.copy()

    if not os.environ.get('FREESURFER_HOME'):
        raise RuntimeError('FREESURFER_HOME environment variable not set')

    env['SUBJECT'] = subject

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    env['SUBJECTS_DIR'] = subjects_dir

    subject_dir = op.join(subjects_dir, subject)
    mri_dir = op.join(subject_dir, 'mri')
    T1_dir = op.join(mri_dir, volume)
    T1_mgz = op.join(mri_dir, volume + '.mgz')
    bem_dir = op.join(subject_dir, 'bem')
    ws_dir = op.join(subject_dir, 'bem', 'watershed')

    if not op.isdir(subject_dir):
        raise RuntimeError('Could not find the MRI data directory "%s"'
                           % subject_dir)
    if not op.isdir(bem_dir):
        os.makedirs(bem_dir)
    if not op.isdir(T1_dir) and not op.isfile(T1_mgz):
        raise RuntimeError('Could not find the MRI data')
    if op.isdir(ws_dir):
        if not overwrite:
            raise RuntimeError('%s already exists. Use the --overwrite option'
                               'to recreate it.' % ws_dir)
        else:
            shutil.rmtree(ws_dir)
    # put together the command
    cmd = ['mri_watershed']
    if preflood:
        cmd += ["-h",  "%s" % int(preflood)]

    if gcaatlas:
        cmd += ['-atlas', '-T1', '-brain_atlas', env['FREESURFER_HOME'] +
                '/average/RB_all_withskull_2007-08-08.gca',
                subject_dir + '/mri/transforms/talairach_with_skull.lta']
    elif atlas:
        cmd += ['-atlas']
    if op.exists(T1_mgz):
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_mgz,
                op.join(ws_dir, 'ws')]
    else:
        cmd += ['-useSRAS', '-surf', op.join(ws_dir, subject), T1_dir,
                op.join(ws_dir, 'ws')]
    # report and run
    logger.info('\nRunning mri_watershed for BEM segmentation with the '
                'following parameters:\n\n'
                'SUBJECTS_DIR = %s\n'
                'SUBJECT = %s\n'
                'Results dir = %s\n' % (subjects_dir, subject, ws_dir))
    os.makedirs(op.join(ws_dir, 'ws'))
    run_subprocess(cmd, env=env, stdout=sys.stdout)
    #
    os.chdir(ws_dir)
    if op.isfile(T1_mgz):
        # XXX : do this with python code
        surfaces = [subject + '_brain_surface', subject +
                    '_inner_skull_surface', subject + '_outer_skull_surface',
                    subject + '_outer_skin_surface']
        for s in surfaces:
            cmd = ['mne_convert_surface', '--surf', s, '--mghmri', T1_mgz,
                   '--surfout', s, "--replacegeom"]
            run_subprocess(cmd, env=env, stdout=sys.stdout)
    os.chdir(bem_dir)
    if op.isfile(subject + '-head.fif'):
        os.remove(subject + '-head.fif')

    # run the equivalent of mne_surf2bem
    points, tris = read_surface(op.join(ws_dir,
                                        subject + '_outer_skin_surface'))
    points *= 1e-3
    surf = dict(coord_frame=5, id=4, nn=None, np=len(points),
                ntri=len(tris), rr=points, sigma=1, tris=tris)
    write_bem_surfaces(subject + '-head.fif', surf)

    logger.info('Created %s/%s-head.fif\n\nComplete.' % (bem_dir, subject))


# ############################################################################
# Read

@verbose
def read_bem_surfaces(fname, patch_stats=False, s_id=None, verbose=None):
    """Read the BEM surfaces from a FIF file

    Parameters
    ----------
    fname : string
        The name of the file containing the surfaces.
    patch_stats : bool, optional (default False)
        Calculate and add cortical patch statistics to the surfaces.
    s_id : int | None
        If int, only read and return the surface with the given s_id.
        An error will be raised if it doesn't exist. If None, all
        surfaces are read and returned.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    surf: list | dict
        A list of dictionaries that each contain a surface. If s_id
        is not None, only the requested surface will be returned.

    See Also
    --------
    write_bem_surfaces, write_bem_solution, make_bem_model
    """
    from .surface import _complete_surface_info
    # Default coordinate frame
    coord_frame = FIFF.FIFFV_COORD_MRI
    # Open the file, create directory
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find BEM
        bem = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if bem is None or len(bem) == 0:
            raise ValueError('BEM data not found')

        bem = bem[0]
        # Locate all surfaces
        bemsurf = dir_tree_find(bem, FIFF.FIFFB_BEM_SURF)
        if bemsurf is None:
            raise ValueError('BEM surface data not found')

        logger.info('    %d BEM surfaces found' % len(bemsurf))
        # Coordinate frame possibly at the top level
        tag = find_tag(fid, bem, FIFF.FIFF_BEM_COORD_FRAME)
        if tag is not None:
            coord_frame = tag.data
        # Read all surfaces
        if s_id is not None:
            surf = [_read_bem_surface(fid, bsurf, coord_frame, s_id)
                    for bsurf in bemsurf]
            surf = [s for s in surf if s is not None]
            if not len(surf) == 1:
                raise ValueError('surface with id %d not found' % s_id)
        else:
            surf = list()
            for bsurf in bemsurf:
                logger.info('    Reading a surface...')
                this = _read_bem_surface(fid, bsurf, coord_frame)
                surf.append(this)
                logger.info('[done]')
            logger.info('    %d BEM surfaces read' % len(surf))
        if patch_stats:
            for this in surf:
                _complete_surface_info(this)
    return surf[0] if s_id is not None else surf


def _read_bem_surface(fid, this, def_coord_frame, s_id=None):
    """Read one bem surface
    """
    # fid should be open as a context manager here
    res = dict()
    # Read all the interesting stuff
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_ID)

    if tag is None:
        res['id'] = FIFF.FIFFV_BEM_SURF_ID_UNKNOWN
    else:
        res['id'] = int(tag.data)

    if s_id is not None and res['id'] != s_id:
        return None

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SIGMA)
    res['sigma'] = 1.0 if tag is None else float(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NNODE)
    if tag is None:
        raise ValueError('Number of vertices not found')

    res['np'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NTRI)
    if tag is None:
        raise ValueError('Number of triangles not found')
    res['ntri'] = int(tag.data)

    tag = find_tag(fid, this, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        tag = find_tag(fid, this, FIFF.FIFF_BEM_COORD_FRAME)
        if tag is None:
            res['coord_frame'] = def_coord_frame
        else:
            res['coord_frame'] = tag.data
    else:
        res['coord_frame'] = tag.data

    # Vertices, normals, and triangles
    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NODES)
    if tag is None:
        raise ValueError('Vertex data not found')

    res['rr'] = tag.data.astype(np.float)  # XXX : double because of mayavi bug
    if res['rr'].shape[0] != res['np']:
        raise ValueError('Vertex information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS)
    if tag is None:
        tag = tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_NORMALS)
    if tag is None:
        res['nn'] = list()
    else:
        res['nn'] = tag.data
        if res['nn'].shape[0] != res['np']:
            raise ValueError('Vertex normal information is incorrect')

    tag = find_tag(fid, this, FIFF.FIFF_BEM_SURF_TRIANGLES)
    if tag is None:
        raise ValueError('Triangulation not found')

    res['tris'] = tag.data - 1  # index start at 0 in Python
    if res['tris'].shape[0] != res['ntri']:
        raise ValueError('Triangulation information is incorrect')

    return res


@verbose
def read_bem_solution(fname, verbose=None):
    """Read the BEM solution from a file

    Parameters
    ----------
    fname : string
        The file containing the BEM solution.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    bem : dict
        The BEM solution.

    See Also
    --------
    write_bem_solution, read_bem_surfaces, write_bem_surfaces,
    make_bem_solution
    """
    # mirrors fwd_bem_load_surfaces from fwd_bem_model.c
    logger.info('Loading surfaces...')
    bem_surfs = read_bem_surfaces(fname, patch_stats=True, verbose=False)
    if len(bem_surfs) == 3:
        logger.info('Three-layer model surfaces loaded.')
        needed = np.array([FIFF.FIFFV_BEM_SURF_ID_HEAD,
                           FIFF.FIFFV_BEM_SURF_ID_SKULL,
                           FIFF.FIFFV_BEM_SURF_ID_BRAIN])
        if not all(x['id'] in needed for x in bem_surfs):
            raise RuntimeError('Could not find necessary BEM surfaces')
        # reorder surfaces as necessary (shouldn't need to?)
        reorder = [None] * 3
        for x in bem_surfs:
            reorder[np.where(x['id'] == needed)[0][0]] = x
        bem_surfs = reorder
    elif len(bem_surfs) == 1:
        if not bem_surfs[0]['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN:
            raise RuntimeError('BEM Surfaces not found')
        logger.info('Homogeneous model surface loaded.')

    # convert from surfaces to solution
    bem = dict(surfs=bem_surfs)
    logger.info('\nLoading the solution matrix...\n')
    f, tree, _ = fiff_open(fname)
    with f as fid:
        # Find the BEM data
        nodes = dir_tree_find(tree, FIFF.FIFFB_BEM)
        if len(nodes) == 0:
            raise RuntimeError('No BEM data in %s' % fname)
        bem_node = nodes[0]

        # Approximation method
        tag = find_tag(f, bem_node, FIFF.FIFF_BEM_APPROX)
        if tag is None:
            raise RuntimeError('No BEM solution found in %s' % fname)
        method = tag.data[0]
        if method not in (FIFF.FIFFV_BEM_APPROX_CONST,
                          FIFF.FIFFV_BEM_APPROX_LINEAR):
            raise RuntimeError('Cannot handle BEM approximation method : %d'
                               % method)

        tag = find_tag(fid, bem_node, FIFF.FIFF_BEM_POT_SOLUTION)
        dims = tag.data.shape
        if len(dims) != 2:
            raise RuntimeError('Expected a two-dimensional solution matrix '
                               'instead of a %d dimensional one' % dims[0])

        dim = 0
        for surf in bem['surfs']:
            if method == FIFF.FIFFV_BEM_APPROX_LINEAR:
                dim += surf['np']
            else:  # method == FIFF.FIFFV_BEM_APPROX_CONST
                dim += surf['ntri']

        if dims[0] != dim or dims[1] != dim:
            raise RuntimeError('Expected a %d x %d solution matrix instead of '
                               'a %d x %d one' % (dim, dim, dims[1], dims[0]))
        sol = tag.data
        nsol = dims[0]

    bem['solution'] = sol
    bem['nsol'] = nsol
    bem['bem_method'] = method

    # Gamma factors and multipliers
    _add_gamma_multipliers(bem)
    kind = {
        FIFF.FIFFV_BEM_APPROX_CONST: 'constant collocation',
        FIFF.FIFFV_BEM_APPROX_LINEAR: 'linear_collocation',
    }[bem['bem_method']]
    logger.info('Loaded %s BEM solution from %s', kind, fname)
    return bem


def _add_gamma_multipliers(bem):
    """Helper to add gamma and multipliers in-place"""
    bem['sigma'] = np.array([surf['sigma'] for surf in bem['surfs']])
    # Dirty trick for the zero conductivity outside
    sigma = np.r_[0.0, bem['sigma']]
    bem['source_mult'] = 2.0 / (sigma[1:] + sigma[:-1])
    bem['field_mult'] = sigma[1:] - sigma[:-1]
    # make sure subsequent "zip"s work correctly
    assert len(bem['surfs']) == len(bem['field_mult'])
    bem['gamma'] = ((sigma[1:] - sigma[:-1])[np.newaxis, :] /
                    (sigma[1:] + sigma[:-1])[:, np.newaxis])
    bem['is_sphere'] = False


_surf_dict = {'inner_skull': FIFF.FIFFV_BEM_SURF_ID_BRAIN,
              'outer_skull': FIFF.FIFFV_BEM_SURF_ID_SKULL,
              'head': FIFF.FIFFV_BEM_SURF_ID_HEAD}


def _bem_find_surface(bem, id_):
    """Find surface from already-loaded BEM"""
    if isinstance(id_, string_types):
        name = id_
        id_ = _surf_dict[id_]
    else:
        name = _bem_explain_surface(id_)
    idx = np.where(np.array([s['id'] for s in bem['surfs']]) == id_)[0]
    if len(idx) != 1:
        raise RuntimeError('BEM model does not have the %s triangulation'
                           % name.replace('_', ' '))
    return bem['surfs'][idx[0]]


def _bem_explain_surface(id_):
    """Return a string corresponding to the given surface ID"""
    _rev_dict = dict((val, key) for key, val in _surf_dict.items())
    return _rev_dict[id_]


# ############################################################################
# Write

@deprecated('write_bem_surface is deprecated and will be removed in 0.11, '
            'use write_bem_surfaces instead')
def write_bem_surface(fname, surf):
    """Write one bem surface

    Parameters
    ----------
    fname : string
        File to write
    surf : dict
        A surface structured as obtained with read_bem_surfaces

    See Also
    --------
    read_bem_surfaces
    """
    write_bem_surfaces(fname, surf)


def write_bem_surfaces(fname, surfs):
    """Write BEM surfaces to a fiff file

    Parameters
    ----------
    fname : str
        Filename to write.
    surfs : dict | list of dict
        The surfaces, or a single surface.
    """
    if isinstance(surfs, dict):
        surfs = [surfs]
    with start_file(fname) as fid:
        start_block(fid, FIFF.FIFFB_BEM)
        write_int(fid, FIFF.FIFF_BEM_COORD_FRAME, surfs[0]['coord_frame'])
        _write_bem_surfaces_block(fid, surfs)
        end_block(fid, FIFF.FIFFB_BEM)
        end_file(fid)


def _write_bem_surfaces_block(fid, surfs):
    """Helper to actually write bem surfaces"""
    for surf in surfs:
        start_block(fid, FIFF.FIFFB_BEM_SURF)
        write_float(fid, FIFF.FIFF_BEM_SIGMA, surf['sigma'])
        write_int(fid, FIFF.FIFF_BEM_SURF_ID, surf['id'])
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, surf['coord_frame'])
        write_int(fid, FIFF.FIFF_BEM_SURF_NNODE, surf['np'])
        write_int(fid, FIFF.FIFF_BEM_SURF_NTRI, surf['ntri'])
        write_float_matrix(fid, FIFF.FIFF_BEM_SURF_NODES, surf['rr'])
        # index start at 0 in Python
        write_int_matrix(fid, FIFF.FIFF_BEM_SURF_TRIANGLES,
                         surf['tris'] + 1)
        if 'nn' in surf and surf['nn'] is not None and len(surf['nn']) > 0:
            write_float_matrix(fid, FIFF.FIFF_BEM_SURF_NORMALS, surf['nn'])
        end_block(fid, FIFF.FIFFB_BEM_SURF)


def write_bem_solution(fname, bem):
    """Write a BEM model with solution

    Parameters
    ----------
    fname : str
        The filename to use.
    bem : dict
        The BEM model with solution to save.

    See Also
    --------
    read_bem_solution
    """
    with start_file(fname) as fid:
        start_block(fid, FIFF.FIFFB_BEM)
        # Coordinate frame (mainly for backward compatibility)
        write_int(fid, FIFF.FIFF_BEM_COORD_FRAME,
                  bem['surfs'][0]['coord_frame'])
        # Surfaces
        _write_bem_surfaces_block(fid, bem['surfs'])
        # The potential solution
        if 'solution' in bem:
            if bem['bem_method'] != FIFF.FWD_BEM_LINEAR_COLL:
                raise RuntimeError('Only linear collocation supported')
            write_int(fid, FIFF.FIFF_BEM_APPROX, FIFF.FIFFV_BEM_APPROX_LINEAR)
            write_float_matrix(fid, FIFF.FIFF_BEM_POT_SOLUTION,
                               bem['solution'])
        end_block(fid, FIFF.FIFFB_BEM)
        end_file(fid)
