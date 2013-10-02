# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from copy import deepcopy

from ..surface import fast_cross_3d, _find_nearest_tri_pt, _get_tri_supp_geom
from ..fiff.constants import FIFF
from ..transforms import apply_trans
from ..utils import logger
from ..parallel import parallel_func


##############################################################################
# Triangle utilities

def _project_to_triangle(s, idx, p, q):
    tri = s['tris'][idx]
    r1 = s['rr'][tri[0]]
    r12 = s['rr'][tri[1]] - r1
    r13 = s['rr'][tri[2]] - r1
    r = r1 + p * r12 + q * r13
    return r


def _triangle_coords(r, geom, best):
    r1 = geom['r1'][best]
    tri_nn = geom['nn'][best]
    r12 = geom['r12'][best]
    r13 = geom['r13'][best]
    a = geom['a'][best]
    b = geom['b'][best]
    c = geom['c'][best]
    rr = r - r1
    z = np.sum(rr * tri_nn)
    v1 = np.sum(rr * r12)
    v2 = np.sum(rr * r13)
    det = a * b - c * c
    x = (b * v1 - c * v2) / det
    y = (a * v2 - c * v1) / det
    return x, y, z


##############################################################################
# COIL DEFINITIONS

def _dup_coil_set(s, t):
    """Make a duplicate"""
    if t is not None and s['coord_frame'] != t['from']:
        raise RuntimeError('Coordinate frame of the transformation does not '
                           'match the coil set')
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


def _calc_beta(rk, rk1):
    rkk1 = rk - rk1
    size = np.sqrt(np.sum(rkk1 * rkk1, axis=1))
    vlrk = np.sqrt(np.sum(rk * rk, axis=1))
    vlrk1 = np.sqrt(np.sum(rk1 * rk1, axis=1))
    return np.log((vlrk * size + np.sum(rk * rkk1, axis=1)) /
                  (vlrk1 * size + np.sum(rk1 * rkk1, axis=1))) / size


def _one_field_coeff(dest, normal, tri_rr):
    """Compute the integral over one triangle"""
    # This looks magical but it is not.
    yy = dest - tri_rr[[0, 1, 2, 0]]
    beta = _calc_beta(yy[:3], yy[1:])
    beta = np.array([beta[2] - beta[0], beta[0] - beta[1], beta[1] - beta[2]])
    return np.dot(np.sum(yy[:3] * beta, axis=0), normal)


def _check_coil_frame(coils, m):
    if coils['coord_frame'] != FIFF.FIFFV_COORD_MRI:
        if coils['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            if m['head_mri_t'] is None:
                raise RuntimeError('head -> mri coordinate transform missing')
            # Make a transformed duplicate
            coils = _dup_coil_set(coils, m['head_mri_t'])
        else:
            raise RuntimeError('Incompatible coil coordinate frame %d'
                               % coils['coord_frame'])
    return coils


def _bem_lin_field_coeff(m, coils, n_jobs,
                         method=FIFF.FWD_BEM_LIN_FIELD_SIMPLE):
    """Compute the weighting factors to obtain the magnetic field"""
    #in the linear potential approximation
    coils = _check_coil_frame(coils, m)

    # leaving this in in case we want to easily add in the future
    if method == FIFF.FWD_BEM_LIN_FIELD_FERGUSON:
        raise NotImplementedError
    elif method == FIFF.FWD_BEM_LIN_FIELD_URANKAR:
        raise NotImplementedError
    else:
        func = _bem_one_lin_field_coeff_simple

    # Process each of the surfaces
    assert len(m['surfs']) == len(m['field_mult'])
    rmags = np.concatenate([coil['rmag'] for coil in coils['coils']])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils['coils']])
    lims = np.cumsum(np.r_[0, [len(coil['rmag']) for coil in coils['coils']]])
    ws = np.concatenate([coil['w'] for coil in coils['coils']])

    # might as well parallelize over surfaces here
    parallel, p_fun, _ = parallel_func(_do_lin_field_coeff, n_jobs)
    coeff = parallel(p_fun(surf, mult, rmags, cosmags, ws, lims, func)
                     for surf, mult in zip(m['surfs'], m['field_mult']))
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


def _bem_specify_coils(m, coils, n_jobs):
    """Set up for computing the solution at a set of coils"""
    if m['bem_method'] == 'linear collocation':
        sol = _bem_lin_field_coeff(m, coils, n_jobs)
    else:
        raise RuntimeError('Only linear collocation supported')
    solution = np.dot(sol, m['solution'])
    csol = dict(ncoil=coils['ncoil'], np=m['nsol'], solution=solution)
    coils['user_data'] = csol


def _bem_specify_els(m, els):
    """Set up for computing the solution at a set of electrodes"""
    sol = dict(ncoil=els['ncoil'], np=m['nsol'])
    els['user_data'] = sol
    solution = np.zeros((sol['ncoil'], sol['np']))
    sol['solution'] = solution
    if m['bem_method'] not in ['linear collocation']:
        raise RuntimeError('Only linear collocation supported')
    # Go through all coils
    scalp = m['surfs'][0]
    scalp['geom'] = _get_tri_supp_geom(scalp['tris'], scalp['rr'])
    inds = np.arange(len(scalp['tris']))

    # In principle this could be parallelized, but pickling overhead is huge
    # (makes it slower than non-parallel)
    for k, el in enumerate(els['coils']):
        # Go through all 'integration points'
        for elw, r in zip(el['w'], el['rmag']):
            if m['head_mri_t'] is not None:
                r = apply_trans(m['head_mri_t']['trans'], r)
            best = _find_nearest_tri_pt(inds, r, scalp['geom'], True)[2]
            # Calculate a linear interpolation between the vertex values
            tri = scalp['tris'][best]
            x, y, z = _triangle_coords(r, scalp['geom'], best)
            w = elw * np.array([(1.0 - x - y), x, y])
            amt = np.dot(w, m['solution'][tri])
            solution[k] += amt


#############################################################################
# FORWARD COMPUTATION

def _make_ctf_comp(dataset, chs, compchs):
    """Make CTF compensator"""
    # mne_make_ctf_comp() from mne_ctf_comp.c
    nch = len(chs)
    logger.info('Setting up compensation data...')
    comps = np.zeros(nch, int)
    need_comp = False
    first_comp = 0
    for k, ch in enumerate(chs):
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
            comps[k] = int(ch['coil_type']) >> 16
            if comps[k] != 0:
                if first_comp == 0:
                    first_comp = comps[k]
                elif comps[k] != first_comp:
                    raise RuntimeError('We do not support nonuniform '
                                       'compensation yet.')
                need_comp = True
        else:
            comps[k] = 0

    if need_comp is False:
        logger.info('    No compensation set. Nothing more to do.')
        return None
    else:
        raise NotImplementedError


def _make_ctf_comp_coils(dataset, coils, comp_coils):
    """Call mne_make_ctf_comp using the information in the coil sets"""
    # Create the fake channel info which contain just enough information
    # for _make_ctf_comp
    chs = list()
    for coil in coils['coils']:
        if coil['coil_class'] == FIFF.FWD_COILC_EEG:
            kind = FIFF.FIFFV_EEG_CH
        else:
            kind = FIFF.FIFFV_MEG_CH
        chs.append(dict(ch_name=coil['chname'], coil_type=coil['type'],
                        kind=kind))

    compchs = list()
    if comp_coils is not None and comp_coils['ncoil'] > 0:
        for coil in coils['coils']:
            if coil['coil_class'] == FIFF.FWD_COILC_EEG:
                kind = FIFF.FIFFV_EEG_CH
            else:
                kind = FIFF.FIFFV_MEG_CH
            compchs.append(dict(ch_name=coil['chname'],
                                coil_type=coil['type'], kind=kind))

    return _make_ctf_comp(dataset, chs, compchs)


def _bem_inf_pot(rd, Q, rp):
    """The infinite medium potential"""
    # NOTE: the (4.0 * np.pi) that was in the denominator has been moved!
    diff = rp - rd
    diff2 = np.sum(diff * diff, axis=1)
    return np.sum(Q * diff, axis=1) / (diff2 * np.sqrt(diff2))


def _bem_inf_field(rd, Q, rp, d):
    """Infinite-medium magnetic field"""
    diff = rp - rd
    diff2 = np.sum(diff * diff, axis=1)
    x = fast_cross_3d(Q[np.newaxis, :], diff)
    return np.sum(x * d, axis=1) / (diff2 * np.sqrt(diff2))


def _apply_ctf_comp():
    # mne_apply_ctf_comp()
    raise NotImplementedError


def _need_comp(comp):
    """Helper for triaging whether coils have compensation"""
    need = (comp['comp_coils'] and comp['comp_coils']['ncoil'] > 0
            and comp['set'] and comp['dataset']['current'])
    return need


def _comp_field(rrs, coils, client, n_jobs):
    """Calculate the compensated field"""
    comp = client
    if not comp['field']:
        raise RuntimeError('Field computation function is missing')

    # First compute the field in the primary set of coils
    res = comp['field'](rrs, coils, comp['client'], n_jobs)

    # Compensation needed?
    if _need_comp(comp):
        # Compute the field in the compensation coils
        comp['work'] = comp['field'](rrs, comp['comp_coils'], comp['work'],
                                     comp['client'], n_jobs)
        _apply_ctf_comp(comp['dataset'], True, res,
                        coils['ncoil'], comp['work'],
                        comp['comp_coils']['ncoil'])
    return res


def _bem_field(rrs, coils, m, n_jobs):
    """Calculate the magnetic field in a set of coils"""
    mults = np.repeat(m['source_mult'] / (4.0 * np.pi),
                      [len(s['rr']) for s in m['surfs']])
    solution = coils['user_data']['solution'] * mults[np.newaxis, :]
    srr = np.concatenate([s['rr'] for s in m['surfs']])

    Qs = np.eye(3)
    rmags = np.concatenate([coil['rmag'] for coil in coils['coils']])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils['coils']])
    ws = np.concatenate([coil['w'] for coil in coils['coils']])
    lims = np.r_[0, np.cumsum([len(coil['rmag']) for coil in coils['coils']])]

    # parallelize here
    ncoil = len(coils['coils'])
    B = np.zeros((3, len(rrs), ncoil))
    for qi, Qq in enumerate(Qs):
        mri_rds, mri_Q = _xform_rd_Q(m['head_mri_t'], rrs, Qq)
        for ri, (mri_rd, rr) in enumerate(zip(mri_rds, rrs)):
            # The dipole location and orientation must be transformed

            # Compute inifinite-medium potentials (see non-vectorized
            # version in _bem_pot_els)
            v0 = _bem_inf_pot(mri_rd, mri_Q, srr)

            # Primary current contribution
            # (can be calculated in the coil/dipole coordinates)

            # The following code is equivalent to this, but vectorized:
            #x = np.array([np.sum(coil['w'] *
            #                     _bem_inf_field(rr, Qq, coil['rmag'],
            #                                    coil['cosmag']))
            #              for coil in coils['coils']])
            x = np.r_[0.0, np.cumsum(ws * _bem_inf_field(rr, Qq, rmags,
                                                         cosmags))]
            B[qi, ri] = np.diff(x[lims])

            # Volume current contribution
            B[qi, ri] += np.dot(solution, v0)
    B = np.reshape(np.swapaxes(B, 0, 1), (len(rrs) * 3, B.shape[2]))
    # Scale correctly
    B *= 1e-7  # MAG_FACTOR in C code
    return B


def _bem_pot_els(rrs, els, m, n_jobs):
    """Compute the potentials due to a current dipole"""
    # n_jobs currently ignored because no speedup gains found from using it
    v0 = np.zeros(m['nsol'])
    Qs = np.eye(3)
    pot = np.zeros((3, len(rrs), len(els['coils'])))
    mults = np.repeat(m['source_mult'] / (4.0 * np.pi),
                      [len(s['rr']) for s in m['surfs']])
    solution = els['user_data']['solution'] * mults[np.newaxis, :]
    srr = np.concatenate([s['rr'] for s in m['surfs']])
    for qi, Qq in enumerate(Qs):
        mri_rds, mri_Q = _xform_rd_Q(m['head_mri_t'], rrs, Qq)
        for ri, mri_rd in enumerate(mri_rds):
            # The below is equivalent to the following, but vectorized
            #soff = 0
            #for surf, mult in zip(m['surfs'], m['source_mult']):
            #    srr = surf['rr']
            #    v0[soff:soff + len(srr)] = mult * _bem_inf_pot(mri_rd, mri_Q,
            #                                                   srr)
            #    soff += len(srr)
            v0 = _bem_inf_pot(mri_rd, mri_Q, srr)  # "mults" in solution
            pot[qi, ri] = np.dot(solution, v0)
    pot = np.reshape(np.swapaxes(pot, 0, 1), (len(rrs) * 3, pot.shape[2]))
    return pot


def _xform_rd_Q(m_trans, rd, Q):
    """Transform rd and Q into mri coords if necessary"""
    if m_trans is not None:
        rd = apply_trans(m_trans['trans'], rd)
        Q = apply_trans(m_trans['trans'], Q, False)
    return rd, Q


def _compute_forward(src, coils_els, comp_coils, comp_data, bem_model, ctype,
                     n_jobs):
    """Compute the MEG forward solution"""
    if ctype == 'meg':
        # Use the new compensated field computation
        # It works the same way independent of whether or not the compensation
        # is in effect

        # Compose a compensation data set
        comp = dict(set=comp_data, comp_coils=comp_coils, field=_bem_field,
                    vec_field=None, client=bem_model)
        _make_ctf_comp_coils(comp['set'], coils_els, comp['comp_coils'])

        # Field computation matrices...
        logger.info('')
        logger.info('Composing the field computation matrix...')
        _bem_specify_coils(bem_model, coils_els, n_jobs)

        if comp['set'] is not None and comp['set']['current'] is True:
            logger.info('Composing the field computation matrix '
                        '(compensation coils)...')
            _bem_specify_coils(bem_model, comp['comp_coils'])
        field = _comp_field
        client = comp
        ftype = 'MEG'
    elif ctype == 'eeg':
        _bem_specify_els(bem_model, coils_els)
        client = bem_model
        field = _bem_pot_els
        ftype = 'EEG'
    else:
        raise ValueError('coil_type must be "meg" or "eeg"')

    # Count the sources and allocate space
    n_ch = len(coils_els['coils'])
    nsource = np.sum([s['nuse'] for s in src])
    n_res = 3 * nsource  # free orientations
    res = np.zeros((n_res, n_ch))

    # Set up arguments for the field computation
    logger.info('Computing %s at %d source locations '
                '(free orientations)...' % (ftype, nsource))
    # keep this split because a) we can and b) reduces cumsum() errors
    res = list()
    for s in src:
        rrs = s['rr'][np.where(s['inuse'])[0]]
        res.append(field(rrs, coils_els, client, n_jobs))
    res = np.concatenate(res)
    return res
