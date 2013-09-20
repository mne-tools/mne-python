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


_mag_factor = 1e-7  # \mu_0/4\pi


##############################################################################
# Triangle utilities

def _project_to_triangle(s, idx, p, q):
    tri = s['tris'][idx]
    r1 = s['rr'][tri[0]]
    r12 = r1 - s['rr'][tri[1]]
    r13 = r1 - s['rr'][tri[2]]
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
    rr = r1 - r
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


def _bem_one_lin_field_coeff_ferg():
    """Skip this, code currently only requires SIMPLE"""
    raise NotImplementedError


def _bem_one_lin_field_coeff_uran():
    """Skip this, code currently only requires SIMPLE"""
    raise NotImplementedError


def _bem_one_lin_field_coeff_simple(dest, normal, tri_rr, tri_nn, tri_area):
    """Simple version..."""
    out = np.zeros((3, len(dest)))
    for rr, o in zip(tri_rr, out):
        diff = rr - dest
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


def _bem_field_coeff(m, coils):
    """Compute the weighting factors to obtain the magnetic field"""
    coils = _check_coil_frame(coils, m)
    coeff = np.zeros((coils['ncoil'], m['nsol']))
    off = 0
    assert len(m['surfs'] == len(m['field_mult']))
    for surf, mult in zip(m['surfs'], m['field_mult']):
        for k, tri in enumerate(surf['tris']):
            tri_rr = surf['rr'][tri]
            for j, coil in enumerate(coils['coils']):
                res = 0.0
                for p in xrange(coil['np']):
                    res += coil['w'][p] * _one_field_coeff(coil['rmag'][p],
                                                           coil['cosmag'][p],
                                                           tri_rr)
                coeff[j][k + off] = mult * res
        off += len(surf['tris'])
    return coeff


def _bem_lin_field_coeff(m, coils, method):
    """Compute the weighting factors to obtain the magnetic field"""
    #in the linear potential approximation
    coils = _check_coil_frame(coils, m)
    coeff = np.zeros((coils['ncoil'], m['nsol']))
    if method == FIFF.FWD_BEM_LIN_FIELD_FERGUSON:
        func = _bem_one_lin_field_coeff_ferg
    elif method == FIFF.FWD_BEM_LIN_FIELD_URANKAR:
        func = _bem_one_lin_field_coeff_uran
    else:
        func = _bem_one_lin_field_coeff_simple

    # Process each of the surfaces
    off = 0
    assert len(m['surfs']) == len(m['field_mult'])
    rmags = np.concatenate([coil['rmag'] for coil in coils['coils']])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils['coils']])
    lims = np.cumsum(np.r_[0, [len(coil['rmag']) for coil in coils['coils']]])
    ws = np.concatenate([coil['w'] for coil in coils['coils']])
    for surf, mult in zip(m['surfs'], m['field_mult']):
        for tri, tri_nn, tri_area in zip(surf['tris'],
                                         surf['tri_nn'], surf['tri_area']):
            """
            # The following is equivalent to:
            tri_rr = surf['rr'][tri]
            xx = func(cosmags, rmags, tri_rr, tri_nn, tri_area)
            for j, coil in enumerate(coils['coils']):
                # Accumulate the coefficients for each triangle node...
                x = func(coil['rmag'], coil['cosmag'],
                         tri_rr, tri_nn, tri_area)
                assert np.array_equal(x, xx[lims[j]:lims[j + 1]])
                res = np.sum(coil['w'][np.newaxis, :] * x, axis=1)
                # Add these to the corresponding coefficient matrix
                coeff[j][tri + off] += mult * res
            """
            tri_rr = surf['rr'][tri]
            # Accumulate the coefficients for each triangle node...
            xx = func(rmags, cosmags, tri_rr, tri_nn, tri_area)
            yy = np.c_[np.zeros((3, 1)), np.cumsum(xx * ws, axis=1)]
            zz = yy[:, lims[1:]] - yy[:, lims[:-1]]
            coeff[:, tri + off] += zz.T
        off += surf['np']
    return coeff


def _bem_specify_coils(m, coils):
    """Set up for computing the solution at a set of coils"""
    if m['bem_method'] == 'constant collocation':
        sol = _bem_field_coeff(m, coils)
    elif m['bem_method'] == 'linear collocation':
        sol = _bem_lin_field_coeff(m, coils, FIFF.FWD_BEM_LIN_FIELD_SIMPLE)
    else:
        raise RuntimeError('Unknown BEM method in fwd_bem_specify_coils')
    solution = np.dot(sol, m['solution'])
    csol = dict(ncoil=coils['ncoil'], np=m['nsol'], solution=solution)
    coils['user_data'] = csol


def _bem_specify_els(m, els):
    """Set up for computing the solution at a set of electrodes"""
    sol = dict(ncoil=els['ncoil'], np=m['nsol'])
    els['user_data'] = sol
    solution = np.zeros((sol['ncoil'], sol['np']))
    sol['solution'] = solution
    if m['bem_method'] not in ['constant collocation',
                               'linear collocation']:
        raise RuntimeError('Unknown BEM approximation method')
    # Go through all coils
    scalp = m['surfs'][0]
    scalp['geom'] = _get_tri_supp_geom(scalp['tris'], scalp['rr'])
    inds = np.arange(len(scalp['rr']))
    for k, el in enumerate(els['coils']):
        # Go through all 'integration points'
        for elw, r in zip(el['w'], el['rmag']):
            if m['head_mri_t'] is not None:
                r = apply_trans(m['head_mri_t']['trans'], r)
            best = _find_nearest_tri_pt(inds, r, scalp['geom'])[2]
            if m['bem_method'] == 'constant collocation':
                # Simply pick the value at the triangle
                solution[k] += elw * m['solution'][best]
            else:  # m['bem_method'] == 'linear collocation'
                # Calculate a linear interpolation between the vertex values
                tri = scalp['tris'][best]
                x, y, z = _triangle_coords(r, scalp['geom'], best)
                w = elw * np.array([(1.0 - x - y), x, y])
                for v in xrange(3):
                    solution[k] += w[v] * m['solution'][tri[v]]


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


def _make_comp_data(dataset, coils, comp_coils, field, vec_field, field_grad,
                    client):
    """Compose a compensation data set"""
    comp = dict(set=dataset, comp_coils=comp_coils, field=field,
                vec_field=vec_field, field_grad=field_grad,
                client=client)
    _make_ctf_comp_coils(comp['set'], coils, comp['comp_coils'])
    return comp


def _bem_inf_pot(rd, Q, rp):
    """The infinite medium potential"""
    diff = rd - rp
    diff2 = np.sqrt(np.sum(diff * diff, axis=1))
    return np.sum(Q * diff, axis=1) / (4.0 * np.pi * diff2 * diff2)


def _bem_inf_field(rd, Q, rp, d):
    """Infinite-medium magnetic field"""
    diff = rd - rp
    diff2 = np.sum(diff * diff, axis=1)
    x = fast_cross_3d(Q[np.newaxis, :], diff)
    return np.sum(x * d, axis=1) / (diff2 * np.sqrt(diff2))


def _bem_inf_pot_der(rd, Q, rp, comp):
    """Derivative of the infinite-medium potential"""
    # with respect to one of the dipole position coordinates
    diff = rd - rp
    diff2 = np.sum(diff * diff, axis=1)
    diff3 = np.sqrt(diff2) * diff2
    diff5 = diff3 * diff2
    res = (3 * np.sum(Q * diff, axis=1) * np.sum(comp * diff, axis=1) / diff5
           - np.sum(comp * Q, axis=1) / diff3)
    return res / (4.0 * np.pi)


def _bem_inf_field_der(rd, Q, rp, d, comp):
    """Derivative of the infinite-medium magnetic field"""
    # with respect to one of the dipole position coordinates
    diff = rd - rp
    diff2 = np.sum(diff * diff, axis=1)
    diff3 = np.sqrt(diff2) * diff2
    diff5 = diff3 * diff2
    x = fast_cross_3d(Q, diff)
    xn = fast_cross_3d(d, Q)
    res = (3 * np.sum(x * d, axis=1) * np.sum(comp * diff, axis=1) / diff5
           - np.sum(comp, xn, axis=1) / diff3)
    return res


def _bem_field(rd, Q, coils, B, m):
    """Calculate the magnetic field in a set of coils"""
    sol = coils['user_data']
    # Infinite-medium potentials
    v0 = _get_v0(m)

    # The dipole location and orientation must be transformed
    mri_rd, mri_Q = _xform_rd_Q(m, rd, Q)

    # Compute the inifinite-medium potentials at the centers of the triangles
    assert len(m['surfs']) == len(m['source_mult'])
    off = 0
    for surf, mult in zip(m['surfs'], m['source_mult']):
        if m['bem_method'] == 'linear collocation':
            rr = surf['rr']
        else:  # m['bem_method'] == 'constant collocation':
            rr = surf['tri_cent']
        v0[off:off + len(rr)] = mult * _bem_inf_pot(mri_rd, mri_Q, rr)
        off += len(rr)

    # Primary current contribution
    # (can be calculated in the coil/dipole coordinates)

    # The following code is equivalent to this, but vectorized:
    # B.fill(0.0)
    # for k, coil in enumerate(coils['coils']):
    #     B[k] += np.sum(coil['w'] * _bem_inf_field(rd, Q, coil['rmag'],
    #                                               coil['cosmag']))
    rmags = np.concatenate([coil['rmag'] for coil in coils['coils']])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils['coils']])
    ws = np.concatenate([coil['w'] for coil in coils['coils']])
    lims = np.r_[0, np.cumsum([len(coil['rmag']) for coil in coils['coils']])]
    x = np.r_[0.0, np.cumsum(ws * _bem_inf_field(rd, Q, rmags, cosmags))]
    B[:] = np.diff(x[lims])

    # Volume current contribution
    B += np.dot(sol['solution'], v0)

    # Scale correctly
    B *= _mag_factor


def _bem_field_grad(rd, Q, coils, m, xgrad, ygrad, zgrad):
    """Calculate the magnetic field in a set of coils"""
    sol = coils['user_data']
    grads = np.array([xgrad, ygrad, zgrad])

    # Infinite-medium potentials
    v0 = _get_v0(m)

    # The dipole location and orientation must be transformed
    mri_rd, mri_Q = _xform_rd_Q(m, rd, Q)

    ees = np.eye(3)
    assert len(m['surfs']) == len(m['source_mults'])
    for pp, ee in enumerate(ees):
        grad = grads[pp]

        # Select the correct gradient component
        mri_ee = ee.copy()
        if m['head_mri_t'] is not None:
            mri_ee = apply_trans(m['head_mri_t']['trans'], mri_ee, False)

        # Compute the inifinite-medium potential derivatives at the
        #centers of the triangles
        off = 0
        for surf, mult in zip(m['surfs'], m['source_mults']):
            ntri = surf['ntri']
            tri_cent = surf['tri_cent']
            v0[off:off + ntri] = mult * _bem_inf_pot_der(mri_rd, mri_Q,
                                                         tri_cent, mri_ee)
            off += ntri

        # Primary current contribution
        # (can be calculated in the coil/dipole coordinates)
        grad.fill(0.0)
        for k, coil in enumerate(coils['coils']):
            grad[k] += coil['w'] * _bem_inf_field_der(rd, Q, coil['rmag'],
                                                      coil['cosmag'], ee)

        # Volume current contribution
        grad += np.dot(sol['solution'], v0)

        # Scale correctly
        grad *= _mag_factor


def _apply_ctf_comp():
    # mne_apply_ctf_comp()
    raise NotImplementedError


def _need_comp(comp):
    """Helper for triaging whether coils have compensation"""
    need = (comp['comp_coils'] and comp['comp_coils']['ncoil'] > 0
            and comp['set'] and comp['dataset']['current'])
    return need


def _comp_field(rd, Q, coils, res, client):
    """Calculate the compensated field (one dipole component)"""
    comp = client
    if not comp['field']:
        raise RuntimeError('Field computation function is missing')

    # First compute the field in the primary set of coils
    comp['field'](rd, Q, coils, res, comp['client'])

    # Compensation needed?
    if not _need_comp(comp):
        return

    if not comp['work']:
        comp['work'] = np.zeros(comp['comp_coils']['ncoil'])

    # Compute the field in the compensation coils
    comp['field'](rd, Q, comp['comp_coils'], comp['work'], comp['client'])
    _apply_ctf_comp(comp['dataset'], True, res, coils['ncoil'], comp['work'],
                    comp['comp_coils']['ncoil'])


def _comp_field_grad(rd, Q, coils, res, xgrad, ygrad, zgrad, client):
    """Calculate the compensated field (one dipole component)"""
    comp = client
    if not comp['field_grad']:
        raise RuntimeError('Field and gradient computation function missing')

    # First compute the field in the primary set of coils
    comp['field_grad'](rd, Q, coils, res, xgrad, ygrad, zgrad, comp['client'])

    # Compensation needed?
    if not _need_comp(comp):
        return

    # Workspace needed?
    if not comp['work']:
        comp['work'] = np.zeros(comp['comp_coils']['ncoil'])

    if not comp['vec_work']:
        comp['vec_work'] = np.zeros((3, comp['comp_coils']['ncoil']))

    # Compute the field in the compensation coils
    comp['field_grad'](rd, Q, comp['comp_coils'], comp['work'],
                       comp['vec_work'][0], comp['vec_work'][1],
                       comp['vec_work'][2], comp['client'])

    # Compute the compensated field
    _apply_ctf_comp(comp['dataset'], True, res, coils['ncoil'],
                    comp['work'], comp['comp_coils']['ncoil'])
    for cvw, grad in zip(comp['vec_work'], [xgrad, ygrad, zgrad]):
        _apply_ctf_comp(comp['dataset'], True, grad, coils['ncoil'],
                        cvw, comp['comp_coils']['ncoil'])


def _get_v0(m):
    """Wrap a common call"""
    if 'v0' not in m:
        m['v0'] = np.zeros(m['nsol'])
    return m['v0']


def _xform_rd_Q(m, rd, Q):
    """Transform rd and Q into mri coords if necessary"""
    mri_rd = rd.copy()
    mri_Q = Q.copy()
    if m['head_mri_t'] is not None:
        mri_rd = apply_trans(m['head_mri_t']['trans'], mri_rd)
        mri_Q = apply_trans(m['head_mri_t']['trans'], mri_Q, False)
    return mri_rd, mri_Q


def _bem_pot(rd, Q, m, els, all_surfs, pot, linear=False):
    """Compute the potentials due to a current dipole"""
    v0 = _get_v0(m)
    mri_rd, mri_Q = _xform_rd_Q(m, rd, Q)
    off = 0
    for surf, mult in zip(m['surfs'], m['source_mult']):
        if linear is False:
            ntri = surf['ntri']
            tri_cent = surf['tri_cent']
            v0[off:off + ntri] = mult * _bem_inf_pot(mri_rd, mri_Q, tri_cent)
            off += ntri
        else:
            rr = surf['rr']
            v0[off:off + len(rr)] = mult * _bem_inf_pot(mri_rd, mri_Q, rr)
            off += len(rr)

    if els is not None:
        sol = els['user_data']
        solution = sol['solution']
    else:
        solution = m['solution']

    pot[:] = np.dot(solution, v0)


def _bem_pot_grad(rd, Q, m, els, all_surfs, xgrad, ygrad, zgrad,
                  linear=False):
    """Compute the potentials due to a current dipole"""
    grads = np.array([xgrad, ygrad, zgrad])
    v0 = _get_v0(m)
    mri_rd, mri_Q = _xform_rd_Q(m, rd, Q)

    ees = np.eye(3)
    for pp, ee in enumerate(ees):
        grad = grads[pp]
        if m['head_mri_t'] is not None:
            ee = apply_trans(m['head_mri_t'], ee, False)

        off = 0
        for surf, mult in zip(m['surfs'], m['source_mult']):
            if linear is False:
                ntri = surf['ntri']
                tri_cent = surf['tri_cent']
                v0[off:off + ntri] = mult * _bem_inf_pot_der(mri_rd, mri_Q,
                                                             tri_cent, ee)
                off += ntri
            else:
                rr = surf['rr']
                v0[off:off + len(rr)] = mult * _bem_inf_pot_der(mri_rd, mri_Q,
                                                                rr, ee)
                off += len(rr)

        if els:
            solution = els['user_data']['solution']
        else:
            solution = m['solution']
        grad[:] = np.dot(solution, v0)


def _bem_pot_els(rd, Q, els, pot, client):
    """This version calculates the potential on all surfaces"""
    m = client
    if m['bem_method'] == 'constant collocation':
        _bem_pot(rd, Q, m, els, False, pot, False)
    elif m['bem_method'] == 'linear collocation':
        _bem_pot(rd, Q, m, els, False, pot, True)
    else:
        raise RuntimeError('Unknown BEM method : %d' % m['bem_method'])


def _bem_pot_grad_els(rd, Q, els, pot, xgrad, ygrad, zgrad, client):
    """This version calculates the potential on all surfaces"""
    m = client
    if m['bem_method'] == 'constant collocation':
        if pot:
            _bem_pot(rd, Q, m, els, False, pot, False)
        _bem_pot_grad(rd, Q, m, els, False, xgrad, ygrad, zgrad, False)
    elif m['bem_method'] == 'linear collocation':
        if pot:
            _bem_pot(rd, Q, m, els, False, pot, True)
        _bem_pot_grad(rd, Q, m, els, False, xgrad, ygrad, zgrad, True)
    else:
        raise RuntimeError('Unknown BEM method : %d' % m['bem_method'])


def _meg_eeg_fwd_one_source_space(a):
    """Compute the MEG or EEG forward solution for one source space"""
    Qs = np.eye(3)
    # XXX Make subfunctions take vectors and vertno
    s = a['s']
    vertno = np.where(s['inuse'])[0]
    p = a['off']
    q = 3 * a['off']
    if a['fixed']:
        if a['field_pot_grad'] is not None and a['res_grad'] is not None:
            for j in vertno:
                a['field_pot_grad'](s['rr'][j], s['nn'][j], a['coils_els'],
                                    a['res'][p], a['res_grad'][q:q + 3],
                                    a['client'])
                q += 3
                p += 1
        else:
            for j in vertno:
                a['field_pot'](s['rr'][j], s['nn'][j], a['coils_els'],
                               a['res'][p], a['client'])
                p += 1
    else:
        if a['field_pot_grad'] and a['res_grad']:
            for j in vertno:
                for k, Qq in enumerate(Qs):
                    if a['comp'] < 0 or a['comp'] == k:
                        a['field_pot_grad'](s['rr'][j], Qq, a['coils_els'],
                                            a['res'][p],
                                            a['res_grad'][q:q + 3],
                                            a['client'])
                    q += 3
                    p += 1
        else:
            for j in vertno:
                if a['vec_field_pot'] is not None:
                    xyz = [a['res'][p], a['res'][p + 1], a['res'][p + 2]]
                    p += 3
                    a['vec_field_pot'](s['rr'][j], a['coils_els'], xyz,
                                       a['client'])
                else:
                    for k, Qq in enumerate(Qs):
                        if a['comp'] < 0 or a['comp'] == k:
                            a['field_pot'](s['rr'][j], Qq, a['coils_els'],
                                           a['res'][p], a['client'])
                        p += 1


def _compute_forward(src, coils_els, comp_coils, comp_data, fixed,
                     bem_model, r0, coil_type, grad):
    """Compute the MEG forward solution"""
    if coil_type == 'meg':
        # Use the new compensated field computation
        # It works the same way independent of whether or not the compensation
        # is in effect
        comp = _make_comp_data(comp_data, coils_els, comp_coils,
                               _bem_field, None, _bem_field_grad,
                               bem_model)

        # Field computation matrices...
        logger.info('')
        logger.info('Composing the field computation matrix...')
        _bem_specify_coils(bem_model, coils_els)

        if comp['set'] is not None and comp['set']['current'] is True:
            logger.info('Composing the field computation matrix '
                        '(compensation coils)...')
            _bem_specify_coils(bem_model, comp['comp_coils'])
        field = _comp_field
        field_grad = _comp_field_grad
        client = comp
        ftype = 'MEG'
    elif coil_type == 'eeg':
        _bem_specify_els(bem_model, coils_els)
        client = bem_model
        field = _bem_pot_els
        field_grad = _bem_pot_grad_els
        ftype = 'EEG'
    else:
        raise ValueError('coil_type must be "meg" or "eeg"')
    vec_field = None

    # Count the sources and allocate space
    n_ch = len(coils_els['coils'])
    nsource = np.sum([s['nuse'] for s in src])
    n_res = nsource if fixed is True else 3 * nsource
    n_res_grad = nsource if fixed is True else 3 * 3 * nsource
    res = np.zeros((n_res, n_ch))
    res_grad = None
    if grad is True:
        res_grad = np.zeros((n_res_grad, n_ch))

    # Set up arguments for the field computation
    extra = 'fixed' if fixed is True else 'free'
    logger.info('Computing %s at %d source locations '
                '(%s orientations)...' % (ftype, nsource, extra))
    one_arg = dict(res=res, res_grad=res_grad, coils_els=coils_els,
                   client=client, fixed=fixed, field_pot=field,
                   vec_field_pot=vec_field, field_pot_grad=field_grad,
                   comp=-1)

    off = 0
    for k, s in enumerate(src):
        one_arg['s'] = s
        one_arg['off'] = off
        _meg_eeg_fwd_one_source_space(one_arg)
        off += s['nuse'] if fixed else 3 * s['nuse']

    return res, res_grad
