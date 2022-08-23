# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD-3-Clause

from collections import defaultdict
from functools import partial

import numpy as np

from ..bem import _check_origin
from ..io.pick import pick_info, pick_types
from ..io import _loc_to_coil_trans, _coil_trans_to_loc, BaseRaw
from ..transforms import _find_vector_rotation
from ..utils import (logger, verbose, check_fname, _check_fname, _pl,
                     _ensure_int, _check_option, _validate_type, _reg_pinv)

from .maxwell import (_col_norm_pinv, _trans_sss_basis, _prep_mf_coils,
                      _get_grad_point_coilsets, _read_cross_talk,
                      _prep_fine_cal)


@verbose
def compute_fine_calibration(raw, n_imbalance=3, t_window=10., ext_order=2,
                             origin=(0., 0., 0.), cross_talk=None,
                             calibration=None, verbose=None):
    """Compute fine calibration from empty-room data.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to use. Should be from an empty-room recording,
        and all channels should be good.
    n_imbalance : int
        Can be 1 or 3 (default), indicating the number of gradiometer
        imbalance components. Only used if gradiometers are present.
    t_window : float
        Time window to use for surface normal rotation in seconds.
        Default is 10.
    %(ext_order_maxwell)s
        Default is 2, which is lower than the default (3) for
        :func:`mne.preprocessing.maxwell_filter` because it tends to yield
        more stable parameter estimates.
    %(origin_maxwell)s
    %(cross_talk_maxwell)s
    calibration : dict | None
        Dictionary with existing calibration. If provided, the magnetometer
        imbalances and adjusted normals will be used and only the gradiometer
        imbalances will be estimated (see step 2 in Notes below).
    %(verbose)s

    Returns
    -------
    calibration : dict
        Fine calibration data.
    count : int
        The number of good segments used to compute the magnetometer
        parameters.

    See Also
    --------
    mne.preprocessing.maxwell_filter

    Notes
    -----
    This algorithm proceeds in two steps, both optimizing the fit between the
    data and a reconstruction of the data based only on an external multipole
    expansion:

    1. Estimate magnetometer normal directions and scale factors. All
       coils (mag and matching grad) are rotated by the adjusted normal
       direction.
    2. Estimate gradiometer imbalance factors. These add point magnetometers
       in just the gradiometer difference direction or in all three directions
       (depending on ``n_imbalance``).

    Magnetometer normal and coefficient estimation (1) is typically the most
    time consuming step. Gradiometer imbalance parameters (2) can be
    iteratively reestimated (for example, first using ``n_imbalance=1`` then
    subsequently ``n_imbalance=3``) by passing the previous ``calibration``
    output to the ``calibration`` input in the second call.

    MaxFilter processes at most 120 seconds of data, so consider cropping
    your raw instance prior to processing. It also checks to make sure that
    there were some minimal usable ``count`` number of segments (default 5)
    that were included in the estimate.

    .. versionadded:: 0.21
    """
    n_imbalance = _ensure_int(n_imbalance, 'n_imbalance')
    _check_option('n_imbalance', n_imbalance, (1, 3))
    _validate_type(raw, BaseRaw, 'raw')
    ext_order = _ensure_int(ext_order, 'ext_order')
    origin = _check_origin(origin, raw.info, 'meg', disp=True)
    _check_option("raw.info['bads']", raw.info['bads'], ([],))
    picks = pick_types(raw.info, meg=True, ref_meg=False)
    if raw.info['dev_head_t'] is not None:
        raise ValueError('info["dev_head_t"] is not None, suggesting that the '
                         'data are not from an empty-room recording')

    info = pick_info(raw.info, picks)  # make a copy and pick MEG channels
    mag_picks = pick_types(info, meg='mag', exclude=())
    grad_picks = pick_types(info, meg='grad', exclude=())

    # Get cross-talk
    ctc, _ = _read_cross_talk(cross_talk, info['ch_names'])

    # Check fine cal
    _validate_type(calibration, (dict, None), 'calibration')

    #
    # 1. Rotate surface normals using magnetometer information (if present)
    #
    cals = np.ones(len(info['ch_names']))
    time_idxs = raw.time_as_index(
        np.arange(0., raw.times[-1], t_window))
    if len(time_idxs) <= 1:
        time_idxs = np.array([0, len(raw.times)], int)
    else:
        time_idxs[-1] = len(raw.times)
    count = 0
    locs = np.array([ch['loc'] for ch in info['chs']])
    zs = locs[mag_picks, -3:].copy()
    if calibration is not None:
        _, calibration, _ = _prep_fine_cal(info, calibration)
        for pi, pick in enumerate(mag_picks):
            idx = calibration['ch_names'].index(info['ch_names'][pick])
            cals[pick] = calibration['imb_cals'][idx]
            zs[pi] = calibration['locs'][idx][-3:]
    elif len(mag_picks) > 0:
        cal_list = list()
        z_list = list()
        logger.info('Adjusting normals for %s magnetometers '
                    '(averaging over %s time intervals)'
                    % (len(mag_picks), len(time_idxs) - 1))
        for start, stop in zip(time_idxs[:-1], time_idxs[1:]):
            logger.info('    Processing interval %0.3f - %0.3f sec'
                        % (start / info['sfreq'], stop / info['sfreq']))
            data = raw[picks, start:stop][0]
            if ctc is not None:
                data = ctc.dot(data)
            z, cal, good = _adjust_mag_normals(info, data, origin, ext_order)
            if good:
                z_list.append(z)
                cal_list.append(cal)
        count = len(cal_list)
        if count == 0:
            raise RuntimeError('No usable segments found')
        cals[:] = np.mean(cal_list, axis=0)
        zs[:] = np.mean(z_list, axis=0)
    if len(mag_picks) > 0:
        for ii, new_z in enumerate(zs):
            z_loc = locs[mag_picks[ii]]
            # Find sensors with same NZ and R0 (should be three for VV)
            idxs = _matched_loc_idx(z_loc, locs)
            # Rotate the direction vectors to the plane defined by new normal
            _rotate_locs(locs, idxs, new_z)
    for ci, loc in enumerate(locs):
        info['chs'][ci]['loc'][:] = loc
    del calibration, zs

    #
    # 2. Estimate imbalance parameters (always done)
    #
    if len(grad_picks) > 0:
        extra = 'X direction' if n_imbalance == 1 else ('XYZ directions')
        logger.info('Computing imbalance for %s gradimeters (%s)'
                    % (len(grad_picks), extra))
        imb_list = list()
        for start, stop in zip(time_idxs[:-1], time_idxs[1:]):
            logger.info('    Processing interval %0.3f - %0.3f sec'
                        % (start / info['sfreq'], stop / info['sfreq']))
            data = raw[picks, start:stop][0]
            if ctc is not None:
                data = ctc.dot(data)
            out = _estimate_imbalance(info, data, cals,
                                      n_imbalance, origin, ext_order)
            imb_list.append(out)
        imb = np.mean(imb_list, axis=0)
    else:
        imb = np.zeros((len(info['ch_names']), n_imbalance))

    #
    # Put in output structure
    #
    assert len(np.intersect1d(mag_picks, grad_picks)) == 0
    imb_cals = [cals[ii:ii + 1] if ii in mag_picks else imb[ii]
                for ii in range(len(info['ch_names']))]
    calibration = dict(ch_names=info['ch_names'], locs=locs, imb_cals=imb_cals)
    return calibration, count


def _matched_loc_idx(mag_loc, all_loc):
    return np.where([np.allclose(mag_loc[-3:], loc[-3:]) and
                     np.allclose(mag_loc[:3], loc[:3]) for loc in all_loc])[0]


def _rotate_locs(locs, idxs, new_z):
    new_z = new_z / np.linalg.norm(new_z)
    old_z = locs[idxs[0]][-3:]
    old_z = old_z / np.linalg.norm(old_z)
    rot = _find_vector_rotation(old_z, new_z)
    for ci in idxs:
        this_trans = _loc_to_coil_trans(locs[ci])
        this_trans[:3, :3] = np.dot(rot, this_trans[:3, :3])
        locs[ci][:] = _coil_trans_to_loc(this_trans)
        np.testing.assert_allclose(locs[ci][-3:], new_z, atol=1e-4)


def _vector_angle(x, y):
    """Get the angle between two vectors in degrees."""
    return np.abs(np.arccos(
        np.clip((x * y).sum(axis=-1) /
                (np.linalg.norm(x, axis=-1) *
                 np.linalg.norm(y, axis=-1)), -1, 1.)))


def _adjust_mag_normals(info, data, origin, ext_order):
    """Adjust coil normals using magnetometers and empty-room data."""
    from scipy.optimize import fmin_cobyla
    # in principle we could allow using just mag or mag+grad, but MF uses
    # just mag so let's follow suit
    mag_scale = 100.
    picks_use = pick_types(info, meg='mag', exclude='bads')
    picks_meg = pick_types(info, meg=True, exclude=())
    picks_mag_orig = pick_types(info, meg='mag', exclude='bads')
    info = pick_info(info, picks_use)  # copy
    data = data[picks_use]
    cals = np.ones((len(data), 1))
    angles = np.zeros(len(cals))
    picks_mag = pick_types(info, meg='mag')
    data[picks_mag] *= mag_scale
    # Transform variables so we're only dealing with good mags
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    all_coils = _prep_mf_coils(info, ignore_ref=True)
    S_tot = _trans_sss_basis(exp, all_coils, coil_scale=mag_scale)
    first_err = _data_err(data, S_tot, cals)
    count = 0
    # two passes: first do the worst, then do all in order
    zs = np.array([ch['loc'][-3:] for ch in info['chs']])
    zs /= np.linalg.norm(zs, axis=-1, keepdims=True)
    orig_zs = zs.copy()
    match_idx = dict()
    locs = np.array([ch['loc'] for ch in info['chs']])
    for pick in picks_mag:
        match_idx[pick] = _matched_loc_idx(locs[pick], locs)
    counts = defaultdict(lambda: 0)
    for ki, kind in enumerate(('worst first', 'in order')):
        logger.info(f'        Magnetometer normal adjustment ({kind}) ...')
        S_tot = _trans_sss_basis(exp, all_coils, coil_scale=mag_scale)
        for pick in picks_mag:
            err = _data_err(data, S_tot, cals, axis=1)

            # First pass: do worst; second pass: do all in order (up to 3x/sen)
            if ki == 0:
                order = list(np.argsort(err[picks_mag]))
                cal_idx = 0
                while len(order) > 0:
                    cal_idx = picks_mag[order.pop(-1)]
                    if counts[cal_idx] < 3:
                        break
                if err[cal_idx] < 2.5:
                    break  # move on to second loop
            else:
                cal_idx = pick
            counts[cal_idx] += 1
            assert cal_idx in picks_mag
            count += 1
            old_z = zs[cal_idx].copy()
            objective = partial(
                _cal_sss_target, old_z=old_z, all_coils=all_coils,
                cal_idx=cal_idx, data=data, cals=cals, match_idx=match_idx,
                S_tot=S_tot, origin=origin, ext_order=ext_order)

            # Figure out the additive term for z-component
            zs[cal_idx] = fmin_cobyla(
                objective, old_z, cons=(), rhobeg=1e-3, rhoend=1e-4,
                disp=False)

            # Do in-place adjustment to all_coils
            cals[cal_idx] = 1. / np.linalg.norm(zs[cal_idx])
            zs[cal_idx] *= cals[cal_idx]
            for idx in match_idx[cal_idx]:
                _rotate_coil(zs[cal_idx], old_z, all_coils, idx, inplace=True)

            # Recalculate S_tot, taking into account rotations
            S_tot = _trans_sss_basis(exp, all_coils)

            # Reprt results
            old_err = err[cal_idx]
            new_err = _data_err(data, S_tot, cals, idx=cal_idx)
            angles[cal_idx] = np.abs(np.rad2deg(_vector_angle(
                zs[cal_idx], orig_zs[cal_idx])))
            ch_name = info['ch_names'][cal_idx]
            logger.debug(
                f'        Optimization step {count:3d} ｜ '
                f'{ch_name} ({counts[cal_idx]}) ｜ '
                f'res {old_err:5.2f}→{new_err:5.2f}% ｜ '
                f'×{cals[cal_idx, 0]:0.3f} ｜ {angles[cal_idx]:0.2f}°')
    last_err = _data_err(data, S_tot, cals)
    # Chunk is usable if all angles and errors are both small
    reason = list()
    max_angle = np.max(angles)
    if max_angle >= 5.:
        reason.append(f'max angle {max_angle:0.2f} >= 5°')
    each_err = _data_err(data, S_tot, cals, axis=-1)[picks_mag]
    n_bad = (each_err > 5.).sum()
    if n_bad:
        reason.append(f'{n_bad} residual{_pl(n_bad)} > 5%')
    reason = ', '.join(reason)
    if reason:
        reason = f' ({reason})'
    good = not bool(reason)
    assert np.allclose(np.linalg.norm(zs, axis=1), 1.)
    logger.info(f'        Fit mismatch {first_err:0.2f}→{last_err:0.2f}%')
    logger.info(f'        Data segment {"" if good else "un"}usable{reason}')
    # Reformat zs and cals to be the n_mags (including bads)
    assert zs.shape == (len(data), 3)
    assert cals.shape == (len(data), 1)
    imb_cals = np.ones(len(picks_meg))
    imb_cals[picks_mag_orig] = cals[:, 0]
    return zs, imb_cals, good


def _data_err(data, S_tot, cals, idx=None, axis=None):
    if idx is None:
        idx = slice(None)
    S_tot = S_tot / cals
    data_model = np.dot(
        np.dot(S_tot[idx], _col_norm_pinv(S_tot.copy())[0]), data)
    err = 100 * (np.linalg.norm(data_model - data[idx], axis=axis) /
                 np.linalg.norm(data[idx], axis=axis))
    return err


def _rotate_coil(new_z, old_z, all_coils, idx, inplace=False):
    """Adjust coils."""
    # Turn NX and NY to the plane determined by NZ
    old_z = old_z / np.linalg.norm(old_z)
    new_z = new_z / np.linalg.norm(new_z)
    rot = _find_vector_rotation(old_z, new_z)  # additional coil rotation
    this_sl = all_coils[5][idx]
    this_rmag = np.dot(rot, all_coils[0][this_sl].T).T
    this_cosmag = np.dot(rot, all_coils[1][this_sl].T).T
    if inplace:
        all_coils[0][this_sl] = this_rmag
        all_coils[1][this_sl] = this_cosmag
    subset = (this_rmag, this_cosmag, np.zeros(this_rmag.shape[0], int),
              1, all_coils[4][[idx]], {0: this_sl})
    return subset


def _cal_sss_target(new_z, old_z, all_coils, cal_idx, data, cals,
                    S_tot, origin, ext_order, match_idx):
    """Evaluate objective function for SSS-based magnetometer calibration."""
    cals[cal_idx] = 1. / np.linalg.norm(new_z)
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    S_tot = S_tot.copy()
    # Rotate necessary coils properly and adjust correct element in c
    for idx in match_idx[cal_idx]:
        this_coil = _rotate_coil(new_z, old_z, all_coils, idx)
        # Replace correct row of S_tot with new value
        S_tot[idx] = _trans_sss_basis(exp, this_coil)
    # Get the GOF
    return _data_err(data, S_tot, cals, idx=cal_idx)


def _estimate_imbalance(info, data, cals, n_imbalance, origin, ext_order):
    """Estimate gradiometer imbalance parameters."""
    mag_scale = 100.
    n_iterations = 3
    mag_picks = pick_types(info, meg='mag', exclude=())
    grad_picks = pick_types(info, meg='grad', exclude=())
    data = data.copy()
    data[mag_picks, :] *= mag_scale
    del mag_picks

    grad_imb = np.zeros((len(grad_picks), n_imbalance))
    exp = dict(origin=origin, int_order=0, ext_order=ext_order)
    all_coils = _prep_mf_coils(info, ignore_ref=True)
    grad_point_coils = _get_grad_point_coilsets(
        info, n_imbalance, ignore_ref=True)
    S_orig = _trans_sss_basis(exp, all_coils, coil_scale=mag_scale)
    S_orig /= cals[:, np.newaxis]
    # Compute point gradiometers for each grad channel
    this_cs = np.array([mag_scale], float)
    S_pt = np.array([_trans_sss_basis(exp, coils, None, this_cs)
                     for coils in grad_point_coils])
    for k in range(n_iterations):
        S_tot = S_orig.copy()
        # In theory we could zero out the homogeneous components with:
        #     S_tot[grad_picks, :3] = 0
        # But in practice it doesn't seem to matter
        S_recon = S_tot[grad_picks]

        # Add influence of point magnetometers
        S_tot[grad_picks, :] += np.einsum('ij,ijk->jk', grad_imb.T, S_pt)

        # Compute multipolar moments
        mm = np.dot(_col_norm_pinv(S_tot.copy())[0], data)

        # Use good channels to recalculate
        prev_imb = grad_imb.copy()
        data_recon = np.dot(S_recon, mm)
        assert S_pt.shape == (n_imbalance, len(grad_picks), S_tot.shape[1])
        khi_pts = (S_pt @ mm).transpose(1, 2, 0)
        assert khi_pts.shape == (len(grad_picks), data.shape[1], n_imbalance)
        residual = data[grad_picks] - data_recon
        assert residual.shape == (len(grad_picks), data.shape[1])
        d = (residual[:, np.newaxis, :] @ khi_pts)[:, 0]
        assert d.shape == (len(grad_picks), n_imbalance)
        dinv, _, _ = _reg_pinv(khi_pts.swapaxes(-1, -2) @ khi_pts, rcond=1e-6)
        assert dinv.shape == (len(grad_picks), n_imbalance, n_imbalance)
        grad_imb[:] = (d[:, np.newaxis] @ dinv)[:, 0]
        # This code is equivalent but hits a np.linalg.pinv bug on old NumPy:
        # grad_imb[:] = np.sum(  # dot product across the time dim
        #     np.linalg.pinv(khi_pts) * residual[:, np.newaxis], axis=-1)
        deltas = (np.linalg.norm(grad_imb - prev_imb) /
                  max(np.linalg.norm(grad_imb), np.linalg.norm(prev_imb)))
        logger.debug(f'        Iteration {k + 1}/{n_iterations}: '
                     f'max ∆ = {100 * deltas.max():7.3f}%')
    imb = np.zeros((len(data), n_imbalance))
    imb[grad_picks] = grad_imb
    return imb


def read_fine_calibration(fname):
    """Read fine calibration information from a .dat file.

    The fine calibration typically includes improved sensor locations,
    calibration coefficients, and gradiometer imbalance information.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    calibration : dict
        Fine calibration information. Key-value pairs are:

        - ``ch_names``
             List of str of the channel names.
        - ``locs``
             Coil location and orientation parameters.
        - ``imb_cals``
             For magnetometers, the calibration coefficients.
             For gradiometers, one or three imbalance parameters.
    """
    # Read new sensor locations
    fname = _check_fname(fname, overwrite='read', must_exist=True)
    check_fname(fname, 'cal', ('.dat',))
    ch_names, locs, imb_cals = list(), list(), list()
    with open(fname, 'r') as fid:
        for line in fid:
            if line[0] in '#\n':
                continue
            vals = line.strip().split()
            if len(vals) not in [14, 16]:
                raise RuntimeError('Error parsing fine calibration file, '
                                   'should have 14 or 16 entries per line '
                                   'but found %s on line:\n%s'
                                   % (len(vals), line))
            # `vals` contains channel number
            ch_name = vals[0]
            if len(ch_name) in (3, 4):  # heuristic for Neuromag fix
                try:
                    ch_name = int(ch_name)
                except ValueError:  # something other than e.g. 113 or 2642
                    pass
                else:
                    ch_name = 'MEG' + '%04d' % ch_name
            # (x, y, z), x-norm 3-vec, y-norm 3-vec, z-norm 3-vec
            # and 1 or 3 imbalance terms
            ch_names.append(ch_name)
            locs.append(np.array(vals[1:13], float))
            imb_cals.append(np.array(vals[13:], float))
    locs = np.array(locs)
    return dict(ch_names=ch_names, locs=locs, imb_cals=imb_cals)


def write_fine_calibration(fname, calibration):
    """Write fine calibration information to a .dat file.

    Parameters
    ----------
    fname : str
        The filename to write out.
    calibration : dict
        Fine calibration information.
    """
    fname = _check_fname(fname, overwrite=True)
    check_fname(fname, 'cal', ('.dat',))
    keys = ('ch_names', 'locs', 'imb_cals')
    with open(fname, 'wb') as cal_file:
        for ch_name, loc, imb_cal in zip(*(calibration[key] for key in keys)):
            cal_line = np.concatenate([loc, imb_cal]).round(6)
            cal_line = ' '.join(f'{c:0.6f}' for c in cal_line)
            cal_file.write(f'{ch_name} {cal_line}\n'.encode('ASCII'))
