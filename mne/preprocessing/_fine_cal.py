# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy import linalg

from ..io.pick import pick_info, pick_types
from ..io import _loc_to_coil_trans, _coil_trans_to_loc, BaseRaw
from ..transforms import _find_vector_rotation, apply_trans
from ..utils import (logger, verbose, check_fname, _check_fname,
                     _ensure_int, _check_option, _validate_type)

from .maxwell import (_col_norm_pinv, _trans_sss_basis, _prep_mf_coils,
                      _get_grad_point_coilsets, _read_cross_talk)


@verbose
def calculate_fine_calibration(raw, n_imbalance=3, t_window=10.,
                               cross_talk=None, verbose=None):
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
    %(maxwell_cross)s
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    fine_cal : dict
        Fine calibration data.
    count : int
        The number of good segments used to compute the magnetometer
        parameters.

    See Also
    --------
    mne.preprocessing.maxwell_filter

    Notes
    -----
    MaxFilter processes at most 120 seconds of data, so consider cropping
    your instance.

    .. versionadded:: 0.21
    """
    n_imbalance = _ensure_int(n_imbalance, 'n_imbalance')
    _check_option('n_imbalance', n_imbalance, (1, 3))
    origin = np.zeros(3)
    _validate_type(raw, BaseRaw, 'raw')
    raw = raw.copy().pick_types(meg=True, ref_meg=False, exclude=())
    _check_option("raw.info['bads']", raw.info['bads'], ([],))
    if raw.info['dev_head_t'] is not None:
        raise ValueError('info["dev_head_t"] is not None, suggesting that the '
                         'data are not from an empty-room recording')

    # Resample to speed up processing
    logger.info('Downsampling to 90 Hz ...')
    raw.resample(90., verbose=False)  # also loads the data
    info = raw.info  # already copied raw, okay to modify this in place
    mag_picks = pick_types(info, meg='mag', exclude=())
    grad_picks = pick_types(info, meg='grad', exclude=())

    # Get cross-talk
    ctc, _ = _read_cross_talk(cross_talk, info['ch_names'])

    #
    # Rotate surface normals using magnetometer information (if present)
    #
    cals = np.ones(len(raw.ch_names))
    time_idxs = raw.time_as_index(
        np.arange(0., raw.times[-1], t_window))
    if len(time_idxs) <= 1:
        time_idxs = np.array([0, len(raw.times)], int)
    else:
        time_idxs[-1] = len(raw.times)
    ext_order = 3
    count = 0
    if len(mag_picks) > 0:
        cal_list = list()
        z_list = list()
        logger.info('Adjusting normals for %s magnetometers '
                    '(averaging over %s time intervals)'
                    % (len(mag_picks), len(time_idxs) - 1))
        for start, stop in zip(time_idxs[:-1], time_idxs[1:]):
            logger.info('    Processing interval %0.3f - %0.3f sec'
                        % (start / info['sfreq'], stop / info['sfreq']))
            data = raw[:, start:stop][0]
            if ctc is not None:
                data = ctc.dot(data)
            z, cal, good = _adjust_mag_normals(info, data, origin, ext_order)
            if good:
                z_list.append(z)
                cal_list.append(cal)
        count = len(cal_list)
        if count == 0:
            raise RuntimeError('No usable segments found')
        cals[mag_picks] = np.mean(cal_list, axis=0)
        zs = np.mean(z_list, axis=0)
        info_ch_locs = np.array([ch['loc'] for ch in info['chs']])
        for ii, new_z in enumerate(zs):
            info_loc = info_ch_locs[mag_picks[ii]]
            # Find sensors with same NZ and R0 (should be three for VV)
            idxs = np.where([np.allclose(info_loc[-3:], info_ch_loc[-3:]) and
                             np.allclose(info_loc[:3], info_ch_loc[:3])
                             for info_ch_loc in info_ch_locs])[0]
            # Rotate the direction vectors to the plane defined by new normal
            rot = np.eye(4)
            rot[:3, :3] = _find_vector_rotation(info_loc[-3:], new_z)
            for ci in idxs:
                this_trans = _loc_to_coil_trans(info_ch_locs[ci])
                this_trans[:3, :3] = apply_trans(rot, this_trans[:3, :3])
                info['chs'][ci]['loc'][:] = _coil_trans_to_loc(this_trans)
    locs = np.array([ch['loc'] for ch in info['chs']])

    #
    # Estimate imbalance parameters
    #
    if len(grad_picks) > 0:
        extra = 'X direction' if n_imbalance == 1 else ('XYZ directions')
        logger.info('Computing imbalance for %s gradimeters (%s)'
                    % (len(grad_picks), extra))
        imb_list = list()
        for start, stop in zip(time_idxs[:-1], time_idxs[1:]):
            logger.info('    Processing interval %0.3f - %0.3f sec'
                        % (start / info['sfreq'], stop / info['sfreq']))
            data = raw[:, start:stop][0]
            if ctc is not None:
                data = ctc.dot(data)
            out = _estimate_imbalance(info, data, cals,
                                      n_imbalance, origin, ext_order)
            imb_list.append(out)
        imb = np.mean(imb_list, axis=0)
    else:
        imb = np.zeros((len(info['ch_names']), n_imbalance))
    # Put in appropriate structure
    assert len(np.intersect1d(mag_picks, grad_picks)) == 0
    imb_cals = [cals[ii:ii + 1] if ii in mag_picks else imb[ii]
                for ii in range(len(info['ch_names']))]
    ch_names = [ch_dict['ch_name'] for ch_dict in info['chs']]
    fine_cal = dict(ch_names=ch_names, locs=locs, imb_cals=imb_cals)

    return fine_cal, count


def _vector_angle(x, y):
    """Get the angle between two vectors in degrees."""
    return np.abs(np.arccos(
        np.clip((x * y).sum(axis=-1) /
                (np.linalg.norm(x, axis=-1) *
                 np.linalg.norm(y, axis=-1)), -1, 1.)))


def _adjust_mag_normals(info, data, origin, ext_order):
    """Adjust coil normals using magnetometers and empty-room data."""
    from scipy.optimize import fmin_cobyla
    picks_mag = pick_types(info, meg='mag', exclude=())
    picks_mag_good = pick_types(info, meg='mag', exclude='bads')
    all_zs = np.array([info['chs'][pick]['loc'][9:12]
                       for pick in picks_mag], float)
    good_subpicks = np.searchsorted(picks_mag, picks_mag_good)
    zs = all_zs[good_subpicks]
    # Transform variables so we're only dealing with good mags
    info = pick_info(info, picks_mag_good)
    data = data[picks_mag_good]
    del picks_mag, picks_mag_good
    # Now get our good sub-picks
    zs /= np.linalg.norm(zs, axis=1, keepdims=True)  # ensure we're unit len
    orig_zs = zs.copy()
    cals = np.ones((len(data), 1))  # this is now true
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    all_coils = _prep_mf_coils(info, ignore_ref=True)
    S_tot = _trans_sss_basis(exp, all_coils)
    first_err = _data_err(data, S_tot, cals)
    count = 0
    # two passes: first do the worst, then do all in order
    angles = np.zeros(len(cals))
    for ki, kind in enumerate(('worst first', 'in order')):
        logger.info(f'        Magnetometer normal adjustment ({kind}) ...')
        S_tot = _trans_sss_basis(exp, all_coils)
        for ii in range(len(cals)):
            err = _data_err(data, S_tot, cals, axis=1)

            # First pass: do worst; second pass: do all in order
            cal_idx = np.argmax(err) if ki == 0 else ii
            if ki == 0 and err[cal_idx] < 2.5:
                break  # move on to second loop
            count += 1
            old_z = zs[cal_idx].copy()
            objective = partial(
                _cal_sss_target, old_z=old_z, all_coils=all_coils,
                cal_idx=cal_idx, data=data, cals=cals,
                S_tot=S_tot, origin=origin, ext_order=ext_order)

            # Figure out the additive term for z-component
            zs[cal_idx] = fmin_cobyla(
                objective, old_z, cons=(), rhobeg=1e-3, rhoend=1e-4,
                disp=False)

            # Do in-place adjustment to all_coils
            _adjust_coils(zs[cal_idx], old_z, all_coils, cal_idx, inplace=True)
            cals[cal_idx] = 1. / np.linalg.norm(zs[cal_idx])

            # Recalculate S_tot, taking into account rotations
            S_tot = _trans_sss_basis(exp, all_coils)

            # Reprt results
            old_err = err[cal_idx]
            new_err = _data_err(data, S_tot, cals, idx=cal_idx)
            angles[cal_idx] = np.abs(np.rad2deg(_vector_angle(
                zs[cal_idx], orig_zs[cal_idx])))
            ch_name = info['ch_names'][cal_idx]
            logger.debug(
                f'        Optimization step {count:3d} ｜ {ch_name} ｜ '
                f'res {old_err:5.2f}→{new_err:5.2f}% ｜ '
                f'×{cals[cal_idx, 0]:0.3f} ｜ {angles[cal_idx]:0.2f}°')
    last_err = _data_err(data, S_tot, cals)
    # Chunk is usable if all angles and errors are both small
    reason = list()
    max_angle = np.max(angles)
    if max_angle >= 5.:
        reason.append(f'max angle {max_angle:0.2f} >= 5°')
    each_err = _data_err(data, S_tot, cals, axis=-1)
    n_bad = (each_err > 5.).sum()
    if n_bad:
        reason.append(f'{n_bad} residual{_pl(n_bad)} > 5%')
    reason = ', '.join(reason)
    if reason:
        reason = f' ({reason})'
    good = not bool(reason)
    zs *= cals
    assert np.allclose(np.linalg.norm(zs, axis=1), 1.)
    logger.info(f'        Fit mismatch {first_err:0.2f}→{last_err:0.2f}%')
    logger.info('        Data segment '
                f'{"" if good else "un"}usable{reason}')
    # Reformat zs and cals to be the n_mags (including bads)
    all_cals = np.ones(len(all_zs))
    all_zs[good_subpicks] = zs
    all_cals[good_subpicks] = cals[:, 0]
    return all_zs, all_cals, good


def _data_err(data, S_tot, cals, idx=None, axis=None):
    if idx is None:
        idx = slice(None)
    S_tot = S_tot / cals
    data_model = np.dot(
        np.dot(S_tot[idx], _col_norm_pinv(S_tot.copy())[0]), data)
    err = 100 * (np.linalg.norm(data_model - data[idx], axis=axis) /
                 np.linalg.norm(data[idx], axis=axis))
    return err


def _adjust_coils(new_z, old_z, all_coils, cal_idx, inplace=False):
    """Adjust coils."""
    # Turn NX and NY to the plane determined by NZ
    old_z = old_z / np.linalg.norm(old_z)
    new_z = new_z / np.linalg.norm(new_z)
    rot = _find_vector_rotation(old_z, new_z)  # additional coil rotation
    this_sl = all_coils[5][cal_idx]
    this_rmag = np.dot(rot, all_coils[0][this_sl].T).T
    this_cosmag = np.dot(rot, all_coils[1][this_sl].T).T
    if inplace:
        all_coils[0][this_sl] = this_rmag
        all_coils[1][this_sl] = this_cosmag
    subset = (this_rmag, this_cosmag, np.zeros(this_rmag.shape[0], int),
              1, np.array([True]), {0: this_sl})
    return subset


def _cal_sss_target(new_z, old_z, all_coils, cal_idx, data, cals,
                    S_tot, origin, ext_order):
    """Evaluate objective function for SSS-based magnetometer calibration."""
    cals[cal_idx] = 1. / np.linalg.norm(new_z)
    # Rotate necessary coils properly and adjust correct element in c
    these_coils = _adjust_coils(new_z, old_z, all_coils, cal_idx)
    # Replace correct row of S_tot with new value
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    S_tot = S_tot.copy()
    S_tot[cal_idx] = _trans_sss_basis(exp, these_coils)
    # Get the GOF
    return _data_err(data, S_tot, cals, idx=cal_idx)


def _estimate_imbalance(info, data, cals, n_imbalance, origin, ext_order,
                        mag_scale=100.):
    """Estimate gradiometer imbalance parameters."""
    n_iterations = 4
    mag_picks = pick_types(info, meg='mag', exclude=())
    grad_picks = pick_types(info, meg='grad', exclude=())
    data = data.copy()
    data[mag_picks, :] *= mag_scale
    del mag_picks
    data_good = data.copy()

    imb = np.zeros((len(data), n_imbalance))
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
    for k in range(n_iterations, 1):
        S_tot = S_orig.copy()
        S_grad = S_tot[grad_picks]

        # Add influence of point magnetometers
        S_tot[grad_picks, :] += np.einsum('ji,ijk->jk', imb[grad_picks], S_pt)

        # Compute multipolar moments
        mm = np.dot(_col_norm_pinv(S_tot.copy())[0], data_good)

        # Use good channels to recalculate
        S_grad[:, :3] = 0  # Homogeneous components
        khis = np.dot(S_grad, mm)
        assert S_pt.shape == (n_imbalance, len(grad_picks), S_tot.shape[1])
        khi_pts = np.dot(S_pt, mm).transpose(1, 2, 0)
        assert khi_pts.shape == (len(grad_picks), data.shape[1], n_imbalance)
        old_imbs = imb.copy()
        for ii, pick in enumerate(grad_picks):
            imb[pick, :] = np.dot(linalg.pinv(khi_pts[ii]),
                                  data[pick] - khis[ii])
        deltas = np.sqrt(np.mean((imb - old_imbs) ** 2, axis=1))
        logger.debug(f'        Iteration {k}/{n_iterations}: '
                     f'max ∆={deltas.max():0.6f}')
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
        Fine calibration information.
    """
    # Read new sensor locations
    fname = _check_fname(fname, overwrite='read', must_exist=True)
    check_fname(fname, 'cal', ('.dat',))
    ch_names = list()
    locs = list()
    imb_cals = list()
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
            ch_names.append(ch_name)
            # (x, y, z), x-norm 3-vec, y-norm 3-vec, z-norm 3-vec
            locs.append(np.array([float(x) for x in vals[1:13]]))
            # and 1 or 3 imbalance terms
            imb_cals.append([float(x) for x in vals[13:]])
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

    with open(fname, 'wb') as cal_file:
        for ci, chan in enumerate(calibration['ch_names']):
            # Write string containing 1) channel, 2) loc info, 3) calib info
            # with field widths (e.g., %.6f) chosen to match how Elekta writes
            # them out
            cal_line = np.concatenate([calibration['locs'][ci],
                                       calibration['imb_cals'][ci]]).round(6)
            # Write string containing 1) channel, 2) loc info, 3) calib info
            cal_str = str(chan) + ' ' + ' '.join(map(lambda x: "%.6f" % x,
                                                     cal_line))

            cal_file.write((cal_str + '\n').encode('ASCII'))
