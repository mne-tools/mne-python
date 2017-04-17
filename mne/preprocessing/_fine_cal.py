# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..io.pick import pick_info, pick_types
from ..io import _loc_to_coil_trans, _coil_trans_to_loc
from ..transforms import _find_vector_rotation
from ..fixes import partial
from ..utils import logger, verbose, check_fname, _check_fname

from .maxwell import (_col_norm_pinv, _trans_sss_basis, _prep_mf_coils,
                      _get_grad_point_coilsets)


@verbose
def calculate_fine_calibration(raw, n_imbalance=3, t_window=10., n_jobs=1,
                               verbose=None):
    """Compute fine calibration from empty-room data

    Parameters
    ----------
    raw : instance of Raw
        The raw data to use. Should be from an empty-room recording.
    n_imbalance : int
        Can be 1 or 3 (default), indicating the number of gradiometer
        imbalance components. Only used if gradiometers are present.
    t_window : float
        Time window to use for surface normal rotation in seconds.
        Default is 10.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized. Currently this
        is only used for the resampling step.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    fine_cal : dict
        Fine calibration data.

    See Also
    --------
    mne.preprocessing.maxwell_filter

    Notes
    -----
    .. versionadded:: 0.12
    """
    n_imbalance = int(n_imbalance)
    if n_imbalance not in (1, 3):
        raise ValueError('Can only have 1 (X) or 3 (X, Y Z) imbalance '
                         'coefficients, not %d' % n_imbalance)

    origin = np.zeros(3)

    # Select MEG channels, use at most 2 minutes of raw data
    meg_picks = pick_types(raw.info, meg=True, exclude=())
    meg_channels = [raw.ch_names[ii] for ii in meg_picks]
    t_max = min(120. + 0.5 / raw.info['sfreq'], raw.times[-1])
    logger.info('Cropping data to %0.1f sec and loading data' % t_max)
    raw = raw.crop(0, t_max, copy=True).load_data().pick_channels(meg_channels)
    logger.info('Downsampling data to 200 Hz')
    raw.resample(200., n_jobs=n_jobs, verbose=False)
    del t_max, meg_picks, meg_channels
    # raw.filter(None, 20., l_trans_bandwidth=5., filter_length='1s')
    mag_picks = pick_types(raw.info, meg='mag', exclude=())
    grad_picks = pick_types(raw.info, meg='grad', exclude=())

    #
    # Rotate surface normals using magnetometer information (if present)
    #
    cals = np.ones(len(raw.ch_names))
    info = deepcopy(raw.info)  # will be modified by _estimate_normals
    data = raw[:][0]
    time_idxs = raw.time_as_index(
        np.arange(0., raw.times[-1], t_window))
    if len(time_idxs) <= 1:
        time_idxs = np.array([0, len(raw.times)], int)
    else:
        time_idxs[-1] = len(raw.times)
    del raw
    # order 2 should be good for numerical stability
    ext_order = 2
    if len(mag_picks) > 0:
        cal_list = list()
        z_list = list()
        logger.info('Adjusting normals for %s magnetometers '
                    '(averaging over %s time intervals)'
                    % (len(mag_picks), len(time_idxs) - 1))
        for start, stop in zip(time_idxs[:-1], time_idxs[1:]):
            logger.info('    Processing interval %0.3f - %0.3f sec'
                        % (start / info['sfreq'], stop / info['sfreq']))
            out = _adjust_normals(info, data[:, start:stop], origin, ext_order)
            z_list.append(out[0])
            cal_list.append(out[1])
        cals[mag_picks] = np.mean(cal_list, axis=0)
        zs = np.mean(z_list, axis=0)
        ch_trans = np.array([_loc_to_coil_trans(ch['loc'])
                             for ch in info['chs']])
        for ii, new_z in enumerate(zs):
            this_trans = ch_trans[mag_picks[ii]]
            old_z = this_trans[:3, 2].copy()
            # Find sensors with same NZ and R0
            idxs = np.where([np.allclose(ct[:3, 2:], this_trans[:3, 2:])
                             for ct in ch_trans])[0]
            # Rotate the direction vectors to the plane defined by new normal
            rot = _find_vector_rotation(old_z, new_z)
            angle = np.rad2deg(np.arccos(np.clip((
                np.trace(rot) - 1) / 2, 0, 1)))
            this_trans[:3, :3] = np.dot(rot, this_trans[:3, :3])
            pl = '' if len(idxs) == 1 else 's'
            logger.info(u'    Rotating %s normal%s (using %s) by %0.1f°'
                        % (len(idxs), pl, info['ch_names'][mag_picks[ii]],
                           angle))
            for ci in idxs:
                info['chs'][ci]['loc'] = _coil_trans_to_loc(this_trans)
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
            out = _estimate_imbalance(info, data[:, start:stop], cals,
                                      n_imbalance, origin, ext_order)
            imb_list.append(out)
        imb = np.mean(imb_list, axis=0)
    else:
        imb = np.zeros((len(info['ch_names']), n_imbalance))
    # Put in appropriate structure
    assert len(np.intersect1d(mag_picks, grad_picks)) == 0
    imb_cals = [np.array([cals[ii]]) if ii in mag_picks else imb[ii]
                for ii in range(len(info['ch_names']))]
    ch_names = [ch_dict['logno'] for ch_dict in info['chs']]
    fine_cal = dict(ch_names=ch_names, locs=locs, imb_cals=imb_cals)

    return fine_cal


def _norm(a, axis=None):
    """Helper to quickly calculate a norm"""
    return np.sqrt(np.sum(a * a, axis=axis))


def _vector_angle(x, y):
    """Helper to get the angle between two vectors in degrees"""
    return np.rad2deg(np.abs(np.arccos(
        np.clip((x * y).sum(axis=-1) /
                (_norm(x, axis=-1) * _norm(y, axis=-1)), -1, 1.))))


def _adjust_normals(info, data, origin, ext_order):
    """Helper to adjust coil normals using magnetometers and empty-room data"""
    from scipy.optimize import fmin_cobyla
    picks_mag = pick_types(info, meg='mag', exclude=())
    info_mag = pick_info(info, picks_mag)
    good = pick_types(info_mag, meg=True, exclude='bads')
    # Transform variables so we're only dealing with mags
    data = data[picks_mag]
    zs = np.array([ch['loc'][9:12] for ch in info_mag['chs']], float)
    zs /= _norm(zs, axis=1)[:, np.newaxis]  # ensure we're unit len
    orig_zs = zs.copy()
    cals = np.ones(len(picks_mag))  # this is now true
    data_cal = data.copy()
    # Subset of goods within mag subselection
    err = np.zeros(len(picks_mag))
    # We use external order 2, as we assume no sources of significant field
    # closer than the walls of the shielded room
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    all_coils = _prep_mf_coils(info_mag, ignore_ref=True)
    rhos = ((1e-2, 1e-3), (1e-3, 1e-4), (1e-3, 1e-4))
    for k, (rhobeg, rhoend) in enumerate(rhos):
        logger.debug('        Iteration %s/%s (rho=%0.0e)'
                     % (k + 1, len(rhos), rhoend))
        done = np.zeros(len(cals), bool)
        data_norm = _norm(data, axis=1)
        for ii in range(len(cals)):
            S_tot = _trans_sss_basis(exp, all_coils)
            data_model = np.dot(
                np.dot(S_tot / cals[:, np.newaxis],
                       _col_norm_pinv(S_tot[good])[0]), data_cal[good])
            with np.errstate(invalid='raise'):
                err = _norm(data_model - data, axis=1) / data_norm
            tot_err = _norm(data_model[good] - data[good]) / _norm(data[good])
            if k == len(rhos) - 1:
                # We want all magnetometers to be calibrated
                cal_idx = ii
            else:
                if k == 0:
                    first_err = tot_err
                cal_idx = np.where(~done)[0][np.argmax(err[~done])]
            done[cal_idx] = True
            if cal_idx in good:
                old_z = zs[cal_idx].copy()
                objective = partial(
                    _cal_sss_target, old_z=old_z, all_coils=all_coils,
                    cal_idx=cal_idx, data_cal=data_cal, data_ch=data[cal_idx],
                    good=good, S_tot=S_tot, origin=origin, ext_order=ext_order)

                # Figure out the additive term for z-component
                zs[cal_idx] = fmin_cobyla(objective, old_z, cons=(),
                                          rhobeg=rhobeg, rhoend=rhoend,
                                          disp=False)

                # Do in-place adjustment to all_coils
                _adjust_coils(zs[cal_idx], old_z, all_coils, cal_idx,
                              return_subset=False)
                cals[cal_idx] = 1. / _norm(zs[cal_idx])
                data_cal[cal_idx] = data[cal_idx] * cals[cal_idx]

                # Report
                angle = _vector_angle(zs[cal_idx], orig_zs[cal_idx])
                logger.debug(
                    u'        Optimized %s | err=%0.5f | ×%0.4f | %0.1f°'
                    % (info_mag['ch_names'][cal_idx], tot_err,
                       cals[cal_idx], np.abs(angle)))
            if k == len(rhos) - 1:
                last_err = _norm(data_model[good] -
                                 data[good]) / _norm(data[good])
    zs *= cals[:, np.newaxis]
    assert np.allclose(_norm(zs, axis=1), 1.)
    logger.info('        Fit mismatch %0.5f -> %0.5f'
                % (first_err, last_err))
    return zs, cals


def _adjust_coils(new_z, old_z, all_coils, cal_idx, return_subset):
    """Helper to adjust coils"""
    # Turn NX and NY to the plane determined by NZ
    old_z = old_z / _norm(old_z)
    new_z = new_z / _norm(new_z)
    rot = _find_vector_rotation(old_z, new_z)  # additional coil rotation
    this_sl = all_coils[5][cal_idx]
    this_rmag = np.dot(rot, all_coils[0][this_sl].T).T
    this_cosmag = np.dot(rot, all_coils[1][this_sl].T).T
    if return_subset:
        all_coils = (this_rmag, this_cosmag, np.zeros(this_rmag.shape[0], int),
                     1, np.array([True]), {0: this_sl})
        return all_coils
    else:
        all_coils[0][this_sl] = this_rmag
        all_coils[1][this_sl] = this_cosmag


def _cal_sss_target(new_z, old_z, all_coils, cal_idx, data_cal, data_ch,
                    good, S_tot, origin, ext_order):
    """Objective function for SSS-based magnetometer calibration"""
    this_cal = 1. / _norm(new_z)
    data_cal[cal_idx] = data_ch * this_cal
    # Rotate necessary coils properly and adjust correct element in c
    these_coils = _adjust_coils(new_z, old_z, all_coils, cal_idx,
                                return_subset=True)
    # Replace correct row of S_tot with new value
    exp = dict(int_order=0, ext_order=ext_order, origin=origin)
    S_tot = S_tot.copy()
    S_tot[cal_idx] = _trans_sss_basis(exp, these_coils)
    # Get the GOF
    data_model = np.dot(np.dot(S_tot[cal_idx] / this_cal,
                               _col_norm_pinv(S_tot[good])[0]), data_cal[good])
    residual = _norm(data_model - data_ch) / _norm(data_ch)
    return residual


def _estimate_imbalance(info, data, cals, n_imbalance, origin, ext_order,
                        mag_scale=100.):
    """Helper to estimate gradiometer imbalance parameters"""
    n_iterations = 3
    mag_picks = pick_types(info, meg='mag', exclude=())
    grad_picks = pick_types(info, meg='grad', exclude=())
    good_picks = pick_types(info, meg=True, exclude='bads')
    data = data.copy()
    data[mag_picks, :] *= mag_scale
    del mag_picks
    data_good_cal = data[good_picks]
    data_good_cal /= cals[good_picks][:, np.newaxis]

    imb = np.zeros((len(data), n_imbalance))
    good_grad_picks = np.intersect1d(grad_picks, good_picks)
    S_pt = np.zeros((len(grad_picks), 1))
    S_pt_good_grad = np.zeros((len(good_grad_picks), n_imbalance))
    exp = dict(origin=origin, int_order=0, ext_order=ext_order)
    all_coils = _prep_mf_coils(info, ignore_ref=True)
    grad_point_coils = _get_grad_point_coilsets(info, n_imbalance,
                                                ignore_ref=True)  # XXX REF?
    S_orig = _trans_sss_basis(exp, all_coils, coil_scale=mag_scale)
    # Compute point gradiometers for each grad channel
    this_cs = np.array([mag_scale], float)
    S_pt = np.array([_trans_sss_basis(exp, coils, None, this_cs)
                     for coils in grad_point_coils])
    good_pt_picks = np.searchsorted(grad_picks, good_grad_picks)
    for k in range(n_iterations):
        logger.debug('        Iteration %s/%s' % (k + 1, n_iterations))
        S_tot = S_orig.copy()
        S_good_grad = S_tot[good_grad_picks]

        # Add influence of point magnetometers
        S_tot[grad_picks, :] += np.einsum('ji,ijk->jk', imb[grad_picks], S_pt)

        # Compute multipolar moments
        mm = np.dot(_col_norm_pinv(S_tot[good_picks])[0], data_good_cal)

        # Use good channels to recalculate
        S_good_grad[:, :3] = 0  # Homogeneous components
        S_pt_good_grad = S_pt[:, good_pt_picks]
        khis = np.dot(S_good_grad, mm)
        khi_pts = np.dot(S_pt_good_grad, mm).transpose(1, 2, 0)
        for ii, pick in enumerate(good_grad_picks):
            old_imb = imb[pick].copy()
            imb[pick, :] = np.dot(linalg.pinv(khi_pts[ii]),
                                  data[pick] - khis[ii])
            imb_str = ', '.join('% 6.4f' % imb_ for imb_ in imb[pick])
            delta = np.sqrt(np.mean((imb[pick] - old_imb) ** 2))
            logger.debug(u'        Imbalance for %s: [%s] (∆=%0.6f)'
                         % (info['ch_names'][pick], imb_str, delta))
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
    _check_fname(fname, overwrite='read', must_exist=True)
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
    _check_fname(fname, overwrite=True)
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
