# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
from scipy import linalg

from .io.pick import pick_types, pick_channels
from .io.base import _BaseRaw
from .io.constants import FIFF
from .forward import (_magnetic_dipole_field_vec, _create_meg_coils,
                      _concatenate_coils)
from .cov import make_ad_hoc_cov, _get_whitener_data
from .transforms import apply_trans, invert_transform
from .utils import verbose, logger, check_version, use_log_level
from .fixes import partial
from .externals.six import string_types

# XXX hpicons?
# XXX use distances from digitization, not initial fit?


# ############################################################################
# Reading from text or FIF file

@verbose
def get_chpi_positions(raw, t_step=None, return_quat=False, verbose=None):
    """Extract head positions

    Note that the raw instance must have CHPI channels recorded.

    Parameters
    ----------
    raw : instance of Raw | str
        Raw instance to extract the head positions from. Can also be a
        path to a Maxfilter head position estimation log file (str).
    t_step : float | None
        Sampling interval to use when converting data. If None, it will
        be automatically determined. By default, a sampling interval of
        1 second is used if processing a raw data. If processing a
        Maxfilter log file, this must be None because the log file
        itself will determine the sampling interval.
    return_quat : bool
        If True, also return the quaternions.

        .. versionadded:: 0.11

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    translation : ndarray, shape (N, 3)
        Translations at each time point.
    rotation : ndarray, shape (N, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (N,)
        The time points.
    quat : ndarray, shape (N, 3)
        The quaternions. Only returned if ``return_quat`` is True.

    Notes
    -----
    The digitized HPI head frame y is related to the frame position X as:

        Y = np.dot(rotation, X) + translation

    Note that if a Maxfilter log file is being processed, the start time
    may not use the same reference point as the rest of mne-python (i.e.,
    it could be referenced relative to raw.first_samp or something else).
    """
    if isinstance(raw, _BaseRaw):
        # for simplicity, we'll sample at 1 sec intervals like maxfilter
        if t_step is None:
            t_step = 1.0
        t_step = float(t_step)
        picks = pick_types(raw.info, meg=False, ref_meg=False,
                           chpi=True, exclude=[])
        if len(picks) == 0:
            raise RuntimeError('raw file has no CHPI channels')
        time_idx = raw.time_as_index(np.arange(0, raw.times[-1], t_step))
        data = [raw[picks, ti] for ti in time_idx]
        t = np.array([d[1] for d in data])
        data = np.array([d[0][:, 0] for d in data])
        data = np.c_[t, data]
    else:
        if not isinstance(raw, string_types):
            raise TypeError('raw must be an instance of Raw or string')
        if not op.isfile(raw):
            raise IOError('File "%s" does not exist' % raw)
        if t_step is not None:
            raise ValueError('t_step must be None if processing a log')
        data = np.loadtxt(raw, skiprows=1)  # first line is header, skip it
        data.shape = (-1, 10)  # ensure it's the right size even if empty
    out = _quats_to_trans_rot_t(data)
    if return_quat:
        out = out + (data[:, 1:4],)
    return out


def _quats_to_trans_rot_t(quats):
    """Convert Maxfilter-formatted head position quaternions

    Parameters
    ----------
    quats : ndarray, shape (N, 10)
        Maxfilter-formatted quaternions.

    Returns
    -------
    translation : ndarray, shape (N, 3)
        Translations at each time point.
    rotation : ndarray, shape (N, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (N,)
        The time points.

    See Also
    --------
    calculate_chpi_positions
    get_chpi_positions
    """
    t = quats[:, 0].copy()
    rotation = _quat_to_rot(quats[:, 1:4])
    translation = quats[:, 4:7].copy()
    return translation, rotation, t


def _quat_to_rot(q):
    """Helper to convert quaternions to rotations"""
    # z = a + bi + cj + dk
    b, c, d = q[..., 0], q[..., 1], q[..., 2]
    bb, cc, dd = b * b, c * c, d * d
    # use max() here to be safe in case roundoff errs put us over
    aa = np.maximum(1. - bb - cc - dd, 0.)
    a = np.sqrt(aa)
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    rotation = np.array([(aa + bb - cc - dd, bc_2 - ad_2, bd_2 + ac_2),
                         (bc_2 + ad_2, aa + cc - bb - dd, cd_2 - ab_2),
                         (bd_2 - ac_2, cd_2 + ab_2, aa + dd - bb - cc),
                         ])
    if q.ndim > 1:
        rotation = np.rollaxis(np.rollaxis(rotation, 1, q.ndim + 1), 0, q.ndim)
    return rotation


def _one_rot_to_quat(rot):
    """Convert a rotation matrix to quaternions"""
    # see e.g. http://www.euclideanspace.com/maths/geometry/rotations/
    #                 conversions/matrixToQuaternion/
    t = 1. + rot[0] + rot[4] + rot[8]
    if t > np.finfo(rot.dtype).eps:
        s = np.sqrt(t) * 2.
        qx = (rot[7] - rot[5]) / s
        qy = (rot[2] - rot[6]) / s
        qz = (rot[3] - rot[1]) / s
        # qw = 0.25 * s
    elif rot[0] > rot[4] and rot[0] > rot[8]:
        s = np.sqrt(1. + rot[0] - rot[4] - rot[8]) * 2.
        qx = 0.25 * s
        qy = (rot[1] + rot[3]) / s
        qz = (rot[2] + rot[6]) / s
        # qw = (rot[7] - rot[5]) / s
    elif rot[4] > rot[8]:
        s = np.sqrt(1. - rot[0] + rot[4] - rot[8]) * 2
        qx = (rot[1] + rot[3]) / s
        qy = 0.25 * s
        qz = (rot[5] + rot[7]) / s
        # qw = (rot[2] - rot[6]) / s
    else:
        s = np.sqrt(1. - rot[0] - rot[4] + rot[8]) * 2.
        qx = (rot[2] + rot[6]) / s
        qy = (rot[5] + rot[7]) / s
        qz = 0.25 * s
        # qw = (rot[3] - rot[1]) / s
    return qx, qy, qz


def _rot_to_quat(rot):
    """Convert a set of rotations to quaternions"""
    rot = rot.reshape(rot.shape[:-2] + (9,))
    return np.apply_along_axis(_one_rot_to_quat, -1, rot)


# ############################################################################
# Estimate positions from data

def _get_hpi_info(info):
    """Helper to get HPI information from raw"""
    if len(info['hpi_meas']) == 0 or \
            ('coil_freq' not in info['hpi_meas'][0]['hpi_coils'][0]):
        raise RuntimeError('Appropriate cHPI information not found in'
                           'raw.info["hpi_meas"], cannot process cHPI')
    hpi_result = info['hpi_results'][-1]
    hpi_coils = sorted(info['hpi_meas'][-1]['hpi_coils'],
                       key=lambda x: x['number'])  # ascending (info) order
    hpi_dig = sorted([d for d in info['dig']
                      if d['kind'] == FIFF.FIFFV_POINT_HPI],
                     key=lambda x: x['ident'])  # ascending (dig) order
    pos_order = hpi_result['order'] - 1  # zero-based indexing, dig->info
    # hpi_result['dig_points'] are in FIFFV_COORD_UNKNOWN coords...?

    # this shouldn't happen, eventually we could add the transforms
    # necessary to put it in head coords
    if not all(d['coord_frame'] == FIFF.FIFFV_COORD_HEAD for d in hpi_dig):
        raise RuntimeError('cHPI coordinate frame incorrect')
    # Give the user some info
    logger.info('HPIFIT: %s coils digitized in order %s'
                % (len(pos_order), ' '.join(str(o + 1) for o in pos_order)))
    logger.debug('HPIFIT: %s coils accepted: %s'
                 % (len(hpi_result['used']),
                    ' '.join(str(h) for h in hpi_result['used'])))
    hpi_rrs = np.array([d['r'] for d in hpi_dig])[pos_order]
    # errors = 1000 * np.sqrt((hpi_rrs - hpi_rrs_fit) ** 2).sum(axis=1)
    # logger.debug('HPIFIT errors:  %s'
    #              % ', '.join('%0.1f' % e for e in errors))
    hpi_freqs = np.array([float(x['coil_freq']) for x in hpi_coils])
    # how cHPI active is indicated in the FIF file
    hpi_sub = info['hpi_subsystem']
    hpi_pick = pick_channels(info['ch_names'], [hpi_sub['event_channel']])[0]
    hpi_on = np.sum([coil['event_bits'][0] for coil in hpi_sub['hpi_coils']])
    logger.info('Using %s HPI coils: %s Hz'
                % (len(hpi_freqs), ' '.join(str(int(s)) for s in hpi_freqs)))
    return hpi_freqs, hpi_rrs, hpi_pick, hpi_on, pos_order


def _magnetic_dipole_objective(x, B, B2, coils, scale, method):
    """Project data onto right eigenvectors of whitened forward"""
    if method == 'forward':
        fwd = _magnetic_dipole_field_vec(x[np.newaxis, :], coils)
    else:
        from .preprocessing.maxwell import _sss_basis
        # Eventually we can try incorporating external bases here, which
        # is why the :3 is on the SVD below
        fwd = _sss_basis(x, coils, 1, 0).T
    fwd = np.dot(fwd, scale.T)
    one = np.dot(linalg.svd(fwd, full_matrices=False)[2][:3], B)
    one *= one
    Bm2 = one.sum()
    return B2 - Bm2


def _fit_magnetic_dipole(B_orig, x0, coils, scale, method):
    """Fit a single bit of data (x0 = pos)"""
    from scipy.optimize import fmin_cobyla
    B = np.dot(scale, B_orig)
    B2 = np.dot(B, B)
    objective = partial(_magnetic_dipole_objective, B=B, B2=B2,
                        coils=coils, scale=scale, method=method)
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-2, rhoend=1e-5, disp=False)
    return x, 1. - objective(x) / B2


def _chpi_objective(x, coil_dev_rrs, coil_head_rrs):
    """Helper objective function"""
    d = np.dot(coil_dev_rrs, _quat_to_rot(x[:3]).T)
    d += x[3:]
    d -= coil_head_rrs
    d *= d
    return d.sum()


def _unit_quat_constraint(x):
    """Constrain our 3 quaternion rot params (ignoring w) to have norm <= 1"""
    return 1 - (x * x).sum()


def _fit_chpi_pos(coil_dev_rrs, coil_head_rrs, x0):
    """Fit rotation and translation parameters for cHPI coils"""
    from scipy.optimize import fmin_cobyla
    denom = np.sum((coil_head_rrs - np.mean(coil_head_rrs, axis=0)) ** 2)
    objective = partial(_chpi_objective, coil_dev_rrs=coil_dev_rrs,
                        coil_head_rrs=coil_head_rrs)
    x = fmin_cobyla(objective, x0, _unit_quat_constraint,
                    rhobeg=1e-2, rhoend=1e-6, disp=False)
    return x, 1. - objective(x) / denom


def _angle_between_quats(x, y):
    """Compute the angle between two quaternions w/3-element representations"""
    # convert to complete quaternion representation
    # use max() here to be safe in case roundoff errs put us over
    x0 = np.sqrt(np.maximum(1. - x[..., 0] ** 2 -
                            x[..., 1] ** 2 - x[..., 2] ** 2, 0.))
    y0 = np.sqrt(np.maximum(1. - y[..., 0] ** 2 -
                            y[..., 1] ** 2 - y[..., 2] ** 2, 0.))
    # the difference z = x * conj(y), and theta = np.arccos(z0)
    z0 = np.maximum(np.minimum(y0 * x0 + (x * y).sum(axis=-1), 1.), -1)
    return 2 * np.arccos(z0)


def _setup_chpi_fits(info, t_window, t_step_min, method='forward'):
    """Helper to set up cHPI fits"""
    from scipy.spatial.distance import cdist
    from .preprocessing.maxwell import _prep_bases
    if not (check_version('numpy', '1.7') and check_version('scipy', '0.11')):
        raise RuntimeError('numpy>=1.7 and scipy>=0.11 required')
    hpi_freqs, coil_head_rrs, hpi_pick, hpi_on = _get_hpi_info(info)[:4]
    line_freqs = np.arange(info['line_freq'], info['sfreq'] / 3.,
                           info['line_freq'])
    logger.info('Line interference frequencies: %s Hz'
                % ' '.join(['%d' % l for l in line_freqs]))
    # initial transforms
    dev_head_t = info['dev_head_t']['trans']
    head_dev_t = invert_transform(info['dev_head_t'])['trans']
    # determine timing
    n_window = int(round(t_window * info['sfreq']))
    n_freqs = len(hpi_freqs)
    logger.debug('Coordinate transformation:')
    for d in (dev_head_t[0, :3], dev_head_t[1, :3], dev_head_t[2, :3],
              dev_head_t[:3, 3] * 1000.):
        logger.debug('{0:8.4f} {1:8.4f} {2:8.4f}'.format(*d))
    # Set up amplitude fits
    slope = np.arange(n_window).astype(np.float64)[:, np.newaxis]
    f_t = 2 * np.pi * hpi_freqs[np.newaxis, :] * (slope / info['sfreq'])
    l_t = 2 * np.pi * line_freqs[np.newaxis, :] * (slope / info['sfreq'])
    model = np.concatenate([np.sin(f_t), np.cos(f_t),  # hpi freqs
                            np.sin(l_t), np.cos(l_t),  # line freqs
                            slope,  # linear slope
                            np.ones((n_window, 1))  # constant
                            ], axis=1)
    inv_model = linalg.pinv(model)
    del slope, f_t, l_t

    # Set up magnetic dipole fits
    picks_good = pick_types(info, meg=True, eeg=False)
    picks = np.concatenate([picks_good, [hpi_pick]])
    megchs = [ch for ci, ch in enumerate(info['chs']) if ci in picks_good]
    coils = _create_meg_coils(megchs, 'normal')
    if method == 'forward':
        coils = _concatenate_coils(coils)
    else:  # == 'multipole'
        coils = _prep_bases(coils, 1, 0)
    scale = make_ad_hoc_cov(info, verbose=False)
    scale = _get_whitener_data(info, scale, picks_good, verbose=False)
    orig_dev_head_quat = np.concatenate([_rot_to_quat(dev_head_t[:3, :3]),
                                         dev_head_t[:3, 3]])
    dists = cdist(coil_head_rrs, coil_head_rrs)
    hpi = dict(dists=dists, scale=scale, picks=picks, model=model,
               inv_model=inv_model, coil_head_rrs=coil_head_rrs,
               coils=coils, on=hpi_on, n_window=n_window, method=method,
               n_freqs=n_freqs)
    last = dict(quat=orig_dev_head_quat, coil_head_rrs=coil_head_rrs,
                coil_dev_rrs=apply_trans(head_dev_t, coil_head_rrs),
                sin_fit=None, fit_time=-t_step_min)
    return hpi, last


def _time_prefix(fit_time):
    """Helper to format log messages"""
    return ('    t=%0.3f:' % fit_time).ljust(17)


@verbose
def calculate_chpi_positions(raw, t_step_min=0.1, t_step_max=10.,
                             t_window=0.2, dist_limit=0.005, gof_limit=0.98,
                             verbose=None):
    """Calculate head positions using cHPI coils

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    t_step_min : float
        Minimum time step to use. If correlations are sufficiently high,
        t_step_max will be used.
    t_step_max : float
        Maximum time step to use.
    t_window : float
        Time window to use to estimate the head positions.
    max_step : float
        Maximum time step to go between estimations.
    dist_limit : float
        Minimum distance (m) to accept for coil position fitting.
    gof_limit : float
        Minimum goodness of fit to accept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fits : ndarray, shape (N, 10)
        The ``[t, q1, q2, q3, x, y, z, gof, err, v]`` for each fit.

    Notes
    -----
    The number of time points ``N`` will depend on the velocity of head
    movements as well as ``t_step_max`` and ``t_step_min``.

    See Also
    --------
    get_chpi_positions
    """
    from scipy.spatial.distance import cdist
    hpi, last = _setup_chpi_fits(raw.info, t_window, t_step_min,
                                 method='forward')
    fit_starts = np.round(np.arange(0, raw.times[-1], t_step_min) *
                          raw.info['sfreq']).astype(int)
    fit_starts = fit_starts[fit_starts < raw.n_times - hpi['n_window']]
    fit_times = (fit_starts + (hpi['n_window'] + 1) // 2) / raw.info['sfreq']
    quats = []
    logger.info('Fitting up to %s time points (%0.1f sec duration)'
                % (len(fit_starts), raw.times[-1]))
    for start, fit_time in zip(fit_starts, fit_times):
        #
        # 1. Fit amplitudes for each channel from each of the N cHPI sinusoids
        #
        with use_log_level(False):
            meg_chpi_data = raw[hpi['picks'], start:start + hpi['n_window']][0]
        this_data = meg_chpi_data[:-1]
        chpi_data = meg_chpi_data[-1]
        if not (chpi_data == hpi['on']).all():
            logger.info(_time_prefix(fit_time) + 'HPI not turned on')
            continue
        X = np.dot(hpi['inv_model'], this_data.T)
        data_diff = np.dot(hpi['model'], X).T - this_data
        data_diff *= data_diff
        this_data *= this_data
        g_chan = (1 - np.sqrt(data_diff.sum(axis=1) / this_data.sum(axis=1)))
        g_sin = (1 - np.sqrt(data_diff.sum() / this_data.sum()))
        del data_diff, this_data
        X_sin, X_cos = X[:hpi['n_freqs']], X[hpi['n_freqs']:2 * hpi['n_freqs']]
        signs = np.sign(np.arctan2(X_sin, X_cos))
        X_sin *= X_sin
        X_cos *= X_cos
        X_sin += X_cos
        sin_fit = np.sqrt(X_sin)
        if last['sin_fit'] is not None:  # first iteration
            corr = np.corrcoef(sin_fit.ravel(), last['sin_fit'].ravel())[0, 1]
            # check to see if we need to continue
            if fit_time - last['fit_time'] <= t_step_max - 1e-7 and \
                    corr * corr > 0.98 and fit_time != fit_times[-1]:
                continue  # don't need to re-fit data
        last['sin_fit'] = sin_fit.copy()  # save *before* inplace sign mult
        sin_fit *= signs
        del signs, X_sin, X_cos, X

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        logger.debug('    HPI amplitude correlation %0.3f: %0.3f '
                     '(%s chnls > 0.950)' % (fit_time, np.sqrt(g_sin),
                                             (np.sqrt(g_chan) > 0.95).sum()))
        outs = [_fit_magnetic_dipole(f, pos, hpi['coils'], hpi['scale'],
                                     hpi['method'])
                for f, pos in zip(sin_fit, last['coil_dev_rrs'])]
        this_coil_dev_rrs = np.array([o[0] for o in outs])
        g_coils = [o[1] for o in outs]
        these_dists = cdist(this_coil_dev_rrs, this_coil_dev_rrs)
        these_dists = np.abs(hpi['dists'] - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        good = False
        use_mask = np.ones(hpi['n_freqs'], bool)
        while not good:
            d = these_dists[use_mask][:, use_mask]
            d_bad = (d > dist_limit)
            good = not d_bad.any()
            if not good:
                if use_mask.sum() == 2:
                    use_mask[:] = False
                    break  # failure
                # exclude next worst point
                badness = (d * d_bad).sum(axis=0)
                exclude = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude] = False
        good = use_mask.sum() >= 3
        if not good:
            logger.warning(_time_prefix(fit_time) + '%s/%s good HPI fits, '
                           'cannot determine the transformation!'
                           % (use_mask.sum(), hpi['n_freqs']))
            continue

        #
        # 3. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization positions)
        #
        this_quat, g = _fit_chpi_pos(this_coil_dev_rrs[use_mask],
                                     hpi['coil_head_rrs'][use_mask],
                                     last['quat'])
        if g < gof_limit:
            logger.info(_time_prefix(fit_time) +
                        'Bad coil fit! (g=%7.3f)' % (g,))
            continue
        this_dev_head_t = np.concatenate(
            (_quat_to_rot(this_quat[:3]),
             this_quat[3:][:, np.newaxis]), axis=1)
        this_dev_head_t = np.concatenate((this_dev_head_t, [[0, 0, 0, 1.]]))
        # velocities, in device coords, of HPI coils
        dt = fit_time - last['fit_time']
        vs = tuple(1000. * np.sqrt(np.sum((last['coil_dev_rrs'] -
                                           this_coil_dev_rrs) ** 2,
                                          axis=1)) / dt)
        logger.info(_time_prefix(fit_time) +
                    ('%s/%s good HPI fits, movements [mm/s] = ' +
                     ' / '.join(['%0.1f'] * hpi['n_freqs']))
                    % ((use_mask.sum(), hpi['n_freqs']) + vs))
        # resulting errors in head coil positions
        est_coil_head_rrs = apply_trans(this_dev_head_t, this_coil_dev_rrs)
        errs = 1000. * np.sqrt(np.sum((hpi['coil_head_rrs'] -
                                       est_coil_head_rrs) ** 2,
                                      axis=1))
        e = 0.  # XXX eventually calculate this
        d = 100 * np.sqrt(np.sum(last['quat'][3:] - this_quat[3:]) ** 2)  # cm
        r = _angle_between_quats(last['quat'][:3], this_quat[:3]) / dt
        v = d / dt  # cm/sec
        for ii in range(hpi['n_freqs']):
            if use_mask[ii]:
                start, end = ' ', '/'
            else:
                start, end = '(', ')'
            log_str = (start +
                       '{0:6.1f} {1:6.1f} {2:6.1f} / ' +
                       '{3:6.1f} {4:6.1f} {5:6.1f} / ' +
                       'g = {6:0.3f} err = {7:4.1f} ' +
                       end)
            if ii <= 2:
                log_str += '{8:6.3f} {9:6.3f} {10:6.3f}'
            elif ii == 3:
                log_str += '{8:6.1f} {9:6.1f} {10:6.1f}'
            vals = np.concatenate((1000 * hpi['coil_head_rrs'][ii],
                                   1000 * est_coil_head_rrs[ii],
                                   [g_coils[ii], errs[ii]]))
            if ii <= 2:
                vals = np.concatenate((vals, this_dev_head_t[ii, :3]))
            elif ii == 3:
                vals = np.concatenate((vals, this_dev_head_t[:3, 3] * 1000.))
            logger.debug(log_str.format(*vals))
        logger.debug('    #t = %0.3f, #e = %0.2f cm, #g = %0.3f, '
                     '#v = %0.2f cm/s, #r = %0.2f rad/s, #d = %0.2f cm'
                     % (fit_time, 100 * e, g, v, r, d))
        quats.append(np.concatenate(([fit_time], this_quat, [g], [e], [v])))
        last['fit_time'] = fit_time
        last['quat'] = this_quat
        last['coil_dev_rrs'] = this_coil_dev_rrs
    logger.info('[done]')
    quats = np.array(quats)
    quats = np.zeros((0, 10)) if quats.size == 0 else quats
    return quats
