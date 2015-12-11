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
from .utils import verbose, logger, check_version
from .fixes import partial
from .externals.six import string_types


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
        path to a Maxfilter log file (str).
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
    _calculate_chpi_positions, get_chpi_positions
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
    hpi_coils = info['hpi_meas'][-1]['hpi_coils']
    hpi_num = np.array([h['number'] for h in hpi_coils])
    pos_order = np.searchsorted(hpi_num, hpi_result['order'])
    hpi_dig = [d for d in info['dig'] if d['kind'] == FIFF.FIFFV_POINT_HPI]
    # this shouldn't happen, eventually we could add the transforms
    # necessary to put it in head coords
    if not all(d['coord_frame'] == FIFF.FIFFV_COORD_HEAD for d in hpi_dig):
        raise RuntimeError('cHPI coordinate frame incorrect')
    hpi_rrs = np.array([d['r'] for d in hpi_dig])[pos_order]
    hpi_freqs = np.array([float(x['coil_freq']) for x in hpi_coils])
    # how cHPI active is indicated in the FIF file
    hpi_sub = info['hpi_subsystem']
    hpi_pick = pick_channels(info['ch_names'], [hpi_sub['event_channel']])[0]
    hpi_on = np.sum([coil['event_bits'][0] for coil in hpi_sub['hpi_coils']])
    return hpi_freqs, hpi_rrs, hpi_pick, hpi_on, pos_order


def _magnetic_dipole_objective(x, B, B2, w, coils):
    """Project data onto right eigenvectors of whitened forward"""
    fwd = np.dot(_magnetic_dipole_field_vec(x[np.newaxis, :], coils), w.T)
    one = np.dot(linalg.svd(fwd, full_matrices=False)[2], B)
    Bm2 = np.sum(one * one)
    return B2 - Bm2


def _fit_magnetic_dipole(B_orig, w, coils, x0):
    """Fit a single bit of data (x0 = pos)"""
    from scipy.optimize import fmin_cobyla
    B = np.dot(w, B_orig)
    B2 = np.dot(B, B)
    objective = partial(_magnetic_dipole_objective, B=B, B2=B2,
                        w=w, coils=coils)
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-2, rhoend=1e-4, disp=False)
    return x, 1. - objective(x) / B2


def _chpi_objective(x, est_pos_dev, hpi_head_rrs):
    """Helper objective function"""
    rot = _quat_to_rot(x[:3]).T
    d = np.dot(est_pos_dev, rot) + x[3:] - hpi_head_rrs
    return np.sum(d * d)


def _fit_chpi_pos(est_pos_dev, hpi_head_rrs, x0):
    """Fit rotation and translation parameters for cHPI coils"""
    from scipy.optimize import fmin_cobyla
    denom = np.sum((hpi_head_rrs - np.mean(hpi_head_rrs, axis=0)) ** 2)
    objective = partial(_chpi_objective, est_pos_dev=est_pos_dev,
                        hpi_head_rrs=hpi_head_rrs)
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-2, rhoend=1e-6, disp=False)
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


@verbose
def _calculate_chpi_positions(raw, t_step_min=0.1, t_step_max=10.,
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
    translation : ndarray, shape (N, 3)
        Translations at each time point.
    rotation : ndarray, shape (N, 3, 3)
        Rotations at each time point.
    t : ndarray, shape (N,)
        The time points.

    Notes
    -----
    The number of time points ``N`` will depend on the velocity of head
    movements as well as ``t_step_max`` and ``t_step_min``.

    See Also
    --------
    get_chpi_positions
    """
    from scipy.spatial.distance import cdist
    if not (check_version('numpy', '1.7') and check_version('scipy', '0.11')):
        raise RuntimeError('numpy>=1.7 and scipy>=0.11 required')
    hpi_freqs, orig_head_rrs, hpi_pick, hpi_on, order = _get_hpi_info(raw.info)
    sfreq, ch_names = raw.info['sfreq'], raw.info['ch_names']
    # initial transforms
    dev_head_t = raw.info['dev_head_t']['trans']
    head_dev_t = invert_transform(raw.info['dev_head_t'])['trans']
    # determine timing
    n_window = int(round(t_window * sfreq))
    fit_starts = np.round(np.arange(0, raw.last_samp / sfreq, t_step_min) *
                          sfreq).astype(int)
    fit_starts = fit_starts[fit_starts < raw.n_times - n_window]
    fit_times = (fit_starts + (n_window + 1) // 2) / sfreq
    n_freqs = len(hpi_freqs)
    logger.info('HPIFIT: %s coils digitized in order %s'
                % (n_freqs, ' '.join(str(o + 1) for o in order)))
    logger.info('Coordinate transformation:')
    for d in (dev_head_t[0, :3], dev_head_t[1, :3], dev_head_t[2, :3],
              dev_head_t[:3, 3] * 1000.):
        logger.info('{0:8.4f} {1:8.4f} {2:8.4f}'.format(*d))
    logger.info('Using %s HPI coils: %s Hz'
                % (n_freqs, ' '.join(str(int(s)) for s in hpi_freqs)))
    # Set up amplitude fits
    slope = np.arange(n_window).astype(np.float64)[:, np.newaxis]
    f_t = 2 * np.pi * hpi_freqs[np.newaxis, :] * (slope / sfreq)
    model = np.concatenate([np.sin(f_t), np.cos(f_t),
                            slope, np.ones((n_window, 1))], axis=1)
    inv_model = linalg.pinv(model)
    del slope, f_t

    # Set up magnetic dipole fits
    picks = pick_types(raw.info, meg=True, eeg=False)
    picks_chpi = np.concatenate([picks, [hpi_pick]])
    logger.info('Found %s total and %s good MEG channels'
                % (len(ch_names), len(picks)))
    megchs = [ch for ci, ch in enumerate(raw.info['chs']) if ci in picks]
    coils = _concatenate_coils(_create_meg_coils(megchs, 'normal'))

    cov = make_ad_hoc_cov(raw.info, verbose=False)
    whitener = _get_whitener_data(raw.info, cov, picks, verbose=False)
    dev_head_quat = np.concatenate([_rot_to_quat(dev_head_t[:3, :3]),
                                    dev_head_t[:3, 3]])
    orig_dists = cdist(orig_head_rrs, orig_head_rrs)
    last_quat = dev_head_quat.copy()
    last_data_fit = None  # this indicates it's the first run
    last_time = -t_step_min
    last_head_rrs = orig_head_rrs.copy()
    corr_limit = 0.98
    quats = []
    est_pos_dev = apply_trans(head_dev_t, orig_head_rrs)
    for start, t in zip(fit_starts, fit_times):
        #
        # 1. Fit amplitudes for each channel from each of the N cHPI sinusoids
        #
        meg_chpi_data = raw[picks_chpi, start:start + n_window][0]
        this_data = meg_chpi_data[:-1]
        chpi_data = meg_chpi_data[-1]
        if not (chpi_data == hpi_on).all():
            logger.info('HPI not turned on (t=%7.3f)' % t)
            continue
        X = np.dot(inv_model, this_data.T)
        data_diff = np.dot(model, X).T - this_data
        data_diff *= data_diff
        this_data *= this_data
        g_chan = (1 - np.sqrt(data_diff.sum(axis=1) / this_data.sum(axis=1)))
        g_sin = (1 - np.sqrt(data_diff.sum() / this_data.sum()))
        del data_diff, this_data
        X_sin, X_cos = X[:n_freqs], X[n_freqs:2 * n_freqs]
        s_fit = np.sqrt(X_cos * X_cos + X_sin * X_sin)
        if last_data_fit is None:  # first iteration
            corr = 0.
        else:
            corr = np.corrcoef(s_fit.ravel(), last_data_fit.ravel())[0, 1]

        # check to see if we need to continue
        if t - last_time <= t_step_max - 1e-7 and corr > corr_limit and \
                t != fit_times[-1]:
            continue  # don't need to re-fit data
        last_data_fit = s_fit.copy()  # save *before* inplace sign transform

        # figure out principal direction of the vectors and align
        # for s, c, fit in zip(X_sin, X_cos, s_fit):
        #     fit *= np.sign(linalg.svd([s, c], full_matrices=False)[2][0])
        s_fit *= np.sign(np.arctan2(X_sin, X_cos))

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        logger.info('HPI amplitude correlation %s: %s (%s chnls > 0.95)'
                    % (t, g_sin, (g_chan > 0.95).sum()))
        outs = [_fit_magnetic_dipole(f, whitener, coils, pos)
                for f, pos in zip(s_fit, est_pos_dev)]
        est_pos_dev = np.array([o[0] for o in outs])
        g_coils = [o[1] for o in outs]
        these_dists = cdist(est_pos_dev, est_pos_dev)
        these_dists = np.abs(orig_dists - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        good = False
        use_mask = np.ones(n_freqs, bool)
        while not good:
            d = (these_dists[use_mask][:, use_mask] <= dist_limit)
            good = d.all()
            if not good:
                if use_mask.sum() == 2:
                    use_mask[:] = False
                    break  # failure
                # exclude next worst point
                badness = these_dists[use_mask][:, use_mask].sum(axis=0)
                exclude = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude] = False
        good = use_mask.sum() >= 3
        if not good:
            logger.warning('    %s/%s acceptable hpi fits found, cannot '
                           'determine the transformation! (t=%7.3f)'
                           % (use_mask.sum(), n_freqs, t))
            continue

        #
        # 3. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization positions)
        #
        dev_head_quat, g = _fit_chpi_pos(est_pos_dev[use_mask],
                                         orig_head_rrs[use_mask],
                                         dev_head_quat)
        if g < gof_limit:
            logger.info('    Bad coil fit for %s! (t=%7.3f)' % t)
            continue
        this_dev_head_t = np.concatenate((_quat_to_rot(dev_head_quat[:3]),
                                          dev_head_quat[3:][:, np.newaxis]),
                                         axis=1)
        this_dev_head_t = np.concatenate((this_dev_head_t, [[0, 0, 0, 1.]]))
        this_head_rrs = apply_trans(this_dev_head_t, est_pos_dev)
        dt = t - last_time
        vs = tuple(1000. * np.sqrt(np.sum((last_head_rrs -
                                           this_head_rrs) ** 2, axis=1)) / dt)
        logger.info('Hpi fit OK, movements [mm/s] = ' +
                    ' / '.join(['%0.1f'] * n_freqs) % vs)
        errs = [0] * n_freqs  # XXX eventually calculate this
        e = 0.  # XXX eventually calculate this
        d = 100 * np.sqrt(np.sum(last_quat[3:] - dev_head_quat[3:]) ** 2)  # cm
        r = _angle_between_quats(last_quat[:3], dev_head_quat[:3]) / dt
        v = d / dt  # cm/sec
        for ii in range(n_freqs):
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
            vals = np.concatenate((1000 * orig_head_rrs[ii],
                                   1000 * this_head_rrs[ii],
                                   [g_coils[ii], errs[ii]]))
            if ii <= 2:
                vals = np.concatenate((vals, this_dev_head_t[ii, :3]))
            elif ii == 3:
                vals = np.concatenate((vals, this_dev_head_t[:3, 3] * 1000.))
            logger.debug(log_str.format(*vals))
        logger.info('#t = %0.3f, #e = %0.2f cm, #g = %0.3f, #v = %0.2f cm/s, '
                    '#r = %0.2f rad/s, #d = %0.2f cm' % (t, e, g, v, r, d))
        quats.append(np.concatenate(([t], dev_head_quat, [g], [1. - g], [v])))
        last_time = t
        last_head_rrs = this_head_rrs.copy()
    quats = np.array(quats)
    quats = np.zeros((0, 10)) if quats.size == 0 else quats
    return _quats_to_trans_rot_t(quats)
