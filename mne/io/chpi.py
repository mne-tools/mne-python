# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Samu Taulu <staulu@uw.edu>
# License: BSD (3-clause)

import numpy as np
from scipy import linalg, optimize
from scipy.spatial.distance import cdist
from os import path as op

from .pick import pick_types
from .constants import FIFF
from .base import _BaseRaw
from ..cov import make_ad_hoc_cov, _get_whitener_data
from ..utils import verbose, logger
from ..transforms import apply_trans, invert_transform
from ..externals.six import string_types


def _fit_mag_dipole(B_orig, w, coils, x0):
    """Fit a single bit of data (x0 = pos)"""
    from ..forward._compute_forward import _mag_dipole_field_vec
    B = np.dot(w, B_orig)
    B2 = np.dot(B, B)

    def fun(x0):
        """Project data onto right eigenvectors of whitened forward"""
        fwd = np.dot(_mag_dipole_field_vec(x0[np.newaxis, :], coils), w.T)
        one = np.dot(linalg.svd(fwd, full_matrices=False)[2], B)
        Bm2 = np.sum(one * one)
        return 1 - Bm2 / B2

    x = optimize.fmin_cobyla(fun, x0, (), rhobeg=1e-3, rhoend=1e-5, disp=False)
    return x


def _fit_chpi_pos(chpi_pos_head, chpi_pos_dev, x0):
    """Fit rotation and translation parameters for cHPI coils"""

    def fun(x):
        trans = x[:3, np.newaxis]
        diff = (np.dot(_quat_to_rot(x[:3]), chpi_pos_head.T) + trans -
                chpi_pos_dev.T)
        return np.sum(diff * diff)

    x = optimize.fmin_cobyla(fun, x0, (), rhobeg=1e-3, rhoend=1e-8, disp=False)
    return x, np.sqrt(fun(x))


@verbose
def calculate_chpi_positions(raw, t_step=0.1, t_window=0.2, verbose=None):
    """Calculate head positions using cHPI coils

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    t_step : float
        Time increments to calculate head positions for, if necessary.
    t_window : float
        Time window to use to estimate the head positions.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    quat : ndarray
        An ``N`` x 10 array with time, 3 quaternions, 3 position values,
        g-value, error, and velocity. The number of time points ``N`` will
        depend on the velocity of head movements as well as `t_step`.
    """
    from ..forward._make_forward import _prep_channels
    n_window = int(round(t_window * raw.info['sfreq']))
    fit_starts = np.round(np.arange(0, raw.last_samp / raw.info['sfreq'],
                                    t_step) * raw.info['sfreq']).astype(int)
    fit_starts = fit_starts[fit_starts < raw.n_times - n_window]
    fit_times = (fit_starts + (n_window + 1) // 2) / raw.info['sfreq']
    dig_order = raw.info['hpi_results'][0]['order'].astype(int)
    freqs = np.array([c['custom_ref'][0]
                      for c in raw.info['hpi_meas'][0]['hpi_coils']], float)
    freq_order = [x['number'] for x in raw.info['hpi_meas'][0]['hpi_coils']]
    dig_reorder = np.searchsorted(freq_order, dig_order)
    freqs = freqs[:, np.newaxis]
    n_freqs = len(freqs)
    logger.info('cHPI Frequencies: %s' % freqs[:, 0])
    t = (np.arange(n_window, dtype=float) / raw.info['sfreq'])[np.newaxis, :]

    # Set up amplitude fits
    model = np.concatenate([np.sin(2 * np.pi * freqs * t),
                            np.cos(2 * np.pi * freqs * t),
                            np.arange(n_window, dtype=float)[np.newaxis, :] /
                            n_window,
                            np.ones(n_window)[np.newaxis, :]], axis=0).T
    inv_model = linalg.pinv(model)

    # Set up magnetic dipole fits
    picks = pick_types(raw.info, meg=True, eeg=False)
    megcoils, _, _, _, _, meg_info = \
        _prep_channels(raw.info, exclude='bads', accurate=False, verbose=False)
    cov = make_ad_hoc_cov(raw.info, verbose=False)
    whitener = _get_whitener_data(raw.info, cov, picks, verbose=False)
    chpi_pos = np.array([d['r'] for d in raw.info['dig']
                         if d['kind'] == FIFF.FIFFV_POINT_HPI],
                        float)[dig_reorder]
    chpi_pos_dev = apply_trans(invert_transform(
        raw.info['dev_head_t'])['trans'], chpi_pos)
    trans = raw.info['hpi_results'][0]['coord_trans']['trans']
    quat_trans = np.concatenate([_rot_to_quat(trans[:3, :3]), trans[:3, 3]])
    orig_dists = cdist(chpi_pos, chpi_pos)
    dist_limit = 0.005
    last_fit = None  # this indicates it's the first run
    last_time = 0
    quats = []
    for fi, (start, t) in enumerate(zip(fit_starts, fit_times)):
        #
        # 1. Fit amplitudes for each channel from each of the N cHPI sinusoids
        #
        this_data = raw[picks, start:start + n_window][0]
        X = np.dot(inv_model, this_data.T)
        X_chpi = X[:n_freqs * 2]
        s_fit = np.sqrt(X_chpi[:n_freqs] ** 2 + X_chpi[n_freqs:] ** 2)
        if last_fit is not None:
            corr = np.corrcoef(s_fit.ravel(), last_fit.ravel())[0, 1]

        # check to see if we need to continue
        if t - last_time >= 1. - 1e-5 or last_fit is None:
            reason = ''
        elif corr < 0.99:
            reason = 'low correlation = %0.4f' % corr
        else:
            continue  # don't need to re-fit data
        last_fit = s_fit.copy()  # save *before* inplace sign transform
        logger.info('Fitting t = %0.3f sec (%s)...' % (t, reason))
        # figure out principal direction of the vectors and align
        for x, y, fit in zip(X_chpi[:n_freqs], X_chpi[n_freqs:], s_fit):
            fit *= np.sign(linalg.svd([x, y], full_matrices=False)[2][0])

        data_model = np.dot(model, X).T
        g = (1 - np.sqrt(np.sum((data_model - this_data) ** 2)) /
             np.sqrt(np.sum(this_data ** 2)))
        logger.info('    Initial sinusoidal GOF: %s%%'
                    % round(100 * g, 1))

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #
        chpi_pos = np.array([_fit_mag_dipole(f, whitener, megcoils, pos)
                             for f, pos in zip(s_fit, chpi_pos)])
        these_dists = cdist(chpi_pos, chpi_pos)
        these_differences = np.abs(orig_dists - these_dists)
        if np.any(these_differences > dist_limit):
            raise RuntimeError('Bad fit')

        #
        # 3. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization positions)
        #
        last_pos = quat_trans[3:]
        quat_trans, err = _fit_chpi_pos(chpi_pos, chpi_pos_dev, quat_trans)
        velocity = (np.sqrt(np.sum(last_pos - quat_trans[3:]) ** 2) /
                    (t - last_time))
        quats.append(np.concatenate(([t], quat_trans, [g], [err], [velocity])))
        last_time = fit_times[fi]
    return np.array(quats)


@verbose
def get_chpi_positions(raw, t_step=None, verbose=None):
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

    Returns
    -------
    translation : array
        A 2-dimensional array of head position vectors (n_time x 3).
    rotation : array
        A 3-dimensional array of rotation matrices (n_time x 3 x 3).
    t : array
        The time points associated with each position (n_time).

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
        if not np.isscalar(t_step):
            raise TypeError('t_step must be a scalar or None')
        picks = pick_types(raw.info, meg=False, ref_meg=False,
                           chpi=True, exclude=[])
        if len(picks) == 0:
            raise RuntimeError('raw file has no CHPI channels')
        time_idx = raw.time_as_index(np.arange(0, raw.n_times /
                                               raw.info['sfreq'], t_step))
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
    t = data[:, 0]
    translation = data[:, 4:7].copy()
    rotation = _quat_to_rot(data[:, 1:4])
    return translation, rotation, t


def _quat_to_rot(q):
    """Helper to convert quaternions to rotations"""
    # z = a + bi + cj + dk
    a = np.sqrt(1 - np.sum(q[..., 0:3] ** 2, axis=-1))
    b, c, d = q[..., 0], q[..., 1], q[..., 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
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
    rotation = np.rollaxis(np.rollaxis(rotation, 1, q.ndim + 1), 0, q.ndim)
    return rotation


def _rot_to_quat(rot):
    """Here we derive qw from qx, qy, qz"""
    qw_4 = np.sqrt(1 + rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]) * 2
    qx = (rot[..., 2, 1] - rot[..., 1, 2]) / qw_4
    qy = (rot[..., 0, 2] - rot[..., 2, 0]) / qw_4
    qz = (rot[..., 1, 0] - rot[..., 0, 1]) / qw_4
    return np.rollaxis(np.array((qx, qy, qz)), 0, rot.ndim - 1)
