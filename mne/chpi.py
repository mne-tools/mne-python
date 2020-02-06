# -*- coding: utf-8 -*-
"""Functions for fitting head positions with (c)HPI coils."""

# To fit head positions (continuously), the procedure using
# ``compute_chpi_locs`` is:
#
#     1. Get HPI coil locations (as digitized in info['dig'] in head coords
#        using ``_get_hpi_initial_fit``.
#     2. Get HPI frequencies,  HPI status channel, HPI status bits,
#        and digitization order using ``_setup_hpi_fitting``.
#     3. Window data using ``t_window`` (half before and half after ``t``) and
#        ``t_step_min``.
#        (Here Elekta high-passes the data, but we omit this step.)
#     4. Use a linear model (DC + linear slope + sin + cos terms set up
#        in ``_setup_hpi_fitting``) to fit sinusoidal amplitudes to MEG
#        channels. Use SVD to determine the phase/amplitude of the sinusoids.
#        This step is accomplished using ``_fit_cHPI_amplitudes``
#     5. If the amplitudes are 98% correlated with last position
#        (and Δt < t_step_max), skip fitting.
#     6. Fit magnetic dipoles using the amplitudes for each coil frequency
#        (calling ``_fit_magnetic_dipole``).
#
# This gives time-varying cHPI coil locations. These can then be used with
# ``compute_head_pos`` to:
#
#     1. Drop coils whose GOF are below ``gof_limit``. If fewer than 3 coils
#        remain, abandon fitting for the chunk.
#     2. Fit dev_head_t quaternion (using ``_fit_chpi_quat_subset``),
#        iteratively dropping coils (as long as 3 remain) to find the best GOF
#        (using ``_fit_chpi_quat``).
#     3. If fewer than 3 coils meet the ``dist_limit`` criteria following
#        projection of the fitted device coil locations into the head frame,
#        abandon fitting for the chunk.
#
# The function ``filter_chpi`` uses the same linear model to filter cHPI
# and (optionally) line frequencies from the data.

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy import linalg
import itertools

from .io.pick import (pick_types, pick_channels, pick_channels_regexp,
                      pick_info)
from .io.proj import Projection, setup_proj
from .io.constants import FIFF
from .io.ctf.trans import _make_ctf_coord_trans_set
from .forward import (_magnetic_dipole_field_vec, _create_meg_coils,
                      _concatenate_coils)
from .cov import make_ad_hoc_cov, compute_whitener
from .dipole import _make_guesses
from .fixes import jit
from .preprocessing.maxwell import (_sss_basis, _prep_mf_coils, _get_mf_picks,
                                    _regularize_out)
from .transforms import (apply_trans, invert_transform, _angle_between_quats,
                         quat_to_rot, rot_to_quat, _fit_matched_points,
                         _quat_to_affine)
from .utils import (verbose, logger, use_log_level, _check_fname, warn,
                    _check_option, _validate_type, ProgressBar)

# Eventually we should add:
#   hpicons
#   high-passing of data during fits
#   parsing cHPI coil information from acq pars, then to PSD if necessary


# ############################################################################
# Reading from text or FIF file

def read_head_pos(fname):
    """Read MaxFilter-formatted head position parameters.

    Parameters
    ----------
    fname : str
        The filename to read. This can be produced by e.g.,
        ``maxfilter -headpos <name>.pos``.

    Returns
    -------
    pos : array, shape (N, 10)
        The position and quaternion parameters from cHPI fitting.

    See Also
    --------
    write_head_pos
    head_pos_to_trans_rot_t

    Notes
    -----
    .. versionadded:: 0.12
    """
    _check_fname(fname, must_exist=True, overwrite='read')
    data = np.loadtxt(fname, skiprows=1)  # first line is header, skip it
    data.shape = (-1, 10)  # ensure it's the right size even if empty
    if np.isnan(data).any():  # make sure we didn't do something dumb
        raise RuntimeError('positions could not be read properly from %s'
                           % fname)
    return data


def write_head_pos(fname, pos):
    """Write MaxFilter-formatted head position parameters.

    Parameters
    ----------
    fname : str
        The filename to write.
    pos : array, shape (N, 10)
        The position and quaternion parameters from cHPI fitting.

    See Also
    --------
    read_head_pos
    head_pos_to_trans_rot_t

    Notes
    -----
    .. versionadded:: 0.12
    """
    _check_fname(fname, overwrite=True)
    pos = np.array(pos, np.float64)
    if pos.ndim != 2 or pos.shape[1] != 10:
        raise ValueError('pos must be a 2D array of shape (N, 10)')
    with open(fname, 'wb') as fid:
        fid.write(' Time       q1       q2       q3       q4       q5       '
                  'q6       g-value  error    velocity\n'.encode('ASCII'))
        for p in pos:
            fmts = ['% 9.3f'] + ['% 8.5f'] * 9
            fid.write(((' ' + ' '.join(fmts) + '\n')
                       % tuple(p)).encode('ASCII'))


def head_pos_to_trans_rot_t(quats):
    """Convert Maxfilter-formatted head position quaternions.

    Parameters
    ----------
    quats : ndarray, shape (N, 10)
        MaxFilter-formatted position and quaternion parameters.

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
    read_head_pos
    write_head_pos
    """
    t = quats[..., 0].copy()
    rotation = quat_to_rot(quats[..., 1:4])
    translation = quats[..., 4:7].copy()
    return translation, rotation, t


def extract_chpi_locs_ctf(raw):
    r"""Extract head position parameters from ctf dataset.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with CTF cHPI information.

    Returns
    -------
    %(chpi_locs)s

    Notes
    -----
    CTF continuous head monitoring stores the x,y,z location (m) of each chpi
    coil as separate channels in the dataset:

    - ``HLC001[123]\\*`` - nasion
    - ``HLC002[123]\\*`` - lpa
    - ``HLC003[123]\\*`` - rpa

    This extracts these positions for use with
    :func:`~mne.chpi.compute_head_pos`.

    .. versionadded:: 0.20
    """
    # Pick channels corresponding to the cHPI positions
    hpi_picks = pick_channels_regexp(raw.info['ch_names'], 'HLC00[123][123].*')

    # make sure we get 9 channels
    if len(hpi_picks) != 9:
        raise RuntimeError('Could not find all 9 cHPI channels')

    # get indices in alphabetical order
    sorted_picks = np.array(sorted(hpi_picks,
                                   key=lambda k: raw.info['ch_names'][k]))

    # make picks to match order of dig cardinial ident codes.
    # LPA (HPIC002[123]-*), NAS(HPIC001[123]-*), RPA(HPIC003[123]-*)
    hpi_picks = sorted_picks[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
    del sorted_picks

    # process the entire run
    time_sl = slice(0, len(raw.times))
    chpi_data = raw[hpi_picks, time_sl][0]

    # transforms
    tmp_trans = _make_ctf_coord_trans_set(None, None)
    ctf_dev_dev_t = tmp_trans['t_ctf_dev_dev']
    del tmp_trans

    # find indices where chpi locations change
    indices = [0]
    indices.extend(np.where(np.all(np.diff(chpi_data, axis=1), axis=0))[0] + 1)
    # data in channels are in ctf device coordinates (cm)
    rrs = chpi_data[:, indices].T.reshape(len(indices), 3, 3)  # m
    # map to mne device coords
    rrs = apply_trans(ctf_dev_dev_t, rrs)
    gofs = np.ones(rrs.shape[:2])  # not encoded, assume all good
    times = raw.times[indices] + raw._first_time
    return dict(rrs=rrs, gofs=gofs, times=times)


# ############################################################################
# Estimate positions from data
@verbose
def _get_hpi_info(info, verbose=None):
    """Get HPI information from raw."""
    if len(info['hpi_meas']) == 0 or \
            ('coil_freq' not in info['hpi_meas'][0]['hpi_coils'][0]):
        raise RuntimeError('Appropriate cHPI information not found in'
                           'info["hpi_meas"] and info["hpi_subsystem"], '
                           'cannot process cHPI')
    hpi_coils = sorted(info['hpi_meas'][-1]['hpi_coils'],
                       key=lambda x: x['number'])  # ascending (info) order

    # get frequencies
    hpi_freqs = np.array([float(x['coil_freq']) for x in hpi_coils])
    logger.info('Using %s HPI coils: %s Hz'
                % (len(hpi_freqs), ' '.join(str(int(s)) for s in hpi_freqs)))

    # how cHPI active is indicated in the FIF file
    hpi_sub = info['hpi_subsystem']
    hpi_pick = None  # there is no pick!
    if hpi_sub is not None:
        if 'event_channel' in hpi_sub:
            hpi_pick = pick_channels(info['ch_names'],
                                     [hpi_sub['event_channel']])
            hpi_pick = hpi_pick[0] if len(hpi_pick) > 0 else None
        # grab codes indicating a coil is active
        hpi_on = [coil['event_bits'][0] for coil in hpi_sub['hpi_coils']]
        # not all HPI coils will actually be used
        hpi_on = np.array([hpi_on[hc['number'] - 1] for hc in hpi_coils])
        # mask for coils that may be active
        hpi_mask = np.array([event_bit != 0 for event_bit in hpi_on])
        hpi_on = hpi_on[hpi_mask]
        hpi_freqs = hpi_freqs[hpi_mask]
    else:
        hpi_on = np.zeros(len(hpi_freqs))

    return hpi_freqs, hpi_pick, hpi_on


@verbose
def _get_hpi_initial_fit(info, adjust=False, verbose=None):
    """Get HPI fit locations from raw."""
    if info['hpi_results'] is None:
        raise RuntimeError('no initial cHPI head localization performed')

    hpi_result = info['hpi_results'][-1]
    hpi_dig = sorted([d for d in info['dig']
                      if d['kind'] == FIFF.FIFFV_POINT_HPI],
                     key=lambda x: x['ident'])  # ascending (dig) order
    if len(hpi_dig) == 0:  # CTF data, probably
        hpi_dig = sorted(hpi_result['dig_points'], key=lambda x: x['ident'])
        if all(d['coord_frame'] in (FIFF.FIFFV_COORD_DEVICE,
                                    FIFF.FIFFV_COORD_UNKNOWN)
               for d in hpi_dig):
            for dig in hpi_dig:
                dig.update(r=apply_trans(info['dev_head_t'], dig['r']),
                           coord_frame=FIFF.FIFFV_COORD_HEAD)

    # zero-based indexing, dig->info
    # CTF does not populate some entries so we use .get here
    pos_order = hpi_result.get('order', np.arange(1, len(hpi_dig) + 1)) - 1
    used = hpi_result.get('used', np.arange(len(hpi_dig)))
    dist_limit = hpi_result.get('dist_limit', 0.005)
    good_limit = hpi_result.get('good_limit', 0.98)
    goodness = hpi_result.get('goodness', np.ones(len(hpi_dig)))

    # this shouldn't happen, eventually we could add the transforms
    # necessary to put it in head coords
    if not all(d['coord_frame'] == FIFF.FIFFV_COORD_HEAD for d in hpi_dig):
        raise RuntimeError('cHPI coordinate frame incorrect')
    # Give the user some info
    logger.info('HPIFIT: %s coils digitized in order %s'
                % (len(pos_order), ' '.join(str(o + 1) for o in pos_order)))
    logger.debug('HPIFIT: %s coils accepted: %s'
                 % (len(used), ' '.join(str(h) for h in used)))
    hpi_rrs = np.array([d['r'] for d in hpi_dig])[pos_order]
    assert len(hpi_rrs) >= 3

    # Fitting errors
    hpi_rrs_fit = sorted([d for d in info['hpi_results'][-1]['dig_points']],
                         key=lambda x: x['ident'])
    hpi_rrs_fit = np.array([d['r'] for d in hpi_rrs_fit])
    # hpi_result['dig_points'] are in FIFFV_COORD_UNKNOWN coords, but this
    # is probably a misnomer because it should be FIFFV_COORD_DEVICE for this
    # to work
    assert hpi_result['coord_trans']['to'] == FIFF.FIFFV_COORD_HEAD
    hpi_rrs_fit = apply_trans(hpi_result['coord_trans']['trans'], hpi_rrs_fit)
    if 'moments' in hpi_result:
        logger.debug('Hpi coil moments (%d %d):'
                     % hpi_result['moments'].shape[::-1])
        for moment in hpi_result['moments']:
            logger.debug("%g %g %g" % tuple(moment))
    errors = np.linalg.norm(hpi_rrs - hpi_rrs_fit, axis=1)
    logger.debug('HPIFIT errors:  %s mm.'
                 % ', '.join('%0.1f' % (1000. * e) for e in errors))
    if errors.sum() < len(errors) * dist_limit:
        logger.info('HPI consistency of isotrak and hpifit is OK.')
    elif not adjust and (len(used) == len(hpi_dig)):
        warn('HPI consistency of isotrak and hpifit is poor.')
    else:
        # adjust HPI coil locations using the hpifit transformation
        for hi, (err, r_fit) in enumerate(zip(errors, hpi_rrs_fit)):
            # transform to head frame
            d = 1000 * err
            if not adjust:
                if err >= dist_limit:
                    warn('Discrepancy of HPI coil %d isotrak and hpifit is '
                         '%.1f mm!' % (hi + 1, d))
            elif hi + 1 not in used:
                if goodness[hi] >= good_limit:
                    logger.info('Note: HPI coil %d isotrak is adjusted by '
                                '%.1f mm!' % (hi + 1, d))
                    hpi_rrs[hi] = r_fit
                else:
                    warn('Discrepancy of HPI coil %d isotrak and hpifit of '
                         '%.1f mm was not adjusted!' % (hi + 1, d))
    logger.debug('HP fitting limits: err = %.1f mm, gval = %.3f.'
                 % (1000 * dist_limit, good_limit))

    return hpi_rrs.astype(float)


def _magnetic_dipole_objective(x, B, B2, coils, whitener, too_close):
    """Project data onto right eigenvectors of whitened forward."""
    fwd = _magnetic_dipole_field_vec(x[np.newaxis], coils, too_close)
    return _magnetic_dipole_delta(fwd, whitener, B, B2)


@jit()
def _magnetic_dipole_delta(fwd, whitener, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    fwd = np.dot(fwd, whitener.T)
    one = np.linalg.svd(fwd, full_matrices=False)[2]
    one = np.dot(one, B)
    Bm2 = np.dot(one, one)
    return B2 - Bm2


def _magnetic_dipole_delta_multi(whitened_fwd_svd, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    one = np.matmul(whitened_fwd_svd, B)
    Bm2 = np.sum(one * one, axis=1)
    return B2 - Bm2


def _fit_magnetic_dipole(B_orig, x0, too_close, hpi, guesses=None):
    """Fit a single bit of data (x0 = pos)."""
    from scipy.optimize import fmin_cobyla
    B = np.dot(hpi['whitener'], B_orig)
    B2 = np.dot(B, B)
    objective = partial(_magnetic_dipole_objective, B=B, B2=B2,
                        coils=hpi['coils'], whitener=hpi['whitener'],
                        too_close=too_close)
    if guesses is not None:
        res0 = objective(x0)
        res = _magnetic_dipole_delta_multi(
            guesses['whitened_fwd_svd'], B, B2)
        assert res.shape == (guesses['rr'].shape[0],)
        idx = np.argmin(res)
        if res[idx] < res0:
            x0 = guesses['rr'][idx]
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-3, rhoend=1e-5, disp=False)
    return x, 1. - objective(x) / B2


@jit()
def _chpi_objective(x, coil_dev_rrs, coil_head_rrs):
    """Compute objective function."""
    d = np.dot(coil_dev_rrs, quat_to_rot(x[:3]).T)
    d += x[3:]
    d -= coil_head_rrs
    d *= d
    return d.sum()


def _fit_chpi_quat(coil_dev_rrs, coil_head_rrs):
    """Fit rotation and translation (quaternion) parameters for cHPI coils."""
    denom = np.linalg.norm(coil_head_rrs - np.mean(coil_head_rrs, axis=0))
    denom *= denom
    # We could try to solve it the analytic way:
    quat = _fit_matched_points(coil_dev_rrs, coil_head_rrs)
    gof = 1. - _chpi_objective(quat, coil_dev_rrs, coil_head_rrs) / denom
    return quat, gof


def _fit_coil_order_dev_head_trans(dev_pnts, head_pnts):
    """Compute Device to Head transform allowing for permutiatons of points."""
    id_quat = np.zeros(6)
    best_order = None
    best_g = -999
    best_quat = id_quat
    for this_order in itertools.permutations(np.arange(len(head_pnts))):
        head_pnts_tmp = head_pnts[np.array(this_order)]
        this_quat, g = _fit_chpi_quat(dev_pnts, head_pnts_tmp)
        assert np.linalg.det(quat_to_rot(this_quat[:3])) > 0.9999
        # For symmetrical arrangements, flips can produce roughly
        # equivalent g values. To avoid this, heavily penalize
        # large rotations.
        rotation = _angle_between_quats(this_quat[:3], np.zeros(3))
        check_g = g * max(1. - rotation / np.pi, 0) ** 0.25
        if check_g > best_g:
            out_g = g
            best_g = check_g
            best_order = np.array(this_order)
            best_quat = this_quat

    # Convert Quaterion to transform
    dev_head_t = _quat_to_affine(best_quat)
    return dev_head_t, best_order, out_g


@verbose
def _setup_hpi_fitting(info, t_window, remove_aliased=False, ext_order=1,
                       verbose=None):
    """Generate HPI structure for HPI localization.

    Returns
    -------
    hpi : dict
        Dictionary of parameters representing the cHPI system and needed to
        perform head localization.
    """
    # grab basic info.
    hpi_freqs, hpi_pick, hpi_ons = _get_hpi_info(info)
    _validate_type(t_window, (str, 'numeric'), 't_window')
    if isinstance(t_window, str):
        if t_window != 'auto':
            raise ValueError('t_window must be "auto" if a string, got %r'
                             % (t_window,))
        t_window = max(5. / min(hpi_freqs), 1. / np.diff(hpi_freqs).min())
    t_window = float(t_window)
    if t_window <= 0:
        raise ValueError('t_window (%s) must be > 0' % (t_window,))
    logger.info('Using time window: %0.1f ms' % (1000 * t_window,))
    model_n_window = int(round(float(t_window) * info['sfreq']))
    # worry about resampled/filtered data.
    # What to do e.g. if Raw has been resampled and some of our
    # HPI freqs would now be aliased
    highest = info.get('lowpass')
    highest = info['sfreq'] / 2. if highest is None else highest
    keepers = hpi_freqs <= highest
    if remove_aliased:
        hpi_freqs = hpi_freqs[keepers]
        hpi_ons = hpi_ons[keepers]
    elif not keepers.all():
        raise RuntimeError('Found HPI frequencies %s above the lowpass '
                           '(or Nyquist) frequency %0.1f'
                           % (hpi_freqs[~keepers].tolist(), highest))
    if info['line_freq'] is not None:
        line_freqs = np.arange(info['line_freq'], info['sfreq'] / 3.,
                               info['line_freq'])
    else:
        line_freqs = np.zeros([0])
    logger.info('Line interference frequencies: %s Hz'
                % ' '.join(['%d' % l for l in line_freqs]))

    # build model to extract sinusoidal amplitudes.
    slope = np.linspace(-0.5, 0.5, model_n_window)[:, np.newaxis]
    rps = np.arange(model_n_window)[:, np.newaxis].astype(float)
    rps *= 2 * np.pi / info['sfreq']  # radians/sec
    f_t = hpi_freqs[np.newaxis, :] * rps
    l_t = line_freqs[np.newaxis, :] * rps
    model = [np.sin(f_t), np.cos(f_t)]  # hpi freqs
    model += [np.sin(l_t), np.cos(l_t)]  # line freqs
    model += [slope, np.ones(slope.shape)]
    model = np.concatenate(model, axis=1)
    inv_model = np.array(linalg.pinv(model))  # ensure aligned, etc.

    # Set up magnetic dipole fits
    meg_picks = pick_types(info, meg=True, eeg=False, exclude='bads')
    info = pick_info(info, meg_picks)  # makes a copy

    coils = _create_meg_coils(info['chs'], 'accurate')
    coils = _concatenate_coils(coils)

    # Set up external model for interference suppression
    projs = list()
    _, _, _, _, mag_or_fine = _get_mf_picks(
        info, int_order=0, ext_order=ext_order, ignore_ref=True,
        verbose='error')
    mf_coils = _prep_mf_coils(
        pick_info(info, pick_types(info)), verbose='error')
    ext = _sss_basis(
        dict(origin=(0., 0., 0.), int_order=0, ext_order=ext_order),
        mf_coils).T
    out_removes = _regularize_out(0, 1, mag_or_fine)
    ext = ext[~np.in1d(np.arange(len(ext)), out_removes)]
    ext = linalg.orth(ext.T).T
    if len(ext):
        projs.append(Projection(
            kind=FIFF.FIFFV_PROJ_ITEM_HOMOG_FIELD, desc='SSS', active=False,
            data=dict(data=ext, ncol=info['nchan'], col_names=info['ch_names'],
                      nrow=len(ext))))
    info['projs'] = projs
    cov = make_ad_hoc_cov(info, verbose=False)
    whitener, _ = compute_whitener(cov, info, verbose=False)
    proj, _ = setup_proj(info, add_eeg_ref=False, verbose=False)
    hpi = dict(meg_picks=meg_picks, hpi_pick=hpi_pick,
               model=model, inv_model=inv_model, t_window=t_window,
               on=hpi_ons, n_window=model_n_window,
               freqs=hpi_freqs, line_freqs=line_freqs,
               whitener=whitener, coils=coils, proj=proj)
    return hpi


def _time_prefix(fit_time):
    """Format log messages."""
    return ('    t=%0.3f:' % fit_time).ljust(17)


def _fit_chpi_amplitudes(raw, time_sl, hpi):
    """Fit amplitudes for each channel from each of the N cHPI sinusoids.

    Returns
    -------
    sin_fit : ndarray, shape (n_freqs, n_channels)) or None :
        The sin amplitudes matching each cHPI frequency
            or None if this time window should be skipped
    """
    # No need to detrend the data because our model has a DC term
    with use_log_level(False):
        # loads good channels
        this_data = raw[hpi['meg_picks'], time_sl][0]

    # which HPI coils to use
    if hpi['hpi_pick'] is not None:
        with use_log_level(False):
            # loads hpi_stim channel
            chpi_data = raw[hpi['hpi_pick'], time_sl][0]

        ons = (np.round(chpi_data).astype(np.int) &
               hpi['on'][:, np.newaxis]).astype(bool)
        n_on = ons.all(axis=-1).sum(axis=0)
        if not (n_on >= 3).all():
            return None
    return _fast_fit(this_data, hpi['proj'], len(hpi['freqs']), hpi['model'],
                     hpi['inv_model'])


@jit()
def _fast_fit(this_data, proj, n_freqs, model, inv_model):
    if this_data.shape[1] == model.shape[0]:
        model, inv_model = model, inv_model
    else:  # first or last window
        model = model[:this_data.shape[1]]
        inv_model = np.linalg.pinv(model)
    proj_data = np.dot(proj, this_data)
    X = np.dot(inv_model, proj_data.T)
    X_sin, X_cos = X[:n_freqs], X[n_freqs:2 * n_freqs]

    # use SVD across all sensors to estimate the sinusoid phase
    sin_fit = np.zeros((n_freqs, X_sin.shape[1]))
    for fi in range(n_freqs):
        u, s, vt = np.linalg.svd(np.vstack((X_sin[fi, :], X_cos[fi, :])),
                                 full_matrices=False)
        # the first component holds the predominant phase direction
        # (so ignore the second, effectively doing s[1] = 0):
        sin_fit[fi, :] = vt[0]

    return sin_fit


def _fit_device_hpi_positions(raw, t_win=None, initial_dev_rrs=None,
                              too_close='raise', ext_order=1):
    """Compute location of HPI coils in device coords for 1 time window.

    Returns
    -------
    coil_dev_rrs : ndarray, shape (n_CHPI, 3)
        Fit locations of each cHPI coil in device coordinates
    """
    _check_option('too_close', too_close, ['raise', 'warning', 'info'])
    # 0. determine samples to fit.
    if t_win is None:  # use the whole window
        i_win = [0, len(raw.times)]
    else:
        i_win = raw.time_as_index(t_win, use_rounding=True)

    # clamp index windows
    i_win = [max(i_win[0], 0), min(i_win[1], len(raw.times))]

    time_sl = slice(i_win[0], i_win[1])

    hpi = _setup_hpi_fitting(
        raw.info, (i_win[1] - i_win[0]) / raw.info['sfreq'],
        ext_order=ext_order)

    if initial_dev_rrs is None:
        initial_dev_rrs = []
        for i in range(len(hpi['freqs'])):
            initial_dev_rrs.append([0.0, 0.0, 0.0])

    # 1. Fit amplitudes for each channel from each of the N cHPI sinusoids
    sin_fit = _fit_chpi_amplitudes(raw, time_sl, hpi)

    # skip this window if it bad.
    # logging has already been done! Maybe turn this into an Exception
    if sin_fit is None:
        return None

    # 2. fit each HPI coil if its turned on
    outs = [_fit_magnetic_dipole(f, pos, too_close, hpi)
            for f, pos, on in zip(sin_fit, initial_dev_rrs, hpi['on'])
            if on > 0]

    coil_dev_rrs = np.array([o[0] for o in outs])
    coil_g = np.array([o[1] for o in outs])

    return coil_dev_rrs, coil_g


def _check_chpi_locs(chpi_locs):
    _validate_type(chpi_locs, dict, 'chpi_locs')
    want_ndims = dict(times=1, rrs=3, gofs=2)
    want_keys = want_ndims.keys()
    if set(want_keys).symmetric_difference(chpi_locs):
        raise ValueError('chpi_locs must be a dict with entries %s, got %s'
                         % (want_keys, sorted(chpi_locs.keys())))
    n_times = None
    for key, want_ndim in want_ndims.items():
        key_str = 'chpi_locs[%s]' % (key,)
        val = chpi_locs[key]
        _validate_type(val, np.ndarray, key_str)
        shape = val.shape
        if val.ndim != want_ndim:
            raise ValueError('%s must have ndim=%d, got %d'
                             % (key_str, want_ndim, val.ndim))
        if n_times is None:
            n_times = shape[0]
        if n_times != shape[0]:
            raise ValueError('chpi_locs have inconsistent number of time '
                             'points in %s' % (want_keys,))
    gofs, rrs = chpi_locs['gofs'], chpi_locs['rrs']
    if gofs.shape[1] != rrs.shape[1]:
        raise ValueError('chpi_locs["rrs"] had values for %d coils but '
                         'chpi_locs["gofs"] had values for %d coils'
                         % (rrs.shape[1], gofs.shape[1]))
    if rrs.shape[2] != 3:
        raise ValueError('chpi_locs["rrs"].shape[2] must be 3, got shape %s'
                         % (key_str, shape))


@verbose
def compute_head_pos(info, chpi_locs, dist_limit=0.005, gof_limit=0.98,
                     adjust_dig=False, verbose=None):
    """Compute time-varying head positions.

    Parameters
    ----------
    info : instance of Info
        Measurement information.
    %(chpi_locs)s
    dist_limit : float
        Minimum distance (m) to accept for coil position fitting.
    gof_limit : float
        Minimum goodness of fit to accept for each coil.
    %(chpi_adjust_dig)s
    %(verbose)s

    Returns
    -------
    quats : ndarray, shape (n_pos, 10)
        The ``[t, q1, q2, q3, x, y, z, gof, err, v]`` for each fit.

    See Also
    --------
    compute_chpi_locs
    extract_chpi_locs_ctf
    read_head_pos
    write_head_pos

    Notes
    -----
    .. versionadded:: 0.20
    """
    _check_chpi_locs(chpi_locs)
    hpi_dig_head_rrs = _get_hpi_initial_fit(info, adjust=adjust_dig,
                                            verbose='error')
    n_coils = len(hpi_dig_head_rrs)
    coil_dev_rrs = apply_trans(invert_transform(info['dev_head_t']),
                               hpi_dig_head_rrs)
    dev_head_t = info['dev_head_t']['trans']
    pos_0 = dev_head_t[:3, 3]
    last = dict(quat_fit_time=-0.1, coil_dev_rrs=coil_dev_rrs,
                quat=np.concatenate([rot_to_quat(dev_head_t[:3, :3]),
                                     dev_head_t[:3, 3]]))
    del coil_dev_rrs
    quats = []
    for fit_time, this_coil_dev_rrs, g_coils in zip(
            *(chpi_locs[key] for key in ('times', 'rrs', 'gofs'))):
        use_idx = np.where(g_coils >= gof_limit)[0]

        #
        # 1. Check number of good ones
        #
        if len(use_idx) < 3:
            msg = (_time_prefix(fit_time) + '%s/%s good HPI fits, cannot '
                   'determine the transformation (%s)!'
                   % (len(use_idx), n_coils,
                      ', '.join('%0.2f' % g for g in g_coils)))
            warn(msg)
            continue

        #
        # 2. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization
        #    positions) iteratively using different sets of coils.
        #
        this_quat, g, use_idx = _fit_chpi_quat_subset(
            this_coil_dev_rrs, hpi_dig_head_rrs, use_idx)

        #
        # 3. Stop if < 3 good
        #

        # Convert quaterion to transform
        this_dev_head_t = _quat_to_affine(this_quat)
        est_coil_head_rrs = apply_trans(this_dev_head_t, this_coil_dev_rrs)
        errs = np.linalg.norm(hpi_dig_head_rrs - est_coil_head_rrs, axis=1)
        n_good = ((g_coils >= gof_limit) & (errs < dist_limit)).sum()
        if n_good < 3:
            warn(_time_prefix(fit_time) + '%s/%s good HPI fits, cannot '
                 'determine the transformation (%s)!'
                 % (n_good, n_coils, ', '.join('%0.2f' % g for g in g_coils)))
            continue

        # velocities, in device coords, of HPI coils
        dt = fit_time - last['quat_fit_time']
        vs = tuple(1000. * np.linalg.norm(last['coil_dev_rrs'] -
                                          this_coil_dev_rrs, axis=1) / dt)
        logger.info(_time_prefix(fit_time) +
                    ('%s/%s good HPI fits, movements [mm/s] = ' +
                    ' / '.join(['% 8.1f'] * n_coils))
                    % ((n_good, n_coils) + vs))

        # Log results
        # MaxFilter averages over a 200 ms window for display, but we don't
        for ii in range(n_coils):
            if ii in use_idx:
                start, end = ' ', '/'
            else:
                start, end = '(', ')'
            log_str = ('    ' + start +
                       '{0:6.1f} {1:6.1f} {2:6.1f} / ' +
                       '{3:6.1f} {4:6.1f} {5:6.1f} / ' +
                       'g = {6:0.3f} err = {7:4.1f} ' +
                       end)
            vals = np.concatenate((1000 * hpi_dig_head_rrs[ii],
                                   1000 * est_coil_head_rrs[ii],
                                   [g_coils[ii], 1000 * errs[ii]]))
            if len(use_idx) >= 3:
                if ii <= 2:
                    log_str += '{8:6.3f} {9:6.3f} {10:6.3f}'
                    vals = np.concatenate(
                        (vals, this_dev_head_t[ii, :3]))
                elif ii == 3:
                    log_str += '{8:6.1f} {9:6.1f} {10:6.1f}'
                    vals = np.concatenate(
                        (vals, this_dev_head_t[:3, 3] * 1000.))
            logger.debug(log_str.format(*vals))

        # resulting errors in head coil positions
        d = 100 * np.linalg.norm(last['quat'][3:] - this_quat[3:])  # cm
        r = _angle_between_quats(last['quat'][:3], this_quat[:3]) / dt
        v = d / dt  # cm/sec
        d = 100 * np.linalg.norm(this_quat[3:] - pos_0)  # dis from 1st
        logger.debug('    #t = %0.3f, #e = %0.2f cm, #g = %0.3f, '
                     '#v = %0.2f cm/s, #r = %0.2f rad/s, #d = %0.2f cm'
                     % (fit_time, 100 * errs.mean(), g, v, r, d))
        logger.debug('    #t = %0.3f, #q = %s '
                     % (fit_time, ' '.join(map('{:8.5f}'.format, this_quat))))

        quats.append(np.concatenate(([fit_time], this_quat, [g],
                                     [errs.mean() * 100], [v])))
        last['quat_fit_time'] = fit_time
        last['quat'] = this_quat
        last['coil_dev_rrs'] = this_coil_dev_rrs
    quats = np.array(quats, np.float64)
    quats = np.zeros((0, 10)) if quats.size == 0 else quats
    return quats


def _fit_chpi_quat_subset(coil_dev_rrs, coil_head_rrs, use_idx):
    quat, g = _fit_chpi_quat(coil_dev_rrs[use_idx], coil_head_rrs[use_idx])
    out_idx = use_idx.copy()
    if len(use_idx) > 3:  # try dropping one (recursively)
        for di in range(len(use_idx)):
            this_use_idx = list(use_idx[:di]) + list(use_idx[di + 1:])
            this_quat, this_g, this_use_idx = _fit_chpi_quat_subset(
                coil_dev_rrs, coil_head_rrs, this_use_idx)
            if this_g > g:
                quat, g, out_idx = this_quat, this_g, this_use_idx
    return quat, g, np.array(out_idx, int)


@jit()
def _unit_quat_constraint(x):
    """Constrain our 3 quaternion rot params (ignoring w) to have norm <= 1."""
    return 1 - (x * x).sum()


@verbose
def compute_chpi_locs(raw, t_step_min=0.01, t_step_max=1.,
                      t_window='auto', too_close='raise',
                      adjust_dig=False, ext_order=1, verbose=None):
    """Compute locations of each cHPI coils over time.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information.
    t_step_min : float
        Minimum time step to use. If correlations are sufficiently high,
        t_step_max will be used.
    t_step_max : float
        Maximum time step to use.
    %(chpi_t_window)s
    too_close : str
        How to handle HPI positions too close to the sensors,
        can be 'raise' (default), 'warning', or 'info'.
    %(chpi_adjust_dig)s
    %(chpi_ext_order)s
    %(verbose)s

    Returns
    -------
    %(chpi_locs)s

    See Also
    --------
    read_head_pos
    write_head_pos
    compute_head_pos
    extract_chpi_locs_ctf

    Notes
    -----
    The number of fitted points ``n_pos`` will depend on the velocity of head
    movements as well as ``t_step_max`` and ``t_step_min``.

    In "auto" mode, ``t_window`` will be set to the longer of:

    1. Five cycles of the lowest HPI frequency.
          Ensures that the frequency estimate is stable.
    2. The reciprocal of the smallest difference between HPI frequencies.
          Ensures that neighboring frequencies can be disambiguated.

    .. versionadded:: 0.20
    """
    _check_option('too_close', too_close, ['raise', 'warning', 'info'])

    # extract initial geometry from info['hpi_results']
    hpi_dig_head_rrs = _get_hpi_initial_fit(raw.info, adjust=adjust_dig)

    # extract hpi system information
    hpi = _setup_hpi_fitting(raw.info, t_window, ext_order=ext_order)
    del t_window

    # move to device coords
    head_dev_t = invert_transform(raw.info['dev_head_t'])['trans']
    hpi_dig_dev_rrs = apply_trans(head_dev_t, hpi_dig_head_rrs)

    # setup last iteration structure
    last = dict(sin_fit=None, coil_fit_time=t_step_min,
                coil_dev_rrs=hpi_dig_dev_rrs)
    del hpi_dig_dev_rrs, hpi_dig_head_rrs

    # Make some location guesses (1 cm grid)
    R = np.linalg.norm(hpi['coils'][0], axis=1).min()
    guesses = _make_guesses(dict(R=R, r0=np.zeros(3)), 0.01, 0., 0.005,
                            verbose=False)[0]['rr']
    logger.info('Computing %d HPI location guesses (1 cm grid in a %0.1f cm '
                'sphere)' % (len(guesses), R * 100))
    fwd = _magnetic_dipole_field_vec(guesses, hpi['coils'], too_close)
    fwd = np.dot(fwd, hpi['whitener'].T)
    fwd.shape = (guesses.shape[0], 3, -1)
    fwd = np.linalg.svd(fwd, full_matrices=False)[2]
    guesses = dict(rr=guesses, whitened_fwd_svd=fwd)
    del fwd, R

    t_begin = raw.times[0]
    t_end = raw.times[-1]
    fit_idxs = raw.time_as_index(np.arange(
        t_begin + hpi['t_window'] / 2., t_end, t_step_min), use_rounding=True)
    logger.info('Fitting %d HPI coil locations at up to %s time points '
                '(%0.1f sec duration)'
                % (len(hpi['freqs']), len(fit_idxs), t_end - t_begin))

    fit_times = (fit_idxs + raw.first_samp -
                 hpi['n_window'] / 2.) / raw.info['sfreq']
    sin_fits = list()
    # Don't use a ProgressBar in debug mode
    with ProgressBar(fit_idxs, mesg='Pass 1: Amplitudes') as pb:
        for mi, midpt in enumerate(pb):
            #
            # 0. determine samples to fit.
            #
            fit_time = fit_times[mi]
            time_sl = midpt - hpi['n_window'] // 2
            time_sl = slice(max(time_sl, 0),
                            min(time_sl + hpi['n_window'], len(raw.times)))

            #
            # 1. Fit amplitudes for each channel from each of the N sinusoids
            #
            sin_fits.append(_fit_chpi_amplitudes(raw, time_sl, hpi))

    iter_ = list(zip(fit_times, sin_fits))
    chpi_locs = dict(times=[], rrs=[], gofs=[])
    with ProgressBar(iter_, mesg='Pass 2: Locations') as pb:
        for fit_time, sin_fit in pb:
            # skip this window if bad
            if sin_fit is None:
                continue

            # check if data has sufficiently changed
            if last['sin_fit'] is not None:  # first iteration
                corrs = np.array(
                    [np.corrcoef(s, l)[0, 1]
                     for s, l in zip(sin_fit, last['sin_fit'])])
                corrs *= corrs
                # check to see if we need to continue
                if fit_time - last['coil_fit_time'] <= t_step_max - 1e-7 and \
                        (corrs > 0.98).sum() >= 3:
                    # don't need to refit data
                    continue

            # update 'last' sin_fit *before* inplace sign mult
            last['sin_fit'] = sin_fit.copy()

            #
            # 2. Fit magnetic dipole for each coil to obtain coil positions
            #    in device coordinates
            #
            rrs, gofs = zip(
                *[_fit_magnetic_dipole(f, x0, too_close, hpi, guesses)
                  for f, x0 in zip(sin_fit, last['coil_dev_rrs'])])
            chpi_locs['times'].append(fit_time)
            chpi_locs['rrs'].append(rrs)
            chpi_locs['gofs'].append(gofs)
            last['coil_fit_time'] = fit_time
            last['coil_dev_rrs'] = rrs
    chpi_locs['times'] = np.array(chpi_locs['times'], float)
    chpi_locs['rrs'] = np.array(chpi_locs['rrs'], float).reshape(
        len(chpi_locs['times']), len(hpi['freqs']), 3)
    chpi_locs['gofs'] = np.array(chpi_locs['gofs'], float)
    return chpi_locs


def _chpi_locs_to_times_dig(chpi_locs):
    """Reformat chpi_locs as list of dig (dict)."""
    dig = list()
    for rrs, gofs in zip(*(chpi_locs[key] for key in ('rrs', 'gofs'))):
        dig.append([{'r': rr, 'ident': idx, 'gof': gof,
                     'kind': FIFF.FIFFV_POINT_HPI,
                     'coord_frame': FIFF.FIFFV_COORD_DEVICE}
                    for idx, (rr, gof) in enumerate(zip(rrs, gofs), 1)])
    return chpi_locs['times'], dig


@verbose
def filter_chpi(raw, include_line=True, t_step=0.01, t_window=None,
                ext_order=1, verbose=None):
    """Remove cHPI and line noise from data.

    .. note:: This function will only work properly if cHPI was on
              during the recording.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information. Must be preloaded. Operates in-place.
    include_line : bool
        If True, also filter line noise.
    t_step : float
        Time step to use for estimation, default is 0.01 (10 ms).
    %(chpi_t_window)s
    %(chpi_ext_order)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The raw data.

    Notes
    -----
    cHPI signals are in general not stationary, because head movements act
    like amplitude modulators on cHPI signals. Thus it is recommended to
    to use this procedure, which uses an iterative fitting method, to
    remove cHPI signals, as opposed to notch filtering.

    .. versionadded:: 0.12
    """
    if not raw.preload:
        raise RuntimeError('raw data must be preloaded')
    if t_window is None:
        warn('The default for t_window is 0.2 in MNE 0.20 but will change '
             'to "auto" in 0.21, set it explicitly to avoid this warning',
             DeprecationWarning)
        t_window = 0.2
    t_step = float(t_step)
    if t_step <= 0:
        raise ValueError('t_step (%s) must be > 0' % (t_step,))
    n_step = int(np.ceil(t_step * raw.info['sfreq']))
    hpi = _setup_hpi_fitting(raw.info, t_window, remove_aliased=True,
                             verbose=False)

    fit_idxs = np.arange(0, len(raw.times) + hpi['n_window'] // 2, n_step)
    n_freqs = len(hpi['freqs'])
    n_remove = 2 * n_freqs
    meg_picks = pick_types(raw.info, meg=True, exclude=())  # filter all chs
    n_times = len(raw.times)

    msg = 'Removing %s cHPI' % n_freqs
    if include_line:
        n_remove += 2 * len(hpi['line_freqs'])
        msg += ' and %s line harmonic' % len(hpi['line_freqs'])
    msg += ' frequencies from %s MEG channels' % len(meg_picks)

    recon = np.dot(hpi['model'][:, :n_remove], hpi['inv_model'][:n_remove]).T
    logger.info(msg)
    chunks = list()  # the chunks to subtract
    last_endpt = 0
    pb = ProgressBar(fit_idxs, mesg='Filtering')
    for ii, midpt in enumerate(pb):
        left_edge = midpt - hpi['n_window'] // 2
        time_sl = slice(max(left_edge, 0),
                        min(left_edge + hpi['n_window'], len(raw.times)))
        this_len = time_sl.stop - time_sl.start
        if this_len == hpi['n_window']:
            this_recon = recon
        else:  # first or last window
            model = hpi['model'][:this_len]
            inv_model = linalg.pinv(model)
            this_recon = np.dot(model[:, :n_remove], inv_model[:n_remove]).T
        this_data = raw._data[meg_picks, time_sl]
        subt_pt = min(midpt + n_step, n_times)
        if last_endpt != subt_pt:
            fit_left_edge = left_edge - time_sl.start + hpi['n_window'] // 2
            fit_sl = slice(fit_left_edge,
                           fit_left_edge + (subt_pt - last_endpt))
            chunks.append((subt_pt, np.dot(this_data, this_recon[:, fit_sl])))
        last_endpt = subt_pt

        # Consume (trailing) chunks that are now safe to remove because
        # our windows will no longer touch them
        if ii < len(fit_idxs) - 1:
            next_left_edge = fit_idxs[ii + 1] - hpi['n_window'] // 2
        else:
            next_left_edge = np.inf
        while len(chunks) > 0 and chunks[0][0] <= next_left_edge:
            right_edge, chunk = chunks.pop(0)
            raw._data[meg_picks,
                      right_edge - chunk.shape[1]:right_edge] -= chunk
    pb.done()
    return raw
