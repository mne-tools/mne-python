# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy import linalg, fftpack

from .io.pick import pick_types, pick_channels
from .io.constants import FIFF
from .forward import (_magnetic_dipole_field_vec, _create_meg_coils,
                      _concatenate_coils, _read_coil_defs)
from .cov import make_ad_hoc_cov, _get_whitener_data
from .transforms import (apply_trans, invert_transform, _angle_between_quats,
                         quat_to_rot, rot_to_quat)
from .utils import (verbose, logger, check_version, use_log_level,
                    _check_fname, warn)

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
    _check_fname(fname, must_exist=True, overwrite=True)
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


# ############################################################################
# Estimate positions from data

@verbose
def _get_hpi_info(info, adjust=False, verbose=None):
    """Helper to get HPI information from raw."""
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
    errors = np.sqrt(((hpi_rrs - hpi_rrs_fit) ** 2).sum(axis=1))
    logger.debug('HPIFIT errors:  %s mm.'
                 % ', '.join('%0.1f' % (1000. * e) for e in errors))
    if errors.sum() < len(errors) * hpi_result['dist_limit']:
        logger.info('HPI consistency of isotrak and hpifit is OK.')
    elif not adjust and (len(hpi_result['used']) == len(hpi_coils)):
        warn('HPI consistency of isotrak and hpifit is poor.')
    else:
        # adjust HPI coil locations using the hpifit transformation
        for hi, (r_dig, r_fit) in enumerate(zip(hpi_rrs, hpi_rrs_fit)):
            # transform to head frame
            d = 1000 * np.sqrt(((r_dig - r_fit) ** 2).sum())
            if not adjust:
                warn('Discrepancy of HPI coil %d isotrak and hpifit is %.1f '
                     'mm!' % (hi + 1, d))
            elif hi + 1 not in hpi_result['used']:
                if hpi_result['goodness'][hi] >= hpi_result['good_limit']:
                    logger.info('Note: HPI coil %d isotrak is adjusted by '
                                '%.1f mm!' % (hi + 1, d))
                    hpi_rrs[hi] = r_fit
                else:
                    warn('Discrepancy of HPI coil %d isotrak and hpifit of '
                         '%.1f mm was not adjusted!' % (hi + 1, d))
    logger.debug('HP fitting limits: err = %.1f mm, gval = %.3f.'
                 % (1000 * hpi_result['dist_limit'], hpi_result['good_limit']))

    # how cHPI active is indicated in the FIF file
    hpi_sub = info['hpi_subsystem']
    if 'event_channel' in hpi_sub:
        hpi_pick = pick_channels(info['ch_names'],
                                 [hpi_sub['event_channel']])
        hpi_pick = hpi_pick[0] if len(hpi_pick) > 0 else None
    else:
        hpi_pick = None  # there is no pick!
    hpi_on = [coil['event_bits'][0] for coil in hpi_sub['hpi_coils']]
    # not all HPI coils will actually be used
    hpi_on = np.array([hpi_on[hc['number'] - 1] for hc in hpi_coils])
    assert len(hpi_coils) == len(hpi_on)

    # get frequencies
    hpi_freqs = np.array([float(x['coil_freq']) for x in hpi_coils])
    logger.info('Using %s HPI coils: %s Hz'
                % (len(hpi_freqs), ' '.join(str(int(s)) for s in hpi_freqs)))
    return hpi_freqs, hpi_rrs, hpi_pick, hpi_on, pos_order


def _magnetic_dipole_objective(x, B, B2, coils, scale, method):
    """Project data onto right eigenvectors of whitened forward."""
    if method == 'forward':
        fwd = _magnetic_dipole_field_vec(x[np.newaxis, :], coils)
    else:
        from .preprocessing.maxwell import _sss_basis
        # Eventually we can try incorporating external bases here, which
        # is why the :3 is on the SVD below
        fwd = _sss_basis(dict(origin=x, int_order=1, ext_order=0), coils).T
    fwd = np.dot(fwd, scale.T)
    one = np.dot(linalg.svd(fwd, full_matrices=False)[2][:3], B)
    one *= one
    Bm2 = one.sum()
    return B2 - Bm2


def _fit_magnetic_dipole(B_orig, x0, coils, scale, method):
    """Fit a single bit of data (x0 = pos)."""
    from scipy.optimize import fmin_cobyla
    B = np.dot(scale, B_orig)
    B2 = np.dot(B, B)
    objective = partial(_magnetic_dipole_objective, B=B, B2=B2,
                        coils=coils, scale=scale, method=method)
    x = fmin_cobyla(objective, x0, (), rhobeg=1e-4, rhoend=1e-5, disp=False)
    return x, 1. - objective(x) / B2


def _chpi_objective(x, coil_dev_rrs, coil_head_rrs):
    """Helper objective function."""
    d = np.dot(coil_dev_rrs, quat_to_rot(x[:3]).T)
    d += x[3:] / 10.  # in decimeters to get quats and head units close
    d -= coil_head_rrs
    d *= d
    return d.sum()


def _unit_quat_constraint(x):
    """Constrain our 3 quaternion rot params (ignoring w) to have norm <= 1."""
    return 1 - (x * x).sum()


def _fit_chpi_pos(coil_dev_rrs, coil_head_rrs, x0):
    """Fit rotation and translation parameters for cHPI coils."""
    from scipy.optimize import fmin_cobyla
    denom = np.sum((coil_head_rrs - np.mean(coil_head_rrs, axis=0)) ** 2)
    objective = partial(_chpi_objective, coil_dev_rrs=coil_dev_rrs,
                        coil_head_rrs=coil_head_rrs)
    x0 = x0.copy()
    x0[3:] *= 10.  # decimeters to get quats and head units close
    x = fmin_cobyla(objective, x0, _unit_quat_constraint,
                    rhobeg=1e-3, rhoend=1e-5, disp=False)
    result = objective(x)
    x[3:] /= 10.
    return x, 1. - result / denom


@verbose
def _setup_chpi_fits(info, t_window, t_step_min, method='forward',
                     exclude='bads', add_hpi_stim_pick=True,
                     remove_aliased=False, verbose=None):
    """Helper to set up cHPI fits."""
    from scipy.spatial.distance import cdist
    from .preprocessing.maxwell import _prep_mf_coils
    if not (check_version('numpy', '1.7') and check_version('scipy', '0.11')):
        raise RuntimeError('numpy>=1.7 and scipy>=0.11 required')
    hpi_freqs, coil_head_rrs, hpi_pick, hpi_ons = _get_hpi_info(info)[:4]
    # What to do e.g. if Raw has been resampled and some of our
    # HPI freqs would now be aliased
    highest = info.get('lowpass')
    highest = info['sfreq'] / 2. if highest is None else highest
    keepers = np.array([h <= highest for h in hpi_freqs], bool)
    if remove_aliased:
        hpi_freqs = hpi_freqs[keepers]
        coil_head_rrs = coil_head_rrs[keepers]
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
    # initial transforms
    dev_head_t = info['dev_head_t']['trans']
    head_dev_t = invert_transform(info['dev_head_t'])['trans']
    # determine timing
    n_window = int(round(t_window * info['sfreq']))
    logger.debug('Coordinate transformation:')
    for d in (dev_head_t[0, :3], dev_head_t[1, :3], dev_head_t[2, :3],
              dev_head_t[:3, 3] * 1000.):
        logger.debug('{0:8.4f} {1:8.4f} {2:8.4f}'.format(*d))
    slope = np.arange(n_window).astype(np.float64)[:, np.newaxis]
    slope -= np.mean(slope)
    rads = slope / info['sfreq']
    rads *= 2 * np.pi
    f_t = hpi_freqs[np.newaxis, :] * rads
    l_t = line_freqs[np.newaxis, :] * rads
    model = [np.sin(f_t), np.cos(f_t)]  # hpi freqs
    model += [np.sin(l_t), np.cos(l_t)]  # line freqs
    model += [slope, np.ones(slope.shape)]
    model = np.concatenate(model, axis=1)
    inv_model = linalg.pinv(model)
    # Set up highpass at half lowest cHPI freq
    hp_n = 2 ** (int(np.ceil(np.log2(n_window))) + 1)
    freqs = fftpack.rfftfreq(hp_n, 1. / info['sfreq'])
    hp_ind = np.where(freqs >= hpi_freqs.min())[0][0] - 2
    hp_window = np.concatenate(
        [[0], np.repeat(np.hanning(hp_ind - 1)[:(hp_ind - 1) // 2],
                        2)])[np.newaxis]

    # Set up magnetic dipole fits
    picks_meg = pick_types(info, meg=True, eeg=False, exclude=exclude)
    if len(exclude) > 0:
        if exclude == 'bads':
            msg = info['bads']
        else:
            msg = exclude
        logger.debug('Static bad channels (%d): %s'
                     % (len(msg), u' '.join(msg)))
    if add_hpi_stim_pick:
        if hpi_pick is None:
            raise RuntimeError('Could not find HPI status channel')
        picks = np.concatenate([picks_meg, [hpi_pick]])
    else:
        picks = picks_meg
    megchs = [ch for ci, ch in enumerate(info['chs']) if ci in picks_meg]
    templates = _read_coil_defs(elekta_defs=True, verbose=False)
    coils = _create_meg_coils(megchs, 'accurate', coilset=templates)
    if method == 'forward':
        coils = _concatenate_coils(coils)
    else:  # == 'multipole'
        coils = _prep_mf_coils(info)
    scale = make_ad_hoc_cov(info, verbose=False)
    scale = _get_whitener_data(info, scale, picks_meg, verbose=False)
    orig_dev_head_quat = np.concatenate([rot_to_quat(dev_head_t[:3, :3]),
                                         dev_head_t[:3, 3]])
    dists = cdist(coil_head_rrs, coil_head_rrs)
    hpi = dict(dists=dists, scale=scale, picks=picks, model=model,
               inv_model=inv_model, coil_head_rrs=coil_head_rrs,
               coils=coils, on=hpi_ons, n_window=n_window, method=method,
               freqs=hpi_freqs, line_freqs=line_freqs,
               hp_ind=hp_ind, hp_n=hp_n, hp_window=hp_window)
    last = dict(quat=orig_dev_head_quat, coil_head_rrs=coil_head_rrs,
                coil_dev_rrs=apply_trans(head_dev_t, coil_head_rrs),
                sin_fit=None, fit_time=-t_step_min)
    return hpi, last


def _time_prefix(fit_time):
    """Helper to format log messages."""
    return ('    t=%0.3f:' % fit_time).ljust(17)


@verbose
def _calculate_chpi_positions(raw, t_step_min=0.1, t_step_max=10.,
                              t_window=0.2, dist_limit=0.005, gof_limit=0.98,
                              verbose=None):
    """Calculate head positions using cHPI coils.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    quats : ndarray, shape (N, 10)
        The ``[t, q1, q2, q3, x, y, z, gof, err, v]`` for each fit.

    Notes
    -----
    The number of time points ``N`` will depend on the velocity of head
    movements as well as ``t_step_max`` and ``t_step_min``.

    See Also
    --------
    read_head_pos
    write_head_pos
    """
    from scipy.spatial.distance import cdist
    hpi, last = _setup_chpi_fits(raw.info, t_window, t_step_min)

    t_begin = raw.times[0]
    t_end = raw.times[-1]
    fit_idxs = raw.time_as_index(np.arange(t_begin + t_window / 2., t_end,
                                           t_step_min),
                                 use_rounding=True)
    quats = []
    logger.info('Fitting up to %s time points (%0.1f sec duration)'
                % (len(fit_idxs), t_end - t_begin))
    pos_0 = None
    n_freqs = len(hpi['freqs'])
    for midpt in fit_idxs:
        #
        # 1. Fit amplitudes for each channel from each of the N cHPI sinusoids
        #
        fit_time = (midpt + raw.first_samp - hpi['n_window'] / 2.) /\
            raw.info['sfreq']
        time_sl = midpt - hpi['n_window'] // 2
        time_sl = slice(max(time_sl, 0),
                        min(time_sl + hpi['n_window'], len(raw.times)))
        with use_log_level(False):
            # loads good channels with hpi_stim
            meg_chpi_data = raw[hpi['picks'], time_sl][0]
        mchpi_data = np.mean(meg_chpi_data[:-1], axis=1, keepdims=True)
        this_data = meg_chpi_data[:-1] - mchpi_data
        chpi_data = meg_chpi_data[-1]  # copies hpi_stim
        ons = (np.round(chpi_data).astype(np.int) &
               hpi['on'][:, np.newaxis]).astype(bool)
        n_on = np.sum(ons, axis=0)
        if not (n_on >= 3).all():
            logger.info(_time_prefix(fit_time) + '%s < 3 HPI coils turned on, '
                        'skipping fit' % (n_on.min(),))
            continue
        # ons = ons.all(axis=1)  # which HPI coils to use
        this_len = time_sl.stop - time_sl.start
        if this_len == hpi['n_window']:
            model, inv_model = hpi['model'], hpi['inv_model']
        else:  # first or last window
            model = hpi['model'][:this_len]
            inv_model = linalg.pinv(model)
        X = np.dot(inv_model, this_data.T)
        X_sin, X_cos = X[:n_freqs], X[n_freqs:2 * n_freqs]

        # use SVD across all sensors to estimate the sinusoid phase
        sin_fit = np.zeros((n_freqs, X_sin.shape[1]))
        for fi in range(n_freqs):
            u, s, vt = np.linalg.svd(np.vstack((X_sin[fi, :], X_cos[fi, :])),
                                     full_matrices=False)
            # the first component holds the predominant phase direction
            # (so ignore the second, effectively doing s[1] = 0):
            X[[fi, fi + n_freqs], :] = np.outer(u[:, 0] * s[0], vt[0])
            sin_fit[fi, :] = vt[0]

        data_diff = np.dot(model, X).T - this_data
        del model, inv_model

        g_sin = 1 - np.sqrt((data_diff**2).sum() / (this_data**2).sum())
        g_chan = 1 - np.sqrt((data_diff**2).sum(axis=1) /
                             (this_data**2).sum(axis=1))

        if last['sin_fit'] is not None:  # first iteration
            corr = np.corrcoef(sin_fit.ravel(), last['sin_fit'].ravel())[0, 1]
            # check to see if we need to continue
            if fit_time - last['fit_time'] <= t_step_max - 1e-7 and \
                    corr * corr > 0.98:
                continue  # don't need to re-fit data
        last['sin_fit'] = sin_fit.copy()  # save *before* inplace sign mult

        del X_sin, X_cos, X

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        logger.debug('    HPI amplitude correlation %0.3f: %0.3f '
                     '(%s chnls > 0.90)' % (fit_time, g_sin,
                                            (g_chan > 0.90).sum()))
        outs = [_fit_magnetic_dipole(f, pos, hpi['coils'], hpi['scale'],
                                     hpi['method'])
                for f, pos in zip(sin_fit, last['coil_dev_rrs'])]
        this_coil_dev_rrs = np.array([o[0] for o in outs])
        g_coils = [o[1] for o in outs]
        these_dists = cdist(this_coil_dev_rrs, this_coil_dev_rrs)
        these_dists = np.abs(hpi['dists'] - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        good = False
        use_mask = np.ones(n_freqs, bool)
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
                exclude_coils = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude_coils] = False
        good = use_mask.sum() >= 3
        if not good:
            warn(_time_prefix(fit_time) + '%s/%s good HPI fits, '
                 'cannot determine the transformation!'
                 % (use_mask.sum(), n_freqs))
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
            (quat_to_rot(this_quat[:3]),
             this_quat[3:][:, np.newaxis]), axis=1)
        this_dev_head_t = np.concatenate((this_dev_head_t, [[0, 0, 0, 1.]]))
        # velocities, in device coords, of HPI coils
        # dt = fit_time - last['fit_time'] #
        dt = t_window
        vs = tuple(1000. * np.sqrt(np.sum((last['coil_dev_rrs'] -
                                           this_coil_dev_rrs) ** 2,
                                          axis=1)) / dt)
        logger.info(_time_prefix(fit_time) +
                    ('%s/%s good HPI fits, movements [mm/s] = ' +
                     ' / '.join(['% 6.1f'] * n_freqs))
                    % ((use_mask.sum(), n_freqs) + vs))
        # resulting errors in head coil positions
        est_coil_head_rrs = apply_trans(this_dev_head_t, this_coil_dev_rrs)
        errs = 1000. * np.sqrt(((hpi['coil_head_rrs'] -
                                 est_coil_head_rrs) ** 2).sum(axis=-1))
        e = errs[use_mask].mean() / 1000.  # mm -> m
        d = 100 * np.sqrt(np.sum(last['quat'][3:] - this_quat[3:]) ** 2)  # cm
        r = _angle_between_quats(last['quat'][:3], this_quat[:3]) / dt
        v = d / dt  # cm/sec
        if pos_0 is None:
            pos_0 = this_quat[3:].copy()
        d = 100 * np.sqrt(np.sum((this_quat[3:] - pos_0) ** 2))  # dis from 1st
        # MaxFilter averages over a 200 ms window for display, but we don't
        for ii in range(n_freqs):
            if use_mask[ii]:
                start, end = ' ', '/'
            else:
                start, end = '(', ')'
            log_str = ('    ' + start +
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
                                   [g_coils[ii], errs[ii]]))  # errs in mm
            if ii <= 2:
                vals = np.concatenate((vals, this_dev_head_t[ii, :3]))
            elif ii == 3:
                vals = np.concatenate((vals, this_dev_head_t[:3, 3] * 1000.))
            logger.debug(log_str.format(*vals))
        logger.debug('    #t = %0.3f, #e = %0.2f cm, #g = %0.3f, '
                     '#v = %0.2f cm/s, #r = %0.2f rad/s, #d = %0.2f cm'
                     % (fit_time, 100 * e, g, v, r, d))
        logger.debug('    #t = %0.3f, #q = %s '
                     % (fit_time, ' '.join(map('{:8.5f}'.format, this_quat))))
        quats.append(np.concatenate(([fit_time], this_quat, [g],
                                     [e * 100], [v])))  # e in centimeters
        last['fit_time'] = fit_time
        last['quat'] = this_quat
        last['coil_dev_rrs'] = this_coil_dev_rrs
    logger.info('[done]')
    quats = np.array(quats, np.float64)
    quats = np.zeros((0, 10)) if quats.size == 0 else quats
    return quats


@verbose
def filter_chpi(raw, include_line=True, verbose=None):
    """Remove cHPI and line noise from data.

    .. note:: This function will only work properly if cHPI was on
              during the recording.

    Parameters
    ----------
    raw : instance of Raw
        Raw data with cHPI information. Must be preloaded. Operates in-place.
    include_line : bool
        If True, also filter line noise.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
    t_window = 0.2
    t_step = 0.01
    n_step = int(np.ceil(t_step * raw.info['sfreq']))
    hpi = _setup_chpi_fits(raw.info, t_window, t_window, exclude='bads',
                           add_hpi_stim_pick=False, remove_aliased=True,
                           verbose=False)[0]
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

    proj = np.dot(hpi['model'][:, :n_remove], hpi['inv_model'][:n_remove]).T
    logger.info(msg)
    chunks = list()  # the chunks to subtract
    last_endpt = 0
    last_done = 0.
    next_done = 60.
    for ii, midpt in enumerate(fit_idxs):
        if midpt / raw.info['sfreq'] >= next_done or ii == len(fit_idxs) - 1:
            logger.info('    Filtering % 5.1f - % 5.1f sec'
                        % (last_done, min(next_done, raw.times[-1])))
            last_done = next_done
            next_done += 60.
        left_edge = midpt - hpi['n_window'] // 2
        time_sl = slice(max(left_edge, 0),
                        min(left_edge + hpi['n_window'], len(raw.times)))
        this_len = time_sl.stop - time_sl.start
        if this_len == hpi['n_window']:
            this_proj = proj
        else:  # first or last window
            model = hpi['model'][:this_len]
            inv_model = linalg.pinv(model)
            this_proj = np.dot(model[:, :n_remove], inv_model[:n_remove]).T
        this_data = raw._data[meg_picks, time_sl]
        subt_pt = min(midpt + n_step, n_times)
        if last_endpt != subt_pt:
            fit_left_edge = left_edge - time_sl.start + hpi['n_window'] // 2
            fit_sl = slice(fit_left_edge,
                           fit_left_edge + (subt_pt - last_endpt))
            chunks.append((subt_pt, np.dot(this_data, this_proj[:, fit_sl])))
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
    return raw
