# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from collections.abc import Iterable

import numpy as np

from ..event import _get_stim_channel
from .._ola import _Interp2
from ..io.pick import (pick_types, pick_info, pick_channels,
                       pick_channels_forward)
from ..cov import make_ad_hoc_cov, read_cov, Covariance
from ..bem import fit_sphere_to_headshape, make_sphere_model, read_bem_solution
from ..io import RawArray, BaseRaw, Info
from ..chpi import (read_head_pos, head_pos_to_trans_rot_t, get_chpi_info,
                    _get_hpi_initial_fit)
from ..io.constants import FIFF
from ..forward import (_magnetic_dipole_field_vec, _merge_meg_eeg_fwds,
                       _stc_src_sel, convert_forward_solution,
                       _prepare_for_forward, _transform_orig_meg_coils,
                       _compute_forwards, _to_forward_dict,
                       restrict_forward_to_stc, _prep_meg_channels)
from ..transforms import _get_trans, transform_surface_to
from ..source_space import (_ensure_src, _set_source_space_vertices,
                            setup_volume_source_space)
from ..source_estimate import _BaseSourceEstimate
from ..surface import _CheckInside
from ..utils import (logger, verbose, check_random_state, _pl, _validate_type,
                     _check_preload)
from ..parallel import check_n_jobs
from .source import SourceSimulator


def _check_cov(info, cov):
    """Check that the user provided a valid covariance matrix for the noise."""
    if isinstance(cov, Covariance) or cov is None:
        pass
    elif isinstance(cov, dict):
        cov = make_ad_hoc_cov(info, cov, verbose=False)
    elif isinstance(cov, str):
        if cov == 'simple':
            cov = make_ad_hoc_cov(info, None, verbose=False)
        else:
            cov = read_cov(cov, verbose=False)
    else:
        raise TypeError('Covariance matrix type not recognized. Valid input '
                        'types are: instance of Covariance, dict, str, None. '
                        ', got %s' % (cov,))
    return cov


def _check_stc_iterable(stc, info):
    # 1. Check that our STC is iterable (or convert it to one using cycle)
    # 2. Do first iter so we can get the vertex subselection
    # 3. Get the list of verts, which must stay the same across iterations
    if isinstance(stc, _BaseSourceEstimate):
        stc = [stc]
    _validate_type(stc, Iterable, 'SourceEstimate, tuple, or iterable')
    stc_enum = enumerate(stc)
    del stc

    try:
        stc_counted = next(stc_enum)
    except StopIteration:
        raise RuntimeError('Iterable did not provide stc[0]')
    _, _, verts = _stc_data_event(stc_counted, 1, info['sfreq'])
    return stc_enum, stc_counted, verts


def _log_ch(start, info, ch):
    """Log channel information."""
    if ch is not None:
        extra, just, ch = ' stored on channel:', 50, info['ch_names'][ch]
    else:
        extra, just, ch = ' not stored', 0, ''
    logger.info((start + extra).ljust(just) + ch)


def _check_head_pos(head_pos, info, first_samp, times=None):
    if head_pos is None:  # use pos from info['dev_head_t']
        head_pos = dict()
    if isinstance(head_pos, str):  # can be a head pos file
        head_pos = read_head_pos(head_pos)
    if isinstance(head_pos, np.ndarray):  # can be head_pos quats
        head_pos = head_pos_to_trans_rot_t(head_pos)
    if isinstance(head_pos, tuple):  # can be quats converted to trans, rot, t
        transs, rots, ts = head_pos
        first_time = first_samp / info['sfreq']
        ts = ts - first_time  # MF files need reref
        dev_head_ts = [np.r_[np.c_[r, t[:, np.newaxis]], [[0, 0, 0, 1]]]
                       for r, t in zip(rots, transs)]
        del transs, rots
    elif isinstance(head_pos, dict):
        ts = np.array(list(head_pos.keys()), float)
        ts.sort()
        dev_head_ts = [head_pos[float(tt)] for tt in ts]
    else:
        raise TypeError('unknown head_pos type %s' % type(head_pos))
    bad = ts < 0
    if bad.any():
        raise RuntimeError('All position times must be >= 0, found %s/%s'
                           '< 0' % (bad.sum(), len(bad)))
    if times is not None:
        bad = ts > times[-1]
        if bad.any():
            raise RuntimeError('All position times must be <= t_end (%0.1f '
                               'sec), found %s/%s bad values (is this a split '
                               'file?)' % (times[-1], bad.sum(), len(bad)))
    # If it starts close to zero, make it zero (else unique(offset) fails)
    if len(ts) > 0 and ts[0] < (0.5 / info['sfreq']):
        ts[0] = 0.
    # If it doesn't start at zero, insert one at t=0
    elif len(ts) == 0 or ts[0] > 0:
        ts = np.r_[[0.], ts]
        dev_head_ts.insert(0, info['dev_head_t']['trans'])
    dev_head_ts = [{'trans': d, 'to': info['dev_head_t']['to'],
                    'from': info['dev_head_t']['from']}
                   for d in dev_head_ts]
    offsets = np.round(ts * info['sfreq']).astype(int)
    assert np.array_equal(offsets, np.unique(offsets))
    assert len(offsets) == len(dev_head_ts)
    offsets = list(offsets)
    return dev_head_ts, offsets


@verbose
def simulate_raw(info, stc=None, trans=None, src=None, bem=None, head_pos=None,
                 mindist=1.0, interp='cos2', n_jobs=1, use_cps=True,
                 forward=None, first_samp=0, max_iter=10000, verbose=None):
    u"""Simulate raw data.

    Head movements can optionally be simulated using the ``head_pos``
    parameter.

    Parameters
    ----------
    %(info_not_none)s Used for simulation.

        .. versionchanged:: 0.18
           Support for :class:`mne.Info`.
    stc : iterable | SourceEstimate | SourceSimulator
        The source estimates to use to simulate data. Each must have the same
        sample rate as the raw data, and the vertices of all stcs in the
        iterable must match. Each entry in the iterable can also be a tuple of
        ``(SourceEstimate, ndarray)`` to allow specifying the stim channel
        (e.g., STI001) data accompany the source estimate.
        See Notes for details.

        .. versionchanged:: 0.18
           Support for tuple, iterable of tuple or `~mne.SourceEstimate`,
           or `~mne.simulation.SourceSimulator`.
    trans : dict | str | None
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of ``.fif`` or ``.fif.gz`` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the ``--trans`` MNE-C
        option). If trans is None, an identity transform will be used.
    src : str | instance of SourceSpaces | None
        Source space corresponding to the stc. If string, should be a source
        space filename. Can also be an instance of loaded or generated
        SourceSpaces. Can be None if ``forward`` is provided.
    bem : str | dict | None
        BEM solution  corresponding to the stc. If string, should be a BEM
        solution filename (e.g., "sample-5120-5120-5120-bem-sol.fif").
        Can be None if ``forward`` is provided.
    %(head_pos)s
        See for example :footcite:`LarsonTaulu2017`.
    mindist : float
        Minimum distance between sources and the inner skull boundary
        to use during forward calculation.
    %(interp)s
    %(n_jobs)s
    %(use_cps)s
    forward : instance of Forward | None
        The forward operator to use. If None (default) it will be computed
        using ``bem``, ``trans``, and ``src``. If not None,
        ``bem``, ``trans``, and ``src`` are ignored.

        .. versionadded:: 0.17
    first_samp : int
        The first_samp property in the output Raw instance.

        .. versionadded:: 0.18
    max_iter : int
        The maximum number of STC iterations to allow.
        This is a sanity parameter to prevent accidental blowups.

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The simulated raw file.

    See Also
    --------
    mne.chpi.read_head_pos
    add_chpi
    add_noise
    add_ecg
    add_eog
    simulate_evoked
    simulate_stc
    simulate_sparse_stc

    Notes
    -----
    **Stim channel encoding**

    By default, the stimulus channel will have the head position number
    (starting at 1) stored in the trigger channel (if available) at the
    t=0 point in each repetition of the ``stc``. If ``stc`` is a tuple of
    ``(SourceEstimate, ndarray)`` the array values will be placed in the
    stim channel aligned with the :class:`mne.SourceEstimate`.

    **Data simulation**

    In the most advanced case where ``stc`` is an iterable of tuples the output
    will be concatenated in time as:

    .. table:: Data alignment and stim channel encoding

       +---------+--------------------------+--------------------------+---------+
       | Channel | Data                                                          |
       +=========+==========================+==========================+=========+
       | M/EEG   | ``fwd @ stc[0][0].data`` | ``fwd @ stc[1][0].data`` | ``...`` |
       +---------+--------------------------+--------------------------+---------+
       | STIM    | ``stc[0][1]``            | ``stc[1][1]``            | ``...`` |
       +---------+--------------------------+--------------------------+---------+
       |         | *time →*                                                      |
       +---------+--------------------------+--------------------------+---------+

    .. versionadded:: 0.10.0

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    _validate_type(info, Info, 'info')
    raw_verbose = verbose

    if len(pick_types(info, meg=False, stim=True)) == 0:
        event_ch = None
    else:
        event_ch = pick_channels(info['ch_names'],
                                 _get_stim_channel(None, info))[0]

    n_jobs = check_n_jobs(n_jobs)
    if forward is not None:
        if any(x is not None for x in (trans, src, bem, head_pos)):
            raise ValueError('If forward is not None then trans, src, bem, '
                             'and head_pos must all be None')
        if not np.allclose(forward['info']['dev_head_t']['trans'],
                           info['dev_head_t']['trans'], atol=1e-6):
            raise ValueError('The forward meg<->head transform '
                             'forward["info"]["dev_head_t"] does not match '
                             'the one in raw.info["dev_head_t"]')
        src = forward['src']

    dev_head_ts, offsets = _check_head_pos(head_pos, info, first_samp, None)

    src = _ensure_src(src, verbose=False)
    if isinstance(bem, str):
        bem = read_bem_solution(bem, verbose=False)

    # Extract necessary info
    meeg_picks = pick_types(info, meg=True, eeg=True, exclude=[])
    logger.info('Setting up raw simulation: %s position%s, "%s" interpolation'
                % (len(dev_head_ts), _pl(dev_head_ts), interp))

    if isinstance(stc, SourceSimulator) and stc.first_samp != first_samp:
        logger.info('SourceSimulator first_samp does not match argument.')

    stc_enum, stc_counted, verts = _check_stc_iterable(stc, info)
    if forward is not None:
        forward = restrict_forward_to_stc(forward, verts)
        src = forward['src']
    else:
        _stc_src_sel(src, verts, on_missing='warn', extra='')
        src = _set_source_space_vertices(src.copy(), verts)

    # array used to store result
    raw_datas = list()
    _log_ch('Event information', info, event_ch)
    # don't process these any more if no MEG present
    n = 1
    get_fwd = _SimForwards(
        dev_head_ts, offsets, info, trans, src, bem, mindist, n_jobs,
        meeg_picks, forward, use_cps)
    interper = _Interp2(offsets, get_fwd, interp)

    this_start = 0
    for n in range(max_iter):
        if isinstance(stc_counted[1], (list, tuple)):
            this_n = stc_counted[1][0].data.shape[1]
        else:
            this_n = stc_counted[1].data.shape[1]
        this_stop = this_start + this_n
        logger.info('    Interval %0.3f-%0.3f sec'
                    % (this_start / info['sfreq'],
                        this_stop / info['sfreq']))
        n_doing = this_stop - this_start
        assert n_doing > 0
        this_data = np.zeros((len(info['ch_names']), n_doing))
        raw_datas.append(this_data)
        # Stim channel
        fwd, fi = interper.feed(this_stop - this_start)
        fi = fi[0]
        stc_data, stim_data, _ = _stc_data_event(
            stc_counted, fi, info['sfreq'], get_fwd.src,
            None if n == 0 else verts)
        if event_ch is not None:
            this_data[event_ch, :] = stim_data[:n_doing]
        this_data[meeg_picks] = np.einsum('svt,vt->st', fwd, stc_data)
        try:
            stc_counted = next(stc_enum)
        except StopIteration:
            logger.info('    %d STC iteration%s provided'
                        % (n + 1, _pl(n + 1)))
            break
        del fwd
    else:
        raise RuntimeError('Maximum number of STC iterations (%d) '
                           'exceeded' % (n,))
    raw_data = np.concatenate(raw_datas, axis=-1)
    raw = RawArray(raw_data, info, first_samp=first_samp, verbose=False)
    raw.set_annotations(raw.annotations)
    raw.verbose = raw_verbose
    logger.info('Done')
    return raw


@verbose
def add_eog(raw, head_pos=None, interp='cos2', n_jobs=1, random_state=None,
            verbose=None):
    """Add blink noise to raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to modify.
    %(head_pos)s
    %(interp)s
    %(n_jobs)s
    %(random_state)s
        The random generator state used for blink, ECG, and sensor noise
        randomization.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The instance, modified in place.

    See Also
    --------
    add_chpi
    add_ecg
    add_noise
    simulate_raw

    Notes
    -----
    The blink artifacts are generated by:

    1. Random activation times are drawn from an inhomogeneous poisson
       process whose blink rate oscillates between 4.5 blinks/minute
       and 17 blinks/minute based on the low (reading) and high (resting)
       blink rates from :footcite:`BentivoglioEtAl1997`.
    2. The activation kernel is a 250 ms Hanning window.
    3. Two activated dipoles are located in the z=0 plane (in head
       coordinates) at ±30 degrees away from the y axis (nasion).
    4. Activations affect MEG and EEG channels.

    The scale-factor of the activation function was chosen based on
    visual inspection to yield amplitudes generally consistent with those
    seen in experimental data. Noisy versions of the activation will be
    stored in the first EOG channel in the raw instance, if it exists.

    References
    ----------
    .. footbibliography::
    """
    return _add_exg(raw, 'blink', head_pos, interp, n_jobs, random_state)


@verbose
def add_ecg(raw, head_pos=None, interp='cos2', n_jobs=1, random_state=None,
            verbose=None):
    """Add ECG noise to raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to modify.
    %(head_pos)s
    %(interp)s
    %(n_jobs)s
    %(random_state)s
        The random generator state used for blink, ECG, and sensor noise
        randomization.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The instance, modified in place.

    See Also
    --------
    add_chpi
    add_eog
    add_noise
    simulate_raw

    Notes
    -----
    The ECG artifacts are generated by:

    1. Random inter-beat intervals are drawn from a uniform distribution
       of times corresponding to 40 and 80 beats per minute.
    2. The activation function is the sum of three Hanning windows with
       varying durations and scales to make a more complex waveform.
    3. The activated dipole is located one (estimated) head radius to
       the left (-x) of head center and three head radii below (+z)
       head center; this dipole is oriented in the +x direction.
    4. Activations only affect MEG channels.

    The scale-factor of the activation function was chosen based on
    visual inspection to yield amplitudes generally consistent with those
    seen in experimental data. Noisy versions of the activation will be
    stored in the first EOG channel in the raw instance, if it exists.

    .. versionadded:: 0.18
    """
    return _add_exg(raw, 'ecg', head_pos, interp, n_jobs, random_state)


def _add_exg(raw, kind, head_pos, interp, n_jobs, random_state):
    assert isinstance(kind, str) and kind in ('ecg', 'blink')
    _validate_type(raw, BaseRaw, 'raw')
    _check_preload(raw, 'Adding %s noise ' % (kind,))
    rng = check_random_state(random_state)
    info, times, first_samp = raw.info, raw.times, raw.first_samp
    data = raw._data
    meg_picks = pick_types(info, meg=True, eeg=False, exclude=())
    meeg_picks = pick_types(info, meg=True, eeg=True, exclude=())
    R, r0 = fit_sphere_to_headshape(info, units='m', verbose=False)[:2]
    bem = make_sphere_model(r0, head_radius=R,
                            relative_radii=(0.97, 0.98, 0.99, 1.),
                            sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)
    trans = None
    dev_head_ts, offsets = _check_head_pos(head_pos, info, first_samp, times)
    if kind == 'blink':
        # place dipoles at 45 degree angles in z=0 plane
        exg_rr = np.array([[np.cos(np.pi / 3.), np.sin(np.pi / 3.), 0.],
                           [-np.cos(np.pi / 3.), np.sin(np.pi / 3), 0.]])
        exg_rr /= np.sqrt(np.sum(exg_rr * exg_rr, axis=1, keepdims=True))
        exg_rr *= 0.96 * R
        exg_rr += r0
        # oriented upward
        nn = np.array([[0., 0., 1.], [0., 0., 1.]])
        # Blink times drawn from an inhomogeneous poisson process
        # by 1) creating the rate and 2) pulling random numbers
        blink_rate = (1 + np.cos(2 * np.pi * 1. / 60. * times)) / 2.
        blink_rate *= 12.5 / 60.
        blink_rate += 4.5 / 60.
        blink_data = rng.uniform(size=len(times)) < blink_rate / info['sfreq']
        blink_data = blink_data * (rng.uniform(size=len(times)) + 0.5)  # amps
        # Activation kernel is a simple hanning window
        blink_kernel = np.hanning(int(0.25 * info['sfreq']))
        exg_data = np.convolve(blink_data, blink_kernel,
                               'same')[np.newaxis, :] * 1e-7
        # Add rescaled noisy data to EOG ch
        ch = pick_types(info, meg=False, eeg=False, eog=True)
        picks = meeg_picks
        del blink_kernel, blink_rate, blink_data
    else:
        if len(meg_picks) == 0:
            raise RuntimeError('Can only add ECG artifacts if MEG data '
                               'channels are present')
        exg_rr = np.array([[-R, 0, -3 * R]])
        max_beats = int(np.ceil(times[-1] * 80. / 60.))
        # activation times with intervals drawn from a uniform distribution
        # based on activation rates between 40 and 80 beats per minute
        cardiac_idx = np.cumsum(rng.uniform(60. / 80., 60. / 40., max_beats) *
                                info['sfreq']).astype(int)
        cardiac_idx = cardiac_idx[cardiac_idx < len(times)]
        cardiac_data = np.zeros(len(times))
        cardiac_data[cardiac_idx] = 1
        # kernel is the sum of three hanning windows
        cardiac_kernel = np.concatenate([
            2 * np.hanning(int(0.04 * info['sfreq'])),
            -0.3 * np.hanning(int(0.05 * info['sfreq'])),
            0.2 * np.hanning(int(0.26 * info['sfreq']))], axis=-1)
        exg_data = np.convolve(cardiac_data, cardiac_kernel,
                               'same')[np.newaxis, :] * 15e-8
        # Add rescaled noisy data to ECG ch
        ch = pick_types(info, meg=False, eeg=False, ecg=True)
        picks = meg_picks
        del cardiac_data, cardiac_kernel, max_beats, cardiac_idx
        nn = np.zeros_like(exg_rr)
        nn[:, 0] = 1  # arbitrarily rightward
    del meg_picks, meeg_picks
    noise = rng.standard_normal(exg_data.shape[1]) * 5e-6
    if len(ch) >= 1:
        ch = ch[-1]
        data[ch, :] = exg_data * 1e3 + noise
    else:
        ch = None
    src = setup_volume_source_space(pos=dict(rr=exg_rr, nn=nn),
                                    sphere_units='mm')
    _log_ch('%s simulated and trace' % kind, info, ch)
    del ch, nn, noise

    used = np.zeros(len(raw.times), bool)
    get_fwd = _SimForwards(
        dev_head_ts, offsets, info, trans, src, bem, 0.005, n_jobs, picks)
    interper = _Interp2(offsets, get_fwd, interp)
    proc_lims = np.concatenate([np.arange(0, len(used), 10000), [len(used)]])
    for start, stop in zip(proc_lims[:-1], proc_lims[1:]):
        fwd, _ = interper.feed(stop - start)
        data[picks, start:stop] += np.einsum(
            'svt,vt->st', fwd, exg_data[:, start:stop])
        assert not used[start:stop].any()
        used[start:stop] = True
    assert used.all()


@verbose
def add_chpi(raw, head_pos=None, interp='cos2', n_jobs=1, verbose=None):
    """Add cHPI activations to raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to be modified.
    %(head_pos)s
    %(interp)s
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The instance, modified in place.

    Notes
    -----
    .. versionadded:: 0.18
    """
    _validate_type(raw, BaseRaw, 'raw')
    _check_preload(raw, 'Adding cHPI signals ')
    info, first_samp, times = raw.info, raw.first_samp, raw.times
    meg_picks = pick_types(info, meg=True, eeg=False, exclude=[])  # for CHPI
    if len(meg_picks) == 0:
        raise RuntimeError('Cannot add cHPI if no MEG picks are present')
    dev_head_ts, offsets = _check_head_pos(head_pos, info, first_samp, times)
    hpi_freqs, hpi_pick, hpi_ons = get_chpi_info(info, on_missing='raise')
    hpi_rrs = _get_hpi_initial_fit(info, verbose='error')
    hpi_nns = hpi_rrs / np.sqrt(np.sum(hpi_rrs * hpi_rrs,
                                       axis=1))[:, np.newaxis]
    # turn on cHPI in file
    data = raw._data
    data[hpi_pick, :] = hpi_ons.sum()
    _log_ch('cHPI status bits enbled and', info, hpi_pick)
    sinusoids = 70e-9 * np.sin(2 * np.pi * hpi_freqs[:, np.newaxis] *
                               (np.arange(len(times)) / info['sfreq']))
    info = pick_info(info, meg_picks)
    with info._unlock():
        info.update(projs=[], bads=[])  # Ensure no 'projs' or 'bads'
    megcoils, _, _, _ = _prep_meg_channels(info, ignore_ref=False)
    used = np.zeros(len(raw.times), bool)
    dev_head_ts.append(dev_head_ts[-1])  # ZOH after time ends
    get_fwd = _HPIForwards(offsets, dev_head_ts, megcoils, hpi_rrs, hpi_nns)
    interper = _Interp2(offsets, get_fwd, interp)
    lims = np.concatenate([offsets, [len(raw.times)]])
    for start, stop in zip(lims[:-1], lims[1:]):
        fwd, = interper.feed(stop - start)
        data[meg_picks, start:stop] += np.einsum(
            'svt,vt->st', fwd, sinusoids[:, start:stop])
        assert not used[start:stop].any()
        used[start:stop] = True
    assert used.all()
    return raw


class _HPIForwards(object):

    def __init__(self, offsets, dev_head_ts, megcoils, hpi_rrs, hpi_nns):
        self.offsets = offsets
        self.dev_head_ts = dev_head_ts
        self.hpi_rrs = hpi_rrs
        self.hpi_nns = hpi_nns
        self.megcoils = megcoils
        self.idx = 0

    def __call__(self, offset):
        assert offset == self.offsets[self.idx]
        _transform_orig_meg_coils(self.megcoils, self.dev_head_ts[self.idx])
        fwd = _magnetic_dipole_field_vec(self.hpi_rrs, self.megcoils).T
        # align cHPI magnetic dipoles in approx. radial direction
        fwd = np.array([np.dot(fwd[:, 3 * ii:3 * (ii + 1)], self.hpi_nns[ii])
                        for ii in range(len(self.hpi_rrs))]).T
        self.idx += 1
        return (fwd,)


def _stc_data_event(stc_counted, head_idx, sfreq, src=None, verts=None):
    stc_idx, stc = stc_counted
    if isinstance(stc, (list, tuple)):
        if len(stc) != 2:
            raise ValueError('stc, if tuple, must be length 2, got %s'
                             % (len(stc),))
        stc, stim_data = stc
    else:
        stim_data = None
    _validate_type(stc, _BaseSourceEstimate, 'stc',
                   'SourceEstimate or tuple with first entry SourceEstimate')
    # Convert event data
    if stim_data is None:
        stim_data = np.zeros(len(stc.times), int)
        stim_data[np.argmin(np.abs(stc.times))] = head_idx
    del head_idx
    _validate_type(stim_data, np.ndarray, 'stim_data')
    if stim_data.dtype.kind != 'i':
        raise ValueError('stim_data in a stc tuple must be an integer ndarray,'
                         ' got dtype %s' % (stim_data.dtype,))
    if stim_data.shape != (len(stc.times),):
        raise ValueError('event data had shape %s but needed to be (%s,) to'
                         'match stc' % (stim_data.shape, len(stc.times)))
    # Validate STC
    if not np.allclose(sfreq, 1. / stc.tstep):
        raise ValueError('stc and info must have same sample rate, '
                         'got %s and %s' % (1. / stc.tstep, sfreq))
    if len(stc.times) <= 2:  # to ensure event encoding works
        raise ValueError('stc must have at least three time points, got %s'
                         % (len(stc.times),))
    verts_ = stc.vertices
    if verts is None:
        assert stc_idx == 0
    else:
        if len(verts) != len(verts_) or not all(
                np.array_equal(a, b) for a, b in zip(verts, verts_)):
            raise RuntimeError('Vertex mismatch for stc[%d], '
                               'all stc.vertices must match' % (stc_idx,))
    stc_data = stc.data
    if src is None:
        assert stc_idx == 0
    else:
        # on_missing depends on whether or not this is the first iteration
        on_missing = 'warn' if verts is None else 'ignore'
        _, stc_sel, _ = _stc_src_sel(src, stc, on_missing=on_missing)
        stc_data = stc_data[stc_sel]
    return stc_data, stim_data, verts_


class _SimForwards(object):

    def __init__(self, dev_head_ts, offsets, info, trans, src, bem, mindist,
                 n_jobs, meeg_picks, forward=None, use_cps=True):
        self.idx = 0
        self.offsets = offsets
        self.use_cps = use_cps
        self.iter = iter(_iter_forward_solutions(
            info, trans, src, bem, dev_head_ts, mindist, n_jobs, forward,
            meeg_picks))

    def __call__(self, offset):
        assert self.offsets[self.idx] == offset
        self.idx += 1
        fwd = next(self.iter)
        self.src = fwd['src']
        # XXX eventually we could speed this up by allowing the forward
        # solution code to only compute the normal direction
        convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                 use_cps=self.use_cps, copy=False,
                                 verbose=False)
        return fwd['sol']['data'], np.array(self.idx, float)


def _iter_forward_solutions(info, trans, src, bem, dev_head_ts, mindist,
                            n_jobs, forward, picks):
    """Calculate a forward solution for a subject."""
    logger.info('Setting up forward solutions')
    info = pick_info(info, picks)
    with info._unlock():
        info.update(projs=[], bads=[])  # Ensure no 'projs' or 'bads'
    mri_head_t, trans = _get_trans(trans)
    megcoils, meg_info, compcoils, megnames, eegels, eegnames, rr, info, \
        update_kwargs, bem = _prepare_for_forward(
            src, mri_head_t, info, bem, mindist, n_jobs, allow_bem_none=True,
            verbose=False)
    del (src, mindist)

    if forward is None:
        eegfwd = _compute_forwards(rr, bem, [eegels], [None],
                                   [None], ['eeg'], n_jobs, verbose=False)[0]
        eegfwd = _to_forward_dict(eegfwd, eegnames)
    else:
        if len(eegnames) > 0:
            eegfwd = pick_channels_forward(forward, eegnames, verbose=False)
        else:
            eegfwd = None

    # short circuit here if there are no MEG channels (don't need to iterate)
    if len(pick_types(info, meg=True)) == 0:
        eegfwd.update(**update_kwargs)
        for _ in dev_head_ts:
            yield eegfwd
        yield eegfwd
        return

    coord_frame = FIFF.FIFFV_COORD_HEAD
    if bem is not None and not bem['is_sphere']:
        idx = np.where(np.array([s['id'] for s in bem['surfs']]) ==
                       FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
        assert len(idx) == 1
        # make a copy so it isn't mangled in use
        bem_surf = transform_surface_to(bem['surfs'][idx[0]], coord_frame,
                                        mri_head_t, copy=True)
    for ti, dev_head_t in enumerate(dev_head_ts):
        # Could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing gain matrix for transform #%s/%s'
                    % (ti + 1, len(dev_head_ts)))
        _transform_orig_meg_coils(megcoils, dev_head_t)
        _transform_orig_meg_coils(compcoils, dev_head_t)

        # Make sure our sensors are all outside our BEM
        coil_rr = np.array([coil['r0'] for coil in megcoils])

        # Compute forward
        if forward is None:
            if not bem['is_sphere']:
                outside = ~_CheckInside(bem_surf)(coil_rr, n_jobs,
                                                  verbose=False)
            elif bem.radius is not None:
                d = coil_rr - bem['r0']
                outside = np.sqrt(np.sum(d * d, axis=1)) > bem.radius
            else:  # only r0 provided
                outside = np.ones(len(coil_rr), bool)
            if not outside.all():
                raise RuntimeError('%s MEG sensors collided with inner skull '
                                   'surface for transform %s'
                                   % (np.sum(~outside), ti))
            megfwd = _compute_forwards(rr, bem, [megcoils], [compcoils],
                                       [meg_info], ['meg'], n_jobs,
                                       verbose=False)[0]
            megfwd = _to_forward_dict(megfwd, megnames)
        else:
            megfwd = pick_channels_forward(forward, megnames, verbose=False)
        fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)
        fwd.update(**update_kwargs)

        yield fwd
    # need an extra one to fill last buffer
    yield fwd
