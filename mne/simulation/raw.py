# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np

from .evoked import _generate_noise
from ..event import _get_stim_channel
from ..filter import _Interp2
from ..io.pick import pick_types, pick_info, pick_channels
from ..source_estimate import VolSourceEstimate
from ..cov import make_ad_hoc_cov, read_cov, Covariance
from ..bem import fit_sphere_to_headshape, make_sphere_model, read_bem_solution
from ..io import RawArray, BaseRaw
from ..chpi import (read_head_pos, head_pos_to_trans_rot_t, _get_hpi_info,
                    _get_hpi_initial_fit)
from ..io.constants import FIFF
from ..fixes import einsum
from ..forward import (_magnetic_dipole_field_vec, _merge_meg_eeg_fwds,
                       _stc_src_sel, convert_forward_solution,
                       _prepare_for_forward, _transform_orig_meg_coils,
                       _compute_forwards, _to_forward_dict)
from ..transforms import _get_trans, transform_surface_to
from ..source_space import (_ensure_src, _points_outside_surface,
                            _adjust_patch_info)
from ..source_estimate import _BaseSourceEstimate
from ..utils import logger, verbose, check_random_state, warn, _pl
from ..parallel import check_n_jobs
from ..externals.six import string_types


def _check_cov(info, cov):
    """Check that the user provided a valid covariance matrix for the noise."""
    if isinstance(cov, Covariance) or cov is None:
        pass
    elif isinstance(cov, dict):
        cov = make_ad_hoc_cov(info, cov, verbose=False)
    elif isinstance(cov, string_types):
        if cov == 'simple':
            cov = make_ad_hoc_cov(info, None, verbose=False)
        else:
            cov = read_cov(cov, verbose=False)
    else:
        raise TypeError('Covariance matrix type not recognized. Valid input '
                        'types are: instance of Covariance, dict, str, None. '
                        ', got %s' % (cov,))
    return cov


def _log_ch(start, info, ch):
    """Log channel information."""
    if ch is not None:
        extra, just, ch = ' stored on channel:', 50, info['ch_names'][ch]
    else:
        extra, just, ch = ' not stored', 0, ''
    logger.info((start + extra).ljust(just) + ch)


@verbose
def simulate_raw(raw, stc, trans, src, bem, cov='simple',
                 blink=False, ecg=False, chpi=False, head_pos=None,
                 mindist=1.0, interp='hann', iir_filter=None, n_jobs=1,
                 random_state=None, use_cps=True, verbose=None):
    u"""Simulate raw data.

    Head movements can optionally be simulated using the ``head_pos``
    parameter.

    Parameters
    ----------
    raw : instance of Raw
        The raw template to use for simulation. The ``info``, ``times``,
        and potentially ``first_samp`` properties will be used.
    stc : instance of SourceEstimate
        The source estimate to use to simulate data. Must have the same
        sample rate as the raw data.
    trans : dict | str | None
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option). If trans is None, an identity transform will be used.
    src : str | instance of SourceSpaces
        Source space corresponding to the stc. If string, should be a source
        space filename. Can also be an instance of loaded or generated
        SourceSpaces.
    bem : str | dict
        BEM solution  corresponding to the stc. If string, should be a BEM
        solution filename (e.g., "sample-5120-5120-5120-bem-sol.fif").
    cov : instance of Covariance | str | dict of float | None
        The sensor covariance matrix used to generate noise. If string, should
        be a filename or 'simple'. If 'simple', an ad hoc covariance matrix
        will be generated with default values. If dict, an ad hoc covariance
        matrix will be generated with the values specified by the dict entries.
        If None, no noise will be added.
    blink : bool
        If true, add simulated blink artifacts. See Notes for details.
    ecg : bool
        If true, add simulated ECG artifacts. See Notes for details.
    chpi : bool
        If true, simulate continuous head position indicator information.
        Valid cHPI information must encoded in ``raw.info['hpi_meas']``
        to use this option.
    head_pos : None | str | dict | tuple | array
        Name of the position estimates file. Should be in the format of
        the files produced by maxfilter. If dict, keys should
        be the time points and entries should be 4x4 ``dev_head_t``
        matrices. If None, the original head position (from
        ``info['dev_head_t']``) will be used. If tuple, should have the
        same format as data returned by `head_pos_to_trans_rot_t`.
        If array, should be of the form returned by
        :func:`mne.chpi.read_head_pos`.
    mindist : float
        Minimum distance between sources and the inner skull boundary
        to use during forward calculation.
    interp : str
        Either 'hann' (default), 'linear', or 'zero', the type of
        forward-solution interpolation to use between forward solutions
        at different head positions.
    iir_filter : None | array
        IIR filter coefficients (denominator) e.g. [1, -1, 0.2].
    n_jobs : int
        Number of jobs to use.
    random_state : None | int | np.random.RandomState
        The random generator state used for blink, ECG, and sensor
        noise randomization.
    use_cps : None | bool (default True)
        Whether to use cortical patch statistics to define normal
        orientations. Only used when surf_ori and/or force_fixed are True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : instance of Raw
        The simulated raw file.

    See Also
    --------
    mne.chpi.read_head_pos
    simulate_evoked
    simulate_stc
    simulate_sparse_stc

    Notes
    -----
    Events coded with the position number (starting at 1) will be stored
    in the trigger channel (if available) at times corresponding to t=0
    in the ``stc``.

    The resulting SNR will be determined by the structure of the noise
    covariance, the amplitudes of ``stc``, and the head position(s) provided.

    The blink and ECG artifacts are generated by 1) placing impulses at
    random times of activation, and 2) convolving with activation kernel
    functions. In both cases, the scale-factors of the activation functions
    (and for the resulting EOG and ECG channel traces) were chosen based on
    visual inspection to yield amplitudes generally consistent with those
    seen in experimental data. Noisy versions of the blink and ECG
    activations will be stored in the first EOG and ECG channel in the
    raw file, respectively, if they exist.

    For blink artifacts:

        1. Random activation times are drawn from an inhomogeneous poisson
           process whose blink rate oscillates between 4.5 blinks/minute
           and 17 blinks/minute based on the low (reading) and high (resting)
           blink rates from [1]_.
        2. The activation kernel is a 250 ms Hanning window.
        3. Two activated dipoles are located in the z=0 plane (in head
           coordinates) at ±30 degrees away from the y axis (nasion).
        4. Activations affect MEG and EEG channels.

    For ECG artifacts:

        1. Random inter-beat intervals are drawn from a uniform distribution
           of times corresponding to 40 and 80 beats per minute.
        2. The activation function is the sum of three Hanning windows with
           varying durations and scales to make a more complex waveform.
        3. The activated dipole is located one (estimated) head radius to
           the left (-x) of head center and three head radii below (+z)
           head center; this dipole is oriented in the +x direction.
        4. Activations only affect MEG channels.

    .. versionadded:: 0.10.0

    References
    ----------
    .. [1] Bentivoglio et al. "Analysis of blink rate patterns in normal
           subjects" Movement Disorders, 1997 Nov;12(6):1028-34.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError('raw should be an instance of Raw')
    times, info, first_samp = raw.times, raw.info, raw.first_samp
    raw_verbose = raw.verbose

    # Check for common flag errors and try to override
    if not isinstance(stc, _BaseSourceEstimate):
        raise TypeError('stc must be a SourceEstimate')
    if not np.allclose(info['sfreq'], 1. / stc.tstep):
        raise ValueError('stc and info must have same sample rate')
    if len(stc.times) <= 2:  # to ensure event encoding works
        raise ValueError('stc must have at least three time points')

    stim = False if len(pick_types(info, meg=False, stim=True)) == 0 else True
    n_jobs = check_n_jobs(n_jobs)

    rng = check_random_state(random_state)

    if head_pos is None:  # use pos from info['dev_head_t']
        head_pos = dict()
    if isinstance(head_pos, string_types):  # can be a head pos file
        head_pos = read_head_pos(head_pos)
    if isinstance(head_pos, np.ndarray):  # can be head_pos quats
        head_pos = head_pos_to_trans_rot_t(head_pos)
    if isinstance(head_pos, tuple):  # can be quats converted to trans, rot, t
        transs, rots, ts = head_pos
        ts -= first_samp / info['sfreq']  # MF files need reref
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
    bad = ts > times[-1]
    if bad.any():
        raise RuntimeError('All position times must be <= t_end (%0.1f '
                           'sec), found %s/%s bad values (is this a split '
                           'file?)' % (times[-1], bad.sum(), len(bad)))
    # If it doesn't start at zero, insert one at t=0
    if len(ts) == 0 or ts[0] > 0:
        ts = np.r_[[0.], ts]
        dev_head_ts.insert(0, info['dev_head_t']['trans'])
    dev_head_ts = [{'trans': d, 'to': info['dev_head_t']['to'],
                    'from': info['dev_head_t']['from']}
                   for d in dev_head_ts]
    offsets = raw.time_as_index(ts)
    assert np.array_equal(offsets, np.unique(offsets))
    assert len(offsets) == len(dev_head_ts)
    del ts

    src = _ensure_src(src, verbose=False)
    if isinstance(bem, string_types):
        bem = read_bem_solution(bem, verbose=False)
    cov = _check_cov(info, cov)
    approx_events = int((len(times) / info['sfreq']) /
                        (stc.times[-1] - stc.times[0]))
    logger.info('Provided parameters will provide approximately %s event%s'
                % (approx_events, _pl(approx_events)))

    # Extract necessary info
    meeg_picks = pick_types(info, meg=True, eeg=True, exclude=[])  # for sim
    meg_picks = pick_types(info, meg=True, eeg=False, exclude=[])  # for CHPI
    fwd_info = pick_info(info, meeg_picks)
    fwd_info['projs'] = []  # Ensure no 'projs' applied
    logger.info('Setting up raw simulation: %s position%s, "%s" interpolation'
                % (len(dev_head_ts), _pl(dev_head_ts), interp))

    verts = stc.vertices
    verts = [verts] if isinstance(stc, VolSourceEstimate) else verts
    src = _restrict_source_space_to(src, verts)

    # array used to store result
    raw_data = np.zeros((len(info['ch_names']), len(times)))

    # figure out our cHPI, ECG, and blink dipoles
    ecg_rr = blink_rrs = blink_nns = exg_bem = hpi_rrs = hpi_nns = None
    ecg = ecg and len(meg_picks) > 0
    chpi = chpi and len(meg_picks) > 0
    if chpi:
        hpi_freqs, hpi_pick, hpi_ons = _get_hpi_info(info)
        hpi_rrs = _get_hpi_initial_fit(info, verbose='error')
        hpi_nns = hpi_rrs / np.sqrt(np.sum(hpi_rrs * hpi_rrs,
                                           axis=1))[:, np.newaxis]
        # turn on cHPI in file
        raw_data[hpi_pick, :] = hpi_ons.sum()
        _log_ch('cHPI status bits enbled and', info, hpi_pick)
    if blink or ecg:
        R, r0 = fit_sphere_to_headshape(info, units='m', verbose=False)[:2]
        exg_bem = make_sphere_model(r0, head_radius=R,
                                    relative_radii=(0.97, 0.98, 0.99, 1.),
                                    sigmas=(0.33, 1.0, 0.004, 0.33),
                                    verbose=False)
    if blink:
        # place dipoles at 45 degree angles in z=0 plane
        blink_rrs = np.array([[np.cos(np.pi / 3.), np.sin(np.pi / 3.), 0.],
                              [-np.cos(np.pi / 3.), np.sin(np.pi / 3), 0.]])
        blink_rrs /= np.sqrt(np.sum(blink_rrs *
                                    blink_rrs, axis=1))[:, np.newaxis]
        blink_rrs *= 0.96 * R
        blink_rrs += r0
        # oriented upward
        blink_nns = np.array([[0., 0., 1.], [0., 0., 1.]])
        # Blink times drawn from an inhomogeneous poisson process
        # by 1) creating the rate and 2) pulling random numbers
        blink_rate = (1 + np.cos(2 * np.pi * 1. / 60. * times)) / 2.
        blink_rate *= 12.5 / 60.
        blink_rate += 4.5 / 60.
        blink_data = rng.rand(len(times)) < blink_rate / info['sfreq']
        blink_data = blink_data * (rng.rand(len(times)) + 0.5)  # varying amps
        # Activation kernel is a simple hanning window
        blink_kernel = np.hanning(int(0.25 * info['sfreq']))
        blink_data = np.convolve(blink_data, blink_kernel,
                                 'same')[np.newaxis, :] * 1e-7
        # Add rescaled noisy data to EOG ch
        ch = pick_types(info, meg=False, eeg=False, eog=True)
        noise = rng.randn(blink_data.shape[1]) * 5e-6
        if len(ch) >= 1:
            ch = ch[-1]
            raw_data[ch, :] = blink_data * 1e3 + noise
        else:
            ch = None
        _log_ch('Blinks simulated and trace', info, ch)
        del blink_kernel, blink_rate, noise
    if ecg:
        ecg_rr = np.array([[-R, 0, -3 * R]])
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
        ecg_data = np.convolve(cardiac_data, cardiac_kernel,
                               'same')[np.newaxis, :] * 15e-8
        # Add rescaled noisy data to ECG ch
        ch = pick_types(info, meg=False, eeg=False, ecg=True)
        noise = rng.randn(ecg_data.shape[1]) * 1.5e-5
        if len(ch) >= 1:
            ch = ch[-1]
            raw_data[ch, :] = ecg_data * 2e3 + noise
        else:
            ch = None
        _log_ch('ECG simulated and trace', info, ch)
        del cardiac_data, cardiac_kernel, max_beats, cardiac_idx

    stc_event_idx = np.argmin(np.abs(stc.times))
    if stim:
        event_ch = pick_channels(info['ch_names'],
                                 _get_stim_channel(None, info))[0]
        raw_data[event_ch, :] = 0.
    else:
        event_ch = None
    _log_ch('Event information', info, event_ch)
    stc_indices = np.arange(len(times)) % len(stc.times)
    raw_data[meeg_picks, :] = 0.
    if chpi:
        sinusoids = 70e-9 * np.sin(2 * np.pi * hpi_freqs[:, np.newaxis] *
                                   (np.arange(len(times)) / info['sfreq']))
    zf = None  # final filter conditions for the noise
    bounds = list(range(0, len(raw.times), 10000)) + [len(raw.times)]
    get_fwd = _SimForwards(dev_head_ts, offsets, info, trans, src, bem,
                           exg_bem, mindist, hpi_rrs, hpi_nns,
                           blink_rrs, blink_nns, ecg_rr, use_cps, n_jobs)
    interper = _Interp2(offsets, get_fwd, interp=interp)
    first = True
    for start, stop in zip(bounds[:-1], bounds[1:]):
        event_idxs = np.where(stc_indices[start:stop] == stc_event_idx)[0]
        logger.info('  Simulating data for %0.3f-%0.3f sec with %s event%s'
                    % (start / info['sfreq'], stop / info['sfreq'],
                       len(event_idxs), _pl(event_idxs)))
        fwd, fwd_blink, fwd_ecg, fwd_chpi, fi = interper.feed(stop - start)
        fi = np.ceil(fi)
        if first:
            src_sel = _stc_src_sel(get_fwd.update_kwargs['src'], stc)
            verts = stc.vertices
            verts = [verts] if isinstance(stc, VolSourceEstimate) else verts
            diff_ = sum([len(v) for v in verts]) - len(src_sel)
            if diff_ != 0:
                warn('%s STC vertices omitted due to fwd calculation' % diff_)
            stc_data = stc.data[src_sel]
            del stc, src_sel, diff_, verts
            first = False

        if stim:
            raw_data[event_ch, event_idxs + start] = fi[event_idxs] + 1
        time_sl = slice(start, stop)
        stc_idxs = stc_indices[time_sl]
        # simulate brain data
        this_data = raw_data[:, time_sl]
        this_data[meeg_picks] = 0.
        this_data[meeg_picks] += einsum(
            'svt,vt->st', fwd, stc_data[:, stc_idxs])
        if blink:
            this_data[meeg_picks] += einsum(
                'svt,vt->st', fwd_blink, blink_data[:, time_sl])
        if ecg:
            this_data[meg_picks] += einsum(
                'svt,vt->st', fwd_ecg, ecg_data[:, time_sl])
        if chpi:
            this_data[meg_picks] += einsum(
                'svt,vt->st', fwd_chpi, sinusoids[:, time_sl])
        # add sensor noise, ECG, blink, cHPI
        if cov is not None:
            noise, zf = _generate_noise(fwd_info, cov, iir_filter, rng,
                                        len(stc_idxs), zi=zf)
            this_data[meeg_picks, time_sl] += noise
    raw = RawArray(raw_data, info, first_samp=first_samp, verbose=False)
    raw.verbose = raw_verbose
    logger.info('Done')
    return raw


class _SimForwards(object):
    """Simulate forwards."""

    def __init__(self, dev_head_ts, offsets, info, trans, src, bem, exg_bem,
                 mindist, hpi_rrs, hpi_nns, blink_rrs, blink_nns, ecg_rr,
                 use_cps, n_jobs):
        self.dev_head_ts = dev_head_ts
        self.offsets = offsets
        mri_head_t, trans = _get_trans(trans)
        self.bem = bem
        self.exg_bem = exg_bem
        self.hpi_rrs = hpi_rrs
        self.hpi_nns = hpi_nns
        self.blink_rrs = blink_rrs
        self.blink_nns = blink_nns
        self.ecg_rr = ecg_rr
        self.use_cps = use_cps
        self.n_jobs = n_jobs
        logger.info('Setting up forward solutions')
        self.megcoils, self.meg_info, self.compcoils, self.megnames, \
            eegels, eegnames, self.rr, self.info, \
            self.update_kwargs, self.bem = _prepare_for_forward(
                src, mri_head_t, info, bem, mindist, self.n_jobs,
                verbose=False)
        if len(eegels) > 0:
            logger.info('Computing EEG gain matrix')
        self.eegfwd = _compute_forwards(
            self.rr, self.bem, [eegels], [None], [None], ['eeg'],
            self.n_jobs, verbose=False)[0]
        self.eegfwd = _to_forward_dict(self.eegfwd, eegnames)
        if self.blink_rrs is not None:
            eegblink = _compute_forwards(
                self.blink_rrs, self.exg_bem, [eegels], [None], [None],
                ['eeg'], self.n_jobs, verbose=False)[0]
            self.eegblink = _to_forward_dict(eegblink, eegnames)
        else:
            self.eegblink = None
        if len(pick_types(self.info, meg=True)) == 0:
            self.has_meg = False
            self.eegfwd.update(**self.update_kwargs)
            self.eegfwd = convert_forward_solution(
                self.eegfwd, surf_ori=True, force_fixed=True,
                use_cps=self.use_cps, verbose=False)['sol']['data']
            self.eegblink = self._blink_fwd_to_array(self.eegblink)
        else:
            self.has_meg = True
            coord_frame = FIFF.FIFFV_COORD_HEAD
            if not bem['is_sphere']:
                idx = np.where(np.array([s['id'] for s in bem['surfs']]) ==
                               FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
                assert len(idx) == 1
                # make a copy so it isn't mangled in use
                self.bem_surf = transform_surface_to(
                    bem['surfs'][idx[0]], coord_frame, mri_head_t, copy=True)

    def __call__(self, offset):
        """Calculate a forward solution for a subject."""
        ti = np.where(self.offsets == offset)[0][0]
        # short circuit here if there are no MEG channels
        if not self.has_meg:
            return self.eegfwd, self.eegblink, None, None, ti

        logger.info('Setting up forward solutions')
        # Could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing MEG gain matrix for transform #%s/%s'
                    % (ti + 1, len(self.dev_head_ts)))
        dev_head_t = self.dev_head_ts[ti]
        _transform_orig_meg_coils(self.megcoils, dev_head_t)
        _transform_orig_meg_coils(self.compcoils, dev_head_t)

        # Make sure our sensors are all outside our BEM
        coil_rr = [coil['r0'] for coil in self.megcoils]
        if not self.bem['is_sphere']:
            outside = _points_outside_surface(
                coil_rr, self.bem_surf, self.n_jobs, verbose=False)
        else:
            d = coil_rr - self.bem['r0']
            outside = np.sqrt(np.sum(d * d, axis=1)) > self.bem.radius
        if not outside.all():
            raise RuntimeError('%s MEG sensors collided with inner skull '
                               'surface for transform %s'
                               % (np.sum(~outside), ti))

        # Compute forward
        megfwd = _compute_forwards(
            self.rr, self.bem, [self.megcoils], [self.compcoils],
            [self.meg_info], ['meg'], self.n_jobs, verbose=False)[0]
        megfwd = _to_forward_dict(megfwd, self.megnames)
        fwd = _merge_meg_eeg_fwds(megfwd, self.eegfwd, verbose=False)
        fwd.update(**self.update_kwargs)
        # must be fixed orientation
        # XXX eventually we could speed this up by allowing the forward
        # solution code to only compute the normal direction

        fwd_blink = fwd_ecg = fwd_chpi = None
        if self.blink_rrs is not None:
            megblink = _compute_forwards(
                self.blink_rrs, self.exg_bem, [self.megcoils],
                [self.compcoils], [self.meg_info], ['meg'], self.n_jobs,
                verbose=False)[0]
            megblink = _to_forward_dict(megblink, self.megnames)
            fwd_blink = _merge_meg_eeg_fwds(megblink, self.eegblink,
                                            verbose=False)
            fwd_blink = self._blink_fwd_to_array(fwd_blink)
        if self.ecg_rr is not None:
            megecg = _compute_forwards(
                self.ecg_rr, self.exg_bem, [self.megcoils], [self.compcoils],
                [self.meg_info], ['meg'], self.n_jobs, verbose=False)[0]
            fwd_ecg = _to_forward_dict(megecg, self.megnames)
            # just use one arbitrary direction
            fwd_ecg = fwd_ecg['sol']['data'][:, [0]]
        if self.hpi_rrs is not None:
            fwd_chpi = _magnetic_dipole_field_vec(
                self.hpi_rrs, self.megcoils).T
            # align cHPI magnetic dipoles in approx. radial direction
            for ii in range(len(self.hpi_rrs)):
                fwd_chpi[:, ii] = np.dot(fwd_chpi[:, 3 * ii:3 * (ii + 1)],
                                         self.hpi_nns[ii])
            fwd_chpi = fwd_chpi[:, :len(self.hpi_rrs)].copy()
        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                       use_cps=self.use_cps, verbose=False)
        fwd = fwd['sol']['data']
        return fwd, fwd_blink, fwd_ecg, fwd_chpi, ti

    def _blink_fwd_to_array(self, fwd_blink):
        fwd_blink = fwd_blink['sol']['data']
        for ii in range(len(self.blink_rrs)):
            fwd_blink[:, ii] = np.dot(fwd_blink[:, 3 * ii:3 * (ii + 1)],
                                      self.blink_nns[ii])
        fwd_blink = fwd_blink[:, :len(self.blink_rrs)]
        return fwd_blink.sum(axis=1)[:, np.newaxis]


def _restrict_source_space_to(src, vertices):
    """Trim down a source space."""
    assert len(src) == len(vertices)
    src = deepcopy(src)
    for s, v in zip(src, vertices):
        s['inuse'].fill(0)
        s['nuse'] = len(v)
        s['vertno'] = v
        s['inuse'][s['vertno']] = 1
        for key in ('nuse_tri', 'use_tris'):
            if key in s:
                del s[key]
        # This will fix 'patch_info' and 'pinfo'
        _adjust_patch_info(s, verbose=False)
    return src
