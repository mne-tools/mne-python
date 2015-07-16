# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <>
#
# License: BSD (3-clause)

import os
from os import path as op
import numpy as np
import warnings
from copy import deepcopy

from mne import (pick_types, pick_info, pick_channels, VolSourceEstimate,
                 convert_forward_solution, get_chpi_positions, EvokedArray,
                 make_ad_hoc_cov)
from mne.bem import fit_sphere_to_headshape, make_sphere_model
from mne.io import read_info, Raw
from mne.externals.six import string_types
from mne.forward.forward import _merge_meg_eeg_fwds, _stc_src_sel
from mne.forward._make_forward import (_prep_channels, _setup_bem,
                                       _compute_forwards, _to_forward_dict)
from mne.forward._compute_forward import _magnetic_dipole_field_vec
from mne.transforms import _get_mri_head_t, transform_surface_to
from mne.source_space import (SourceSpaces, read_source_spaces,
                              _filter_source_spaces, _points_outside_surface)
from mne.bem import _bem_find_surface
from mne.io.constants import FIFF
from mne.io.meas_info import Info
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import logger, verbose, check_random_state
from mne.simulation import generate_noise_evoked


@verbose
def simulate_raw(raw, stc, trans, src, bem, cov='simple', pos=None, ecg=False,
                 blink=False, chpi=False, mindist=1.0, interp='linear',
                 random_state=None, n_jobs=1, verbose=None):
    """Simulate raw data with head movements

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to use. The measurement info, including the
        head positions, will be used to simulate data.
    stc : instance of SourceEstimate
        The source estimate to use to simulate data. Must have the same
        sample rate as the raw data.
    trans : dict | str
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option).
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : str
        Filename of BEM solution (e.g., "sample-5120-5120-5120-bem-sol.fif").
    cov : instance of Covariance | 'simple' | None
        The sensor covariance matrix used to generate noise. If None,
        no noise will be added. If 'simple', a basic (diagonal) ad-hoc
        noise covariance will be used.
    blink : bool
        Add simulated blink artifacts
    ecg : bool
        Add simulated ECG artifacts
    pos : None | str | dict
        Name of the position estimates file. Should be in the format of
        the files produced by maxfilter-produced. If dict, keys should
        be the time points and entries should be 4x3 ``dev_head_t``
        matrices. If None, the original head position (from
        ``raw.info['dev_head_t']``) will be used.
    chpi : bool
        If true, use continuous head position indicator information.
    mindist : float
        Minimum distance between sources and the inner skull boundary
        to use during forward calculation.
    interp : str
        Either 'linear' or 'zero', the type of forward-solution
        interpolation to use between provided time points.
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
    n_jobs : int
        Number of jobs to use.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of Raw
        The simulated raw file.

    Notes
    -----
    Events coded with the number of the forward solution used will be placed
    in the raw files in the trigger channel STI101 at the t=0 times of the
    SourceEstimates.

    The resulting SNR will be determined by the structure of the noise
    covariance, and the amplitudes of the SourceEstimate. Note that this
    will vary as a function of position.
    """
    # Check for common flag errors and override
    if isinstance(raw, string_types):
        with warnings.catch_warnings(record=True):
            raw = Raw(raw, allow_maxshield=True, preload=True, verbose=False)
    else:
        raw = raw.copy()

    if not isinstance(stc, _BaseSourceEstimate):
        raise TypeError('stc must be a SourceEstimate')
    if not np.allclose(raw.info['sfreq'], 1. / stc.tstep):
        raise ValueError('stc and raw must have same sample rate')
    # Only use cHPI if custom frequency is in HPI information
    if chpi and np.all(['custom_ref' in x.keys()
                        for x in raw.info['hpi_meas'][0]['hpi_coils']]):
        raise ValueError("`custom_ref` must be in "
                         "raw.info['hpi_meas'][0]['hpi_coils'] to use cHPI")
    rng = check_random_state(random_state)
    if interp not in ('linear', 'zero'):
        raise ValueError('interp must be "linear" or "zero"')

    # Use stationary head
    if pos is None:
        dev_head_ts = [raw.info['dev_head_t']] * 2
        offsets = np.array([0, raw.n_times])
        interp = 'zero'
    # Use pos data to simulate head movement
    else:
        if isinstance(pos, string_types):
            pos = get_chpi_positions(pos, verbose=False)
        if isinstance(pos, tuple):  # can be an already-loaded pos file
            transs, rots, ts = pos
            ts -= raw.first_samp / raw.info['sfreq']  # MF files need reref
            dev_head_ts = [np.r_[np.c_[r, t[:, np.newaxis]], [[0, 0, 0, 1]]]
                           for r, t in zip(rots, transs)]
            del transs, rots
        elif isinstance(pos, dict):
            ts = np.array(list(pos.keys()), float)
            ts.sort()
            dev_head_ts = [pos[float(tt)] for tt in ts]
        else:
            raise TypeError('unknown pos type %s' % type(pos))
        if not (ts >= 0).all():  # pathological if not
            raise RuntimeError('Cannot have t < 0 in transform file')
        tend = raw.times[-1]
        assert not (ts < 0).any()
        assert not (ts > tend).any()
        if ts[0] > 0:
            ts = np.r_[[0.], ts]
            dev_head_ts.insert(0, raw.info['dev_head_t']['trans'])
        dev_head_ts = [{'trans': d, 'to': raw.info['dev_head_t']['to'],
                        'from': raw.info['dev_head_t']['from']}
                       for d in dev_head_ts]
        if ts[-1] < tend:
            dev_head_ts.append(dev_head_ts[-1])
            ts = np.r_[ts, [tend]]
        offsets = raw.time_as_index(ts)
        offsets[-1] = raw.n_times  # fix for roundoff error
        assert offsets[-2] != offsets[-1]
        del ts

    if isinstance(cov, string_types):
        assert cov == 'simple'
        cov = make_ad_hoc_cov(raw.info, verbose=False)
    assert np.array_equal(offsets, np.unique(offsets))
    assert len(offsets) == len(dev_head_ts)
    approx_events = int((raw.n_times / raw.info['sfreq']) /
                        (stc.times[-1] - stc.times[0]))
    logger.info('Provided parameters will provide approximately %s event%s'
                % (approx_events, '' if approx_events == 1 else 's'))

    # Get chpi freqs and reorder
    if chpi:
        hpi_freqs = np.array([x['custom_ref'][0]
                            for x in raw.info['hpi_meas'][0]['hpi_coils']])
        n_freqs = len(hpi_freqs)
        order = [x['number'] - 1 for x in raw.info['hpi_meas'][0]['hpi_coils']]
        assert np.array_equal(np.unique(order), np.arange(n_freqs))
        hpi_freqs = hpi_freqs[order]
        hpi_order = raw.info['hpi_results'][0]['order'] - 1
        assert np.array_equal(np.unique(hpi_order), np.arange(n_freqs))
        hpi_freqs = hpi_freqs[hpi_order]


    # Extract necessary info
    picks = pick_types(raw.info, meg=True, eeg=True)  # for simulation
    meg_picks = pick_types(raw.info, meg=True, eeg=False)  # for CHPI
    fwd_info = pick_info(raw.info, picks)
    fwd_info['projs'] = []  # Ensure no 'projs' applied
    logger.info('Setting up raw data simulation using %s head position%s'
                % (len(dev_head_ts), 's' if len(dev_head_ts) != 1 else ''))
    raw.preload_data(verbose=False)

    if isinstance(stc, VolSourceEstimate):
        verts = [stc.vertices]
    else:
        verts = stc.vertices
    src = _restrict_source_space_to(src, verts)

    blink_bem = blink_rr = ecg_rr = chpi_rrs = None

    # Simulate Blink, ECG, and head-movement artifacts
    # Oscillate blink artifact between resting (17 bpm) and reading (4.5 bpm)
    # http://www.ncbi.nlm.nih.gov/pubmed/9399231

    # Create blink_bem (used for eog and blink artifact) if necessary
    if ecg or blink:

        # Figure out our cHPI, ECG, and Blink dipoles
        dig = raw.info['dig']
        assert all([d['coord_frame'] == FIFF.FIFFV_COORD_HEAD
                    for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI])
        chpi_rrs = [d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI]
        R, r0 = fit_sphere_to_headshape(raw.info, verbose=False)[:2]
        R /= 1000.
        r0 /= 1000.

        blink_rr = [d['r'] for d in raw.info['dig']
                if d['ident'] == FIFF.FIFFV_POINT_NASION][0]
        blink_rr = blink_rr - r0
        blink_rr = (blink_rr / np.sqrt(np.sum(blink_rr * blink_rr)) *
                0.98 * R)[np.newaxis, :]
        blink_rr += r0
        blink_bem = make_sphere_model(
            r0, head_radius=R, relative_radii=(0.99, 1.), sigmas=(0.33, 0.33),
            verbose=False)

    if blink:
        blink_rate = np.cos(2 * np.pi * 1. / 60. * raw.times)
        blink_rate *= 12.5 / 60.
        blink_rate += 4.5 / 60.
        blink_times = rng.rand(raw.n_times) < blink_rate / raw.info['sfreq']
        # vary amplitudes
        blink_times = blink_times * (rng.rand(raw.n_times) + 0.5)
        # Convolve blink times with kernel to get blink traces
        blink_kernel = np.hanning(int(0.25 * raw.info['sfreq']))
        blink_data = np.convolve(blink_times, blink_kernel,
                                 'same')[np.newaxis, :]
        blink_data += rng.randn(blink_data.shape[1]) * 0.05
        blink_data *= 100e-6  # Scale to blink EMG scale
        del blink_times,

    if ecg:
        ecg_rr = np.array([[-R, 0, -3 * R]])

        max_beats = int(np.ceil(raw.times[-1] * 70. / 60.))
        cardiac_idx = np.cumsum(rng.uniform(60. / 70., 60. / 50., max_beats) *
                                raw.info['sfreq']).astype(int)
        cardiac_idx = cardiac_idx[cardiac_idx < raw.n_times]
        cardiac_data = np.zeros(raw.n_times)
        cardiac_data[cardiac_idx] = 1
        cardiac_kernel = np.concatenate([
            2 * np.hanning(int(0.04 * raw.info['sfreq'])),
            -0.3 * np.hanning(int(0.05 * raw.info['sfreq'])),
            0.2 * np.hanning(int(0.26 * raw.info['sfreq']))], axis=-1)
        ecg_data = np.convolve(cardiac_data, cardiac_kernel,
                               'same')[np.newaxis, :]
        ecg_data += rng.randn(ecg_data.shape[1]) * 0.05
        ecg_data *= 3e-4
        del cardiac_data

    # TODO: This is just scaling to correct physiological scale, correct?
    #       Can we just roll the scaling into above funcs if statements
    #       and handle the data assignment there?
    #       We should also check that the correct channels exist

    # Add to data file, then rescale for simulation
    '''
    for data, scale, exg_ch in zip([blink_data, ecg_data],
                                   [1e-3, 5e-4],
                                   ['EOG062', 'ECG063']):
        ch = pick_channels(raw.ch_names, [exg_ch])
        if len(ch) == 1:
            raw._data[ch[0], :] = data
        data *= scale
    '''

    evoked = EvokedArray(np.zeros((len(picks), len(stc.times))), fwd_info,
                         stc.tmin, verbose=False)
    stc_event_idx = np.argmin(np.abs(stc.times))
    event_chs = pick_channels(raw.info['ch_names'], ['STI101'])
    if len(event_chs) == 0:
        # When functionality is complete, expand raw object with this channel
        #raise RuntimeError('No `STI` channels found.')
        event_ch = 0
    else:
        event_ch = pick_channels(raw.info['ch_names'], ['STI101'])[0]

    used = np.zeros(raw.n_times, bool)
    stc_indices = np.arange(raw.n_times) % len(stc.times)
    raw._data[event_ch, ].fill(0)
    hpi_mag = 25e-9
    last_fwd = last_fwd_chpi = last_fwd_blink = last_fwd_ecg = src_sel = None

    # Simulate raw data in sets of event
    for fi, (fwd, fwd_blink, fwd_ecg, fwd_chpi) in enumerate(
        _make_forward_solutions(fwd_info, trans, src, bem, mindist,
                                dev_head_ts, n_jobs, blink_bem, ecg_rr,
                                blink_rr, chpi_rrs)):

        # Must be fixed orientation
        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                       verbose=False)

        # Just use one arbitrary direction for each
        if fwd_ecg is not None:
            fwd_ecg = fwd_ecg['sol']['data'][:, ::3]
        if fwd_blink is not None:
            fwd_blink = fwd_blink['sol']['data'][:, ::3]
        if fwd_chpi is not None:
            fwd_chpi = fwd_chpi[:, ::3]

        if src_sel is None:
            src_sel = _stc_src_sel(fwd['src'], stc)
            if isinstance(stc, VolSourceEstimate):
                verts = [stc.vertices]
            else:
                verts = stc.vertices
            diff_ = sum([len(v) for v in verts]) - len(src_sel)
            if diff_ != 0:
                warnings.warn('%s STC vertices omitted due to fwd calculation'
                              % (diff_,))

        # Skip to next iteration if this is first pass in loop
        if last_fwd is None:
            last_fwd, last_fwd_blink, last_fwd_ecg, last_fwd_chpi = \
                fwd, fwd_blink, fwd_ecg, fwd_chpi
            continue

        n_time = offsets[fi] - offsets[fi-1]

        time_slice = slice(offsets[fi-1], offsets[fi])
        assert not used[time_slice].any()
        stc_idxs = stc_indices[time_slice]
        event_idxs = np.where(stc_idxs == stc_event_idx)[0] + offsets[fi-1]
        used[time_slice] = True
        logger.info('  Simulating data for %0.3f-%0.3f sec with %s event%s'
                    % (tuple(offsets[fi-1:fi+1] / raw.info['sfreq']) +
                       (len(event_idxs), '' if len(event_idxs) == 1 else 's')))

        # Simulate brain data
        stc_data = stc.data[:, stc_idxs][src_sel]
        data = _interp(last_fwd['sol']['data'], fwd['sol']['data'], stc_data,
                       interp)
        simulated = EvokedArray(data, evoked.info, 0)
        if cov is not None:
            noise = generate_noise_evoked(simulated, cov, [1, -1, 0.2], rng)
            simulated.data += noise.data
        assert simulated.data.shape[0] == len(picks)
        assert simulated.data.shape[1] == len(stc_idxs)
        raw._data[picks, time_slice] = simulated.data

        # Add ECG, Blink, and CHPI traces
        # TODO: Should conditionals depend on fwds or param bool flags?
        if ecg:
            raw._data[meg_picks, time_slice] += \
                _interp(last_fwd_ecg, fwd_ecg, ecg_data[:, time_slice], interp)
            last_fwd_ecg = fwd_ecg

        if blink:
            raw._data[picks, time_slice] += \
                _interp(last_fwd_blink, fwd_blink, blink_data[:, time_slice],
                        interp)
            last_fwd_blink = fwd_blink

        if chpi:
            this_t = np.arange(offsets[fi-1], offsets[fi]) / raw.info['sfreq']
            sinusoids = np.zeros((n_freqs, n_time))
            for fi, freq in enumerate(hpi_freqs):
                sinusoids[fi] = 2 * np.pi * freq * this_t
                sinusoids[fi] = hpi_mag * np.sin(sinusoids[fi])
            raw._data[meg_picks, time_slice] += \
                _interp(last_fwd_chpi, fwd_chpi, sinusoids, interp)
            last_fwd_chpi = fwd_chpi

        # Add events
        raw._data[event_ch, event_idxs] = fi

        # Prepare for next iteration
        last_fwd, last_fwd_blink, last_fwd_ecg, last_fwd_chpi = \
            fwd, fwd_blink, fwd_ecg, fwd_chpi

    assert used.all()
    logger.info('Done')
    return raw


def _make_forward_solutions(info, mri, src, bem, mindist, dev_head_ts,
                            n_jobs=1, bem_blink=None, ecg_rrs=None,
                            blink_rrs=None, chpi_rrs=None):
    """Calculate a forward solution for a subject

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    mri : dict | str
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option).
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif") to
        use.
    mindist : float
        Minimum distance of sources from inner skull surface (in mm).
    dev_head_ts : list
        List of device<->head transforms.
    n_jobs : int
        Number of jobs to run in parallel.
    bem_blink : dict | None
        Spherical BEM to use for blink (and ECG) simulation.
    chpi_rrs : ndarray | None
        CHPI dipoles to simulate (magnetic dipoles).
    blink_rrs : ndarray | None
        Blink dipoles to simulate.
    ecg_rrs : ndarray | None
        ECG dipoles to simulate.

    Returns
    -------
    fwd : generator
        A generator for each forward solution in dev_head_ts.

    Notes
    -----
    Some of the forward solution calculation options from the C code
    (e.g., `--grad`, `--fixed`) are not implemented here. For those,
    consider using the C command line tools or the Python wrapper
    `do_forward_solution`.
    """

    mri_head_t, mri = _get_mri_head_t(mri)
    assert mri_head_t['from'] == FIFF.FIFFV_COORD_MRI

    if not isinstance(src, string_types):
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or SourceSpaces')
    else:
        if not op.isfile(src):
            raise IOError('Source space file "%s" not found' % src)
    if isinstance(bem, dict):
        bem_extra = 'dict'
    else:
        bem_extra = bem
        if not op.isfile(bem):
            raise IOError('BEM file "%s" not found' % bem)
    if not isinstance(info, (dict, string_types)):
        raise TypeError('info should be a dict or string')
    if isinstance(info, string_types):
        info = read_info(info, verbose=False)

    # set default forward solution coordinate frame to HEAD
    # this could, in principle, be an option
    coord_frame = FIFF.FIFFV_COORD_HEAD

    # Report the setup
    logger.info('Setting up forward solutions')

    # Read the source locations
    if isinstance(src, string_types):
        src = read_source_spaces(src, verbose=False)
    else:
        # let's make a copy in case we modify something
        src = src.copy()
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')
    logger.info('Read %d source spaces a total of %d active source locations'
                % (len(src), nsource))

    # Make a new dict with the relevant information
    mri_id = dict(machid=np.zeros(2, np.int32), version=0, secs=0, usecs=0)
    info = dict(nchan=info['nchan'], chs=info['chs'], comps=info['comps'],
                ch_names=info['ch_names'],
                mri_file='', mri_id=mri_id, meas_file='',
                meas_id=None, working_dir=os.getcwd(),
                command_line='', bads=info['bads'])
    info = Info(info)

    # Only get the EEG channels here b/c we can do MEG later
    _, _, eegels, _, eegnames, _ = \
        _prep_channels(info, False, True, True, verbose=False)

    # Transform source spaces into the appropriate coordinates (HEAD or MRI)
    for s in src:
        transform_surface_to(s, coord_frame, mri_head_t)

    # Prepare the BEM model
    bem = _setup_bem(bem, bem_extra, len(eegnames), mri_head_t, verbose=False)

    # Circumvent numerical problems by excluding points too close to the skull
    if not bem['is_sphere']:
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        _filter_source_spaces(inner_skull, mindist, mri_head_t, src, n_jobs,
                              verbose=False)

    # Time to do the heavy lifting: EEG first, then MEG
    rr = np.concatenate([s['rr'][s['vertno']] for s in src])
    eegfwd = _compute_forwards(rr, bem, [eegels], [None],
                               [None], ['eeg'], n_jobs, verbose=False)[0]
    eegfwd = _to_forward_dict(eegfwd, None, eegnames, coord_frame,
                              FIFF.FIFFV_MNE_FREE_ORI)
    if blink_rrs is not None:
        eegblink = _compute_forwards(blink_rrs, bem_blink, [eegels], [None],
                                [None], ['eeg'], n_jobs, verbose=False)[0]
        eegblink = _to_forward_dict(eegblink, None, eegnames, coord_frame,
                                FIFF.FIFFV_MNE_FREE_ORI)

    for ti, dev_head_t in enumerate(dev_head_ts):
        # Could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing gain matrix for transform #%s/%s'
                    % (ti + 1, len(dev_head_ts)))
        info = deepcopy(info)
        info['dev_head_t'] = dev_head_t
        megcoils, compcoils, _, megnames, _, meg_info = \
            _prep_channels(info, True, False, False, verbose=False)

        # Make sure our sensors are all outside our BEM
        coil_rr = [coil['r0'] for coil in megcoils]
        if not bem['is_sphere']:
            idx = np.where(np.array([s['id'] for s in bem['surfs']]) ==
                           FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
            assert len(idx) == 1
            bem_surf = transform_surface_to(bem['surfs'][idx[0]], coord_frame,
                                            mri_head_t)
            outside = _points_outside_surface(coil_rr, bem_surf, n_jobs,
                                              verbose=False)
        else:
            rad = bem['layers'][-1]['rad']
            outside = np.sqrt(np.sum((coil_rr - bem['r0']) ** 2)) >= rad
        if not np.all(outside):
            raise RuntimeError('MEG sensors collided with inner skull '
                               'surface for transform %s' % ti)

        # Compute forward
        megfwd = _compute_forwards(rr, bem, [megcoils], [compcoils],
                                   [meg_info], ['meg'], n_jobs,
                                   verbose=False)[0]
        megfwd = _to_forward_dict(megfwd, None, megnames, coord_frame,
                                  FIFF.FIFFV_MNE_FREE_ORI)
        fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)

        # pick out final dict info
        nsource = fwd['sol']['data'].shape[1] // 3
        source_nn = np.tile(np.eye(3), (nsource, 1))
        fwd.update(dict(nchan=fwd['sol']['data'].shape[0], nsource=nsource,
                        info=info, src=src, source_nn=source_nn,
                        source_rr=rr, surf_ori=False, mri_head_t=mri_head_t))
        fwd['info']['mri_head_t'] = mri_head_t
        fwd['info']['dev_head_t'] = dev_head_t

        # Compute forward solutions for each type of artifact
        fwd_blink = fwd_ecg = fwd_chpi = None
        if blink_rrs is not None:
            megblink = _compute_forwards(blink_rrs, bem_blink, [megcoils],
                                         [compcoils], [meg_info], ['meg'],
                                         n_jobs, verbose=False)[0]
            megblink = _to_forward_dict(megblink, None, megnames, coord_frame,
                                    FIFF.FIFFV_MNE_FREE_ORI)
            fwd_blink = _merge_meg_eeg_fwds(megblink, eegblink, verbose=False)

        if ecg_rrs is not None:
            megecg = _compute_forwards(ecg_rrs, bem_blink, [megcoils],
                                       [compcoils], [meg_info], ['meg'],
                                       n_jobs, verbose=False)[0]
            fwd_ecg = _to_forward_dict(megecg, None, megnames, coord_frame,
                                    FIFF.FIFFV_MNE_FREE_ORI)
        if chpi_rrs is not None:
            fwd_chpi = _magnetic_dipole_field_vec(chpi_rrs, megcoils).T

        yield fwd, fwd_blink, fwd_ecg, fwd_chpi


def _restrict_source_space_to(src, vertices):
    """Helper to trim down a source space"""
    assert len(src) == len(vertices)
    src = deepcopy(src)
    for s, v in zip(src, vertices):
        s['inuse'].fill(0)
        s['nuse'] = len(v)
        s['vertno'] = v
        s['inuse'][s['vertno']] = 1
        del s['pinfo']
        del s['nuse_tri']
        del s['use_tris']
        del s['patch_inds']
    return src


def _interp(data_1, data_2, stc_data, interp):
    """Helper to interpolate"""

    n_time = stc_data.shape[1]
    lin_interp_1 = np.linspace(1, 0, n_time, endpoint=False)
    lin_interp_2 = 1 - lin_interp_1
    if interp == 'zero':
        return np.dot(data_1, stc_data)
    else:  # interp == 'linear':
        data_1 = np.dot(data_1, stc_data)
        data_2 = np.dot(data_2, stc_data)
        return data_1 * lin_interp_1 + data_2 * lin_interp_2

