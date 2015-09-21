# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import warnings
from copy import deepcopy

from .evoked import _generate_noise
from ..event import _get_stim_channel
from ..io.pick import pick_types, pick_info, pick_channels
from ..source_estimate import VolSourceEstimate
from ..cov import make_ad_hoc_cov
from ..bem import fit_sphere_to_headshape, make_sphere_model, read_bem_solution
from ..io import RawArray, _BaseRaw, get_chpi_positions, FIFF
from ..forward import (_magnetic_dipole_field_vec, _merge_meg_eeg_fwds,
                       _stc_src_sel, convert_forward_solution,
                       _prepare_for_forward, _prep_meg_channels,
                       _compute_forwards, _to_forward_dict)
from ..transforms import _get_mri_head_t, transform_surface_to
from ..source_space import read_source_spaces, _points_outside_surface
from ..source_estimate import _BaseSourceEstimate
from ..utils import logger, verbose, check_random_state
from ..externals.six import string_types


@verbose
def simulate_raw(raw, stc, trans, src, bem, cov='simple',
                 blink=False, ecg=False, chpi=False, head_pos=None,
                 mindist=1.0, interp='cos2', n_jobs=1, random_state=None,
                 verbose=None):
    """Simulate raw data with head movements

    Parameters
    ----------
    raw : instance of Raw
        The raw template to use for simulation. The ``info``, ``times``,
        and potentially ``first_samp`` properties will be used.
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
        Source space corresponding to the stc. If string, should be a source
        space filename. Can also be an instance of loaded or generated
        SourceSpaces.
    bem : str | dict
        BEM solution  corresponding to the stc. If string, should be a BEM
        solution filename (e.g., "sample-5120-5120-5120-bem-sol.fif").
    cov : instance of Covariance | 'simple' | None
        The sensor covariance matrix used to generate noise. If None,
        no noise will be added. If 'simple', a basic (diagonal) ad-hoc
        noise covariance will be used.
    blink : bool
        If true, add simulated blink artifacts.
    ecg : bool
        If true, add simulated ECG artifacts.
    chpi : bool
        If true, simulate continuous head position indicator information.
    head_pos : None | str | dict
        Name of the position estimates file. Should be in the format of
        the files produced by maxfilter-produced. If dict, keys should
        be the time points and entries should be 4x3 ``dev_head_t``
        matrices. If None, the original head position (from
        ``info['dev_head_t']``) will be used.
    mindist : float
        Minimum distance between sources and the inner skull boundary
        to use during forward calculation.
    interp : str
        Either 'cos2', 'linear', or 'zero', the type of forward-solution
        interpolation to use between provided time points.
    n_jobs : int
        Number of jobs to use.
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
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

    if not isinstance(raw, _BaseRaw):
        raise TypeError('raw should be an instance of Raw')
    times, info, first_samp = raw.times, raw.info, raw.first_samp

    # Check for common flag errors and try to override
    if not isinstance(stc, _BaseSourceEstimate):
        raise TypeError('stc must be a SourceEstimate')
    if not np.allclose(info['sfreq'], 1. / stc.tstep):
        raise ValueError('stc and info must have same sample rate')
    if len(stc.times) <= 2:  # to ensure event encoding works
        raise ValueError('stc must have at least three time points')

    # Only use cHPI if custom frequency is in HPI information
    if chpi and np.all(['custom_ref' in x.keys()
                        for x in info['hpi_meas'][0]['hpi_coils']]):
        raise ValueError("`custom_ref` must be in "
                         "info['hpi_meas'][0]['hpi_coils'] to use cHPI")

    if len(pick_types(info, meg=False, stim=True)) == 0:
        # TODO: To further functionality, eventually expand raw object with
        #     this channel if it doesn't exist
        raise ValueError('At least one stim channel must be present to '
                         'record events.')

    rng = check_random_state(random_state)
    if interp not in ('cos2', 'linear', 'zero'):
        raise ValueError('interp must be "cos2", "linear", or "zero"')

    if head_pos is None:  # use pos from file
        dev_head_ts = [info['dev_head_t']] * 2
        offsets = np.array([0, len(times)])
        interp = 'zero'
    # Use position data to simulate head movement
    else:
        if isinstance(head_pos, string_types):
            head_pos = get_chpi_positions(head_pos, verbose=False)
        if isinstance(head_pos, tuple):  # can be an already-loaded pos file
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
        if not (ts >= 0).all():  # pathological if not
            raise RuntimeError('Cannot have t < 0 in transform file')
        tend = times[-1]
        bad = ts < 0
        if bad.any():
            raise RuntimeError('All position times must be >= 0, found %s/%s'
                               '< 0' % (bad.sum(), len(bad)))
        bad = ts > tend
        if bad.any():
            raise RuntimeError('All position times must be <= t_end (%0.1f '
                               'sec), found %s/%s bad values (is this a split '
                               'file?)' % (tend, bad.sum(), len(bad)))
        if ts[0] > 0:
            ts = np.r_[[0.], ts]
            dev_head_ts.insert(0, info['dev_head_t']['trans'])
        dev_head_ts = [{'trans': d, 'to': info['dev_head_t']['to'],
                        'from': info['dev_head_t']['from']}
                       for d in dev_head_ts]
        if ts[-1] < tend:
            dev_head_ts.append(dev_head_ts[-1])
            ts = np.r_[ts, [tend]]
        offsets = raw.time_as_index(ts)
        offsets[-1] = len(times)  # fix for roundoff error
        assert offsets[-2] != offsets[-1]
        del ts

    if isinstance(src, string_types):
        src = read_source_spaces(src, verbose=verbose)
    if isinstance(bem, string_types):
        bem = read_bem_solution(bem, verbose)
    if isinstance(cov, string_types):
        assert cov == 'simple'
        cov = make_ad_hoc_cov(info, verbose=False)
    assert np.array_equal(offsets, np.unique(offsets))
    assert len(offsets) == len(dev_head_ts)
    approx_events = int((len(times) / info['sfreq']) /
                        (stc.times[-1] - stc.times[0]))
    logger.info('Provided parameters will provide approximately %s event%s'
                % (approx_events, '' if approx_events == 1 else 's'))

    # Get chpi freqs and reorder
    if chpi:
        hpi_freqs = np.array([x['custom_ref'][0]
                             for x in info['hpi_meas'][0]['hpi_coils']])
        n_freqs = len(hpi_freqs)
        order = [x['number'] - 1 for x in info['hpi_meas'][0]['hpi_coils']]
        assert np.array_equal(np.unique(order), np.arange(n_freqs))
        hpi_freqs = hpi_freqs[order]
        hpi_order = info['hpi_results'][0]['order'] - 1
        assert np.array_equal(np.unique(hpi_order), np.arange(n_freqs))
        hpi_freqs = hpi_freqs[hpi_order]

    # Extract necessary info
    picks = pick_types(info, meg=True, eeg=True, exclude=[])  # for simulation
    meg_picks = pick_types(info, meg=True, eeg=False)  # for CHPI
    fwd_info = pick_info(info, picks)
    fwd_info['projs'] = []  # Ensure no 'projs' applied
    logger.info('Setting up raw data simulation using %s head position%s'
                % (len(dev_head_ts), 's' if len(dev_head_ts) != 1 else ''))

    if isinstance(stc, VolSourceEstimate):
        verts = [stc.vertices]
    else:
        verts = stc.vertices
    src = _restrict_source_space_to(src, verts)

    # array used to store result
    raw_data = np.zeros((len(info['ch_names']), len(times)))

    # figure out our cHPI, ECG, and blink dipoles
    dig = info['dig']
    assert all([d['coord_frame'] == FIFF.FIFFV_COORD_HEAD
                for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI])
    chpi_rrs = np.array([d['r'] for d in dig
                        if d['kind'] == FIFF.FIFFV_POINT_HPI])
    R, r0 = fit_sphere_to_headshape(info, verbose=False)[:2]
    R /= 1000.
    r0 /= 1000.
    ecg_rr = blink_rr = blink_bem = None
    if blink:
        blink_rr = [d['r'] for d in info['dig']
                    if d['ident'] == FIFF.FIFFV_POINT_NASION][0]
        blink_rr = blink_rr - r0
        blink_rr = (blink_rr / np.sqrt(np.sum(blink_rr * blink_rr)) *
                    0.98 * R)[np.newaxis, :]
        blink_rr += r0
        blink_bem = make_sphere_model(r0, head_radius=R,
                                      relative_radii=(0.99, 1.),
                                      sigmas=(0.33, 0.33), verbose=False)
        # let's oscillate between resting (17 bpm) and reading (4.5 bpm) rate
        # http://www.ncbi.nlm.nih.gov/pubmed/9399231
        blink_rate = np.cos(2 * np.pi * 1. / 60. * times)
        blink_rate *= 12.5 / 60.
        blink_rate += 4.5 / 60.
        blink_data = rng.rand(len(times)) < blink_rate / info['sfreq']
        blink_data = blink_data * (rng.rand(len(times)) + 0.5)  # varying amps
        blink_kernel = np.hanning(int(0.25 * info['sfreq']))
        blink_data = np.convolve(blink_data, blink_kernel,
                                 'same')[np.newaxis, :]
        blink_data += rng.randn(blink_data.shape[1]) * 0.05
        blink_data *= 100e-6

        # Add to file, rescale for simulation
        ch = pick_types(info, meg=False, eeg=False, eog=True)
        if len(ch) >= 1:
            raw_data[ch[-1], :] = blink_data
        blink_data *= 1e-3
        del blink_kernel, blink_rate
    if ecg:
        ecg_rr = np.array([[-R, 0, -3 * R]])
        max_beats = int(np.ceil(times[-1] * 70. / 60.))
        cardiac_idx = np.cumsum(rng.uniform(60. / 70., 60. / 50., max_beats) *
                                info['sfreq']).astype(int)
        cardiac_idx = cardiac_idx[cardiac_idx < len(times)]
        cardiac_data = np.zeros(len(times))
        cardiac_data[cardiac_idx] = 1
        cardiac_kernel = np.concatenate([
            2 * np.hanning(int(0.04 * info['sfreq'])),
            -0.3 * np.hanning(int(0.05 * info['sfreq'])),
            0.2 * np.hanning(int(0.26 * info['sfreq']))], axis=-1)
        ecg_data = np.convolve(cardiac_data, cardiac_kernel,
                               'same')[np.newaxis, :]
        ecg_data += rng.randn(ecg_data.shape[1]) * 0.05
        ecg_data *= 3e-4
        del cardiac_data, cardiac_kernel, max_beats, cardiac_idx

        # Add to data file, rescale for simulation
        ch = pick_types(info, meg=False, eeg=False, ecg=True)
        if len(ch) >= 1:
            raw_data[ch[-1], :] = ecg_data
        ecg_data *= 5e-4

    stc_event_idx = np.argmin(np.abs(stc.times))
    event_ch = _get_stim_channel(None, raw.info)  # XXX make more robust
    event_ch = pick_channels(info['ch_names'], event_ch)[0]
    used = np.zeros(len(times), bool)
    stc_indices = np.arange(len(times)) % len(stc.times)
    raw_data[event_ch, :] = 0.
    raw_data[picks, :] = 0.
    hpi_mag = 70e-9
    last_fwd = last_fwd_chpi = last_fwd_blink = last_fwd_ecg = src_sel = None
    zf = None  # final filter conditions for the noise
    chpi_nns = chpi_rrs / np.sqrt(np.sum(chpi_rrs * chpi_rrs,
                                         axis=1))[:, np.newaxis]
    for fi, (fwd, fwd_blink, fwd_ecg, fwd_chpi) in \
        enumerate(_iter_forward_solutions(
            fwd_info, trans, src, bem, blink_bem, dev_head_ts, mindist,
            chpi_rrs, blink_rr, ecg_rr, n_jobs)):
        # must be fixed orientation
        fwd = convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=True, verbose=False)
        # just use one arbitrary direction
        if blink:
            fwd_blink = fwd_blink['sol']['data'][:, ::3]
        if ecg:
            fwd_ecg = fwd_ecg['sol']['data'][:, ::3]

        # align cHPI magnetic dipoles in approx. radial direction
        for ii in range(len(chpi_rrs)):
            fwd_chpi[:, ii] = np.dot(fwd_chpi[:, 3 * ii:3 * (ii + 1)],
                                     chpi_nns[ii])
        fwd_chpi = fwd_chpi[:, :len(chpi_rrs)].copy()

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
        if last_fwd is None:
            last_fwd, last_fwd_blink, last_fwd_ecg, last_fwd_chpi = \
                fwd, fwd_blink, fwd_ecg, fwd_chpi
            continue

        # set up interpolation
        n_pts = offsets[fi] - offsets[fi - 1]
        if interp == 'zero':
            interps = None
        else:
            if interp == 'linear':
                interps = np.linspace(1, 0, n_pts, endpoint=False)
            else:  # interp == 'cos2':
                interps = np.cos(0.5 * np.pi * np.arange(n_pts)) ** 2
            interps = np.array([interps, 1 - interps])

        assert not used[offsets[fi - 1]:offsets[fi]].any()
        event_idxs = np.where(stc_indices[offsets[fi - 1]:offsets[fi]] ==
                              stc_event_idx)[0] + offsets[fi - 1]
        raw_data[event_ch, event_idxs] = fi

        logger.info('  Simulating data for %0.3f-%0.3f sec with %s event%s'
                    % (tuple(offsets[fi - 1:fi + 1] / info['sfreq']) +
                       (len(event_idxs), '' if len(event_idxs) == 1 else 's')))

        # Process data in large chunks to save on memory
        chunk_size = 10000
        chunks = np.concatenate((np.arange(offsets[fi - 1], offsets[fi],
                                           chunk_size), [offsets[fi]]))
        for start, stop in zip(chunks[:-1], chunks[1:]):
            assert stop - start <= chunk_size

            used[start:stop] = True
            if interp == 'zero':
                this_interp = None
            else:
                this_interp = interps[:, start - chunks[0]:stop - chunks[0]]
            time_sl = slice(start, stop)
            this_t = np.arange(start, stop) / info['sfreq']
            stc_idxs = stc_indices[time_sl]

            # simulate brain data
            raw_data[picks, time_sl] = \
                _interp(last_fwd['sol']['data'], fwd['sol']['data'],
                        stc.data[:, stc_idxs][src_sel], this_interp)

            # add sensor noise, ECG, blink, cHPI
            if cov is not None:
                noise, zf = _generate_noise(fwd_info, cov, [1, -1, 0.2], rng,
                                            len(stc_idxs), zi=zf)
                raw_data[picks, time_sl] += noise
            if blink:
                raw_data[picks, time_sl] += \
                    _interp(last_fwd_blink, fwd_blink, blink_data[:, time_sl],
                            this_interp)
            if ecg:
                raw_data[meg_picks, time_sl] += \
                    _interp(last_fwd_ecg, fwd_ecg, ecg_data[:, time_sl],
                            this_interp)
            if chpi:
                sinusoids = np.zeros((n_freqs, len(stc_idxs)))
                for fidx, freq in enumerate(hpi_freqs):
                    sinusoids[fidx] = 2 * np.pi * freq * this_t
                    sinusoids[fidx] = hpi_mag * np.sin(sinusoids[fidx])
                raw_data[meg_picks, time_sl] += \
                    _interp(last_fwd_chpi, fwd_chpi, sinusoids, this_interp)

        assert used[offsets[fi - 1]:offsets[fi]].all()

        # prepare for next iteration
        last_fwd, last_fwd_blink, last_fwd_ecg, last_fwd_chpi = \
            fwd, fwd_blink, fwd_ecg, fwd_chpi
    assert used.all()
    logger.info('Done')

    raw = RawArray(raw_data, info)
    return raw


def _iter_forward_solutions(info, trans, src, bem, bem_blink, dev_head_ts,
                            mindist, chpi_rrs, blink_rrs, ecg_rrs, n_jobs):
    """Calculate a forward solution for a subject"""
    mri_head_t, trans = _get_mri_head_t(trans)
    logger.info('Setting up forward solutions')
    megcoils, meg_info, compcoils, megnames, eegels, eegnames, rr, info, \
        update_kwargs, bem = _prepare_for_forward(
            src, mri_head_t, info, bem, mindist, n_jobs)
    del (src, mindist)

    eegfwd = _compute_forwards(rr, bem, [eegels], [None],
                               [None], ['eeg'], n_jobs, verbose=False)[0]
    eegfwd = _to_forward_dict(eegfwd, eegnames)
    if blink_rrs is not None:
        eegblink = _compute_forwards(blink_rrs, bem_blink, [eegels], [None],
                                     [None], ['eeg'], n_jobs,
                                     verbose=False)[0]
        eegblink = _to_forward_dict(eegblink, eegnames)

    coord_frame = FIFF.FIFFV_COORD_HEAD
    for ti, dev_head_t in enumerate(dev_head_ts):
        # Could be *slightly* more efficient not to do this N times,
        # but the cost here is tiny compared to actual fwd calculation
        logger.info('Computing gain matrix for transform #%s/%s'
                    % (ti + 1, len(dev_head_ts)))
        info = deepcopy(info)
        info['dev_head_t'] = dev_head_t
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, True, [], False, verbose=False)

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
        megfwd = _to_forward_dict(megfwd, megnames)
        fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)
        fwd.update(**update_kwargs)

        fwd_blink = fwd_ecg = None
        if blink_rrs is not None:
            megblink = _compute_forwards(blink_rrs, bem_blink, [megcoils],
                                         [compcoils], [meg_info], ['meg'],
                                         n_jobs, verbose=False)[0]
            megblink = _to_forward_dict(megblink, megnames)
            fwd_blink = _merge_meg_eeg_fwds(megblink, eegblink, verbose=False)
        if ecg_rrs is not None:
            megecg = _compute_forwards(ecg_rrs, bem_blink, [megcoils],
                                       [compcoils], [meg_info], ['meg'],
                                       n_jobs, verbose=False)[0]
            fwd_ecg = _to_forward_dict(megecg, megnames)
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


def _interp(data_1, data_2, stc_data, interps):
    """Helper to interpolate"""
    out_data = np.dot(data_1, stc_data)
    if interps is not None:
        out_data *= interps[0]
        data_1 = np.dot(data_1, stc_data)
        data_1 *= interps[1]
        out_data += data_1
        del data_1
    return out_data
