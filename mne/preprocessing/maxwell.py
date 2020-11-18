# -*- coding: utf-8 -*-
# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)

from collections import Counter, OrderedDict
from functools import partial
from math import factorial
from os import path as op

import numpy as np
from scipy import linalg

from .. import __version__
from ..annotations import _annotations_starts_stops
from ..bem import _check_origin
from ..transforms import (_str_to_frame, _get_trans, Transform, apply_trans,
                          _find_vector_rotation, _cart_to_sph, _get_n_moments,
                          _sph_to_cart_partials, _deg_ord_idx, _average_quats,
                          _sh_complex_to_real, _sh_real_to_complex, _sh_negate,
                          quat_to_rot, rot_to_quat)
from ..forward import _concatenate_coils, _prep_meg_channels, _create_meg_coils
from ..surface import _normalize_vectors
from ..io.constants import FIFF, FWD
from ..io.meas_info import _simplify_info
from ..io.proc_history import _read_ctc
from ..io.write import _generate_meas_id, DATE_NONE
from ..io import (_loc_to_coil_trans, _coil_trans_to_loc, BaseRaw, RawArray,
                  Projection)
from ..io.pick import pick_types, pick_info
from ..utils import (verbose, logger, _clean_names, warn, _time_mask, _pl,
                     _check_option, _ensure_int, _validate_type, use_log_level)
from ..fixes import _get_args, _safe_svd, einsum, bincount
from ..channels.channels import _get_T1T2_mag_inds, fix_mag_coil_types


# Note: MF uses single precision and some algorithms might use
# truncated versions of constants (e.g., μ0), which could lead to small
# differences between algorithms


# Changes to arguments here should also be made in find_bad_channels_maxwell
@verbose
def maxwell_filter(raw, origin='auto', int_order=8, ext_order=3,
                   calibration=None, cross_talk=None, st_duration=None,
                   st_correlation=0.98, coord_frame='head', destination=None,
                   regularize='in', ignore_ref=False, bad_condition='error',
                   head_pos=None, st_fixed=True, st_only=False, mag_scale=100.,
                   skip_by_annotation=('edge', 'bad_acq_skip'),
                   extended_proj=(), verbose=None):
    """Maxwell filter data using multipole moments.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered.

        .. warning:: It is critical to mark bad channels in
                     ``raw.info['bads']`` prior to processing in order to
                     prevent artifact spreading. Manual inspection and use
                     of :func:`~find_bad_channels_maxwell` is recommended.
    %(maxwell_origin)s
    %(maxwell_int)s
    %(maxwell_ext)s
    %(maxwell_cal)s
    %(maxwell_cross)s
    st_duration : float | None
        If not None, apply spatiotemporal SSS with specified buffer duration
        (in seconds). MaxFilter™'s default is 10.0 seconds in v2.2.
        Spatiotemporal SSS acts as implicitly as a high-pass filter where the
        cut-off frequency is 1/st_duration Hz. For this (and other) reasons,
        longer buffers are generally better as long as your system can handle
        the higher memory usage. To ensure that each window is processed
        identically, choose a buffer length that divides evenly into your data.
        Any data at the trailing edge that doesn't fit evenly into a whole
        buffer window will be lumped into the previous buffer.
    st_correlation : float
        Correlation limit between inner and outer subspaces used to reject
        ovwrlapping intersecting inner/outer signals during spatiotemporal SSS.
    %(maxwell_coord)s
    %(maxwell_dest)s
    %(maxwell_reg)s
    %(maxwell_ref)s
    %(maxwell_cond)s
    %(maxwell_pos)s

        .. versionadded:: 0.12
    %(maxwell_st_fixed_only)s
    %(maxwell_mag)s

        .. versionadded:: 0.13
    %(maxwell_skip)s

        .. versionadded:: 0.17
    %(maxwell_extended)s
    %(verbose)s

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        The raw data with Maxwell filtering applied.

    See Also
    --------
    mne.preprocessing.annotate_flat
    mne.preprocessing.find_bad_channels_maxwell
    mne.chpi.filter_chpi
    mne.chpi.read_head_pos
    mne.epochs.average_movements

    Notes
    -----
    .. versionadded:: 0.11

    Some of this code was adapted and relicensed (with BSD form) with
    permission from Jussi Nurminen. These algorithms are based on work
    from [1]_ and [2]_. It will likely use multiple CPU cores, see the
    :ref:`FAQ <faq_cpu>` for more information.

    .. warning:: Maxwell filtering in MNE is not designed or certified
                 for clinical use.

    Compared to the MEGIN MaxFilter™ software, the MNE Maxwell filtering
    routines currently provide the following features:

    .. table::
       :widths: auto

       +-----------------------------------------------------------------------------+-----+-----------+
       | Feature                                                                     | MNE | MaxFilter |
       +=============================================================================+=====+===========+
       | Maxwell filtering software shielding                                        | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Bad channel reconstruction                                                  | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Cross-talk cancellation                                                     | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Fine calibration correction (1D)                                            | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Fine calibration correction (3D)                                            | ✓   |           |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Spatio-temporal SSS (tSSS)                                                  | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Coordinate frame translation                                                | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Regularization using information theory                                     | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Movement compensation (raw)                                                 | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Movement compensation (:func:`epochs <mne.epochs.average_movements>`)       | ✓   |           |
       +-----------------------------------------------------------------------------+-----+-----------+
       | :func:`cHPI subtraction <mne.chpi.filter_chpi>`                             | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Double floating point precision                                             | ✓   |           |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Seamless processing of split (``-1.fif``) and concatenated files            | ✓   |           |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Automatic bad channel detection (:func:`~find_bad_channels_maxwell`)        | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Head position estimation (:func:`~mne.chpi.compute_head_pos`)               | ✓   | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Certified for clinical use                                                  |     | ✓         |
       +-----------------------------------------------------------------------------+-----+-----------+
       | Extended external basis (eSSS)                                              | ✓   |           |
       +-----------------------------------------------------------------------------+-----+-----------+

    Epoch-based movement compensation is described in [1]_.

    Use of Maxwell filtering routines with non-Neuromag systems is currently
    **experimental**. Worse results for non-Neuromag systems are expected due
    to (at least):

    * Missing fine-calibration and cross-talk cancellation data for
      other systems.
    * Processing with reference sensors has not been vetted.
    * Regularization of components may not work well for all systems.
    * Coil integration has not been optimized using Abramowitz/Stegun
      definitions.

    .. note:: Various Maxwell filtering algorithm components are covered by
              patents owned by MEGIN. These patents include, but may not be
              limited to:

              - US2006031038 (Signal Space Separation)
              - US6876196 (Head position determination)
              - WO2005067789 (DC fields)
              - WO2005078467 (MaxShield)
              - WO2006114473 (Temporal Signal Space Separation)

              These patents likely preclude the use of Maxwell filtering code
              in commercial applications. Consult a lawyer if necessary.

    Currently, in order to perform Maxwell filtering, the raw data must not
    have any projectors applied. During Maxwell filtering, the spatial
    structure of the data is modified, so projectors are discarded (unless
    in ``st_only=True`` mode).

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.
           https://doi.org/10.1063/1.1935742

    .. [2] Taulu S. and Simola J. "Spatiotemporal signal space separation
           method for rejecting nearby interference in MEG measurements,"
           Physics in Medicine and Biology, vol. 51, pp. 1759-1768, 2006.
           https://doi.org/10.1088/0031-9155/51/7/008
    """  # noqa: E501
    logger.info('Maxwell filtering raw data')
    params = _prep_maxwell_filter(
        raw=raw, origin=origin, int_order=int_order, ext_order=ext_order,
        calibration=calibration, cross_talk=cross_talk,
        st_duration=st_duration, st_correlation=st_correlation,
        coord_frame=coord_frame, destination=destination,
        regularize=regularize, ignore_ref=ignore_ref,
        bad_condition=bad_condition, head_pos=head_pos, st_fixed=st_fixed,
        st_only=st_only, mag_scale=mag_scale,
        skip_by_annotation=skip_by_annotation, extended_proj=extended_proj)
    raw_sss = _run_maxwell_filter(raw, **params)
    # Update info
    _update_sss_info(raw_sss, **params['update_kwargs'])
    logger.info('[done]')
    return raw_sss


@verbose
def _prep_maxwell_filter(
        raw, origin='auto', int_order=8, ext_order=3,
        calibration=None, cross_talk=None, st_duration=None,
        st_correlation=0.98, coord_frame='head', destination=None,
        regularize='in', ignore_ref=False, bad_condition='error',
        head_pos=None, st_fixed=True, st_only=False,
        mag_scale=100.,
        skip_by_annotation=('edge', 'bad_acq_skip'), extended_proj=(),
        reconstruct='in', verbose=None):
    # There are an absurd number of different possible notations for spherical
    # coordinates, which confounds the notation for spherical harmonics.  Here,
    # we purposefully stay away from shorthand notation in both and use
    # explicit terms (like 'azimuth' and 'polar') to avoid confusion.
    # See mathworld.wolfram.com/SphericalHarmonic.html for more discussion.
    # Our code follows the same standard that ``scipy`` uses for ``sph_harm``.

    # triage inputs ASAP to avoid late-thrown errors
    _validate_type(raw, BaseRaw, 'raw')
    _check_usable(raw)
    _check_regularize(regularize)
    st_correlation = float(st_correlation)
    if st_correlation <= 0. or st_correlation > 1.:
        raise ValueError('Need 0 < st_correlation <= 1., got %s'
                         % st_correlation)
    _check_option('coord_frame', coord_frame, ['head', 'meg'])
    head_frame = True if coord_frame == 'head' else False
    recon_trans = _check_destination(destination, raw.info, head_frame)
    if st_duration is not None:
        st_duration = float(st_duration)
        st_correlation = float(st_correlation)
        st_duration = int(round(st_duration * raw.info['sfreq']))
        if not 0. < st_correlation <= 1:
            raise ValueError('st_correlation must be between 0. and 1.')
    _check_option('bad_condition', bad_condition,
                  ['error', 'warning', 'ignore', 'info'])
    if raw.info['dev_head_t'] is None and coord_frame == 'head':
        raise RuntimeError('coord_frame cannot be "head" because '
                           'info["dev_head_t"] is None; if this is an '
                           'empty room recording, consider using '
                           'coord_frame="meg"')
    if st_only and st_duration is None:
        raise ValueError('st_duration must not be None if st_only is True')
    head_pos = _check_pos(head_pos, head_frame, raw, st_fixed,
                          raw.info['sfreq'])
    _check_info(raw.info, sss=not st_only, tsss=st_duration is not None,
                calibration=not st_only and calibration is not None,
                ctc=not st_only and cross_talk is not None)

    # Now we can actually get moving
    info = raw.info.copy()
    meg_picks, mag_picks, grad_picks, good_mask, mag_or_fine = \
        _get_mf_picks_fix_mags(info, int_order, ext_order, ignore_ref)

    # Magnetometers are scaled to improve numerical stability
    coil_scale, mag_scale = _get_coil_scale(
        meg_picks, mag_picks, grad_picks, mag_scale, info)

    #
    # Extended projection vectors
    #
    _validate_type(extended_proj, (list, tuple), 'extended_proj')
    good_names = [info['ch_names'][c] for c in meg_picks[good_mask]]
    if len(extended_proj) > 0:
        extended_proj_ = list()
        for pi, proj in enumerate(extended_proj):
            item = 'extended_proj[%d]' % (pi,)
            _validate_type(proj, Projection, item)
            got_names = proj['data']['col_names']
            diff = sorted(set(got_names).symmetric_difference(set(good_names)))
            if diff:
                raise ValueError('%s channel names (length %d) do not match '
                                 'the good MEG channel names (length %d):\n%s'
                                 % (item, len(got_names), len(good_names),
                                    ', '.join(diff)))
            extended_proj_.append(proj['data']['data'])
        extended_proj = np.concatenate(extended_proj_)
        logger.info('    Extending external SSS basis using %d projection '
                    'vectors' % (len(extended_proj),))

    #
    # Fine calibration processing (load fine cal and overwrite sensor geometry)
    #
    sss_cal = dict()
    if calibration is not None:
        calibration, sss_cal = _update_sensor_geometry(
            info, calibration, ignore_ref)
        mag_or_fine.fill(True)  # all channels now have some mag-type data

    # Determine/check the origin of the expansion
    origin = _check_origin(origin, info, coord_frame, disp=True)
    # Convert to the head frame
    if coord_frame == 'meg' and info['dev_head_t'] is not None:
        origin_head = apply_trans(info['dev_head_t'], origin)
    else:
        origin_head = origin
    update_kwargs = dict(
        origin=origin, coord_frame=coord_frame, sss_cal=sss_cal,
        int_order=int_order, ext_order=ext_order,
        extended_proj=extended_proj)
    del origin, coord_frame, sss_cal
    origin_head.setflags(write=False)

    #
    # Cross-talk processing
    #
    meg_ch_names = [info['ch_names'][p] for p in meg_picks]
    ctc, sss_ctc = _read_cross_talk(cross_talk, meg_ch_names)
    update_kwargs['sss_ctc'] = sss_ctc
    del sss_ctc

    #
    # Translate to destination frame (always use non-fine-cal bases)
    #
    exp = dict(origin=origin_head, int_order=int_order, ext_order=0)
    all_coils = _prep_mf_coils(info, ignore_ref)
    S_recon = _trans_sss_basis(exp, all_coils, recon_trans, coil_scale)
    exp['ext_order'] = ext_order
    exp['extended_proj'] = extended_proj
    del extended_proj
    # Reconstruct data from internal space only (Eq. 38), and rescale S_recon
    S_recon /= coil_scale
    if recon_trans is not None:
        # warn if we have translated too far
        diff = 1000 * (info['dev_head_t']['trans'][:3, 3] -
                       recon_trans['trans'][:3, 3])
        dist = np.sqrt(np.sum(_sq(diff)))
        if dist > 25.:
            warn('Head position change is over 25 mm (%s) = %0.1f mm'
                 % (', '.join('%0.1f' % x for x in diff), dist))

    # Reconstruct raw file object with spatiotemporal processed data
    max_st = dict()
    if st_duration is not None:
        if st_only:
            job = FIFF.FIFFV_SSS_JOB_TPROJ
        else:
            job = FIFF.FIFFV_SSS_JOB_ST
        max_st.update(job=job, subspcorr=st_correlation,
                      buflen=st_duration / info['sfreq'])
        logger.info('    Processing data using tSSS with st_duration=%s'
                    % max_st['buflen'])
        st_when = 'before' if st_fixed else 'after'  # relative to movecomp
    else:
        # st_duration from here on will act like the chunk size
        st_duration = min(max(int(round(10. * info['sfreq'])), 1),
                          len(raw.times))
        st_correlation = None
        st_when = 'never'
    update_kwargs['max_st'] = max_st
    del st_fixed, max_st

    # Figure out which transforms we need for each tSSS block
    # (and transform pos[1] to times)
    head_pos[1] = raw.time_as_index(head_pos[1], use_rounding=True)
    # Compute the first bit of pos_data for cHPI reporting
    if info['dev_head_t'] is not None and head_pos[0] is not None:
        this_pos_quat = np.concatenate([
            rot_to_quat(info['dev_head_t']['trans'][:3, :3]),
            info['dev_head_t']['trans'][:3, 3],
            np.zeros(3)])
    else:
        this_pos_quat = None
    _get_this_decomp_trans = partial(
        _get_decomp, all_coils=all_coils,
        cal=calibration, regularize=regularize,
        exp=exp, ignore_ref=ignore_ref, coil_scale=coil_scale,
        grad_picks=grad_picks, mag_picks=mag_picks, good_mask=good_mask,
        mag_or_fine=mag_or_fine, bad_condition=bad_condition,
        mag_scale=mag_scale)
    update_kwargs.update(
        nchan=good_mask.sum(), st_only=st_only, recon_trans=recon_trans)
    params = dict(
        skip_by_annotation=skip_by_annotation,
        st_duration=st_duration, st_correlation=st_correlation,
        st_only=st_only, st_when=st_when, ctc=ctc, coil_scale=coil_scale,
        this_pos_quat=this_pos_quat, meg_picks=meg_picks,
        good_mask=good_mask, grad_picks=grad_picks, head_pos=head_pos,
        info=info, _get_this_decomp_trans=_get_this_decomp_trans,
        S_recon=S_recon, update_kwargs=update_kwargs)
    return params


def _run_maxwell_filter(
        raw, skip_by_annotation, st_duration, st_correlation, st_only,
        st_when, ctc, coil_scale, this_pos_quat, meg_picks, good_mask,
        grad_picks, head_pos, info, _get_this_decomp_trans, S_recon,
        update_kwargs,
        reconstruct='in', copy=True):
    # Eventually find_bad_channels_maxwell could be sped up by moving this
    # outside the loop (e.g., in the prep function) but regularization depends
    # on which channels are being used, so easier just to include it here.
    # The time it takes to recompute S and pS themselves is roughly on par
    # with the np.dot with the data, so not a huge gain to be made there.
    S_decomp, S_decomp_full, pS_decomp, reg_moments, n_use_in = \
        _get_this_decomp_trans(info['dev_head_t'], t=0.)
    update_kwargs.update(reg_moments=reg_moments.copy())
    if ctc is not None:
        ctc = ctc[good_mask][:, good_mask]

    add_channels = (head_pos[0] is not None) and (not st_only) and copy
    raw_sss, pos_picks = _copy_preload_add_channels(
        raw, add_channels, copy, info)
    sfreq = info['sfreq']
    del raw
    if not st_only:
        # remove MEG projectors, they won't apply now
        _remove_meg_projs(raw_sss)
    # Figure out which segments of data we can use
    onsets, ends = _annotations_starts_stops(
        raw_sss, skip_by_annotation, invert=True)
    max_samps = (ends - onsets).max()
    if not 0. < st_duration <= max_samps + 1.:
        raise ValueError('st_duration (%0.1fs) must be between 0 and the '
                         'longest contiguous duration of the data '
                         '(%0.1fs).' % (st_duration / sfreq,
                                        max_samps / sfreq))
    # Generate time points to break up data into equal-length windows
    starts, stops = list(), list()
    for onset, end in zip(onsets, ends):
        read_lims = np.arange(onset, end + 1, st_duration)
        if len(read_lims) == 1:
            read_lims = np.concatenate([read_lims, [end]])
        if read_lims[-1] != end:
            read_lims[-1] = end
            # fold it into the previous buffer
            n_last_buf = read_lims[-1] - read_lims[-2]
            if st_correlation is not None and len(read_lims) > 2:
                if n_last_buf >= st_duration:
                    logger.info(
                        '    Spatiotemporal window did not fit evenly into'
                        'contiguous data segment. %0.2f seconds were lumped '
                        'into the previous window.'
                        % ((n_last_buf - st_duration) / sfreq,))
                else:
                    logger.info(
                        '    Contiguous data segment of duration %0.2f '
                        'seconds is too short to be processed with tSSS '
                        'using duration %0.2f'
                        % (n_last_buf / sfreq, st_duration / sfreq))
        assert len(read_lims) >= 2
        assert read_lims[0] == onset and read_lims[-1] == end
        starts.extend(read_lims[:-1])
        stops.extend(read_lims[1:])
        del read_lims
    st_duration = min(max_samps, st_duration)

    # Loop through buffer windows of data
    n_sig = int(np.floor(np.log10(max(len(starts), 0)))) + 1
    logger.info(
        '    Processing %s data chunk%s' % (len(starts), _pl(starts)))
    for ii, (start, stop) in enumerate(zip(starts, stops)):
        tsss_valid = (stop - start) >= st_duration
        rel_times = raw_sss.times[start:stop]
        t_str = '%8.3f - %8.3f sec' % tuple(rel_times[[0, -1]])
        t_str += ('(#%d/%d)' % (ii + 1, len(starts))).rjust(2 * n_sig + 5)

        # Get original data
        orig_data = raw_sss._data[meg_picks[good_mask], start:stop]
        # This could just be np.empty if not st_only, but shouldn't be slow
        # this way so might as well just always take the original data
        out_meg_data = raw_sss._data[meg_picks, start:stop]
        # Apply cross-talk correction
        if ctc is not None:
            orig_data = ctc.dot(orig_data)
        out_pos_data = np.empty((len(pos_picks), stop - start))

        # Figure out which positions to use
        t_s_s_q_a = _trans_starts_stops_quats(head_pos, start, stop,
                                              this_pos_quat)
        n_positions = len(t_s_s_q_a[0])

        # Set up post-tSSS or do pre-tSSS
        if st_correlation is not None:
            # If doing tSSS before movecomp...
            resid = orig_data.copy()  # to be safe let's operate on a copy
            if st_when == 'after':
                orig_in_data = np.empty((len(meg_picks), stop - start))
            else:  # 'before'
                avg_trans = t_s_s_q_a[-1]
                if avg_trans is not None:
                    # if doing movecomp
                    S_decomp_st, _, pS_decomp_st, _, n_use_in_st = \
                        _get_this_decomp_trans(avg_trans, t=rel_times[0])
                else:
                    S_decomp_st, pS_decomp_st = S_decomp, pS_decomp
                    n_use_in_st = n_use_in
                orig_in_data = np.dot(np.dot(S_decomp_st[:, :n_use_in_st],
                                             pS_decomp_st[:n_use_in_st]),
                                      resid)
                resid -= np.dot(np.dot(S_decomp_st[:, n_use_in_st:],
                                       pS_decomp_st[n_use_in_st:]), resid)
                resid -= orig_in_data
                # Here we operate on our actual data
                proc = out_meg_data if st_only else orig_data
                _do_tSSS(proc, orig_in_data, resid, st_correlation,
                         n_positions, t_str, tsss_valid)

        if not st_only or st_when == 'after':
            # Do movement compensation on the data
            for trans, rel_start, rel_stop, this_pos_quat in \
                    zip(*t_s_s_q_a[:4]):
                # Recalculate bases if necessary (trans will be None iff the
                # first position in this interval is the same as last of the
                # previous interval)
                if trans is not None:
                    S_decomp, S_decomp_full, pS_decomp, reg_moments, \
                        n_use_in = _get_this_decomp_trans(
                            trans, t=rel_times[rel_start])

                # Determine multipole moments for this interval
                mm_in = np.dot(pS_decomp[:n_use_in],
                               orig_data[:, rel_start:rel_stop])

                # Our output data
                if not st_only:
                    if reconstruct == 'in':
                        proj = S_recon.take(reg_moments[:n_use_in], axis=1)
                        mult = mm_in
                    else:
                        assert reconstruct == 'orig'
                        proj = S_decomp_full  # already picked reg
                        mm_out = np.dot(pS_decomp[n_use_in:],
                                        orig_data[:, rel_start:rel_stop])
                        mult = np.concatenate((mm_in, mm_out))
                    out_meg_data[:, rel_start:rel_stop] = \
                        np.dot(proj, mult)
                if len(pos_picks) > 0:
                    out_pos_data[:, rel_start:rel_stop] = \
                        this_pos_quat[:, np.newaxis]

                # Transform orig_data to store just the residual
                if st_when == 'after':
                    # Reconstruct data using original location from external
                    # and internal spaces and compute residual
                    rel_resid_data = resid[:, rel_start:rel_stop]
                    orig_in_data[:, rel_start:rel_stop] = \
                        np.dot(S_decomp[:, :n_use_in], mm_in)
                    rel_resid_data -= np.dot(np.dot(S_decomp[:, n_use_in:],
                                                    pS_decomp[n_use_in:]),
                                             rel_resid_data)
                    rel_resid_data -= orig_in_data[:, rel_start:rel_stop]

        # If doing tSSS at the end
        if st_when == 'after':
            _do_tSSS(out_meg_data, orig_in_data, resid, st_correlation,
                     n_positions, t_str, tsss_valid)
        elif st_when == 'never' and head_pos[0] is not None:
            logger.info('        Used % 2d head position%s for %s'
                        % (n_positions, _pl(n_positions), t_str))
        raw_sss._data[meg_picks, start:stop] = out_meg_data
        raw_sss._data[pos_picks, start:stop] = out_pos_data
    return raw_sss


def _get_coil_scale(meg_picks, mag_picks, grad_picks, mag_scale, info):
    """Get the magnetometer scale factor."""
    if isinstance(mag_scale, str):
        if mag_scale != 'auto':
            raise ValueError('mag_scale must be a float or "auto", got "%s"'
                             % mag_scale)
        if len(mag_picks) in (0, len(meg_picks)):
            mag_scale = 100.  # only one coil type, doesn't matter
            logger.info('    Setting mag_scale=%0.2f because only one '
                        'coil type is present' % mag_scale)
        else:
            # Find our physical distance between gradiometer pickup loops
            # ("base line")
            coils = _create_meg_coils([info['chs'][pick]
                                       for pick in meg_picks], 'accurate')
            grad_base = {coils[pick]['base'] for pick in grad_picks}
            if len(grad_base) != 1 or list(grad_base)[0] <= 0:
                raise RuntimeError('Could not automatically determine '
                                   'mag_scale, could not find one '
                                   'proper gradiometer distance from: %s'
                                   % list(grad_base))
            grad_base = list(grad_base)[0]
            mag_scale = 1. / grad_base
            logger.info('    Setting mag_scale=%0.2f based on gradiometer '
                        'distance %0.2f mm' % (mag_scale, 1000 * grad_base))
    mag_scale = float(mag_scale)
    coil_scale = np.ones((len(meg_picks), 1))
    coil_scale[mag_picks] = mag_scale
    return coil_scale, mag_scale


def _remove_meg_projs(inst):
    """Remove inplace existing MEG projectors (assumes inactive)."""
    meg_picks = pick_types(inst.info, meg=True, exclude=[])
    meg_channels = [inst.ch_names[pi] for pi in meg_picks]
    non_meg_proj = list()
    for proj in inst.info['projs']:
        if not any(c in meg_channels for c in proj['data']['col_names']):
            non_meg_proj.append(proj)
    inst.add_proj(non_meg_proj, remove_existing=True, verbose=False)


def _check_destination(destination, info, head_frame):
    """Triage our reconstruction trans."""
    if destination is None:
        return info['dev_head_t']
    if not head_frame:
        raise RuntimeError('destination can only be set if using the '
                           'head coordinate frame')
    if isinstance(destination, str):
        recon_trans = _get_trans(destination, 'meg', 'head')[0]
    elif isinstance(destination, Transform):
        recon_trans = destination
    else:
        destination = np.array(destination, float)
        if destination.shape != (3,):
            raise ValueError('destination must be a 3-element vector, '
                             'str, or None')
        recon_trans = np.eye(4)
        recon_trans[:3, 3] = destination
        recon_trans = Transform('meg', 'head', recon_trans)
    if recon_trans.to_str != 'head' or recon_trans.from_str != 'MEG device':
        raise RuntimeError('Destination transform is not MEG device -> head, '
                           'got %s -> %s' % (recon_trans.from_str,
                                             recon_trans.to_str))
    return recon_trans


@verbose
def _prep_mf_coils(info, ignore_ref=True, verbose=None):
    """Get all coil integration information loaded and sorted."""
    coils, comp_coils = _prep_meg_channels(
        info, accurate=True, head_frame=False,
        ignore_ref=ignore_ref, do_picking=False, verbose=False)[:2]
    mag_mask = _get_mag_mask(coils)
    if len(comp_coils) > 0:
        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=[])
        ref_picks = pick_types(info, meg=False, ref_meg=True, exclude=[])
        inserts = np.searchsorted(meg_picks, ref_picks)
        # len(inserts) == len(comp_coils)
        for idx, comp_coil in zip(inserts[::-1], comp_coils[::-1]):
            coils.insert(idx, comp_coil)
        # Now we have:
        # [c['chname'] for c in coils] ==
        # [info['ch_names'][ii]
        #  for ii in pick_types(info, meg=True, ref_meg=True)]

    # Now coils is a sorted list of coils. Time to do some vectorization.
    n_coils = len(coils)
    rmags = np.concatenate([coil['rmag'] for coil in coils])
    cosmags = np.concatenate([coil['cosmag'] for coil in coils])
    ws = np.concatenate([coil['w'] for coil in coils])
    cosmags *= ws[:, np.newaxis]
    del ws
    n_int = np.array([len(coil['rmag']) for coil in coils])
    bins = np.repeat(np.arange(len(n_int)), n_int)
    bd = np.concatenate(([0], np.cumsum(n_int)))
    slice_map = {ii: slice(start, stop)
                 for ii, (start, stop) in enumerate(zip(bd[:-1], bd[1:]))}
    return rmags, cosmags, bins, n_coils, mag_mask, slice_map


def _trans_starts_stops_quats(pos, start, stop, this_pos_data):
    """Get all trans and limits we need."""
    pos_idx = np.arange(*np.searchsorted(pos[1], [start, stop]))
    used = np.zeros(stop - start, bool)
    trans = list()
    rel_starts = list()
    rel_stops = list()
    quats = list()
    weights = list()
    for ti in range(-1, len(pos_idx)):
        # first iteration for this block of data
        if ti < 0:
            rel_start = 0
            rel_stop = pos[1][pos_idx[0]] if len(pos_idx) > 0 else stop
            rel_stop = rel_stop - start
            if rel_start == rel_stop:
                continue  # our first pos occurs on first time sample
            # Don't calculate S_decomp here, use the last one
            trans.append(None)  # meaning: use previous
            quats.append(this_pos_data)
        else:
            rel_start = pos[1][pos_idx[ti]] - start
            if ti == len(pos_idx) - 1:
                rel_stop = stop - start
            else:
                rel_stop = pos[1][pos_idx[ti + 1]] - start
            trans.append(pos[0][pos_idx[ti]])
            quats.append(pos[2][pos_idx[ti]])
        assert 0 <= rel_start
        assert rel_start < rel_stop
        assert rel_stop <= stop - start
        assert not used[rel_start:rel_stop].any()
        used[rel_start:rel_stop] = True
        rel_starts.append(rel_start)
        rel_stops.append(rel_stop)
        weights.append(rel_stop - rel_start)
    assert used.all()
    # Use weighted average for average trans over the window
    if this_pos_data is None:
        avg_trans = None
    else:
        weights = np.array(weights)
        quats = np.array(quats)
        weights = weights / weights.sum().astype(float)  # int -> float
        avg_quat = _average_quats(quats[:, :3], weights)
        avg_t = np.dot(weights, quats[:, 3:6])
        avg_trans = np.vstack([
            np.hstack([quat_to_rot(avg_quat), avg_t[:, np.newaxis]]),
            [[0., 0., 0., 1.]]])
    return trans, rel_starts, rel_stops, quats, avg_trans


def _do_tSSS(clean_data, orig_in_data, resid, st_correlation,
             n_positions, t_str, tsss_valid):
    """Compute and apply SSP-like projection vectors based on min corr."""
    if not tsss_valid:
        t_proj = np.empty((clean_data.shape[1], 0))
    else:
        np.asarray_chkfinite(resid)
        t_proj = _overlap_projector(orig_in_data, resid, st_correlation)
    # Apply projector according to Eq. 12 in [2]_
    msg = ('        Projecting %2d intersecting tSSS component%s '
           'for %s' % (t_proj.shape[1], _pl(t_proj.shape[1], ' '), t_str))
    if n_positions > 1:
        msg += ' (across %2d position%s)' % (n_positions,
                                             _pl(n_positions, ' '))
    logger.info(msg)
    clean_data -= np.dot(np.dot(clean_data, t_proj), t_proj.T)


def _copy_preload_add_channels(raw, add_channels, copy, info):
    """Load data for processing and (maybe) add cHPI pos channels."""
    if copy:
        raw = raw.copy()
    raw.info['chs'] = info['chs']  # updated coil types
    if add_channels:
        kinds = [FIFF.FIFFV_QUAT_1, FIFF.FIFFV_QUAT_2, FIFF.FIFFV_QUAT_3,
                 FIFF.FIFFV_QUAT_4, FIFF.FIFFV_QUAT_5, FIFF.FIFFV_QUAT_6,
                 FIFF.FIFFV_HPI_G, FIFF.FIFFV_HPI_ERR, FIFF.FIFFV_HPI_MOV]
        out_shape = (len(raw.ch_names) + len(kinds), len(raw.times))
        out_data = np.zeros(out_shape, np.float64)
        msg = '    Appending head position result channels and '
        if raw.preload:
            logger.info(msg + 'copying original raw data')
            out_data[:len(raw.ch_names)] = raw._data
            raw._data = out_data
        else:
            logger.info(msg + 'loading raw data from disk')
            raw._preload_data(out_data[:len(raw.ch_names)], verbose=False)
            raw._data = out_data
        assert raw.preload is True
        off = len(raw.ch_names)
        chpi_chs = [
            dict(ch_name='CHPI%03d' % (ii + 1), logno=ii + 1,
                 scanno=off + ii + 1, unit_mul=-1, range=1., unit=-1,
                 kind=kinds[ii], coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                 cal=1e-4, coil_type=FWD.COIL_UNKNOWN, loc=np.zeros(12))
            for ii in range(len(kinds))]
        raw.info['chs'].extend(chpi_chs)
        raw.info._update_redundant()
        raw.info._check_consistency()
        assert raw._data.shape == (raw.info['nchan'], len(raw.times))
        # Return the pos picks
        pos_picks = np.arange(len(raw.ch_names) - len(chpi_chs),
                              len(raw.ch_names))
        return raw, pos_picks
    else:
        if copy:
            if not raw.preload:
                logger.info('    Loading raw data from disk')
                raw.load_data(verbose=False)
            else:
                logger.info('    Using loaded raw data')
        return raw, np.array([], int)


def _check_pos(pos, head_frame, raw, st_fixed, sfreq):
    """Check for a valid pos array and transform it to a more usable form."""
    _validate_type(pos, (np.ndarray, None), 'head_pos')
    if pos is None:
        return [None, np.array([-1])]
    if not head_frame:
        raise ValueError('positions can only be used if coord_frame="head"')
    if not st_fixed:
        warn('st_fixed=False is untested, use with caution!')
    if not isinstance(pos, np.ndarray):
        raise TypeError('pos must be an ndarray')
    if pos.ndim != 2 or pos.shape[1] != 10:
        raise ValueError('pos must be an array of shape (N, 10)')
    t = pos[:, 0]
    if not np.array_equal(t, np.unique(t)):
        raise ValueError('Time points must unique and in ascending order')
    # We need an extra 1e-3 (1 ms) here because MaxFilter outputs values
    # only out to 3 decimal places
    if not _time_mask(t, tmin=raw._first_time - 1e-3, tmax=None,
                      sfreq=sfreq).all():
        raise ValueError('Head position time points must be greater than '
                         'first sample offset, but found %0.4f < %0.4f'
                         % (t[0], raw._first_time))
    max_dist = np.sqrt(np.sum(pos[:, 4:7] ** 2, axis=1)).max()
    if max_dist > 1.:
        warn('Found a distance greater than 1 m (%0.3g m) from the device '
             'origin, positions may be invalid and Maxwell filtering could '
             'fail' % (max_dist,))
    dev_head_ts = np.zeros((len(t), 4, 4))
    dev_head_ts[:, 3, 3] = 1.
    dev_head_ts[:, :3, 3] = pos[:, 4:7]
    dev_head_ts[:, :3, :3] = quat_to_rot(pos[:, 1:4])
    pos = [dev_head_ts, t - raw._first_time, pos[:, 1:]]
    return pos


def _get_decomp(trans, all_coils, cal, regularize, exp, ignore_ref,
                coil_scale, grad_picks, mag_picks, good_mask, mag_or_fine,
                bad_condition, t, mag_scale):
    """Get a decomposition matrix and pseudoinverse matrices."""
    #
    # Fine calibration processing (point-like magnetometers and calib. coeffs)
    #
    S_decomp_full = _get_s_decomp(
        exp, all_coils, trans, coil_scale, cal, ignore_ref, grad_picks,
        mag_picks, mag_scale)
    S_decomp = S_decomp_full[good_mask]
    #
    # Extended SSS basis (eSSS)
    #
    extended_proj = exp.get('extended_proj', ())
    if len(extended_proj) > 0:
        rcond = 1e-4
        thresh = 1e-4
        extended_proj = extended_proj.T * coil_scale[good_mask]
        extended_proj /= np.linalg.norm(extended_proj, axis=0)
        n_int = _get_n_moments(exp['int_order'])
        if S_decomp.shape[1] > n_int:
            S_ext = S_decomp[:, n_int:].copy()
            S_ext /= np.linalg.norm(S_ext, axis=0)
            S_ext_orth = linalg.orth(S_ext, rcond=rcond)
            assert S_ext_orth.shape[1] == S_ext.shape[1]
            extended_proj -= np.dot(S_ext_orth,
                                    np.dot(S_ext_orth.T, extended_proj))
            scale = np.mean(np.linalg.norm(S_decomp[n_int:], axis=0))
        else:
            scale = np.mean(np.linalg.norm(S_decomp[:n_int], axis=0))
        mask = np.linalg.norm(extended_proj, axis=0) > thresh
        extended_remove = list(np.where(~mask)[0] + S_decomp.shape[1])
        logger.debug('    Reducing %d -> %d'
                     % (extended_proj.shape[1], mask.sum()))
        extended_proj /= np.linalg.norm(extended_proj, axis=0) / scale
        S_decomp = np.concatenate([S_decomp, extended_proj], axis=-1)
        if extended_proj.shape[1]:
            S_decomp_full = np.pad(
                S_decomp_full, ((0, 0), (0, extended_proj.shape[1])),
                'constant')
            S_decomp_full[good_mask, -extended_proj.shape[1]:] = extended_proj
    else:
        extended_remove = list()
    del extended_proj

    #
    # Regularization
    #
    S_decomp, reg_moments, n_use_in = _regularize(
        regularize, exp, S_decomp, mag_or_fine, extended_remove, t=t)
    S_decomp_full = S_decomp_full.take(reg_moments, axis=1)

    #
    # Pseudo-inverse of total multipolar moment basis set (Part of Eq. 37)
    #
    pS_decomp, sing = _col_norm_pinv(S_decomp.copy())
    cond = sing[0] / sing[-1]
    if bad_condition != 'ignore' and cond >= 1000.:
        msg = 'Matrix is badly conditioned: %0.0f >= 1000' % cond
        if bad_condition == 'error':
            raise RuntimeError(msg)
        elif bad_condition == 'warning':
            warn(msg)
        else:  # condition == 'info'
            logger.info(msg)

    # Build in our data scaling here
    pS_decomp *= coil_scale[good_mask].T
    S_decomp /= coil_scale[good_mask]
    S_decomp_full /= coil_scale
    return S_decomp, S_decomp_full, pS_decomp, reg_moments, n_use_in


def _get_s_decomp(exp, all_coils, trans, coil_scale, cal, ignore_ref,
                  grad_picks, mag_picks, mag_scale):
    """Get S_decomp."""
    S_decomp = _trans_sss_basis(exp, all_coils, trans, coil_scale)
    if cal is not None:
        # Compute point-like mags to incorporate gradiometer imbalance
        grad_cals = _sss_basis_point(exp, trans, cal, ignore_ref, mag_scale)
        # Add point like magnetometer data to bases.
        if len(grad_picks) > 0:
            S_decomp[grad_picks, :] += grad_cals
        # Scale magnetometers by calibration coefficient
        if len(mag_picks) > 0:
            S_decomp[mag_picks, :] /= cal['mag_cals']
        # We need to be careful about KIT gradiometers
    return S_decomp


@verbose
def _regularize(regularize, exp, S_decomp, mag_or_fine, extended_remove, t,
                verbose=None):
    """Regularize a decomposition matrix."""
    # ALWAYS regularize the out components according to norm, since
    # gradiometer-only setups (e.g., KIT) can have zero first-order
    # (homogeneous field) components
    int_order, ext_order = exp['int_order'], exp['ext_order']
    n_in = _get_n_moments(int_order)
    n_out = S_decomp.shape[1] - n_in
    t_str = '%8.3f' % t
    if regularize is not None:  # regularize='in'
        in_removes, out_removes = _regularize_in(
            int_order, ext_order, S_decomp, mag_or_fine, extended_remove)
    else:
        in_removes = []
        out_removes = _regularize_out(int_order, ext_order, mag_or_fine,
                                      extended_remove)
    reg_in_moments = np.setdiff1d(np.arange(n_in), in_removes)
    reg_out_moments = np.setdiff1d(np.arange(n_in, S_decomp.shape[1]),
                                   out_removes)
    n_use_in = len(reg_in_moments)
    n_use_out = len(reg_out_moments)
    reg_moments = np.concatenate((reg_in_moments, reg_out_moments))
    S_decomp = S_decomp.take(reg_moments, axis=1)
    if regularize is not None or n_use_out != n_out:
        logger.info('        Using %s/%s harmonic components for %s  '
                    '(%s/%s in, %s/%s out)'
                    % (n_use_in + n_use_out, n_in + n_out, t_str,
                       n_use_in, n_in, n_use_out, n_out))
    return S_decomp, reg_moments, n_use_in


@verbose
def _get_mf_picks_fix_mags(info, int_order, ext_order, ignore_ref=False,
                           verbose=None):
    """Pick types for Maxwell filtering and fix magnetometers."""
    # Check for T1/T2 mag types
    mag_inds_T1T2 = _get_T1T2_mag_inds(info, use_cal=True)
    if len(mag_inds_T1T2) > 0:
        fix_mag_coil_types(info, use_cal=True)
    # Get indices of channels to use in multipolar moment calculation
    ref = not ignore_ref
    meg_picks = pick_types(info, meg=True, ref_meg=ref, exclude=[])
    meg_info = pick_info(_simplify_info(info), meg_picks)
    del info
    good_mask = np.zeros(len(meg_picks,), bool)
    good_mask[pick_types(meg_info, meg=True, ref_meg=ref, exclude='bads')] = 1
    n_bases = _get_n_moments([int_order, ext_order]).sum()
    if n_bases > good_mask.sum():
        raise ValueError('Number of requested bases (%s) exceeds number of '
                         'good sensors (%s)' % (str(n_bases), good_mask.sum()))
    recons = [ch for ch in meg_info['bads']]
    if len(recons) > 0:
        msg = '    Bad MEG channels being reconstructed: %s' % recons
    else:
        msg = '    No bad MEG channels'
    logger.info(msg)
    ref_meg = False if ignore_ref else 'mag'
    mag_picks = pick_types(meg_info, meg='mag', ref_meg=ref_meg, exclude=[])
    ref_meg = False if ignore_ref else 'grad'
    grad_picks = pick_types(meg_info, meg='grad', ref_meg=ref_meg, exclude=[])
    assert len(mag_picks) + len(grad_picks) == len(meg_info['ch_names'])
    # Determine which are magnetometers for external basis purposes
    mag_or_fine = np.zeros(len(meg_picks), bool)
    mag_or_fine[mag_picks] = True
    # KIT gradiometers are marked as having units T, not T/M (argh)
    # We need a separate variable for this because KIT grads should be
    # treated mostly like magnetometers (e.g., scaled by 100) for reg
    coil_types = np.array([ch['coil_type'] for ch in meg_info['chs']])
    mag_or_fine[(coil_types & 0xFFFF) == FIFF.FIFFV_COIL_KIT_GRAD] = False
    # The same thing goes for CTF gradiometers...
    ctf_grads = [FIFF.FIFFV_COIL_CTF_GRAD,
                 FIFF.FIFFV_COIL_CTF_REF_GRAD,
                 FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD]
    mag_or_fine[np.in1d(coil_types, ctf_grads)] = False
    msg = ('    Processing %s gradiometers and %s magnetometers'
           % (len(grad_picks), len(mag_picks)))
    n_kit = len(mag_picks) - mag_or_fine.sum()
    if n_kit > 0:
        msg += ' (of which %s are actually KIT gradiometers)' % n_kit
    logger.info(msg)
    return meg_picks, mag_picks, grad_picks, good_mask, mag_or_fine


def _check_regularize(regularize):
    """Ensure regularize is valid."""
    if not (regularize is None or (isinstance(regularize, str) and
                                   regularize in ('in',))):
        raise ValueError('regularize must be None or "in"')


def _check_usable(inst):
    """Ensure our data are clean."""
    if inst.proj:
        raise RuntimeError('Projectors cannot be applied to data during '
                           'Maxwell filtering.')
    current_comp = inst.compensation_grade
    if current_comp not in (0, None):
        raise RuntimeError('Maxwell filter cannot be done on compensated '
                           'channels, but data have been compensated with '
                           'grade %s.' % current_comp)


def _col_norm_pinv(x):
    """Compute the pinv with column-normalization to stabilize calculation.

    Note: will modify/overwrite x.
    """
    norm = np.sqrt(np.sum(x * x, axis=0))
    x /= norm
    u, s, v = _safe_svd(x, full_matrices=False, **check_disable)
    v /= norm
    return np.dot(v.T * 1. / s, u.T), s


def _sq(x):
    """Square quickly."""
    return x * x


def _check_finite(data):
    """Ensure data is finite."""
    if not np.isfinite(data).all():
        raise RuntimeError('data contains non-finite numbers')


def _sph_harm_norm(order, degree):
    """Compute normalization factor for spherical harmonics."""
    # we could use scipy.special.poch(degree + order + 1, -2 * order)
    # here, but it's slower for our fairly small degree
    norm = np.sqrt((2 * degree + 1.) / (4 * np.pi))
    if order != 0:
        norm *= np.sqrt(factorial(degree - order) /
                        float(factorial(degree + order)))
    return norm


def _concatenate_sph_coils(coils):
    """Concatenate MEG coil parameters for spherical harmoncs."""
    rs = np.concatenate([coil['r0_exey'] for coil in coils])
    wcoils = np.concatenate([coil['w'] for coil in coils])
    ezs = np.concatenate([np.tile(coil['ez'][np.newaxis, :],
                                  (len(coil['rmag']), 1))
                          for coil in coils])
    bins = np.repeat(np.arange(len(coils)),
                     [len(coil['rmag']) for coil in coils])
    return rs, wcoils, ezs, bins


_mu_0 = 4e-7 * np.pi  # magnetic permeability


def _get_mag_mask(coils):
    """Get the coil_scale for Maxwell filtering."""
    return np.array([coil['coil_class'] == FWD.COILC_MAG for coil in coils])


def _sss_basis_basic(exp, coils, mag_scale=100., method='standard'):
    """Compute SSS basis using non-optimized (but more readable) algorithms."""
    from scipy.special import sph_harm
    int_order, ext_order = exp['int_order'], exp['ext_order']
    origin = exp['origin']
    assert 'extended_proj' not in exp  # advanced option not supported
    # Compute vector between origin and coil, convert to spherical coords
    if method == 'standard':
        # Get position, normal, weights, and number of integration pts.
        rmags, cosmags, ws, bins = _concatenate_coils(coils)
        rmags -= origin
        # Convert points to spherical coordinates
        rad, az, pol = _cart_to_sph(rmags).T
        cosmags *= ws[:, np.newaxis]
        del rmags, ws
        out_type = np.float64
    else:  # testing equivalence method
        rs, wcoils, ezs, bins = _concatenate_sph_coils(coils)
        rs -= origin
        rad, az, pol = _cart_to_sph(rs).T
        ezs *= wcoils[:, np.newaxis]
        del rs, wcoils
        out_type = np.complex128
    del origin

    # Set up output matrices
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_tot = np.empty((len(coils), n_in + n_out), out_type)
    S_in = S_tot[:, :n_in]
    S_out = S_tot[:, n_in:]
    coil_scale = np.ones((len(coils), 1))
    coil_scale[_get_mag_mask(coils)] = mag_scale

    # Compute internal/external basis vectors (exclude degree 0; L/RHS Eq. 5)
    for degree in range(1, max(int_order, ext_order) + 1):
        # Only loop over positive orders, negative orders are handled
        # for efficiency within
        for order in range(degree + 1):
            S_in_out = list()
            grads_in_out = list()
            # Same spherical harmonic is used for both internal and external
            sph = sph_harm(order, degree, az, pol)
            sph_norm = _sph_harm_norm(order, degree)
            # Compute complex gradient for all integration points
            # in spherical coordinates (Eq. 6). The gradient for rad, az, pol
            # is obtained by taking the partial derivative of Eq. 4 w.r.t. each
            # coordinate.
            az_factor = 1j * order * sph / np.sin(np.maximum(pol, 1e-16))
            pol_factor = (-sph_norm * np.sin(pol) * np.exp(1j * order * az) *
                          _alegendre_deriv(order, degree, np.cos(pol)))
            if degree <= int_order:
                S_in_out.append(S_in)
                in_norm = _mu_0 * rad ** -(degree + 2)
                g_rad = in_norm * (-(degree + 1.) * sph)
                g_az = in_norm * az_factor
                g_pol = in_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            if degree <= ext_order:
                S_in_out.append(S_out)
                out_norm = _mu_0 * rad ** (degree - 1)
                g_rad = out_norm * degree * sph
                g_az = out_norm * az_factor
                g_pol = out_norm * pol_factor
                grads_in_out.append(_sph_to_cart_partials(az, pol,
                                                          g_rad, g_az, g_pol))
            for spc, grads in zip(S_in_out, grads_in_out):
                # We could convert to real at the end, but it's more efficient
                # to do it now
                if method == 'standard':
                    grads_pos_neg = [_sh_complex_to_real(grads, order)]
                    orders_pos_neg = [order]
                    # Deal with the negative orders
                    if order > 0:
                        # it's faster to use the conjugation property for
                        # our normalized spherical harmonics than recalculate
                        grads_pos_neg.append(_sh_complex_to_real(
                            _sh_negate(grads, order), -order))
                        orders_pos_neg.append(-order)
                    for gr, oo in zip(grads_pos_neg, orders_pos_neg):
                        # Gradients dotted w/integration point weighted normals
                        gr = einsum('ij,ij->i', gr, cosmags)
                        vals = np.bincount(bins, gr, len(coils))
                        spc[:, _deg_ord_idx(degree, oo)] = -vals
                else:
                    grads = einsum('ij,ij->i', grads, ezs)
                    v = (np.bincount(bins, grads.real, len(coils)) +
                         1j * np.bincount(bins, grads.imag, len(coils)))
                    spc[:, _deg_ord_idx(degree, order)] = -v
                    if order > 0:
                        spc[:, _deg_ord_idx(degree, -order)] = \
                            -_sh_negate(v, order)

    # Scale magnetometers
    S_tot *= coil_scale
    if method != 'standard':
        # Eventually we could probably refactor this for 2x mem (and maybe CPU)
        # savings by changing how spc/S_tot is assigned above (real only)
        S_tot = _bases_complex_to_real(S_tot, int_order, ext_order)
    return S_tot


def _sss_basis(exp, all_coils):
    """Compute SSS basis for given conditions.

    Parameters
    ----------
    exp : dict
        Must contain the following keys:

            origin : ndarray, shape (3,)
                Origin of the multipolar moment space in meters
            int_order : int
                Order of the internal multipolar moment space
            ext_order : int
                Order of the external multipolar moment space

    coils : list
        List of MEG coils. Each should contain coil information dict specifying
        position, normals, weights, number of integration points and channel
        type. All coil geometry must be in the same coordinate frame
        as ``origin`` (``head`` or ``meg``).

    Returns
    -------
    bases : ndarray, shape (n_coils, n_mult_moments)
        Internal and external basis sets as a single ndarray.

    Notes
    -----
    Does not incorporate magnetometer scaling factor or normalize spaces.

    Adapted from code provided by Jukka Nenonen.
    """
    rmags, cosmags, bins, n_coils = all_coils[:4]
    int_order, ext_order = exp['int_order'], exp['ext_order']
    n_in, n_out = _get_n_moments([int_order, ext_order])
    rmags = rmags - exp['origin']

    # do the heavy lifting
    max_order = max(int_order, ext_order)
    L = _tabular_legendre(rmags, max_order)
    phi = np.arctan2(rmags[:, 1], rmags[:, 0])
    r_n = np.sqrt(np.sum(rmags * rmags, axis=1))
    r_xy = np.sqrt(rmags[:, 0] * rmags[:, 0] + rmags[:, 1] * rmags[:, 1])
    cos_pol = rmags[:, 2] / r_n  # cos(theta); theta 0...pi
    sin_pol = np.sqrt(1. - cos_pol * cos_pol)  # sin(theta)
    z_only = (r_xy <= 1e-16)
    sin_pol_nz = sin_pol.copy()
    sin_pol_nz[z_only] = 1.  # will be overwritten later
    r_xy[z_only] = 1.
    cos_az = rmags[:, 0] / r_xy  # cos(phi)
    cos_az[z_only] = 1.
    sin_az = rmags[:, 1] / r_xy  # sin(phi)
    sin_az[z_only] = 0.
    # Appropriate vector spherical harmonics terms
    #  JNE 2012-02-08: modified alm -> 2*alm, blm -> -2*blm
    r_nn2 = r_n.copy()
    r_nn1 = 1.0 / (r_n * r_n)
    S_tot = np.empty((n_coils, n_in + n_out), np.float64)
    S_in = S_tot[:, :n_in]
    S_out = S_tot[:, n_in:]
    for degree in range(max_order + 1):
        if degree <= ext_order:
            r_nn1 *= r_n  # r^(l-1)
        if degree <= int_order:
            r_nn2 *= r_n  # r^(l+2)

        # mu_0*sqrt((2l+1)/4pi (l-m)!/(l+m)!)
        mult = 2e-7 * np.sqrt((2 * degree + 1) * np.pi)

        if degree > 0:
            idx = _deg_ord_idx(degree, 0)
            # alpha
            if degree <= int_order:
                b_r = mult * (degree + 1) * L[degree][0] / r_nn2
                b_pol = -mult * L[degree][1] / r_nn2
                S_in[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, 0., b_pol,
                    cosmags, bins, n_coils)
            # beta
            if degree <= ext_order:
                b_r = -mult * degree * L[degree][0] * r_nn1
                b_pol = -mult * L[degree][1] * r_nn1
                S_out[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, 0., b_pol,
                    cosmags, bins, n_coils)
        for order in range(1, degree + 1):
            ord_phi = order * phi
            sin_order = np.sin(ord_phi)
            cos_order = np.cos(ord_phi)
            mult /= np.sqrt((degree - order + 1) * (degree + order))
            factor = mult * np.sqrt(2)  # equivalence fix (MF uses 2.)

            # Real
            idx = _deg_ord_idx(degree, order)
            r_fact = factor * L[degree][order] * cos_order
            az_fact = factor * order * sin_order * L[degree][order]
            pol_fact = -factor * (L[degree][order + 1] -
                                  (degree + order) * (degree - order + 1) *
                                  L[degree][order - 1]) * cos_order
            # alpha
            if degree <= int_order:
                b_r = (degree + 1) * r_fact / r_nn2
                b_az = az_fact / (sin_pol_nz * r_nn2)
                b_az[z_only] = 0.
                b_pol = pol_fact / (2 * r_nn2)
                S_in[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol,
                    cosmags, bins, n_coils)
            # beta
            if degree <= ext_order:
                b_r = -degree * r_fact * r_nn1
                b_az = az_fact * r_nn1 / sin_pol_nz
                b_az[z_only] = 0.
                b_pol = pol_fact * r_nn1 / 2.
                S_out[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol,
                    cosmags, bins, n_coils)

            # Imaginary
            idx = _deg_ord_idx(degree, -order)
            r_fact = factor * L[degree][order] * sin_order
            az_fact = factor * order * cos_order * L[degree][order]
            pol_fact = factor * (L[degree][order + 1] -
                                 (degree + order) * (degree - order + 1) *
                                 L[degree][order - 1]) * sin_order
            # alpha
            if degree <= int_order:
                b_r = -(degree + 1) * r_fact / r_nn2
                b_az = az_fact / (sin_pol_nz * r_nn2)
                b_az[z_only] = 0.
                b_pol = pol_fact / (2 * r_nn2)
                S_in[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol,
                    cosmags, bins, n_coils)
            # beta
            if degree <= ext_order:
                b_r = degree * r_fact * r_nn1
                b_az = az_fact * r_nn1 / sin_pol_nz
                b_az[z_only] = 0.
                b_pol = pol_fact * r_nn1 / 2.
                S_out[:, idx] = _integrate_points(
                    cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol,
                    cosmags, bins, n_coils)
    return S_tot


def _integrate_points(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol,
                      cosmags, bins, n_coils):
    """Integrate points in spherical coords."""
    grads = _sp_to_cart(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol).T
    grads = (grads * cosmags).sum(axis=1)
    return bincount(bins, grads, n_coils)


def _tabular_legendre(r, nind):
    """Compute associated Legendre polynomials."""
    r_n = np.sqrt(np.sum(r * r, axis=1))
    x = r[:, 2] / r_n  # cos(theta)
    L = list()
    for degree in range(nind + 1):
        L.append(np.zeros((degree + 2, len(r))))
    L[0][0] = 1.
    pnn = np.ones(x.shape)
    fact = 1.
    sx2 = np.sqrt((1. - x) * (1. + x))
    for degree in range(nind + 1):
        L[degree][degree] = pnn
        pnn *= (-fact * sx2)
        fact += 2.
        if degree < nind:
            L[degree + 1][degree] = x * (2 * degree + 1) * L[degree][degree]
        if degree >= 2:
            for order in range(degree - 1):
                L[degree][order] = (x * (2 * degree - 1) *
                                    L[degree - 1][order] -
                                    (degree + order - 1) *
                                    L[degree - 2][order]) / (degree - order)
    return L


def _sp_to_cart(cos_az, sin_az, cos_pol, sin_pol, b_r, b_az, b_pol):
    """Convert spherical coords to cartesian."""
    out = np.empty((3,) + sin_pol.shape)
    out[0] = sin_pol * cos_az * b_r + cos_pol * cos_az * b_pol - sin_az * b_az
    out[1] = sin_pol * sin_az * b_r + cos_pol * sin_az * b_pol + cos_az * b_az
    out[2] = cos_pol * b_r - sin_pol * b_pol
    return out


def _get_degrees_orders(order):
    """Get the set of degrees used in our basis functions."""
    degrees = np.zeros(_get_n_moments(order), int)
    orders = np.zeros_like(degrees)
    for degree in range(1, order + 1):
        # Only loop over positive orders, negative orders are handled
        # for efficiency within
        for order in range(degree + 1):
            ii = _deg_ord_idx(degree, order)
            degrees[ii] = degree
            orders[ii] = order
            ii = _deg_ord_idx(degree, -order)
            degrees[ii] = degree
            orders[ii] = -order
    return degrees, orders


def _alegendre_deriv(order, degree, val):
    """Compute the derivative of the associated Legendre polynomial at a value.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    val : float
        Value to evaluate the derivative at.

    Returns
    -------
    dPlm : float
        Associated Legendre function derivative
    """
    from scipy.special import lpmv
    assert order >= 0
    return (order * val * lpmv(order, degree, val) + (degree + order) *
            (degree - order + 1.) * np.sqrt(1. - val * val) *
            lpmv(order - 1, degree, val)) / (1. - val * val)


def _bases_complex_to_real(complex_tot, int_order, ext_order):
    """Convert complex spherical harmonics to real."""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    complex_in = complex_tot[:, :n_in]
    complex_out = complex_tot[:, n_in:]
    real_tot = np.empty(complex_tot.shape, np.float64)
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    for comp, real, exp_order in zip([complex_in, complex_out],
                                     [real_in, real_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            for order in range(deg + 1):
                idx_pos = _deg_ord_idx(deg, order)
                idx_neg = _deg_ord_idx(deg, -order)
                real[:, idx_pos] = _sh_complex_to_real(comp[:, idx_pos], order)
                if order != 0:
                    # This extra mult factor baffles me a bit, but it works
                    # in round-trip testing, so we'll keep it :(
                    mult = (-1 if order % 2 == 0 else 1)
                    real[:, idx_neg] = mult * _sh_complex_to_real(
                        comp[:, idx_neg], -order)
    return real_tot


def _bases_real_to_complex(real_tot, int_order, ext_order):
    """Convert real spherical harmonics to complex."""
    n_in, n_out = _get_n_moments([int_order, ext_order])
    real_in = real_tot[:, :n_in]
    real_out = real_tot[:, n_in:]
    comp_tot = np.empty(real_tot.shape, np.complex128)
    comp_in = comp_tot[:, :n_in]
    comp_out = comp_tot[:, n_in:]
    for real, comp, exp_order in zip([real_in, real_out],
                                     [comp_in, comp_out],
                                     [int_order, ext_order]):
        for deg in range(1, exp_order + 1):
            # only loop over positive orders, figure out neg from pos
            for order in range(deg + 1):
                idx_pos = _deg_ord_idx(deg, order)
                idx_neg = _deg_ord_idx(deg, -order)
                this_comp = _sh_real_to_complex([real[:, idx_pos],
                                                 real[:, idx_neg]], order)
                comp[:, idx_pos] = this_comp
                comp[:, idx_neg] = _sh_negate(this_comp, order)
    return comp_tot


def _check_info(info, sss=True, tsss=True, calibration=True, ctc=True):
    """Ensure that Maxwell filtering has not been applied yet."""
    for ent in info['proc_history']:
        for msg, key, doing in (('SSS', 'sss_info', sss),
                                ('tSSS', 'max_st', tsss),
                                ('fine calibration', 'sss_cal', calibration),
                                ('cross-talk cancellation', 'sss_ctc', ctc)):
            if not doing:
                continue
            if len(ent['max_info'][key]) > 0:
                raise RuntimeError('Maxwell filtering %s step has already '
                                   'been applied, cannot reapply' % msg)


def _update_sss_info(raw, origin, int_order, ext_order, nchan, coord_frame,
                     sss_ctc, sss_cal, max_st, reg_moments, st_only,
                     recon_trans, extended_proj):
    """Update info inplace after Maxwell filtering.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Data to be filtered
    origin : array-like, shape (3,)
        Origin of internal and external multipolar moment space in head coords
        (in meters)
    int_order : int
        Order of internal component of spherical expansion
    ext_order : int
        Order of external component of spherical expansion
    nchan : int
        Number of sensors
    sss_ctc : dict
        The cross talk information.
    sss_cal : dict
        The calibration information.
    max_st : dict
        The tSSS information.
    reg_moments : ndarray | slice
        The moments that were used.
    st_only : bool
        Whether tSSS only was performed.
    recon_trans : instance of Transformation
        The reconstruction trans.
    extended_proj : ndarray
        Extended external bases.
    """
    n_in, n_out = _get_n_moments([int_order, ext_order])
    raw.info['maxshield'] = False
    components = np.zeros(n_in + n_out + len(extended_proj)).astype('int32')
    components[reg_moments] = 1
    sss_info_dict = dict(in_order=int_order, out_order=ext_order,
                         nchan=nchan, origin=origin.astype('float32'),
                         job=FIFF.FIFFV_SSS_JOB_FILTER,
                         nfree=np.sum(components[:n_in]),
                         frame=_str_to_frame[coord_frame],
                         components=components)
    max_info_dict = dict(max_st=max_st)
    if st_only:
        max_info_dict.update(sss_info=dict(), sss_cal=dict(), sss_ctc=dict())
    else:
        max_info_dict.update(sss_info=sss_info_dict, sss_cal=sss_cal,
                             sss_ctc=sss_ctc)
        # Reset 'bads' for any MEG channels since they've been reconstructed
        _reset_meg_bads(raw.info)
        # set the reconstruction transform
        raw.info['dev_head_t'] = recon_trans
    block_id = _generate_meas_id()
    raw.info['proc_history'].insert(0, dict(
        max_info=max_info_dict, block_id=block_id, date=DATE_NONE,
        creator='mne-python v%s' % __version__, experimenter=''))


def _reset_meg_bads(info):
    """Reset MEG bads."""
    meg_picks = pick_types(info, meg=True, exclude=[])
    info['bads'] = [bad for bad in info['bads']
                    if info['ch_names'].index(bad) not in meg_picks]


check_disable = dict()  # not available on really old versions of SciPy
if 'check_finite' in _get_args(linalg.svd):
    check_disable['check_finite'] = False


def _orth_overwrite(A):
    """Create a slightly more efficient 'orth'."""
    # adapted from scipy/linalg/decomp_svd.py
    u, s = _safe_svd(A, full_matrices=False, **check_disable)[:2]
    M, N = A.shape
    eps = np.finfo(float).eps
    tol = max(M, N) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    return u[:, :num]


def _overlap_projector(data_int, data_res, corr):
    """Calculate projector for removal of subspace intersection in tSSS."""
    # corr necessary to deal with noise when finding identical signal
    # directions in the subspace. See the end of the Results section in [2]_

    # Note that the procedure here is an updated version of [2]_ (and used in
    # MF's tSSS) that uses residuals instead of internal/external spaces
    # directly. This provides more degrees of freedom when analyzing for
    # intersections between internal and external spaces.

    # Normalize data, then compute orth to get temporal bases. Matrices
    # must have shape (n_samps x effective_rank) when passed into svd
    # computation

    # we use np.linalg.norm instead of sp.linalg.norm here: ~2x faster!
    n = np.linalg.norm(data_int)
    n = 1. if n == 0 else n  # all-zero data should gracefully continue
    data_int = _orth_overwrite((data_int / n).T)
    n = np.linalg.norm(data_res)
    n = 1. if n == 0 else n
    data_res = _orth_overwrite((data_res / n).T)
    if data_int.shape[1] == 0 or data_res.shape[1] == 0:
        return np.empty((data_int.shape[0], 0))
    Q_int = linalg.qr(data_int,
                      overwrite_a=True, mode='economic', **check_disable)[0].T
    Q_res = linalg.qr(data_res,
                      overwrite_a=True, mode='economic', **check_disable)[0]
    C_mat = np.dot(Q_int, Q_res)
    del Q_int

    # Compute angles between subspace and which bases to keep
    S_intersect, Vh_intersect = _safe_svd(C_mat, full_matrices=False,
                                          **check_disable)[1:]
    del C_mat
    intersect_mask = (S_intersect >= corr)
    del S_intersect

    # Compute projection operator as (I-LL_T) Eq. 12 in [2]_
    # V_principal should be shape (n_time_pts x n_retained_inds)
    Vh_intersect = Vh_intersect[intersect_mask].T
    V_principal = np.dot(Q_res, Vh_intersect)
    return V_principal


def _prep_fine_cal(info, fine_cal):
    from ._fine_cal import read_fine_calibration
    _validate_type(fine_cal, (dict, 'path-like'))
    if not isinstance(fine_cal, dict):
        extra = op.basename(str(fine_cal))
        fine_cal = read_fine_calibration(fine_cal)
    else:
        extra = 'dict'
    logger.info(f'    Using fine calibration {extra}')
    ch_names = _clean_names(info['ch_names'], remove_whitespace=True)
    info_to_cal = OrderedDict()
    missing = list()
    for ci, name in enumerate(fine_cal['ch_names']):
        if name not in ch_names:
            missing.append(name)
        else:
            oi = ch_names.index(name)
            info_to_cal[oi] = ci
    meg_picks = pick_types(info, meg=True, exclude=[])
    if len(info_to_cal) != len(meg_picks):
        raise RuntimeError(
            'Not all MEG channels found in fine calibration file, missing:\n%s'
            % sorted(list({ch_names[pick] for pick in meg_picks} -
                          set(fine_cal['ch_names']))))
    if len(missing):
        warn('Found cal channel%s not in data: %s' % (_pl(missing), missing))
    return info_to_cal, fine_cal, ch_names


def _update_sensor_geometry(info, fine_cal, ignore_ref):
    """Replace sensor geometry information and reorder cal_chs."""
    info_to_cal, fine_cal, ch_names = _prep_fine_cal(info, fine_cal)
    grad_picks = pick_types(info, meg='grad', exclude=())
    mag_picks = pick_types(info, meg='mag', exclude=())

    # Determine gradiometer imbalances and magnetometer calibrations
    grad_imbalances = np.array([fine_cal['imb_cals'][info_to_cal[gi]]
                                for gi in grad_picks]).T
    if grad_imbalances.shape[0] not in [0, 1, 3]:
        raise ValueError('Must have 1 (x) or 3 (x, y, z) point-like ' +
                         'magnetometers. Currently have %i' %
                         grad_imbalances.shape[0])
    mag_cals = np.array([fine_cal['imb_cals'][info_to_cal[mi]]
                         for mi in mag_picks])
    # Now let's actually construct our point-like adjustment coils for grads
    grad_coilsets = _get_grad_point_coilsets(
        info, n_types=len(grad_imbalances), ignore_ref=ignore_ref)
    calibration = dict(grad_imbalances=grad_imbalances,
                       grad_coilsets=grad_coilsets, mag_cals=mag_cals)

    # Replace sensor locations (and track differences) for fine calibration
    ang_shift = list()
    used = np.zeros(len(info['chs']), bool)
    cal_corrs = list()
    cal_chans = list()
    adjust_logged = False
    for oi, ci in info_to_cal.items():
        assert not used[oi]
        used[oi] = True
        info_ch = info['chs'][oi]
        ch_num = int(fine_cal['ch_names'][ci].lstrip('MEG').lstrip('0'))
        cal_chans.append([ch_num, info_ch['coil_type']])

        # Some .dat files might only rotate EZ, so we must check first that
        # EX and EY are orthogonal to EZ. If not, we find the rotation between
        # the original and fine-cal ez, and rotate EX and EY accordingly:
        ch_coil_rot = _loc_to_coil_trans(info_ch['loc'])[:3, :3]
        cal_loc = fine_cal['locs'][ci].copy()
        cal_coil_rot = _loc_to_coil_trans(cal_loc)[:3, :3]
        if np.max([np.abs(np.dot(cal_coil_rot[:, ii], cal_coil_rot[:, 2]))
                   for ii in range(2)]) > 1e-6:  # X or Y not orthogonal
            if not adjust_logged:
                logger.info('        Adjusting non-orthogonal EX and EY')
                adjust_logged = True
            # find the rotation matrix that goes from one to the other
            this_trans = _find_vector_rotation(
                ch_coil_rot[:, 2], cal_coil_rot[:, 2])
            cal_loc[3:] = np.dot(this_trans, ch_coil_rot).T.ravel()

        # calculate shift angle
        v1 = _loc_to_coil_trans(cal_loc)[:3, :3]
        _normalize_vectors(v1)
        v2 = _loc_to_coil_trans(info_ch['loc'])[:3, :3]
        _normalize_vectors(v2)
        ang_shift.append(np.sum(v1 * v2, axis=0))
        if oi in grad_picks:
            extra = [1., fine_cal['imb_cals'][ci][0]]
        else:
            extra = [fine_cal['imb_cals'][ci][0], 0.]
        cal_corrs.append(np.concatenate([extra, cal_loc]))
        # Adjust channel normal orientations with those from fine calibration
        # Channel positions are not changed
        info_ch['loc'][3:] = cal_loc[3:]
        assert (info_ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE)
    meg_picks = pick_types(info, meg=True, exclude=())
    assert used[meg_picks].all()
    assert not used[np.setdiff1d(np.arange(len(used)), meg_picks)].any()
    # This gets written to the Info struct
    sss_cal = dict(cal_corrs=np.array(cal_corrs),
                   cal_chans=np.array(cal_chans))

    # Log quantification of sensor changes
    # Deal with numerical precision giving absolute vals slightly more than 1.
    ang_shift = np.array(ang_shift)
    np.clip(ang_shift, -1., 1., ang_shift)
    np.rad2deg(np.arccos(ang_shift), ang_shift)  # Convert to degrees
    logger.info('        Adjusted coil positions by (μ ± σ): '
                '%0.1f° ± %0.1f° (max: %0.1f°)' %
                (np.mean(ang_shift), np.std(ang_shift),
                 np.max(np.abs(ang_shift))))
    return calibration, sss_cal


def _get_grad_point_coilsets(info, n_types, ignore_ref):
    """Get point-type coilsets for gradiometers."""
    _rotations = dict(
        x=np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]]),
        y=np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]]),
        z=np.eye(4))
    grad_coilsets = list()
    grad_picks = pick_types(info, meg='grad', exclude=[])
    if len(grad_picks) == 0:
        return grad_coilsets
    grad_info = pick_info(_simplify_info(info), grad_picks)
    # Coil_type values for x, y, z point magnetometers
    # Note: 1D correction files only have x-direction corrections
    for ch in grad_info['chs']:
        ch['coil_type'] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
    orig_locs = [ch['loc'].copy() for ch in grad_info['chs']]
    for rot in 'xyz'[:n_types]:
        # Rotate the Z magnetometer orientation to the destination orientation
        for ci, ch in enumerate(grad_info['chs']):
            ch['loc'][3:] = _coil_trans_to_loc(np.dot(
                _loc_to_coil_trans(orig_locs[ci]),
                _rotations[rot]))[3:]
        grad_coilsets.append(_prep_mf_coils(grad_info, ignore_ref))
    return grad_coilsets


def _sss_basis_point(exp, trans, cal, ignore_ref=False, mag_scale=100.):
    """Compute multipolar moments for point-like mags (in fine cal)."""
    # Loop over all coordinate directions desired and create point mags
    S_tot = 0.
    # These are magnetometers, so use a uniform coil_scale of 100.
    this_cs = np.array([mag_scale], float)
    for imb, coils in zip(cal['grad_imbalances'], cal['grad_coilsets']):
        S_add = _trans_sss_basis(exp, coils, trans, this_cs)
        # Scale spaces by gradiometer imbalance
        S_add *= imb[:, np.newaxis]
        S_tot += S_add

    # Return point-like mag bases
    return S_tot


def _regularize_out(int_order, ext_order, mag_or_fine, extended_remove):
    """Regularize out components based on norm."""
    n_in = _get_n_moments(int_order)
    remove_homog = ext_order > 0 and not mag_or_fine.any()
    return list(range(n_in, n_in + 3 * remove_homog)) + extended_remove


def _regularize_in(int_order, ext_order, S_decomp, mag_or_fine,
                   extended_remove):
    """Regularize basis set using idealized SNR measure."""
    n_in, n_out = _get_n_moments([int_order, ext_order])

    # The "signal" terms depend only on the inner expansion order
    # (i.e., not sensor geometry or head position / expansion origin)
    a_lm_sq, rho_i = _compute_sphere_activation_in(
        np.arange(int_order + 1))
    degrees, orders = _get_degrees_orders(int_order)
    a_lm_sq = a_lm_sq[degrees]

    I_tots = np.zeros(n_in)  # we might not traverse all, so use np.zeros
    in_keepers = list(range(n_in))
    out_removes = _regularize_out(int_order, ext_order, mag_or_fine,
                                  extended_remove)
    out_keepers = list(np.setdiff1d(np.arange(n_in, S_decomp.shape[1]),
                                    out_removes))
    remove_order = []
    S_decomp = S_decomp.copy()
    use_norm = np.sqrt(np.sum(S_decomp * S_decomp, axis=0))
    S_decomp /= use_norm
    eigs = np.zeros((n_in, 2))

    # plot = False  # for debugging
    # if plot:
    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(3, figsize=[6, 12])
    #     plot_ord = np.empty(n_in, int)
    #     plot_ord.fill(-1)
    #     count = 0
    #     # Reorder plot to match MF
    #     for degree in range(1, int_order + 1):
    #         for order in range(0, degree + 1):
    #             assert plot_ord[count] == -1
    #             plot_ord[count] = _deg_ord_idx(degree, order)
    #             count += 1
    #             if order > 0:
    #                 assert plot_ord[count] == -1
    #                 plot_ord[count] = _deg_ord_idx(degree, -order)
    #                 count += 1
    #     assert count == n_in
    #     assert (plot_ord >= 0).all()
    #     assert len(np.unique(plot_ord)) == n_in
    noise_lev = 5e-13  # noise level in T/m
    noise_lev *= noise_lev  # effectively what would happen by earlier multiply
    for ii in range(n_in):
        this_S = S_decomp.take(in_keepers + out_keepers, axis=1)
        u, s, v = _safe_svd(this_S, full_matrices=False, **check_disable)
        del this_S
        eigs[ii] = s[[0, -1]]
        v = v.T[:len(in_keepers)]
        v /= use_norm[in_keepers][:, np.newaxis]
        eta_lm_sq = np.dot(v * 1. / s, u.T)
        del u, s, v
        eta_lm_sq *= eta_lm_sq
        eta_lm_sq = eta_lm_sq.sum(axis=1)
        eta_lm_sq *= noise_lev

        # Mysterious scale factors to match MF, likely due to differences
        # in the basis normalizations...
        eta_lm_sq[orders[in_keepers] == 0] *= 2
        eta_lm_sq *= 0.0025
        snr = a_lm_sq[in_keepers] / eta_lm_sq
        I_tots[ii] = 0.5 * np.log2(snr + 1.).sum()
        remove_order.append(in_keepers[np.argmin(snr)])
        in_keepers.pop(in_keepers.index(remove_order[-1]))
        # heuristic to quit if we're past the peak to save cycles
        if ii > 10 and (I_tots[ii - 1:ii + 1] < 0.95 * I_tots.max()).all():
            break
        # if plot and ii == 0:
        #     axs[0].semilogy(snr[plot_ord[in_keepers]], color='k')
    # if plot:
    #     axs[0].set(ylabel='SNR', ylim=[0.1, 500], xlabel='Component')
    #     axs[1].plot(I_tots)
    #     axs[1].set(ylabel='Information', xlabel='Iteration')
    #     axs[2].plot(eigs[:, 0] / eigs[:, 1])
    #     axs[2].set(ylabel='Condition', xlabel='Iteration')
    # Pick the components that give at least 98% of max info
    # This is done because the curves can be quite flat, and we err on the
    # side of including rather than excluding components
    max_info = np.max(I_tots)
    lim_idx = np.where(I_tots >= 0.98 * max_info)[0][0]
    in_removes = remove_order[:lim_idx]
    for ii, ri in enumerate(in_removes):
        logger.debug('            Condition %0.3f/%0.3f = %03.1f, '
                     'Removing in component %s: l=%s, m=%+0.0f'
                     % (tuple(eigs[ii]) + (eigs[ii, 0] / eigs[ii, 1],
                                           ri, degrees[ri], orders[ri])))
    logger.debug('        Resulting information: %0.1f bits/sample '
                 '(%0.1f%% of peak %0.1f)'
                 % (I_tots[lim_idx], 100 * I_tots[lim_idx] / max_info,
                    max_info))
    return in_removes, out_removes


def _compute_sphere_activation_in(degrees):
    u"""Compute the "in" power from random currents in a sphere.

    Parameters
    ----------
    degrees : ndarray
        The degrees to evaluate.

    Returns
    -------
    a_power : ndarray
        The a_lm associated for the associated degrees (see [1]_).
    rho_i : float
        The current density.

    References
    ----------
    .. [1] A 122-channel whole-cortex SQUID system for measuring the brain’s
       magnetic fields. Knuutila et al. IEEE Transactions on Magnetics,
       Vol 29 No 6, Nov 1993.
    """
    r_in = 0.080  # radius of the randomly-activated sphere

    # set the observation point r=r_s, az=el=0, so we can just look at m=0 term
    # compute the resulting current density rho_i

    # This is the "surface" version of the equation:
    # b_r_in = 100e-15  # fixed radial field amplitude at distance r_s = 100 fT
    # r_s = 0.13  # 5 cm from the surface
    # rho_degrees = np.arange(1, 100)
    # in_sum = (rho_degrees * (rho_degrees + 1.) /
    #           ((2. * rho_degrees + 1.)) *
    #           (r_in / r_s) ** (2 * rho_degrees + 2)).sum() * 4. * np.pi
    # rho_i = b_r_in * 1e7 / np.sqrt(in_sum)
    # rho_i = 5.21334885574e-07  # value for r_s = 0.125
    rho_i = 5.91107375632e-07  # deterministic from above, so just store it
    a_power = _sq(rho_i) * (degrees * r_in ** (2 * degrees + 4) /
                            (_sq(2. * degrees + 1.) *
                            (degrees + 1.)))
    return a_power, rho_i


def _trans_sss_basis(exp, all_coils, trans=None, coil_scale=100.):
    """Compute SSS basis (optionally) using a dev<->head trans."""
    if trans is not None:
        if not isinstance(trans, Transform):
            trans = Transform('meg', 'head', trans)
        assert not np.isnan(trans['trans']).any()
        all_coils = (apply_trans(trans, all_coils[0]),
                     apply_trans(trans, all_coils[1], move=False),
                     ) + all_coils[2:]
    if not isinstance(coil_scale, np.ndarray):
        # Scale all magnetometers (with `coil_class` == 1.0) by `mag_scale`
        cs = coil_scale
        coil_scale = np.ones((all_coils[3], 1))
        coil_scale[all_coils[4]] = cs
    S_tot = _sss_basis(exp, all_coils)
    S_tot *= coil_scale
    return S_tot


# intentionally omitted: st_duration, st_correlation, destination, st_fixed,
# st_only
@verbose
def find_bad_channels_maxwell(
        raw, limit=7., duration=5., min_count=5, return_scores=False,
        origin='auto', int_order=8, ext_order=3, calibration=None,
        cross_talk=None, coord_frame='head', regularize='in', ignore_ref=False,
        bad_condition='error', head_pos=None, mag_scale=100.,
        skip_by_annotation=('edge', 'bad_acq_skip'), h_freq=40.0,
        extended_proj=(), verbose=None):
    r"""Find bad channels using Maxwell filtering.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to process.
    limit : float
        Detection limit for noisy segments (default is 7.). Smaller values will
        find more bad channels at increased risk of including good ones. This
        value can be interpreted as the standard score of differences between
        the original and Maxwell-filtered data. See the ``Notes`` section for
        details.

        .. note:: This setting only concerns *noisy* channel detection.
                  The limit for *flat* channel detection currently cannot be
                  controlled by the user. Flat channel detection is always run
                  before noisy channel detection.
    duration : float
        Duration of the segments into which to slice the data for processing,
        in seconds. Default is 5.
    min_count : int
        Minimum number of times a channel must show up as bad in a chunk.
        Default is 5.
    return_scores : bool
        If ``True``, return a dictionary with scoring information for each
        evaluated segment of the data. Default is ``False``.

        .. warning:: This feature is experimental and may change in a future
                     version of MNE-Python without prior notice. Please
                     report any problems and enhancement proposals to the
                     developers.

        .. versionadded:: 0.21
    %(maxwell_origin)s
    %(maxwell_int)s
    %(maxwell_ext)s
    %(maxwell_cal)s
    %(maxwell_cross)s
    %(maxwell_coord)s
    %(maxwell_reg)s
    %(maxwell_ref)s
    %(maxwell_cond)s
    %(maxwell_pos)s
    %(maxwell_mag)s
    %(maxwell_skip)s
    h_freq : float | None
        The cutoff frequency (in Hz) of the low-pass filter that will be
        applied before processing the data. This defaults to ``40.``, which
        should provide similar results to MaxFilter. If you do not wish to
        apply a filter, set this to ``None``.
    %(maxwell_extended)s
    %(verbose)s

    Returns
    -------
    noisy_chs : list
        List of bad MEG channels that were automatically detected as being
        noisy among the good MEG channels.
    flat_chs : list
        List of MEG channels that were detected as being flat in at least
        ``min_count`` segments.
    scores : dict
        A dictionary with information produced by the scoring algorithms.
        Only returned when ``return_scores`` is ``True``. It contains the
        following keys:

        - ``ch_names`` : ndarray, shape (n_meg,)
            The names of the MEG channels. Their order corresponds to the
            order of rows in the ``scores`` and ``limits`` arrays.
        - ``ch_types`` : ndarray, shape (n_meg,)
            The types of the MEG channels in ``ch_names`` (``'mag'``,
            ``'grad'``).
        - ``bins`` : ndarray, shape (n_windows, 2)
            The inclusive window boundaries (start and stop; in seconds) used
            to calculate the scores.
        - ``scores_flat`` : ndarray, shape (n_meg, n_windows)
            The scores for testing whether MEG channels are flat. These values
            correspond to the standard deviation of a segment.
            See the ``Notes`` section for details.
        - ``limits_flat`` : ndarray, shape (n_meg, 1)
            The score thresholds (in standard deviation) above which a segment
            was classified as "flat".
        - ``scores_noisy`` : ndarray, shape (n_meg, n_windows)
            The scores for testing whether MEG channels are noisy. These values
            correspond to the standard score of a segment.
            See the ``Notes`` section for details.
        - ``limits_noisy`` : ndarray, shape (n_meg, 1)
            The score thresholds (in standard scores) above which a segment was
            classified as "noisy".

        .. note:: The scores and limits for channels marked as ``bad`` in the
                  input data will be set to ``np.nan``.

    See Also
    --------
    annotate_flat
    maxwell_filter

    Notes
    -----
    All arguments after ``raw``, ``limit``, ``duration``, ``min_count``, and
    ``return_scores`` are the same as :func:`~maxwell_filter`, except that the
    following are not allowed in this function because they are unused:
    ``st_duration``, ``st_correlation``, ``destination``, ``st_fixed``, and
    ``st_only``.

    This algorithm, for a given chunk of data:

    1. Runs SSS on the data, without removing external components.
    2. Excludes channels as *flat* that have had low variability
       (standard deviation < 0.01 fT or fT/cm in a 30 ms window) in the given
       or any previous chunk.
    3. For each channel :math:`k`, computes the *range* or peak-to-peak
       :math:`d_k` of the difference between the reconstructed and original
       data.
    4. Computes the average :math:`\mu_d` and standard deviation
       :math:`\sigma_d` of the differences (after scaling magnetometer data
       to roughly match the scale of the gradiometer data using ``mag_scale``).
    5. Marks channels as bad for the chunk when
       :math:`d_k > \mu_d + \textrm{limit} \times \sigma_d`. Note that this
       expression can be easily transformed into
       :math:`(d_k - \mu_d) / \sigma_d > \textrm{limit}`, which is equivalent
       to :math:`z(d_k) > \textrm{limit}`, with :math:`z(d_k)` being the
       standard or z-score of the difference.

    Data are processed in chunks of the given ``duration``, and channels that
    are bad for at least ``min_count`` chunks are returned.

    Channels marked as *flat* in step 2 are excluded from all subsequent steps
    of noisy channel detection.

    This algorithm gives results similar to, but not identical with,
    MaxFilter. Differences arise because MaxFilter processes on a
    buffer-by-buffer basis (using buffer-size-dependent downsampling logic),
    uses different filtering characteristics, and possibly other factors.
    Channels that are near the ``limit`` for a given ``min_count`` are
    particularly susceptible to being different between the two
    implementations.

    .. versionadded:: 0.20
    """
    if h_freq is not None:
        if raw.info.get('lowpass') and raw.info['lowpass'] < h_freq:
            msg = (f'The input data has already been low-pass filtered with a '
                   f'{raw.info["lowpass"]} Hz cutoff frequency, which is '
                   f'below the requested cutoff of {h_freq} Hz. Not applying '
                   f'low-pass filter.')
            logger.info(msg)
        else:
            logger.info(f'Applying low-pass filter with {h_freq} Hz cutoff '
                        f'frequency ...')
            raw = raw.copy().load_data().filter(l_freq=None, h_freq=h_freq)

    limit = float(limit)
    onsets, ends = _annotations_starts_stops(
        raw, skip_by_annotation, invert=True)
    del skip_by_annotation
    # operate on chunks
    starts = list()
    stops = list()
    step = int(round(raw.info['sfreq'] * duration))
    for onset, end in zip(onsets, ends):
        if end - onset >= step:
            ss = np.arange(onset, end - step + 1, step)
            starts.extend(ss)
            ss = ss + step
            ss[-1] = end
            stops.extend(ss)
    min_count = min(_ensure_int(min_count, 'min_count'), len(starts))
    logger.info('Scanning for bad channels in %d interval%s (%0.1f sec) ...'
                % (len(starts), _pl(starts), step / raw.info['sfreq']))
    params = _prep_maxwell_filter(
        raw, skip_by_annotation=[],  # already accounted for
        origin=origin, int_order=int_order, ext_order=ext_order,
        calibration=calibration, cross_talk=cross_talk,
        coord_frame=coord_frame, regularize=regularize,
        ignore_ref=ignore_ref, bad_condition=bad_condition, head_pos=head_pos,
        mag_scale=mag_scale, extended_proj=extended_proj)
    del origin, int_order, ext_order, calibration, cross_talk, coord_frame
    del regularize, ignore_ref, bad_condition, head_pos, mag_scale
    good_meg_picks = params['meg_picks'][params['good_mask']]
    assert len(params['meg_picks']) == len(params['coil_scale'])
    assert len(params['good_mask']) == len(params['meg_picks'])
    noisy_chs = Counter()
    flat_chs = Counter()
    flat_limits = dict(grad=0.01e-13, mag=0.01e-15)
    these_limits = np.array([
        flat_limits['grad']
        if pick in params['grad_picks'] else
        flat_limits['mag']
        for pick in good_meg_picks])

    flat_step = max(20, int(30 * raw.info['sfreq'] / 1000.))
    all_flats = set()

    # Prepare variables to return if `return_scores=True`.
    bins = np.empty((len(starts), 2))  # To store start, stop of each segment
    # We create ndarrays with one row per channel, regardless of channel type
    # and whether the channel has been marked as "bad" in info or not. This
    # makes indexing in the loop easier. We only filter this down to the subset
    # of MEG channels after all processing is done.
    ch_names = np.array(raw.ch_names)
    ch_types = np.array(raw.get_channel_types())

    scores_flat = np.full((len(ch_names), len(starts)), np.nan)
    scores_noisy = np.full_like(scores_flat, fill_value=np.nan)

    thresh_flat = np.full((len(ch_names), 1), np.nan)
    thresh_noisy = np.full_like(thresh_flat, fill_value=np.nan)

    for si, (start, stop) in enumerate(zip(starts, stops)):
        n_iter = 0
        orig_data = raw.get_data(None, start, stop, verbose=False)
        chunk_raw = RawArray(
            orig_data, params['info'],
            first_samp=raw.first_samp + start, copy='data', verbose=False)

        t = chunk_raw.times[[0, -1]] + start / raw.info['sfreq']
        logger.info('        Interval %3d: %8.3f - %8.3f'
                    % ((si + 1,) + tuple(t[[0, -1]])))

        # Flat pass: SD < 0.01 fT/cm or 0.01 fT for at 30 ms (or 20 samples)
        n = stop - start
        flat_stop = n - (n % flat_step)
        data = chunk_raw.get_data(good_meg_picks, 0, flat_stop)
        data.shape = (data.shape[0], -1, flat_step)
        delta = np.std(data, axis=-1).min(-1)  # min std across segments

        # We may want to return this later if `return_scores=True`.
        bins[si, :] = t[0], t[-1]
        scores_flat[good_meg_picks, si] = delta
        thresh_flat[good_meg_picks] = these_limits.reshape(-1, 1)

        chunk_flats = delta < these_limits
        chunk_flats = np.where(chunk_flats)[0]
        chunk_flats = [raw.ch_names[good_meg_picks[chunk_flat]]
                       for chunk_flat in chunk_flats]
        flat_chs.update(chunk_flats)
        all_flats |= set(chunk_flats)
        chunk_flats = sorted(all_flats)
        these_picks = [pick for pick in good_meg_picks
                       if raw.ch_names[pick] not in chunk_flats]
        # Bad pass
        chunk_noisy = list()
        params['st_duration'] = int(round(
            chunk_raw.times[-1] * raw.info['sfreq']))
        for n_iter in range(1, 101):  # iteratively exclude the worst ones
            assert set(raw.info['bads']) & set(chunk_noisy) == set()
            params['good_mask'][:] = [
                chunk_raw.ch_names[pick] not in
                raw.info['bads'] + chunk_noisy + chunk_flats
                for pick in params['meg_picks']]
            chunk_raw._data[:] = orig_data
            delta = chunk_raw.get_data(these_picks)
            with use_log_level(False):
                _run_maxwell_filter(
                    chunk_raw, reconstruct='orig', copy=False, **params)

            if n_iter == 1 and len(chunk_flats):
                logger.info('            Flat (%2d): %s'
                            % (len(chunk_flats), ' '.join(chunk_flats)))
            delta -= chunk_raw.get_data(these_picks)
            # p2p
            range_ = np.ptp(delta, axis=-1)
            cs_picks = np.searchsorted(params['meg_picks'], these_picks)
            range_ *= params['coil_scale'][cs_picks, 0]
            mean, std = np.mean(range_), np.std(range_)
            # z score
            z = (range_ - mean) / std
            idx = np.argmax(z)
            max_ = z[idx]

            # We may want to return this later if `return_scores=True`.
            scores_noisy[these_picks, si] = z
            thresh_noisy[these_picks] = limit

            if max_ < limit:
                break

            name = raw.ch_names[these_picks[idx]]
            logger.debug('            Bad:       %s %0.1f'
                         % (name, max_))
            these_picks.pop(idx)
            chunk_noisy.append(name)
        noisy_chs.update(chunk_noisy)
    noisy_chs = sorted((b for b, c in noisy_chs.items() if c >= min_count),
                       key=lambda x: raw.ch_names.index(x))
    flat_chs = sorted((f for f, c in flat_chs.items() if c >= min_count),
                      key=lambda x: raw.ch_names.index(x))

    # Only include MEG channels.
    ch_names = ch_names[params['meg_picks']]
    ch_types = ch_types[params['meg_picks']]
    scores_flat = scores_flat[params['meg_picks']]
    thresh_flat = thresh_flat[params['meg_picks']]
    scores_noisy = scores_noisy[params['meg_picks']]
    thresh_noisy = thresh_noisy[params['meg_picks']]

    logger.info('    Static bad channels:  %s' % (noisy_chs,))
    logger.info('    Static flat channels: %s' % (flat_chs,))
    logger.info('[done]')

    if return_scores:
        scores = dict(ch_names=ch_names,
                      ch_types=ch_types,
                      bins=bins,
                      scores_flat=scores_flat,
                      limits_flat=thresh_flat,
                      scores_noisy=scores_noisy,
                      limits_noisy=thresh_noisy)
        return noisy_chs, flat_chs, scores
    else:
        return noisy_chs, flat_chs


def _read_cross_talk(cross_talk, ch_names):
    sss_ctc = dict()
    ctc = None
    if cross_talk is not None:
        sss_ctc = _read_ctc(cross_talk)
        ctc_chs = sss_ctc['proj_items_chs']
        # checking for extra space ambiguity in channel names
        # between old and new fif files
        if ch_names[0] not in ctc_chs:
            ctc_chs = _clean_names(ctc_chs, remove_whitespace=True)
            ch_names = _clean_names(ch_names, remove_whitespace=True)
        missing = sorted(list(set(ch_names) - set(ctc_chs)))
        if len(missing) != 0:
            raise RuntimeError('Missing MEG channels in cross-talk matrix:\n%s'
                               % missing)
        missing = sorted(list(set(ctc_chs) - set(ch_names)))
        if len(missing) > 0:
            warn('Not all cross-talk channels in raw:\n%s' % missing)
        ctc_picks = [ctc_chs.index(name) for name in ch_names]
        ctc = sss_ctc['decoupler'][ctc_picks][:, ctc_picks]
        # I have no idea why, but MF transposes this for storage..
        sss_ctc['decoupler'] = sss_ctc['decoupler'].T.tocsc()
    return ctc, sss_ctc
