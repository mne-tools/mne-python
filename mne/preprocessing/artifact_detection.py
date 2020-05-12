# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)


import numpy as np
from ..annotations import (Annotations, _annotations_starts_stops)
from ..transforms import (quat_to_rot, _average_quats, _angle_between_quats,
                          apply_trans, _quat_to_affine)
from ..filter import filter_data
from .. import Transform
from ..utils import (_mask_to_onsets_offsets, logger, verbose)


@verbose
def annotate_muscle_zscore(raw, threshold=4, ch_type=None, min_length_good=0.1,
                           filter_freq=(110, 140), n_jobs=1, verbose=None):
    """Create annotations for segments that likely contain muscle artifacts.

    Detects data segments containing activity in the frequency range given by
    ``filter_freq`` whose envelope magnitude exceeds the specified z-score
    threshold, when summed across channels and divided by ``sqrt(n_channels)``.
    False-positive transient peaks are prevented by low-pass filtering the
    resulting z-score time series at 4 Hz. Only operates on a single channel
    type, if ``ch_type`` is ``None`` it will select the first type in the list
    ``mag``, ``grad``, ``eeg``.
    See :footcite:`Muthukumaraswamy2013` for background on choosing
    ``filter_freq`` and ``threshold``.

    Parameters
    ----------
    raw : instance of Raw
        Data to estimate segments with muscle artifacts.
    threshold : float
        The threshold in z-scores for marking segments as containing muscle
        activity artifacts.
    ch_type : 'mag' | 'grad' | 'eeg' | None
        The type of sensors to use. If ``None`` it will take the first type in
        ``mag``, ``grad``, ``eeg``.
    min_length_good : float | None
        The shortest allowed duration of "good data" (in seconds) between
        adjacent annotations; shorter segments will be incorporated into the
        surrounding annotations.``None`` is equivalent to ``0``.
        Default is ``0.1``.
    filter_freq : array-like, shape (2,)
        The lower and upper frequencies of the band-pass filter.
        Default is ``(110, 140)``.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    annot : mne.Annotations
        Periods with muscle artifacts annotated as BAD_muscle.
    scores_muscle : array
        Z-score values averaged across channels for each sample.

    References
    ----------
    .. footbibliography::
    """
    from scipy.stats import zscore
    from scipy.ndimage.measurements import label

    raw_copy = raw.copy()

    if ch_type is None:
        raw_ch_type = raw_copy.get_channel_types()
        if 'mag' in raw_ch_type:
            ch_type = 'mag'
        elif 'grad' in raw_ch_type:
            ch_type = 'grad'
        elif 'eeg' in raw_ch_type:
            ch_type = 'eeg'
        else:
            raise ValueError('No M/EEG channel types found, please specify a'
                             ' ch_type or provide M/EEG sensor data')
        logger.info('Using %s sensors for muscle artifact detection'
                    % (ch_type))

    if ch_type in ('mag', 'grad'):
        raw_copy.pick_types(meg=ch_type, ref_meg=False)
    else:
        ch_type = {'meg': False, ch_type: True}
        raw_copy.pick_types(**ch_type)

    raw_copy.filter(filter_freq[0], filter_freq[1], fir_design='firwin',
                    pad="reflect_limited", n_jobs=n_jobs)
    raw_copy.apply_hilbert(envelope=True, n_jobs=n_jobs)

    data = raw_copy.get_data(reject_by_annotation="NaN")
    nan_mask = ~np.isnan(data[0])
    sfreq = raw_copy.info['sfreq']

    art_scores = zscore(data[:, nan_mask], axis=1)
    art_scores = art_scores.sum(axis=0) / np.sqrt(art_scores.shape[0])
    art_scores = filter_data(art_scores, sfreq, None, 4)

    scores_muscle = np.zeros(data.shape[1])
    scores_muscle[nan_mask] = art_scores

    art_mask = scores_muscle > threshold
    # return muscle scores with NaNs
    scores_muscle[~nan_mask] = np.nan

    # remove artifact free periods shorter than min_length_good
    min_length_good = 0 if min_length_good is None else min_length_good
    min_samps = min_length_good * sfreq
    comps, num_comps = label(art_mask == 0)
    for com in range(1, num_comps + 1):
        l_idx = np.nonzero(comps == com)[0]
        if len(l_idx) < min_samps:
            art_mask[l_idx] = True

    annot = _annotations_from_mask(raw_copy.times, art_mask, 'BAD_muscle')

    return annot, scores_muscle


def annotate_movement(raw, pos, rotation_velocity_limit=None,
                      translation_velocity_limit=None,
                      mean_distance_limit=None, use_dev_head_trans='average'):
    """Detect segments with movement.

    Detects segments periods further from rotation_velocity_limit,
    translation_velocity_limit and mean_distance_limit. It returns an
    annotation with the bad segments.

    Parameters
    ----------
    raw : instance of Raw
        Data to compute head position.
    pos : array, shape (N, 10)
        The position and quaternion parameters from cHPI fitting. Obtained
        with `mne.chpi` functions.
    rotation_velocity_limit : float
        Head rotation velocity limit in radians per second.
    translation_velocity_limit : float
        Head translation velocity limit in radians per second.
    mean_distance_limit : float
        Head position limit from mean recording in meters.
    use_dev_head_trans : 'average' (default) | 'info'
        Identify the device to head transform used to define the
        fixed HPI locations for computing moving distances.
        If ``average`` the average device to head transform is
        computed using ``compute_average_dev_head_t``.
        If ``info``, ``raw.info['dev_head_t']`` is used.

    Returns
    -------
    annot : mne.Annotations
        Periods with head motion.
    hpi_disp : array
        Head position over time with respect to the mean head pos.

    See Also
    --------
    compute_average_dev_head_t
    """
    sfreq = raw.info['sfreq']
    hp_ts = pos[:, 0].copy()
    hp_ts -= raw.first_samp / sfreq
    dt = np.diff(hp_ts)
    hp_ts = np.concatenate([hp_ts, [hp_ts[-1] + 1. / sfreq]])

    annot = Annotations([], [], [], orig_time=None)  # rel to data start

    # Annotate based on rotational velocity
    t_tot = raw.times[-1]
    if rotation_velocity_limit is not None:
        assert rotation_velocity_limit > 0
        # Rotational velocity (radians / sec)
        r = _angle_between_quats(pos[:-1, 1:4], pos[1:, 1:4])
        r /= dt
        bad_mask = (r >= np.deg2rad(rotation_velocity_limit))
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(u'Omitting %5.1f%% (%3d segments): '
                    u'ω >= %5.1f°/s (max: %0.1f°/s)'
                    % (bad_pct, len(onsets), rotation_velocity_limit,
                       np.rad2deg(r.max())))
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_mov_rotat_vel')

    # Annotate based on translational velocity limit
    if translation_velocity_limit is not None:
        assert translation_velocity_limit > 0
        v = np.linalg.norm(np.diff(pos[:, 4:7], axis=0), axis=-1)
        v /= dt
        bad_mask = (v >= translation_velocity_limit)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(u'Omitting %5.1f%% (%3d segments): '
                    u'v >= %5.4fm/s (max: %5.4fm/s)'
                    % (bad_pct, len(onsets), translation_velocity_limit,
                       v.max()))
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_mov_trans_vel')

    # Annotate based on displacement from mean head position
    disp = []
    if mean_distance_limit is not None:
        assert mean_distance_limit > 0

        # compute dev to head transform for fixed points
        use_dev_head_trans = use_dev_head_trans.lower()
        if use_dev_head_trans not in ['average', 'info']:
            raise ValueError('use_dev_head_trans must be either' +
                             ' \'average\' or \'info\': got \'%s\''
                             % (use_dev_head_trans,))

        if use_dev_head_trans == 'average':
            fixed_dev_head_t = compute_average_dev_head_t(raw, pos)
        elif use_dev_head_trans == 'info':
            fixed_dev_head_t = raw.info['dev_head_t']

        # Get static head pos from file, used to convert quat to cartesian
        chpi_pos = sorted([d for d in raw.info['hpi_results'][-1]
                          ['dig_points']], key=lambda x: x['ident'])
        chpi_pos = np.array([d['r'] for d in chpi_pos])

        # Get head pos changes during recording
        chpi_pos_mov = np.array([apply_trans(_quat_to_affine(quat), chpi_pos)
                                for quat in pos[:, 1:7]])

        # get fixed position
        chpi_pos_fix = apply_trans(fixed_dev_head_t, chpi_pos)

        # get movement displacement from mean pos
        hpi_disp = chpi_pos_mov - np.tile(chpi_pos_fix, (pos.shape[0], 1, 1))

        # get positions above threshold distance
        disp = np.sqrt((hpi_disp ** 2).sum(axis=2))
        bad_mask = np.any(disp > mean_distance_limit, axis=1)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(u'Omitting %5.1f%% (%3d segments): '
                    u'disp >= %5.4fm (max: %5.4fm)'
                    % (bad_pct, len(onsets), mean_distance_limit, disp.max()))
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_mov_dist')
    return annot, disp


def compute_average_dev_head_t(raw, pos):
    """Get new device to head transform based on good segments.

    Segments starting with "BAD" annotations are not included for calculating
    the mean head position.

    Parameters
    ----------
    raw : instance of Raw
        Data to compute head position.
    pos : array, shape (N, 10)
        The position and quaternion parameters from cHPI fitting.

    Returns
    -------
    dev_head_t : array
        New trans matrix using the averaged good head positions.
    """
    sfreq = raw.info['sfreq']
    seg_good = np.ones(len(raw.times))
    trans_pos = np.zeros(3)
    hp = pos.copy()
    hp_ts = hp[:, 0] - raw._first_time

    # Check rounding issues at 0 time
    if hp_ts[0] < 0:
        hp_ts[0] = 0
        assert hp_ts[1] > 1. / sfreq

    # Mask out segments if beyond scan time
    mask = hp_ts <= raw.times[-1]
    if not mask.all():
        logger.info(
            '          Removing %d samples > raw.times[-1] (%s)'
            % (np.sum(~mask), raw.times[-1]))
        hp = hp[mask]
    del mask, hp_ts

    # Get time indices
    ts = np.concatenate((hp[:, 0], [(raw.last_samp + 1) / sfreq]))
    assert (np.diff(ts) > 0).all()
    ts -= raw.first_samp / sfreq
    idx = raw.time_as_index(ts, use_rounding=True)
    del ts
    if idx[0] == -1:  # annoying rounding errors
        idx[0] = 0
        assert idx[1] > 0
    assert (idx >= 0).all()
    assert idx[-1] == len(seg_good)
    assert (np.diff(idx) > 0).all()

    # Mark times bad that are bad according to annotations
    onsets, ends = _annotations_starts_stops(raw, 'bad')
    for onset, end in zip(onsets, ends):
        seg_good[onset:end] = 0
    dt = np.diff(np.cumsum(np.concatenate([[0], seg_good]))[idx])
    assert (dt >= 0).all()
    dt = dt / sfreq
    del seg_good, idx

    # Get weighted head pos trans and rot
    trans_pos += np.dot(dt, hp[:, 4:7])

    rot_qs = hp[:, 1:4]
    best_q = _average_quats(rot_qs, weights=dt)

    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    trans[:3, 3] = trans_pos / dt.sum()
    assert np.linalg.norm(trans[:3, 3]) < 1  # less than 1 meter is sane
    dev_head_t = Transform('meg', 'head', trans)
    return dev_head_t


def _annotations_from_mask(times, art_mask, art_name):
    """Construct annotations from boolean mask of the data."""
    from scipy.ndimage.measurements import label
    comps, num_comps = label(art_mask)
    onsets, durations, desc = [], [], []
    n_times = len(times)
    for lbl in range(1, num_comps + 1):
        l_idx = np.nonzero(comps == lbl)[0]
        onsets.append(times[l_idx[0]])
        # duration is to the time after the last labeled time
        # or to the end of the times.
        if 1 + l_idx[-1] < n_times:
            durations.append(times[1 + l_idx[-1]] - times[l_idx[0]])
        else:
            durations.append(times[l_idx[-1]] - times[l_idx[0]])
        desc.append(art_name)
    return Annotations(onsets, durations, desc)
