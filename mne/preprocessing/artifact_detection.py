# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.signal import find_peaks
from scipy.stats import zscore

from ..annotations import (
    Annotations,
    _adjust_onset_meas_date,
    _annotations_starts_stops,
    annotations_from_events,
)
from ..filter import filter_data
from ..io.base import BaseRaw
from ..transforms import (
    Transform,
    _angle_between_quats,
    _average_quats,
    _quat_to_affine,
    apply_trans,
    quat_to_rot,
)
from ..utils import (
    _check_option,
    _mask_to_onsets_offsets,
    _pl,
    _validate_type,
    logger,
    verbose,
    warn,
)


@verbose
def annotate_muscle_zscore(
    raw,
    threshold=4,
    ch_type=None,
    min_length_good=0.1,
    filter_freq=(110, 140),
    n_jobs=None,
    verbose=None,
):
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
    raw_copy = raw.copy()

    if ch_type is None:
        raw_ch_type = raw_copy.get_channel_types()
        if "mag" in raw_ch_type:
            ch_type = "mag"
        elif "grad" in raw_ch_type:
            ch_type = "grad"
        elif "eeg" in raw_ch_type:
            ch_type = "eeg"
        else:
            raise ValueError(
                "No M/EEG channel types found, please specify a 'ch_type' or provide "
                "M/EEG sensor data."
            )
        logger.info("Using %s sensors for muscle artifact detection", ch_type)
    else:
        _check_option("ch_type", ch_type, ["mag", "grad", "eeg"])
    raw_copy.pick(ch_type)

    raw_copy.filter(
        filter_freq[0],
        filter_freq[1],
        fir_design="firwin",
        pad="reflect_limited",
        n_jobs=n_jobs,
    )
    raw_copy.apply_hilbert(envelope=True, n_jobs=n_jobs)

    data = raw_copy.get_data(reject_by_annotation="NaN")
    nan_mask = ~np.isnan(data[0])
    sfreq = raw_copy.info["sfreq"]

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

    annot = _annotations_from_mask(
        raw_copy.times, art_mask, "BAD_muscle", orig_time=raw.info["meas_date"]
    )
    _adjust_onset_meas_date(annot, raw)
    return annot, scores_muscle


def annotate_movement(
    raw,
    pos,
    rotation_velocity_limit=None,
    translation_velocity_limit=None,
    mean_distance_limit=None,
    use_dev_head_trans="average",
):
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
        Head rotation velocity limit in degrees per second.
    translation_velocity_limit : float
        Head translation velocity limit in meters per second.
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
    sfreq = raw.info["sfreq"]
    hp_ts = pos[:, 0].copy() - raw.first_time
    dt = np.diff(hp_ts)
    hp_ts = np.concatenate([hp_ts, [hp_ts[-1] + 1.0 / sfreq]])
    orig_time = raw.info["meas_date"]
    annot = Annotations([], [], [], orig_time=orig_time)

    # Annotate based on rotational velocity
    t_tot = raw.times[-1]
    if rotation_velocity_limit is not None:
        assert rotation_velocity_limit > 0
        # Rotational velocity (radians / s)
        r = _angle_between_quats(pos[:-1, 1:4], pos[1:, 1:4])
        r /= dt
        bad_mask = r >= np.deg2rad(rotation_velocity_limit)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(
            "Omitting %5.1f%% (%3d segments): " "ω >= %5.1f°/s (max: %0.1f°/s)",
            bad_pct,
            len(onsets),
            rotation_velocity_limit,
            np.rad2deg(r.max()),
        )
        annot += _annotations_from_mask(
            hp_ts, bad_mask, "BAD_mov_rotat_vel", orig_time=orig_time
        )

    # Annotate based on translational velocity limit
    if translation_velocity_limit is not None:
        assert translation_velocity_limit > 0
        v = np.linalg.norm(np.diff(pos[:, 4:7], axis=0), axis=-1)
        v /= dt
        bad_mask = v >= translation_velocity_limit
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(
            "Omitting %5.1f%% (%3d segments): " "v >= %5.4fm/s (max: %5.4fm/s)",
            bad_pct,
            len(onsets),
            translation_velocity_limit,
            v.max(),
        )
        annot += _annotations_from_mask(
            hp_ts, bad_mask, "BAD_mov_trans_vel", orig_time=orig_time
        )

    # Annotate based on displacement from mean head position
    disp = []
    if mean_distance_limit is not None:
        assert mean_distance_limit > 0

        # compute dev to head transform for fixed points
        use_dev_head_trans = use_dev_head_trans.lower()
        if use_dev_head_trans not in ["average", "info"]:
            raise ValueError(
                "use_dev_head_trans must be either"
                f" 'average' or 'info': got '{use_dev_head_trans}'"
            )

        if use_dev_head_trans == "average":
            fixed_dev_head_t = compute_average_dev_head_t(raw, pos)
        elif use_dev_head_trans == "info":
            fixed_dev_head_t = raw.info["dev_head_t"]

        # Get static head pos from file, used to convert quat to cartesian
        chpi_pos = sorted(
            [d for d in raw.info["hpi_results"][-1]["dig_points"]],
            key=lambda x: x["ident"],
        )
        chpi_pos = np.array([d["r"] for d in chpi_pos])

        # Get head pos changes during recording
        chpi_pos_mov = np.array(
            [apply_trans(_quat_to_affine(quat), chpi_pos) for quat in pos[:, 1:7]]
        )

        # get fixed position
        chpi_pos_fix = apply_trans(fixed_dev_head_t, chpi_pos)

        # get movement displacement from mean pos
        hpi_disp = chpi_pos_mov - np.tile(chpi_pos_fix, (pos.shape[0], 1, 1))

        # get positions above threshold distance
        disp = np.sqrt((hpi_disp**2).sum(axis=2))
        bad_mask = np.any(disp > mean_distance_limit, axis=1)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        logger.info(
            "Omitting %5.1f%% (%3d segments): " "disp >= %5.4fm (max: %5.4fm)",
            bad_pct,
            len(onsets),
            mean_distance_limit,
            disp.max(),
        )
        annot += _annotations_from_mask(
            hp_ts, bad_mask, "BAD_mov_dist", orig_time=orig_time
        )
    _adjust_onset_meas_date(annot, raw)
    return annot, disp


@verbose
def compute_average_dev_head_t(raw, pos, *, verbose=None):
    """Get new device to head transform based on good segments.

    Segments starting with "BAD" annotations are not included for calculating
    the mean head position.

    Parameters
    ----------
    raw : instance of Raw | list of Raw
        Data to compute head position. Can be a list containing multiple raw
        instances.
    pos : array, shape (N, 10) | list of ndarray
        The position and quaternion parameters from cHPI fitting. Can be
        a list containing multiple position arrays, one per raw instance passed.
    %(verbose)s

    Returns
    -------
    dev_head_t : instance of Transform
        New ``dev_head_t`` transformation using the averaged good head positions.

    Notes
    -----
    .. versionchanged:: 1.7
       Support for multiple raw instances and position arrays was added.
    """
    # Get weighted head pos trans and rot
    if not isinstance(raw, list | tuple):
        raw = [raw]
    if not isinstance(pos, list | tuple):
        pos = [pos]
    if len(pos) != len(raw):
        raise ValueError(
            f"Number of head positions ({len(pos)}) must match the number of raw "
            f"instances ({len(raw)})"
        )
    hp = list()
    dt = list()
    for ri, (r, p) in enumerate(zip(raw, pos)):
        _validate_type(r, BaseRaw, f"raw[{ri}]")
        _validate_type(p, np.ndarray, f"pos[{ri}]")
        hp_, dt_ = _raw_hp_weights(r, p)
        hp.append(hp_)
        dt.append(dt_)
    hp = np.concatenate(hp, axis=0)
    dt = np.concatenate(dt, axis=0)
    dt /= dt.sum()
    best_q = _average_quats(hp[:, 1:4], weights=dt)
    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    trans[:3, 3] = dt @ hp[:, 4:7]
    dist = np.linalg.norm(trans[:3, 3])
    if dist > 1:  # less than 1 meter is sane
        warn(f"Implausible head position detected: {dist} meters from device origin")
    dev_head_t = Transform("meg", "head", trans)
    return dev_head_t


def _raw_hp_weights(raw, pos):
    sfreq = raw.info["sfreq"]
    seg_good = np.ones(len(raw.times))
    hp = pos.copy()
    hp_ts = hp[:, 0] - raw._first_time

    # Check rounding issues at 0 time
    if hp_ts[0] < 0:
        hp_ts[0] = 0
        assert hp_ts[1] > 1.0 / sfreq

    # Mask out segments if beyond scan time
    mask = hp_ts <= raw.times[-1]
    if not mask.all():
        logger.info(
            "          Removing %d samples > raw.times[-1] (%s)",
            np.sum(~mask),
            raw.times[-1],
        )
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
    onsets, ends = _annotations_starts_stops(raw, "bad")
    for onset, end in zip(onsets, ends):
        seg_good[onset:end] = 0
    dt = np.diff(np.cumsum(np.concatenate([[0], seg_good]))[idx])
    assert (dt >= 0).all()
    dt = dt / sfreq
    del seg_good, idx
    return hp, dt


def _annotations_from_mask(times, mask, annot_name, orig_time=None):
    """Construct annotations from boolean mask of the data."""
    mask_tf = distance_transform_edt(mask)
    # Overcome the shortcoming of find_peaks
    # in finding a marginal peak, by
    # inserting 0s at the front and the
    # rear, then subtracting in index
    ins_mask_tf = np.concatenate((np.zeros(1), mask_tf, np.zeros(1)))
    left_midpt_index = find_peaks(ins_mask_tf)[0] - 1
    right_midpt_index = (
        np.flip(len(ins_mask_tf) - 1 - find_peaks(ins_mask_tf[::-1])[0]) - 1
    )
    onsets_index = left_midpt_index - mask_tf[left_midpt_index].astype(int) + 1
    ends_index = right_midpt_index + mask_tf[right_midpt_index].astype(int)
    # Ensure onsets_index >= 0,
    # otherwise the duration starts from the beginning
    onsets_index[onsets_index < 0] = 0
    # Ensure ends_index < len(times),
    # otherwise the duration is to the end of times
    if len(times) == len(mask):
        ends_index[ends_index >= len(times)] = len(times) - 1
    # To be consistent with the original code,
    # possibly a bug in tests code
    else:
        ends_index[ends_index >= len(mask)] = len(mask)
    onsets = times[onsets_index]
    ends = times[ends_index]
    durations = ends - onsets
    desc = [annot_name] * len(durations)
    return Annotations(onsets, durations, desc, orig_time=orig_time)


@verbose
def annotate_break(
    raw,
    events=None,
    min_break_duration=15.0,
    t_start_after_previous=5.0,
    t_stop_before_next=5.0,
    ignore=("bad", "edge"),
    *,
    verbose=None,
):
    """Create `~mne.Annotations` for breaks in an ongoing recording.

    This function first searches for segments in the data that are not
    annotated or do not contain any events and are at least
    ``min_break_duration`` seconds long, and then proceeds to creating
    annotations for those break periods.

    Parameters
    ----------
    raw : instance of Raw
        The continuous data to analyze.
    events : None | array, shape (n_events, 3)
        If ``None`` (default), operate based solely on the annotations present
        in ``raw``. If an events array, ignore any annotations in the raw data,
        and operate based on these events only.
    min_break_duration : float
        The minimum time span in seconds between the offset of one and the
        onset of the subsequent annotation (if ``events`` is ``None``) or
        between two consecutive events (if ``events`` is an array) to consider
        this period a "break". Defaults to 15 seconds.

        .. note:: This value defines the minimum duration of a break period in
                  the data, **not** the minimum duration of the generated
                  annotations! See also ``t_start_after_previous`` and
                  ``t_stop_before_next`` for details.

    t_start_after_previous, t_stop_before_next : float
        Specifies how far the to-be-created "break" annotation extends towards
        the two annotations or events spanning the break. This can be used to
        ensure e.g. that the break annotation doesn't start and end immediately
        with a stimulation event. If, for example, your data contains a break
        of 30 seconds between two stimuli, and ``t_start_after_previous`` is
        set to ``5`` and ``t_stop_before_next`` is set to ``3``, the break
        annotation will start 5 seconds after the first stimulus, and end 3
        seconds before the second stimulus, yielding an annotated break of
        ``30 - 5 - 3 = 22`` seconds. Both default to 5 seconds.

        .. note:: The beginning and the end of the recording will be annotated
                  as breaks, too, if the period from recording start until the
                  first annotation or event (or from last annotation or event
                  until recording end) is at least ``min_break_duration``
                  seconds long.

    ignore : iterable of str
        Annotation descriptions starting with these strings will be ignored by
        the break-finding algorithm. The string comparison is case-insensitive,
        i.e., ``('bad',)`` and ``('BAD',)`` are equivalent. By default, all
        annotation descriptions starting with "bad" and annotations
        indicating "edges" (produced by data concatenation) will be
        ignored. Pass an empty list or tuple to take all existing annotations
        into account. If ``events`` is passed, this parameter has no effect.
    %(verbose)s

    Returns
    -------
    break_annotations : instance of Annotations
        The break annotations, each with the description ``'BAD_break'``. If
        no breaks could be found given the provided function parameters, an
        empty `~mne.Annotations` object will be returned.

    Notes
    -----
    .. versionadded:: 0.24
    """
    _validate_type(item=raw, item_name="raw", types=BaseRaw, type_name="Raw")
    _validate_type(item=events, item_name="events", types=(None, np.ndarray))

    if min_break_duration - t_start_after_previous - t_stop_before_next <= 0:
        annot_dur = min_break_duration - t_start_after_previous - t_stop_before_next
        raise ValueError(
            f"The result of "
            f"min_break_duration - t_start_after_previous - "
            f"t_stop_before_next must be greater than 0, but it is: "
            f"{annot_dur}"
        )

    if events is not None and events.size == 0:
        raise ValueError("The events array must not be empty.")

    if events is not None or not ignore:
        ignore = tuple()
    else:
        ignore = tuple(ignore)

    for item in ignore:
        _validate_type(item=item, types="str", item_name='All elements of "ignore"')

    if events is None:
        annotations = raw.annotations.copy()
        if ignore:
            logger.info(
                f"Ignoring annotations with descriptions starting "
                f'with: {", ".join(ignore)}'
            )
    else:
        annotations = annotations_from_events(
            events=events, sfreq=raw.info["sfreq"], orig_time=raw.info["meas_date"]
        )

    if not annotations:
        raise ValueError("Could not find (or generate) any annotations in your data.")

    # Only keep annotations of interest and extract annotated time periods
    # Ignore case
    ignore = tuple(i.lower() for i in ignore)
    keep_mask = [True] * len(annotations)
    for idx, description in enumerate(annotations.description):
        description = description.lower()
        if any(description.startswith(i) for i in ignore):
            keep_mask[idx] = False

    annotated_intervals = [
        [onset, onset + duration]
        for onset, duration in zip(
            annotations.onset[keep_mask], annotations.duration[keep_mask]
        )
    ]

    # Merge overlapping annotation intervals
    # Pre-load `merged_intervals` with the first interval to simplify
    # processing
    merged_intervals = [annotated_intervals[0]]
    for interval in annotated_intervals:
        merged_interval_stop = merged_intervals[-1][1]
        interval_start, interval_stop = interval

        if interval_stop < merged_interval_stop:
            # Current interval ends sooner than the merged one; skip it
            continue
        elif (
            interval_start <= merged_interval_stop
            and interval_stop >= merged_interval_stop
        ):
            # Expand duration of the merged interval
            merged_intervals[-1][1] = interval_stop
        else:
            # No overlap between the current interval and the existing merged
            # time period; proceed to the next interval
            merged_intervals.append(interval)

    merged_intervals = np.array(merged_intervals)
    merged_intervals -= raw.first_time  # work in zero-based time

    # Now extract the actual break periods
    break_onsets = []
    break_durations = []

    # Handle the time period up until the first annotation
    if 0 < merged_intervals[0][0] and merged_intervals[0][0] >= min_break_duration:
        onset = 0  # don't add t_start_after_previous here
        offset = merged_intervals[0][0] - t_stop_before_next
        duration = offset - onset
        break_onsets.append(onset)
        break_durations.append(duration)

    # Handle the time period between first and last annotation
    for idx, _ in enumerate(merged_intervals[1:, :], start=1):
        this_start = merged_intervals[idx, 0]
        previous_stop = merged_intervals[idx - 1, 1]
        if this_start - previous_stop < min_break_duration:
            continue

        onset = previous_stop + t_start_after_previous
        offset = this_start - t_stop_before_next
        duration = offset - onset
        break_onsets.append(onset)
        break_durations.append(duration)

    # Handle the time period after the last annotation
    if (
        raw.times[-1] > merged_intervals[-1][1]
        and raw.times[-1] - merged_intervals[-1][1] >= min_break_duration
    ):
        onset = merged_intervals[-1][1] + t_start_after_previous
        offset = raw.times[-1]  # don't subtract t_stop_before_next here
        duration = offset - onset
        break_onsets.append(onset)
        break_durations.append(duration)

    # Finally, create the break annotations
    break_annotations = Annotations(
        onset=break_onsets,
        duration=break_durations,
        description=["BAD_break"],
        orig_time=raw.info["meas_date"],
    )

    # Log some info
    n_breaks = len(break_annotations)
    break_times = [
        f"{o:.1f} – {o + d:.1f} s [{d:.1f} s]"
        for o, d in zip(break_annotations.onset, break_annotations.duration)
    ]
    break_times = "\n    ".join(break_times)
    total_break_dur = sum(break_annotations.duration)
    fraction_breaks = total_break_dur / raw.times[-1]
    logger.info(
        f"\nDetected {n_breaks} break period{_pl(n_breaks)} of >= "
        f"{min_break_duration} s duration:\n    {break_times}\n"
        f"In total, {round(100 * fraction_breaks, 1):.1f}% of the "
        f"data ({round(total_break_dur, 1):.1f} s) have been marked "
        f"as a break.\n"
    )
    _adjust_onset_meas_date(break_annotations, raw)

    return break_annotations
