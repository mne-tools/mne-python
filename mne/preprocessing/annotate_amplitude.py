# License: BSD-3-Clause

import numpy as np

from ..io import BaseRaw
from ..annotations import Annotations, _adjust_onset_meas_date
from ..io.pick import _picks_to_idx, _picks_by_type, _get_channel_types
from ..utils import _validate_type, verbose, logger, _mask_to_onsets_offsets


@verbose
def annotate_amplitude(raw, peak=None, flat=None, bad_percent=5,
                       min_duration=0.005, picks=None, *, verbose=None):
    """
    Annotate segments of raw data which PTP amplitudes between consecutive
    samples exceeds thresholds in ``peak`` or fall below thresholds in
    ``flat``.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    peak : float | dict | None
        Reject segments based on **maximum** peak-to-peak signal amplitude
        (PTP). Valid **keys** can be any channel type present in the object.
        The **values** are floats that set the minimum acceptable PTP. If the
        PTP is smaller than this threshold, the segment will be annotated.
        If float, the minimum acceptable PTP is applied to all channels.
    flat : float | dict | None
        Reject segments based on **minimum** peak-to-peak signal amplitude
        (PTP). Valid **keys** can be any channel type present in the object.
        The **values** are floats that set the minimum acceptable PTP. If the
        PTP is smaller than this threshold, the segment will be annotated.
        If float, the minimum acceptable PTP is applied to all channels.
    bad_percent : float
        The percentage of the time a channel can be bad.
        Below this percentage, temporal bad marking (:class:`~mne.Annotations`)
        will be used. Above this percentage, spatial bad marking
        (:class:`info['bads'] <mne.Info>`) will be used.
        Defaults to ``5`` (5%%).
    min_duration : float
        The minimum duration (sec) to consider as above or below threshold.
        For some systems with low bit data representations, adjacent time
        samples with exactly the same value are not totally uncommon.
        Defaults to ``0.005`` (5 ms).
    %(picks_good_data)s
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        The annotated bad segments.
    bads : list
        The channels detected as bad.

    Notes
    -----
    This function may perform much faster if data is loaded in memory, as it
    loads data one channel type at a time (across all time points), which is
    typically not an efficient way to read raw data from disk.

    .. versionadded:: 1.0
    """
    _validate_type(raw, BaseRaw, 'raw')
    picks_ = _picks_to_idx(raw.info, picks, 'data_or_ica', exclude='bads')
    peak = _check_ptp(peak, 'peak', raw.info, picks_)
    flat = _check_ptp(flat, 'flat', raw.info, picks_)
    if peak is None and flat is None:
        raise ValueError(
            "At least one of the arguments 'peak' or 'flat' must not be None.")
    bad_percent = _check_bad_percent(bad_percent)
    min_duration = _check_min_duration(min_duration,
                                       raw.times.size * 1 / raw.info['sfreq'])
    min_duration_samples = int(np.round(min_duration * raw.info['sfreq']))
    bads = list()

    # grouping picks by channel types to avoid operating on each channel
    # individually
    picks = {
        ch_type: np.intersect1d(picks_of_type, picks_, assume_unique=True)
        for ch_type, picks_of_type in _picks_by_type(raw.info, exclude='bads')
        }
    del picks_  # re-using this variable name in for loop

    # size matching the diff a[i+1] - a[i]
    any_flat = np.zeros(len(raw.times) - 1, bool)
    any_peak = np.zeros(len(raw.times) - 1, bool)

    # look for discrete difference above or below thresholds
    logger.info('Finding segments below or above PTP threshold.')
    for ch_type, picks_ in picks.items():
        diff = np.abs(np.diff(raw.get_data(picks=picks_), axis=1))

        if flat is not None:
            flat_ = diff <= flat[ch_type]
            # reject too short segments
            flat_ = _reject_short_segments(flat_, min_duration_samples)
            # reject channels above maximum bad_percentage
            flat_count = flat_.sum(axis=1)
            flat_count[np.nonzero(flat_count)] += 1  # offset by 1 due to diff
            flat_mean = flat_count / raw.times.size * 100
            flat_ch_to_set_bad = picks_[np.where(flat_mean >= bad_percent)[0]]
            bads.extend(flat_ch_to_set_bad)
            # add onset/offset for annotations
            flat_ch_to_annotate = picks_[
                np.where((0 < flat_mean) & (flat_mean < bad_percent))[0]]
            idx = np.where(flat_[flat_ch_to_annotate, :])[1]
            any_flat[idx] = True

        if peak is not None:
            peak_ = diff >= peak[ch_type]
            # reject too short segments
            peak_ = _reject_short_segments(peak_, min_duration_samples)
            # reject channels above maximum bad_percentage
            peak_count = peak_.sum(axis=1)
            peak_count[np.nonzero(peak_count)] += 1  # offset by 1 due to diff
            peak_mean = peak_count / raw.times.size * 100
            peak_ch_to_set_bad = picks_[np.where(peak_mean >= bad_percent)[0]]
            bads.extend(peak_ch_to_set_bad)
            # add onset/offset for annotations
            peak_ch_to_annotate = picks_[
                np.where((0 < peak_mean) & (peak_mean < bad_percent))[0]]
            idx = np.where(peak_[peak_ch_to_annotate, :])[1]
            any_peak[idx] = True

    # annotation for flat
    annotation_flat = _create_annotations(any_flat, 'flat', raw)
    # annotation for peak
    annotation_peak = _create_annotations(any_peak, 'peak', raw)
    # group
    annotations = annotation_flat + annotation_peak
    # bads
    bads = [raw.ch_names[bad] for bad in bads if bad not in raw.info['bads']]

    return annotations, bads


def _check_ptp(ptp, name, info, picks):
    """Check the PTP threhsold argument, and converts it to dict if needed."""
    _validate_type(ptp, ('numeric', dict, None))

    if ptp is not None and not isinstance(ptp, dict):
        if ptp < 0:
            raise ValueError(
                f"Argument '{name}' should define a positive threshold. "
                f"Provided: '{ptp}'.")
        ch_types = set(_get_channel_types(info, picks))
        ptp = {ch_type: ptp for ch_type in ch_types}
    elif isinstance(ptp, dict):
        for key, value in ptp.items():
            if value < 0:
                raise ValueError(
                    f"Argument '{name}' should define positive thresholds. "
                    f"Provided for channel type '{key}': '{value}'.")
    return ptp


def _check_bad_percent(bad_percent):
    """Check that bad_percent is a valid percentage and converts to float."""
    _validate_type(bad_percent, 'numeric', 'bad_percent')
    bad_percent = float(bad_percent)
    if not 0 <= bad_percent <= 100:
        raise ValueError(
            "Argument 'bad_percent' should define a percentage between 0% "
            f"and 100%. Provided: {bad_percent}%.")
    return bad_percent


def _check_min_duration(min_duration, raw_duration):
    """Check that min_duration is a valid duration and converts to float."""
    _validate_type(min_duration, 'numeric', 'min_duration')
    min_duration = float(min_duration)
    if min_duration < 0:
        raise ValueError(
            "Argument 'min_duration' should define a positive duration in "
            f"seconds. Provided: '{min_duration}' seconds.")
    if min_duration >= raw_duration:
        raise ValueError(
            "Argument 'min_duration' should define a positive duration in "
            f"seconds shorter than the raw duration ({raw_duration} seconds). "
            f"Provided: '{min_duration}' seconds.")
    return min_duration


def _reject_short_segments(arr, min_duration_samples):
    """Check if flat or peak segments are longer than the minimum duration."""
    assert arr.dtype == bool and arr.ndim == 2
    for k, ch in enumerate(arr):
        onsets, offsets = _mask_to_onsets_offsets(ch)
        for start, stop in zip(onsets, offsets):
            if stop - start <= min_duration_samples:
                arr[k, start:stop] = False
    return arr


def _create_annotations(any_arr, type_, raw):
    """Create the peak of flat annotations from the any_arr."""
    assert type_ in ('peak', 'flat')
    starts, stops = _mask_to_onsets_offsets(any_arr)
    starts, stops = np.array(starts), np.array(stops)
    onsets = starts / raw.info['sfreq']
    durations = (stops - starts) / raw.info['sfreq']
    annot = Annotations(onsets, durations, [f'BAD_{type_}'] * len(onsets),
                        orig_time=raw.info['meas_date'])
    _adjust_onset_meas_date(annot, raw)
    return annot
