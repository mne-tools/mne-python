# License: BSD-3-Clause

import numpy as np

from ..annotations import (_annotations_starts_stops, Annotations,
                           _adjust_onset_meas_date)
from ..io import BaseRaw
from ..io.pick import (_picks_to_idx, _picks_by_type, _get_channel_types,
                       channel_type)
from ..utils import (_validate_type, verbose, logger, _mask_to_onsets_offsets,
                     ProgressBar)


@verbose
def annotate_amplitude(raw, peak=None, flat=None, bad_percent=5,
                       min_duration=0.005, picks=None, *, verbose):
    """Annotate segments of raw data which PTP amplitudes exceeds thresholds
    in ``peak`` or fall below thresholds in ``flat``.

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
        Defaults to 5 (5%%).
    min_duration : float
        The minimum duration (sec) to consider as above or below threshold.
        For some systems with low bit data representations, adjacent
        time samples with exactly the same value are not totally uncommon.
        Defaults to 0.005 (5 ms).
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
    This function may perform much faster if data are loaded in memory, as it
    loads data one channel at a time (across all time points), which is
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
    min_duration = _check_min_duration(min_duration, raw.times[-1])
    min_duration_samples = int(np.round(min_duration * raw.info['sfreq']))
    bads = list()

    # grouping picks by channel types to avoid operating on each channel
    # individually
    picks = {
        ch_type: np.intersect1d(picks_of_type, picks_, assume_unique=True)
        for ch_type, picks_of_type in _picks_by_type(raw.info, exclude='bads')
        }
    del picks_  # re-using this variable name below

    # look for discrete difference above or below thresholds
    logger.info('Finding segments below or above PTP threshold.')
    for ch_type, picks_ in picks.items():
        diff = np.abs(np.diff(raw._data[picks_, :], axis=1))

        if flat is not None:
            flat_ = diff <= flat[ch_type]
            # reject too short segments
            for start, stop in _2dim_mask_to_onsets_offsets(flat_):
                if stop - start < min_duration_samples:
                    flat_[start:stop] = False
            # reject channels above maximum bad_percentage
            flat_mean = flat_.mean(axis=1) * 100
            flat_ch = picks_[np.where(flat_mean >= bad_percent)[0]]
            bads.extend(flat_ch)

        if peak is not None:
            peak_ = diff >= peak[ch_type]
            # reject too short segments
            for start, stop in _2dim_mask_to_onsets_offsets(peak_):
                if stop - start < min_duration_samples:
                    peak_[start:stop] = False
            # reject channels above maximum bad_percentage
            peak_mean = peak_.mean(axis=1) * 100
            peak_ch = picks_[np.where(peak_mean >= bad_percent)[0]]
            bads.extend(peak_ch)


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
                    "Provided for channel type '{key}': '{value}'.")
    return ptp


def _check_bad_percent(bad_percent):
    """Check that bad_percent is a valid percentage and converts to float."""
    bad_percent = float(bad_percent)
    if not 0 <= bad_percent <= 100:
        raise ValueError(
            "Argument 'bad_percent' should define a percentage between 0% "
            f"and 100%. Provided: {bad_percent}%.")
    return bad_percent


def _check_min_duration(min_duration, raw_duration):
    """Check that min_duration is a valid duration and converts to float."""
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


def _2dim_mask_to_onsets_offsets(mask):
    """
    Similar to utils.numerics._mask_to_onsets_offsets but for 2D mask
    (n_channels, n_samples).

    Examples
    --------
    >>> mask = np.zeros((3, 10))
    >>> mask[0, 2:6] = 1.
    >>> mask[0, 7:9] = 1.
    >>> mask[1, :4] = 1.
    >>> mask[1, 8:] = 1.
    >>> mask[2, 2:8] = 1.

    >>> mask
    array([[0., 0., 1., 1., 1., 1., 0., 1., 1., 0.],
           [1., 1., 1., 1., 0., 0., 0., 0., 1., 1.],
           [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.]])

    >>> np.diff(mask)
    array([[ 0.,  1.,  0.,  0.,  0., -1.,  1.,  0., -1.],
           [ 0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.]])

    >>> onsets_offsets
    [(0, 4), (2, 6), (2, 8), (7, 9), (8, 10)]
    """
    assert mask.dtype == bool and mask.ndim == 2
    mask = mask.astype(int)
    diff = np.diff(mask)
    onsets = np.where(diff > 0)[1] + 1
    if any(mask[:, 0]):
        onsets = np.concatenate([[0], onsets])
    offsets = np.where(diff < 0)[1] + 1
    if any(mask[:, -1]):
        offsets = np.concatenate([offsets, [mask.shape[-1]]])
    assert len(onsets) == len(offsets)
    return onsets, offsets
