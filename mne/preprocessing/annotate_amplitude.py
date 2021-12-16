# License: BSD-3-Clause

import numpy as np

from ..annotations import (_annotations_starts_stops, Annotations,
                           _adjust_onset_meas_date)
from ..io import BaseRaw
from ..io.pick import _picks_to_idx, _get_channel_types, channel_type
from ..utils import (_validate_type, verbose, logger, _mask_to_onsets_offsets,
                     ProgressBar)


@verbose
def annotate_amplitude(raw, peak, flat, bad_percent=5, min_duration=0.005,
                       picks=None, *, verbose):
    """Annotate segments of raw data which PTP amplitudes exceeds thresholds
    in ``peak`` or fall below thresholds in ``flat``.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    peak : float | dict
        Reject segments based on **maximum** peak-to-peak signal amplitude
        (PTP). Valid **keys** can be any channel type present in the object.
        The **values** are floats that set the minimum acceptable PTP. If the
        PTP is smaller than this threshold, the segment will be annotated.
        If float, the minimum acceptable PTP is applied to all channels.
    flat : float | dict
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
    picks = _picks_to_idx(raw.info, picks, 'data_or_ica', exclude='bads')
    peak = _check_ptp(peak, 'peak', raw.info, picks)
    flat = _check_ptp(flat, 'flat', raw.info, picks)
    bad_percent = float(bad_percent)
    min_duration = float(min_duration)
    min_duration_samples = int(np.round(min_duration * raw.info['sfreq']))
    bads = list()

    logger.info('Finding segments below or above PTP threshold.')


def _check_ptp(ptp, name, info, picks):
    """Check the PTP threhsold argument, and converts it to dict if needed."""
    _validate_type(ptp, ('numeric', dict))

    if not isinstance(ptp, dict):
        if ptp < 0:
            raise ValueError(
                f"Argument '{name}' should define a positive threshold. "
                f"Provided: '{ptp}'.")
        ch_types = _get_channel_types(info, picks)
        ptp = {ch_type: ptp for ch_type in ch_types}
    else:
        for key, value in ptp.items():
            if value < 0:
                raise ValueError(
                    f"Argument '{name}' should define positive thresholds. "
                    "Provided for channel type '{key}': '{value}'.")
    return ptp
