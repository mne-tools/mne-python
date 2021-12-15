# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ..annotations import (_annotations_starts_stops, Annotations,
                           _adjust_onset_meas_date)
from ..io import BaseRaw
from ..io.pick import _picks_to_idx, _get_channel_types, channel_type
from ..utils import (_validate_type, verbose, logger, _pl,
                     _mask_to_onsets_offsets, ProgressBar)


@verbose
def annotate_flat(raw, bad_percent=5., min_duration=0.005, picks=None,
                  flatness=0, *, verbose=None):
    """Annotate flat segments of raw data (or add to a bad channel list).

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    bad_percent : float
        The percentage of the time a channel can be bad.
        Below this percentage, temporal bad marking (:class:`~mne.Annotations`)
        will be used. Above this percentage, spatial bad marking
        (:class:`info['bads'] <mne.Info>`) will be used.
        Defaults to 5 (5%%).
    min_duration : float
        The minimum duration (sec) to consider as actually flat.
        For some systems with low bit data representations, adjacent
        time samples with exactly the same value are not totally uncommon.
        Defaults to 0.005 (5 ms).
    %(picks_good_data)s
    flatness : dict | float
        Reject segments based on **minimum** peak-to-peak signal amplitude
        (PTP). Valid **keys** can be any channel type present in the object.
        The **values** are floats that set the minimum acceptable PTP. If the
        PTP is smaller than this threshold, the segment will be annotated.
        If float, the minimum acceptable PTP is applied to all channels.
        Default to ``0.`` to annotate completely flat segments.

        .. versionadded:: 1.0
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        The annotated bad segments.
    bads : list
        The channels detected as bad.

    Notes
    -----
    This function is useful both for removing short segments of data where
    the acquisition system clipped (i.e., hit the ADC limit of the hardware)
    and for automatically identifying channels that were flat for a large
    proportion of a given recording.

    This function may perform much faster if data are loaded
    in memory, as it loads data one channel at a time (across all
    time points), which is typically not an efficient way to read
    raw data from disk.

    .. versionadded:: 0.18
    """
    _validate_type(raw, BaseRaw, 'raw')
    bad_percent = float(bad_percent)
    min_duration = float(min_duration)
    picks = _picks_to_idx(raw.info, picks, 'data_or_ica', exclude='bads')
    flatness = _check_flatness(flatness, raw.info, picks)
    # This will not be so efficient for most readers, but we can optimize
    # it later
    any_flat = np.zeros(len(raw.times), bool)
    bads = list()
    time_thresh = int(np.round(min_duration * raw.info['sfreq']))
    onsets, ends = _annotations_starts_stops(raw, 'bad_acq_skip', invert=True)
    idx = np.concatenate([np.arange(onset, end)
                          for onset, end in zip(onsets, ends)])
    logger.info('Finding flat segments')
    for pick in ProgressBar(picks, mesg='Channels'):
        data = np.concatenate([raw[pick, onset:end][0][0]
                               for onset, end in zip(onsets, ends)])
        flat = np.diff(data) <= flatness[channel_type(raw.info, pick)]
        flat = np.concatenate(
            [flat[[0]], flat[1:] | flat[:-1], flat[[-1]]])
        starts, stops = _mask_to_onsets_offsets(flat)
        for start, stop in zip(starts, stops):
            if stop - start < time_thresh:
                flat[start:stop] = False
        flat_mean = flat.mean()
        if flat_mean:  # only do something if there are actually flat parts
            flat_mean *= 100
            if flat_mean > bad_percent:
                kind, comp = 'bads', '>'
                bads.append(raw.ch_names[pick])
            else:
                kind, comp = 'BAD_', '≤'
                any_flat[idx] |= flat
            logger.debug('%s: %s (%s %s %s)'
                         % (kind, raw.ch_names[pick],
                            flat_mean, comp, bad_percent))
    starts, stops = _mask_to_onsets_offsets(any_flat)
    logger.info('Marking %0.2f%% of time points (%d segment%s) and '
                '%d/%d channel%s bad%s'
                % (100 * any_flat[idx].mean(), len(starts), _pl(starts),
                   len(bads), len(picks), _pl(bads),
                   (': %s' % (bads,)) if bads else ''))
    bads = [bad for bad in bads if bad not in raw.info['bads']]
    starts, stops = np.array(starts), np.array(stops)
    onsets = starts / raw.info['sfreq']
    durations = (stops - starts) / raw.info['sfreq']
    annot = Annotations(onsets, durations, ['BAD_flat'] * len(onsets),
                        orig_time=raw.info['meas_date'])
    _adjust_onset_meas_date(annot, raw)
    return annot, bads


def _check_flatness(flatness, info, picks):
    """Check the flatness argument, and converts it to dict if needed."""
    _validate_type(flatness, ('numeric', dict), 'flatness')

    if not isinstance(flatness, dict):
        if flatness < 0:
            raise ValueError(
                "Argument 'flatness' should be a positive threshold. "
                f"Provided: '{flatness}'.")
        ch_types = _get_channel_types(info, picks)
        flatness = {ch_type: flatness for ch_type in ch_types}
    else:
        for key, value in flatness.items():
            if value < 0:
                raise ValueError(
                    "Argument 'flatness' should define positive thresholds. "
                    "Provided for channel type '{key}': '{value}'.")
    return flatness
