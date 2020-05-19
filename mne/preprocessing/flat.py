# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..annotations import _annotations_starts_stops
from ..io import BaseRaw
from ..io.pick import _picks_to_idx
from ..utils import (_validate_type, verbose, logger, _pl,
                     _mask_to_onsets_offsets, ProgressBar)


@verbose
def mark_flat(raw, bad_percent=5., min_duration=0.005, picks=None,
              verbose=None):
    r"""Mark flat segments of raw data using annotations or in info['bads'].

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
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The modified raw instance. Operates in place.

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
        flat = np.diff(data) == 0
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
    add_bads = [bad for bad in bads if bad not in raw.info['bads']]
    raw.info['bads'] = list(raw.info['bads']) + add_bads
    if len(starts) > 0:
        starts, stops = np.array(starts), np.array(stops)
        onsets = (starts + raw.first_samp) / raw.info['sfreq']
        durations = (stops - starts) / raw.info['sfreq']
        raw.annotations.append(onsets, durations, ['BAD_flat'] * len(onsets))
    return raw
