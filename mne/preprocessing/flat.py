# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..io import BaseRaw
from ..io.pick import _picks_to_idx
from ..utils import (_validate_type, verbose, logger, _pl,
                     _mask_to_onsets_offsets, ProgressBar)


@verbose
def mark_flat(raw, ratio=1., min_duration=0.005, picks=None, verbose=None):
    r"""Mark flat segments of raw data using annotations or in info['bads'].

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    ratio : float
        The ratio of the temporal proportion (time spent flat) to spatial
        proportion (the reciprocal of the number of channels in picks)
        below which temporal bad marking (:class:`~mne.Annotations`) will be
        used instead of spatial bad marking (:class:`info['bads'] <mne.Info>`).
        See Notes.
    min_duration : float
        The minimum duration (sec) to consider as actually flat.
        For some systems with low bit data representations, adjacent
        channels with exactly the same value are not totally uncommon.
    %(picks_good_data)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The modified raw instance. Operates in place.

    Notes
    -----
    For a given channel :math:`i` in `picks` that is flat (considering only
    flat segments of at least `min_duration`)
    for :math:`p_{i,t}` proportion of time points, given the spatial fraction
    of picks that the channel represents (constant
    :math:`p_{i,s} = 1 / N_\mathrm{picks}`), the `ratio`
    parameter sets the ratio :math:`\frac{p_{i,t}}{p_{i,s}}` below which
    temporal marking (:class:`~mne.Annotations`) will be used instead of
    spatial marking (:clasS:`~mne.Info`). Or more simply:

    - ``ratio < 1. / len(picks)`` always marks temporally.
        - This risks the entire recording being marked bad if there is
          (at least) one channel that is flat at all times.
    - ``1. / len(picks) <= ratio < len(picks)`` marks spatially or temporally.
        - Values in the range ``0 < ratio < 1`` prefer temporal marking to
          spatial marking.
        - The default value ``ratio = 1.`` uses the temporal method
          if and only if the proportion of time a given channel is flat is
          less than the reciprocal of the number of channels in ``picks``.
        - Values in the range ``1 < ratio < len(picks)`` prefer spatial
          marking to temporal marking.
    - ``ratio >= len(picks)`` always marks spatially.
        - This risks a channel being marked as bad for the entire recording if
          it is flat for a contiguous `min_duration` chunk of time.

    .. note:: This function may perform much faster if data are loaded
              in memory, as it loads data one channel at a time (across all
              time points), which is typically not an efficient way to read
              raw data from disk.

    .. versionadded:: 0.18
    """
    _validate_type(raw, BaseRaw, 'raw')
    ratio = float(ratio)
    min_duration = float(min_duration)
    picks = _picks_to_idx(raw.info, picks, 'data_or_ica', exclude='bads')
    # This will not be so efficient for most readers, but we can optimize
    # it later
    any_flat = np.zeros(len(raw.times), bool)
    bads = list()
    time_thresh = int(np.round(min_duration * raw.info['sfreq']))
    with ProgressBar(picks, mesg='Finding flat segments', spinner=True,
                     verbose_bool='auto') as pb:
        for pick in pb:
            data = raw[pick][0][0]
            flat = np.diff(data) == 0
            flat = np.concatenate(
                [flat[[0]], flat[1:] | flat[:-1], flat[[-1]]])
            starts, stops = _mask_to_onsets_offsets(flat)
            for start, stop in zip(starts, stops):
                if stop - start < time_thresh:
                    flat[start:stop] = False
            flat_mean = flat.mean()
            with np.errstate(divide='ignore'):
                bad_ratio = np.divide(1., flat_mean * len(picks))
            if flat_mean:  # only do something if there are actually flat parts
                if bad_ratio < ratio:
                    kind, comp = 'bads', '<'
                    bads.append(raw.ch_names[pick])
                else:
                    kind, comp = 'BAD_', '≥'
                    any_flat |= flat
                logger.debug('%s: %s (%s %s %s)' % (kind, raw.ch_names[pick],
                                                    bad_ratio, comp, ratio))
    starts, stops = _mask_to_onsets_offsets(any_flat)
    logger.info('Marking %0.2f%% of time points (%d segment%s) and '
                '%d/%d channel%s bad: %s'
                % (100 * any_flat.mean(), len(starts), _pl(starts),
                   len(bads), len(picks), _pl(bads), bads))
    add_bads = [bad for bad in bads if bad not in raw.info['bads']]
    raw.info['bads'] = list(raw.info['bads']) + add_bads
    if len(starts) > 0:
        starts, stops = np.array(starts), np.array(stops)
        onsets = starts / raw.info['sfreq']
        durations = (stops - starts) / raw.info['sfreq']
        raw.annotations.append(onsets, durations, ['BAD_flat'] * len(onsets))
    return raw
