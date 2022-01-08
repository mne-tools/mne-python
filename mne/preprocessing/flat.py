# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from .annotate_amplitude import annotate_amplitude
from ..utils import verbose, deprecated


@verbose
@deprecated('use mne.preprocessing.annotate_amplitude instead.')
def annotate_flat(raw, bad_percent=5., min_duration=0.005, picks=None,
                  verbose=None):
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
    return annotate_amplitude(raw, None, 0., bad_percent, min_duration, picks,
                              verbose=verbose)
