# Author: David Julien <david.julien@ifsttar.fr>
#
# License: BSD (3-clause)

import numpy as np

import warnings

from ..annotations import Annotations
from ..utils import _mask_to_onsets_offsets, verbose, warn


@verbose
def annotate_nan(raw, verbose=None):
    """Detect segments with NaN and return a new Annotations instance.

    Parameters
    ----------
    raw : instance of Raw
        Data to find segments with NaN values.
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        Updated annotations for raw data.
    """
    annot = raw.annotations.copy()
    data, times = raw.get_data(return_times=True)
    sampling_duration = 1 / raw.info['sfreq']

    nans = np.any(np.isnan(data), axis=0)
    starts, stops = _mask_to_onsets_offsets(nans)

    if len(starts) == 0:
        warn("The dataset you provided does not contain 'NaN' values. "
             "No annotations were made.")
        return annot

    starts, stops = np.array(starts), np.array(stops)
    onsets = (starts + raw.first_samp) * sampling_duration
    durations = (stops - starts) * sampling_duration

    annot.append(onsets, durations, 'bad_NAN')

    return annot
