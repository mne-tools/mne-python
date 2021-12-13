# Author: David Julien <david.julien@ifsttar.fr>
#
# License: BSD-3-Clause

import numpy as np

from ..annotations import Annotations, _adjust_onset_meas_date
from ..utils import verbose
from .artifact_detection import _annotations_from_mask


@verbose
def annotate_nan(raw, *, verbose=None):
    """Detect segments with NaN and return a new Annotations instance.

    Parameters
    ----------
    raw : instance of Raw
        Data to find segments with NaN values.
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        New channel-specific annotations for the data.
    """
    data, times = raw.get_data(return_times=True)
    onsets, durations, ch_names = list(), list(), list()
    for row, ch_name in zip(data, raw.ch_names):
        annot = _annotations_from_mask(times, np.isnan(row), 'BAD_NAN')
        onsets.extend(annot.onset)
        durations.extend(annot.duration)
        ch_names.extend([[ch_name]] * len(annot))
    annot = Annotations(onsets, durations, 'BAD_NAN', ch_names=ch_names,
                        orig_time=raw.info['meas_date'])
    _adjust_onset_meas_date(annot, raw)
    return annot
