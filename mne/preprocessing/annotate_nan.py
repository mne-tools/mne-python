# Author: David Julien <david.julien@ifsttar.fr>
#
# License: BSD (3-clause)

import numpy as np

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
        Updated annotations for raw data.
    """
    data, times = raw.get_data(return_times=True)
    nans = np.any(np.isnan(data), axis=0)
    annot = _annotations_from_mask(times, nans, 'BAD_NAN')
    return annot
