# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD-3-Clause

import numpy as np

from ..utils import _check_option
from mne.utils import deprecated


def _check_stc(stc1, stc2):
    """Check that stcs are compatible."""
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')


@deprecated("This function is deprecated and will be removed in version 1.3, "
            "use the mne.simulation.metrics module instead.")
def source_estimate_quantification(stc1, stc2, metric='rms'):
    """Calculate matrix similarities.

    Parameters
    ----------
    stc1 : SourceEstimate
        First source estimate for comparison.
    stc2 : SourceEstimate
        Second source estimate for comparison.
    metric : str
        Metric to calculate, 'rms' or 'cosine'.

    Returns
    -------
    score : float | array
        Calculated metric.

    Notes
    -----
    Metric calculation has multiple options:

        * rms: Root mean square of difference between stc data matrices.
        * cosine: Normalized correlation of all elements in stc data matrices.

    .. versionadded:: 0.10.0
    """
    _check_option('metric', metric, ['rms', 'cosine'])

    # This is checking that the datas are having the same size meaning
    # no comparison between distributed and sparse can be done so far.
    _check_stc(stc1, stc2)
    data1, data2 = stc1.data, stc2.data

    # Calculate root mean square difference between two matrices
    if metric == 'rms':
        score = np.sqrt(np.mean((data1 - data2) ** 2))

    # Calculate correlation coefficient between matrix elements
    elif metric == 'cosine':
        score = 1. - (np.dot(data1.flatten(), data2.flatten()) /
                      (np.linalg.norm(data1) * np.linalg.norm(data2)))
    return score
