# Authors: Yousra Bekhti
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def _check_stc(stc1, stc2):
    # XXX What should we check? that the data is having the same size?
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if stc1.times != stc2.times:
        raise ValueError('Times of two stcs must match.')



def source_estimate_quantification(stc1, stc2, metric='rms'):
    """Helper function to calculate matrix similarities.

    Parameters
    ----------
    stc1 : SourceEstimate
        First source estimate for comparison
    stc2 : SourceEstimate
        First source estimate for comparison
    metric : str
        Metric to calculate. 'rms', 'avg_corrcoef',

    Returns
    -------

    """

    # TODO Add checks for source space
    _check_stc(stc1, stc2)

    score = _calc_metric(stc1.data, stc2.data, metric)

def _calc_metric(data1, data2, metric):
    """Helper to calculate metric of choice.

    Parameters
    ----------
    data1 : ndarray, shape(n_sources, ntimes)
        Second data matrix
    data2 : ndarray, shape(n_sources, ntimes)
        Second data matrix
    metric : str
        Metric to calculate. 'rms', 'corr',

    Returns
    -------
    score : float
        Calculated metric
    """

    # Calculate root mean square difference between two matrices
    if metric == 'rms':
        return np.mean((stc1.data - stc2.data) ** 2)

    # Calculate correlation coefficient between matrix elements
    elif metric == 'corr':
        return np.correlate(stc1.data.flatten(), stc2.data.flatten())
