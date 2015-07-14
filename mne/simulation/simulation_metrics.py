# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def _check_stc(stc1, stc2):
    # TODO Add checks for source space
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')


def source_estimate_quantification(stc1, stc2, metric='rms', src=None):
    """Helper function to calculate matrix similarities.

    Parameters
    ----------
    stc1 : SourceEstimate
        First source estimate for comparison
    stc2 : SourceEstimate
        First source estimate for comparison
    metric : str
        Metric to calculate. 'rms', 'corr',
    src : None | list of dict
        The source space. The default value is None. It must be provided when
        using those metrics: "distance_err", "weighted_distance_err"

    Returns
    -------
    score : float | array
        Calculated metric
    """
    _check_stc(stc1, stc2)
    # This is checking that the datas are having the same size meaning
    # no comparison between distributed and sparse can be done so far.
    data1, data2 = stc1.data, stc2.data

    # Calculate root mean square difference between two matrices
    if metric == 'rms':
        return np.sqrt(np.mean((data1 - data2) ** 2))

    # Calculate root mean square difference between two normalized matrices
    elif metric == 'rms_normed':
        data1 = data1 / np.max(data1)
        data2 = data2 / np.max(data2)
        return np.sqrt(np.mean((data1 - data2) ** 2))

    # Calculate correlation coefficient between matrix elements
    elif metric == 'corr':
        return np.correlate(data1.flatten(), data2.flatten())

    # Calculate distance error between the vertices.
    # Will not have anysense in case where the vertices of the whole cortex
    # are present. That case, use the weighted distance error.
    # Usefull to check how far dipoles are different.
    elif metric == 'distance_err':
        dist_lh = src[0]['rr'][stc1.vertices[0]] - \
            src[0]['rr'][stc2.vertices[0]]
        dist_rh = src[1]['rr'][stc1.vertices[1]] - \
            src[1]['rr'][stc2.vertices[1]]

        return np.concatenate([dist_lh, dist_rh])

    # Calculate the distance error weighted by the amplitude at each time point
    elif metric == "weighted_distance_err":
        dist_lh = src[0]['rr'][stc1.vertices[0]] - \
            src[0]['rr'][stc2.vertices[0]]
        dist_rh = src[1]['rr'][stc1.vertices[1]] - \
            src[1]['rr'][stc2.vertices[1]]
        dist_err = np.concatenate([dist_lh, dist_rh])

        weights = np.abs(data1 - data2)
        score = list()
        [score.append(dist_err * np.tile(w, (3, 1)).T) for w in weights.T]
        return score


# score = _calc_metric(stc1.data, stc2.data, metric, src)
# return score

# def _calc_metric(data1, data2, metric, src=None):
#     """Helper to calculate metric of choice.

#     Parameters
#     ----------
#     data1 : ndarray, shape(n_sources, ntimes)
#         Second data matrix
#     data2 : ndarray, shape(n_sources, ntimes)
#         Second data matrix
#     metric : str
#         Metric to calculate. 'rms', 'rms_normed', 'corr',
#     src : None | list of dict
#         The source space. The default value is None. It must be provided when
#         using those metrics: "distance_err", "weighted_distance_err"

#     Returns
#     -------
#     score : float
#         Calculated metric
#     """
