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
<<<<<<< HEAD:mne/simulation/simulation_metrics.py
        Metric to calculate. 'rms', 'rms_normed', 'corr', 'distance_err', or
        'weighted_distance_err'.
=======
        Metric to calculate. 'rms', 'rms_normed', 'corr', 'distance_err',
        'weighted_distance_err', ...
>>>>>>> Fix tests + Eric's comments: 1st pass:mne/simulation/metrics.py
    src : None | list of dict
        The source space. The default value is None. It must be provided when
        using those metrics: "distance_err", "weighted_distance_err"

    Returns
    -------
    score : float | array
        Calculated metric

    Notes
    -----
    Metric calculation has multiple options:
        rms: Root mean square of difference between stc data matrices
        rms_normed: Root mean square of difference between (activity
            normalized) stc data matrices
        corr: Correlation of all elements in stc data matrices
        distance_err: Distance between most active dipoles
        weighted_distance_err: Distance between most active dipoles weighted by
            difference in activity
    """
    known_metrics = ['rms', 'rms_normed', 'corr', 'distance_err',
                     'weighted_distance_err']
    if metric not in known_metrics:
        raise ValueError('metric must be a str from the known metrics: '
                         '"rms", "rms_normed", "corr", "distance_err", '
                         '"weighted_distance_err" or "..."')

    # This is checking that the datas are having the same size meaning
    # no comparison between distributed and sparse can be done so far.
    _check_stc(stc1, stc2)
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
    # or a full label are present. That case, use the weighted distance error.
    # Usefull to check how far dipoles are from each other.
    elif metric == 'distance_err':

        pos_concat = np.c_[src[0]['rr'], src[1]['rr']]

        # Get vertex inds that need distance comparison
        stc1_verts = get_largest_n(1, stc1.data)
        stc2_verts = get_largest_n(1, stc2.data)

        # Get distances between vertices needed
        #verts =
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

def closest_dipole_pos(point, src):
    """Find the closest dipole position to a point in space."""
    # TODO: Is this function general enough to go somewhere else?
    # Variant used in mne.label.split_label
    from scipy.spatial.distance import cdist
    if src['dist'] is None:
        raise RuntimeError('Source space distances must exist to calculate '
                           'nearest dipole to a point in space')
    pos_concat = np.c_[src[0]['rr'], src[1]['rr']]

    # Calculate distance to all points
    distance = cdist(point, pos_concat)
    return pos_concat[np.argmin(distance)]


def get_largest_n(n_pts, data):
    inds = np.argpartition(data, n_pts, axis=1)
    return data(inds)

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
