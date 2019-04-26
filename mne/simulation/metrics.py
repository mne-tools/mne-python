# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import norm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_distances
from ..utils import _check_option

# TODO: Add more localization accuracy functions. For example, distance between
#       true dipole position (in simulated stc) and the centroid of the
#       estimated activity.


def _check_stc(stc1, stc2):
    """Check that stcs are compatible."""
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')

def _uniform_stc(stc1, stc2):
    if len(stc1.vertices) != len(stc2.vertices):
        raise ValueError('Data in stcs must have the same number of vertices components. '
                         'Got %d != %d.' % (len(stc1.vertices), len(stc2.vertices)))
    idx_start1 = 0
    idx_start2 = 0
    stc1 = stc1.copy()
    stc2 = stc2.copy()
    all_data1 = []
    all_data2 = []
    for i, (vert1, vert2) in enumerate(zip(stc1.vertices, stc2.vertices)):
        vert  = np.union1d(vert1, vert2)
        data1 = np.zeros([len(vert), stc1.data.shape[1]])
        data2 = np.zeros([len(vert), stc2.data.shape[1]])
        data1[np.searchsorted(vert, vert1)] = stc1.data[idx_start1:idx_start1 + len(vert1)]
        data2[np.searchsorted(vert, vert2)] = stc2.data[idx_start2:idx_start2 + len(vert2)]
        idx_start1 += len(vert1)
        idx_start2 += len(vert2)
        stc1.vertices[i] = vert
        stc2.vertices[i] = vert
        all_data1.append(data1)
        all_data2.append(data2)

    stc1._data = np.concatenate(all_data1, axis=0)
    stc2._data = np.concatenate(all_data2, axis=0)
    return stc1, stc2


def _apply(func, P, Q, per_sample):
    if per_sample:
        metric = np.zeros(P.data.shape[1])
        for i in range(P.data.shape[1]):
            metric[i] = func(P.data[:, i:i + 1], Q.data[:, i:i + 1])
    else:
        metric = func(P.data, Q.data)
    return metric


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
                      (norm(data1) * norm(data2)))
    return score

def _kl(x, y):
    p = np.reshape(x, (-1, 1))
    q = np.reshape(y, (-1, 1))

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def stc_kl(stc_true, stc_est, per_sample=True):
    P, Q = _uniform_stc(stc_true, stc_est)
    func = partial(_kl)
    metric = _apply(func, P, Q, per_sample=per_sample)
    return metric

def _cosine(x, y):
    p = np.reshape(x, (-1, 1))
    q = np.reshape(y, (-1, 1))
    return np.dot(p.T, q) / (norm(p) * norm(q))

def stc_cosine(stc_true, stc_est, per_sample=True):
    P, Q = _uniform_stc(stc_true, stc_est)
    func = partial(_cosine)
    metric = _apply(func, P, Q, per_sample=per_sample)
    return metric

def _check_threshold(x):
    if isinstance(x, str):
        return float(x.strip('%')) / 100.0
    else:
        return x


def _dle(p, q, src, stc):
    p = np.sum(p, axis=1)
    q = np.sum(q, axis=1)
    idx1 = np.nonzero(p)[0]
    idx2 = np.nonzero(q)[0]
    points = np.empty([0, 3], dtype=float)
    for i in range(2):
        points = np.vstack([points, src[i]['rr'][stc.vertices[i]]])
    if len(idx1) and len(idx2):
        D = pairwise_distances(points[idx1], points[idx2])
        D_min_1 = np.min(D, axis=0)
        D_min_2 = np.min(D, axis=1)
        return (np.mean(D_min_1) + np.mean(D_min_2)) / 2.
    elif len(idx1):
        return -np.inf
    else:
        return np.inf



def stc_dipole_localization_error(stc_true, stc_est, src, threshold='90%', per_sample=True):
    P, Q = _uniform_stc(stc_true, stc_est)
    if isinstance(threshold, str):
        t = float(threshold.strip('%')) / 100.0
        P._data[np.where(np.abs(P._data) <= t * np.max(np.abs(P._data)))] = 0
        Q._data[np.where(np.abs(Q._data) <= t * np.max(np.abs(Q._data)))] = 0
    else:
        t = threshold
        P._data[np.where(np.abs(P._data) <= t)] = 0
        Q._data[np.where(np.abs(Q._data) <= t)] = 0
    print(P.data)
    print(Q.data)
    func = partial(_dle, src=src, stc=P)
    metric = _apply(func, P, Q, per_sample=per_sample)
    return metric


def _roc_auc_score(p, q):
    return roc_auc_score(np.abs(p) > 0, q)


def stc_roc_auc_score(stc_true, stc_est, per_sample=True):
    P, Q = _uniform_stc(stc_true, stc_est)
    func = partial(_roc_auc_score)
    metric = _apply(func, P, Q, per_sample=per_sample)
    return metric

