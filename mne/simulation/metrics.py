# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#          Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
from scipy.linalg import norm
from ..utils import _check_option, fill_doc

# TODO: Add more localization accuracy functions. For example, distance between
#       true dipole position (in simulated stc) and the centroid of the
#       estimated activity.


def _check_stc(stc1, stc2):
    """Check that stcs are compatible."""
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')


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


def _uniform_stc(stc1, stc2):
    """This function returns the stcs with the same vertices by
    inserting zeros in data for missing vertices."""
    if len(stc1.vertices) != len(stc2.vertices):
        raise ValueError('Data in stcs must have the same number of vertices '
                         'components. Got %d != %d.' %
                         (len(stc1.vertices), len(stc2.vertices)))
    idx_start1 = 0
    idx_start2 = 0
    stc1 = stc1.copy()
    stc2 = stc2.copy()
    all_data1 = []
    all_data2 = []
    for i, (vert1, vert2) in enumerate(zip(stc1.vertices, stc2.vertices)):
        vert = np.union1d(vert1, vert2)
        data1 = np.zeros([len(vert), stc1.data.shape[1]])
        data2 = np.zeros([len(vert), stc2.data.shape[1]])
        data1[np.searchsorted(vert, vert1)] = \
            stc1.data[idx_start1:idx_start1 + len(vert1)]
        data2[np.searchsorted(vert, vert2)] = \
            stc2.data[idx_start2:idx_start2 + len(vert2)]
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
    """Applies a metric to each pair of columns of P and Q
    if per_sample is True. Otherwise it applies it to P and Q
    directly."""
    if per_sample:
        metric = np.zeros(P.data.shape[1])  # one value per time point
        for i in range(P.data.shape[1]):
            metric[i] = func(P.data[:, i:i + 1], Q.data[:, i:i + 1])
    else:
        metric = func(P.data, Q.data)
    return metric


def _thresholding(stc_true, stc_est, threshold):
    stc_true._data = np.abs(stc_true._data)
    stc_est._data = np.abs(stc_est._data)
    if isinstance(threshold, str):
        t = _check_threshold(threshold)
        stc_true._data[stc_true._data <= t * np.max(stc_true._data)] = 0.
        stc_est._data[stc_est._data <= t * np.max(stc_est._data)] = 0.
    else:
        stc_true._data[stc_true._data <= threshold] = 0.
        stc_est._data[stc_est._data <= threshold] = 0.
    return stc_true, stc_est


def _cosine(x, y):
    p = np.reshape(x, (-1, 1))
    q = np.reshape(y, (-1, 1))
    return np.dot(p.T, q) / (norm(p) * norm(q))


@fill_doc
def stc_cosine(stc_true, stc_est, per_sample=True):
    """Compute cosine similarity between 2 source estimates

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    P, Q = _uniform_stc(stc_true, stc_est)
    func = partial(_cosine)
    metric = _apply(func, P, Q, per_sample=per_sample)
    return metric


def _check_threshold(threshold):
    """Accepts a float or a string that ends with %"""
    if isinstance(threshold, str):
        if threshold.endswith("%"):
            return float(threshold[:-1]) / 100.0
        else:
            raise ValueError('Threshold if a string must end with '
                             '"%%". Got %s.' % threshold)
    else:
        return threshold


def _dle(p, q, src, stc):
    """Aux function to compute dipole localization error"""
    from sklearn.metrics import pairwise_distances
    p = np.sum(p, axis=1)
    q = np.sum(q, axis=1)
    idx1 = np.nonzero(p)[0]
    idx2 = np.nonzero(q)[0]
    points = []
    for i in range(len(src)):
        points.append(src[i]['rr'][stc.vertices[i]])
    points = np.concatenate(points, axis=0)
    if len(idx1) and len(idx2):
        D = pairwise_distances(points[idx1], points[idx2])
        D_min_1 = np.min(D, axis=0)
        D_min_2 = np.min(D, axis=1)
        return (np.mean(D_min_1) + np.mean(D_min_2)) / 2.
    else:
        return np.inf


@fill_doc
def stc_dipole_localization_error(stc_true, stc_est, src, threshold='90%',
                                  per_sample=True):
    """Compute dipole localization error (DLE) between 2 source estimates

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    src : instance of SourceSpaces
        The source space on which the source estimates are defined.
    threshold : float | str
        The threshold to apply to source estimates before computing
        the dipole localization error. If a string the threshold is
        a percentage and it should end with the percent character.
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    _thresholding(stc_true, stc_est, threshold)
    func = partial(_dle, src=src, stc=stc_true)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _roc_auc_score(p, q):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.abs(p) > 0, np.abs(q))


@fill_doc
def stc_roc_auc_score(stc_true, stc_est, per_sample=True):
    """Compute ROC AUC between 2 source estimates

    ROC stands for receiver operating curve and AUC is Area under the curve.
    When computing this metric the stc_true must be thresholded
    as any non-zero value will be considered as a positive.


    The ROC-AUC metric is computed between amplitudes of the source
    estimates, i.e. after taking the absolute values.

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    func = partial(_roc_auc_score)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _f1_score(p, q):
    from sklearn.metrics import f1_score
    p = np.sum(p, axis=1)
    q = np.sum(q, axis=1)
    return f1_score(np.abs(p) > 0, np.abs(q) > 0)


@fill_doc
def stc_f1_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the F1 score, also known as balanced F-score or F-measure

    The F1 score can be interpreted as a weighted average of the precision and recall,
    where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are equal.
    The formula for the F1 score is:

        F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the dipole localization error. If a string the threshold is
        a percentage and it should end with the percent character.
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    _thresholding(stc_true, stc_est, threshold)
    func = partial(_f1_score)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _precision_score(p, q):
    from sklearn.metrics import precision_score
    p = np.sum(p, axis=1)
    q = np.sum(q, axis=1)
    return precision_score(np.abs(p) > 0, np.abs(q) > 0)


@fill_doc
def stc_precision_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the precision

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the dipole localization error. If a string the threshold is
        a percentage and it should end with the percent character.
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    stc_true, stc_est = _thresholding(stc_true, stc_est, threshold)
    func = partial(_precision_score)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _recall_score(p, q):
    from sklearn.metrics import recall_score
    p = np.sum(p, axis=1)
    q = np.sum(q, axis=1)
    return recall_score(np.abs(p) > 0, np.abs(q) > 0)


@fill_doc
def stc_recall_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the recall

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    %(metric_stc_true)s
    %(metric_stc_est)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the dipole localization error. If a string the threshold is
        a percentage and it should end with the percent character.
    %(metric_per_sample)s

    Returns
    -------
    %(stc_metric)s
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    _thresholding(stc_true, stc_est, threshold)
    func = partial(_recall_score)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric
