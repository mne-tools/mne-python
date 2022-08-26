# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#          Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from functools import partial

import numpy as np
from mne.utils import _check_option, fill_doc, _validate_type


def _check_stc(stc1, stc2):
    """Check that stcs are compatible."""
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')


def source_estimate_quantification(stc1, stc2, metric='rms'):
    """Calculate STC similarities across all sources and times.

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
        score = 1. - _cosine(data1, data2)
    return score


def _uniform_stc(stc1, stc2):
    """Uniform vertices of two stcs.

    This function returns the stcs with the same vertices by
    inserting zeros in data for missing vertices.
    """
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


def _apply(func, stc_true, stc_est, per_sample):
    """Apply metric to stcs.

    Applies a metric to each pair of columns of stc_true and stc_est
    if per_sample is True. Otherwise it applies it to stc_true and stc_est
    directly.
    """
    if per_sample:
        metric = np.empty(stc_true.data.shape[1])  # one value per time point
        for i in range(stc_true.data.shape[1]):
            metric[i] = func(stc_true.data[:, i:i + 1],
                             stc_est.data[:, i:i + 1])
    else:
        metric = func(stc_true.data, stc_est.data)
    return metric


def _thresholding(stc_true, stc_est, threshold):
    relative = isinstance(threshold, str)
    threshold = _check_threshold(threshold)
    if relative:
        if stc_true is not None:
            stc_true._data[np.abs(stc_true._data) <=
                           threshold * np.max(np.abs(stc_true._data))] = 0.
        stc_est._data[np.abs(stc_est._data) <=
                      threshold * np.max(np.abs(stc_est._data))] = 0.
    else:
        if stc_true is not None:
            stc_true._data[np.abs(stc_true._data) <= threshold] = 0.
        stc_est._data[np.abs(stc_est._data) <= threshold] = 0.
    return stc_true, stc_est


def _cosine(x, y):
    p = x.ravel()
    q = y.ravel()
    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    if p_norm * q_norm:
        return (p.T @ q) / (p_norm * q_norm)
    elif p_norm == q_norm:
        return 1
    else:
        return 0


@fill_doc
def cosine_score(stc_true, stc_est, per_sample=True):
    """Compute cosine similarity between 2 source estimates.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    .. versionadded:: 1.2
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    metric = _apply(_cosine, stc_true, stc_est, per_sample=per_sample)
    return metric


def _check_threshold(threshold):
    """Accept a float or a string that ends with %."""
    _validate_type(threshold, ('numeric', str), 'threshold')
    if isinstance(threshold, str):
        if not threshold.endswith("%"):
            raise ValueError('Threshold if a string must end with '
                             '"%%". Got %s.' % threshold)
        threshold = float(threshold[:-1]) / 100.0
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError(
            'Threshold proportion must be between 0 and 1 (inclusive), but '
            f'got {threshold}')
    return threshold


def _abs_col_sum(x):
    return np.abs(x).sum(axis=1)


def _dle(p, q, src, stc):
    """Aux function to compute dipole localization error."""
    from scipy.spatial.distance import cdist
    p = _abs_col_sum(p)
    q = _abs_col_sum(q)
    idx1 = np.nonzero(p)[0]
    idx2 = np.nonzero(q)[0]
    points = []
    for i in range(len(src)):
        points.append(src[i]['rr'][stc.vertices[i]])
    points = np.concatenate(points, axis=0)
    if len(idx1) and len(idx2):
        D = cdist(points[idx1], points[idx2])
        D_min_1 = np.min(D, axis=0)
        D_min_2 = np.min(D, axis=1)
        return (np.mean(D_min_1) + np.mean(D_min_2)) / 2.
    else:
        return np.inf


@fill_doc
def region_localization_error(stc_true, stc_est, src, threshold='90%',
                              per_sample=True):
    r"""Compute region localization error (RLE) between 2 source estimates.

    .. math::

        RLE = \frac{1}{2Q}\sum_{k \in I} \min_{l \in \hat{I}}{||r_k - r_l||} + \frac{1}{2\hat{Q}}\sum_{l \in \hat{I}} \min_{k \in I}{||r_k - r_l||}

    where :math:`I` and :math:`\hat{I}` denote respectively the original and
    estimated indexes of active sources, :math:`Q` and :math:`\hat{Q}` are
    the numbers of original and estimated active sources.
    :math:`r_k` denotes the position of the k-th source dipole in space
    and :math:`||\cdot||` is an Euclidean norm in :math:`\mathbb{R}^3`.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    src : instance of SourceSpaces
        The source space on which the source estimates are defined.
    threshold : float | str
        The threshold to apply to source estimates before computing
        the dipole localization error. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    Papers :footcite:`MaksymenkoEtAl2017` and :footcite:`BeckerEtAl2017`
    use term Dipole Localization Error (DLE) for the same formula. Paper
    :footcite:`YaoEtAl2005` uses term Error Distance (ED) for the same formula.
    To unify the terminology and to avoid confusion with other cases
    of using term DLE but for different metric :footcite:`MolinsEtAl2008`, we
    use term Region Localization Error (RLE).

    .. versionadded:: 1.2

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    stc_true, stc_est = _thresholding(stc_true, stc_est, threshold)
    func = partial(_dle, src=src, stc=stc_true)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _roc_auc_score(p, q):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.abs(p) > 0, np.abs(q))


@fill_doc
def roc_auc_score(stc_true, stc_est, per_sample=True):
    """Compute ROC AUC between 2 source estimates.

    ROC stands for receiver operating curve and AUC is Area under the curve.
    When computing this metric the stc_true must be thresholded
    as any non-zero value will be considered as a positive.

    The ROC-AUC metric is computed between amplitudes of the source
    estimates, i.e. after taking the absolute values.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    .. versionadded:: 1.2
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    metric = _apply(_roc_auc_score, stc_true, stc_est, per_sample=per_sample)
    return metric


def _f1_score(p, q):
    from sklearn.metrics import f1_score
    return f1_score(_abs_col_sum(p) > 0, _abs_col_sum(q) > 0)


@fill_doc
def f1_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a weighted average of the precision
    and recall, where an F1 score reaches its best value at 1 and worst score
    at 0. The relative contribution of precision and recall to the F1
    score are equal.
    The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Threshold is used first for data binarization.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the f1 score. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    .. versionadded:: 1.2
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    stc_true, stc_est = _thresholding(stc_true, stc_est, threshold)
    metric = _apply(_f1_score, stc_true, stc_est, per_sample=per_sample)
    return metric


def _precision_score(p, q):
    from sklearn.metrics import precision_score
    return precision_score(_abs_col_sum(p) > 0, _abs_col_sum(q) > 0)


@fill_doc
def precision_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Threshold is used first for data binarization.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the precision. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    .. versionadded:: 1.2
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    stc_true, stc_est = _thresholding(stc_true, stc_est, threshold)
    metric = _apply(_precision_score, stc_true, stc_est, per_sample=per_sample)
    return metric


def _recall_score(p, q):
    from sklearn.metrics import recall_score
    return recall_score(_abs_col_sum(p) > 0, _abs_col_sum(q) > 0)


@fill_doc
def recall_score(stc_true, stc_est, threshold='90%', per_sample=True):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Threshold is used first for data binarization.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    threshold : float | str
        The threshold to apply to source estimates before computing
        the recall. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    .. versionadded:: 1.2
    """
    stc_true, stc_est = _uniform_stc(stc_true, stc_est)
    stc_true, stc_est = _thresholding(stc_true, stc_est, threshold)
    metric = _apply(_recall_score, stc_true, stc_est, per_sample=per_sample)
    return metric


def _prepare_ppe_sd(stc_true, stc_est, src, threshold='50%'):
    stc_true = stc_true.copy()
    stc_est = stc_est.copy()
    n_dipoles = 0
    for i, v in enumerate(stc_true.vertices):
        if len(v):
            n_dipoles += len(v)
            r_true = src[i]['rr'][v]
    if n_dipoles != 1:
        raise ValueError('True source must contain only one dipole, got %d.'
                         % n_dipoles)

    _, stc_est = _thresholding(None, stc_est, threshold)

    r_est = np.empty([0, 3])
    for i, v in enumerate(stc_est.vertices):
        if len(v):
            r_est = np.vstack([r_est, src[i]['rr'][v]])
    return stc_est, r_true, r_est


def _peak_position_error(p, q, r_est, r_true):
    q = _abs_col_sum(q)
    if np.sum(q):
        q /= np.sum(q)
        r_est_mean = np.dot(q, r_est)
        return np.linalg.norm(r_est_mean - r_true)
    else:
        return np.inf


@fill_doc
def peak_position_error(stc_true, stc_est, src, threshold='50%',
                        per_sample=True):
    r"""Compute the peak position error.

    The peak position error measures the distance between the center-of-mass
    of the estimated and the true source.

    .. math::

        PPE = \| \dfrac{\sum_i|s_i|r_{i}}{\sum_i|s_i|}
        - r_{true}\|,

    where :math:`r_{true}` is a true dipole position,
    :math:`r_i` and :math:`|s_i|` denote respectively the position
    and amplitude of i-th dipole in source estimate.

    Threshold is used on estimated source for focusing the metric to strong
    amplitudes and omitting the low-amplitude values.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    src : instance of SourceSpaces
        The source space on which the source estimates are defined.
    threshold : float | str
        The threshold to apply to source estimates before computing
        the recall. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    These metrics are documented in :footcite:`StenroosHauk2013` and
    :footcite:`LinEtAl2006a`.

    .. versionadded:: 1.2

    References
    ----------
    .. footbibliography::
    """
    stc_est, r_true, r_est = _prepare_ppe_sd(stc_true, stc_est, src, threshold)
    func = partial(_peak_position_error, r_est=r_est, r_true=r_true)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric


def _spatial_deviation(p, q, r_est, r_true):
    q = _abs_col_sum(q)
    if np.sum(q):
        q /= np.sum(q)
        r_true_tile = np.tile(r_true, (r_est.shape[0], 1))
        r_diff = r_est - r_true_tile
        r_diff_norm = np.sum(r_diff ** 2, axis=1)
        return np.sqrt(np.dot(q, r_diff_norm))
    else:
        return np.inf


@fill_doc
def spatial_deviation_error(stc_true, stc_est, src, threshold='50%',
                            per_sample=True):
    r"""Compute the spatial deviation.

    The spatial deviation characterizes the spread of the estimate source
    around the true source.

    .. math::

        SD = \dfrac{\sum_i|s_i|\|r_{i} - r_{true}\|^2}{\sum_i|s_i|}.

    where :math:`r_{true}` is a true dipole position,
    :math:`r_i` and :math:`|s_i|` denote respectively the position
    and amplitude of i-th dipole in source estimate.

    Threshold is used on estimated source for focusing the metric to strong
    amplitudes and omitting the low-amplitude values.

    Parameters
    ----------
    %(stc_true_metric)s
    %(stc_est_metric)s
    src : instance of SourceSpaces
        The source space on which the source estimates are defined.
    threshold : float | str
        The threshold to apply to source estimates before computing
        the recall. If a string the threshold is
        a percentage and it should end with the percent character.
    %(per_sample_metric)s

    Returns
    -------
    %(stc_metric)s

    Notes
    -----
    These metrics are documented in :footcite:`StenroosHauk2013` and
    :footcite:`LinEtAl2006a`.

    .. versionadded:: 1.2

    References
    ----------
    .. footbibliography::
    """
    stc_est, r_true, r_est = _prepare_ppe_sd(stc_true, stc_est, src, threshold)
    func = partial(_spatial_deviation, r_est=r_est, r_true=r_true)
    metric = _apply(func, stc_true, stc_est, per_sample=per_sample)
    return metric
