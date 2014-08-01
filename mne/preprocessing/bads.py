# Authors: Denis Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)


import numpy as np
from scipy import stats


def find_outliers(X, threshold=3.0):
    """Find outliers based on Gaussian mixture

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    max_iter = 2
    my_mask = np.zeros(len(X), dtype=np.bool)
    X = np.abs(X)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = stats.zscore(X)
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx
