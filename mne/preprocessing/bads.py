# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.stats import zscore


def _find_outliers(X, threshold=3.0, max_iter=2, tail=0):
    """Find outliers based on iterated Z-scoring.

    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.
    tail : {0, 1, -1}
        Whether to search for outliers on both extremes of the z-scores (0),
        or on just the positive (1) or negative (-1) side.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    my_mask = np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        if tail == 0:
            this_z = np.abs(zscore(X))
        elif tail == 1:
            this_z = zscore(X)
        elif tail == -1:
            this_z = -zscore(X)
        else:
            raise ValueError(f"Tail parameter {tail} not recognised.")
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx
