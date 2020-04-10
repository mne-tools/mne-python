# Authors: Denis Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)


import numpy as np


def _find_outliers(X, threshold=3.0, max_iter=2):
    from scipy.stats import zscore
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx
