# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy.stats import pearsonr as sp_pearsonr

from .utils import check_idx


def pearsonr(data, idx=None):
    """

    """
    n_signals, n_samples = data.shape

    if idx is None:
        # only compute r for lower-triangular region
        idx_use = np.tril_indices(n_signals, -1)
    else:
        idx_use = check_idx(idx)

    corr = np.zeros((len(idx_use[0])))
    pval = np.zeros_like(corr)

    for i in xrange(len(corr)):
        corr[i], pval[i] = sp_pearsonr(data[idx_use[0][i]],
                                       data[idx_use[1][i]])

    # if idx was supplied we return 1D arrays, otherwise 2D arrays
    if idx is None:
        corr_1d, pval_1d = corr, pval
        corr = np.zeros((n_signals, n_signals))
        pval = np.zeros_like(corr)
        corr[idx_use] = corr_1d
        pval[idx_use] = pval_1d

    return corr, pval

