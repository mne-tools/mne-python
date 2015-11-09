# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_almost_equal

from mne.inverse_sparse.mxne_debiasing import compute_bias


def test_compute_debiasing():
    """Test source amplitude debiasing"""
    rng = np.random.RandomState(42)
    G = rng.randn(10, 4)
    X = rng.randn(4, 20)
    debias_true = np.arange(1, 5, dtype=np.float)
    M = np.dot(G, X * debias_true[:, np.newaxis])
    debias = compute_bias(M, G, X, max_iter=10000, n_orient=1, tol=1e-7)
    assert_almost_equal(debias, debias_true, decimal=5)
    debias = compute_bias(M, G, X, max_iter=10000, n_orient=2, tol=1e-5)
    assert_almost_equal(debias, [1.8, 1.8, 3.72, 3.72], decimal=2)
