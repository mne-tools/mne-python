# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from numpy.testing import assert_almost_equal

from mne.inverse_sparse.mxne_debiasing import compute_bias


def test_compute_debiasing():
    """Test source amplitude debiasing."""
    rng = np.random.default_rng(42)
    G = rng.standard_normal((10, 4))
    X = rng.standard_normal((4, 20))
    debias_true = np.arange(1, 5, dtype=np.float64)
    M = np.dot(G, X * debias_true[:, np.newaxis])
    debias = compute_bias(M, G, X, max_iter=10000, n_orient=1, tol=1e-7)
    assert_almost_equal(debias, debias_true, decimal=5)
    debias = compute_bias(M, G, X, max_iter=10000, n_orient=2, tol=1e-5)
    assert_almost_equal(debias, [1.63, 1.63, 3.33, 3.33], decimal=2)
