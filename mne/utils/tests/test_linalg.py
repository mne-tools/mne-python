"""Test linalg utilities."""
# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
from mne.utils import _sym_inv


def test_sym_inv():
    mat_real = np.array([[3., 5., 8.], [5., 1., 9.], [1., 8., 3.]])

    # _sym_inv with full rank should behave like pinv
    mat_pinv = np.linalg.pinv(mat_real)
    mat_symv = _sym_inv(mat_real, reduce_rank=False)

    assert_allclose(mat_pinv, mat_symv)
