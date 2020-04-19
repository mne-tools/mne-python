"""Test linalg utilities."""
# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
from mne.utils import _sym_inv


def test_sym_inv():
    # make 3 dimensional  matrices that are positive semidefinite
    random = np.random.RandomState(73)

    # matrix with real values
    real_part = random.rand(3, 3) * 10
    real_part = real_part.dot(real_part.T)
    mat_real = np.array([real_part, real_part])  # _pos_def_inv expects >2 dim.

    # matrix with complex values
    random.seed(21)
    imag_part = random.rand(3, 3) * 10
    mat_complex = real_part.copy()
    mat_complex = np.array(imag_part, dtype=complex)
    mat_complex.imag = imag_part
    mat_complex = np.dot(mat_complex, mat_complex.conj().T)
    mat_complex = np.array([mat_complex, mat_complex])

    # _sym_inv should behave like pinv
    # Test for real-valued matrices:
    mat_pinv = np.linalg.pinv(mat_real)
    mat_symv = _sym_inv(mat_real, reduce_rank=False)
    assert_allclose(mat_pinv, mat_symv)

    # Test for complex values:
    mat_pinv = np.linalg.pinv(mat_complex)
    mat_symv = _sym_inv(mat_complex, reduce_rank=False)
    assert_allclose(mat_pinv, mat_symv)
