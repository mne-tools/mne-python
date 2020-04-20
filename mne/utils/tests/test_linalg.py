"""Test linalg utilities."""
# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
from mne.utils import run_tests_if_main, _pos_semidef_inv


def test_pos_semidef_inv():
    # make 3 dimensional  matrices that are positive semidefinite
    random = np.random.RandomState(73)

    # matrix with real values
    real_part = random.rand(3, 3)
    real_part = real_part.dot(real_part.T)
    mat_real = np.array([real_part, real_part])  # _pos_def_inv expects >2 dim.

    # matrix with complex values
    random.seed(21)
    imag_part = random.rand(3, 3)
    mat_complex = real_part.copy()
    mat_complex = np.array(imag_part, dtype=complex)
    mat_complex.imag = imag_part
    mat_complex = np.dot(mat_complex, mat_complex.conj().T)
    mat_complex = np.array([mat_complex, mat_complex])

    # _pos_semidef_inv should behave like pinv
    # Test for real-valued matrices:
    mat_pinv = np.linalg.pinv(mat_real)
    mat_symv = _pos_semidef_inv(mat_real, reduce_rank=False)
    assert_allclose(mat_pinv, mat_symv, rtol=1e-7,
                    atol=1e-7 * np.median(mat_pinv[0].ravel()))

    # Test for complex values:
    mat_pinv = np.linalg.pinv(mat_complex)
    mat_symv = _pos_semidef_inv(mat_complex, reduce_rank=False)
    assert_allclose(np.real(mat_pinv), np.real(mat_symv), rtol=1e-7,
                    atol=1e-7 * np.median(np.real(mat_pinv).ravel()))
    assert_allclose(np.imag(mat_pinv), np.imag(mat_symv), rtol=1e-7,
                    atol=1e-7 * np.median(np.imag(mat_pinv).ravel()))


run_tests_if_main()
