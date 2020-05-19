"""Test linalg utilities."""
# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from mne.utils import run_tests_if_main, _pos_semidef_inv, requires_version


@requires_version('numpy', '1.17')  # hermitian kwarg
@pytest.mark.parametrize('dtype', (np.float64, np.complex128))  # real, complex
@pytest.mark.parametrize('ndim', (2, 3, 4))
@pytest.mark.parametrize('n', (3, 4))
@pytest.mark.parametrize('deficient, reduce_rank', [
    (False, False),
    (True, False),  # should auto-remove the reduced component
    (True, True),  # force removal of one component (though redundant here)
])
def test_pos_semidef_inv(ndim, dtype, n, deficient, reduce_rank):
    """Test positive semidefinite inverse."""
    # make n-dimensional matrix
    n_extra = 2  # how many we add along the other dims
    rng = np.random.RandomState(73)
    shape = (n_extra,) * (ndim - 2) + (n, n)
    mat = rng.randn(*shape) + 1j * rng.randn(*shape)
    proj = np.eye(n)
    if deficient:
        vec = np.ones(n) / np.sqrt(n)
        proj -= np.outer(vec, vec)
    with pytest.warns(None):  # intentionally discard imag
        mat = mat.astype(dtype)
    # make it rank deficient (maybe)
    if deficient:
        mat = np.matmul(proj, mat)
    # now make it positive definite
    mat = np.matmul(mat, mat.conj().swapaxes(-2, -1))
    # if the dtype is complex, the conjugate transpose != transpose
    kwargs = dict(atol=1e-10, rtol=1e-10)
    orig_eq_t = np.allclose(
        mat, mat.swapaxes(-2, -1), **kwargs)
    t_eq_ct = np.allclose(
        mat.swapaxes(-2, -1), mat.conj().swapaxes(-2, -1), **kwargs)
    if np.iscomplexobj(mat):
        assert not orig_eq_t
        assert not t_eq_ct
    else:
        assert t_eq_ct
        assert orig_eq_t
    assert mat.shape == shape
    # ensure pos-semidef
    s = np.linalg.svd(mat, compute_uv=False)
    assert s.shape == shape[:-1]
    rank = (s > s[..., :1] * 1e-12).sum(-1)
    want_rank = n - deficient
    assert_array_equal(rank, want_rank)
    # assert equiv with NumPy
    mat_pinv = np.linalg.pinv(mat, hermitian=True)
    mat_symv = _pos_semidef_inv(mat, reduce_rank=reduce_rank)
    assert_allclose(mat_pinv, mat_symv, **kwargs)
    want = np.dot(proj, np.eye(n))
    if deficient:
        want -= want.mean(axis=0)
    for _ in range(ndim - 2):
        want = np.repeat(want[np.newaxis], n_extra, axis=0)
    assert_allclose(np.matmul(mat_symv, mat), want, **kwargs)
    assert_allclose(np.matmul(mat, mat_symv), want, **kwargs)


run_tests_if_main()
