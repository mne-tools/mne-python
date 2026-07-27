"""Test linalg utilities."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from mne.utils import _record_warnings, _reg_pinv, _sym_mat_pow


@pytest.mark.parametrize("dtype", (np.float64, np.complex128))  # real, complex
@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("n", (3, 4))
@pytest.mark.parametrize("psdef", (True, False))
@pytest.mark.parametrize(
    "deficient, reduce_rank",
    [
        (False, False),
        (True, False),  # should auto-remove the reduced component
        (True, True),  # force removal of one component (though redundant here)
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        _sym_mat_pow,
        _reg_pinv,
    ],
)
def test_pos_semidef_inv(ndim, dtype, n, deficient, reduce_rank, psdef, func):
    """Test positive semidefinite matrix inverses."""
    svd = np.linalg.svd
    # make n-dimensional matrix
    n_extra = 2  # how many we add along the other dims
    rng = np.random.default_rng(73)
    shape = (n_extra,) * (ndim - 2) + (n, n)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    proj = np.eye(n)
    if deficient:
        vec = np.ones(n) / np.sqrt(n)
        proj -= np.outer(vec, vec)
    with _record_warnings():  # intentionally discard imag
        mat = mat.astype(dtype)
    # now make it conjugate symmetric or positive semi-definite
    if psdef:
        mat = np.matmul(mat, mat.swapaxes(-2, -1).conj())
    else:
        mat += mat.swapaxes(-2, -1).conj()
    assert_allclose(mat, mat.swapaxes(-2, -1).conj(), atol=1e-6)
    s = svd(mat, hermitian=True)[1]
    assert (s >= 0).all()
    # make it rank deficient (maybe)
    if deficient:
        mat = np.matmul(np.matmul(proj, mat), proj)
    # if the dtype is complex, the conjugate transpose != transpose
    kwargs = dict(atol=1e-10, rtol=1e-10)
    orig_eq_t = np.allclose(mat, mat.swapaxes(-2, -1), **kwargs)
    t_eq_ct = np.allclose(mat.swapaxes(-2, -1), mat.conj().swapaxes(-2, -1), **kwargs)
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
    mat_pinv = np.linalg.pinv(mat)
    if func is _sym_mat_pow:
        if not psdef:
            with pytest.raises(ValueError, match="not positive semi-"):
                func(mat, -1)
            return
        mat_symv = func(mat, -1, reduce_rank=reduce_rank)
        mat_sqrt = func(mat, 0.5)
        if ndim == 2:
            mat_sqrt_scipy = linalg.sqrtm(mat)
            assert_allclose(mat_sqrt, mat_sqrt_scipy, atol=1e-6)
        mat_2 = np.matmul(mat_sqrt, mat_sqrt)
        assert_allclose(mat, mat_2, atol=1e-6)
        mat_symv_2 = func(mat, -0.5, reduce_rank=reduce_rank)
        mat_symv_2 = np.matmul(mat_symv_2, mat_symv_2)
        assert_allclose(mat_symv_2, mat_symv, atol=1e-6)
    else:
        assert func is _reg_pinv
        mat_symv, _, _ = func(mat, rank=None)
    assert_allclose(mat_pinv, mat_symv, **kwargs)
    want = np.dot(proj, np.eye(n))
    if deficient:
        want -= want.mean(axis=0)
    for _ in range(ndim - 2):
        want = np.repeat(want[np.newaxis], n_extra, axis=0)
    assert_allclose(np.matmul(mat_symv, mat), want, **kwargs)
    assert_allclose(np.matmul(mat, mat_symv), want, **kwargs)


def test_limit_blas_threads_logic(monkeypatch):
    """Test when BLAS thread limiting engages and when it defers."""
    pytest.importorskip("threadpoolctl")
    from mne.utils import linalg

    controller = MagicMock()
    controller.info.return_value = [dict(internal_api="openblas", num_threads=8)]
    monkeypatch.setattr(linalg, "_blas_thread_controller", lambda: controller)
    monkeypatch.setattr(linalg, "_n_available_cpus", lambda: 8)
    for var in linalg._BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with linalg._limit_blas_threads():
        pass
    controller.limit.assert_called_once_with(limits=3)

    # defer to an explicit user preference
    controller.reset_mock()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with linalg._limit_blas_threads():
        pass
    controller.limit.assert_not_called()

    # defer when something already limited us (outer limits, joblib worker, ...)
    monkeypatch.delenv("OMP_NUM_THREADS")
    controller.info.return_value = [dict(internal_api="openblas", num_threads=2)]
    with linalg._limit_blas_threads():
        pass
    controller.limit.assert_not_called()

    # but do not oversubscribe a machine smaller than the cap
    monkeypatch.setattr(linalg, "_n_available_cpus", lambda: 2)
    with linalg._limit_blas_threads():
        pass
    controller.limit.assert_called_once_with(limits=2)

    # no-op when threads cannot be controlled at all (no threadpoolctl, Accelerate)
    monkeypatch.setattr(linalg, "_blas_thread_controller", lambda: None)
    with linalg._limit_blas_threads():
        pass


def test_limit_blas_threads(monkeypatch):
    """Test that BLAS threads are actually limited and restored."""
    threadpool_info = pytest.importorskip("threadpoolctl").threadpool_info
    from mne.utils import linalg

    if linalg._blas_thread_controller() is None:
        pytest.skip("BLAS threads are not controllable (e.g. Accelerate)")
    for var in linalg._BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # 1 so that no ambient limit (e.g. from conftest) counts as already-limited
    monkeypatch.setattr(linalg, "_n_available_cpus", lambda: 1)
    before = [pool["num_threads"] for pool in threadpool_info()]
    with linalg._limit_blas_threads():
        assert [pool["num_threads"] for pool in threadpool_info()] == [1] * len(before)
    assert [pool["num_threads"] for pool in threadpool_info()] == before
