# -*- coding: utf-8 -*-
"""Utility functions to speed up linear algebraic operations.

In general, things like np.dot and linalg.svd should be used directly
because they are smart about checking for bad values. However, in cases where
things are done repeatedly (e.g., thousands of times on tiny matrices), the
overhead can become problematic from a performance standpoint. Examples:

- Optimization routines:
  - Dipole fitting
  - Sparse solving
  - cHPI fitting
- Inverse computation
  - Beamformers (LCMV/DICS)
  - eLORETA minimum norm

Significant performance gains can be achieved by ensuring that inputs
are Fortran contiguous because that's what LAPACK requires. Without this,
inputs will be memcopied.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg


_d = np.empty(0, np.float64)
_z = np.empty(0, np.complex128)
dgemm = linalg.get_blas_funcs('gemm', (_d,))
zgemm = linalg.get_blas_funcs('gemm', (_z,))
dgemv = linalg.get_blas_funcs('gemv', (_d,))
ddot = linalg.get_blas_funcs('dot', (_d,))
dgesdd, dgesdd_lwork = linalg.get_lapack_funcs(('gesdd', 'gesdd_lwork'), (_d,))
zgesdd, zgesdd_lwork = linalg.get_lapack_funcs(('gesdd', 'gesdd_lwork'), (_z,))
dgeev, dgeev_lwork = linalg.get_lapack_funcs(('geev', 'geev_lwork'), (_d,))
zgeev, zgeev_lwork = linalg.get_lapack_funcs(('geev', 'geev_lwork'), (_z,))
_I = np.cast['F'](1j)


def _gesdd_lwork(shape, dtype=np.float64):
    """Set up SVD calculations on identical-shape float64/complex128 arrays."""
    if dtype == np.float64:
        gesdd_lwork = dgesdd_lwork
    else:
        assert dtype == np.complex128
        gesdd_lwork = zgesdd_lwork
    lwork = linalg.decomp_svd._compute_lwork(
        gesdd_lwork, *shape, compute_uv=True, full_matrices=False)
    return lwork


def _repeated_svd(x, lwork, overwrite_a=False):
    """Mimic scipy.linalg.svd, avoid lwork and get_lapack_funcs overhead."""
    if x.dtype == np.float64:
        gesdd = dgesdd
    else:
        assert x.dtype == np.complex128
        gesdd = zgesdd
    u, s, v, info = gesdd(x, compute_uv=True, lwork=lwork,
                          full_matrices=False, overwrite_a=True)
    if info > 0:
        raise linalg.LinAlgError("SVD did not converge")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gesdd'
                         % -info)
    return u, s, v


def _repeated_pinv2(x, lwork, rcond=None):
    """Mimic scipy.linalg.pinv2, avoid lwork and get_lapack_funcs overhead."""
    # Adapted from SciPy
    u, s, vh = _repeated_svd(x, lwork)
    if rcond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        rcond = factor[t] * np.finfo(t).eps
    rank = np.sum(s > rcond * np.max(s))
    psigma_diag = 1.0 / s[: rank]
    u[:, :rank] *= psigma_diag
    B = np.transpose(np.conjugate(np.dot(u[:, :rank], vh[:rank])))
    return B


def _geev_lwork(shape, dtype=np.float64):
    """Set up SVD calculations on identical-shape float64/complex128 arrays."""
    if dtype == np.float64:
        geev_lwork = dgeev_lwork
    else:
        assert dtype == np.complex128
        geev_lwork = zgeev_lwork
    lwork = linalg.decomp._compute_lwork(
        geev_lwork, shape[0], compute_vl=False, compute_vr=True)
    return lwork


def _repeated_eig(a, lwork, overwrite_a=False):
    """Mimic scipy.linalg.eig, avoid lwork and get_lapack_funcs overhead."""
    if a.dtype == np.float64:
        geev = dgeev
    else:
        assert a.dtype == np.complex128
        geev = zgeev
    a1 = a
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    if geev.typecode in 'cz':
        w, vl, vr, info = geev(
            a1, lwork=lwork, compute_vl=False, compute_vr=True,
            overwrite_a=overwrite_a)
    else:
        wr, wi, vl, vr, info = geev(
            a1, lwork=lwork, compute_vl=False, compute_vr=True,
            overwrite_a=overwrite_a)
        t = {'f': 'F', 'd': 'D'}[wr.dtype.char]
        w = wr + _I * wi
    linalg.decomp._check_info(
        info, 'eig algorithm (geev)',
        positive='did not converge (only eigenvalues '
                 'with order >= %d have converged)')

    only_real = np.all(w.imag == 0.0)
    if not (geev.typecode in 'cz' or only_real):
        t = w.dtype.char
        vr = linalg.decomp._make_complex_eigvecs(w, vr, t)
    return w, vr
