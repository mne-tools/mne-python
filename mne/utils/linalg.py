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


_x = np.empty(0)
dgemm = linalg.get_blas_funcs('gemm', (_x,))
dgemv = linalg.get_blas_funcs('gemv', (_x,))
ddot = linalg.get_blas_funcs('dot', (_x,))
dgesdd = linalg.get_lapack_funcs('gesdd', (_x,))
dgesdd, dgesdd_lwork = linalg.get_lapack_funcs(('gesdd', 'gesdd_lwork'), (_x,))


def _dgesdd_lwork(shape):
    """Set up repeated SVD calculations on identical shape float64 arrays."""
    lwork = linalg.decomp_svd._compute_lwork(
        dgesdd_lwork, *shape, compute_uv=True, full_matrices=False)
    return lwork


def _repeated_svd(x, lwork, overwrite_a=False):
    """Mimic scipy.linalg.svd, avoid lwork and get_lapack_funcs overhead."""
    assert lwork is not None
    u, s, v, info = dgesdd(x, compute_uv=True, lwork=lwork,
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
    assert x.dtype == np.float64
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
