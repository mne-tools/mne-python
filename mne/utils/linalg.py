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
from scipy.linalg import LinAlgError
from scipy._lib._util import _asarray_validated

_d = np.empty(0, np.float64)
_z = np.empty(0, np.complex128)
dgemm = linalg.get_blas_funcs('gemm', (_d,))
zgemm = linalg.get_blas_funcs('gemm', (_z,))
dgemv = linalg.get_blas_funcs('gemv', (_d,))
ddot = linalg.get_blas_funcs('dot', (_d,))
_I = np.cast['F'](1j)


###############################################################################
# linalg.svd and linalg.pinv2
dgesdd, dgesdd_lwork = linalg.get_lapack_funcs(('gesdd', 'gesdd_lwork'), (_d,))
zgesdd, zgesdd_lwork = linalg.get_lapack_funcs(('gesdd', 'gesdd_lwork'), (_z,))
dgesvd, dgesvd_lwork = linalg.get_lapack_funcs(('gesvd', 'gesvd_lwork'), (_d,))
zgesvd, zgesvd_lwork = linalg.get_lapack_funcs(('gesvd', 'gesvd_lwork'), (_z,))


def _svd_lwork(shape, dtype=np.float64):
    """Set up SVD calculations on identical-shape float64/complex128 arrays."""
    if dtype == np.float64:
        gesdd_lwork, gesvd_lwork = dgesdd_lwork, dgesvd_lwork
    else:
        assert dtype == np.complex128
        gesdd_lwork, gesvd_lwork = zgesdd_lwork, zgesvd_lwork
    sdd_lwork = linalg.decomp_svd._compute_lwork(
        gesdd_lwork, *shape, compute_uv=True, full_matrices=False)
    svd_lwork = linalg.decomp_svd._compute_lwork(
        gesvd_lwork, *shape, compute_uv=True, full_matrices=False)
    return (sdd_lwork, svd_lwork)


def _repeated_svd(x, lwork, overwrite_a=False):
    """Mimic scipy.linalg.svd, avoid lwork and get_lapack_funcs overhead."""
    if x.dtype == np.float64:
        gesdd, gesvd = dgesdd, zgesdd
    else:
        assert x.dtype == np.complex128
        gesdd, gesvd = zgesdd, zgesvd
    # this has to use overwrite_a=False in case we need to fall back to gesvd
    u, s, v, info = gesdd(x, compute_uv=True, lwork=lwork[0],
                          full_matrices=False, overwrite_a=False)
    if info > 0:
        # Fall back to slower gesvd, sometimes gesdd fails
        u, s, v, info = gesvd(x, compute_uv=True, lwork=lwork[1],
                              full_matrices=False, overwrite_a=overwrite_a)
    if info > 0:
        raise LinAlgError("SVD did not converge")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gesdd'
                         % -info)
    return u, s, v


###############################################################################
# linalg.eigh

dsyevd, = linalg.get_lapack_funcs(('syevd',), (_d,))
zheevd, = linalg.get_lapack_funcs(('heevd',), (_z,))


def eigh(a, overwrite_a=False, check_finite=True):
    """Efficient wrapper for eigh.

    Parameters
    ----------
    a : ndarray, shape (n_components, n_components)
        The symmetric array operate on.
    overwrite_a : bool
        If True, the contents of a can be overwritten for efficiency.
    check_finite : bool
        If True, check that all elements are finite.

    Returns
    -------
    w : ndarray, shape (n_components,)
        The N eigenvalues, in ascending order, each repeated according to
        its multiplicity.
    v : ndarray, shape (n_components, n_components)
        The normalized eigenvector corresponding to the eigenvalue ``w[i]``
        is the column ``v[:, i]``.
    """
    # We use SYEVD, see https://github.com/scipy/scipy/issues/9212
    if check_finite:
        a = _asarray_validated(a, check_finite=check_finite)
    if a.dtype == np.float64:
        evr, driver = dsyevd, 'syevd'
    else:
        assert a.dtype == np.complex128
        evr, driver = zheevd, 'heevd'
    w, v, info = evr(a, lower=1, overwrite_a=overwrite_a)
    if info == 0:
        return w, v
    if info < 0:
        raise ValueError('illegal value in argument %d of internal %s'
                         % (-info, driver))
    else:
        raise LinAlgError("internal fortran routine failed to converge: "
                          "%i off-diagonal elements of an "
                          "intermediate tridiagonal form did not converge"
                          " to zero." % info)


def sqrtm_sym(A, rcond=1e-7, inv=False):
    """Compute the sqrt of a positive, semi-definite matrix (or its inverse).

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        The array to take the square root of.
    rcond : float
        The relative condition number used during reconstruction.
    inv : bool
        If True, compute the inverse of the square root rather than the
        square root itself.

    Returns
    -------
    A_sqrt : ndarray, shape (..., n, n)
        The (possibly inverted) square root of A.
    s : ndarray, shape (..., n)
        The original square root singular values (not inverted).
    """
    # Same as linalg.sqrtm(C) but faster, also yields the eigenvalues
    return _sym_mat_pow(A, -0.5 if inv else 0.5, rcond, return_s=True)


def _sym_mat_pow(A, power, rcond=1e-7, reduce_rank=False, return_s=False):
    """Exponentiate Hermitian matrices with optional rank reduction."""
    assert power in (-1, 0.5, -0.5)  # only used internally
    s, u = np.linalg.eigh(A)  # eigenvalues in ascending order
    # Is it positive semi-defidite? If so, keep real
    limit = s[..., -1:] * rcond
    if not (s >= -limit).all():  # allow some tiny small negative ones
        raise ValueError('Matrix is not positive semi-definite')
    s[s <= limit] = np.inf if power < 0 else 0
    if reduce_rank:
        # These are ordered smallest to largest, so we set the first one
        # to inf -- then the 1. / s below will turn this to zero, as needed.
        s[..., 0] = np.inf
    if power in (-0.5, 0.5):
        np.sqrt(s, out=s)
    use_s = 1. / s if power < 0 else s
    out = np.matmul(u * use_s[..., np.newaxis, :], u.swapaxes(-2, -1).conj())
    if return_s:
        out = (out, s)
    return out
