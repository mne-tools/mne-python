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

import functools

import numpy as np


# For efficiency, names should be str or tuple of str, dtype a builtin
# NumPy dtype

@functools.lru_cache(None)
def _get_blas_funcs(dtype, names):
    from scipy import linalg
    return linalg.get_blas_funcs(names, (np.empty(0, dtype),))


@functools.lru_cache(None)
def _get_lapack_funcs(dtype, names):
    from scipy import linalg
    assert dtype in (np.float64, np.complex128)
    x = np.empty(0, dtype)
    return linalg.get_lapack_funcs(names, (x,))


###############################################################################
# linalg.svd and linalg.pinv2

def _svd_lwork(shape, dtype=np.float64):
    """Set up SVD calculations on identical-shape float64/complex128 arrays."""
    from scipy import linalg
    gesdd_lwork, gesvd_lwork = _get_lapack_funcs(
        dtype, ('gesdd_lwork', 'gesvd_lwork'))
    sdd_lwork = linalg.decomp_svd._compute_lwork(
        gesdd_lwork, *shape, compute_uv=True, full_matrices=False)
    svd_lwork = linalg.decomp_svd._compute_lwork(
        gesvd_lwork, *shape, compute_uv=True, full_matrices=False)
    return sdd_lwork, svd_lwork


def _repeated_svd(x, lwork, overwrite_a=False):
    """Mimic scipy.linalg.svd, avoid lwork and get_lapack_funcs overhead."""
    gesdd, gesvd = _get_lapack_funcs(
        x.dtype, ('gesdd', 'gesvd'))
    # this has to use overwrite_a=False in case we need to fall back to gesvd
    u, s, v, info = gesdd(x, compute_uv=True, lwork=lwork[0],
                          full_matrices=False, overwrite_a=False)
    if info > 0:
        # Fall back to slower gesvd, sometimes gesdd fails
        u, s, v, info = gesvd(x, compute_uv=True, lwork=lwork[1],
                              full_matrices=False, overwrite_a=overwrite_a)
    if info > 0:
        raise np.linalg.LinAlgError("SVD did not converge")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gesdd'
                         % -info)
    return u, s, v


###############################################################################
# linalg.eigh

@functools.lru_cache(None)
def _get_evd(dtype):
    from scipy import linalg
    x = np.empty(0, dtype)
    if dtype == np.float64:
        driver = 'syevd'
    else:
        assert dtype == np.complex128
        driver = 'heevd'
    evr, = linalg.get_lapack_funcs((driver,), (x,))
    return evr, driver


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
    from scipy.linalg import LinAlgError
    from scipy._lib._util import _asarray_validated
    # We use SYEVD, see https://github.com/scipy/scipy/issues/9212
    if check_finite:
        a = _asarray_validated(a, check_finite=check_finite)
    evd, driver = _get_evd(a.dtype)
    w, v, info = evd(a, lower=1, overwrite_a=overwrite_a)
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
