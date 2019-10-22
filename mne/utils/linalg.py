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
from ..fixes import _check_info

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


def _repeated_pinv2(x, lwork, rcond=None):
    """Mimic scipy.linalg.pinv2, avoid lwork and get_lapack_funcs overhead."""
    # Adapted from SciPy
    u, s, vh = _repeated_svd(x, lwork)
    if rcond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        rcond = factor[t] * np.finfo(t).eps
    rank = np.sum(s > rcond * s[0])
    psigma_diag = 1.0 / s[:rank]
    u[:, :rank] *= psigma_diag
    B = np.transpose(np.conjugate(np.dot(u[:, :rank], vh[:rank])))
    return B


###############################################################################
# linalg.eig

dgeev, dgeev_lwork = linalg.get_lapack_funcs(('geev', 'geev_lwork'), (_d,))
zgeev, zgeev_lwork = linalg.get_lapack_funcs(('geev', 'geev_lwork'), (_z,))


def _eig_lwork(shape, dtype=np.float64):
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
        wr, wi, vl, vr, info = dgeev(
            a, lwork=lwork, compute_vl=False, compute_vr=True,
            overwrite_a=overwrite_a)
        w = wr + _I * wi
        need_complex = np.any(wi)
    else:
        assert a.dtype == np.complex128
        w, vl, vr, info = zgeev(
            a, lwork=lwork, compute_vl=False, compute_vr=True,
            overwrite_a=overwrite_a)
        need_complex = False
    _check_info(
        info, 'eig algorithm (geev)',
        positive='did not converge (only eigenvalues '
                 'with order >= %d have converged)')
    if need_complex:
        t = w.dtype.char
        vr = linalg.decomp._make_complex_eigvecs(w, vr, t)
    return w, vr


###############################################################################
# linalg.inv

dgetrf, dgetri, dgetri_lwork = linalg.get_lapack_funcs(
    ('getrf', 'getri', 'getri_lwork'), (_d,))
zgetrf, zgetri, zgetri_lwork = linalg.get_lapack_funcs(
    ('getrf', 'getri', 'getri_lwork'), (_z,))


def _inv_lwork(shape, dtype=np.float64):
    if dtype == np.float64:
        getri_lwork = dgetri_lwork
    else:
        assert dtype == np.complex128
        getri_lwork = zgetri_lwork
    return linalg.lapack._compute_lwork(getri_lwork, shape[0])


def _repeated_inv(a, lwork, overwrite_a=False):
    if a.dtype == np.float64:
        getrf, getri = dgetrf, dgetri
    else:
        assert a.dtype == np.complex128
        getrf, getri = zgetrf, zgetri
    lu, piv, info = getrf(a, overwrite_a=overwrite_a)
    if info == 0:
        # XXX: the following line fixes curious SEGFAULT when
        # benchmarking 500x500 matrix inverse. This seems to
        # be a bug in LAPACK ?getri routine because if lwork is
        # minimal (when using lwork[0] instead of lwork[1]) then
        # all tests pass. Further investigation is required if
        # more such SEGFAULTs occur.
        lwork = int(1.01 * lwork)
        inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
    if info > 0:
        raise LinAlgError("singular matrix")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'getrf|getri' % -info)
    return inv_a


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
