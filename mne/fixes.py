"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.

# XXX : copied from scikit-learn

"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD

from __future__ import division
import collections
from operator import itemgetter
import inspect

import warnings
import numpy as np
import scipy
from scipy import linalg, sparse
from math import ceil, log
from numpy.fft import irfft
from distutils.version import LooseVersion
from functools import partial
from .externals import six
from .externals.six.moves import copyreg
from gzip import GzipFile


###############################################################################
# Misc

class gzip_open(GzipFile):  # python2.6 doesn't have context managing

    def __enter__(self):
        if hasattr(GzipFile, '__enter__'):
            return GzipFile.__enter__(self)
        else:
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(GzipFile, '__exit__'):
            return GzipFile.__exit__(self, exc_type, exc_value, traceback)
        else:
            return self.close()


class _Counter(collections.defaultdict):
    """Partial replacement for Python 2.7 collections.Counter."""
    def __init__(self, iterable=(), **kwargs):
        super(_Counter, self).__init__(int, **kwargs)
        self.update(iterable)

    def most_common(self):
        return sorted(six.iteritems(self), key=itemgetter(1), reverse=True)

    def update(self, other):
        """Adds counts for elements in other"""
        if isinstance(other, self.__class__):
            for x, n in six.iteritems(other):
                self[x] += n
        else:
            for x in other:
                self[x] += 1

try:
    Counter = collections.Counter
except AttributeError:
    Counter = _Counter


def _unique(ar, return_index=False, return_inverse=False):
    """A replacement for the np.unique that appeared in numpy 1.4.

    While np.unique existed long before, keyword return_inverse was
    only added in 1.4.
    """
    try:
        ar = ar.flatten()
    except AttributeError:
        if not return_inverse and not return_index:
            items = sorted(set(ar))
            return np.asarray(items)
        else:
            ar = np.asarray(ar).flatten()

    if ar.size == 0:
        if return_inverse and return_index:
            return ar, np.empty(0, np.bool), np.empty(0, np.bool)
        elif return_inverse or return_index:
            return ar, np.empty(0, np.bool)
        else:
            return ar

    if return_inverse or return_index:
        perm = ar.argsort()
        aux = ar[perm]
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            iperm = perm.argsort()
            if return_index:
                return aux[flag], perm[flag], iflag[iperm]
            else:
                return aux[flag], iflag[iperm]
        else:
            return aux[flag], perm[flag]

    else:
        ar.sort()
        flag = np.concatenate(([True], ar[1:] != ar[:-1]))
        return ar[flag]

if LooseVersion(np.__version__) < LooseVersion('1.5'):
    unique = _unique
else:
    unique = np.unique


def _bincount(X, weights=None, minlength=None):
    """Replacing np.bincount in numpy < 1.6 to provide minlength."""
    result = np.bincount(X, weights)
    if minlength is None or len(result) >= minlength:
        return result
    out = np.zeros(minlength, np.int)
    out[:len(result)] = result
    return out

if LooseVersion(np.__version__) < LooseVersion('1.6'):
    bincount = _bincount
else:
    bincount = np.bincount


def _copysign(x1, x2):
    """Slow replacement for np.copysign, which was introduced in numpy 1.4"""
    return np.abs(x1) * np.sign(x2)

if not hasattr(np, 'copysign'):
    copysign = _copysign
else:
    copysign = np.copysign


def _in1d(ar1, ar2, assume_unique=False, invert=False):
    """Replacement for in1d that is provided for numpy >= 1.4"""
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # This code is significantly faster when the condition is satisfied.
    if len(ar2) < 10 * len(ar1) ** 0.145:
        if invert:
            mask = np.ones(len(ar1), dtype=np.bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=np.bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]

    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]


if not hasattr(np, 'in1d') or LooseVersion(np.__version__) < '1.8':
    in1d = _in1d
else:
    in1d = np.in1d


def _digitize(x, bins, right=False):
    """Replacement for digitize with right kwarg (numpy < 1.7).

    Notes
    -----
    This fix is only meant for integer arrays. If ``right==True`` but either
    ``x`` or ``bins`` are of a different type, a NotImplementedError will be
    raised.
    """
    if right:
        x = np.asarray(x)
        bins = np.asarray(bins)
        if (x.dtype.kind not in 'ui') or (bins.dtype.kind not in 'ui'):
            raise NotImplementedError("Only implemented for integer input")
        return np.digitize(x - 1e-5, bins)
    else:
        return np.digitize(x, bins)

if LooseVersion(np.__version__) < LooseVersion('1.7'):
    digitize = _digitize
else:
    digitize = np.digitize


def _tril_indices(n, k=0):
    """Replacement for tril_indices that is provided for numpy >= 1.4"""
    mask = np.greater_equal(np.subtract.outer(np.arange(n), np.arange(n)), -k)
    indices = np.where(mask)

    return indices

if not hasattr(np, 'tril_indices'):
    tril_indices = _tril_indices
else:
    tril_indices = np.tril_indices


def _unravel_index(indices, dims):
    """Add support for multiple indices in unravel_index that is provided
    for numpy >= 1.4"""
    indices_arr = np.asarray(indices)
    if indices_arr.size == 1:
        return np.unravel_index(indices, dims)
    else:
        if indices_arr.ndim != 1:
            raise ValueError('indices should be one dimensional')

        ndims = len(dims)
        unraveled_coords = np.empty((indices_arr.size, ndims), dtype=np.int)
        for coord, idx in zip(unraveled_coords, indices_arr):
            coord[:] = np.unravel_index(idx, dims)
        return tuple(unraveled_coords.T)


if LooseVersion(np.__version__) < LooseVersion('1.4'):
    unravel_index = _unravel_index
else:
    unravel_index = np.unravel_index


def _qr_economic_old(A, **kwargs):
    """
    Compat function for the QR-decomposition in economic mode
    Scipy 0.9 changed the keyword econ=True to mode='economic'
    """
    with warnings.catch_warnings(record=True):
        return linalg.qr(A, econ=True, **kwargs)


def _qr_economic_new(A, **kwargs):
    return linalg.qr(A, mode='economic', **kwargs)


if LooseVersion(scipy.__version__) < LooseVersion('0.9'):
    qr_economic = _qr_economic_old
else:
    qr_economic = _qr_economic_new


def savemat(file_name, mdict, oned_as="column", **kwargs):
    """MATLAB-format output routine that is compatible with SciPy 0.7's.

    0.7.2 (or .1?) added the oned_as keyword arg with 'column' as the default
    value. It issues a warning if this is not provided, stating that "This will
    change to 'row' in future versions."
    """
    import scipy.io
    try:
        return scipy.io.savemat(file_name, mdict, oned_as=oned_as, **kwargs)
    except TypeError:
        return scipy.io.savemat(file_name, mdict, **kwargs)

if hasattr(np, 'count_nonzero'):
    from numpy import count_nonzero
else:
    def count_nonzero(X):
        return len(np.flatnonzero(X))

# little dance to see if np.copy has an 'order' keyword argument
if 'order' in inspect.getargspec(np.copy)[0]:
    def safe_copy(X):
        # Copy, but keep the order
        return np.copy(X, order='K')
else:
    # Before an 'order' argument was introduced, numpy wouldn't muck with
    # the ordering
    safe_copy = np.copy


def _meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from coordinate vectors.
    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.
    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory.
        Default is False.
        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.
        .. versionadded:: 1.7.0
    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.
    """
    ndim = len(xi)

    copy_ = kwargs.pop('copy', True)
    sparse = kwargs.pop('sparse', False)
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x in enumerate(xi)]

    shape = [x.size for x in output]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + (1,) * (ndim - 2)
        output[1].shape = (-1, 1) + (1,) * (ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)

if LooseVersion(np.__version__) < LooseVersion('1.7'):
    meshgrid = _meshgrid
else:
    meshgrid = np.meshgrid


###############################################################################
# Back porting firwin2 for older scipy

# Original version of firwin2 from scipy ticket #457, submitted by "tash".
#
# Rewritten by Warren Weckesser, 2010.


def _firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=1.0):
    """FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.  If the gain at the Nyquist rate, `gain[-1]`, is not 0,
        then `numtaps` must be odd.

    freq : array-like, 1D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency can be redefined with the argument
        `nyq`.

        The values in `freq` must be nondecreasing.  A value can be repeated
        once to implement a discontinuity.  The first value in `freq` must
        be 0, and the last value must be `nyq`.

    gain : array-like
        The filter gains at the frequency sampling points.

    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc).  The default is one more than the smallest
        power of 2 that is not less than `numtaps`.  `nfreqs` must be greater
        than `numtaps`.

    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming".  See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.

    nyq : float
        Nyquist frequency.  Each frequency in `freq` must be between 0 and
        `nyq` (inclusive).

    Returns
    -------
    taps : numpy 1D array of length `numtaps`
        The filter coefficients of the FIR filter.

    Examples
    --------
    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
    that decreases linearly on [0.5, 1.0] from 1 to 0:

    >>> taps = firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])  # doctest: +SKIP
    >>> print(taps[72:78])  # doctest: +SKIP
    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]

    See also
    --------
    scipy.signal.firwin

    Notes
    -----

    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain.  The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.

    The FIR filter will have linear phase.  The filter is Type I if `numtaps`
    is odd and Type II if `numtaps` is even.  Because Type II filters always
    have a zero at the Nyquist frequency, `numtaps` must be odd if `gain[-1]`
    is not zero.

    .. versionadded:: 0.9.0

    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)

    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm

    """

    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError('ntaps must be less than nfreqs, but firwin2 was '
                         'called with ntaps=%d and nfreqs=%s'
                         % (numtaps, nfreqs))

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with `nyq`.')
    d = np.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')

    if numtaps % 2 == 0 and gain[-1] != 0.0:
        raise ValueError("A filter with an even number of coefficients must "
                         "have zero gain at the Nyquist rate.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    # Tweak any repeated values in freq so that interp works.
    eps = np.finfo(float).eps
    for k in range(len(freq)):
        if k < len(freq) - 1 and freq[k] == freq[k + 1]:
            freq[k] = freq[k] - eps
            freq[k + 1] = freq[k + 1] + eps

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = np.linspace(0.0, nyq, nfreqs)
    fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = irfft(fx2)

    if window is not None:
        # Create the window to apply to the filter coefficients.
        from scipy.signal.signaltools import get_window
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    return out


def get_firwin2():
    """Helper to get firwin2"""
    try:
        from scipy.signal import firwin2
    except ImportError:
        firwin2 = _firwin2
    return firwin2


def _filtfilt(*args, **kwargs):
    """wrap filtfilt, excluding padding arguments"""
    from scipy.signal import filtfilt
    # cut out filter args
    if len(args) > 4:
        args = args[:4]
    if 'padlen' in kwargs:
        del kwargs['padlen']
    return filtfilt(*args, **kwargs)


def get_filtfilt():
    """Helper to get filtfilt from scipy"""
    from scipy.signal import filtfilt

    if 'padlen' in inspect.getargspec(filtfilt)[0]:
        return filtfilt

    return _filtfilt


###############################################################################
# Back porting matrix_rank for numpy < 1.7


def _matrix_rank(M, tol=None):
    """ Return matrix rank of array using SVD method

    Rank of the array is the number of SVD singular values of the array that
    are greater than `tol`.

    Parameters
    ----------
    M : {(M,), (M, N)} array_like
        array of <=2 dimensions
    tol : {None, float}, optional
       threshold below which SVD values are considered zero. If `tol` is
       None, and ``S`` is an array with singular values for `M`, and
       ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
       set to ``S.max() * max(M.shape) * eps``.

    Notes
    -----
    The default threshold to detect rank deficiency is a test on the magnitude
    of the singular values of `M`. By default, we identify singular values less
    than ``S.max() * max(M.shape) * eps`` as indicating rank deficiency (with
    the symbols defined above). This is the algorithm MATLAB uses [1]. It also
    appears in *Numerical recipes* in the discussion of SVD solutions for
    linear least squares [2].

    This default threshold is designed to detect rank deficiency accounting
    for the numerical errors of the SVD computation. Imagine that there is a
    column in `M` that is an exact (in floating point) linear combination of
    other columns in `M`. Computing the SVD on `M` will not produce a
    singular value exactly equal to 0 in general: any difference of the
    smallest SVD value from 0 will be caused by numerical imprecision in the
    calculation of the SVD. Our threshold for small SVD values takes this
    numerical imprecision into account, and the default threshold will detect
    such numerical rank deficiency. The threshold may declare a matrix `M`
    rank deficient even if the linear combination of some columns of `M` is
    not exactly equal to another column of `M` but only numerically very
    close to another column of `M`.

    We chose our default threshold because it is in wide use. Other
    thresholds are possible. For example, elsewhere in the 2007 edition of
    *Numerical recipes* there is an alternative threshold of ``S.max() *
    np.finfo(M.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe
    this threshold as being based on "expected roundoff error" (p 71).

    The thresholds above deal with floating point roundoff error in the
    calculation of the SVD. However, you may have more information about the
    sources of error in `M` that would make you consider other tolerance
    values to detect *effective* rank deficiency. The most useful measure of
    the tolerance depends on the operations you intend to use on your matrix.
    For example, if your data come from uncertain measurements with
    uncertainties greater than floating point epsilon, choosing a tolerance
    near that uncertainty may be preferable. The tolerance may be absolute if
    the uncertainties are absolute rather than relative.

    References
    ----------
    .. [1] MATLAB reference documention, "Rank"
           http://www.mathworks.com/help/techdoc/ref/rank.html
    .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
           "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
           page 795.

    Examples
    --------
    >>> from numpy.linalg import matrix_rank
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    """
    M = np.asarray(M)
    if M.ndim > 2:
        raise TypeError('array should have 2 or fewer dimensions')
    if M.ndim < 2:
        return np.int(not all(M == 0))
    S = np.linalg.svd(M, compute_uv=False)
    if tol is None:
        tol = S.max() * np.max(M.shape) * np.finfo(S.dtype).eps
    return np.sum(S > tol)

if LooseVersion(np.__version__) > '1.7.1':
    from numpy.linalg import matrix_rank
else:
    matrix_rank = _matrix_rank


def _reconstruct_partial(func, args, kwargs):
    """Helper to pickle partial functions"""
    return partial(func, *args, **(kwargs or {}))


def _reduce_partial(p):
    """Helper to pickle partial functions"""
    return _reconstruct_partial, (p.func, p.args, p.keywords)

# This adds pickling functionality to older Python 2.6
# Please always import partial from here.
copyreg.pickle(partial, _reduce_partial)


def normalize_colors(vmin, vmax, clip=False):
    """Helper to handle matplotlib API"""
    import matplotlib.pyplot as plt
    try:
        return plt.Normalize(vmin, vmax, clip=clip)
    except AttributeError:
        return plt.normalize(vmin, vmax, clip=clip)


def assert_true(expr, msg='False is not True'):
    """Fake assert_true without message"""
    if not expr:
        raise AssertionError(msg)


def assert_is(expr1, expr2, msg=None):
    """Fake assert_is without message"""
    assert_true(expr2 is expr2, msg)


def assert_is_not(expr1, expr2, msg=None):
    """Fake assert_is_not without message"""
    assert_true(expr1 is not expr2, msg)


def _sparse_block_diag(mats, format=None, dtype=None):
    """An implementation of scipy.sparse.block_diag since old versions of
    scipy don't have it. Forms a sparse matrix by stacking matrices in block
    diagonal form.

    Parameters
    ----------
    mats : list of matrices
        Input matrices.
    format : str, optional
        The sparse format of the result (e.g. "csr"). If not given, the
        matrix is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of blocks.

    Returns
    -------
    res : sparse matrix
    """
    nmat = len(mats)
    rows = []
    for ia, a in enumerate(mats):
        row = [None] * nmat
        row[ia] = a
        rows.append(row)
    return sparse.bmat(rows, format=format, dtype=dtype)

try:
    from scipy.sparse import block_diag as sparse_block_diag
except Exception:
    sparse_block_diag = _sparse_block_diag


def _isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose

    Notes
    -----
    .. versionadded:: 1.7.0

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `isclose(a, b)` might be different from `isclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> isclose([1e10,1e-7], [1.00001e10,1e-8])
    array([ True, False], dtype=bool)
    >>> isclose([1e10,1e-8], [1.00001e10,1e-9])
    array([ True,  True], dtype=bool)
    >>> isclose([1e10,1e-8], [1.0001e10,1e-9])
    array([False,  True], dtype=bool)
    >>> isclose([1.0, np.nan], [1.0, np.nan])
    array([ True, False], dtype=bool)
    >>> isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    array([ True,  True], dtype=bool)
    """
    def within_tol(x, y, atol, rtol):
        with np.errstate(invalid='ignore'):
            result = np.less_equal(abs(x - y), atol + rtol * abs(y))
        if np.isscalar(a) and np.isscalar(b):
            result = bool(result)
        return result

    x = np.array(a, copy=False, subok=True, ndmin=1)
    y = np.array(b, copy=False, subok=True, ndmin=1)

    # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
    # This will cause casting of x later. Also, make sure to allow subclasses
    # (e.g., for numpy.ma).
    dt = np.core.multiarray.result_type(y, 1.)
    y = np.array(y, dtype=dt, copy=False, subok=True)

    xfin = np.isfinite(x)
    yfin = np.isfinite(y)
    if all(xfin) and all(yfin):
        return within_tol(x, y, atol, rtol)
    else:
        finite = xfin & yfin
        cond = np.zeros_like(finite, subok=True)
        # Because we're using boolean indexing, x & y must be the same shape.
        # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
        # lib.stride_tricks, though, so we can't import it here.
        x = x * np.ones_like(cond)
        y = y * np.ones_like(cond)
        # Avoid subtraction with infinite/nan values...
        cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
        # Check for equality of infinite values...
        cond[~finite] = (x[~finite] == y[~finite])
        if equal_nan:
            # Make NaN == NaN
            both_nan = np.isnan(x) & np.isnan(y)
            cond[both_nan] = both_nan[both_nan]
        return cond


if LooseVersion(np.__version__) < LooseVersion('1.7'):
    isclose = _isclose
else:
    isclose = np.isclose
