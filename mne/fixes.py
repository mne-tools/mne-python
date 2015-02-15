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
from scipy.signal import filtfilt as sp_filtfilt
from distutils.version import LooseVersion
from functools import partial
from .externals import six
from .externals.six.moves import copyreg
from gzip import GzipFile


###############################################################################
# Misc

class gzip_open(GzipFile):  # python2.6 doesn't have context managing
    def __init__(self, *args, **kwargs):
        return GzipFile.__init__(self, *args, **kwargs)

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


def _in1d(ar1, ar2, assume_unique=False):
    """Replacement for in1d that is provided for numpy >= 1.4"""
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)
    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    equal_adj = (sar[1:] == sar[:-1])
    flag = np.concatenate((equal_adj, [False]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]

    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]

if not hasattr(np, 'in1d'):
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


# wrap filtfilt, excluding padding arguments
def _filtfilt(*args, **kwargs):
    # cut out filter args
    if len(args) > 4:
        args = args[:4]
    if 'padlen' in kwargs:
        del kwargs['padlen']
    return sp_filtfilt(*args, **kwargs)

if 'padlen' not in inspect.getargspec(sp_filtfilt)[0]:
    filtfilt = _filtfilt
else:
    filtfilt = sp_filtfilt


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

    >>> taps = firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    >>> print(taps[72:78])
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

if hasattr(scipy.signal, 'firwin2'):
    from scipy.signal import firwin2
else:
    firwin2 = _firwin2


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
    if 'Normalize' in vars(plt):
        return plt.Normalize(vmin, vmax, clip=clip)
    else:
        return plt.normalize(vmin, vmax, clip=clip)


def _assert_true(expr, msg):
    """Fake assert_true without message"""
    assert expr

try:
    from nose.tools import assert_true
except ImportError:
    assert_true = _assert_true


def _assert_is(expr1, expr2, msg=None):
    """Fake assert_is without message"""
    assert_true(expr2 is expr2, msg)


def _assert_is_not(expr1, expr2, msg=None):
    """Fake assert_is_not without message"""
    assert_true(expr2 is not expr2, msg)

try:
    from nose.tools import assert_is, assert_is_not
except ImportError:
    assert_is = _assert_is
    assert_is_not = _assert_is_not


def _sparse_block_diag(mats, fmt=None, dtype=None):
    """An implementation of scipy.sparse.block_diag since old versions of
    scipy don't have it. Forms a sparse matrix by stacking matrices in block
    diagonal form.

    Parameters
    ----------
    mats : list of matrices
        Input matrices.
    fmt : str, optional
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
    return sparse.bmat(rows, fmt=fmt, dtype=dtype)

try:
    from scipy.sparse import block_diag as sparse_block_diag
except Exception:
    sparse_block_diag = _sparse_block_diag


"""Numpy nanmean"""


def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    """
    is_new = not isinstance(a, np.ndarray)
    if is_new:
        a = np.array(a)
    if not issubclass(a.dtype.type, np.inexact):
        return a, None
    if not is_new:
        # need copy
        a = np.array(a, subok=True)

    mask = np.isnan(a)
    np.copyto(a, val, where=mask)
    return a, mask


def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    output. If out is None, then then a is used instead so that the
    division is in place. Note that this is only called with `a` an inexact
    type.

    Parameters
    ----------
    a : {ndarray, numpy scalar}
        Numerator. Expected to be of inexact type but not checked.
    b : {ndarray, numpy scalar}
        Denominator.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    Returns
    -------
    ret : {ndarray, numpy scalar}
        The return value is a/b. If `a` was an ndarray the division is done
        in place. If `a` is a numpy scalar, the division preserves its type.

    """
    with np.errstate(invalid='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / float(b))
            else:
                # This is questionable, but currently a numpy scalar can
                # be output to a zero dimensional array.
                return np.divide(a, b, out=out, casting='unsafe')


def _nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original `arr`.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    higher-precision accumulator using the `dtype` keyword can alleviate
    this issue.
    """
    if keepdims is True:
        keepdims_ = [slice(None, None, None) for _ in arr.ndim]
        keepdims_[axis] = np.newaxis
    else:
        keepdims = Ellipsis
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out)[keepdims_]

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    # The warning context speeds things up.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cnt = np.sum(~mask, axis=axis, dtype=np.intp)[keepdims_]
        tot = np.sum(arr, axis=axis, dtype=dtype, out=out)[keepdims_]
        avg = _divide_by_count(tot, cnt, out=out)

    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg


try:
    from numpy import nanmean
except ImportError:
    nanmean = _nanmean
