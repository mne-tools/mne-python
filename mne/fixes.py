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

import collections
from operator import itemgetter
import inspect

import numpy as np

try:
    Counter = collections.Counter
except AttributeError:
    class Counter(collections.defaultdict):
        """Partial replacement for Python 2.7 collections.Counter."""
        def __init__(self, iterable=(), **kwargs):
            super(Counter, self).__init__(int, **kwargs)
            self.update(iterable)

        def most_common(self):
            return sorted(self.iteritems(), key=itemgetter(1), reverse=True)

        def update(self, other):
            """Adds counts for elements in other"""
            if isinstance(other, self.__class__):
                for x, n in other.iteritems():
                    self[x] += n
            else:
                for x in other:
                    self[x] += 1


def lsqr(X, y, tol=1e-3):
    import scipy.sparse.linalg as sp_linalg
    from ..utils.extmath import safe_sparse_dot

    if hasattr(sp_linalg, 'lsqr'):
        # scipy 0.8 or greater
        return sp_linalg.lsqr(X, y)
    else:
        n_samples, n_features = X.shape
        if n_samples > n_features:
            coef, _ = sp_linalg.cg(safe_sparse_dot(X.T, X),
                                   safe_sparse_dot(X.T, y),
                                   tol=tol)
        else:
            coef, _ = sp_linalg.cg(safe_sparse_dot(X, X.T), y, tol=tol)
            coef = safe_sparse_dot(X.T, coef)

        residues = y - safe_sparse_dot(X, coef)
        return coef, None, None, residues


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

np_version = []
for x in np.__version__.split('.'):
    try:
        np_version.append(int(x))
    except ValueError:
        # x may be of the form dev-1ea1592
        np_version.append(x)

if np_version[:2] < (1, 5):
    unique = _unique
else:
    unique = np.unique


def _bincount(X, weights=None, minlength=None):
    """Replacing np.bincount in numpy < 1.6 to provide minlength."""
    result = np.bincount(X, weights)
    if len(result) >= minlength:
        return result
    out = np.zeros(minlength, np.int)
    out[:len(result)] = result
    return out

if np_version[:2] < (1, 6):
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


if np_version[:2] < (1, 4):
    unravel_index = _unravel_index
else:
    unravel_index = np.unravel_index


def qr_economic(A, **kwargs):
    """Compat function for the QR-decomposition in economic mode

    Scipy 0.9 changed the keyword econ=True to mode='economic'
    """
    import scipy.linalg
    # trick: triangular solve has introduced in 0.9
    if hasattr(scipy.linalg, 'solve_triangular'):
        return scipy.linalg.qr(A, mode='economic', **kwargs)
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return scipy.linalg.qr(A, econ=True, **kwargs)


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

try:
    from numpy import count_nonzero
except ImportError:
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


# imported from scipy git master on 2012/12/12
def _filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
    """A forward-backward filter.

    This function applies a linear filter twice, once forward
    and once backwards.  The combined filter has linear phase.

    Before applying the filter, the function can pad the data along the
    given axis in one of three ways: odd, even or constant.  The odd
    and even extensions have the corresponding symmetry about the end point
    of the data.  The constant extension extends the data with the values
    at end points.  On both the forward and backwards passes, the
    initial condition of the filter is found by using `lfilter_zi` and
    scaling it by the end point of the extended data.

    Parameters
    ----------
    b : array_like, 1-D
        The numerator coefficient vector of the filter.
    a : array_like, 1-D
        The denominator coefficient vector of the filter.  If a[0]
        is not 1, then both a and b are normalized by a[0].
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter. This value must be less than
        `x.shape[axis]-1`.  `padlen=0` implies no padding.
        The default value is 3*max(len(a),len(b)).

    Returns
    -------
    y : ndarray
        The filtered output, an array of type numpy.float64 with the same
        shape as `x`.

    See Also
    --------
    lfilter_zi, lfilter

    Examples
    --------
    First we create a one second signal that is the sum of two pure sine
    waves, with frequencies 5 Hz and 250 Hz, sampled at 2000 Hz.

    >>> t = np.linspace(0, 1.0, 2001)
    >>> xlow = np.sin(2 * np.pi * 5 * t)
    >>> xhigh = np.sin(2 * np.pi * 250 * t)
    >>> x = xlow + xhigh

    Now create a lowpass Butterworth filter with a cutoff of 0.125 times
    the Nyquist rate, or 125 Hz, and apply it to x with filtfilt.  The
    result should be approximately xlow, with no phase shift.

    >>> from scipy import signal
    >>> b, a = signal.butter(8, 0.125)
    >>> y = signal.filtfilt(b, a, x, padlen=150)
    >>> np.abs(y - xlow).max()
    9.1086182074789912e-06

    We get a fairly clean result for this artificial example because
    the odd extension is exact, and with the moderately long padding,
    the filter's transients have dissipated by the time the actual data
    is reached.  In general, transient effects at the edges are
    unavoidable.
    """

    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype must "
                         "be 'even', 'odd', 'constant', or None.") %
                            padtype)

    b = np.asarray(b)
    a = np.asarray(a)
    x = np.asarray(x)

    ntaps = max(len(a), len(b))

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be at least "
                         "padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x

    # Get the steady state of the filter's step response.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y

from scipy.signal import filtfilt
if 'padlen' not in inspect.getargspec(filtfilt)[0]:
    filtfilt = _filtfilt
