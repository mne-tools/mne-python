"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.

# XXX : originally copied from scikit-learn

"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD

from __future__ import division
import collections
from distutils.version import LooseVersion
from functools import partial
from gzip import GzipFile
import inspect
from math import ceil, log
from operator import itemgetter
import re
import warnings

import numpy as np
from numpy.fft import irfft
import scipy
from scipy import linalg, sparse

from .externals import six
from .externals.six.moves import copyreg, xrange


###############################################################################
# Misc

# helpers to get function arguments
if hasattr(inspect, 'signature'):  # py35
    def _get_args(function, varargs=False):
        params = inspect.signature(function).parameters
        args = [key for key, param in params.items()
                if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
        if varargs:
            varargs = [param.name for param in params.values()
                       if param.kind == param.VAR_POSITIONAL]
            if len(varargs) == 0:
                varargs = None
            return args, varargs
        else:
            return args
else:
    def _get_args(function, varargs=False):
        out = inspect.getargspec(function)  # args, varargs, keywords, defaults
        if varargs:
            return out[:2]
        else:
            return out[0]


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


def _safe_svd(A, **kwargs):
    """Wrapper to get around the SVD did not converge error of death"""
    # Intel has a bug with their GESVD driver:
    #     https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/628049  # noqa
    # For SciPy 0.18 and up, we can work around it by using
    # lapack_driver='gesvd' instead.
    if kwargs.get('overwrite_a', False):
        raise ValueError('Cannot set overwrite_a=True with this function')
    try:
        return linalg.svd(A, **kwargs)
    except np.linalg.LinAlgError as exp:
        if 'lapack_driver' in _get_args(linalg.svd):
            warn('SVD error (%s), attempting to use GESVD instead of GESDD'
                 % (exp,))
            return linalg.svd(A, lapack_driver='gesvd', **kwargs)
        else:
            raise


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
if 'order' in _get_args(np.copy):
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


def _filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
    """copy of modern SciPy filtfilt without "method" or "irlen" arguments"""
    from scipy.signal import lfilter_zi, lfilter
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    x = np.asarray(x)

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=max(len(a), len(b)))

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


def _sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """copy of SciPy sosfiltfilt"""
    sos, n_sections = _validate_sos(sos)

    # `method` is "pad"...
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=ntaps)

    # These steps follow the same form as filtfilt with modifications
    zi = sosfilt_zi(sos)  # shape (n_sections, 2) --> (n_sections, ..., 2, ...)
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 2
    zi.shape = [n_sections] + zi_shape
    x_0 = axis_slice(ext, stop=1, axis=axis)
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x_0)
    y_0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0)
    y = axis_reverse(y, axis=axis)
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)
    return y


def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a'"""
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[a_slice]
    return b


def axis_reverse(a, axis=-1):
    """Reverse the 1-d slices of `a` along axis `axis`."""
    return axis_slice(a, step=-1, axis=axis)


def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype "
                          "must be 'even', 'odd', 'constant', or None.") %
                         padtype)

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
    return edge, ext


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


def odd_ext(x, n, axis=-1):
    """Generate a new ndarray by making an odd extension of x along an axis."""
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = np.concatenate((2 * left_end - left_ext,
                          x,
                          2 * right_end - right_ext),
                         axis=axis)
    return ext


def even_ext(x, n, axis=-1):
    """Create an ndarray that is an even extension of x along an axis."""
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = np.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    return ext


def const_ext(x, n, axis=-1):
    """Create an ndarray that is a constant extension of x along an axis"""
    if n < 1:
        return x
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = np.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = np.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    return ext


def sosfilt_zi(sos):
    """Compute an initial state `zi` for the sosfilt function"""
    from scipy.signal import lfilter_zi
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2))
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section] = scale * lfilter_zi(b, a)
        # If H(z) = B(z)/A(z) is this section's transfer function, then
        # b.sum()/a.sum() is H(1), the gain at omega=0.  That's the steady
        # state value of this section's step response.
        scale *= b.sum() / a.sum()

    return zi


def sosfilt(sos, x, axis=-1, zi=None):
    """Filter data along one dimension using cascaded second-order sections"""
    from scipy.signal import lfilter
    x = np.asarray(x)

    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')

    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')

    use_zi = zi is not None
    if use_zi:
        zi = np.asarray(zi)
        x_zi_shape = list(x.shape)
        x_zi_shape[axis] = 2
        x_zi_shape = tuple([n_sections] + x_zi_shape)
        if zi.shape != x_zi_shape:
            raise ValueError('Invalid zi shape.  With axis=%r, an input with '
                             'shape %r, and an sos array with %d sections, zi '
                             'must have shape %r.' %
                             (axis, x.shape, n_sections, x_zi_shape))
        zf = np.zeros_like(zi)

    for section in range(n_sections):
        if use_zi:
            x, zf[section] = lfilter(sos[section, :3], sos[section, 3:],
                                     x, axis, zi=zi[section])
        else:
            x = lfilter(sos[section, :3], sos[section, 3:], x, axis)
    out = (x, zf) if use_zi else x
    return out


def get_filtfilt():
    """Helper to get filtfilt from scipy"""
    from scipy.signal import filtfilt
    if 'padlen' in _get_args(filtfilt):
        return filtfilt
    else:
        return _filtfilt


def get_sosfiltfilt():
    """Helper to get sosfiltfilt from scipy"""
    try:
        from scipy.signal import sosfiltfilt
    except ImportError:
        sosfiltfilt = _sosfiltfilt
    return sosfiltfilt


def _get_argrelmax():
    try:
        from scipy.signal import argrelmax
    except ImportError:
        argrelmax = _argrelmax
    return argrelmax


def _argrelmax(data, axis=0, order=1, mode='clip'):
    """Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'.  See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.
    """
    comparator = np.greater
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in xrange(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return np.where(results)


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


assert_raises_regex_impl = None


# from numpy 1.9.1
def assert_raises_regex(exception_class, expected_regexp,
                        callable_obj=None, *args, **kwargs):
    """
    Fail unless an exception of class exception_class and with message that
    matches expected_regexp is thrown by callable when invoked with arguments
    args and keyword arguments kwargs.
    Name of this function adheres to Python 3.2+ reference, but should work in
    all versions down to 2.6.
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import nose

    global assert_raises_regex_impl
    if assert_raises_regex_impl is None:
        try:
            # Python 3.2+
            assert_raises_regex_impl = nose.tools.assert_raises_regex
        except AttributeError:
            try:
                # 2.7+
                assert_raises_regex_impl = nose.tools.assert_raises_regexp
            except AttributeError:
                # 2.6

                # This class is copied from Python2.7 stdlib almost verbatim
                class _AssertRaisesContext(object):

                    def __init__(self, expected, expected_regexp=None):
                        self.expected = expected
                        self.expected_regexp = expected_regexp

                    def failureException(self, msg):
                        return AssertionError(msg)

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc_value, tb):
                        if exc_type is None:
                            try:
                                exc_name = self.expected.__name__
                            except AttributeError:
                                exc_name = str(self.expected)
                            raise self.failureException(
                                "{0} not raised".format(exc_name))
                        if not issubclass(exc_type, self.expected):
                            # let unexpected exceptions pass through
                            return False
                        self.exception = exc_value  # store for later retrieval
                        if self.expected_regexp is None:
                            return True

                        expected_regexp = self.expected_regexp
                        if isinstance(expected_regexp, basestring):
                            expected_regexp = re.compile(expected_regexp)
                        if not expected_regexp.search(str(exc_value)):
                            raise self.failureException(
                                '"%s" does not match "%s"' %
                                (expected_regexp.pattern, str(exc_value)))
                        return True

                def impl(cls, regex, callable_obj, *a, **kw):
                    mgr = _AssertRaisesContext(cls, regex)
                    if callable_obj is None:
                        return mgr
                    with mgr:
                        callable_obj(*a, **kw)
                assert_raises_regex_impl = impl

    return assert_raises_regex_impl(exception_class, expected_regexp,
                                    callable_obj, *args, **kwargs)


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
    if np.all(xfin) and np.all(yfin):
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
