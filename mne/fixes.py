"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.

# XXX : originally copied from scikit-learn

"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD

from __future__ import division

import inspect
from distutils.version import LooseVersion
import re
import warnings

import numpy as np
from scipy import linalg, __version__ as sp_version


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


def _safe_svd(A, **kwargs):
    """Wrapper to get around the SVD did not converge error of death"""
    # Intel has a bug with their GESVD driver:
    #     https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/628049  # noqa: E501
    # For SciPy 0.18 and up, we can work around it by using
    # lapack_driver='gesvd' instead.
    if kwargs.get('overwrite_a', False):
        raise ValueError('Cannot set overwrite_a=True with this function')
    try:
        return linalg.svd(A, **kwargs)
    except np.linalg.LinAlgError as exp:
        from .utils import warn
        if 'lapack_driver' in _get_args(linalg.svd):
            warn('SVD error (%s), attempting to use GESVD instead of GESDD'
                 % (exp,))
            return linalg.svd(A, lapack_driver='gesvd', **kwargs)
        else:
            raise


###############################################################################
# Back porting scipy.signal.sosfilt (0.17) and sosfiltfilt (0.18)


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


def get_sosfiltfilt():
    """Helper to get sosfiltfilt from scipy"""
    try:
        from scipy.signal import sosfiltfilt
    except ImportError:
        sosfiltfilt = _sosfiltfilt
    return sosfiltfilt


###############################################################################
# scipy.special.sph_harm ()

def _sph_harm(order, degree, az, pol):
    """Evaluate point in specified multipolar moment.

    When using, pay close attention to inputs. Spherical harmonic notation for
    order/degree, and theta/phi are both reversed in original SSS work compared
    to many other sources. See mathworld.wolfram.com/SphericalHarmonic.html for
    more discussion.

    Note that scipy has ``scipy.special.sph_harm``, but that function is
    too slow on old versions (< 0.15) for heavy use.

    Parameters
    ----------
    order : int
        Order of spherical harmonic. (Usually) corresponds to 'm'.
    degree : int
        Degree of spherical harmonic. (Usually) corresponds to 'l'.
    az : float
        Azimuthal (longitudinal) spherical coordinate [0, 2*pi]. 0 is aligned
        with x-axis.
    pol : float
        Polar (or colatitudinal) spherical coordinate [0, pi]. 0 is aligned
        with z-axis.
    norm : bool
        If True, include normalization factor.

    Returns
    -------
    base : complex float
        The spherical harmonic value.
    """
    from scipy.special import lpmv
    from .preprocessing.maxwell import _sph_harm_norm

    # Error checks
    if np.abs(order) > degree:
        raise ValueError('Absolute value of order must be <= degree')
    # Ensure that polar and azimuth angles are arrays
    az = np.asarray(az)
    pol = np.asarray(pol)
    if (np.abs(az) > 2 * np.pi).any():
        raise ValueError('Azimuth coords must lie in [-2*pi, 2*pi]')
    if(pol < 0).any() or (pol > np.pi).any():
        raise ValueError('Polar coords must lie in [0, pi]')
    # This is the "seismology" convention on Wikipedia, w/o Condon-Shortley
    sph = lpmv(order, degree, np.cos(pol)) * np.exp(1j * order * az)
    sph *= _sph_harm_norm(order, degree)
    return sph


def _get_sph_harm():
    """Helper to get a usable spherical harmonic function."""
    if LooseVersion(sp_version) < LooseVersion('0.17.1'):
        sph_harm = _sph_harm
    else:
        from scipy.special import sph_harm
    return sph_harm


###############################################################################
# Misc utilities

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


def _read_volume_info(fobj):
    """An implementation of nibabel.freesurfer.io._read_volume_info, since old
    versions of nibabel (<=2.1.0) don't have it.
    """
    volume_info = dict()
    head = np.fromfile(fobj, '>i4', 1)
    if not np.array_equal(head, [20]):  # Read two bytes more
        head = np.concatenate([head, np.fromfile(fobj, '>i4', 2)])
        if not np.array_equal(head, [2, 0, 20]):
            warnings.warn("Unknown extension code.")
            return volume_info

    volume_info['head'] = head
    for key in ['valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras',
                'zras', 'cras']:
        pair = fobj.readline().decode('utf-8').split('=')
        if pair[0].strip() != key or len(pair) != 2:
            raise IOError('Error parsing volume info.')
        if key in ('valid', 'filename'):
            volume_info[key] = pair[1].strip()
        elif key == 'volume':
            volume_info[key] = np.array(pair[1].split()).astype(int)
        else:
            volume_info[key] = np.array(pair[1].split()).astype(float)
    # Ignore the rest
    return volume_info


def _serialize_volume_info(volume_info):
    """An implementation of nibabel.freesurfer.io._serialize_volume_info, since
    old versions of nibabel (<=2.1.0) don't have it."""
    keys = ['head', 'valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras',
            'zras', 'cras']
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError('Invalid volume info: %s.' % diff.pop())

    strings = list()
    for key in keys:
        if key == 'head':
            if not (np.array_equal(volume_info[key], [20]) or np.array_equal(
                    volume_info[key], [2, 0, 20])):
                warnings.warn("Unknown extension code.")
            strings.append(np.array(volume_info[key], dtype='>i4').tostring())
        elif key in ('valid', 'filename'):
            val = volume_info[key]
            strings.append('{0} = {1}\n'.format(key, val).encode('utf-8'))
        elif key == 'volume':
            val = volume_info[key]
            strings.append('{0} = {1} {2} {3}\n'.format(
                key, val[0], val[1], val[2]).encode('utf-8'))
        else:
            val = volume_info[key]
            strings.append('{0} = {1:0.10g} {2:0.10g} {3:0.10g}\n'.format(
                key.ljust(6), val[0], val[1], val[2]).encode('utf-8'))
    return b''.join(strings)
