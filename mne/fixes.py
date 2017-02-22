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

from .externals.six import string_types, iteritems


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
# Backporting nibabel's read_geometry

def _get_read_geometry():
    """Get the geometry reading function."""
    try:
        import nibabel as nib
        has_nibabel = True
    except ImportError:
        has_nibabel = False
    if has_nibabel and LooseVersion(nib.__version__) > LooseVersion('2.1.0'):
        from nibabel.freesurfer import read_geometry
    else:
        read_geometry = _read_geometry
    return read_geometry


def _read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Backport from nibabel."""
    from .surface import _fread3, _fread3_many
    volume_info = dict()

    TRIANGLE_MAGIC = 16777214
    QUAD_MAGIC = 16777215
    NEW_QUAD_MAGIC = 16777213
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):  # Quad file
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            (fmt, div) = (">i2", 100.) if magic == QUAD_MAGIC else (">f4", 1.)
            coords = np.fromfile(fobj, fmt, nvert * 3).astype(np.float) / div
            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            #
            #   Face splitting follows
            #
            faces = np.zeros((2 * nquad, 3), dtype=np.int)
            nface = 0
            for quad in quads:
                if (quad[0] % 2) == 0:
                    faces[nface] = quad[0], quad[1], quad[3]
                    nface += 1
                    faces[nface] = quad[2], quad[3], quad[1]
                    nface += 1
                else:
                    faces[nface] = quad[0], quad[1], quad[2]
                    nface += 1
                    faces[nface] = quad[0], quad[2], quad[3]
                    nface += 1

        elif magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline().rstrip(b'\n').decode('utf-8')
            fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

            if read_metadata:
                volume_info = _read_volume_info(fobj)
        else:
            raise ValueError("File does not appear to be a Freesurfer surface")

    coords = coords.astype(np.float)  # XXX: due to mayavi bug on mac 32bits

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn('No volume information contained in the file')
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret


###############################################################################
# Backporting scipy.signal.sosfilt (0.17) and sosfiltfilt (0.18)


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


def minimum_phase(h):
    """Convert a linear-phase FIR filter to minimum phase.

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.
    """
    try:
        from scipy.signal import minimum_phase
    except Exception:
        pass
    else:
        return minimum_phase(h)
    from scipy.fftpack import fft, ifft
    h = np.asarray(h)
    if np.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1D and at least 2 samples long')
    n_half = len(h) // 2
    if not np.allclose(h[-n_half:][::-1], h[:n_half]):
        warnings.warn('h does not appear to by symmetric, conversion may '
                      'fail', RuntimeWarning)
    n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
    # zero-pad; calculate the DFT
    h_temp = np.abs(fft(h, n_fft))
    # take 0.25*log(|H|**2) = 0.5*log(|H|)
    h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
    np.log(h_temp, out=h_temp)
    h_temp *= 0.5
    # IDFT
    h_temp = ifft(h_temp).real
    # multiply pointwise by the homomorphic filter
    # lmin[n] = 2u[n] - d[n]
    win = np.zeros(n_fft)
    win[0] = 1
    stop = (len(h) + 1) // 2
    win[1:stop] = 2
    if len(h) % 2:
        win[stop] = 1
    h_temp *= win
    h_temp = ifft(np.exp(fft(h_temp)))
    h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]


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
# Scipy spectrogram (for mne.time_frequency.psd_welch) needed for scipy < 0.16

def _spectrogram(x, fs=1.0, window=('tukey',.25), nperseg=256, noverlap=None,
                nfft=None, detrend='constant', return_onesided=True,
                scaling='density', axis=-1, mode='psd'):
    """
    Compute a spectrogram with consecutive Fourier transforms.
    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment.  Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.  If None,
        the FFT length is `nperseg`. Defaults to None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. If False return
        a two-sided spectrum. Note that for complex data, a two-sided
        spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power spectrum
        ('spectrum') where `Pxx` has units of V**2, if `x` is measured in V
        and fs is measured in Hz.  Defaults to 'density'
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are ['psd',
        'complex', 'magnitude', 'angle', 'phase'].

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds to the
        segment times.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the entire
    data stream is averaged over, one may wish to use a smaller overlap (or
    perhaps none at all) when computing a spectrogram, to maintain some
    statistical independence between individual segments.
    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck "Discrete-Time
           Signal Processing", Prentice Hall, 1999.
    """
    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    freqs, time, Pxy = _spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided, scaling,
                                        axis, mode=mode)

    return freqs, time, Pxy


def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=256,
                    noverlap=None, nfft=None, detrend='constant',
                    return_onesided=True, scaling='spectrum', axis=-1,
                    mode='psd'):
    """
    Calculate various forms of windowed FFTs for PSD, CSD, etc.
    This is a helper function that implements the commonality between the
    psd, csd, and spectrogram functions. It is not designed to be called
    externally. The windows are not averaged over; the result from each window
    is returned.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memoery as x (i.e. _spectral_helper(x, x, ...)),
        the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to 'hann'.
    nperseg : int, optional
        Length of each segment.  Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 2``.  Defaults to None.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.  If None,
        the FFT length is `nperseg`. Defaults to None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. If False return
        a two-sided spectrum. Note that for complex data, a two-sided
        spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and fs is measured in Hz.  Defaults to 'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are ['psd',
        'complex', 'magnitude', 'angle', 'phase'].

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    References
    ----------
    .. [1] Stack Overflow, "Rolling window for 1D arrays in Numpy?",
        http://stackoverflow.com/a/6811241
    .. [2] Stack Overflow, "Using strides for an efficient moving average
        filter", http://stackoverflow.com/a/4947453

    Notes
    -----
    Adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    from scipy import fftpack
    from scipy.signal import signaltools
    from scipy.signal.windows import get_window

    if mode not in ['psd', 'complex', 'magnitude', 'angle', 'phase']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "'default', 'psd', 'complex', "
                         "'magnitude', 'angle', 'phase'" % mode)

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x,y,np.complex64)
    else:
        outdtype = np.result_type(x,np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError:
            raise ValueError('x and y cannot be broadcast together.')

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.rollaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = np.rollaxis(y, axis, len(y.shape))

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    # X and Y are same length now, can test nperseg with either
    if x.shape[-1] < nperseg:
        warnings.warn('nperseg = {0:d}, is greater than input length = {1:d}, '
                      'using nperseg = {1:d}'.format(nperseg, x.shape[-1]))
        nperseg = x.shape[-1]

    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    elif noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    else:
        noverlap = int(noverlap)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend

    if isinstance(window, string_types) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    if np.result_type(win,np.complex64) != outdtype:
        win = win.astype(outdtype)

    if mode == 'psd':
        if scaling == 'density':
            scale = 1.0 / (fs * (win*win).sum())
        elif scaling == 'spectrum':
            scale = 1.0 / win.sum()**2
        else:
            raise ValueError('Unknown scaling: %r' % scaling)
    else:
        scale = 1

    if return_onesided is True:
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
    else:
        sides = 'twosided'

    if sides == 'twosided':
        num_freqs = nfft
    elif sides == 'onesided':
        if nfft % 2:
            num_freqs = (nfft + 1)//2
        else:
            num_freqs = nfft//2 + 1

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft)
    result = result[..., :num_freqs]
    freqs = fftpack.fftfreq(nfft, 1/fs)[:num_freqs]

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft)
        result_y = result_y[..., :num_freqs]
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result
    elif mode == 'magnitude':
        result = np.absolute(result)
    elif mode == 'angle' or mode == 'phase':
        result = np.angle(result)
    elif mode == 'complex':
        pass

    result *= scale
    if sides == 'onesided':
        if nfft % 2:
            result[...,1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[...,1:-1] *= 2

    t = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1, nperseg - noverlap)/float(fs)

    if sides != 'twosided' and not nfft % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # we unwrap the phase here to handle the onesided vs. twosided case
    if mode == 'phase':
        result = np.unwrap(result, axis=-1)

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'complex':
        result = result.real

    # Output is going to have new last axis for window index
    if axis != -1:
        # Specify as positive axis index
        if axis < 0:
            axis = len(result.shape)-1-axis

        # Roll frequency axis back to axis where the data came from
        result = np.rollaxis(result, -1, axis)
    else:
        # Make sure window/time index is last axis
        result = np.rollaxis(result, -1, -2)

    return freqs, t, result


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft):
    """
    Calculate windowed FFT, for internal use by scipy.signal._spectral_helper
    This is a helper function that does the main FFT calculation for
    _spectral helper. All input valdiation is performed there, and the data
    axis is assumed to be the last axis of x. It is not designed to be called
    externally. The windows are not averaged over; the result from each window
    is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    References
    ----------
    .. [1] Stack Overflow, "Repeat NumPy array without replicating data?",
        http://stackoverflow.com/a/5568169

    Notes
    -----
    Adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    from scipy import fftpack

    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    result = fftpack.fft(result, n=nfft)

    return result


def get_spectrogram():
    '''helper function to get relevant spectrogram'''
    from .utils import check_version
    if check_version('scipy', '0.16.0'):
        from scipy.signal import spectrogram
    else:
        spectrogram = _spectrogram
    return spectrogram


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


##############################################################################
# adapted from scikit-learn

class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        try:
            from inspect import signature
        except ImportError:
            from .externals.funcsigs import signature
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        from sklearn.base import _pprint
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    # __getstate__ and __setstate__ are omitted because they only contain
    # conditionals that are not satisfied by our objects (e.g.,
    # ``if type(self).__module__.startswith('sklearn.')``.
