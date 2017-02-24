"""IIR and FIR filtering and resampling functions."""

from copy import deepcopy
from functools import partial

import numpy as np
from scipy.fftpack import fft, ifftshift, fftfreq, ifft

from .cuda import (setup_cuda_fft_multiply_repeated, fft_multiply_repeated,
                   setup_cuda_fft_resample, fft_resample, _smart_pad)
from .externals.six import string_types, integer_types
from .fixes import get_sosfiltfilt, minimum_phase
from .parallel import parallel_func, check_n_jobs
from .time_frequency.multitaper import dpss_windows, _mt_spectra
from .utils import (logger, verbose, sum_squared, check_version, warn,
                    deprecated)

# These values are *double* what is given in Ifeachor and Jervis.
_length_factors = dict(hann=6.2, hamming=6.6, blackman=11.0)


def is_power2(num):
    """Test if number is a power of 2.

    Parameters
    ----------
    num : int
        Number.

    Returns
    -------
    b : bool
        True if is power of 2.

    Examples
    --------
    >>> is_power2(2 ** 3)
    True
    >>> is_power2(5)
    False
    """
    num = int(num)
    return num != 0 and ((num & (num - 1)) == 0)


def next_fast_len(target):
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    Copied from SciPy with minor modifications.
    """
    from bisect import bisect_left
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            p2 = 2 ** (quotient - 1).bit_length()

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _overlap_add_filter(x, h, n_fft=None, phase='zero', picks=None,
                        n_jobs=1, copy=True):
    """Filter the signal x using h with overlap-add FFTs.

    Parameters
    ----------
    x : array, shape (n_signals, n_times)
        Signals to filter.
    h : 1d array
        Filter impulse response (FIR filter coefficients). Must be odd length
        if phase == 'linear'.
    n_fft : int
        Length of the FFT. If None, the best size is determined automatically.
    phase : str
        If 'zero', the delay for the filter is compensated (and it must be
        an odd-length symmetric filter). If 'linear', the response is
        uncompensated. If 'zero-double', the filter is applied in the
        forward and reverse directions. If 'minimum', a minimum-phase
        filter will be used.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.

    Returns
    -------
    xf : array, shape (n_signals, n_times)
        x filtered.
    """
    n_jobs = check_n_jobs(n_jobs, allow_cuda=True)
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    # Extend the signal by mirroring the edges to reduce transient filter
    # response
    _check_zero_phase_length(len(h), phase)
    if len(h) == 1:
        return x * h ** 2 if phase == 'zero-double' else x * h
    n_edge = max(min(len(h), x.shape[1]) - 1, 0)
    logger.debug('Smart-padding with:  %s samples on each edge' % n_edge)
    n_x = x.shape[1] + 2 * n_edge

    if phase == 'zero-double':
        h = np.convolve(h, h[::-1])

    # Determine FFT length to use
    min_fft = 2 * len(h) - 1
    if n_fft is None:
        max_fft = n_x
        if max_fft >= min_fft:
            # cost function based on number of multiplications
            N = 2 ** np.arange(np.ceil(np.log2(min_fft)),
                               np.ceil(np.log2(max_fft)) + 1, dtype=int)
            cost = (np.ceil(n_x / (N - len(h) + 1).astype(np.float)) *
                    N * (np.log2(N) + 1))

            # add a heuristic term to prevent too-long FFT's which are slow
            # (not predicted by mult. cost alone, 4e-5 exp. determined)
            cost += 4e-5 * N * n_x

            n_fft = N[np.argmin(cost)]
        else:
            # Use only a single block
            n_fft = next_fast_len(min_fft)
    logger.debug('FFT block length:   %s' % n_fft)
    if n_fft < min_fft:
        raise ValueError('n_fft is too short, has to be at least '
                         '2 * len(h) - 1 (%s), got %s' % (min_fft, n_fft))

    # Filter in frequency domain
    h_fft = fft(np.concatenate([h, np.zeros(n_fft - len(h), dtype=h.dtype)]))

    # Figure out if we should use CUDA
    n_jobs, cuda_dict, h_fft = setup_cuda_fft_multiply_repeated(n_jobs, h_fft)

    # Process each row separately
    picks = np.arange(len(x)) if picks is None else picks
    if n_jobs == 1:
        for p in picks:
            x[p] = _1d_overlap_filter(x[p], h_fft, len(h), n_edge, phase,
                                      cuda_dict)
    else:
        parallel, p_fun, _ = parallel_func(_1d_overlap_filter, n_jobs)
        data_new = parallel(p_fun(x[p], h_fft, len(h), n_edge, phase,
                                  cuda_dict) for p in picks)
        for pp, p in enumerate(picks):
            x[p] = data_new[pp]

    x.shape = orig_shape
    return x


def _1d_overlap_filter(x, h_fft, n_h, n_edge, phase, cuda_dict):
    """Do one-dimensional overlap-add FFT FIR filtering."""
    # pad to reduce ringing
    if cuda_dict['use_cuda']:
        n_fft = cuda_dict['x'].size  # account for CUDA's modification of h_fft
    else:
        n_fft = len(h_fft)
    x_ext = _smart_pad(x, np.array([n_edge, n_edge]))
    n_x = len(x_ext)
    x_filtered = np.zeros_like(x_ext)

    n_seg = n_fft - n_h + 1
    n_segments = int(np.ceil(n_x / float(n_seg)))
    shift = ((n_h - 1) // 2 if phase.startswith('zero') else 0) + n_edge

    # Now the actual filtering step is identical for zero-phase (filtfilt-like)
    # or single-pass
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = (seg_idx + 1) * n_seg
        seg = x_ext[start:stop]
        seg = np.concatenate([seg, np.zeros(n_fft - len(seg))])

        prod = fft_multiply_repeated(h_fft, seg, cuda_dict)

        start_filt = max(0, start - shift)
        stop_filt = min(start - shift + n_fft, n_x)
        start_prod = max(0, shift - start)
        stop_prod = start_prod + stop_filt - start_filt
        x_filtered[start_filt:stop_filt] += prod[start_prod:stop_prod]

    # Remove mirrored edges that we added and cast (n_edge can be zero)
    x_filtered = x_filtered[:n_x - 2 * n_edge].astype(x.dtype)
    return x_filtered


def _filter_attenuation(h, freq, gain):
    """Compute minimum attenuation at stop frequency."""
    from scipy.signal import freqz
    _, filt_resp = freqz(h.ravel(), worN=np.pi * freq)
    filt_resp = np.abs(filt_resp)  # use amplitude response
    filt_resp[np.where(gain == 1)] = 0
    idx = np.argmax(filt_resp)
    att_db = -20 * np.log10(np.maximum(filt_resp[idx], 1e-20))
    att_freq = freq[idx]
    return att_db, att_freq


def _prep_for_filtering(x, copy, picks=None):
    """Set up array as 2D for filtering ease."""
    if x.dtype != np.float64:
        raise TypeError("Arrays passed for filtering must have a dtype of "
                        "np.float64")
    if copy is True:
        x = x.copy()
    orig_shape = x.shape
    x = np.atleast_2d(x)
    x.shape = (np.prod(x.shape[:-1]), x.shape[-1])
    if picks is None:
        picks = np.arange(x.shape[0])
    elif len(orig_shape) == 3:
        n_epochs, n_channels, n_times = orig_shape
        offset = np.repeat(np.arange(0, n_channels * n_epochs, n_channels),
                           len(picks))
        picks = np.tile(picks, n_epochs) + offset
    elif len(orig_shape) > 3:
        raise ValueError('picks argument is not supported for data with more'
                         ' than three dimensions')
    picks = np.array(picks, int).ravel()
    if not all(0 <= pick < x.shape[0] for pick in picks) or \
            len(set(picks)) != len(picks):
        raise ValueError('bad argument for "picks": %s' % (picks,))

    return x, orig_shape, picks


def _construct_fir_filter(sfreq, freq, gain, filter_length, phase, fir_window):
    """Filter signal using gain control points in the frequency domain.

    The filter impulse response is constructed from a Hann window (window
    used in "firwin2" function) to avoid ripples in the frequency response
    (windowing is a smoothing in frequency domain).

    If x is multi-dimensional, this operates along the last dimension.

    Parameters
    ----------
    Fs : float
        Sampling rate in Hz.
    freq : 1d array
        Frequency sampling points in Hz.
    gain : 1d array
        Filter gain at frequency sampling points.
    filter_length : int
        Length of the filter to use. Must be odd length if phase == "zero".
    phase : str
        If 'zero', the delay for the filter is compensated (and it must be
        an odd-length symmetric filter). If 'linear', the response is
        uncompensated. If 'zero-double', the filter is applied in the
        forward and reverse directions. If 'minimum', a minimum-phase
        filter will be used.
    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann", or "blackman".

    Returns
    -------
    xf : array
        x filtered.
    """
    from scipy.signal import firwin2

    # issue a warning if attenuation is less than this
    min_att_db = 12 if phase == 'minimum' else 20

    # normalize frequencies
    freq = np.array(freq) / (sfreq / 2.)
    if freq[0] != 0 or freq[-1] != 1:
        raise ValueError('freq must start at 0 and end an Nyquist (%s), got %s'
                         % (sfreq / 2., freq))
    gain = np.array(gain)

    # Use overlap-add filter with a fixed length
    N = _check_zero_phase_length(filter_length, phase, gain[-1])
    # construct symmetric (linear phase) filter
    if phase == 'minimum':
        h = firwin2(N * 2 - 1, freq, gain, window=fir_window)
        h = minimum_phase(h)
    else:
        h = firwin2(N, freq, gain, window=fir_window)
    assert h.size == N
    att_db, att_freq = _filter_attenuation(h, freq, gain)
    if phase == 'zero-double':
        att_db += 6
    if att_db < min_att_db:
        att_freq *= sfreq / 2.
        warn('Attenuation at stop frequency %0.1fHz is only %0.1fdB. '
             'Increase filter_length for higher attenuation.'
             % (att_freq, att_db))
    return h


def _check_zero_phase_length(N, phase, gain_nyq=0):
    N = int(N)
    if N % 2 == 0:
        if phase == 'zero':
            raise RuntimeError('filter_length must be odd if phase="zero", '
                               'got %s' % N)
        elif phase == 'zero-double' and gain_nyq == 1:
            N += 1
    return N


def _check_coefficients(system):
    """Check for filter stability."""
    if isinstance(system, tuple):
        from scipy.signal import tf2zpk
        z, p, k = tf2zpk(*system)
    else:  # sos
        from scipy.signal import sos2zpk
        z, p, k = sos2zpk(system)
    if np.any(np.abs(p) > 1.0):
        raise RuntimeError('Filter poles outside unit circle, filter will be '
                           'unstable. Consider using different filter '
                           'coefficients.')


def _filtfilt(x, iir_params, picks, n_jobs, copy):
    """Helper to more easily call filtfilt."""
    # set up array for filtering, reshape to 2D, operate on last axis
    from scipy.signal import filtfilt
    padlen = min(iir_params['padlen'], len(x))
    n_jobs = check_n_jobs(n_jobs)
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    if 'sos' in iir_params:
        sosfiltfilt = get_sosfiltfilt()
        fun = partial(sosfiltfilt, sos=iir_params['sos'], padlen=padlen)
        _check_coefficients(iir_params['sos'])
    else:
        fun = partial(filtfilt, b=iir_params['b'], a=iir_params['a'],
                      padlen=padlen)
        _check_coefficients((iir_params['b'], iir_params['a']))
    if n_jobs == 1:
        for p in picks:
            x[p] = fun(x=x[p])
    else:
        parallel, p_fun, _ = parallel_func(fun, n_jobs)
        data_new = parallel(p_fun(x=x[p]) for p in picks)
        for pp, p in enumerate(picks):
            x[p] = data_new[pp]
    x.shape = orig_shape
    return x


def estimate_ringing_samples(system, max_try=100000):
    """Estimate filter ringing.

    Parameters
    ----------
    system : tuple | ndarray
        A tuple of (b, a) or ndarray of second-order sections coefficients.
    max_try : int
        Approximate maximum number of samples to try.
        This will be changed to a multple of 1000.

    Returns
    -------
    n : int
        The approximate ringing.
    """
    from scipy import signal
    if isinstance(system, tuple):  # TF
        kind = 'ba'
        b, a = system
        zi = [0.] * (len(a) - 1)
    else:
        kind = 'sos'
        sos = system
        zi = [[0.] * 2] * len(sos)
    n_per_chunk = 1000
    n_chunks_max = int(np.ceil(max_try / float(n_per_chunk)))
    x = np.zeros(n_per_chunk)
    x[0] = 1
    last_good = n_per_chunk
    thresh_val = 0
    for ii in range(n_chunks_max):
        if kind == 'ba':
            h, zi = signal.lfilter(b, a, x, zi=zi)
        else:
            h, zi = signal.sosfilt(sos, x, zi=zi)
        x[0] = 0  # for subsequent iterations we want zero input
        h = np.abs(h)
        thresh_val = max(0.001 * np.max(h), thresh_val)
        idx = np.where(np.abs(h) > thresh_val)[0]
        if len(idx) > 0:
            last_good = idx[-1]
        else:  # this iteration had no sufficiently lange values
            idx = (ii - 1) * n_per_chunk + last_good
            break
    else:
        warn('Could not properly estimate ringing for the filter')
        idx = n_per_chunk * n_chunks_max
    return idx


def construct_iir_filter(iir_params, f_pass=None, f_stop=None, sfreq=None,
                         btype=None, return_copy=True):
    """Use IIR parameters to get filtering coefficients.

    This function works like a wrapper for iirdesign and iirfilter in
    scipy.signal to make filter coefficients for IIR filtering. It also
    estimates the number of padding samples based on the filter ringing.
    It creates a new iir_params dict (or updates the one passed to the
    function) with the filter coefficients ('b' and 'a') and an estimate
    of the padding necessary ('padlen') so IIR filtering can be performed.

    .. note:: As of 0.14, second-order sections will be used in filter
              design by default (replacing ``output='ba'`` by
              ``output='sos'``) to help ensure filter stability and
              reduce numerical error. Second-order sections filtering
              requires SciPy >= 16.0.


    Parameters
    ----------
    iir_params : dict
        Dictionary of parameters to use for IIR filtering.

            * If ``iir_params['sos']`` exists, it will be used as
              second-order sections to perform IIR filtering.

              .. versionadded:: 0.13

            * Otherwise, if ``iir_params['b']`` and ``iir_params['a']``
              exist, these will be used as coefficients to perform IIR
              filtering.
            * Otherwise, if ``iir_params['order']`` and
              ``iir_params['ftype']`` exist, these will be used with
              `scipy.signal.iirfilter` to make a filter.
            * Otherwise, if ``iir_params['gpass']`` and
              ``iir_params['gstop']`` exist, these will be used with
              `scipy.signal.iirdesign` to design a filter.
            * ``iir_params['padlen']`` defines the number of samples to pad
              (and an estimate will be calculated if it is not given).
              See Notes for more details.
            * ``iir_params['output']`` defines the system output kind when
              designing filters, either "sos" or "ba". For 0.13 the
              default is 'ba' but will change to 'sos' in 0.14.

    f_pass : float or list of float
        Frequency for the pass-band. Low-pass and high-pass filters should
        be a float, band-pass should be a 2-element list of float.
    f_stop : float or list of float
        Stop-band frequency (same size as f_pass). Not used if 'order' is
        specified in iir_params.
    sfreq : float | None
        The sample rate.
    btype : str
        Type of filter. Should be 'lowpass', 'highpass', or 'bandpass'
        (or analogous string representations known to
        :func:`scipy.signal.iirfilter`).
    return_copy : bool
        If False, the 'sos', 'b', 'a', and 'padlen' entries in
        ``iir_params`` will be set inplace (if they weren't already).
        Otherwise, a new ``iir_params`` instance will be created and
        returned with these entries.

    Returns
    -------
    iir_params : dict
        Updated iir_params dict, with the entries (set only if they didn't
        exist before) for 'sos' (or 'b', 'a'), and 'padlen' for
        IIR filtering.

    See Also
    --------
    mne.filter.filter_data
    mne.io.Raw.filter

    Notes
    -----
    This function triages calls to :func:`scipy.signal.iirfilter` and
    :func:`scipy.signal.iirdesign` based on the input arguments (see
    linked functions for more details).

    Examples
    --------
    iir_params can have several forms. Consider constructing a low-pass
    filter at 40 Hz with 1000 Hz sampling rate.

    In the most basic (2-parameter) form of iir_params, the order of the
    filter 'N' and the type of filtering 'ftype' are specified. To get
    coefficients for a 4th-order Butterworth filter, this would be:

    >>> iir_params = dict(order=4, ftype='butter', output='sos')  # doctest:+SKIP
    >>> iir_params = construct_iir_filter(iir_params, 40, None, 1000, 'low', return_copy=False)  # doctest:+SKIP
    >>> print((2 * len(iir_params['sos']), iir_params['padlen']))  # doctest:+SKIP
    (4, 82)

    Filters can also be constructed using filter design methods. To get a
    40 Hz Chebyshev type 1 lowpass with specific gain characteristics in the
    pass and stop bands (assuming the desired stop band is at 45 Hz), this
    would be a filter with much longer ringing:

    >>> iir_params = dict(ftype='cheby1', gpass=3, gstop=20, output='sos')  # doctest:+SKIP
    >>> iir_params = construct_iir_filter(iir_params, 40, 50, 1000, 'low')  # doctest:+SKIP
    >>> print((2 * len(iir_params['sos']), iir_params['padlen']))  # doctest:+SKIP
    (6, 439)

    Padding and/or filter coefficients can also be manually specified. For
    a 10-sample moving window with no padding during filtering, for example,
    one can just do:

    >>> iir_params = dict(b=np.ones((10)), a=[1, 0], padlen=0)
    >>> iir_params = construct_iir_filter(iir_params, return_copy=False)
    >>> print((iir_params['b'], iir_params['a'], iir_params['padlen']))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), [1, 0], 0)

    For more information, see the tutorials :ref:`tut_background_filtering`
    and :ref:`tut_artifacts_filter`.
    """  # noqa: E501
    from scipy.signal import iirfilter, iirdesign
    known_filters = ('bessel', 'butter', 'butterworth', 'cauer', 'cheby1',
                     'cheby2', 'chebyshev1', 'chebyshev2', 'chebyshevi',
                     'chebyshevii', 'ellip', 'elliptic')
    if not isinstance(iir_params, dict):
        raise TypeError('iir_params must be a dict, got %s' % type(iir_params))
    system = None
    # if the filter has been designed, we're good to go
    if 'sos' in iir_params:
        system = iir_params['sos']
        output = 'sos'
    elif 'a' in iir_params and 'b' in iir_params:
        system = (iir_params['b'], iir_params['a'])
        output = 'ba'
    else:
        output = iir_params.get('output', None)
        if output is None:
            warn('The default output type is "ba" in 0.13 but will change '
                 'to "sos" in 0.14')
            output = 'ba'
        if not isinstance(output, string_types) or output not in ('ba', 'sos'):
            raise ValueError('Output must be "ba" or "sos", got %s'
                             % (output,))
        # ensure we have a valid ftype
        if 'ftype' not in iir_params:
            raise RuntimeError('ftype must be an entry in iir_params if ''b'' '
                               'and ''a'' are not specified')
        ftype = iir_params['ftype']
        if ftype not in known_filters:
            raise RuntimeError('ftype must be in filter_dict from '
                               'scipy.signal (e.g., butter, cheby1, etc.) not '
                               '%s' % ftype)

        # use order-based design
        Wp = np.asanyarray(f_pass) / (float(sfreq) / 2)
        if 'order' in iir_params:
            system = iirfilter(iir_params['order'], Wp, btype=btype,
                               ftype=ftype, output=output)
        else:
            # use gpass / gstop design
            Ws = np.asanyarray(f_stop) / (float(sfreq) / 2)
            if 'gpass' not in iir_params or 'gstop' not in iir_params:
                raise ValueError('iir_params must have at least ''gstop'' and'
                                 ' ''gpass'' (or ''N'') entries')
            system = iirdesign(Wp, Ws, iir_params['gpass'],
                               iir_params['gstop'], ftype=ftype, output=output)

    if system is None:
        raise RuntimeError('coefficients could not be created from iir_params')
    # do some sanity checks
    _check_coefficients(system)

    # now deal with padding
    if 'padlen' not in iir_params:
        padlen = estimate_ringing_samples(system)
    else:
        padlen = iir_params['padlen']

    if return_copy:
        iir_params = deepcopy(iir_params)

    iir_params.update(dict(padlen=padlen))
    if output == 'sos':
        iir_params.update(sos=system)
    else:
        iir_params.update(b=system[0], a=system[1])
    return iir_params


def _check_method(method, iir_params, extra_types=()):
    """Helper to parse method arguments."""
    allowed_types = ['iir', 'fir', 'fft'] + list(extra_types)
    if not isinstance(method, string_types):
        raise TypeError('method must be a string')
    if method not in allowed_types:
        raise ValueError('method must be one of %s, not "%s"'
                         % (allowed_types, method))
    if method == 'fft':
        method = 'fir'  # use the better name
    if method == 'iir':
        if iir_params is None:
            iir_params = dict()
        if len(iir_params) == 0 or (len(iir_params) == 1 and
                                    'output' in iir_params):
            iir_params = dict(order=4, ftype='butter',
                              output=iir_params.get('output', 'sos'))
    elif iir_params is not None:
        raise ValueError('iir_params must be None if method != "iir"')
    return iir_params, method


@verbose
def filter_data(data, sfreq, l_freq, h_freq, picks=None, filter_length='auto',
                l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
                method='fir', iir_params=None, copy=True, phase='zero',
                fir_window='hamming', verbose=None):
    """Filter a subset of channels.

    Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
    filter to the channels selected by ``picks``.

    ``l_freq`` and ``h_freq`` are the frequencies below which and above
    which, respectively, to filter out of the data. Thus the uses are:

        * ``l_freq < h_freq``: band-pass filter
        * ``l_freq > h_freq``: band-stop filter
        * ``l_freq is not None and h_freq is None``: high-pass filter
        * ``l_freq is None and h_freq is not None``: low-pass filter

    .. note:: If n_jobs > 1, more memory is required as
              ``len(picks) * n_times`` additional time points need to
              be temporaily stored in memory.

    Parameters
    ----------
    data : ndarray, shape (..., n_times)
        The data to filter.
    sfreq : float
        The sample frequency in Hz.
    l_freq : float | None
        Low cut-off frequency in Hz. If None the data are only low-passed.
    h_freq : float | None
        High cut-off frequency in Hz. If None the data are only
        high-passed.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Currently this is only supported for
        2D (n_channels, n_times) and 3D (n_epochs, n_channels, n_times)
        arrays.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default in 0.14): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: (default in 0.13 is "10s") a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'``, then this filter
        is applied twice, once forward, and once backward. If 'minimum',
        then a minimum-phase, causal filter will be used.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.

    Returns
    -------
    data : ndarray, shape (..., n_times)
        The filtered data.

    See Also
    --------
    construct_iir_filter
    create_filter
    mne.io.Raw.filter
    notch_filter
    resample

    Notes
    -----
    For more information, see the tutorials :ref:`tut_background_filtering`
    and :ref:`tut_artifacts_filter`, and :func:`mne.filter.create_filter`.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError('data must be an array')
    iir_params, method = _check_method(method, iir_params)
    filt = create_filter(
        data, sfreq, l_freq, h_freq, filter_length, l_trans_bandwidth,
        h_trans_bandwidth, method, iir_params, phase, fir_window)
    if method in ('fir', 'fft'):
        data = _overlap_add_filter(data, filt, None, phase, picks, n_jobs,
                                   copy)
    else:
        data = _filtfilt(data, filt, picks, n_jobs, copy)
    return data


@verbose
def create_filter(data, sfreq, l_freq, h_freq, filter_length='auto',
                  l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                  method='fir', iir_params=None, phase='zero',
                  fir_window='hamming', verbose=None):
    r"""Create a FIR or IIR filter.

    ``l_freq`` and ``h_freq`` are the frequencies below which and above
    which, respectively, to filter out of the data. Thus the uses are:

        * ``l_freq < h_freq``: band-pass filter
        * ``l_freq > h_freq``: band-stop filter
        * ``l_freq is not None and h_freq is None``: high-pass filter
        * ``l_freq is None and h_freq is not None``: low-pass filter

    Parameters
    ----------
    data : ndarray, shape (..., n_times)
        The data that will be filtered. This is used for sanity checking
        only.
    sfreq : float
        The sample frequency in Hz.
    l_freq : float | None
        Low cut-off frequency in Hz. If None the data are only low-passed.
    h_freq : float | None
        High cut-off frequency in Hz. If None the data are only
        high-passed.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'``, then this filter
        is applied twice, once forward, and once backward. If 'minimum',
        then a minimum-phase, causal filter will be used.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann", or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.

    Returns
    -------
    filt : array or dict
        Will be an array of FIR coefficients for method='fir', and dict
        with IIR parameters for method='iir'.

    See Also
    --------
    filter_data

    Notes
    -----
    **Band-pass filter**

    The frequency response is (approximately) given by::

       1-|               ----------
         |             /|         | \
     |H| |            / |         |  \
         |           /  |         |   \
         |          /   |         |    \
       0-|----------    |         |     --------------
         |         |    |         |     |            |
         0        Fs1  Fp1       Fp2   Fs2          Nyq

    Where:

        * Fs1 = Fp1 - l_trans_bandwidth in Hz
        * Fs2 = Fp2 + h_trans_bandwidth in Hz


    **Band-stop filter**

    The frequency response is (approximately) given by::

        1-|---------                   ----------
          |         \                 /
      |H| |          \               /
          |           \             /
          |            \           /
        0-|             -----------
          |        |    |         |    |        |
          0       Fp1  Fs1       Fs2  Fp2      Nyq

    Where ``Fs1 = Fp1 + l_trans_bandwidth`` and
    ``Fs2 = Fp2 - h_trans_bandwidth``.

    Multiple stop bands can be specified using arrays.

    **Low-pass filter**

    The frequency response is (approximately) given by::

        1-|------------------------
          |                        \
      |H| |                         \
          |                          \
          |                           \
        0-|                            ----------------
          |                       |    |              |
          0                      Fp  Fstop           Nyq

    Where ``Fstop = Fp + trans_bandwidth``.

    **High-pass filter**

    The frequency response is (approximately) given by::

        1-|             -----------------------
          |            /
      |H| |           /
          |          /
          |         /
        0-|---------
          |        |    |                     |
          0      Fstop  Fp                   Nyq

    Where ``Fstop = Fp - trans_bandwidth``.

    .. versionadded:: 0.14
    """
    sfreq = float(sfreq)
    if sfreq < 0:
        raise ValueError('sfreq must be positive')
    if h_freq is not None:
        h_freq = np.array(h_freq, float).ravel()
        if (h_freq > (sfreq / 2.)).any():
            raise ValueError('h_freq (%s) must be less than the Nyquist '
                             'frequency %s' % (h_freq, sfreq / 2.))
    if l_freq is not None:
        l_freq = np.array(l_freq, float).ravel()
        if (l_freq == 0).all():
            l_freq = None
    iir_params, method = _check_method(method, iir_params)
    if l_freq is None and h_freq is None:
        data, sfreq, _, _, _, _, filter_length, phase, fir_window = \
            _triage_filter_params(
                data, sfreq, None, None, None, None,
                filter_length, method, phase, fir_window)
        if method == 'iir':
            out = dict() if iir_params is None else deepcopy(iir_params)
            out.update(b=np.array([1.]), a=np.array([1.]))
        else:
            freq = [0, sfreq / 2.]
            gain = [1., 1.]
    if l_freq is None and h_freq is not None:
        logger.info('Setting up low-pass filter at %0.2g Hz' % (h_freq,))
        data, sfreq, _, f_p, _, f_s, filter_length, phase, fir_window = \
            _triage_filter_params(
                data, sfreq, None, h_freq, None, h_trans_bandwidth,
                filter_length, method, phase, fir_window)
        if method == 'iir':
            out = construct_iir_filter(iir_params, f_p, f_s, sfreq, 'low')
        else:  # 'fir'
            freq = [0, f_p, f_s]
            gain = [1, 1, 0]
            if f_s != sfreq / 2.:
                freq += [sfreq / 2.]
                gain += [0]
    elif l_freq is not None and h_freq is None:
        logger.info('Setting up high-pass filter at %0.2g Hz' % (l_freq,))
        data, sfreq, pass_, _, stop, _, filter_length, phase, fir_window = \
            _triage_filter_params(
                data, sfreq, l_freq, None, l_trans_bandwidth, None,
                filter_length, method, phase, fir_window)
        if method == 'iir':
            out = construct_iir_filter(iir_params, pass_, stop, sfreq,
                                       'high')
        else:  # 'fir'
            freq = [stop, pass_, sfreq / 2.]
            gain = [0, 1, 1]
            if stop != 0:
                freq = [0] + freq
                gain = [0] + gain
    elif l_freq is not None and h_freq is not None:
        if (l_freq < h_freq).any():
            logger.info('Setting up band-pass filter from %0.2g - %0.2g Hz'
                        % (l_freq, h_freq))
            data, sfreq, f_p1, f_p2, f_s1, f_s2, filter_length, phase, \
                fir_window = _triage_filter_params(
                    data, sfreq, l_freq, h_freq, l_trans_bandwidth,
                    h_trans_bandwidth, filter_length, method, phase,
                    fir_window)
            if method == 'iir':
                out = construct_iir_filter(iir_params, [f_p1, f_p2],
                                           [f_s1, f_s2], sfreq, 'bandpass')
            else:  # 'fir'
                freq = [f_s1, f_p1, f_p2, f_s2]
                gain = [0, 1, 1, 0]
                if f_s2 != sfreq / 2.:
                    freq += [sfreq / 2.]
                    gain += [0]
                if f_s1 != 0:
                    freq = [0] + freq
                    gain = [0] + gain
        else:
            # This could possibly be removed after 0.14 release, but might
            # as well leave it in to sanity check notch_filter
            if len(l_freq) != len(h_freq):
                raise ValueError('l_freq and h_freq must be the same length')
            msg = 'Setting up band-stop filter'
            if len(l_freq) == 1:
                msg += ' from %0.2g - %0.2g Hz' % (h_freq, l_freq)
            logger.info(msg)
            # Note: order of outputs is intentionally switched here!
            data, sfreq, f_s1, f_s2, f_p1, f_p2, filter_length, phase, \
                fir_window = _triage_filter_params(
                    data, sfreq, h_freq, l_freq, h_trans_bandwidth,
                    l_trans_bandwidth, filter_length, method, phase,
                    fir_window, bands='arr', reverse=True)
            if method == 'iir':
                if len(f_p1) != 1:
                    raise ValueError('Multiple stop-bands can only be used '
                                     'with FIR filtering')
                out = construct_iir_filter(iir_params, [f_p1[0], f_p2[0]],
                                           [f_s1[0], f_s2[0]], sfreq,
                                           'bandstop')
            else:  # 'fir'
                freq = np.r_[f_p1, f_s1, f_s2, f_p2]
                gain = np.r_[np.ones_like(f_p1), np.zeros_like(f_s1),
                             np.zeros_like(f_s2), np.ones_like(f_p2)]
                order = np.argsort(freq)
                freq = freq[order]
                gain = gain[order]
                if freq[0] != 0:
                    freq = np.r_[[0.], freq]
                    gain = np.r_[[1.], gain]
                if freq[-1] != sfreq / 2.:
                    freq = np.r_[freq, [sfreq / 2.]]
                    gain = np.r_[gain, [1.]]
                if np.any(np.abs(np.diff(gain, 2)) > 1):
                    raise ValueError('Stop bands are not sufficiently '
                                     'separated.')
    if method == 'fir':
        out = _construct_fir_filter(sfreq, freq, gain, filter_length, phase,
                                    fir_window)
    return out


@deprecated('band_pass_filter is deprecated and will be removed in 0.15, '
            'use filter_data instead.')
@verbose
def band_pass_filter(x, Fs, Fp1, Fp2, filter_length='auto',
                     l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                     method='fir', iir_params=None, picks=None, n_jobs=1,
                     copy=True, phase='zero', fir_window='hamming',
                     verbose=None):
    """Bandpass filter for the signal x.

    Applies a zero-phase bandpass filter to the signal x, operating on the
    last dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    Fp1 : float
        Low cut-off frequency in Hz.
    Fp2 : float
        High cut-off frequency in Hz.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default in 0.14): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: (default in 0.13 is "10s") a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        Can be "auto" (default in 0.14) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        Can be "auto" (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default in 0.14), the delay of this filter
        is compensated for. If ``phase=='zero-double'`` (default in 0.13
        and before), then this filter is applied twice, once forward, and
        once backward.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default in
        0.14), "hann" (default in 0.13), or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    filter_data
    notch_filter
    resample

    Notes
    -----
    The frequency response is (approximately) given by::

       1-|               ----------
         |             /|         | \
     |H| |            / |         |  \
         |           /  |         |   \
         |          /   |         |    \
       0-|----------    |         |     --------------
         |         |    |         |     |            |
         0        Fs1  Fp1       Fp2   Fs2          Nyq

    Where:

        * Fs1 = Fp1 - l_trans_bandwidth in Hz
        * Fs2 = Fp2 + h_trans_bandwidth in Hz

    """
    return filter_data(
        x, Fs, Fp1, Fp2, picks, filter_length, l_trans_bandwidth,
        h_trans_bandwidth, n_jobs, method, iir_params, copy, phase,
        fir_window)


@deprecated('band_stop_filter is deprecated and will be removed in 0.15, '
            'use filter_data instead.')
@verbose
def band_stop_filter(x, Fs, Fp1, Fp2, filter_length='auto',
                     l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                     method='fir', iir_params=None, picks=None, n_jobs=1,
                     copy=True, phase='zero', fir_window='hamming',
                     verbose=None):
    """Bandstop filter for the signal x.

    Applies a zero-phase bandstop filter to the signal x, operating on the
    last dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    Fp1 : float | array of float
        Low cut-off frequency in Hz.
    Fp2 : float | array of float
        High cut-off frequency in Hz.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        Can be "auto" (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        Can be "auto" (default) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'`` then this filter
        is applied twice, once forward, and once backward.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann", or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    filter_data
    notch_filter
    resample

    Notes
    -----
    The frequency response is (approximately) given by::

        1-|---------                   ----------
          |         \                 /
      |H| |          \               /
          |           \             /
          |            \           /
        0-|             -----------
          |        |    |         |    |        |
          0       Fp1  Fs1       Fs2  Fp2      Nyq

    Where ``Fs1 = Fp1 + l_trans_bandwidth`` and
    ``Fs2 = Fp2 - h_trans_bandwidth``.

    Multiple stop bands can be specified using arrays.
    """
    return filter_data(x, Fs, Fp2, Fp1, picks, filter_length,
                       h_trans_bandwidth, l_trans_bandwidth, n_jobs, method,
                       iir_params, copy, phase, fir_window)


@deprecated('low_pass_filter is deprecated and will be removed in 0.15, '
            'use filter_data instead.')
@verbose
def low_pass_filter(x, Fs, Fp, filter_length='auto', trans_bandwidth='auto',
                    method='fir', iir_params=None, picks=None, n_jobs=1,
                    copy=True, phase='zero', fir_window='hamming',
                    verbose=None):
    """Lowpass filter for the signal x.

    Applies a zero-phase lowpass filter to the signal x, operating on the
    last dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    Fp : float
        Cut-off frequency in Hz.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    trans_bandwidth : float | str
        Width of the transition band in Hz. Can be "auto"
        (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'`` then this filter
        is applied twice, once forward, and once backward.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    filter_data
    notch_filter
    resample

    Notes
    -----
    The frequency response is (approximately) given by::

        1-|------------------------
          |                        \
      |H| |                         \
          |                          \
          |                           \
        0-|                            ----------------
          |                       |    |              |
          0                      Fp  Fstop           Nyq

    Where ``Fstop = Fp + trans_bandwidth``.
    """
    return filter_data(
        x, Fs, None, Fp, picks, filter_length, 'auto', trans_bandwidth, n_jobs,
        method, iir_params, copy, phase, fir_window)


@deprecated('high_pass_filter is deprecated and will be removed in 0.15, '
            'use filter_data instead.')
@verbose
def high_pass_filter(x, Fs, Fp, filter_length='auto', trans_bandwidth='auto',
                     method='fir', iir_params=None, picks=None, n_jobs=1,
                     copy=True, phase='zero', fir_window='hamming',
                     verbose=None):
    """Highpass filter for the signal x.

    Applies a zero-phase highpass filter to the signal x, operating on the
    last dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    Fp : float
        Cut-off frequency in Hz.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    trans_bandwidth : float | str
        Width of the transition band in Hz. Can be "auto"
        (default) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'`` then this filter
        is applied twice, once forward, and once backward.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    filter_data
    notch_filter
    resample

    Notes
    -----
    The frequency response is (approximately) given by::

        1-|             -----------------------
          |            /
      |H| |           /
          |          /
          |         /
        0-|---------
          |        |    |                     |
          0      Fstop  Fp                   Nyq

    Where ``Fstop = Fp - trans_bandwidth``.
    """
    return filter_data(
        x, Fs, Fp, None, picks, filter_length, trans_bandwidth, 'auto', n_jobs,
        method, iir_params, copy, phase, fir_window)


@verbose
def notch_filter(x, Fs, freqs, filter_length='auto', notch_widths=None,
                 trans_bandwidth=1, method='fir', iir_params=None,
                 mt_bandwidth=None, p_value=0.05, picks=None, n_jobs=1,
                 copy=True, phase='zero', fir_window='hamming',
                 verbose=None):
    r"""Notch filter for the signal x.

    Applies a zero-phase notch filter to the signal x, operating on the last
    dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    freqs : float | array of float | None
        Frequencies to notch filter in Hz, e.g. np.arange(60, 241, 60).
        None can only be used with the mode 'spectrum_fit', where an F
        test is used to find sinusoidal components.
    filter_length : str | int
        Length of the FIR filter to use (if applicable):

            * int: specified length in samples.
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming').
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.

    notch_widths : float | array of float | None
        Width of the stop band (centred at each freq in freqs) in Hz.
        If None, freqs / 200 is used.
    trans_bandwidth : float
        Width of the transition band in Hz.
        Only used for ``method='fir'``.
    method : str
        'fir' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt). 'spectrum_fit' will
        use multi-taper estimation of sinusoidal components. If freqs=None
        and method='spectrum_fit', significant sinusoidal components
        are detected using an F test, and noted by logging.
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'spectrum_fit' mode.
    p_value : float
        p-value to use in F-test thresholding to determine significant
        sinusoidal components to remove when method='spectrum_fit' and
        freqs=None. Note that this will be Bonferroni corrected for the
        number of frequencies, so large p-values may be justified.
    picks : array-like of int | None
        Indices of channels to filter. If None all channels will be
        filtered. Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fir'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    phase : str
        Phase of the filter, only used if ``method='fir'``.
        By default, a symmetric linear-phase FIR filter is constructed.
        If ``phase='zero'`` (default), the delay of this filter
        is compensated for. If ``phase=='zero-double'``, then this filter
        is applied twice, once forward, and once backward. If 'minimum',
        then a minimum-phase, causal filter will be used.

        .. versionadded:: 0.13

    fir_window : str
        The window to use in FIR design, can be "hamming" (default),
        "hann" (default in 0.13), or "blackman".

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    filter_data
    resample

    Notes
    -----
    The frequency response is (approximately) given by::

        1-|----------         -----------
          |          \       /
      |H| |           \     /
          |            \   /
          |             \ /
        0-|              -
          |         |    |    |         |
          0        Fp1 freq  Fp2       Nyq

    For each freq in freqs, where ``Fp1 = freq - trans_bandwidth / 2`` and
    ``Fs2 = freq + trans_bandwidth / 2``.

    References
    ----------
    Multi-taper removal is inspired by code from the Chronux toolbox, see
    www.chronux.org and the book "Observed Brain Dynamics" by Partha Mitra
    & Hemant Bokil, Oxford University Press, New York, 2008. Please
    cite this in publications if method 'spectrum_fit' is used.
    """
    iir_params, method = _check_method(method, iir_params, ['spectrum_fit'])

    if freqs is not None:
        freqs = np.atleast_1d(freqs)
    elif method != 'spectrum_fit':
        raise ValueError('freqs=None can only be used with method '
                         'spectrum_fit')

    # Only have to deal with notch_widths for non-autodetect
    if freqs is not None:
        if notch_widths is None:
            notch_widths = freqs / 200.0
        elif np.any(notch_widths < 0):
            raise ValueError('notch_widths must be >= 0')
        else:
            notch_widths = np.atleast_1d(notch_widths)
            if len(notch_widths) == 1:
                notch_widths = notch_widths[0] * np.ones_like(freqs)
            elif len(notch_widths) != len(freqs):
                raise ValueError('notch_widths must be None, scalar, or the '
                                 'same length as freqs')

    if method in ('fir', 'iir'):
        # Speed this up by computing the fourier coefficients once
        tb_2 = trans_bandwidth / 2.0
        lows = [freq - nw / 2.0 - tb_2
                for freq, nw in zip(freqs, notch_widths)]
        highs = [freq + nw / 2.0 + tb_2
                 for freq, nw in zip(freqs, notch_widths)]
        xf = filter_data(x, Fs, highs, lows, picks, filter_length, tb_2, tb_2,
                         n_jobs, method, iir_params, copy, phase, fir_window)
    elif method == 'spectrum_fit':
        xf = _mt_spectrum_proc(x, Fs, freqs, notch_widths, mt_bandwidth,
                               p_value, picks, n_jobs, copy)

    return xf


def _mt_spectrum_proc(x, sfreq, line_freqs, notch_widths, mt_bandwidth,
                      p_value, picks, n_jobs, copy):
    """Helper to more easily call _mt_spectrum_remove."""
    from scipy import stats
    # set up array for filtering, reshape to 2D, operate on last axis
    n_jobs = check_n_jobs(n_jobs)
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)

    # XXX need to implement the moving window version for raw files
    n_times = x.shape[1]

    # max taper size chosen because it has an max error < 1e-3:
    # >>> np.max(np.diff(dpss_windows(953, 4, 100)[0]))
    # 0.00099972447657578449
    # so we use 1000 because it's the first "nice" number bigger than 953:
    dpss_n_times_max = 1000

    # figure out what tapers to use
    if mt_bandwidth is not None:
        half_nbw = float(mt_bandwidth) * n_times / (2 * sfreq)
    else:
        half_nbw = 4

    # compute dpss windows
    n_tapers_max = int(2 * half_nbw)
    window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                       low_bias=False,
                                       interp_from=min(n_times,
                                                       dpss_n_times_max))
    # F-stat of 1-p point
    threshold = stats.f.ppf(1 - p_value / n_times, 2, 2 * len(window_fun) - 2)

    if n_jobs == 1:
        freq_list = list()
        for ii, x_ in enumerate(x):
            if ii in picks:
                x[ii], f = _mt_spectrum_remove(x_, sfreq, line_freqs,
                                               notch_widths, window_fun,
                                               threshold)
                freq_list.append(f)
    else:
        parallel, p_fun, _ = parallel_func(_mt_spectrum_remove, n_jobs)
        data_new = parallel(p_fun(x_, sfreq, line_freqs, notch_widths,
                                  window_fun, threshold)
                            for xi, x_ in enumerate(x)
                            if xi in picks)
        freq_list = [d[1] for d in data_new]
        data_new = np.array([d[0] for d in data_new])
        x[picks, :] = data_new

    # report found frequencies
    for rm_freqs in freq_list:
        if line_freqs is None:
            if len(rm_freqs) > 0:
                logger.info('Detected notch frequencies:\n%s'
                            % ', '.join([str(rm_f) for rm_f in rm_freqs]))
            else:
                logger.info('Detected notch frequecies:\nNone')

    x.shape = orig_shape
    return x


def _mt_spectrum_remove(x, sfreq, line_freqs, notch_widths,
                        window_fun, threshold):
    """Use MT-spectrum to remove line frequencies.

    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """
    # drop the even tapers
    n_tapers = len(window_fun)
    tapers_odd = np.arange(0, n_tapers, 2)
    tapers_even = np.arange(1, n_tapers, 2)
    tapers_use = window_fun[tapers_odd]

    # sum tapers for (used) odd prolates across time (n_tapers, 1)
    H0 = np.sum(tapers_use, axis=1)

    # sum of squares across tapers (1, )
    H0_sq = sum_squared(H0)

    # make "time" vector
    rads = 2 * np.pi * (np.arange(x.size) / float(sfreq))

    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = _mt_spectra(x[np.newaxis, :], window_fun, sfreq)

    # sum of the product of x_p and H0 across tapers (1, n_freqs)
    x_p_H0 = np.sum(x_p[:, tapers_odd, :] *
                    H0[np.newaxis, :, np.newaxis], axis=1)

    # resulting calculated amplitudes for all freqs
    A = x_p_H0 / H0_sq

    if line_freqs is None:
        # figure out which freqs to remove using F stat

        # estimated coefficient
        x_hat = A * H0[:, np.newaxis]

        # numerator for F-statistic
        num = (n_tapers - 1) * (A * A.conj()).real * H0_sq
        # denominator for F-statistic
        den = (np.sum(np.abs(x_p[:, tapers_odd, :] - x_hat) ** 2, 1) +
               np.sum(np.abs(x_p[:, tapers_even, :]) ** 2, 1))
        den[den == 0] = np.inf
        f_stat = num / den

        # find frequencies to remove
        indices = np.where(f_stat > threshold)[1]
        rm_freqs = freqs[indices]
    else:
        # specify frequencies
        indices_1 = np.unique([np.argmin(np.abs(freqs - lf))
                               for lf in line_freqs])
        notch_widths /= 2.0
        indices_2 = [np.logical_and(freqs > lf - nw, freqs < lf + nw)
                     for lf, nw in zip(line_freqs, notch_widths)]
        indices_2 = np.where(np.any(np.array(indices_2), axis=0))[0]
        indices = np.unique(np.r_[indices_1, indices_2])
        rm_freqs = freqs[indices]

    fits = list()
    for ind in indices:
        c = 2 * A[0, ind]
        fit = np.abs(c) * np.cos(freqs[ind] * rads + np.angle(c))
        fits.append(fit)

    if len(fits) == 0:
        datafit = 0.0
    else:
        # fitted sinusoids are summed, and subtracted from data
        datafit = np.sum(np.atleast_2d(fits), axis=0)

    return x - datafit, rm_freqs


@verbose
def resample(x, up, down, npad=100, axis=-1, window='boxcar', n_jobs=1,
             verbose=None):
    """Resample an array.

    Operates along the last dimension of the array.

    Parameters
    ----------
    x : n-d array
        Signal to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.
    npad : int | str
        Number of samples to use at the beginning and end for padding.
        Can be "auto" to pad to the next highest power of 2.
    axis : int
        Axis along which to resample (default is the last axis).
    window : string or tuple
        See :func:`scipy.signal.resample` for description.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    xf : array
        x resampled.

    Notes
    -----
    This uses (hopefully) intelligent edge padding and frequency-domain
    windowing improve scipy.signal.resample's resampling method, which
    we have adapted for our use here. Choices of npad and window have
    important consequences, and the default choices should work well
    for most natural signals.

    Resampling arguments are broken into "up" and "down" components for future
    compatibility in case we decide to use an upfirdn implementation. The
    current implementation is functionally equivalent to passing
    up=up/down and down=1.
    """
    from scipy.signal import get_window
    # check explicitly for backwards compatibility
    if not isinstance(axis, int):
        err = ("The axis parameter needs to be an integer (got %s). "
               "The axis parameter was missing from this function for a "
               "period of time, you might be intending to specify the "
               "subsequent window parameter." % repr(axis))
        raise TypeError(err)

    # make sure our arithmetic will work
    x = np.asanyarray(x)
    ratio = float(up) / down
    if axis < 0:
        axis = x.ndim + axis
    orig_last_axis = x.ndim - 1
    if axis != orig_last_axis:
        x = x.swapaxes(axis, orig_last_axis)
    orig_shape = x.shape
    x_len = orig_shape[-1]
    if x_len == 0:
        warn('x has zero length along last axis, returning a copy of x')
        return x.copy()
    bad_msg = 'npad must be "auto" or an integer'
    if isinstance(npad, string_types):
        if npad != 'auto':
            raise ValueError(bad_msg)
        # Figure out reasonable pad that gets us to a power of 2
        min_add = min(x_len // 8, 100) * 2
        npad = 2 ** int(np.ceil(np.log2(x_len + min_add))) - x_len
        npad, extra = divmod(npad, 2)
        npads = np.array([npad, npad + extra], int)
    else:
        if npad != int(npad):
            raise ValueError(bad_msg)
        npads = np.array([npad, npad], int)
    del npad

    # prep for resampling now
    x_flat = x.reshape((-1, x_len))
    orig_len = x_len + npads.sum()  # length after padding
    new_len = int(round(ratio * orig_len))  # length after resampling
    final_len = int(round(ratio * x_len))
    to_removes = [int(round(ratio * npads[0]))]
    to_removes.append(new_len - final_len - to_removes[0])
    to_removes = np.array(to_removes)
    # This should hold:
    # assert np.abs(to_removes[1] - to_removes[0]) <= int(np.ceil(ratio))

    # figure out windowing function
    if window is not None:
        if callable(window):
            W = window(fftfreq(orig_len))
        elif isinstance(window, np.ndarray) and \
                window.shape == (orig_len,):
            W = window
        else:
            W = ifftshift(get_window(window, orig_len))
    else:
        W = np.ones(orig_len)
    W *= (float(new_len) / float(orig_len))
    W = W.astype(np.complex128)

    # figure out if we should use CUDA
    n_jobs, cuda_dict, W = setup_cuda_fft_resample(n_jobs, W, new_len)

    # do the resampling using an adaptation of scipy's FFT-based resample()
    # use of the 'flat' window is recommended for minimal ringing
    if n_jobs == 1:
        y = np.zeros((len(x_flat), new_len - to_removes.sum()), dtype=x.dtype)
        for xi, x_ in enumerate(x_flat):
            y[xi] = fft_resample(x_, W, new_len, npads, to_removes,
                                 cuda_dict)
    else:
        parallel, p_fun, _ = parallel_func(fft_resample, n_jobs)
        y = parallel(p_fun(x_, W, new_len, npads, to_removes, cuda_dict)
                     for x_ in x_flat)
        y = np.array(y)

    # Restore the original array shape (modified for resampling)
    y.shape = orig_shape[:-1] + (y.shape[1],)
    if axis != orig_last_axis:
        y = y.swapaxes(axis, orig_last_axis)

    return y


def _resample_stim_channels(stim_data, up, down):
    """Resample stim channels, carefully.

    Parameters
    ----------
    stim_data : array, shape (n_samples,) or (n_stim_channels, n_samples)
        Stim channels to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.

    Returns
    -------
    stim_resampled : array, shape (n_stim_channels, n_samples_resampled)
        The resampled stim channels.

    Note
    ----
    The approach taken here is equivalent to the approach in the C-code.
    See the decimate_stimch function in MNE/mne_browse_raw/save.c
    """
    stim_data = np.atleast_2d(stim_data)
    n_stim_channels, n_samples = stim_data.shape

    ratio = float(up) / down
    resampled_n_samples = int(round(n_samples * ratio))

    stim_resampled = np.zeros((n_stim_channels, resampled_n_samples))

    # Figure out which points in old data to subsample protect against
    # out-of-bounds, which can happen (having one sample more than
    # expected) due to padding
    sample_picks = np.minimum(
        (np.arange(resampled_n_samples) / ratio).astype(int),
        n_samples - 1
    )

    # Create windows starting from sample_picks[i], ending at sample_picks[i+1]
    windows = zip(sample_picks, np.r_[sample_picks[1:], n_samples])

    # Use the first non-zero value in each window
    for window_i, window in enumerate(windows):
        for stim_num, stim in enumerate(stim_data):
            nonzero = stim[window[0]:window[1]].nonzero()[0]
            if len(nonzero) > 0:
                val = stim[window[0] + nonzero[0]]
            else:
                val = stim[window[0]]
            stim_resampled[stim_num, window_i] = val

    return stim_resampled


def detrend(x, order=1, axis=-1):
    """Detrend the array x.

    Parameters
    ----------
    x : n-d array
        Signal to detrend.
    order : int
        Fit order. Currently must be '0' or '1'.
    axis : integer
        Axis of the array to operate on.

    Returns
    -------
    xf : array
        x detrended.

    Examples
    --------
    As in scipy.signal.detrend:
        >>> randgen = np.random.RandomState(9)
        >>> npoints = int(1e3)
        >>> noise = randgen.randn(npoints)
        >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
        >>> (detrend(x) - noise).max() < 0.01
        True
    """
    from scipy.signal import detrend
    if axis > len(x.shape):
        raise ValueError('x does not have %d axes' % axis)
    if order == 0:
        fit = 'constant'
    elif order == 1:
        fit = 'linear'
    else:
        raise ValueError('order must be 0 or 1')

    y = detrend(x, axis=axis, type=fit)

    return y


def _triage_filter_params(x, sfreq, l_freq, h_freq,
                          l_trans_bandwidth, h_trans_bandwidth,
                          filter_length, method, phase, fir_window,
                          bands='scalar', reverse=False):
    """Helper to validate and automate filter parameter selection."""
    if not isinstance(phase, string_types) or phase not in \
            ('linear', 'zero', 'zero-double', 'minimum', ''):
        raise ValueError('phase must be "linear", "zero", "zero-double", '
                         'or "minimum", got "%s"' % (phase,))
    if not isinstance(fir_window, string_types) or fir_window not in \
            ('hann', 'hamming', 'blackman', ''):
        raise ValueError('fir_window must be "hamming", "hann", or "blackman",'
                         'got "%s"' % (fir_window,))

    def float_array(c):
        return np.array(c, float).ravel()

    if bands == 'arr':
        cast = float_array
    else:
        cast = float
    x = np.asanyarray(x)
    len_x = x.shape[-1]
    sfreq = float(sfreq)
    if l_freq is not None:
        l_freq = cast(l_freq)
        if np.any(l_freq <= 0):
            raise ValueError('highpass frequency %s must be greater than zero'
                             % (l_freq,))
    if h_freq is not None:
        h_freq = cast(h_freq)
        if np.any(h_freq >= sfreq / 2.):
            raise ValueError('lowpass frequency %s must be less than Nyquist '
                             '(%s)' % (h_freq, sfreq / 2.))
    if method == 'iir':
        # Ignore these parameters, effectively
        l_stop, h_stop = l_freq, h_freq
    else:  # method == 'fir'
        l_stop = h_stop = None
        if l_freq is not None:  # high-pass component
            if isinstance(l_trans_bandwidth, string_types):
                if l_trans_bandwidth != 'auto':
                    raise ValueError('l_trans_bandwidth must be "auto" if '
                                     'string, got "%s"' % l_trans_bandwidth)
                l_trans_bandwidth = np.minimum(np.maximum(0.25 * l_freq, 2.),
                                               l_freq)
                logger.info('l_trans_bandwidth chosen to be %0.1f Hz'
                            % (l_trans_bandwidth,))
            l_trans_bandwidth = cast(l_trans_bandwidth)
            if np.any(l_trans_bandwidth <= 0):
                raise ValueError('l_trans_bandwidth must be positive, got %s'
                                 % (l_trans_bandwidth,))
            l_stop = l_freq - l_trans_bandwidth
            if reverse:  # band-stop style
                l_stop += l_trans_bandwidth
                l_freq += l_trans_bandwidth
            if np.any(l_stop < 0):
                raise ValueError('Filter specification invalid: Lower stop '
                                 'frequency negative (%0.1fHz). Increase pass '
                                 'frequency or reduce the transition '
                                 'bandwidth (l_trans_bandwidth)' % l_stop)
        if h_freq is not None:  # low-pass component
            if isinstance(h_trans_bandwidth, string_types):
                if h_trans_bandwidth != 'auto':
                    raise ValueError('h_trans_bandwidth must be "auto" if '
                                     'string, got "%s"' % h_trans_bandwidth)
                h_trans_bandwidth = np.minimum(np.maximum(0.25 * h_freq, 2.),
                                               sfreq / 2. - h_freq)
                logger.info('h_trans_bandwidth chosen to be %0.1f Hz'
                            % (h_trans_bandwidth))
            h_trans_bandwidth = cast(h_trans_bandwidth)
            if np.any(h_trans_bandwidth <= 0):
                raise ValueError('h_trans_bandwidth must be positive, got %s'
                                 % (h_trans_bandwidth,))
            h_stop = h_freq + h_trans_bandwidth
            if reverse:  # band-stop style
                h_stop -= h_trans_bandwidth
                h_freq -= h_trans_bandwidth
            if np.any(h_stop > sfreq / 2):
                raise ValueError('Effective band-stop frequency (%s) is too '
                                 'high (maximum based on Nyquist is %s)'
                                 % (h_stop, sfreq / 2.))
        if isinstance(filter_length, string_types):
            filter_length = filter_length.lower()
            if filter_length == 'auto':
                h_check = h_trans_bandwidth if h_freq is not None else np.inf
                l_check = l_trans_bandwidth if l_freq is not None else np.inf
                filter_length = max(int(round(
                    _length_factors[fir_window] * sfreq /
                    float(min(h_check, l_check)))), 1)
                logger.info('Filter length of %s samples (%0.3f sec) selected'
                            % (filter_length, filter_length / sfreq))
            else:
                err_msg = ('filter_length, if a string, must be a '
                           'human-readable time, e.g. "10s", or "auto", not '
                           '"%s"' % filter_length)
                if filter_length.lower().endswith('ms'):
                    mult_fact = 1e-3
                    filter_length = filter_length[:-2]
                elif filter_length[-1].lower() == 's':
                    mult_fact = 1
                    filter_length = filter_length[:-1]
                else:
                    raise ValueError(err_msg)
                # now get the number
                try:
                    filter_length = float(filter_length)
                except ValueError:
                    raise ValueError(err_msg)
                if phase == 'zero-double':  # old mode
                    filter_length = 2 ** int(np.ceil(np.log2(
                        filter_length * mult_fact * sfreq)))
                else:
                    filter_length = max(int(np.ceil(filter_length * mult_fact *
                                                    sfreq)), 1)
        elif not isinstance(filter_length, integer_types):
            raise ValueError('filter_length must be a str, int, or None, got '
                             '%s' % (type(filter_length),))
    if method != 'fir':
        filter_length = len_x
    if phase == 'zero' and method == 'fir':
        filter_length += (filter_length % 2 == 0)
    if filter_length <= 0:
        raise ValueError('filter_length must be positive, got %s'
                         % (filter_length,))
    if filter_length > len_x:
        warn('filter_length (%s) is longer than the signal (%s), '
             'distortion is likely. Reduce filter length or filter a '
             'longer signal.' % (filter_length, len_x))
    logger.debug('Using filter length: %s' % filter_length)
    return (x, sfreq, l_freq, h_freq, l_stop, h_stop, filter_length, phase,
            fir_window)


class FilterMixin(object):
    """Object for Epoch/Evoked filtering."""

    def savgol_filter(self, h_freq, copy=False):
        """Filter the data using Savitzky-Golay polynomial method.

        Parameters
        ----------
        h_freq : float
            Approximate high cut-off frequency in Hz. Note that this
            is not an exact cutoff, since Savitzky-Golay filtering [1]_ is
            done using polynomial fits instead of FIR/IIR filtering.
            This parameter is thus used to determine the length of the
            window over which a 5th-order polynomial smoothing is used.
        copy : bool
            If True, a copy of the object, filtered, is returned.
            If False (default), it operates on the object in place.

        Returns
        -------
        inst : instance of Epochs or Evoked
            The object with the filtering applied.

        See Also
        --------
        mne.io.Raw.filter

        Notes
        -----
        For Savitzky-Golay low-pass approximation, see:

            https://gist.github.com/Eric89GXL/bbac101d50176611136b


        .. versionadded:: 0.9.0

        Examples
        --------
        >>> import mne
        >>> from os import path as op
        >>> evoked_fname = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis-ave.fif')  # doctest:+SKIP
        >>> evoked = mne.read_evokeds(evoked_fname, baseline=(None, 0))[0]  # doctest:+SKIP
        >>> evoked.savgol_filter(10.)  # low-pass at around 10 Hz # doctest:+SKIP
        >>> evoked.plot()  # doctest:+SKIP

        References
        ----------
        .. [1] Savitzky, A., Golay, M.J.E. (1964). "Smoothing and
               Differentiation of Data by Simplified Least Squares
               Procedures". Analytical Chemistry 36 (8): 1627-39.
        """  # noqa: E501
        inst = self.copy() if copy else self
        if not inst.preload:
            raise RuntimeError('data must be preloaded to filter')
        data = inst._data

        h_freq = float(h_freq)
        if h_freq >= inst.info['sfreq'] / 2.:
            raise ValueError('h_freq must be less than half the sample rate')

        # savitzky-golay filtering
        if not check_version('scipy', '0.14'):
            raise RuntimeError('scipy >= 0.14 must be installed for savgol')
        from scipy.signal import savgol_filter
        window_length = (int(np.round(inst.info['sfreq'] /
                                      h_freq)) // 2) * 2 + 1
        data[...] = savgol_filter(data, axis=-1, polyorder=5,
                                  window_length=window_length)
        return inst


@verbose
def design_mne_c_filter(sfreq, l_freq=None, h_freq=40.,
                        l_trans_bandwidth=None, h_trans_bandwidth=5.,
                        verbose=None):
    """Create a FIR filter like that used by MNE-C.

    Parameters
    ----------
    sfreq : float
        The sample frequency.
    l_freq : float | None
        The low filter frequency in Hz, default None.
        Can be None to avoid high-passing.
    h_freq : float
        The high filter frequency in Hz, default 40.
        Can be None to avoid low-passing.
    l_trans_bandwidth : float | None
        Low transition bandwidthin Hz. Can be None (default) to use 3 samples.
    h_trans_bandwidth : float
        High transition bandwidth in Hz.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.

    Returns
    -------
    h : ndarray, shape (8193,)
        The linear-phase (symmetric) FIR filter coefficients.

    Notes
    -----
    This function is provided mostly for reference purposes.

    MNE-C uses a frequency-domain filter design technique by creating a
    linear-phase filter of length 8193. In the frequency domain, the
    4197 frequencies are directly constructed, with zeroes in the stop-band
    and ones in the pass-band, with squared cosine ramps in between.
    """
    n_freqs = (4096 + 2 * 2048) // 2 + 1
    freq_resp = np.ones(n_freqs)
    l_freq = 0 if l_freq is None else float(l_freq)
    if l_trans_bandwidth is None:
        l_width = 3
    else:
        l_width = (int(((n_freqs - 1) * l_trans_bandwidth) /
                       (0.5 * sfreq)) + 1) // 2
    l_start = int(((n_freqs - 1) * l_freq) / (0.5 * sfreq))
    h_freq = sfreq / 2. if h_freq is None else float(h_freq)
    h_width = (int(((n_freqs - 1) * h_trans_bandwidth) /
                   (0.5 * sfreq)) + 1) // 2
    h_start = int(((n_freqs - 1) * h_freq) / (0.5 * sfreq))
    logger.info('filter : %7.3f ... %6.1f Hz   bins : %d ... %d of %d '
                'hpw : %d lpw : %d' % (l_freq, h_freq, l_start, h_start,
                                       n_freqs, l_width, h_width))
    if l_freq > 0:
        start = l_start - l_width + 1
        stop = start + 2 * l_width - 1
        if start < 0 or stop >= n_freqs:
            raise RuntimeError('l_freq too low or l_trans_bandwidth too large')
        freq_resp[:start] = 0.
        k = np.arange(-l_width + 1, l_width) / float(l_width) + 3.
        freq_resp[start:stop] = np.cos(np.pi / 4. * k) ** 2

    if h_freq < sfreq / 2.:
        start = h_start - h_width + 1
        stop = start + 2 * h_width - 1
        if start < 0 or stop >= n_freqs:
            raise RuntimeError('h_freq too high or h_trans_bandwidth too '
                               'large')
        k = np.arange(-h_width + 1, h_width) / float(h_width) + 1.
        freq_resp[start:stop] *= np.cos(np.pi / 4. * k) ** 2
        freq_resp[stop:] = 0.0
    # Get the time-domain version of this signal
    h = ifft(np.concatenate((freq_resp, freq_resp[::-1][:-1]))).real
    h = np.roll(h, n_freqs - 1)  # center the impulse like a linear-phase filt
    return h


###############################################################################
# Class for interpolation between adjacent points

class _Interp2(object):
    r"""Interpolate between two points.

    Parameters
    ----------
    interp : str
        Can be 'zero', 'linear', 'hann', or 'cos2'.

    Notes
    -----
    This will process data using overlapping windows of potentially
    different sizes to achieve a constant output value using different
    2-point interpolation schemes. For example, for linear interpolation,
    and window sizes of 6 and 17, this would look like::

        1 _     _
          |\   / '-.           .-'
          | \ /     '-.     .-'
          |  x         |-.-|
          | / \     .-'     '-.
          |/   \_.-'           '-.
        0 +----|----|----|----|---
          0    5   10   15   20   25

    """

    @verbose
    def __init__(self, interp='hann'):
        # set up interpolation
        self._last = dict()
        self._current = dict()
        self._count = dict()
        self._n_samp = None
        self.interp = interp

    def __setitem__(self, key, value):
        """Update an item."""
        if value is None:
            assert key not in self._current
            return
        if key in self._current:
            self._last[key] = self._current[key].copy()
        self._current[key] = value.copy()
        self._count[key] = self._count.get(key, 0) + 1

    @property
    def n_samp(self):
        return self._n_samp

    @n_samp.setter
    def n_samp(self, n_samp):
        # all up to date
        assert len(set(self._count.values())) == 1
        self._n_samp = n_samp
        self.interp = self.interp
        self._chunks = np.concatenate((np.arange(0, n_samp, 10000), [n_samp]))

    @property
    def interp(self):
        return self._interp

    @interp.setter
    def interp(self, interp):
        known_types = ('cos2', 'linear', 'zero', 'hann')
        if interp not in known_types:
            raise ValueError('interp must be one of %s, got "%s"'
                             % (known_types, interp))
        self._interp = interp
        if self.n_samp is not None:
            if self._interp == 'zero':
                self._interpolators = None
            else:
                if self._interp == 'linear':
                    interp = np.linspace(1, 0, self.n_samp, endpoint=False)
                elif self._interp == 'cos2':
                    interp = np.cos(0.5 * np.pi * np.arange(self.n_samp)) ** 2
                else:  # interp == 'hann'
                    interp = np.hanning(self.n_samp * 2 + 1)[self.n_samp:-1]
                self._interpolators = np.array([interp, 1 - interp])

    def interpolate(self, key, data, out, picks=None, data_idx=None):
        """Interpolate."""
        picks = slice(None) if picks is None else picks
        # Process data in large chunks to save on memory
        for start, stop in zip(self._chunks[:-1], self._chunks[1:]):
            time_sl = slice(start, stop)
            if data_idx is not None:
                # This is useful e.g. when circularly accessing the same data.
                # This prevents STC blowups in raw data simulation.
                data_sl = data[:, data_idx[time_sl]]
            else:
                data_sl = data[:, time_sl]
            this_data = np.dot(self._last[key], data_sl)
            if self._interpolators is not None:
                this_data *= self._interpolators[0][time_sl]
            out[picks, time_sl] += this_data
            if self._interpolators is not None:
                this_data = np.dot(self._current[key], data_sl)
                this_data *= self._interpolators[1][time_sl]
                out[picks, time_sl] += this_data
            del this_data
