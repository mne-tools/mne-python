"""IIR and FIR filtering and resampling functions."""

from collections import Counter
from copy import deepcopy
from functools import partial

import numpy as np

from .annotations import _annotations_starts_stops
from .io.pick import _picks_to_idx
from .cuda import (_setup_cuda_fft_multiply_repeated, _fft_multiply_repeated,
                   _setup_cuda_fft_resample, _fft_resample, _smart_pad)
from .parallel import parallel_func
from .utils import (logger, verbose, sum_squared, warn, _pl,
                    _check_preload, _validate_type, _check_option, _ensure_int)
from ._ola import _COLA

# These values from Ifeachor and Jervis.
_length_factors = dict(hann=3.1, hamming=3.3, blackman=5.0)


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
    """Find the next fast size of input data to `fft`, for zero-padding, etc.

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

            p2 = 2 ** int(quotient - 1).bit_length()

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
                        n_jobs=None, copy=True, pad='reflect_limited'):
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
    picks : list | None
        See calling functions.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if ``cupy``
        is installed properly.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    pad : str
        Padding type for ``_smart_pad``.

    Returns
    -------
    x : array, shape (n_signals, n_times)
        x filtered.
    """
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
            cost = (np.ceil(n_x / (N - len(h) + 1).astype(np.float64)) *
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

    # Figure out if we should use CUDA
    n_jobs, cuda_dict = _setup_cuda_fft_multiply_repeated(n_jobs, h, n_fft)

    # Process each row separately
    picks = _picks_to_idx(len(x), picks)
    parallel, p_fun, _ = parallel_func(_1d_overlap_filter, n_jobs)
    if n_jobs == 1:
        for p in picks:
            x[p] = _1d_overlap_filter(x[p], len(h), n_edge, phase,
                                      cuda_dict, pad, n_fft)
    else:
        data_new = parallel(p_fun(x[p], len(h), n_edge, phase,
                                  cuda_dict, pad, n_fft) for p in picks)
        for pp, p in enumerate(picks):
            x[p] = data_new[pp]

    x.shape = orig_shape
    return x


def _1d_overlap_filter(x, n_h, n_edge, phase, cuda_dict, pad, n_fft):
    """Do one-dimensional overlap-add FFT FIR filtering."""
    # pad to reduce ringing
    x_ext = _smart_pad(x, (n_edge, n_edge), pad)
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

        prod = _fft_multiply_repeated(seg, cuda_dict)

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
    x = _check_filterable(x)
    if copy is True:
        x = x.copy()
    orig_shape = x.shape
    x = np.atleast_2d(x)
    picks = _picks_to_idx(x.shape[-2], picks)
    x.shape = (np.prod(x.shape[:-1]), x.shape[-1])
    if len(orig_shape) == 3:
        n_epochs, n_channels, n_times = orig_shape
        offset = np.repeat(np.arange(0, n_channels * n_epochs, n_channels),
                           len(picks))
        picks = np.tile(picks, n_epochs) + offset
    elif len(orig_shape) > 3:
        raise ValueError('picks argument is not supported for data with more'
                         ' than three dimensions')
    assert all(0 <= pick < x.shape[0] for pick in picks)  # guaranteed by above

    return x, orig_shape, picks


def _firwin_design(N, freq, gain, window, sfreq):
    """Construct a FIR filter using firwin."""
    from scipy.signal import firwin
    assert freq[0] == 0
    assert len(freq) > 1
    assert len(freq) == len(gain)
    assert N % 2 == 1
    h = np.zeros(N)
    prev_freq = freq[-1]
    prev_gain = gain[-1]
    if gain[-1] == 1:
        h[N // 2] = 1  # start with "all up"
    assert prev_gain in (0, 1)
    for this_freq, this_gain in zip(freq[::-1][1:], gain[::-1][1:]):
        assert this_gain in (0, 1)
        if this_gain != prev_gain:
            # Get the correct N to satistify the requested transition bandwidth
            transition = (prev_freq - this_freq) / 2.
            this_N = int(round(_length_factors[window] / transition))
            this_N += (1 - this_N % 2)  # make it odd
            if this_N > N:
                raise ValueError('The requested filter length %s is too short '
                                 'for the requested %0.2f Hz transition band, '
                                 'which requires %s samples'
                                 % (N, transition * sfreq / 2., this_N))
            # Construct a lowpass
            this_h = firwin(this_N, (prev_freq + this_freq) / 2.,
                            window=window, pass_zero=True, fs=freq[-1] * 2)
            assert this_h.shape == (this_N,)
            offset = (N - this_N) // 2
            if this_gain == 0:
                h[offset:N - offset] -= this_h
            else:
                h[offset:N - offset] += this_h
        prev_gain = this_gain
        prev_freq = this_freq
    return h


def _construct_fir_filter(sfreq, freq, gain, filter_length, phase, fir_window,
                          fir_design):
    """Filter signal using gain control points in the frequency domain.

    The filter impulse response is constructed from a Hann window (window
    used in "firwin2" function) to avoid ripples in the frequency response
    (windowing is a smoothing in frequency domain).

    If x is multi-dimensional, this operates along the last dimension.

    Parameters
    ----------
    sfreq : float
        Sampling rate in Hz.
    freq : 1d array
        Frequency sampling points in Hz.
    gain : 1d array
        Filter gain at frequency sampling points.
        Must be all 0 and 1 for fir_design=="firwin".
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
    fir_design : str
        Can be "firwin2" or "firwin".

    Returns
    -------
    h : array
        Filter coefficients.
    """
    assert freq[0] == 0
    if fir_design == 'firwin2':
        from scipy.signal import firwin2 as fir_design
    else:
        assert fir_design == 'firwin'
        fir_design = partial(_firwin_design, sfreq=sfreq)
    from scipy.signal import minimum_phase

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
        h = fir_design(N * 2 - 1, freq, gain, window=fir_window)
        h = minimum_phase(h)
    else:
        h = fir_design(N, freq, gain, window=fir_window)
    assert h.size == N
    att_db, att_freq = _filter_attenuation(h, freq, gain)
    if phase == 'zero-double':
        att_db += 6
    if att_db < min_att_db:
        att_freq *= sfreq / 2.
        warn('Attenuation at stop frequency %0.2f Hz is only %0.2f dB. '
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
    """Call filtfilt."""
    # set up array for filtering, reshape to 2D, operate on last axis
    from scipy.signal import filtfilt, sosfiltfilt
    padlen = min(iir_params['padlen'], x.shape[-1] - 1)
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    if 'sos' in iir_params:
        fun = partial(sosfiltfilt, sos=iir_params['sos'], padlen=padlen,
                      axis=-1)
        _check_coefficients(iir_params['sos'])
    else:
        fun = partial(filtfilt, b=iir_params['b'], a=iir_params['a'],
                      padlen=padlen, axis=-1)
        _check_coefficients((iir_params['b'], iir_params['a']))
    parallel, p_fun, n_jobs = parallel_func(fun, n_jobs)
    if n_jobs == 1:
        for p in picks:
            x[p] = fun(x=x[p])
    else:
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
        This will be changed to a multiple of 1000.

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


_ftype_dict = {
    'butter': 'Butterworth',
    'cheby1': 'Chebyshev I',
    'cheby2': 'Chebyshev II',
    'ellip': 'Cauer/elliptic',
    'bessel': 'Bessel/Thomson',
}


@verbose
def construct_iir_filter(iir_params, f_pass=None, f_stop=None, sfreq=None,
                         btype=None, return_copy=True, verbose=None):
    """Use IIR parameters to get filtering coefficients.

    This function works like a wrapper for iirdesign and iirfilter in
    scipy.signal to make filter coefficients for IIR filtering. It also
    estimates the number of padding samples based on the filter ringing.
    It creates a new iir_params dict (or updates the one passed to the
    function) with the filter coefficients ('b' and 'a') and an estimate
    of the padding necessary ('padlen') so IIR filtering can be performed.

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
              You should also supply ``iir_params['rs']`` and
              ``iir_params['rp']`` if using elliptic or Chebychev filters.
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
    %(verbose)s

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

    .. versionchanged:: 0.14
       Second-order sections are used in filter design by default (replacing
       ``output='ba'`` by ``output='sos'``) to help ensure filter stability
       and reduce numerical error.

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

    >>> iir_params = dict(b=np.ones((10)), a=[1, 0], padlen=0)  # doctest:+SKIP
    >>> iir_params = construct_iir_filter(iir_params, return_copy=False)  # doctest:+SKIP
    >>> print((iir_params['b'], iir_params['a'], iir_params['padlen']))  # doctest:+SKIP
    (array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), [1, 0], 0)

    For more information, see the tutorials
    :ref:`disc-filtering` and :ref:`tut-filter-resample`.
    """  # noqa: E501
    from scipy.signal import iirfilter, iirdesign, freqz, sosfreqz
    known_filters = ('bessel', 'butter', 'butterworth', 'cauer', 'cheby1',
                     'cheby2', 'chebyshev1', 'chebyshev2', 'chebyshevi',
                     'chebyshevii', 'ellip', 'elliptic')
    if not isinstance(iir_params, dict):
        raise TypeError('iir_params must be a dict, got %s' % type(iir_params))
    # if the filter has been designed, we're good to go
    Wp = None
    if 'sos' in iir_params:
        system = iir_params['sos']
        output = 'sos'
    elif 'a' in iir_params and 'b' in iir_params:
        system = (iir_params['b'], iir_params['a'])
        output = 'ba'
    else:
        output = iir_params.get('output', 'sos')
        _check_option('output', output, ('ba', 'sos'))
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
        f_pass = np.atleast_1d(f_pass)
        if f_pass.ndim > 1:
            raise ValueError('frequencies must be 1D, got %dD' % f_pass.ndim)
        edge_freqs = ', '.join('%0.2f' % (f,) for f in f_pass)
        Wp = f_pass / (float(sfreq) / 2)
        # IT will de designed
        ftype_nice = _ftype_dict.get(ftype, ftype)
        logger.info('')
        logger.info('IIR filter parameters')
        logger.info('---------------------')
        logger.info('%s %s zero-phase (two-pass forward and reverse) '
                    'non-causal filter:' % (ftype_nice, btype))
        # SciPy designs for -3dB but we do forward-backward, so this is -6dB
        if 'order' in iir_params:
            kwargs = dict(N=iir_params['order'], Wn=Wp, btype=btype,
                          ftype=ftype, output=output)
            for key in ('rp', 'rs'):
                if key in iir_params:
                    kwargs[key] = iir_params[key]
            system = iirfilter(**kwargs)
            logger.info('- Filter order %d (effective, after forward-backward)'
                        % (2 * iir_params['order'] * len(Wp),))
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

    # get the gains at the cutoff frequencies
    if Wp is not None:
        if output == 'sos':
            cutoffs = sosfreqz(system, worN=Wp * np.pi)[1]
        else:
            cutoffs = freqz(system[0], system[1], worN=Wp * np.pi)[1]
        # 2 * 20 here because we do forward-backward filtering
        cutoffs = 40 * np.log10(np.abs(cutoffs))
        cutoffs = ', '.join(['%0.2f' % (c,) for c in cutoffs])
        logger.info('- Cutoff%s at %s Hz: %s dB'
                    % (_pl(f_pass), edge_freqs, cutoffs))
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
    logger.info('')
    return iir_params


def _check_method(method, iir_params, extra_types=()):
    """Parse method arguments."""
    allowed_types = ['iir', 'fir', 'fft'] + list(extra_types)
    _validate_type(method, 'str', 'method')
    _check_option('method', method, allowed_types)
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
                l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                n_jobs=None, method='fir', iir_params=None, copy=True,
                phase='zero', fir_window='hamming', fir_design='firwin',
                pad='reflect_limited', *, verbose=None):
    """Filter a subset of channels.

    Parameters
    ----------
    data : ndarray, shape (..., n_times)
        The data to filter.
    sfreq : float
        The sample frequency in Hz.
    %(l_freq)s
    %(h_freq)s
    %(picks_nostr)s
        Currently this is only supported for 2D (n_channels, n_times) and
        3D (n_epochs, n_channels, n_times) arrays.
    %(filter_length)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(n_jobs_fir)s
    %(method_fir)s
    %(iir_params)s
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    %(phase)s
    %(fir_window)s
    %(fir_design)s
    %(pad_fir)s
        The default is ``'reflect_limited'``.

        .. versionadded:: 0.15
    %(verbose)s

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
              be temporarily stored in memory.

    For more information, see the tutorials
    :ref:`disc-filtering` and :ref:`tut-filter-resample` and
    :func:`mne.filter.create_filter`.
    """
    data = _check_filterable(data)
    iir_params, method = _check_method(method, iir_params)
    filt = create_filter(
        data, sfreq, l_freq, h_freq, filter_length, l_trans_bandwidth,
        h_trans_bandwidth, method, iir_params, phase, fir_window, fir_design)
    if method in ('fir', 'fft'):
        data = _overlap_add_filter(data, filt, None, phase, picks, n_jobs,
                                   copy, pad)
    else:
        data = _filtfilt(data, filt, picks, n_jobs, copy)
    return data


@verbose
def create_filter(data, sfreq, l_freq, h_freq, filter_length='auto',
                  l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                  method='fir', iir_params=None, phase='zero',
                  fir_window='hamming', fir_design='firwin', verbose=None):
    r"""Create a FIR or IIR filter.

    ``l_freq`` and ``h_freq`` are the frequencies below which and above
    which, respectively, to filter out of the data. Thus the uses are:

        * ``l_freq < h_freq``: band-pass filter
        * ``l_freq > h_freq``: band-stop filter
        * ``l_freq is not None and h_freq is None``: high-pass filter
        * ``l_freq is None and h_freq is not None``: low-pass filter

    Parameters
    ----------
    data : ndarray, shape (..., n_times) | None
        The data that will be filtered. This is used for sanity checking
        only. If None, no sanity checking related to the length of the signal
        relative to the filter order will be performed.
    sfreq : float
        The sample frequency in Hz.
    %(l_freq)s
    %(h_freq)s
    %(filter_length)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(method_fir)s
    %(iir_params)s
    %(phase)s
    %(fir_window)s
    %(fir_design)s
    %(verbose)s

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
    .. note:: For FIR filters, the *cutoff frequency*, i.e. the -6 dB point,
              is in the middle of the transition band (when using phase='zero'
              and fir_design='firwin'). For IIR filters, the cutoff frequency
              is given by ``l_freq`` or ``h_freq`` directly, and
              ``l_trans_bandwidth`` and ``h_trans_bandwidth`` are ignored.

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
    # If no data specified, sanity checking will be skipped
    if data is None:
        logger.info('No data specified. Sanity checks related to the length of'
                    ' the signal relative to the filter order will be'
                    ' skipped.')
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
        data, sfreq, _, _, _, _, filter_length, phase, fir_window, \
            fir_design = _triage_filter_params(
                data, sfreq, None, None, None, None,
                filter_length, method, phase, fir_window, fir_design)
        if method == 'iir':
            out = dict() if iir_params is None else deepcopy(iir_params)
            out.update(b=np.array([1.]), a=np.array([1.]))
        else:
            freq = [0, sfreq / 2.]
            gain = [1., 1.]
    if l_freq is None and h_freq is not None:
        logger.info('Setting up low-pass filter at %0.2g Hz' % (h_freq,))
        data, sfreq, _, f_p, _, f_s, filter_length, phase, fir_window, \
            fir_design = _triage_filter_params(
                data, sfreq, None, h_freq, None, h_trans_bandwidth,
                filter_length, method, phase, fir_window, fir_design)
        if method == 'iir':
            out = construct_iir_filter(iir_params, f_p, f_s, sfreq, 'lowpass')
        else:  # 'fir'
            freq = [0, f_p, f_s]
            gain = [1, 1, 0]
            if f_s != sfreq / 2.:
                freq += [sfreq / 2.]
                gain += [0]
    elif l_freq is not None and h_freq is None:
        logger.info('Setting up high-pass filter at %0.2g Hz' % (l_freq,))
        data, sfreq, pass_, _, stop, _, filter_length, phase, fir_window, \
            fir_design = _triage_filter_params(
                data, sfreq, l_freq, None, l_trans_bandwidth, None,
                filter_length, method, phase, fir_window, fir_design)
        if method == 'iir':
            out = construct_iir_filter(iir_params, pass_, stop, sfreq,
                                       'highpass')
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
                fir_window, fir_design = _triage_filter_params(
                    data, sfreq, l_freq, h_freq, l_trans_bandwidth,
                    h_trans_bandwidth, filter_length, method, phase,
                    fir_window, fir_design)
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
                fir_window, fir_design = _triage_filter_params(
                    data, sfreq, h_freq, l_freq, h_trans_bandwidth,
                    l_trans_bandwidth, filter_length, method, phase,
                    fir_window, fir_design, bands='arr', reverse=True)
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
                                    fir_window, fir_design)
    return out


@verbose
def notch_filter(x, Fs, freqs, filter_length='auto', notch_widths=None,
                 trans_bandwidth=1, method='fir', iir_params=None,
                 mt_bandwidth=None, p_value=0.05, picks=None, n_jobs=None,
                 copy=True, phase='zero', fir_window='hamming',
                 fir_design='firwin', pad='reflect_limited', *,
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
    %(filter_length_notch)s
    notch_widths : float | array of float | None
        Width of the stop band (centred at each freq in freqs) in Hz.
        If None, freqs / 200 is used.
    trans_bandwidth : float
        Width of the transition band in Hz.
        Only used for ``method='fir'``.
    %(method_fir)s
        'spectrum_fit' will use multi-taper estimation of sinusoidal
        components. If freqs=None and method='spectrum_fit', significant
        sinusoidal components are detected using an F test, and noted by
        logging.
    %(iir_params)s
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'spectrum_fit' mode.
    p_value : float
        P-value to use in F-test thresholding to determine significant
        sinusoidal components to remove when method='spectrum_fit' and
        freqs=None. Note that this will be Bonferroni corrected for the
        number of frequencies, so large p-values may be justified.
    %(picks_nostr)s
        Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    %(n_jobs_fir)s
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    %(phase)s
    %(fir_window)s
    %(fir_design)s
    %(pad_fir)s
        The default is ``'reflect_limited'``.
    %(verbose)s

    Returns
    -------
    xf : array
        The x array filtered.

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
    x = _check_filterable(x, 'notch filtered', 'notch_filter')
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
                         n_jobs, method, iir_params, copy, phase, fir_window,
                         fir_design, pad=pad)
    elif method == 'spectrum_fit':
        xf = _mt_spectrum_proc(x, Fs, freqs, notch_widths, mt_bandwidth,
                               p_value, picks, n_jobs, copy, filter_length)

    return xf


def _get_window_thresh(n_times, sfreq, mt_bandwidth, p_value):
    # max taper size chosen because it has an max error < 1e-3:
    # >>> np.max(np.diff(dpss_windows(953, 4, 100)[0]))
    # 0.00099972447657578449
    # so we use 1000 because it's the first "nice" number bigger than 953.
    # but if we have a new enough scipy,
    # it's only ~0.175 sec for 8 tapers even with 100000 samples
    from scipy import stats
    from .time_frequency.multitaper import _compute_mt_params
    dpss_n_times_max = 100000

    # figure out what tapers to use
    window_fun, _, _ = _compute_mt_params(
        n_times, sfreq, mt_bandwidth, False, False,
        interp_from=min(n_times, dpss_n_times_max), verbose=False)

    # F-stat of 1-p point
    threshold = stats.f.ppf(1 - p_value / n_times, 2, 2 * len(window_fun) - 2)
    return window_fun, threshold


def _mt_spectrum_proc(x, sfreq, line_freqs, notch_widths, mt_bandwidth,
                      p_value, picks, n_jobs, copy, filter_length):
    """Call _mt_spectrum_remove."""
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    if isinstance(filter_length, str) and filter_length == 'auto':
        filter_length = '10s'
    if filter_length is None:
        filter_length = x.shape[-1]
    filter_length = min(_to_samples(filter_length, sfreq, '', ''), x.shape[-1])
    get_wt = partial(
        _get_window_thresh, sfreq=sfreq, mt_bandwidth=mt_bandwidth,
        p_value=p_value)
    window_fun, threshold = get_wt(filter_length)
    parallel, p_fun, n_jobs = parallel_func(_mt_spectrum_remove_win, n_jobs)
    if n_jobs == 1:
        freq_list = list()
        for ii, x_ in enumerate(x):
            if ii in picks:
                x[ii], f = _mt_spectrum_remove_win(
                    x_, sfreq, line_freqs, notch_widths, window_fun, threshold,
                    get_wt)
                freq_list.append(f)
    else:
        data_new = parallel(p_fun(x_, sfreq, line_freqs, notch_widths,
                                  window_fun, threshold, get_wt)
                            for xi, x_ in enumerate(x)
                            if xi in picks)
        freq_list = [d[1] for d in data_new]
        data_new = np.array([d[0] for d in data_new])
        x[picks, :] = data_new

    # report found frequencies, but do some sanitizing first by binning into
    # 1 Hz bins
    counts = Counter(sum((np.unique(np.round(ff)).tolist()
                          for f in freq_list for ff in f), list()))
    kind = 'Detected' if line_freqs is None else 'Removed'
    found_freqs = '\n'.join(f'    {freq:6.2f} : '
                            f'{counts[freq]:4d} window{_pl(counts[freq])}'
                            for freq in sorted(counts)) or '    None'
    logger.info(f'{kind} notch frequencies (Hz):\n{found_freqs}')

    x.shape = orig_shape
    return x


def _mt_spectrum_remove_win(x, sfreq, line_freqs, notch_widths,
                            window_fun, threshold, get_thresh):
    n_times = x.shape[-1]
    n_samples = window_fun.shape[1]
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x)
    rm_freqs = list()
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = _mt_spectrum_remove(
            x_, sfreq, line_freqs, notch_widths, window_fun, threshold,
            get_thresh)
        rm_freqs.append(out[1])
        return (out[0],)  # must return a tuple

    # Define how to store a chunk of fully processed data (it's trivial)
    def store(x_):
        stop = idx[0] + x_.shape[-1]
        x_out[..., idx[0]:stop] += x_
        idx[0] = stop

    _COLA(process, store, n_times, n_samples, n_overlap, sfreq,
          verbose=False).feed(x)
    assert idx[0] == n_times
    return x_out, rm_freqs


def _mt_spectrum_remove(x, sfreq, line_freqs, notch_widths,
                        window_fun, threshold, get_thresh):
    """Use MT-spectrum to remove line frequencies.

    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """
    from .time_frequency.multitaper import _mt_spectra
    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
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
        indices_2 = [np.logical_and(freqs > lf - nw / 2., freqs < lf + nw / 2.)
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
        datafit = np.sum(fits, axis=0)

    return x - datafit, rm_freqs


def _check_filterable(x, kind='filtered', alternative='filter'):
    # Let's be fairly strict about this -- users can easily coerce to ndarray
    # at their end, and we already should do it internally any time we are
    # using these low-level functions. At the same time, let's
    # help people who might accidentally use low-level functions that they
    # shouldn't use by pushing them in the right direction
    from .io.base import BaseRaw
    from .epochs import BaseEpochs
    from .evoked import Evoked
    if isinstance(x, (BaseRaw, BaseEpochs, Evoked)):
        try:
            name = x.__class__.__name__
        except Exception:
            pass
        else:
            raise TypeError(
                'This low-level function only operates on np.ndarray '
                f'instances. To get a {kind} {name} instance, use a method '
                f'like `inst_new = inst.copy().{alternative}(...)` '
                'instead.')
    _validate_type(x, (np.ndarray, list, tuple), f'Data to be {kind}')
    x = np.asanyarray(x)
    if x.dtype != np.float64:
        raise ValueError('Data to be %s must be real floating, got %s'
                         % (kind, x.dtype,))
    return x


def _resamp_ratio_len(up, down, n):
    ratio = float(up) / down
    return ratio, max(int(round(ratio * n)), 1)


@verbose
def resample(x, up=1., down=1., npad=100, axis=-1, window='boxcar',
             n_jobs=None, pad='reflect_limited', *, verbose=None):
    """Resample an array.

    Operates along the last dimension of the array.

    Parameters
    ----------
    x : ndarray
        Signal to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.
    %(npad)s
    axis : int
        Axis along which to resample (default is the last axis).
    %(window_resample)s
    %(n_jobs_cuda)s
    %(pad)s
        The default is ``'reflect_limited'``.

        .. versionadded:: 0.15
    %(verbose)s

    Returns
    -------
    y : array
        The x array resampled.

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
    from scipy.fft import ifftshift, fftfreq
    # check explicitly for backwards compatibility
    if not isinstance(axis, int):
        err = ("The axis parameter needs to be an integer (got %s). "
               "The axis parameter was missing from this function for a "
               "period of time, you might be intending to specify the "
               "subsequent window parameter." % repr(axis))
        raise TypeError(err)

    # make sure our arithmetic will work
    x = _check_filterable(x, 'resampled', 'resample')
    ratio, final_len = _resamp_ratio_len(up, down, x.shape[axis])
    del up, down
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
    if isinstance(npad, str):
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
    new_len = max(int(round(ratio * orig_len)), 1)  # length after resampling
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

    # figure out if we should use CUDA
    n_jobs, cuda_dict = _setup_cuda_fft_resample(n_jobs, W, new_len)

    # do the resampling using an adaptation of scipy's FFT-based resample()
    # use of the 'flat' window is recommended for minimal ringing
    parallel, p_fun, n_jobs = parallel_func(_fft_resample, n_jobs)
    if n_jobs == 1:
        y = np.zeros((len(x_flat), new_len - to_removes.sum()), dtype=x.dtype)
        for xi, x_ in enumerate(x_flat):
            y[xi] = _fft_resample(x_, new_len, npads, to_removes,
                                  cuda_dict, pad)
    else:
        y = parallel(p_fun(x_, new_len, npads, to_removes, cuda_dict, pad)
                     for x_ in x_flat)
        y = np.array(y)

    # Restore the original array shape (modified for resampling)
    y.shape = orig_shape[:-1] + (y.shape[1],)
    if axis != orig_last_axis:
        y = y.swapaxes(axis, orig_last_axis)
    assert y.shape[axis] == final_len

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
    axis : int
        Axis of the array to operate on.

    Returns
    -------
    y : array
        The x array detrended.

    Examples
    --------
    As in :func:`scipy.signal.detrend`::

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


# Taken from Ifeachor and Jervis p. 356.
# Note that here the passband ripple and stopband attenuation are
# rendundant. The scalar passband ripple δp is expressed in dB as
# 20 * log10(1+δp), but the scalar stopband ripple δs is expressed in dB as
# -20 * log10(δs). So if we know that our stopband attenuation is 53 dB
# (Hamming) then δs = 10 ** (53 / -20.), which means that the passband
# deviation should be 20 * np.log10(1 + 10 ** (53 / -20.)) == 0.0194.
_fir_window_dict = {
    'hann': dict(name='Hann', ripple=0.0546, attenuation=44),
    'hamming': dict(name='Hamming', ripple=0.0194, attenuation=53),
    'blackman': dict(name='Blackman', ripple=0.0017, attenuation=74),
}
_known_fir_windows = tuple(sorted(_fir_window_dict.keys()))
_known_phases = ('linear', 'zero', 'zero-double', 'minimum')
_known_fir_designs = ('firwin', 'firwin2')
_fir_design_dict = {
    'firwin': 'Windowed time-domain',
    'firwin2': 'Windowed frequency-domain',
}


def _to_samples(filter_length, sfreq, phase, fir_design):
    _validate_type(filter_length, (str, 'int-like'), 'filter_length')
    if isinstance(filter_length, str):
        filter_length = filter_length.lower()
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
        filter_length = max(int(np.ceil(filter_length * mult_fact *
                                        sfreq)), 1)
        if fir_design == 'firwin':
            filter_length += (filter_length - 1) % 2
    filter_length = _ensure_int(filter_length, 'filter_length')
    return filter_length


def _triage_filter_params(x, sfreq, l_freq, h_freq,
                          l_trans_bandwidth, h_trans_bandwidth,
                          filter_length, method, phase, fir_window,
                          fir_design, bands='scalar', reverse=False):
    """Validate and automate filter parameter selection."""
    _validate_type(phase, 'str', 'phase')
    _check_option('phase', phase, _known_phases)
    _validate_type(fir_window, 'str', 'fir_window')
    _check_option('fir_window', fir_window, _known_fir_windows)
    _validate_type(fir_design, 'str', 'fir_design')
    _check_option('fir_design', fir_design, _known_fir_designs)

    # Helpers for reporting
    report_phase = 'non-linear phase' if phase == 'minimum' else 'zero-phase'
    causality = 'causal' if phase == 'minimum' else 'non-causal'
    if phase == 'zero-double':
        report_pass = 'two-pass forward and reverse'
    else:
        report_pass = 'one-pass'
    if l_freq is not None:
        if h_freq is not None:
            kind = 'bandstop' if reverse else 'bandpass'
        else:
            kind = 'highpass'
            assert not reverse
    elif h_freq is not None:
        kind = 'lowpass'
        assert not reverse
    else:
        kind = 'allpass'

    def float_array(c):
        return np.array(c, float).ravel()

    if bands == 'arr':
        cast = float_array
    else:
        cast = float
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

    dB_cutoff = False  # meaning, don't try to compute or report
    if bands == 'scalar' or (len(h_freq) == 1 and len(l_freq) == 1):
        if phase == 'zero':
            dB_cutoff = '-6 dB'
        elif phase == 'zero-double':
            dB_cutoff = '-12 dB'

    # we go to the next power of two when in FIR and zero-double mode
    if method == 'iir':
        # Ignore these parameters, effectively
        l_stop, h_stop = l_freq, h_freq
    else:  # method == 'fir'
        l_stop = h_stop = None
        logger.info('')
        logger.info('FIR filter parameters')
        logger.info('---------------------')
        logger.info('Designing a %s, %s, %s %s filter:'
                    % (report_pass, report_phase, causality, kind))
        logger.info('- %s design (%s) method'
                    % (_fir_design_dict[fir_design], fir_design))
        this_dict = _fir_window_dict[fir_window]
        if fir_design == 'firwin':
            logger.info('- {name:s} window with {ripple:0.4f} passband ripple '
                        'and {attenuation:d} dB stopband attenuation'
                        .format(**this_dict))
        else:
            logger.info('- {name:s} window'.format(**this_dict))

        if l_freq is not None:  # high-pass component
            if isinstance(l_trans_bandwidth, str):
                if l_trans_bandwidth != 'auto':
                    raise ValueError('l_trans_bandwidth must be "auto" if '
                                     'string, got "%s"' % l_trans_bandwidth)
                l_trans_bandwidth = np.minimum(np.maximum(0.25 * l_freq, 2.),
                                               l_freq)
            msg = ('- Lower transition bandwidth: %0.2f Hz'
                   % (l_trans_bandwidth))
            if dB_cutoff:
                logger.info('- Lower passband edge: %0.2f' % (l_freq,))
                msg += ' (%s cutoff frequency: %0.2f Hz)' % (
                    dB_cutoff, l_freq - l_trans_bandwidth / 2.)
            logger.info(msg)
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
                                 'frequency negative (%0.2f Hz). Increase pass'
                                 ' frequency or reduce the transition '
                                 'bandwidth (l_trans_bandwidth)' % l_stop)
        if h_freq is not None:  # low-pass component
            if isinstance(h_trans_bandwidth, str):
                if h_trans_bandwidth != 'auto':
                    raise ValueError('h_trans_bandwidth must be "auto" if '
                                     'string, got "%s"' % h_trans_bandwidth)
                h_trans_bandwidth = np.minimum(np.maximum(0.25 * h_freq, 2.),
                                               sfreq / 2. - h_freq)
            msg = ('- Upper transition bandwidth: %0.2f Hz'
                   % (h_trans_bandwidth))
            if dB_cutoff:
                logger.info('- Upper passband edge: %0.2f Hz' % (h_freq,))
                msg += ' (%s cutoff frequency: %0.2f Hz)' % (
                    dB_cutoff, h_freq + h_trans_bandwidth / 2.)
            logger.info(msg)
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

        if isinstance(filter_length, str) and filter_length.lower() == 'auto':
            filter_length = filter_length.lower()
            h_check = h_trans_bandwidth if h_freq is not None else np.inf
            l_check = l_trans_bandwidth if l_freq is not None else np.inf
            mult_fact = 2. if fir_design == 'firwin2' else 1.
            filter_length = '%ss' % (_length_factors[fir_window] * mult_fact /
                                     float(min(h_check, l_check)),)
            next_pow_2 = False  # disable old behavior
        else:
            next_pow_2 = (
                isinstance(filter_length, str) and phase == 'zero-double')

        filter_length = _to_samples(filter_length, sfreq, phase, fir_design)

        # use correct type of filter (must be odd length for firwin and for
        # zero phase)
        if fir_design == 'firwin' or phase == 'zero':
            filter_length += (filter_length - 1) % 2

        logger.info('- Filter length: %s samples (%0.3f sec)'
                    % (filter_length, filter_length / sfreq))
        logger.info('')

        if filter_length <= 0:
            raise ValueError('filter_length must be positive, got %s'
                             % (filter_length,))

        if next_pow_2:
            filter_length = 2 ** int(np.ceil(np.log2(filter_length)))
            if fir_design == 'firwin':
                filter_length += (filter_length - 1) % 2

    # If we have data supplied, do a sanity check
    if x is not None:
        x = _check_filterable(x)
        len_x = x.shape[-1]
        if method != 'fir':
            filter_length = len_x
        if filter_length > len_x and not (l_freq is None and h_freq is None):
            warn('filter_length (%s) is longer than the signal (%s), '
                 'distortion is likely. Reduce filter length or filter a '
                 'longer signal.' % (filter_length, len_x))

    logger.debug('Using filter length: %s' % filter_length)
    return (x, sfreq, l_freq, h_freq, l_stop, h_stop, filter_length, phase,
            fir_window, fir_design)


class FilterMixin(object):
    """Object for Epoch/Evoked filtering."""

    @verbose
    def savgol_filter(self, h_freq, verbose=None):
        """Filter the data using Savitzky-Golay polynomial method.

        Parameters
        ----------
        h_freq : float
            Approximate high cut-off frequency in Hz. Note that this
            is not an exact cutoff, since Savitzky-Golay filtering
            :footcite:`SavitzkyGolay1964` is done using polynomial fits
            instead of FIR/IIR filtering. This parameter is thus used to
            determine the length of the window over which a 5th-order
            polynomial smoothing is used.
        %(verbose)s

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

            https://gist.github.com/larsoner/bbac101d50176611136b

        .. versionadded:: 0.9.0

        References
        ----------
        .. footbibliography::

        Examples
        --------
        >>> import mne
        >>> from os import path as op
        >>> evoked_fname = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis-ave.fif')  # doctest:+SKIP
        >>> evoked = mne.read_evokeds(evoked_fname, baseline=(None, 0))[0]  # doctest:+SKIP
        >>> evoked.savgol_filter(10.)  # low-pass at around 10 Hz # doctest:+SKIP
        >>> evoked.plot()  # doctest:+SKIP
        """  # noqa: E501
        from scipy.signal import savgol_filter
        _check_preload(self, 'inst.savgol_filter')
        h_freq = float(h_freq)
        if h_freq >= self.info['sfreq'] / 2.:
            raise ValueError('h_freq must be less than half the sample rate')

        # savitzky-golay filtering
        window_length = (int(np.round(self.info['sfreq'] /
                                      h_freq)) // 2) * 2 + 1
        logger.info('Using savgol length %d' % window_length)
        self._data[:] = savgol_filter(self._data, axis=-1, polyorder=5,
                                      window_length=window_length)
        return self

    @verbose
    def filter(self, l_freq, h_freq, picks=None, filter_length='auto',
               l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=None,
               method='fir', iir_params=None, phase='zero',
               fir_window='hamming', fir_design='firwin',
               skip_by_annotation=('edge', 'bad_acq_skip'), pad='edge', *,
               verbose=None):
        """Filter a subset of channels.

        Parameters
        ----------
        %(l_freq)s
        %(h_freq)s
        %(picks_all_data)s
        %(filter_length)s
        %(l_trans_bandwidth)s
        %(h_trans_bandwidth)s
        %(n_jobs_fir)s
        %(method_fir)s
        %(iir_params)s
        %(phase)s
        %(fir_window)s
        %(fir_design)s
        skip_by_annotation : str | list of str
            If a string (or list of str), any annotation segment that begins
            with the given string will not be included in filtering, and
            segments on either side of the given excluded annotated segment
            will be filtered separately (i.e., as independent signals).
            The default (``('edge', 'bad_acq_skip')`` will separately filter
            any segments that were concatenated by :func:`mne.concatenate_raws`
            or :meth:`mne.io.Raw.append`, or separated during acquisition.
            To disable, provide an empty list. Only used if ``inst`` is raw.

            .. versionadded:: 0.16.
        %(pad_fir)s
        %(verbose)s

        Returns
        -------
        inst : instance of Epochs, Evoked, or Raw
            The filtered data.

        See Also
        --------
        mne.filter.create_filter
        mne.Evoked.savgol_filter
        mne.io.Raw.notch_filter
        mne.io.Raw.resample
        mne.filter.create_filter
        mne.filter.filter_data
        mne.filter.construct_iir_filter

        Notes
        -----
        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by ``picks``.
        The data are modified inplace.

        The object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        ``l_freq`` and ``h_freq`` are the frequencies below which and above
        which, respectively, to filter out of the data. Thus the uses are:

            * ``l_freq < h_freq``: band-pass filter
            * ``l_freq > h_freq``: band-stop filter
            * ``l_freq is not None and h_freq is None``: high-pass filter
            * ``l_freq is None and h_freq is not None``: low-pass filter

        ``self.info['lowpass']`` and ``self.info['highpass']`` are only
        updated with picks=None.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporarily stored in memory.

        For more information, see the tutorials
        :ref:`disc-filtering` and :ref:`tut-filter-resample` and
        :func:`mne.filter.create_filter`.

        .. versionadded:: 0.15
        """
        from .io.base import BaseRaw
        _check_preload(self, 'inst.filter')
        if pad is None and method != 'iir':
            pad = 'edge'
        update_info, picks = _filt_check_picks(self.info, picks,
                                               l_freq, h_freq)
        if isinstance(self, BaseRaw):
            # Deal with annotations
            onsets, ends = _annotations_starts_stops(
                self, skip_by_annotation, invert=True)
            logger.info('Filtering raw data in %d contiguous segment%s'
                        % (len(onsets), _pl(onsets)))
        else:
            onsets, ends = np.array([0]), np.array([self._data.shape[1]])
        max_idx = (ends - onsets).argmax()
        for si, (start, stop) in enumerate(zip(onsets, ends)):
            # Only output filter params once (for info level), and only warn
            # once about the length criterion (longest segment is too short)
            use_verbose = verbose if si == max_idx else 'error'
            filter_data(
                self._data[:, start:stop], self.info['sfreq'], l_freq, h_freq,
                picks, filter_length, l_trans_bandwidth, h_trans_bandwidth,
                n_jobs, method, iir_params, copy=False, phase=phase,
                fir_window=fir_window, fir_design=fir_design, pad=pad,
                verbose=use_verbose)
        # update info if filter is applied to all data channels,
        # and it's not a band-stop filter
        _filt_update_info(self.info, update_info, l_freq, h_freq)
        return self

    @verbose
    def resample(self, sfreq, npad='auto', window='boxcar', n_jobs=None,
                 pad='edge', *, verbose=None):
        """Resample data.

        If appropriate, an anti-aliasing filter is applied before resampling.
        See :ref:`resampling-and-decimating` for more information.

        .. note:: Data must be loaded.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        %(npad)s
        %(window_resample)s
        %(n_jobs_cuda)s
        %(pad)s
            The default is ``'edge'``, which pads with the edge values of each
            vector.

            .. versionadded:: 0.15
        %(verbose)s

        Returns
        -------
        inst : instance of Epochs or Evoked
            The resampled object.

        See Also
        --------
        mne.io.Raw.resample

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        from .epochs import BaseEpochs
        from .evoked import Evoked
        # Should be guaranteed by our inheritance, and the fact that
        # mne.io.base.BaseRaw overrides this method
        assert isinstance(self, (BaseEpochs, Evoked))

        _check_preload(self, 'inst.resample')

        sfreq = float(sfreq)
        o_sfreq = self.info['sfreq']
        self._data = resample(self._data, sfreq, o_sfreq, npad, window=window,
                              n_jobs=n_jobs, pad=pad)
        lowpass = self.info.get('lowpass')
        lowpass = np.inf if lowpass is None else lowpass
        with self.info._unlock():
            self.info['lowpass'] = min(lowpass, sfreq / 2.)
            self.info['sfreq'] = float(sfreq)
        new_times = (np.arange(self._data.shape[-1], dtype=np.float64) /
                     sfreq + self.times[0])
        # adjust indirectly affected variables
        self._set_times(new_times)
        self._raw_times = self.times
        self._update_first_last()
        return self

    @verbose
    def apply_hilbert(self, picks=None, envelope=False, n_jobs=None,
                      n_fft='auto', *, verbose=None):
        """Compute analytic signal or envelope for a subset of channels.

        Parameters
        ----------
        %(picks_all_data_noref)s
        envelope : bool
            Compute the envelope signal of each channel. Default False.
            See Notes.
        %(n_jobs)s
        n_fft : int | None | str
            Points to use in the FFT for Hilbert transformation. The signal
            will be padded with zeros before computing Hilbert, then cut back
            to original length. If None, n == self.n_times. If 'auto',
            the next highest fast FFT length will be use.
        %(verbose)s

        Returns
        -------
        self : instance of Raw, Epochs, or Evoked
            The raw object with transformed data.

        Notes
        -----
        **Parameters**

        If ``envelope=False``, the analytic signal for the channels defined in
        ``picks`` is computed and the data of the Raw object is converted to
        a complex representation (the analytic signal is complex valued).

        If ``envelope=True``, the absolute value of the analytic signal for the
        channels defined in ``picks`` is computed, resulting in the envelope
        signal.

        .. warning: Do not use ``envelope=True`` if you intend to compute
                    an inverse solution from the raw data. If you want to
                    compute the envelope in source space, use
                    ``envelope=False`` and compute the envelope after the
                    inverse solution has been obtained.

        If envelope=False, more memory is required since the original raw data
        as well as the analytic signal have temporarily to be stored in memory.
        If n_jobs > 1, more memory is required as ``len(picks) * n_times``
        additional time points need to be temporarily stored in memory.

        Also note that the ``n_fft`` parameter will allow you to pad the signal
        with zeros before performing the Hilbert transform. This padding
        is cut off, but it may result in a slightly different result
        (particularly around the edges). Use at your own risk.

        **Analytic signal**

        The analytic signal "x_a(t)" of "x(t)" is::

            x_a = F^{-1}(F(x) 2U) = x + i y

        where "F" is the Fourier transform, "U" the unit step function,
        and "y" the Hilbert transform of "x". One usage of the analytic
        signal is the computation of the envelope signal, which is given by
        "e(t) = abs(x_a(t))". Due to the linearity of Hilbert transform and the
        MNE inverse solution, the enevlope in source space can be obtained
        by computing the analytic signal in sensor space, applying the MNE
        inverse, and computing the envelope in source space.
        """
        _check_preload(self, 'inst.apply_hilbert')
        if n_fft is None:
            n_fft = len(self.times)
        elif isinstance(n_fft, str):
            if n_fft != 'auto':
                raise ValueError('n_fft must be an integer, string, or None, '
                                 'got %s' % (type(n_fft),))
            n_fft = next_fast_len(len(self.times))
        n_fft = int(n_fft)
        if n_fft < len(self.times):
            raise ValueError("n_fft (%d) must be at least the number of time "
                             "points (%d)" % (n_fft, len(self.times)))
        dtype = None if envelope else np.complex128
        picks = _picks_to_idx(self.info, picks, exclude=(), with_ref_meg=False)
        args, kwargs = (), dict(n_fft=n_fft, envelope=envelope)

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        parallel, p_fun, n_jobs = parallel_func(_check_fun, n_jobs)
        if n_jobs == 1:
            # modify data inplace to save memory
            for idx in picks:
                self._data[..., idx, :] = _check_fun(
                    _my_hilbert, data_in[..., idx, :], *args, **kwargs)
        else:
            # use parallel function
            data_picks_new = parallel(
                p_fun(_my_hilbert, data_in[..., p, :], *args, **kwargs)
                for p in picks)
            for pp, p in enumerate(picks):
                self._data[..., p, :] = data_picks_new[pp]
        return self


def _check_fun(fun, d, *args, **kwargs):
    """Check shapes."""
    want_shape = d.shape
    d = fun(d, *args, **kwargs)
    if not isinstance(d, np.ndarray):
        raise TypeError('Return value must be an ndarray')
    if d.shape != want_shape:
        raise ValueError('Return data must have shape %s not %s'
                         % (want_shape, d.shape))
    return d


def _my_hilbert(x, n_fft=None, envelope=False):
    """Compute Hilbert transform of signals w/ zero padding.

    Parameters
    ----------
    x : array, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.

    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
    from scipy.signal import hilbert
    n_x = x.shape[-1]
    out = hilbert(x, N=n_fft, axis=-1)[..., :n_x]
    if envelope:
        out = np.abs(out)
    return out


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
    %(verbose)s

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
    and ones in the passband, with squared cosine ramps in between.
    """
    from scipy.fft import irfft
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
    h = irfft(freq_resp, n=2 * len(freq_resp) - 1)
    h = np.roll(h, n_freqs - 1)  # center the impulse like a linear-phase filt
    return h


def _filt_check_picks(info, picks, h_freq, l_freq):
    from .io.pick import _picks_to_idx
    update_info = False
    # This will pick *all* data channels
    picks = _picks_to_idx(info, picks, 'data_or_ica', exclude=())
    if h_freq is not None or l_freq is not None:
        data_picks = _picks_to_idx(info, None, 'data_or_ica', exclude=(),
                                   allow_empty=True)
        if len(data_picks) == 0:
            logger.info('No data channels found. The highpass and '
                        'lowpass values in the measurement info will not '
                        'be updated.')
        elif np.in1d(data_picks, picks).all():
            update_info = True
        else:
            logger.info('Filtering a subset of channels. The highpass and '
                        'lowpass values in the measurement info will not '
                        'be updated.')
    return update_info, picks


def _filt_update_info(info, update_info, l_freq, h_freq):
    if update_info:
        if h_freq is not None and (l_freq is None or l_freq < h_freq) and \
                (info["lowpass"] is None or h_freq < info['lowpass']):
            with info._unlock():
                info['lowpass'] = float(h_freq)
        if l_freq is not None and (h_freq is None or l_freq < h_freq) and \
                (info["highpass"] is None or l_freq > info['highpass']):
            with info._unlock():
                info['highpass'] = float(l_freq)
