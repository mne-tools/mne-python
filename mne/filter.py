"""IIR and FIR filtering functions"""

from copy import deepcopy

import numpy as np
from scipy.fftpack import fft, ifftshift, fftfreq

from .cuda import (setup_cuda_fft_multiply_repeated, fft_multiply_repeated,
                   setup_cuda_fft_resample, fft_resample, _smart_pad)
from .externals.six import string_types, integer_types
from .fixes import get_firwin2, get_filtfilt
from .parallel import parallel_func, check_n_jobs
from .time_frequency.multitaper import dpss_windows, _mt_spectra
from .utils import logger, verbose, sum_squared, check_version, warn


def is_power2(num):
    """Test if number is a power of 2

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


def _overlap_add_filter(x, h, n_fft=None, zero_phase=True, picks=None,
                        n_jobs=1):
    """ Filter using overlap-add FFTs.

    Filters the signal x using a filter with the impulse response h.
    If zero_phase==True, the the filter is applied twice, once in the forward
    direction and once backward , resulting in a zero-phase filter.

    .. warning:: This operates on the data in-place.

    Parameters
    ----------
    x : 2d array
        Signal to filter.
    h : 1d array
        Filter impulse response (FIR filter coefficients).
    n_fft : int
        Length of the FFT. If None, the best size is determined automatically.
    zero_phase : bool
        If True: the filter is applied in forward and backward direction,
        resulting in a zero-phase filter.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.

    Returns
    -------
    xf : 2d array
        x filtered.
    """
    if picks is None:
        picks = np.arange(x.shape[0])

    # Extend the signal by mirroring the edges to reduce transient filter
    # response
    n_h = len(h)
    if n_h == 1:
        return x * h ** 2 if zero_phase else x * h
    if x.shape[1] < len(h):
        raise ValueError('Overlap add should only be used for signals '
                         'longer than the requested filter')
    n_edge = max(min(n_h, x.shape[1]) - 1, 0)

    n_x = x.shape[1] + 2 * n_edge

    # Determine FFT length to use
    if n_fft is None:
        min_fft = 2 * n_h - 1
        max_fft = n_x
        if max_fft >= min_fft:
            n_tot = 2 * n_x if zero_phase else n_x

            # cost function based on number of multiplications
            N = 2 ** np.arange(np.ceil(np.log2(min_fft)),
                               np.ceil(np.log2(max_fft)) + 1, dtype=int)
            # if doing zero-phase, h needs to be thought of as ~ twice as long
            n_h_cost = 2 * n_h - 1 if zero_phase else n_h
            cost = (np.ceil(n_tot / (N - n_h_cost + 1).astype(np.float)) *
                    N * (np.log2(N) + 1))

            # add a heuristic term to prevent too-long FFT's which are slow
            # (not predicted by mult. cost alone, 4e-5 exp. determined)
            cost += 4e-5 * N * n_tot

            n_fft = N[np.argmin(cost)]
        else:
            # Use only a single block
            n_fft = 2 ** int(np.ceil(np.log2(n_x + n_h - 1)))

    if zero_phase and n_fft <= 2 * n_h - 1:
        raise ValueError("n_fft is too short, has to be at least "
                         "2 * len(h) - 1 if zero_phase == True")
    elif not zero_phase and n_fft <= n_h:
        raise ValueError("n_fft is too short, has to be at least "
                         "len(h) if zero_phase == False")

    if not is_power2(n_fft):
        warn("FFT length is not a power of 2. Can be slower.")

    # Filter in frequency domain
    h_fft = fft(np.concatenate([h, np.zeros(n_fft - n_h, dtype=h.dtype)]))
    assert(len(h_fft) == n_fft)

    if zero_phase:
        """Zero-phase filtering is now done in one pass by taking the squared
        magnitude of h_fft. This gives equivalent results to the old two-pass
        method but theoretically doubles the speed for long fft lengths. To
        compensate for this, overlapping must be done both before and after
        each segment. When zero_phase == False it only needs to be done after.
        """
        h_fft = (h_fft * h_fft.conj()).real
        # equivalent to convolving h(t) and h(-t) in the time domain

    # Figure out if we should use CUDA
    n_jobs, cuda_dict, h_fft = setup_cuda_fft_multiply_repeated(n_jobs, h_fft)

    # Process each row separately
    if n_jobs == 1:
        for p in picks:
            x[p] = _1d_overlap_filter(x[p], h_fft, n_h, n_edge, zero_phase,
                                      cuda_dict)
    else:
        parallel, p_fun, _ = parallel_func(_1d_overlap_filter, n_jobs)
        data_new = parallel(p_fun(x[p], h_fft, n_h, n_edge, zero_phase,
                                  cuda_dict)
                            for p in picks)
        for pp, p in enumerate(picks):
            x[p] = data_new[pp]

    return x


def _1d_overlap_filter(x, h_fft, n_h, n_edge, zero_phase, cuda_dict):
    """Do one-dimensional overlap-add FFT FIR filtering"""
    # pad to reduce ringing
    if cuda_dict['use_cuda']:
        n_fft = cuda_dict['x'].size  # account for CUDA's modification of h_fft
    else:
        n_fft = len(h_fft)
    x_ext = _smart_pad(x, np.array([n_edge, n_edge]))
    n_x = len(x_ext)
    x_filtered = np.zeros_like(x_ext)

    if zero_phase:
        # Segment length for signal x (convolving twice)
        n_seg = n_fft - 2 * (n_h - 1) - 1

        # Number of segments (including fractional segments)
        n_segments = int(np.ceil(n_x / float(n_seg)))

        # padding parameters to ensure filtering is done properly
        pre_pad = n_h - 1
        post_pad = n_fft - (n_h - 1)
    else:
        n_seg = n_fft - n_h + 1
        n_segments = int(np.ceil(n_x / float(n_seg)))
        pre_pad = 0
        post_pad = n_fft

    # Now the actual filtering step is identical for zero-phase (filtfilt-like)
    # or single-pass
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = (seg_idx + 1) * n_seg
        seg = x_ext[start:stop]
        seg = np.concatenate([np.zeros(pre_pad), seg,
                              np.zeros(post_pad - len(seg))])

        prod = fft_multiply_repeated(h_fft, seg, cuda_dict)

        start_filt = max(0, start - pre_pad)
        stop_filt = min(start - pre_pad + n_fft, n_x)
        start_prod = max(0, pre_pad - start)
        stop_prod = start_prod + stop_filt - start_filt
        x_filtered[start_filt:stop_filt] += prod[start_prod:stop_prod]

    # Remove mirrored edges that we added and cast
    if n_edge > 0:
        x_filtered = x_filtered[n_edge:-n_edge]
    x_filtered = x_filtered.astype(x.dtype)
    return x_filtered


def _filter_attenuation(h, freq, gain):
    """Compute minimum attenuation at stop frequency"""
    from scipy.signal import freqz
    _, filt_resp = freqz(h.ravel(), worN=np.pi * freq)
    filt_resp = np.abs(filt_resp)  # use amplitude response
    filt_resp /= np.max(filt_resp)
    filt_resp[np.where(gain == 1)] = 0
    idx = np.argmax(filt_resp)
    att_db = -20 * np.log10(filt_resp[idx])
    att_freq = freq[idx]
    return att_db, att_freq


def _1d_fftmult_ext(x, B, extend_x, cuda_dict):
    """Helper to parallelize FFT FIR, with extension if necessary"""
    # extend, if necessary
    if extend_x is True:
        x = np.r_[x, x[-1]]

    # do Fourier transforms
    xf = fft_multiply_repeated(B, x, cuda_dict)

    # put back to original size and type
    if extend_x is True:
        xf = xf[:-1]

    xf = xf.astype(x.dtype)
    return xf


def _prep_for_filtering(x, copy, picks=None):
    """Set up array as 2D for filtering ease"""
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

    return x, orig_shape, picks


def _filter(x, Fs, freq, gain, filter_length='10s', picks=None, n_jobs=1,
            copy=True):
    """Filter signal using gain control points in the frequency domain.

    The filter impulse response is constructed from a Hamming window (window
    used in "firwin2" function) to avoid ripples in the frequency response
    (windowing is a smoothing in frequency domain). The filter is zero-phase.

    If x is multi-dimensional, this operates along the last dimension.

    Parameters
    ----------
    x : array
        Signal to filter.
    Fs : float
        Sampling rate in Hz.
    freq : 1d array
        Frequency sampling points in Hz.
    gain : 1d array
        Filter gain at frequency sampling points.
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.

    Returns
    -------
    xf : array
        x filtered.
    """
    firwin2 = get_firwin2()
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)

    # issue a warning if attenuation is less than this
    min_att_db = 20

    # normalize frequencies
    freq = np.array(freq) / (Fs / 2.)
    gain = np.array(gain)
    filter_length = _get_filter_length(filter_length, Fs, len_x=x.shape[1])
    n_jobs = check_n_jobs(n_jobs, allow_cuda=True)

    if filter_length is None or x.shape[1] <= filter_length:
        # Use direct FFT filtering for short signals

        Norig = x.shape[1]

        extend_x = False
        if (gain[-1] == 0.0 and Norig % 2 == 1) \
                or (gain[-1] == 1.0 and Norig % 2 != 1):
            # Gain at Nyquist freq: 1: make x EVEN, 0: make x ODD
            extend_x = True

        N = x.shape[1] + (extend_x is True)

        h = firwin2(N, freq, gain)[np.newaxis, :]

        att_db, att_freq = _filter_attenuation(h, freq, gain)
        if att_db < min_att_db:
            att_freq *= Fs / 2
            warn('Attenuation at stop frequency %0.1fHz is only %0.1fdB.'
                 % (att_freq, att_db))

        # Make zero-phase filter function
        B = np.abs(fft(h)).ravel()

        # Figure out if we should use CUDA
        n_jobs, cuda_dict, B = setup_cuda_fft_multiply_repeated(n_jobs, B)

        if n_jobs == 1:
            for p in picks:
                x[p] = _1d_fftmult_ext(x[p], B, extend_x, cuda_dict)
        else:
            parallel, p_fun, _ = parallel_func(_1d_fftmult_ext, n_jobs)
            data_new = parallel(p_fun(x[p], B, extend_x, cuda_dict)
                                for p in picks)
            for pp, p in enumerate(picks):
                x[p] = data_new[pp]
    else:
        # Use overlap-add filter with a fixed length
        N = filter_length

        if (gain[-1] == 0.0 and N % 2 == 1) \
                or (gain[-1] == 1.0 and N % 2 != 1):
            # Gain at Nyquist freq: 1: make N EVEN, 0: make N ODD
            N += 1

        # construct filter with gain resulting from forward-backward filtering
        h = firwin2(N, freq, gain, window='hann')

        att_db, att_freq = _filter_attenuation(h, freq, gain)
        att_db += 6  # the filter is applied twice (zero phase)
        if att_db < min_att_db:
            att_freq *= Fs / 2
            warn('Attenuation at stop frequency %0.1fHz is only %0.1fdB. '
                 'Increase filter_length for higher attenuation.'
                 % (att_freq, att_db))

        # reconstruct filter, this time with appropriate gain for fwd-bkwd
        gain = np.sqrt(gain)
        h = firwin2(N, freq, gain, window='hann')
        x = _overlap_add_filter(x, h, zero_phase=True, picks=picks,
                                n_jobs=n_jobs)

    x.shape = orig_shape
    return x


def _check_coefficients(b, a):
    """Check for filter stability"""
    from scipy.signal import tf2zpk
    z, p, k = tf2zpk(b, a)
    if np.any(np.abs(p) > 1.0):
        raise RuntimeError('Filter poles outside unit circle, filter will be '
                           'unstable. Consider using different filter '
                           'coefficients.')


def _filtfilt(x, b, a, padlen, picks, n_jobs, copy):
    """Helper to more easily call filtfilt"""
    # set up array for filtering, reshape to 2D, operate on last axis
    filtfilt = get_filtfilt()
    n_jobs = check_n_jobs(n_jobs)
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    _check_coefficients(b, a)
    if n_jobs == 1:
        for p in picks:
            x[p] = filtfilt(b, a, x[p], padlen=padlen)
    else:
        parallel, p_fun, _ = parallel_func(filtfilt, n_jobs)
        data_new = parallel(p_fun(b, a, x[p], padlen=padlen)
                            for p in picks)
        for pp, p in enumerate(picks):
            x[p] = data_new[pp]
    x.shape = orig_shape
    return x


def _estimate_ringing_samples(b, a):
    """Helper function for determining IIR padding"""
    from scipy.signal import lfilter
    x = np.zeros(1000)
    x[0] = 1
    h = lfilter(b, a, x)
    return np.where(np.abs(h) > 0.001 * np.max(np.abs(h)))[0][-1]


def construct_iir_filter(iir_params=dict(b=[1, 0], a=[1, 0], padlen=0),
                         f_pass=None, f_stop=None, sfreq=None, btype=None,
                         return_copy=True):
    """Use IIR parameters to get filtering coefficients

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
        If iir_params['b'] and iir_params['a'] exist, these will be used
        as coefficients to perform IIR filtering. Otherwise, if
        iir_params['order'] and iir_params['ftype'] exist, these will be
        used with scipy.signal.iirfilter to make a filter. Otherwise, if
        iir_params['gpass'] and iir_params['gstop'] exist, these will be
        used with scipy.signal.iirdesign to design a filter.
        iir_params['padlen'] defines the number of samples to pad (and
        an estimate will be calculated if it is not given). See Notes for
        more details.
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
        (or analogous string representations known to scipy.signal).
    return_copy : bool
        If False, the 'b', 'a', and 'padlen' entries in iir_params will be
        set inplace (if they weren't already). Otherwise, a new iir_params
        instance will be created and returned with these entries.

    Returns
    -------
    iir_params : dict
        Updated iir_params dict, with the entries (set only if they didn't
        exist before) for 'b', 'a', and 'padlen' for IIR filtering.

    Notes
    -----
    This function triages calls to scipy.signal.iirfilter and iirdesign
    based on the input arguments (see descriptions of these functions
    and scipy's scipy.signal.filter_design documentation for details).

    Examples
    --------
    iir_params can have several forms. Consider constructing a low-pass
    filter at 40 Hz with 1000 Hz sampling rate.

    In the most basic (2-parameter) form of iir_params, the order of the
    filter 'N' and the type of filtering 'ftype' are specified. To get
    coefficients for a 4th-order Butterworth filter, this would be:

    >>> iir_params = dict(order=4, ftype='butter')
    >>> iir_params = construct_iir_filter(iir_params, 40, None, 1000, 'low', return_copy=False)
    >>> print((len(iir_params['b']), len(iir_params['a']), iir_params['padlen']))
    (5, 5, 82)

    Filters can also be constructed using filter design methods. To get a
    40 Hz Chebyshev type 1 lowpass with specific gain characteristics in the
    pass and stop bands (assuming the desired stop band is at 45 Hz), this
    would be a filter with much longer ringing:

    >>> iir_params = dict(ftype='cheby1', gpass=3, gstop=20)
    >>> iir_params = construct_iir_filter(iir_params, 40, 50, 1000, 'low')
    >>> print((len(iir_params['b']), len(iir_params['a']), iir_params['padlen']))
    (6, 6, 439)

    Padding and/or filter coefficients can also be manually specified. For
    a 10-sample moving window with no padding during filtering, for example,
    one can just do:

    >>> iir_params = dict(b=np.ones((10)), a=[1, 0], padlen=0)
    >>> iir_params = construct_iir_filter(iir_params, return_copy=False)
    >>> print((iir_params['b'], iir_params['a'], iir_params['padlen']))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), [1, 0], 0)

    """  # noqa
    from scipy.signal import iirfilter, iirdesign
    known_filters = ('bessel', 'butter', 'butterworth', 'cauer', 'cheby1',
                     'cheby2', 'chebyshev1', 'chebyshev2', 'chebyshevi',
                     'chebyshevii', 'ellip', 'elliptic')
    a = None
    b = None
    # if the filter has been designed, we're good to go
    if 'a' in iir_params and 'b' in iir_params:
        [b, a] = [iir_params['b'], iir_params['a']]
    else:
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
            [b, a] = iirfilter(iir_params['order'], Wp, btype=btype,
                               ftype=ftype)
        else:
            # use gpass / gstop design
            Ws = np.asanyarray(f_stop) / (float(sfreq) / 2)
            if 'gpass' not in iir_params or 'gstop' not in iir_params:
                raise ValueError('iir_params must have at least ''gstop'' and'
                                 ' ''gpass'' (or ''N'') entries')
            [b, a] = iirdesign(Wp, Ws, iir_params['gpass'],
                               iir_params['gstop'], ftype=ftype)

    if a is None or b is None:
        raise RuntimeError('coefficients could not be created from iir_params')

    # now deal with padding
    if 'padlen' not in iir_params:
        padlen = _estimate_ringing_samples(b, a)
    else:
        padlen = iir_params['padlen']

    if return_copy:
        iir_params = deepcopy(iir_params)

    iir_params.update(dict(b=b, a=a, padlen=padlen))
    return iir_params


def _check_method(method, iir_params, extra_types):
    """Helper to parse method arguments"""
    allowed_types = ['iir', 'fft'] + extra_types
    if not isinstance(method, string_types):
        raise TypeError('method must be a string')
    if method not in allowed_types:
        raise ValueError('method must be one of %s, not "%s"'
                         % (allowed_types, method))
    if method == 'iir':
        if iir_params is None:
            iir_params = dict(order=4, ftype='butter')
        if not isinstance(iir_params, dict):
            raise ValueError('iir_params must be a dict')
    elif iir_params is not None:
        raise ValueError('iir_params must be None if method != "iir"')
    method = method.lower()
    return iir_params


@verbose
def band_pass_filter(x, Fs, Fp1, Fp2, filter_length='10s',
                     l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                     method='fft', iir_params=None,
                     picks=None, n_jobs=1, copy=True, verbose=None):
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
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    l_trans_bandwidth : float
        Width of the transition band at the low cut-off frequency in Hz.
        Not used if 'order' is specified in iir_params.
    h_trans_bandwidth : float
        Width of the transition band at the high cut-off frequency in Hz.
        Not used if 'order' is specified in iir_params.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    low_pass_filter, high_pass_filter

    Notes
    -----
    The frequency response is (approximately) given by::

                     ----------
                   /|         | \
                  / |         |  \
                 /  |         |   \
                /   |         |    \
      ----------    |         |     -----------------
                    |         |
              Fs1  Fp1       Fp2   Fs2

    Where:

        Fs1 = Fp1 - l_trans_bandwidth in Hz
        Fs2 = Fp2 + h_trans_bandwidth in Hz
    """
    iir_params = _check_method(method, iir_params, [])

    Fs = float(Fs)
    Fp1 = float(Fp1)
    Fp2 = float(Fp2)
    Fs1 = Fp1 - l_trans_bandwidth if method == 'fft' else Fp1
    Fs2 = Fp2 + h_trans_bandwidth if method == 'fft' else Fp2
    if Fs2 > Fs / 2:
        raise ValueError('Effective band-stop frequency (%s) is too high '
                         '(maximum based on Nyquist is %s)' % (Fs2, Fs / 2.))

    if Fs1 <= 0:
        raise ValueError('Filter specification invalid: Lower stop frequency '
                         'too low (%0.1fHz). Increase Fp1 or reduce '
                         'transition bandwidth (l_trans_bandwidth)' % Fs1)

    if method == 'fft':
        freq = [0, Fs1, Fp1, Fp2, Fs2, Fs / 2]
        gain = [0, 0, 1, 1, 0, 0]
        xf = _filter(x, Fs, freq, gain, filter_length, picks, n_jobs, copy)
    else:
        iir_params = construct_iir_filter(iir_params, [Fp1, Fp2],
                                          [Fs1, Fs2], Fs, 'bandpass')
        padlen = min(iir_params['padlen'], len(x))
        xf = _filtfilt(x, iir_params['b'], iir_params['a'], padlen,
                       picks, n_jobs, copy)

    return xf


@verbose
def band_stop_filter(x, Fs, Fp1, Fp2, filter_length='10s',
                     l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                     method='fft', iir_params=None,
                     picks=None, n_jobs=1, copy=True, verbose=None):
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
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    l_trans_bandwidth : float
        Width of the transition band at the low cut-off frequency in Hz.
        Not used if 'order' is specified in iir_params.
    h_trans_bandwidth : float
        Width of the transition band at the high cut-off frequency in Hz.
        Not used if 'order' is specified in iir_params.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x filtered.

    Notes
    -----
    The frequency response is (approximately) given by::

      ----------                   ----------
               |\                 /|
               | \               / |
               |  \             /  |
               |   \           /   |
               |    -----------    |
               |    |         |    |
              Fp1  Fs1       Fs2  Fp2

    Where:

        Fs1 = Fp1 + l_trans_bandwidth in Hz
        Fs2 = Fp2 - h_trans_bandwidth in Hz

    Note that multiple stop bands can be specified using arrays.
    """
    iir_params = _check_method(method, iir_params, [])

    Fp1 = np.atleast_1d(Fp1)
    Fp2 = np.atleast_1d(Fp2)
    if not len(Fp1) == len(Fp2):
        raise ValueError('Fp1 and Fp2 must be the same length')

    Fs = float(Fs)
    Fp1 = Fp1.astype(float)
    Fp2 = Fp2.astype(float)
    Fs1 = Fp1 + l_trans_bandwidth if method == 'fft' else Fp1
    Fs2 = Fp2 - h_trans_bandwidth if method == 'fft' else Fp2

    if np.any(Fs1 <= 0):
        raise ValueError('Filter specification invalid: Lower stop frequency '
                         'too low (%0.1fHz). Increase Fp1 or reduce '
                         'transition bandwidth (l_trans_bandwidth)' % Fs1)

    if method == 'fft':
        freq = np.r_[0, Fp1, Fs1, Fs2, Fp2, Fs / 2]
        gain = np.r_[1, np.ones_like(Fp1), np.zeros_like(Fs1),
                     np.zeros_like(Fs2), np.ones_like(Fp2), 1]
        order = np.argsort(freq)
        freq = freq[order]
        gain = gain[order]
        if np.any(np.abs(np.diff(gain, 2)) > 1):
            raise ValueError('Stop bands are not sufficiently separated.')
        xf = _filter(x, Fs, freq, gain, filter_length, picks, n_jobs, copy)
    else:
        for fp_1, fp_2, fs_1, fs_2 in zip(Fp1, Fp2, Fs1, Fs2):
            iir_params_new = construct_iir_filter(iir_params, [fp_1, fp_2],
                                                  [fs_1, fs_2], Fs, 'bandstop')
            padlen = min(iir_params_new['padlen'], len(x))
            xf = _filtfilt(x, iir_params_new['b'], iir_params_new['a'], padlen,
                           picks, n_jobs, copy)

    return xf


@verbose
def low_pass_filter(x, Fs, Fp, filter_length='10s', trans_bandwidth=0.5,
                    method='fft', iir_params=None,
                    picks=None, n_jobs=1, copy=True, verbose=None):
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
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    trans_bandwidth : float
        Width of the transition band in Hz. Not used if 'order' is specified
        in iir_params.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    resample
    band_pass_filter, high_pass_filter

    Notes
    -----
    The frequency response is (approximately) given by::

      -------------------------
                              | \
                              |  \
                              |   \
                              |    \
                              |     -----------------
                              |
                              Fp  Fp+trans_bandwidth

    """
    iir_params = _check_method(method, iir_params, [])
    Fs = float(Fs)
    Fp = float(Fp)
    Fstop = Fp + trans_bandwidth if method == 'fft' else Fp
    if Fstop > Fs / 2.:
        raise ValueError('Effective stop frequency (%s) is too high '
                         '(maximum based on Nyquist is %s)' % (Fstop, Fs / 2.))

    if method == 'fft':
        freq = [0, Fp, Fstop, Fs / 2]
        gain = [1, 1, 0, 0]
        xf = _filter(x, Fs, freq, gain, filter_length, picks, n_jobs, copy)
    else:
        iir_params = construct_iir_filter(iir_params, Fp, Fstop, Fs, 'low')
        padlen = min(iir_params['padlen'], len(x))
        xf = _filtfilt(x, iir_params['b'], iir_params['a'], padlen,
                       picks, n_jobs, copy)

    return xf


@verbose
def high_pass_filter(x, Fs, Fp, filter_length='10s', trans_bandwidth=0.5,
                     method='fft', iir_params=None,
                     picks=None, n_jobs=1, copy=True, verbose=None):
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
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    trans_bandwidth : float
        Width of the transition band in Hz. Not used if 'order' is
        specified in iir_params.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    picks : array-like of int | None
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x filtered.

    See Also
    --------
    low_pass_filter, band_pass_filter

    Notes
    -----
    The frequency response is (approximately) given by::

                       -----------------------
                     /|
                    / |
                   /  |
                  /   |
        ----------    |
                      |
               Fstop  Fp

    Where Fstop = Fp - trans_bandwidth.
    """
    iir_params = _check_method(method, iir_params, [])
    Fs = float(Fs)
    Fp = float(Fp)

    Fstop = Fp - trans_bandwidth if method == 'fft' else Fp
    if Fstop <= 0:
        raise ValueError('Filter specification invalid: Stop frequency too low'
                         '(%0.1fHz). Increase Fp or reduce transition '
                         'bandwidth (trans_bandwidth)' % Fstop)

    if method == 'fft':
        freq = [0, Fstop, Fp, Fs / 2]
        gain = [0, 0, 1, 1]
        xf = _filter(x, Fs, freq, gain, filter_length, picks, n_jobs, copy)
    else:
        iir_params = construct_iir_filter(iir_params, Fp, Fstop, Fs, 'high')
        padlen = min(iir_params['padlen'], len(x))
        xf = _filtfilt(x, iir_params['b'], iir_params['a'], padlen,
                       picks, n_jobs, copy)

    return xf


@verbose
def notch_filter(x, Fs, freqs, filter_length='10s', notch_widths=None,
                 trans_bandwidth=1, method='fft',
                 iir_params=None, mt_bandwidth=None,
                 p_value=0.05, picks=None, n_jobs=1, copy=True, verbose=None):
    """Notch filter for the signal x.

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
    filter_length : str (Default: '10s') | int | None
        Length of the filter to use. If None or "len(x) < filter_length",
        the filter length used is len(x). Otherwise, if int, overlap-add
        filtering with a filter of the specified length in samples) is
        used (faster for long signals). If str, a human-readable time in
        units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
        to the shortest power-of-two length at least that duration.
        Not used for 'iir' filters.
    notch_widths : float | array of float | None
        Width of the stop band (centred at each freq in freqs) in Hz.
        If None, freqs / 200 is used.
    trans_bandwidth : float
        Width of the transition band in Hz. Not used if 'order' is
        specified in iir_params.
    method : str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
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
        Indices to filter. If None all indices will be filtered.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly, CUDA is initialized, and method='fft'.
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x filtered.

    Notes
    -----
    The frequency response is (approximately) given by::

      ----------         -----------
               |\       /|
               | \     / |
               |  \   /  |
               |   \ /   |
               |    -    |
               |    |    |
              Fp1 freq  Fp2

    For each freq in freqs, where:

        Fp1 = freq - trans_bandwidth / 2 in Hz
        Fs2 = freq + trans_bandwidth / 2 in Hz

    References
    ----------
    Multi-taper removal is inspired by code from the Chronux toolbox, see
    www.chronux.org and the book "Observed Brain Dynamics" by Partha Mitra
    & Hemant Bokil, Oxford University Press, New York, 2008. Please
    cite this in publications if method 'spectrum_fit' is used.
    """
    iir_params = _check_method(method, iir_params, ['spectrum_fit'])

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

    if method in ['fft', 'iir']:
        # Speed this up by computing the fourier coefficients once
        tb_2 = trans_bandwidth / 2.0
        lows = [freq - nw / 2.0 - tb_2
                for freq, nw in zip(freqs, notch_widths)]
        highs = [freq + nw / 2.0 + tb_2
                 for freq, nw in zip(freqs, notch_widths)]
        xf = band_stop_filter(x, Fs, lows, highs, filter_length, tb_2, tb_2,
                              method, iir_params, picks, n_jobs, copy)
    elif method == 'spectrum_fit':
        xf = _mt_spectrum_proc(x, Fs, freqs, notch_widths, mt_bandwidth,
                               p_value, picks, n_jobs, copy)

    return xf


def _mt_spectrum_proc(x, sfreq, line_freqs, notch_widths, mt_bandwidth,
                      p_value, picks, n_jobs, copy):
    """Helper to more easily call _mt_spectrum_remove"""
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
    """Use MT-spectrum to remove line frequencies

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
    """Resample the array x

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
        See scipy.signal.resample for description.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

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
    stim_data : 1D array, shape (n_samples,) |
                2D array, shape (n_stim_channels, n_samples)
        Stim channels to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.

    Returns
    -------
    stim_resampled : 2D array, shape (n_stim_channels, n_samples_resampled)
        The resampled stim channels

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


def _get_filter_length(filter_length, sfreq, min_length=128, len_x=np.inf):
    """Helper to determine a reasonable filter length"""
    if not isinstance(min_length, int):
        raise ValueError('min_length must be an int')
    if isinstance(filter_length, string_types):
        # parse time values
        if filter_length[-2:].lower() == 'ms':
            mult_fact = 1e-3
            filter_length = filter_length[:-2]
        elif filter_length[-1].lower() == 's':
            mult_fact = 1
            filter_length = filter_length[:-1]
        else:
            raise ValueError('filter_length, if a string, must be a '
                             'human-readable time (e.g., "10s"), not '
                             '"%s"' % filter_length)
        # now get the number
        try:
            filter_length = float(filter_length)
        except ValueError:
            raise ValueError('filter_length, if a string, must be a '
                             'human-readable time (e.g., "10s"), not '
                             '"%s"' % filter_length)
        filter_length = 2 ** int(np.ceil(np.log2(filter_length *
                                                 mult_fact * sfreq)))
        # shouldn't make filter longer than length of x
        if filter_length >= len_x:
            filter_length = len_x
        # only need to check min_length if the filter is shorter than len_x
        elif filter_length < min_length:
            filter_length = min_length
            warn('filter_length was too short, using filter of length %d '
                 'samples ("%0.1fs")'
                 % (filter_length, filter_length / float(sfreq)))

    if filter_length is not None:
        if not isinstance(filter_length, integer_types):
            raise ValueError('filter_length must be str, int, or None')
    return filter_length


class FilterMixin(object):
    """Object for Epoch/Evoked filtering"""

    def savgol_filter(self, h_freq, copy=False):
        """Filter the data using Savitzky-Golay polynomial method

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
        """  # noqa
        from .evoked import Evoked
        from .epochs import _BaseEpochs
        inst = self.copy() if copy else self
        if isinstance(inst, Evoked):
            data = inst.data
            axis = 1
        elif isinstance(inst, _BaseEpochs):
            if not inst.preload:
                raise RuntimeError('data must be preloaded to filter')
            data = inst._data
            axis = 2

        h_freq = float(h_freq)
        if h_freq >= inst.info['sfreq'] / 2.:
            raise ValueError('h_freq must be less than half the sample rate')

        # savitzky-golay filtering
        if not check_version('scipy', '0.14'):
            raise RuntimeError('scipy >= 0.14 must be installed for savgol')
        from scipy.signal import savgol_filter
        window_length = (int(np.round(inst.info['sfreq'] /
                                      h_freq)) // 2) * 2 + 1
        data[...] = savgol_filter(data, axis=axis, polyorder=5,
                                  window_length=window_length)
        return inst
