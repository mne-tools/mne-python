import warnings
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import freqz, iirdesign, iirfilter, filter_dict
from fixes import filtfilt
from scipy import signal
from copy import deepcopy

from .utils import firwin2  # back port for old scipy


def is_power2(num):
    """Test if number is a power of 2

    Parameters
    ----------
    num : int
        Number

    Returns
    -------
    b : bool
        True if is power of 2

    Example
    -------
    >>> is_power2(2 ** 3)
    True
    >>> is_power2(5)
    False
    """
    num = int(num)
    return num != 0 and ((num & (num - 1)) == 0)


def _overlap_add_filter(x, h, n_fft=None, zero_phase=True):
    """ Filter using overlap-add FFTs.

    Filters the signal x using a filter with the impulse response h.
    If zero_phase==True, the amplitude response is scaled and the filter is
    applied in forward and backward direction, resulting in a zero-phase
    filter.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    h : 1d array
        Filter impulse response (FIR filter coefficients)
    n_fft : int
        Length of the FFT. If None, the best size is determined automatically.
    zero_phase : bool
        If True: the filter is applied in forward and backward direction,
        resulting in a zero-phase filter

    Returns
    -------
    xf : 1d array
        x filtered
    """
    n_h = len(h)

    # Extend the signal by mirroring the edges to reduce transient filter
    # response
    n_edge = min(n_h, len(x))

    x_ext = np.r_[2 * x[0] - x[n_edge - 1:0:-1],\
                  x, 2 * x[-1] - x[-2:-n_edge - 1:-1]]

    n_x = len(x_ext)

    # Determine FFT length to use
    if n_fft is None:
        if n_x > n_h:
            n_tot = 2 * n_x if zero_phase else n_x

            N = 2 ** np.arange(np.ceil(np.log2(n_h)),
                               np.floor(np.log2(n_tot)))
            cost = np.ceil(n_tot / (N - n_h + 1)) * N * (np.log2(N) + 1)
            n_fft = N[np.argmin(cost)]
        else:
            # Use only a single block
            n_fft = 2 ** np.ceil(np.log2(n_x + n_h - 1))

    if n_fft <= 0:
        raise ValueError('n_fft is too short, has to be at least len(h)')

    if not is_power2(n_fft):
        warnings.warn("FFT length is not a power of 2. Can be slower.")

    # Filter in frequency domain
    h_fft = fft(np.r_[h, np.zeros(n_fft - n_h, dtype=h.dtype)])

    if zero_phase:
        # We will apply the filter in forward and backward direction: Scale
        # frequency response of the filter so that the shape of the amplitude
        # response stays the same when it is applied twice

        # be careful not to divide by too small numbers
        idx = np.where(np.abs(h_fft) > 1e-6)
        h_fft[idx] = h_fft[idx] / np.sqrt(np.abs(h_fft[idx]))

    x_filtered = np.zeros_like(x_ext)

    # Segment length for signal x
    n_seg = n_fft - n_h + 1

    # Number of segments (including fractional segments)
    n_segments = int(np.ceil(n_x / float(n_seg)))

    filter_input = x_ext

    for pass_no in range(2) if zero_phase else range(1):

        if pass_no == 1:
            # second pass: flip signal
            filter_input = np.flipud(x_filtered)
            x_filtered = np.zeros_like(x_ext)

        for seg_idx in range(n_segments):
            seg = filter_input[seg_idx * n_seg:(seg_idx + 1) * n_seg]
            seg_fft = fft(np.r_[seg, np.zeros(n_fft - len(seg))])

            if seg_idx * n_seg + n_fft < n_x:
                x_filtered[seg_idx * n_seg:seg_idx * n_seg + n_fft]\
                    += np.real(ifft(h_fft * seg_fft))
            else:
                # Last segment
                x_filtered[seg_idx * n_seg:] \
                    += np.real(ifft(h_fft * seg_fft))[:n_x - seg_idx * n_seg]

    # Remove mirrored edges that we added
    x_filtered = x_filtered[n_edge - 1:-n_edge + 1]

    if zero_phase:
        # flip signal back
        x_filtered = np.flipud(x_filtered)

    return x_filtered


def _filter_attenuation(h, freq, gain):
    """Compute minimum attenuation at stop frequency"""

    _, filt_resp = freqz(h, worN=np.pi * freq)
    filt_resp = np.abs(filt_resp)  # use amplitude response
    filt_resp /= np.max(filt_resp)
    filt_resp[np.where(gain == 1)] = 0
    idx = np.argmax(filt_resp)
    att_db = -20 * np.log10(filt_resp[idx])
    att_freq = freq[idx]

    return att_db, att_freq


def _filter(x, Fs, freq, gain, filter_length=None):
    """Filter signal using gain control points in the frequency domain.

    The filter impulse response is constructed from a Hamming window (window
    used in "firwin2" function) to avoid ripples in the frequency reponse
    (windowing is a smoothing in frequency domain). The filter is zero-phase.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        Sampling rate in Hz
    freq : 1d array
        Frequency sampling points in Hz
    gain : 1d array
        Filter gain at frequency sampling points
    filter_length : int (default: None)
        Length of the filter to use. If None or "len(x) < filter_length", the
        filter length used is len(x). Otherwise, overlap-add filtering with a
        filter of the specified length is used (faster for long signals).

    Returns
    -------
    xf : 1d array
        x filtered
    """

    # issue a warning if attenuation is less than this
    min_att_db = 20

    assert x.ndim == 1

    # normalize frequencies
    freq = np.array([f / (Fs / 2) for f in freq])
    gain = np.array(gain)

    if filter_length is None or len(x) <= filter_length:
        # Use direct FFT filtering for short signals

        Norig = len(x)

        if (gain[-1] == 0.0 and Norig % 2 == 1) \
                or (gain[-1] == 1.0 and Norig % 2 != 1):
            # Gain at Nyquist freq: 1: make x EVEN, 0: make x ODD
            x = np.r_[x, x[-1]]

        N = len(x)

        H = firwin2(N, freq, gain)

        att_db, att_freq = _filter_attenuation(H, freq, gain)
        if att_db < min_att_db:
            att_freq *= Fs / 2
            warnings.warn('Attenuation at stop frequency %0.1fHz is only '
                          '%0.1fdB.' % (att_freq, att_db))

        # Make zero-phase filter function
        B = np.abs(fft(H))

        xf = np.real(ifft(fft(x) * B))
        xf = np.array(xf[:Norig], dtype=x.dtype)
        x = x[:Norig]
    else:
        # Use overlap-add filter with a fixed length
        N = filter_length

        if (gain[-1] == 0.0 and N % 2 == 1) \
                or (gain[-1] == 1.0 and N % 2 != 1):
            # Gain at Nyquist freq: 1: make N EVEN, 0: make N ODD
            N += 1

        H = firwin2(N, freq, gain)

        att_db, att_freq = _filter_attenuation(H, freq, gain)
        att_db += 6  # the filter is applied twice (zero phase)
        if att_db < min_att_db:
            att_freq *= Fs / 2
            warnings.warn('Attenuation at stop frequency %0.1fHz is only '
                          '%0.1fdB. Increase filter_length for higher '
                          'attenuation.' % (att_freq, att_db))

        xf = _overlap_add_filter(x, H, zero_phase=True)

    return xf


def _estimate_ringing_samples(b, a):
    """Helper function for determining IIR padding"""
    x = np.zeros(1000)
    x[0] = 1
    h = signal.lfilter(b, a, x)
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
    iir_params: dict
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
    f_pass: float or list of float
        Frequency for the pass-band. Low-pass and high-pass filters should
        be a float, band-pass should be a 2-element list of float.
    f_stop: float or list of float
        Stop-band frequency (same size as f_pass). Not used if 'order' is
        specified in iir_params.
    btype: str
        Type of filter. Should be 'lowpass', 'highpass', or 'bandpass'
        (or analogous string representations known to scipy.signal).
    return_copy: bool
        If False, the 'b', 'a', and 'padlen' entries in iir_params will be
        set inplace (if they weren't already). Otherwise, a new iir_params
        instance will be created and returned with these entries.

    Returns
    -------
    iir_params: dict
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
    >>> print (len(iir_params['b']), len(iir_params['a']), iir_params['padlen'])
    (5, 5, 82)

    Filters can also be constructed using filter design methods. To get a
    40 Hz Chebyshev type 1 lowpass with specific gain characteristics in the
    pass and stop bands (assuming the desired stop band is at 45 Hz), this
    would be a filter with much longer ringing:

    >>> iir_params = dict(ftype='cheby1', gpass=3, gstop=20)
    >>> iir_params = construct_iir_filter(iir_params, 40, 50, 1000, 'low')
    >>> print (len(iir_params['b']), len(iir_params['a']), iir_params['padlen'])
    (6, 6, 439)

    Padding and/or filter coefficients can also be manually specified. For
    a 10-sample moving window with no padding during filtering, for example,
    one can just do:

    >>> iir_params = dict(b=np.ones((10)), a=[1, 0], padlen=0)
    >>> iir_params = construct_iir_filter(iir_params, return_copy=False)
    >>> print (iir_params['b'], iir_params['a'], iir_params['padlen'])
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), [1, 0], 0)

    """
    a = None
    b = None
    # if the filter has been designed, we're good to go
    if 'a' in iir_params and 'b' in iir_params:
        [b, a] = [iir_params['b'], iir_params['a']]
    else:
        # ensure we have a valid ftype
        if not 'ftype' in iir_params:
            raise RuntimeError('ftype must be an entry in iir_params if ''b'' '
                               'and ''a'' are not specified')
        ftype = iir_params['ftype']
        if not ftype in filter_dict:
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
            if not 'gpass' in iir_params or not 'gstop' in iir_params:
                raise ValueError('iir_params must have at least ''gstop'' and'
                                 ' ''gpass'' (or ''N'') entries')
            [b, a] = iirdesign(Wp, Ws, iir_params['gpass'],
                               iir_params['gstop'], ftype=ftype)

    if a is None or b is None:
        raise RuntimeError('coefficients could not be created from iir_params')

    # now deal with padding
    if not 'padlen' in iir_params:
        padlen = _estimate_ringing_samples(b, a)
    else:
        padlen = iir_params['padlen']

    if return_copy:
        iir_params = deepcopy(iir_params)

    iir_params.update(dict(b=b, a=a, padlen=padlen))
    return iir_params


def band_pass_filter(x, Fs, Fp1, Fp2, filter_length=None,
                     l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                     method='fft', iir_params=dict(order=4, ftype='butter')):
    """Bandpass filter for the signal x.

    Applies a zero-phase bandpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        Sampling rate in Hz
    Fp1 : float
        Low cut-off frequency in Hz
    Fp2 : float
        High cut-off frequency in Hz
    filter_length : int (default: None)
        Length of the filter to use. If None or "len(x) < filter_length", the
        filter length used is len(x). Otherwise, overlap-add filtering with a
        filter of the specified length is used (faster for long signals).
    l_trans_bandwidth : float
        Width of the transition band at the low cut-off frequency in Hz.
    h_trans_bandwidth : float
        Width of the transition band at the high cut-off frequency in Hz.
    method: str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params: dict
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details.

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The frequency response is (approximately) given by
                     ----------
                   /|         | \
                  / |         |  \
                 /  |         |   \
                /   |         |    \
      ----------    |         |     -----------------
                    |         |
              Fs1  Fp1       Fp2   Fs2

    Where
    Fs1 = Fp1 - l_trans_bandwidth in Hz
    Fs2 = Fp2 + h_trans_bandwidth in Hz
    """

    method = method.lower()
    if method not in ['fft', 'iir']:
        raise RuntimeError('method should be fft or iir (not %s)' % method)

    Fs = float(Fs)
    Fp1 = float(Fp1)
    Fp2 = float(Fp2)
    Fs1 = Fp1 - l_trans_bandwidth
    Fs2 = Fp2 + h_trans_bandwidth

    if Fs1 <= 0:
        raise ValueError('Filter specification invalid: Lower stop frequency '
                         'too low (%0.1fHz). Increase Fp1 or reduce '
                         'transition bandwidth (l_trans_bandwidth)' % Fs1)

    if method == 'fft':
        xf = _filter(x, Fs, [0, Fs1, Fp1, Fp2, Fs2, Fs / 2],
                     [0, 0, 1, 1, 0, 0], filter_length)
    else:
        iir_params = construct_iir_filter(iir_params, [Fp1, Fp2],
                                          [Fs1, Fs2], Fs, 'bandpass')
        padlen = min(iir_params['padlen'], len(x))
        xf = filtfilt(iir_params['b'], iir_params['a'], x, padlen=padlen)

    return xf


def low_pass_filter(x, Fs, Fp, filter_length=None, trans_bandwidth=0.5,
                    method='fft', iir_params=dict(order=4, ftype='butter')):
    """Lowpass filter for the signal x.

    Applies a zero-phase lowpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        Sampling rate in Hz
    Fp : float
        Cut-off frequency in Hz
    filter_length : int (default: None)
        Length of the filter to use. If None or "len(x) < filter_length", the
        filter length used is len(x). Otherwise, overlap-add filtering with a
        filter of the specified length is used (faster for long signals).
    trans_bandwidth : float
        Width of the transition band in Hz.
    method: str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params: dict
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details.

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The frequency response is (approximately) given by
      -------------------------
                              | \
                              |  \
                              |   \
                              |    \
                              |     -----------------
                              |
                              Fp  Fp+trans_bandwidth

    """

    method = method.lower()
    if method not in ['fft', 'iir']:
        raise RuntimeError('method should be fft or iir (not %s)' % method)

    Fs = float(Fs)
    Fp = float(Fp)
    Fstop = Fp + trans_bandwidth
    if method == 'fft':
        xf = _filter(x, Fs, [0, Fp, Fstop, Fs / 2], [1, 1, 0, 0],
                     filter_length)
    else:
        iir_params = construct_iir_filter(iir_params, Fp, Fstop, Fs, 'low')
        padlen = min(iir_params['padlen'], len(x))
        xf = filtfilt(iir_params['b'], iir_params['a'], x, padlen=padlen)

    return xf


def high_pass_filter(x, Fs, Fp, filter_length=None, trans_bandwidth=0.5,
                     method='fft', iir_params=dict(order=4, ftype='butter')):
    """Highpass filter for the signal x.

    Applies a zero-phase highpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        Sampling rate in Hz
    Fp : float
        Cut-off frequency in Hz
    filter_length : int (default: None)
        Length of the filter to use. If None or "len(x) < filter_length", the
        filter length used is len(x). Otherwise, overlap-add filtering with a
        filter of the specified length is used (faster for long signals).
    trans_bandwidth : float
        Width of the transition band in Hz.
    method: str
        'fft' will use overlap-add FIR filtering, 'iir' will use IIR
        forward-backward filtering (via filtfilt).
    iir_params: dict
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details.

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The frequency response is (approximately) given by
                   -----------------------
                 /|
                / |
               /  |
              /   |
    ----------    |
                  |
           Fstop  Fp

    where Fstop = Fp - trans_bandwidth
    """

    method = method.lower()
    if method not in ['fft', 'iir']:
        raise RuntimeError('method should be fft or iir (not %s)' % method)

    Fs = float(Fs)
    Fp = float(Fp)

    Fstop = Fp - trans_bandwidth
    if Fstop <= 0:
        raise ValueError('Filter specification invalid: Stop frequency too low'
                         '(%0.1fHz). Increase Fp or reduce transition '
                         'bandwidth (trans_bandwidth)' % Fstop)

    if method == 'fft':
        xf = _filter(x, Fs, [0, Fstop, Fp, Fs / 2], [0, 0, 1, 1],
                     filter_length)
    else:
        iir_params = construct_iir_filter(iir_params, Fp, Fstop, Fs, 'high')
        padlen = min(iir_params['padlen'], len(x))
        xf = filtfilt(iir_params['b'], iir_params['a'], x, padlen=padlen)

    return xf


def resample(x, up, down, npad=100, axis=0, window='boxcar'):
    """Resample the array x.

    Parameters
    ----------
    x: n-d array
        Signal to resample.
    up: float
        Factor to upsample by.
    down: float
        Factor to downsample by.
    npad: integer
        Number of samples to use at the beginning and end for padding.
    axis: integer
        Axis of the array to operate on.
    window: string or tuple
        See scipy.signal.resample for description.

    Returns
    -------
    xf: array
        x filtered.

    Notes
    -----
    This uses (hopefully) intelligent edge padding and frequency-domain
    windowing improve scipy.signal.resample's resampling. Choices of
    npad and window have important consequences, and these choices should
    work well for most natural signals.

    Resampling arguments are broken into "up" and "down" components for future
    compatibility in case we decide to use an upfirdn implementation. The
    current implementation is functionally equivalent to passing
    up=up/down and down=1.

    """
    # make sure our arithmetic will work
    ratio = float(up) / down

    if axis > len(x.shape):
        raise ValueError('x does not have %d axes' % axis)

    x_len = x.shape[axis]
    if x_len > 0:
        # add some padding at beginning and end to make scipy's FFT
        # method work a little cleaner
        pad_shape = np.array(x.shape, dtype=np.int)
        pad_shape[axis] = npad
        keep = np.zeros(x_len, dtype='bool')
        # pad both ends because signal is not assumed periodic
        keep[0] = True
        pad = np.ones(pad_shape) * np.compress(keep, x, axis=axis)
        # do the padding
        x_padded = np.concatenate((pad, x, pad), axis=axis)
        new_len = ratio * x_padded.shape[axis]

        # do the resampling using scipy's FFT-based resample function
        # use of the 'flat' window is recommended for minimal ringing
        y = signal.resample(x_padded, new_len, axis=axis, window=window)

        # now let's trim it back to the correct size (if there was padding)
        to_remove = np.round(ratio * npad)
        if to_remove > 0:
            keep = np.ones((new_len), dtype='bool')
            keep[:to_remove] = False
            keep[-to_remove:] = False
            y = np.compress(keep, y, axis=axis)
    else:
        warnings.warn('x has zero length along axis=%d, returning a copy of '
                      'x' % axis)
        y = x.copy()
    return y
