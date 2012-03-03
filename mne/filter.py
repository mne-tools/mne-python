import warnings
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


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
    assert x.ndim == 1

    # normalize frequencies
    freq = [f / (Fs / 2) for f in freq]

    if filter_length is None or len(x) <= filter_length:
        # Use direct FFT filtering for short signals

        Norig = len(x)

        if (gain[-1] == 0.0 and Norig % 2 == 1) \
                or (gain[-1] == 1.0 and Norig % 2 != 1):
            # Gain at Nyquist freq: 1: make x EVEN, 0: make x ODD
            x = np.r_[x, x[-1]]

        N = len(x)

        H = signal.firwin2(N, freq, gain)

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

        H = signal.firwin2(N, freq, gain)
        xf = _overlap_add_filter(x, H, zero_phase=True)

    return xf


def band_pass_filter(x, Fs, Fp1, Fp2, filter_length=None):
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

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The passbands (Fp1 Fp2) frequencies are defined in Hz as
                     ----------
                   /|         | \
                  / |         |  \
                 /  |         |   \
                /   |         |    \
      ----------    |         |     -----------------
                    |         |
              Fs1  Fp1       Fp2   Fs2

    DEFAULTS values
    Fs1 = Fp1 - 0.5 in Hz
    Fs2 = Fp2 + 0.5 in Hz
    """
    Fs = float(Fs)
    Fp1 = float(Fp1)
    Fp2 = float(Fp2)

    # Default values in Hz
    Fs1 = Fp1 - 0.5
    Fs2 = Fp2 + 0.5

    xf = _filter(x, Fs, [0, Fs1, Fp1, Fp2, Fs2, Fs / 2], [0, 0, 1, 1, 0, 0],
                 filter_length)

    return xf


def low_pass_filter(x, Fs, Fp, filter_length=None):
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

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The passbands (Fp1 Fp2) frequencies are defined in Hz as
      -------------------------
                              | \
                              |  \
                              |   \
                              |    \
                              |     -----------------
                              |
                              Fp  Fp+0.5

    """
    Fs = float(Fs)
    Fp = float(Fp)

    Fstop = Fp + 0.5

    xf = _filter(x, Fs, [0, Fp, Fstop, Fs / 2], [1, 1, 0, 0], filter_length)

    return xf


def high_pass_filter(x, Fs, Fp, filter_length=None):
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

    Returns
    -------
    xf : array
        x filtered

    Notes
    -----
    The passbands (Fp1 Fp2) frequencies are defined in Hz as
                   -----------------------
                 /|
                / |
               /  |
              /   |
    ----------    |
                  |
          Fp-0.5  Fp

    """

    Fs = float(Fs)
    Fp = float(Fp)

    Fstop = Fp - 0.5

    xf = _filter(x, Fs, [0, Fstop, Fp, Fs / 2], [0, 0, 1, 1], filter_length)

    return xf
