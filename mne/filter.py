import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


def _overlap_add_filter(x, h, N_fft=None, zero_phase=True):
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
        Filter impule response (FIR filter coefficients)
    N_fft : int
        Length of the FFT. If None, the best size is determined automatically.
    zero_phase : boolean
        If True: the filter is applied in forward and backward direction,
        resulting in a zero-phase filter

    Returns
    -------
    xf : 1d array
        x filtered
    """

    N_h = len(h)

    # Extend the signal by mirroring the edges to reduce transient filter
    # response
    N_edge = min(N_h, len(x))

    x_ext = np.r_[2*x[0]-x[N_edge-1:0:-1], x, 2*x[-1]-x[-2:-N_edge-1:-1]]

    N_x = len(x_ext)

    # Determine FFT length to use
    if N_fft == None:
        if N_x > N_h:
            N_tot = 2*N_x if zero_phase else N_x

            N = 2**np.arange(np.ceil(np.log2(N_h)),
                             np.floor(np.log2(N_tot)))
            cost = np.ceil(N_tot / (N - N_h + 1)) * N  * (np.log2(N) + 1)
            N_fft = N[np.argmin(cost)]
        else:
            # Use only a single block
            N_fft = 2**np.ceil(np.log2(N_x + N_h - 1))

    if N_fft <= 0:
        raise ValueError('N_fft is too short, has to be at least len(h)')

    # Filter in frequency domain
    h_fft = fft(np.r_[h, np.zeros(N_fft - N_h)])

    if zero_phase:
        # We will apply the filter in forward and backward direction: Scale 
        # frequency response of the filter so that the shape of the amplitude
        # response stays the same when it is applied twice

        # be careful not to divide by too small numbers
        idx = np.where(np.abs(h_fft) > 1e-6)
        h_fft[idx] = h_fft[idx] / np.sqrt(np.abs(h_fft[idx]))

    x_filtered = np.zeros_like(x_ext)

    # Segment length for signal x
    N_seg = N_fft - N_h + 1

    # Number of segements (including fractional segments)
    num_segments = int(np.ceil(N_x / float(N_seg)))

    filter_input = x_ext

    for pass_no in range(2) if zero_phase else range(1):

        if pass_no == 1:
            # second pass: flip signal
            filter_input = np.flipud(x_filtered)
            x_filtered = np.zeros_like(x_ext)

        for seg_ind in range(num_segments):
            seg = filter_input[seg_ind*N_seg:(seg_ind+1)*N_seg]
            seg_fft = fft(np.r_[seg, np.zeros(N_fft - len(seg))])

            if seg_ind*N_seg+N_fft < N_x:
                x_filtered[seg_ind*N_seg:seg_ind*N_seg+N_fft] \
                    += np.real(ifft(h_fft * seg_fft))
            else:
                # Last segment
                x_filtered[seg_ind*N_seg:] \
                    += np.real(ifft(h_fft * seg_fft))[:N_x-seg_ind*N_seg]

    # Remove mirrored edges that we added
    x_filtered = x_filtered[N_edge-1:-N_edge+1]

    if zero_phase:
        # flip signal back
        x_filtered = np.flipud(x_filtered)

    return x_filtered


def _filter(x, Fs, freq, gain):
    """ Filter signal using gain control points in the frequency domain.

    The filter impulse response is constructed from a Hamming window (window
    used in "firwin2" function) to avoid ripples in the frequency reponse
    (windowing is a smoothing in frequency domain).

    If the signal is shorter than 1 minute, it is filtered directly using
    FFTs (zero-phase). If the signal is longer, zero-phase overlap-add
    filtering is used.

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

    Returns
    -------
    xf : 1d array
        x filtered
    """
    assert x.ndim == 1

    # normalize frequencies
    freq = [f / (Fs / 2) for f in freq]

    if len(x) < 60*Fs:
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
        xf = xf[:Norig]
        x = x[:Norig]
    else:
        # Use overlap-add filter
        N = int(10 * Fs)

        if (gain[-1] == 0.0 and N % 2 == 1) \
            or (gain[-1] == 1.0 and N % 2 != 1):
            # Gain at Nyquist freq: 1: make N EVEN, 0: make N ODD
            N += 1

        H = signal.firwin2(N, freq, gain)
        xf = _overlap_add_filter(x, H, zero_phase=True)

    return xf


def band_pass_filter(x, Fs, Fp1, Fp2):
    """Bandpass filter for the signal x.

    Applies a zero-phase bandpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        sampling rate
    Fp1 : float
        low cut-off frequency
    Fp2 : float
        high cut-off frequency

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
    Fs  = float(Fs)
    Fp1 = float(Fp1)
    Fp2 = float(Fp2)

    # Default values in Hz
    Fs1 = Fp1 - 0.5
    Fs2 = Fp2 + 0.5

    xf = _filter(x, Fs, [0, Fs1, Fp1, Fp2, Fs2, Fs/2], [0, 0, 1, 1, 0, 0])

    return xf


def low_pass_filter(x, Fs, Fp):
    """Lowpass filter for the signal x.

    Applies a zero-phase lowpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        sampling rate
    Fp : float
        cut-off frequency

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

    xf = _filter(x, Fs, [0, Fp, Fstop, Fs/2], [1, 1, 0, 0])

    return xf


def high_pass_filter(x, Fs, Fp):
    """Highpass filter for the signal x.

    Applies a zero-phase highpass filter to the signal x.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    Fs : float
        sampling rate
    Fp : float
        cut-off frequency

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

    xf = _filter(x, Fs, [0, Fstop, Fp, Fs/2], [0, 0, 1, 1])

    return xf
