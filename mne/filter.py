import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


def band_pass_filter(x, Fs, Fp1, Fp2):
    """Bandpass filter for the signal x.

    An acausal fft algorithm is applied (i.e. no phase shift). The filter
    functions is constructed from a Hamming window (window used in "firwin2"
    function) to avoid ripples in the frequency reponse (windowing is a
    smoothing in frequency domain)

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

    # Default values in Hz
    Fs1 = Fp1 - 0.5
    Fs2 = Fp2 + 0.5

    assert x.ndim == 1

    # Make x EVEN
    Norig = len(x)
    if Norig % 2 == 1:
        x = np.c_[x, 1]

    # Normalize frequencies
    Ns1 = Fs1 / (Fs / 2)
    Ns2 = Fs2 / (Fs / 2)
    Np1 = Fp1 / (Fs / 2)
    Np2 = Fp2 / (Fs / 2)

    # Construct the filter function H(f)
    N = len(x)

    B = signal.firwin2(N, [0, Ns1, Np1, Np2, Ns2, 1], [0, 0, 1, 1, 0, 0])

    # Make zero-phase filter function
    H = np.abs(fft(B))

    xf = np.real(ifft(fft(x) * H))
    xf = xf[:Norig]
    x = x[:Norig]

    return xf
