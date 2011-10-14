import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


def overlap_add_filter(x, h, N_fft=None, zero_phase=True):
    """ Calculate linear convolution using overlap-add FFTs

    Implements the linear convolution between x and h using overlap-add FFTs.

    Parameters
    ----------
    x : 1d array
        Signal to filter
    h : 1d array
        Filter impule response (FIR filter coefficients)
    N_fft : int
        Length of the FFT. If None, the best size is determined automatically.
    """
    
    # https://ccrma.stanford.edu/~jos/fp/Forward_Backward_Filtering.html
    # http://www.mathworks.com/matlabcentral/fileexchange/17061-filtfilthd/content/filtfilthd.m
    N_h = len(h)
    N_x = len(x)
    
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
            
    print 'FFT length: %d' % (N_fft)        
    
    N_seg = N_fft - N_h + 1

    if N_fft <= 0:
        raise ValueError('N_fft is too short, has to be at least len(h)')

    # Filter in frequency domain    
    h_fft = fft(np.concatenate((h, np.zeros(N_fft - N_h)))) 
    
    if zero_phase:
        # We will apply the filter in forward and backward direction: Scale 
        # frequency response of the filter so that the shape of the amplitude
        # response stays the same when it is applied twice
        
        # be careful not to divide by too small numbers
        #idx = np.where(np.abs(h_fft) > 1e-6)
        #h_fft[idx] = h_fft[idx] / np.sqrt(np.abs(h_fft[idx])) 
        pass
    
    x_filtered = np.zeros_like(x)

    # Number of segements (including fractional segments)
    num_segments = int(np.ceil(N_x / float(N_seg)))
    
    filter_input = x

    for pass_no in range(2) if zero_phase else range(1):
        
        if pass_no == 1:
            # second pass: flip signal
            filter_input = x_filtered[::-1]
            x_filtered = np.zeros_like(x)
            
        for seg_ind in range(num_segments):
            seg = filter_input[seg_ind*N_seg:(seg_ind+1)*N_seg]
            seg_fft = fft(np.concatenate((seg, np.zeros(N_fft - len(seg)))))
    
    
            if seg_ind*N_seg+N_fft < N_x:                               
                x_filtered[seg_ind*N_seg:seg_ind*N_seg+N_fft] \
                    += np.real(ifft(h_fft * seg_fft))            
            else:
                # Last segment
                x_filtered[seg_ind*N_seg:] \
                    += np.real(ifft(h_fft * seg_fft))[:N_x-seg_ind*N_seg]
    if zero_phase:
        # flip signal back
        x_filtered = x_filtered[::-1]
        
    return x_filtered           
        
    

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
    Fp1 = float(Fp1)
    Fp2 = float(Fp2)

    # Default values in Hz
    Fs1 = Fp1 - 0.5
    Fs2 = Fp2 + 0.5

    assert x.ndim == 1

    # Make x EVEN
    Norig = len(x)
    if Norig % 2 == 1:
        x = np.r_[x, 1]

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


def low_pass_filter(x, Fs, Fp):
    """Lowpass filter for the signal x.

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
    Fp = float(Fp)

    assert x.ndim == 1

    # Make x EVEN
    Norig = len(x)
    if Norig % 2 == 1:
        x = np.r_[x, 1]

    # Normalize frequencies
    Ns = (Fp + 0.5) / (Fs / 2)
    Np = Fp / (Fs / 2)

    # Construct the filter function H(f)
    N = len(x)

    B = signal.firwin2(N, [0, Np, Ns, 1], [1, 1, 0, 0])

    # Make zero-phase filter function
    H = np.abs(fft(B))

    xf = np.real(ifft(fft(x) * H))
    xf = xf[:Norig]
    x = x[:Norig]

    return xf


def high_pass_filter(x, Fs, Fp):
    """Highpass filter for the signal x.

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
    Fp = float(Fp)

    assert x.ndim == 1

    # Make x ODD
    Norig = len(x)
    if Norig % 2 == 0:
        x = np.r_[x, 1]

    # Normalize frequencies
    Ns = (Fp - 0.5) / (Fs / 2)
    Np = Fp / (Fs / 2)

    # Construct the filter function H(f)
    N = len(x)

    B = signal.firwin2(N, [0, Ns, Np, 1], [0, 0, 1, 1])

    # Make zero-phase filter function
    H = np.abs(fft(B))

    xf = np.real(ifft(fft(x) * H))
    xf = xf[:Norig]
    x = x[:Norig]

    return xf
