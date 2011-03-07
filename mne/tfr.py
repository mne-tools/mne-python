"""A module which implements the continuous wavelet transform
with complex Morlet wavelets.

Author : Alexandre Gramfort, gramfort@nmr.mgh.harvard.edu (2011)
License : BSD 3-clause

inspired by Matlab code from Sheraz Khan & Brainstorm & SPM
"""

from math import sqrt
import numpy as np
from scipy import linalg
from scipy.fftpack import fftn, ifftn


def morlet(Fs, freqs, n_cycles=7, sigma=None):
    """Compute Wavelets for the given frequency range

    Parameters
    ----------
    Fs : float
        Sampling Frequency

    freqs : array
        frequency range of interest (1 x Frequencies)

    n_cycles : float
        Number of oscillations if wavelet

    sigma : float, (optional)
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.

    Returns
    -------
    Ws : list of array
        Wavelets time series
    """
    Ws = list()
    for f in freqs:
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0, 5*sigma_t, 1.0 / Fs)
        t = np.r_[-t[::-1], t[1:]]
        W = np.exp(2.0 * 1j * np.pi * f *t)
        W *= np.exp(-t**2 / (2.0 * sigma_t**2))
        W /= sqrt(0.5) * linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _cwt_fft(X, Ws, mode="same"):
    """Compute cwt with fft based convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))

    # precompute FFTs of Ws
    fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        fft_Ws[i] = fftn(W, [fsize])

    for k, x in enumerate(X):
        if mode == "full":
            tfr = np.zeros((n_freqs, fsize), dtype=np.complex128)
        elif mode == "same" or mode == "valid":
            tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)

        fft_x = fftn(x, [fsize])
        for i, W in enumerate(Ws):
            ret = ifftn(fft_x * fft_Ws[i])[:n_times + W.size - 1]
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = _centered(ret, sz)
            else:
                tfr[i, :] = _centered(ret, n_times)
        yield tfr


def _cwt_convolve(X, Ws, mode='same'):
    """Compute time freq decomposition with temporal convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    # Compute convolutions
    for x in X:
        tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)
        for i, W in enumerate(Ws):
            ret = np.convolve(x, W, mode=mode)
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = ret
            else:
                tfr[i] = ret
        yield tfr


def cwt_morlet(X, Fs, freqs, use_fft=True, n_cycles=7.0):
    """Compute time freq decomposition with Morlet wavelets

    Parameters
    ----------
    X : array of shape [n_signals, n_times]
        signals (one per line)

    Fs : float
        sampling Frequency

    freqs : array
        Array of frequencies of interest

    Returns
    -------
    tfr : 3D array
        Time Frequency Decompositions (n_signals x n_frequencies x n_times)
    """
    mode = 'same'
    # mode = "valid"
    n_signals, n_times = X.shape
    n_frequencies = len(freqs)

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, freqs, n_cycles=n_cycles)

    if use_fft:
        coefs = _cwt_fft(X, Ws, mode)
    else:
        coefs = _cwt_convolve(X, Ws, mode)

    tfrs = np.empty((n_signals, n_frequencies, n_times))
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs

def _time_frequency(X, Ws, use_fft):
    """Aux of time_frequency for parallel computing over channels
    """
    n_epochs, n_times = X.shape
    n_frequencies = len(Ws)
    psd = np.zeros((n_frequencies, n_times)) # PSD
    plf = np.zeros((n_frequencies, n_times), dtype=np.complex) # phase lock

    mode = 'same'
    if use_fft:
        tfrs = _cwt_fft(X, Ws, mode)
    else:
        tfrs = _cwt_convolve(X, Ws, mode)

    for tfr in tfrs:
        tfr_abs = np.abs(tfr)
        psd += tfr_abs**2
        plf += tfr / tfr_abs

    return psd, plf

def single_trial_power(epochs, Fs, frequencies, use_fft=True, n_cycles=25,
                           n_jobs=1):
    """Compute time-frequency power on single epochs
    """
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = epochs.shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles)

    power = np.empty((n_epochs, n_channels, n_frequencies, n_times),
                     dtype=np.float)

    mode = 'same'
    if use_fft:
        _cwt = _cwt_fft
    else:
        _cwt = _cwt_convolve

    for k, e in enumerate(epochs):
        mode = 'same'
        power[k] = np.abs(_cwt(e, Ws, mode))**2

    return power

def time_frequency(data, Fs, frequencies, use_fft=True, n_cycles=25,
                   n_jobs=1):
    """Compute time induced power and inter-trial phase-locking factor

    The time frequency decomposition is done with Morlet wavelets

    Parameters
    ----------
    data : array
        3D array of shape [n_epochs, n_channels, n_times]

    Fs : float
        sampling Frequency

    frequencies : array
        Array of frequencies of interest

    use_fft : bool
        Compute transform with fft based convolutions or temporal
        convolutions.

    n_cycles : int
        The number of cycles in the wavelet

    n_jobs : int
        The number of CPUs used in parallel. All CPUs are used in -1.
        Requires joblib package.

    Returns
    -------
    power : 2D array
        Induced power (Channels x Frequencies x Timepoints).
        Squared amplitude of time-frequency coefficients.
    phase_lock : 2D array
        Phase locking factor in [0, 1] (Channels x Frequencies x Timepoints)
    """
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data.shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles)

    try:
        import joblib
    except ImportError:
        print "joblib not installed. Cannot run in parallel."
        n_jobs = 1

    if n_jobs == 1:
        psd = np.empty((n_channels, n_frequencies, n_times))
        plf = np.empty((n_channels, n_frequencies, n_times), dtype=np.complex)

        for c in range(n_channels):
            X = np.squeeze(data[:,c,:])
            psd[c], plf[c] = _time_frequency(X, Ws, use_fft)

    else:
        from joblib import Parallel, delayed
        psd_plf = Parallel(n_jobs=n_jobs)(
                    delayed(_time_frequency)(
                            np.squeeze(data[:,c,:]), Ws, use_fft)
                    for c in range(n_channels))

        psd = np.zeros((n_channels, n_frequencies, n_times))
        plf = np.zeros((n_channels, n_frequencies, n_times), dtype=np.complex)
        for c, (psd_c, plf_c) in enumerate(psd_plf):
            psd[c,:,:], plf[c,:,:] = psd_c, plf_c

    psd /= n_epochs
    plf = np.abs(plf) / n_epochs
    return psd, plf
