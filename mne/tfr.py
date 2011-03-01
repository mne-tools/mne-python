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


def _cwt_morlet_fft(x, Fs, freqs, mode="same", Ws=None):
    """Compute cwt with fft based convolutions
    """
    x = np.asarray(x)
    freqs = np.asarray(freqs)

    # Precompute wavelets for given frequency range to save time
    n_samples = x.size
    n_freqs = freqs.size

    if Ws is None:
        Ws = morlet(Fs, freqs)

    Ws_max_size = max(W.size for W in Ws)
    size = n_samples + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    fft_x = fftn(x, [fsize])

    if mode == "full":
        tfr = np.zeros((n_freqs, fsize), dtype=np.complex128)
    elif mode == "same" or mode == "valid":
        tfr = np.zeros((n_freqs, n_samples), dtype=np.complex128)

    for i, W in enumerate(Ws):
        ret = ifftn(fft_x * fftn(W, [fsize]))[:n_samples + W.size - 1]
        if mode == "valid":
            sz = abs(W.size - n_samples) + 1
            offset = (n_samples - sz) / 2
            tfr[i, offset:(offset + sz)] = _centered(ret, sz)
        else:
            tfr[i] = _centered(ret, n_samples)
    return tfr


def _cwt_morlet_convolve(x, Fs, freqs, mode='same', Ws=None):
    """Compute time freq decomposition with temporal convolutions
    """
    x = np.asarray(x)
    freqs = np.asarray(freqs)

    if Ws is None:
        Ws = morlet(Fs, freqs)

    n_samples = x.size
    # Compute convolutions
    tfr = np.zeros((freqs.size, len(x)), dtype=np.complex128)
    for i, W in enumerate(Ws):
        ret = np.convolve(x, W, mode=mode)
        if mode == "valid":
            sz = abs(W.size - n_samples) + 1
            offset = (n_samples - sz) / 2
            tfr[i, offset:(offset + sz)] = ret
        else:
            tfr[i] = ret
    return tfr


def cwt_morlet(x, Fs, freqs, use_fft=True, n_cycles=7.0):
    """Compute time freq decomposition with Morlet wavelets

    Parameters
    ----------
    x : array
        signal

    Fs : float
        sampling Frequency

    freqs : array
        Array of frequencies of interest

    Returns
    -------
    tfr : 2D array
        Time Frequency Decomposition (Frequencies x Timepoints)
    """
    mode = 'same'
    # mode = "valid"

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, freqs, n_cycles=n_cycles)

    if use_fft:
        return _cwt_morlet_fft(x, Fs, freqs, mode, Ws)
    else:
        return _cwt_morlet_convolve(x, Fs, freqs, mode, Ws)


def _time_frequency_one_channel(epochs, c, Fs, frequencies, use_fft, n_cycles):
    """Aux of time_frequency for parallel computing"""
    n_epochs, _, n_times = epochs.shape
    n_frequencies = len(frequencies)
    psd_c = np.zeros((n_frequencies, n_times)) # PSD
    plf_c = np.zeros((n_frequencies, n_times), dtype=np.complex) # phase lock

    for e in range(n_epochs):
        tfr = cwt_morlet(epochs[e, c, :].ravel(), Fs, frequencies,
                                  use_fft=use_fft, n_cycles=n_cycles)
        tfr_abs = np.abs(tfr)
        psd_c += tfr_abs**2
        plf_c += tfr / tfr_abs
    return psd_c, plf_c


def time_frequency(epochs, Fs, frequencies, use_fft=True, n_cycles=25,
                   n_jobs=1):
    """Compute time induced power and inter-trial phase-locking factor

    The time frequency decomposition is done with Morlet wavelets

    Parameters
    ----------
    epochs : array
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
    n_epochs, n_channels, n_times = epochs.shape

    try:
        import joblib
    except ImportError:
        print "joblib not installed. Cannot run in parallel."
        n_jobs = 1

    if n_jobs == 1:
        psd = np.empty((n_channels, n_frequencies, n_times))
        plf = np.empty((n_channels, n_frequencies, n_times), dtype=np.complex)

        for c in range(n_channels):
            psd[c,:,:], plf[c,:,:] = _time_frequency_one_channel(epochs, c, Fs,
                                                frequencies, use_fft, n_cycles)
    else:
        from joblib import Parallel, delayed
        psd_plf = Parallel(n_jobs=n_jobs)(
                    delayed(_time_frequency_one_channel)(
                            epochs, c, Fs, frequencies, use_fft, n_cycles)
                    for c in range(n_channels))

        psd = np.zeros((n_channels, n_frequencies, n_times))
        plf = np.zeros((n_channels, n_frequencies, n_times), dtype=np.complex)
        for c, (psd_c, plf_c) in enumerate(psd_plf):
            psd[c,:,:], plf[c,:,:] = psd_c, plf_c

    psd /= n_epochs
    plf = np.abs(plf) / n_epochs
    return psd, plf
