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
from ..baseline import rescale
from ..parallel import parallel_func


def morlet(Fs, freqs, n_cycles=7, sigma=None, zero_mean=False):
    """Compute Wavelets for the given frequency range

    Parameters
    ----------
    Fs : float
        Sampling Frequency

    freqs : array
        frequency range of interest (1 x Frequencies)

    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.

    sigma : float, (optional)
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.

    zero_mean : bool
        Make sure the wavelet is zero mean

    Returns
    -------
    Ws : list of array
        Wavelets time series
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)
    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0, 5 * sigma_t, 1.0 / Fs)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
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
    fsize = 2 ** np.ceil(np.log2(size))

    # precompute FFTs of Ws
    fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('Wavelet is too long for such a short signal. '
                             'Reduce the number of cycles.')
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
            if len(W) > len(x):
                raise ValueError('Wavelet is too long for such a short '
                                 'signal. Reduce the number of cycles.')
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = ret
            else:
                tfr[i] = ret
        yield tfr


def cwt_morlet(X, Fs, freqs, use_fft=True, n_cycles=7.0, zero_mean=False):
    """Compute time freq decomposition with Morlet wavelets

    Parameters
    ----------
    X : array of shape [n_signals, n_times]
        signals (one per line)
    Fs : float
        sampling Frequency
    freqs : array
        Array of frequencies of interest
    use_fft : bool
        Compute convolution with FFT or temoral convolution.
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    zero_mean : bool
        Make sure the wavelets are zero mean.

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
    Ws = morlet(Fs, freqs, n_cycles=n_cycles, zero_mean=zero_mean)

    if use_fft:
        coefs = _cwt_fft(X, Ws, mode)
    else:
        coefs = _cwt_convolve(X, Ws, mode)

    tfrs = np.empty((n_signals, n_frequencies, n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def cwt(X, Ws, use_fft=True, mode='same'):
    """Compute time freq decomposition with continuous wavelet transform

    Parameters
    ----------
    X : array of shape [n_signals, n_times]
        signals (one per line)
    Ws : list of array
        Wavelets time series
    use_fft : bool
        Use FFT for convolutions
    mode : 'same' | 'valid' | 'full'
        Convention for convolution

    Returns
    -------
    tfr : 3D array
        Time Frequency Decompositions (n_signals x n_frequencies x n_times)
    """
    n_signals, n_times = X.shape
    n_frequencies = len(Ws)

    if use_fft:
        coefs = _cwt_fft(X, Ws, mode)
    else:
        coefs = _cwt_convolve(X, Ws, mode)

    tfrs = np.empty((n_signals, n_frequencies, n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def _time_frequency(X, Ws, use_fft):
    """Aux of time_frequency for parallel computing over channels
    """
    n_epochs, n_times = X.shape
    n_frequencies = len(Ws)
    psd = np.zeros((n_frequencies, n_times))  # PSD
    plf = np.zeros((n_frequencies, n_times), dtype=np.complex)  # phase lock

    mode = 'same'
    if use_fft:
        tfrs = _cwt_fft(X, Ws, mode)
    else:
        tfrs = _cwt_convolve(X, Ws, mode)

    for tfr in tfrs:
        tfr_abs = np.abs(tfr)
        psd += tfr_abs ** 2
        plf += tfr / tfr_abs

    return psd, plf


def single_trial_power(data, Fs, frequencies, use_fft=True, n_cycles=7,
                       baseline=None, baseline_mode='ratio', times=None,
                       n_jobs=1, zero_mean=False):
    """Compute time-frequency power on single epochs

    Parameters
    ----------
    data : array of shape [n_epochs, n_channels, n_times]
        The epochs
    Fs : float
        Sampling rate
    frequencies : array-like
        The frequencies
    use_fft : bool
        Use the FFT for convolutions or not.
    n_cycles: float | array of float
        Number of cycles  in the Morlet wavelet. Fixed number
        or one per frequency.
    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'ratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    times : array
        Required to define baseline
    n_jobs : int
        The number of epochs to process at the same time
    zero_mean : bool
        Make sure the wavelets are zero mean.

    Returns
    -------
    power : list of 2D array
        Each element of the list the the power estimate for an epoch.
    """
    mode = 'same'
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data.shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    parallel, my_cwt, _ = parallel_func(cwt, n_jobs)

    print "Computing time-frequency power on single epochs..."

    power = np.empty((n_epochs, n_channels, n_frequencies, n_times),
                     dtype=np.float)
    if n_jobs == 1:
        for k, e in enumerate(data):
            power[k] = np.abs(cwt(e, Ws, mode)) ** 2
    else:
        # Precompute tf decompositions in parallel
        tfrs = parallel(my_cwt(e, Ws, use_fft, mode) for e in data)
        for k, tfr in enumerate(tfrs):
            power[k] = np.abs(tfr) ** 2

    # Run baseline correction
    power = rescale(power, times, baseline, baseline_mode,
                    verbose=True, copy=False)
    return power


def induced_power(data, Fs, frequencies, use_fft=True, n_cycles=7,
                  decim=1, n_jobs=1, zero_mean=False):
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
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim: int
        Temporal decimation factor
    n_jobs : int
        The number of CPUs used in parallel. All CPUs are used in -1.
        Requires joblib package.
    zero_mean : bool
        Make sure the wavelets are zero mean.

    Returns
    -------
    power : 2D array
        Induced power (Channels x Frequencies x Timepoints).
        Squared amplitude of time-frequency coefficients.
    phase_lock : 2D array
        Phase locking factor in [0, 1] (Channels x Frequencies x Timepoints)
    """
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data[:, :, ::decim].shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    if n_jobs == 1:
        psd = np.empty((n_channels, n_frequencies, n_times))
        plf = np.empty((n_channels, n_frequencies, n_times), dtype=np.complex)

        for c in range(n_channels):
            X = np.squeeze(data[:, c, :])
            this_psd, this_plf = _time_frequency(X, Ws, use_fft)
            psd[c], plf[c] = this_psd[:, ::decim], this_plf[:, ::decim]
    else:
        parallel, my_time_frequency, _ = parallel_func(_time_frequency, n_jobs)

        psd_plf = parallel(my_time_frequency(np.squeeze(data[:, c, :]),
                                             Ws, use_fft)
                           for c in range(n_channels))

        psd = np.zeros((n_channels, n_frequencies, n_times))
        plf = np.zeros((n_channels, n_frequencies, n_times), dtype=np.complex)
        for c, (psd_c, plf_c) in enumerate(psd_plf):
            psd[c, :, :], plf[c, :, :] = psd_c[:, ::decim], plf_c[:, ::decim]

    psd /= n_epochs
    plf = np.abs(plf) / n_epochs
    return psd, plf
