# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#
# License : BSD 3-clause

import numpy as np
from scipy import fftpack
from scipy.linalg import toeplitz

from .multitaper import sine_tapers
from ..utils import logger, verbose
from ..parallel import parallel_func, check_n_jobs


def _st(x_in, n_fft, freqs):
    """Aux function"""

    f_half = int(np.fix(n_fft // 2))

    #  Compute Toeplitz matrix with the shifted fft(h)
    Hft = fftpack.fft(x_in, n_fft)[None]
    HW = toeplitz(Hft.conj().T[:f_half + 1], Hft)
    freqs = freqs[np.newaxis].T
    # Compute all frequency domain Gaussians as one matrix
    invfk = 1. / freqs[1:f_half + 1]
    W = 2 * np.pi * (freqs * invfk.T).T  # broadcast
    G = np.exp(-np.power(W, 2, W) / 2)  # Gaussian in freq domain

    #  Exclude the first row, corresponding to zero frequency
    #  and compute Stockwell Transform
    ST = fftpack.ifft(HW[1:f_half + 1, :] * G, axis=1)  # Compute voice

    # Add the zero freq row
    st_zero = np.mean(x_in) * np.ones((1, n_fft))

    return np.concatenate([st_zero, ST])


def _st_parallel(x_in, n_fft, freqs):
    """Aux function"""
    return np.array([_st(x, n_fft, freqs) for x in x_in])


def _st_mt(tapers, x_in, n_fft, freqs, K2):
    """Aux function"""
    n, st = 0., 0.
    for k, taper in enumerate(tapers, 1):
        X = _st(taper * x_in, n_fft, freqs)
        mu = 1. - k * k / K2
        st += mu * np.abs(X) ** 2
        n += mu
    st *= len(x_in) / n
    return st


def _st_mt_parallel(tapers, x_in, n_fft, freqs, K2):
    """Aux function"""
    return np.array([_st_mt(tapers, x, n_fft, freqs, K2) for x in x_in])


@verbose
def _check_input_st(x_in, n_fft, verbose):
    """Aux function"""
    # flatten to 2 D and memorize original shape
    x_outer_shape = x_in.shape[:-1]  # non time dimension
    if x_in.ndim > 1:
        x_in = x_in.reshape(x_in.size / x_in.shape[-1], x_in.shape[-1])
    if x_in.shape[-1] < n_fft:
        msg = ('The input signal is shorter ({}) than "n_fft" ({}). '
               'Using actual signal length instead. This will be slow.')
        logger.warn(msg.format(len(x_in), n_fft))
        return len(x_in)
    return n_fft, x_in, x_outer_shape


def _restore_shape(x_out, x_outer_shape):
    """Aux function"""
    _reshape = list(x_outer_shape) + list(x_out.shape[-2:])
    return x_out.reshape(tuple(_reshape))


@verbose
def stockwell(data, sfreq, fmin=0, fmax=np.inf, n_fft=512, n_jobs=1,
              verbose=None):
    """Computes Stockwell a.k.a. S transform

    Based on MATLAB code by Kalyan S. Das

    Parameters
    ----------
    data : ndarray
        The signal to transform. Any dimensionality supported as long
        as the last dimension is time.
    sfreq : float
        The sampling frequency.
    fmin : None, float
        The minimum frequency to include. If None defaults to 0.
    fmax : None, float
        The maximum frequency to include. If None defaults to
        n_samples / 2.
    n_fft : int
        The length of the windows used for FFT. The smaller
        it is the smoother are the PSDs, i.e., the worse
        the spectral resolution. Defaults to 512. Note.
        If the signal is shorter than `n_fft`, the length
        of the signal will be used instead. This will lead
        to longer computation times.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    st : ndarray
        The complex Stockwell transformed data.
        The last two dimensions are frequency and time.

    References
    ----------
    Stockwell, R. G. "Why use the S-transform." AMS Pseudo-differential
        operators: Partial differential equations and time-frequency
        analysis 52 (2007): 279-309.
    """
    n_fft, x_in, x_outer_shape = _check_input_st(data, n_fft, verbose)

    freqs = fftpack.fftfreq(n_fft, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    n_jobs = check_n_jobs(n_jobs)

    if n_jobs == 1 and x_in.ndim == 1:
        st = _st(x_in, n_fft, freqs)[freq_mask]
    elif n_jobs == 1 and x_in.ndim == 2:
        st = np.array([_st(x, n_fft, freqs) for x in x_in])
        st = _restore_shape(st[:, freq_mask], x_outer_shape)
    elif n_jobs > 1 and x_in.ndim == 2:
        parallel, my_st, n_jobs = parallel_func(_st_parallel, n_jobs)
        out = parallel(my_st(x, n_fft, freqs) for x in
                       np.array_split(x_in, n_jobs))
        st = _restore_shape(np.concatenate(out)[:, freq_mask], x_outer_shape)
    else:
        raise RuntimeError('Something terrible happened')

    return st


@verbose
def stockwell_power(data, sfreq, n_tapers=3, fmin=0, fmax=np.inf,
                    n_fft=512, n_jobs=1, verbose=None):
    """Computes multiataper power using Stockwell a.k.a. S transform

    Based on MATLAB code by Kalyan S. Das and Python code by the NIH

    Parameters
    ----------
    data : ndarray
        The signal to transform. Any dimensionality supported as long
        as the last dimension is time.
    sfreq : float
        The sampling frequency.
    n_tapers : int
        The number of tapers to be used. If 0, only the power
        of the S transform will be returned without applying tapers.
    fmin : None, float
        The minimum frequency to include. If None defaults to 0.
    fmax : None, float
        The maximum frequency to include. If None defaults to
        n_samples / 2.
    n_fft : int
        The length of the windows used for FFT. The smaller
        it is the smoother are the PSDs, i.e., the worse
        the spectral resolution. Defaults to 512. Note.
        If the signal is shorter than `n_fft`, the length
        of the signal will be used instead. This will lead
        to longer computation times.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    st_power : ndarray
        The multitaper power of the Stockwell transformed data.
        The last two dimensions are frequency and time.

    References
    ----------
    Stockwell, R. G. "Why use the S-transform." AMS Pseudo-differential
        operators: Partial differential equations and time-frequency
        analysis 52 (2007): 279-309.
    """

    if n_tapers == 0:
        st = stockwell(data=data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                       n_fft=n_fft, n_jobs=n_jobs, verbose=verbose)
        return np.abs(st) ** 2

    n_fft, x_in, x_outer_shape = _check_input_st(data, n_fft, verbose)
    n_times = x_in.shape[-1]

    tapers = sine_tapers(n_tapers, n_times)
    K2 = float(n_tapers * n_tapers)
    freqs = fftpack.fftfreq(n_fft, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    n_jobs = check_n_jobs(n_jobs)

    if n_jobs == 1 and x_in.ndim == 1:
        st = _st_mt(tapers, x_in, n_fft, freqs, K2)[freq_mask]
    elif n_jobs == 1 and x_in.ndim == 2:
        st = np.array([_st_mt(tapers, x, n_fft, freqs, K2) for x in x_in])
        st = _restore_shape(st[:, freq_mask], x_outer_shape)
    elif n_jobs > 1 and x_in.ndim == 2:
        parallel, my_st_mt, n_jobs = parallel_func(_st_mt_parallel, n_jobs)
        out = parallel(my_st_mt(tapers, x, n_fft, freqs, K2)
                       for x in np.array_split(x_in, n_jobs))
        st = _restore_shape(np.concatenate(out)[:, freq_mask], x_outer_shape)
    else:
        raise RuntimeError('Something terrible happened')

    return st
