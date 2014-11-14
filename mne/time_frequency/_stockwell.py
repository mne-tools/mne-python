# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License : BSD 3-clause

import math
import numpy as np
from scipy import fftpack
from scipy.linalg import toeplitz

from .multitaper import sine_tapers
from ..io.pick import pick_types, pick_info
from ..utils import logger, verbose
from ..parallel import parallel_func, check_n_jobs
from .tfr import AverageTFR


def _is_power_of_two(n):
    """Returns True if n is a power of two"""
    return not (n > 0 and ((n & (n - 1))))


def _st(x_in, n_fft, freqs, width):
    """Compute Stockwell on one or multiple signals"""
    ndim = x_in.ndim

    if ndim == 1:
        x_in = x_in[np.newaxis, :]

    n_signals = len(x_in)
    f_half = n_fft // 2

    # Compute all frequency domain Gaussians as one matrix
    W = (width * np.pi * (freqs[np.newaxis, :] /
         freqs[1:f_half + 1, np.newaxis]))

    W *= W  # faster than W = np.pow(W, 2)
    G = np.exp(-W / width)  # Gaussian in freq domain

    #  Exclude the first row, corresponding to zero frequency
    #  and compute S Transform
    ST = np.empty((n_signals, f_half + 1, n_fft), dtype=np.complex64)

    for k in range(n_signals):
        Hft = fftpack.fft(x_in[k], n_fft)
        #  Compute Toeplitz matrix with the shifted fft(h)
        HW = toeplitz(Hft[:f_half + 1].conj(), Hft)

        ST[k, 1:] = fftpack.ifft(HW[1:f_half + 1, :] * G, axis=1)  # voice
        ST[k, 0] = np.mean(x_in[k])  # Add zero freq row

    if ndim == 1:
        ST = ST[0]

    return ST


def _st_mt(tapers, x_in, n_fft, freqs, K2, width):
    """Compute stockwell power with multitaper"""

    n, st = 0., 0.
    for k, taper in enumerate(tapers):
        X = _st(taper * x_in, n_fft, freqs, width)
        mu = 1. - k * k / K2
        st += mu * np.abs(X) ** 2
        n += mu

    st *= len(x_in) / n
    return st


def _st_mt_parallel(tapers, x_in, n_fft, freqs, K2, width):
    """Aux function"""
    out = np.zeros((len(x_in), n_fft // 2 + 1, n_fft),
                   dtype=np.float64)
    for ii, x in enumerate(x_in):
        out[ii] = _st_mt(tapers, x, n_fft, freqs, K2, width)
    return out


@verbose
def _check_input_st(x_in, n_fft, verbose):
    """Aux function"""
    # flatten to 2 D and memorize original shape
    x_outer_shape = x_in.shape[:-1]  # non time dimension
    n_times = x_in.shape[-1]
    x_in = x_in.reshape(x_in.size // n_times, n_times)

    if n_fft is None:
        # Compute next power of 2
        n_fft = 2 ** int(math.ceil(math.log(n_times, 2)))
    elif n_fft < n_times:
        raise ValueError("n_fft cannot be smaller than signal size. "
                         "Got %s < %s." % (n_fft, n_times))

    zero_pad = None
    if n_times < n_fft:
        msg = ('The input signal is shorter ({0}) than "n_fft" ({1}). '
               'Applying zero padding.').format(x_in.shape[-1], n_fft)
        logger.warn(msg)
        if not _is_power_of_two(n_fft):
            raise ValueError("n_fft larger than signal size should be "
                             "a power of 2. Got %s." % n_fft)
        zero_pad = n_fft - n_times
        pad_width = ([(0, 0)] * (x_in.ndim - 1)) + [(0, zero_pad)]
        x_in = np.pad(x_in, pad_width, mode='constant', constant_values=0)

    return n_fft, x_in, x_outer_shape, zero_pad


def _restore_shape(x_out, x_outer_shape):
    """Aux function"""
    _reshape = x_outer_shape + x_out.shape[-2:]
    return x_out.reshape(_reshape)


@verbose
def stockwell(data, sfreq, fmin=0, fmax=np.inf, n_fft=None, n_jobs=1,
              width=2.0, verbose=None):
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
        The maximum frequency to include. If None defaults to np.inf
    n_fft : int | None
        The length of the windows used for FFT. If None,
        it defaults to the next power of 2 larger than
        the signal length.
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
    n_fft_, x_in, x_outer_shape, zero_pad = _check_input_st(data, n_fft,
                                                            verbose)

    freqs = fftpack.fftfreq(n_fft_, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    n_jobs = check_n_jobs(n_jobs)
    if n_jobs > 1 and x_in.shape[0] == 1:
        n_jobs = 1

    parallel, my_st, n_jobs = parallel_func(_st, n_jobs)
    out = parallel(my_st(x, n_fft_, freqs, width=width) for x in
                   np.array_split(x_in, n_jobs))

    st = _restore_shape(np.concatenate(out)[:, freq_mask], x_outer_shape)

    if zero_pad is not None:
        st = st[..., :-zero_pad]

    return st


@verbose
def stockwell_power(data, sfreq, n_tapers=3, fmin=0, fmax=np.inf,
                    n_fft=None, width=2.0, n_jobs=1, verbose=None):
    """Computes multitaper power using Stockwell a.k.a. S transform

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
        The maximum frequency to include. If None defaults to np.inf
    n_fft : int | None
        The length of the windows used for FFT. If None,
        it defaults to the next power of 2 larger than
        the signal length.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    st_power : ndarray
        The multitaper power of the Stockwell transformed data.
        The last two dimensions are frequency and time.
    freqs : ndarray
        The frequencies.

    References
    ----------
    Stockwell, R. G. "Why use the S-transform." AMS Pseudo-differential
        operators: Partial differential equations and time-frequency
        analysis 52 (2007): 279-309.
    """
    n_fft_, x_in, x_outer_shape, zero_pad = _check_input_st(data, n_fft,
                                                            verbose)
    n_times = x_in.shape[-1]

    if n_tapers is None:
        tapers = np.ones((1, n_times))
    else:
        tapers = sine_tapers(n_tapers, n_times)

    n_tapers_ = len(tapers)
    K2 = float(n_tapers_ * n_tapers_)
    freqs = fftpack.fftfreq(n_fft_, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    n_jobs = check_n_jobs(n_jobs)
    if n_jobs > 1 and x_in.shape[0] == 1:
        n_jobs = 1

    parallel, my_st_mt, n_jobs = parallel_func(_st_mt_parallel, n_jobs)
    out = parallel(my_st_mt(tapers, x, n_fft_, freqs, K2, width=width)
                   for x in np.array_split(x_in, n_jobs))
    st = _restore_shape(np.concatenate(out)[:, freq_mask], x_outer_shape)
    freqs = freqs[freq_mask]

    if zero_pad is not None:
        st = st[..., :-zero_pad]

    return st, freqs


def tfr_stockwell(epochs, n_tapers=3, fmin=None, fmax=None, n_fft=None,
                  width=2.0, decim=1, n_jobs=1):
    """Compute Time-Frequency Representation (TFR) using Stockwell Transform

    Parameters
    ----------
    epochs : Epochs
        The epochs.
    n_tapers : int
        The number of tapers to be used. If 0, only the power
        of the S transform will be returned without applying tapers.
    fmin : None, float
        The minimum frequency to include. If None defaults to 0.
    fmax : None, float
        The maximum frequency to include. If None defaults to np.inf
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    power : AverageTFR
        The averaged power.
    """
    data = epochs.get_data()
    picks = pick_types(epochs.info, meg=True, eeg=True)
    info = pick_info(epochs.info, picks)
    data = data[:, picks, :]
    times = epochs.times[::decim].copy()
    power, freqs = stockwell_power(data, sfreq=info['sfreq'],
                                   fmin=fmin, fmax=fmax,
                                   n_tapers=n_tapers, n_fft=n_fft,
                                   width=width,
                                   n_jobs=n_jobs)
    power = np.mean(power[..., ::decim], axis=0)
    power = np.array(power)
    nave = len(data)
    out = AverageTFR(info, power, times, freqs, nave)
    return out
