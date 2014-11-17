# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License : BSD 3-clause

import math
import numpy as np
from scipy import fftpack
from scipy.linalg import toeplitz

from ..io.pick import pick_types, pick_info
from ..utils import logger, verbose
from ..parallel import parallel_func, check_n_jobs
from .tfr import AverageTFR


def _is_power_of_two(n):
    """Returns True if n is a power of two"""
    return not (n > 0 and ((n & (n - 1))))


def _finalize_st(x_in, n_fft, f_half, G):
    """Aux function"""
    Hft = fftpack.fft(x_in, n_fft)
    #  Compute Toeplitz matrix with the shifted fft(h)
    HW = toeplitz(Hft[:f_half + 1].conj(), Hft)
    out = fftpack.ifft(HW[1:f_half + 1, :] * G, axis=1)
    return out


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
    ST = np.empty((n_signals, f_half, n_fft), dtype=np.complex64)
    for k in range(n_signals):
        ST[k] = _finalize_st(x_in[k], n_fft, f_half, G)
    if ndim == 1:
        ST = ST[0]

    return ST


@verbose
def _check_input_st(x_in, n_fft, verbose=None):
    """Aux function"""
    # flatten to 2 D and memorize original shape
    n_times = x_in.shape[-1]

    if n_fft is None or (not _is_power_of_two(n_fft) and
                         n_times > n_fft):
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
        zero_pad = n_fft - n_times
        pad_width = ([(0, 0)] * (x_in.ndim - 1)) + [(0, zero_pad)]
        x_in = np.pad(x_in, pad_width, mode='constant', constant_values=0)

    return x_in, n_fft, zero_pad


@verbose
def _induced_power_stockwell(data, sfreq, fmin=0, fmax=np.inf,
                             n_fft=None, width=1.0, n_jobs=1, decim=1,
                             verbose=None):
    """Computes multitaper power using Stockwell a.k.a. S transform

    Based on MATLAB code by Kalyan S. Das and Python code by the NIH

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
    n_epochs, n_channels, n_times = data[:, :, ::decim].shape
    data, n_fft_, zero_pad = _check_input_st(data, n_fft, verbose)

    freqs = fftpack.fftfreq(n_fft_, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    n_frequencies = len(freqs)
    n_jobs = check_n_jobs(n_jobs)

    n_frequencies = np.sum(freq_mask)
    parallel, my_st, _ = parallel_func(_st, n_jobs)
    tfrs = parallel(my_st(np.squeeze(data[:, c, :]),
                          n_fft_, freqs, width)
                    for c in range(n_channels))

    psd = np.zeros((n_channels, n_frequencies, n_times))
    itc = np.zeros((n_channels, n_frequencies, n_times),
                   dtype=np.complex)
    for i_chan, tfrs in enumerate(tfrs):
        if zero_pad is not None:
            tfrs = tfrs[..., :-zero_pad]
        tfrs = tfrs[:, freq_mask][..., ::decim]
        for tfr in tfrs:
            tfr_abs = np.abs(tfr)
            psd[i_chan, ...] += tfr_abs ** 2
            itc[i_chan, ...] += tfr / tfr_abs

    psd /= n_epochs
    itc = np.abs(itc) / n_epochs
    freqs = freqs[freq_mask]
    return psd, itc, freqs


def tfr_stockwell(epochs, fmin=0, fmax=np.inf, n_fft=None,
                  width=1.0, decim=1, n_jobs=1, return_itc=False):
    """Time-Frequency Representation (TFR) using Stockwell Transform

    Parameters
    ----------
    epochs : Epochs
        The epochs.
    fmin : float
        The minimum frequency to include.
    fmax : loat
        The maximum frequency to include.
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    n_jobs : int
        The number of jobs to run in parallel (over channels).

    Returns
    -------
    power : AverageTFR
        The averaged power.
    """
    data = epochs.get_data()
    picks = pick_types(epochs.info, meg=True, eeg=True)
    info = pick_info(epochs.info, picks)
    data = data[:, picks, :]
    power, itc, freqs = _induced_power_stockwell(data,
                                                 sfreq=info['sfreq'],
                                                 fmin=fmin, fmax=fmax,
                                                 n_fft=n_fft,
                                                 width=width,
                                                 n_jobs=n_jobs)
    times = epochs.times[::decim].copy()
    nave = len(data)
    out = AverageTFR(info, power, times, freqs, nave)
    if return_itc:
        out = (out, AverageTFR(info, itc, times, freqs, nave))
    return out
