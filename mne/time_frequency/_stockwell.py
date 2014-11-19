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


def _st(x, fmin, fmax, sfreq, width):
    """Implementation based on Matlab cpde by Ali Moukadem"""
    if x.ndim == 1:
        x = x[None]
    M = x.shape[-1]
    tw = fftpack.fftfreq(M, 1. / sfreq) / M
    tw = np.r_[tw[:1], tw[1:][::-1]]
    Fx = fftpack.fft(x)
    XF = np.c_[Fx, Fx]

    start_f = int(np.round((fmin * M / sfreq)))
    stop_f = int(fmax * M / sfreq)
    k = width  # 1 for classical stowckwell transform

    f_range = np.arange(start_f, stop_f + 1, 1)
    ST = np.empty((x.shape[0], len(f_range), M), dtype=np.complex)
    for i_f, f in enumerate(f_range):
        window = ((f / (np.sqrt(2. * np.pi) * k)) *
                  np.exp(-0.5 * (1. / k ** 2.) * (f ** 2.) * tw ** 2.))
        window /= window.sum()  # normalisation
        for i in range(len(x)):
            ST[i, i_f, :] = fftpack.ifft(XF[i, f:f + M] * fftpack.fft(window))
    if ST.shape[0] == 1:
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
def _induced_power_stockwell(data, sfreq, fmin, fmax,
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
        The minimum frequency to include. If None defaults to the minimum fft
        frequency greater than zero.
    fmax : None, float
        The maximum frequency to include. If None defaults to the maximum fft.
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
    if fmin is None:
        fmin = freqs[freqs > 0][0]
    if fmax is None:
        fmax = freqs.max()
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    n_jobs = check_n_jobs(n_jobs)

    parallel, my_st, _ = parallel_func(_st, n_jobs)
    tfrs = parallel(my_st(np.squeeze(data[:, c, :]),
                          fmin, fmax, sfreq, width)
                    for c in range(n_channels))
    n_frequencies = np.sum(freq_mask)
    psd = np.zeros((n_channels, n_frequencies, n_times))
    itc = np.zeros((n_channels, n_frequencies, n_times),
                   dtype=np.complex)
    for i_chan, tfrs in enumerate(tfrs):
        if zero_pad is not None:
            tfrs = tfrs[..., :-zero_pad]
        tfrs = tfrs[..., ::decim]
        for tfr in tfrs:
            tfr_abs = np.abs(tfr)
            psd[i_chan, ...] += tfr_abs ** 2
            itc[i_chan, ...] += tfr / tfr_abs

    psd /= n_epochs
    itc = np.abs(itc) / n_epochs
    freqs = freqs[freq_mask]
    return psd, itc, freqs


def tfr_stockwell(epochs, fmin=None, fmax=None, n_fft=None,
                  width=1.0, decim=1, n_jobs=1, return_itc=False):
    """Time-Frequency Representation (TFR) using Stockwell Transform

    Parameters
    ----------
    epochs : Epochs
        The epochs.
    fmin : None, float
        The minimum frequency to include. If None defaults to the minimum fft
        frequency greater than zero.
    fmax : None, float
        The maximum frequency to include. If None defaults to the maximum fft.
    n_fft : int | None
        The length of the windows used for FFT. If None,
        it defaults to the next power of 2 larger than
        the signal length.
    width : float
        The width of the Gaussian window. If < 1, increased temporal resolution,
        if > 1, increased frequency resolution. Defaults to 1.
        (classical S-Transform).
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
                                                 decim=decim,
                                                 n_jobs=n_jobs)
    times = epochs.times[::decim].copy()
    nave = len(data)
    out = AverageTFR(info, power, times, freqs, nave)
    if return_itc:
        out = (out, AverageTFR(info, itc, times, freqs, nave))
    return out
