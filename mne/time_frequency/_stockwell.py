# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License : BSD 3-clause

from copy import deepcopy
import math
import numpy as np
from scipy import fftpack
# XXX explore cuda optimization at some point.

from ..io.pick import _pick_data_channels, pick_info
from ..utils import verbose, warn, fill_doc, _validate_type
from ..parallel import parallel_func, check_n_jobs
from .tfr import AverageTFR, _get_data


def _check_input_st(x_in, n_fft):
    """Aux function."""
    # flatten to 2 D and memorize original shape
    n_times = x_in.shape[-1]

    def _is_power_of_two(n):
        return not (n > 0 and ((n & (n - 1))))

    if n_fft is None or (not _is_power_of_two(n_fft) and n_times > n_fft):
        # Compute next power of 2
        n_fft = 2 ** int(math.ceil(math.log(n_times, 2)))
    elif n_fft < n_times:
        raise ValueError("n_fft cannot be smaller than signal size. "
                         "Got %s < %s." % (n_fft, n_times))
    if n_times < n_fft:
        warn('The input signal is shorter ({}) than "n_fft" ({}). '
             'Applying zero padding.'.format(x_in.shape[-1], n_fft))
        zero_pad = n_fft - n_times
        pad_array = np.zeros(x_in.shape[:-1] + (zero_pad,), x_in.dtype)
        x_in = np.concatenate((x_in, pad_array), axis=-1)
    else:
        zero_pad = 0
    return x_in, n_fft, zero_pad


def _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width):
    """Precompute stockwell Gaussian windows (in the freq domain)."""
    tw = fftpack.fftfreq(n_samp, 1. / sfreq) / n_samp
    tw = np.r_[tw[:1], tw[1:][::-1]]

    k = width  # 1 for classical stowckwell transform
    f_range = np.arange(start_f, stop_f, 1)
    windows = np.empty((len(f_range), len(tw)), dtype=np.complex128)
    for i_f, f in enumerate(f_range):
        if f == 0.:
            window = np.ones(len(tw))
        else:
            window = ((f / (np.sqrt(2. * np.pi) * k)) *
                      np.exp(-0.5 * (1. / k ** 2.) * (f ** 2.) * tw ** 2.))
        window /= window.sum()  # normalisation
        windows[i_f] = fftpack.fft(window)
    return windows


def _st(x, start_f, windows):
    """Compute ST based on Ali Moukadem MATLAB code (used in tests)."""
    n_samp = x.shape[-1]
    ST = np.empty(x.shape[:-1] + (len(windows), n_samp), dtype=np.complex128)
    # do the work
    Fx = fftpack.fft(x)
    XF = np.concatenate([Fx, Fx], axis=-1)
    for i_f, window in enumerate(windows):
        f = start_f + i_f
        ST[..., i_f, :] = fftpack.ifft(XF[..., f:f + n_samp] * window)
    return ST


def _st_power_itc(x, start_f, compute_itc, zero_pad, decim, W):
    """Aux function."""
    n_samp = x.shape[-1]
    n_out = (n_samp - zero_pad)
    n_out = n_out // decim + bool(n_out % decim)
    psd = np.empty((len(W), n_out))
    itc = np.empty_like(psd) if compute_itc else None
    X = fftpack.fft(x)
    XX = np.concatenate([X, X], axis=-1)
    for i_f, window in enumerate(W):
        f = start_f + i_f
        ST = fftpack.ifft(XX[:, f:f + n_samp] * window)
        if zero_pad > 0:
            TFR = ST[:, :-zero_pad:decim]
        else:
            TFR = ST[:, ::decim]
        TFR_abs = np.abs(TFR)
        TFR_abs[TFR_abs == 0] = 1.
        if compute_itc:
            TFR /= TFR_abs
            itc[i_f] = np.abs(np.mean(TFR, axis=0))
        TFR_abs *= TFR_abs
        psd[i_f] = np.mean(TFR_abs, axis=0)
    return psd, itc


@fill_doc
def tfr_array_stockwell(data, sfreq, fmin=None, fmax=None, n_fft=None,
                        width=1.0, decim=1, return_itc=False, n_jobs=1):
    """Compute power and intertrial coherence using Stockwell (S) transform.

    Same computation as `~mne.time_frequency.tfr_stockwell`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.

    See :footcite:`Stockwell2007,MoukademEtAl2014,WheatEtAl2010,JonesEtAl2006`
    for more information.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        The signal to transform.
    sfreq : float
        The sampling frequency.
    fmin : None, float
        The minimum frequency to include. If None defaults to the minimum fft
        frequency greater than zero.
    fmax : None, float
        The maximum frequency to include. If None defaults to the maximum fft.
    n_fft : int | None
        The length of the windows used for FFT. If None, it defaults to the
        next power of 2 larger than the signal length.
    width : float
        The width of the Gaussian window. If < 1, increased temporal
        resolution, if > 1, increased frequency resolution. Defaults to 1.
        (classical S-Transform).
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    %(n_jobs)s

    Returns
    -------
    st_power : ndarray
        The multitaper power of the Stockwell transformed data.
        The last two dimensions are frequency and time.
    itc : ndarray
        The intertrial coherence. Only returned if return_itc is True.
    freqs : ndarray
        The frequencies.

    See Also
    --------
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet

    References
    ----------
    .. footbibliography::
    """
    _validate_type(data, np.ndarray, 'data')
    if data.ndim != 3:
        raise ValueError(
            'data must be 3D with shape (n_epochs, n_channels, n_times), '
            f'got {data.shape}')
    n_epochs, n_channels = data.shape[:2]
    n_out = data.shape[2] // decim + bool(data.shape[-1] % decim)
    data, n_fft_, zero_pad = _check_input_st(data, n_fft)

    freqs = fftpack.fftfreq(n_fft_, 1. / sfreq)
    if fmin is None:
        fmin = freqs[freqs > 0][0]
    if fmax is None:
        fmax = freqs.max()

    start_f = np.abs(freqs - fmin).argmin()
    stop_f = np.abs(freqs - fmax).argmin()
    freqs = freqs[start_f:stop_f]

    W = _precompute_st_windows(data.shape[-1], start_f, stop_f, sfreq, width)
    n_freq = stop_f - start_f
    psd = np.empty((n_channels, n_freq, n_out))
    itc = np.empty((n_channels, n_freq, n_out)) if return_itc else None

    parallel, my_st, _ = parallel_func(_st_power_itc, n_jobs)
    tfrs = parallel(my_st(data[:, c, :], start_f, return_itc, zero_pad,
                          decim, W)
                    for c in range(n_channels))
    for c, (this_psd, this_itc) in enumerate(iter(tfrs)):
        psd[c] = this_psd
        if this_itc is not None:
            itc[c] = this_itc

    return psd, itc, freqs


@verbose
def tfr_stockwell(inst, fmin=None, fmax=None, n_fft=None,
                  width=1.0, decim=1, return_itc=False, n_jobs=1,
                  verbose=None):
    """Compute Time-Frequency Representation (TFR) using Stockwell Transform.

    Same computation as `~mne.time_frequency.tfr_array_stockwell`, but operates
    on `~mne.Epochs` objects instead of :class:`NumPy arrays <numpy.ndarray>`.

    See :footcite:`Stockwell2007,MoukademEtAl2014,WheatEtAl2010,JonesEtAl2006`
    for more information.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    fmin : None, float
        The minimum frequency to include. If None defaults to the minimum fft
        frequency greater than zero.
    fmax : None, float
        The maximum frequency to include. If None defaults to the maximum fft.
    n_fft : int | None
        The length of the windows used for FFT. If None, it defaults to the
        next power of 2 larger than the signal length.
    width : float
        The width of the Gaussian window. If < 1, increased temporal
        resolution, if > 1, increased frequency resolution. Defaults to 1.
        (classical S-Transform).
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    n_jobs : int
        The number of jobs to run in parallel (over channels).
    %(verbose)s

    Returns
    -------
    power : AverageTFR
        The averaged power.
    itc : AverageTFR
        The intertrial coherence. Only returned if return_itc is True.

    See Also
    --------
    mne.time_frequency.tfr_array_stockwell
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet

    Notes
    -----
    .. versionadded:: 0.9.0

    References
    ----------
    .. footbibliography::
    """
    # verbose dec is used b/c subfunctions are verbose
    data = _get_data(inst, return_itc)
    picks = _pick_data_channels(inst.info)
    info = pick_info(inst.info, picks)
    data = data[:, picks, :]
    n_jobs = check_n_jobs(n_jobs)
    power, itc, freqs = tfr_array_stockwell(data, sfreq=info['sfreq'],
                                            fmin=fmin, fmax=fmax, n_fft=n_fft,
                                            width=width, decim=decim,
                                            return_itc=return_itc,
                                            n_jobs=n_jobs)
    times = inst.times[::decim].copy()
    nave = len(data)
    out = AverageTFR(info, power, times, freqs, nave, method='stockwell-power')
    if return_itc:
        out = (out, AverageTFR(deepcopy(info), itc, times.copy(),
                               freqs.copy(), nave, method='stockwell-itc'))
    return out
