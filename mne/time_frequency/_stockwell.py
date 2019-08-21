# Authors : Denis A. Engemann <denis.engemann@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License : BSD 3-clause

from inspect import isgenerator
import math
import numpy as np
from scipy import fftpack
# XXX explore cuda optimization at some point.

from ..io.pick import _pick_data_channels, pick_info
from ..utils import verbose, warn, fill_doc
from ..parallel import parallel_func, check_n_jobs
from .tfr import _get_data, _check_stfr_list_elem, _assign_tfr_class


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
    windows = np.empty((len(f_range), len(tw)), dtype=np.complex)
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
    ST = np.empty(x.shape[:-1] + (len(windows), n_samp), dtype=np.complex)
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


def _tfr_list_stockwell(inst, fmin, fmax, n_fft, width, decim, return_itc,
                        n_jobs):
    """Perform stockwell transform for stc lists/generator objects."""
    from ..source_estimate import _BaseSourceEstimate

    for ep_idx, obj in enumerate(inst):

        if not isinstance(obj, _BaseSourceEstimate):
            raise TypeError("List or generator input must consist of "
                            "SourceEstimate objects. Got {}."
                            .format(type(inst)))

        # load the data. Set return_itc=False to omit an Error
        data, kernel = _get_data(obj, return_itc=False, fill_dims=False)

        data, n_fft_, zero_pad = _check_input_st(data, n_fft)

        if ep_idx == 0:

            n_channels = len(kernel) if kernel is not None else len(data)
            sfreq = obj.sfreq
            type_ref = type(obj)
            tmin_ref = obj._tmin

            freqs = fftpack.fftfreq(n_fft_, 1. / sfreq)
            if fmin is None:
                fmin = freqs[freqs > 0][0]
            if fmax is None:
                fmax = freqs.max()

            start_f = np.abs(freqs - fmin).argmin()
            stop_f = np.abs(freqs - fmax).argmin()
            freqs = freqs[start_f:stop_f]

            n_samp = data.shape[-1]

            W = _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width)

            n_out = (n_samp - zero_pad)
            n_out = n_out // decim + bool(n_out % decim)
            psd = np.zeros((n_channels, len(W), n_out))
            itc = np.zeros_like(psd, dtype=np.complex) if return_itc else None

        else:
            # make sure all elements got the same properties as the first one
            _check_stfr_list_elem(obj, type_ref, sfreq, tmin_ref)

        X = fftpack.fft(data)
        XX = np.concatenate([X, X], axis=-1)

        for i_f, window in enumerate(W):
            f = start_f + i_f
            ST = fftpack.ifft(XX[:, f:f + n_samp] * window)
            if zero_pad > 0:
                TFR = ST[:, :-zero_pad:decim]
            else:
                TFR = ST[:, ::decim]

            # compute the full source time series from kernel and tfr
            if kernel is not None:
                TFR = np.tensordot(kernel, TFR, [-1, 0])

            TFR_abs = np.abs(TFR)
            TFR_abs[TFR_abs == 0] = 1.
            if return_itc:
                TFR /= TFR_abs
                itc[:, i_f, :] += TFR
            TFR_abs *= TFR_abs
            psd[:, i_f, :] += TFR_abs

    psd /= ep_idx + 1

    if return_itc:
        itc /= ep_idx + 1
        for i_f, window in enumerate(W):
            itc[:, i_f, :] = np.abs(itc[:, i_f, :])
        itc = itc.real

    # one list object is passed for type references etc.
    return psd, itc, freqs, obj


@fill_doc
def tfr_array_stockwell(data, sfreq, fmin=None, fmax=None, n_fft=None,
                        width=1.0, decim=1, return_itc=False, n_jobs=1):
    """Compute power and intertrial coherence using Stockwell (S) transform.

    See [1]_, [2]_, [3]_, [4]_ for more information.

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

    References
    ----------
    .. [1] Stockwell, R. G. "Why use the S-transform." AMS Pseudo-differential
       operators: Partial differential equations and time-frequency
       analysis 52 (2007): 279-309.
    .. [2] Moukadem, A., Bouguila, Z., Abdeslam, D. O, and Dieterlen, A.
       Stockwell transform optimization applied on the detection of split in
       heart sounds (2014). Signal Processing Conference (EUSIPCO), 2013
       Proceedings of the 22nd European, pages 2015--2019.
    .. [3] Wheat, K., Cornelissen, P. L., Frost, S.J, and Peter C. Hansen
       (2010). During Visual Word Recognition, Phonology Is Accessed
       within 100 ms and May Be Mediated by a Speech Production
       Code: Evidence from Magnetoencephalography. The Journal of
       Neuroscience, 30 (15), 5229-5233.
    .. [4] K. A. Jones and B. Porjesz and D. Chorlian and M. Rangaswamy and C.
       Kamarajan and A. Padmanabhapillai and A. Stimus and H. Begleiter
       (2006). S-transform time-frequency analysis of P300 reveals deficits in
       individuals diagnosed with alcoholism.
       Clinical Neurophysiology 117 2128--2143

    See Also
    --------
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet
    """
    n_epochs, n_channels = data.shape[:2]
    n_out = data.shape[2] // decim + bool(data.shape[2] % decim)
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
    """Time-Frequency Representation (TFR) using Stockwell Transform.

    Parameters
    ----------
    inst : Epochs | Evoked | SourceEstimate | list of SourceEstimate
        The object to be computed. Can be Epochs, Evoked, any type of
        SourceEstimate, or a list of multiple SourceEstimates of the same type.
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
    power : AverageTFR | SourceTFR
        The averaged power.
    itc : AverageTFR | SourceTFR
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
    """
    from ..source_estimate import _BaseSourceEstimate
    # verbose dec is used b/c subfunctions are verbose

    n_jobs = check_n_jobs(n_jobs)

    info = None
    nave = None
    if isinstance(inst, list) or isgenerator(inst):

        power, itc, freqs, inst = \
            _tfr_list_stockwell(inst, fmin, fmax, n_fft, width, decim,
                                return_itc, n_jobs)

    else:
        data, _ = _get_data(inst, return_itc)
        if isinstance(inst, _BaseSourceEstimate):
            sfreq = inst.sfreq
        else:
            nave = len(data)
            picks = _pick_data_channels(inst.info)
            data = data[:, picks, :]
            info = pick_info(inst.info, picks)
            sfreq = info['sfreq']

        power, itc, freqs = \
            tfr_array_stockwell(data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                n_fft=n_fft, width=width, decim=decim,
                                return_itc=return_itc, n_jobs=n_jobs)

    times = inst.times[::decim].copy()
    out = _assign_tfr_class(power, inst, info, freqs, times, average=True,
                            nave=nave, method='stockwell-power')
    if return_itc:
        out = (out, _assign_tfr_class(itc, inst, info, freqs, times,
                                      average=True, nave=nave,
                                      method='stockwell-itc'))
    return out
