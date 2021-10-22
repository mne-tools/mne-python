# Authors : Alexandre Gramfort, alexandre.gramfort@inria.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD-3-Clause

from functools import partial
import numpy as np

from ..parallel import parallel_func, check_n_jobs
from ..io.pick import _picks_to_idx
from ..utils import logger, verbose, _time_mask, _check_option
from .multitaper import psd_array_multitaper


def _decomp_aggregate_mask(epoch, func, average, freq_sl):
    _, _, spect = func(epoch)
    spect = spect[..., freq_sl, :]
    # Do the averaging here (per epoch) to save memory
    if average == 'mean':
        spect = np.nanmean(spect, axis=-1)
    elif average == 'median':
        spect = np.nanmedian(spect, axis=-1)
    return spect


def _spect_func(epoch, func, freq_sl, average):
    """Aux function."""
    # Decide if we should split this to save memory or not, since doing
    # multiple calls will incur some performance overhead. Eventually we might
    # want to write (really, go back to) our own spectrogram implementation
    # that, if possible, averages after each transform, but this will incur
    # a lot of overhead because of the many Python calls required.
    kwargs = dict(func=func, average=average, freq_sl=freq_sl)
    if epoch.nbytes > 10e6:
        spect = np.apply_along_axis(
            _decomp_aggregate_mask, -1, epoch, **kwargs)
    else:
        spect = _decomp_aggregate_mask(epoch, **kwargs)
    return spect


def _check_nfft(n, n_fft, n_per_seg, n_overlap):
    """Ensure n_fft, n_per_seg and n_overlap make sense."""
    if n_per_seg is None and n_fft > n:
        raise ValueError(('If n_per_seg is None n_fft is not allowed to be > '
                          'n_times. If you want zero-padding, you have to set '
                          'n_per_seg to relevant length. Got n_fft of %d while'
                          ' signal length is %d.') % (n_fft, n))
    n_per_seg = n_fft if n_per_seg is None or n_per_seg > n_fft else n_per_seg
    n_per_seg = n if n_per_seg > n else n_per_seg
    if n_overlap >= n_per_seg:
        raise ValueError(('n_overlap cannot be greater than n_per_seg (or '
                          'n_fft). Got n_overlap of %d while n_per_seg is '
                          '%d.') % (n_overlap, n_per_seg))
    return n_fft, n_per_seg, n_overlap


def _check_psd_data(inst, tmin, tmax, picks, proj, reject_by_annotation=False):
    """Check PSD data / pull arrays from inst."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    if not isinstance(inst, (BaseEpochs, BaseRaw, Evoked)):
        raise ValueError('epochs must be an instance of Epochs, Raw, or'
                         'Evoked. Got type {}'.format(type(inst)))

    time_mask = _time_mask(inst.times, tmin, tmax, sfreq=inst.info['sfreq'])
    picks = _picks_to_idx(inst.info, picks, 'data', with_ref_meg=False)
    if proj:
        # Copy first so it's not modified
        inst = inst.copy().apply_proj()

    sfreq = inst.info['sfreq']
    if isinstance(inst, BaseRaw):
        start, stop = np.where(time_mask)[0][[0, -1]]
        rba = 'NaN' if reject_by_annotation else None
        data = inst.get_data(picks, start, stop + 1, reject_by_annotation=rba)
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data(picks=picks)[:, :, time_mask]
    else:  # Evoked
        data = inst.data[picks][:, time_mask]

    return data, sfreq


@verbose
def psd_array_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
                    n_per_seg=None, n_jobs=1, average='mean', window='hamming',
                    verbose=None):
    """Compute power spectral density (PSD) using Welch's method.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    n_fft : int
        The length of FFT used, must be ``>= n_per_seg`` (default: 256).
        The segments will be zero-padded if ``n_fft > n_per_seg``.
    n_overlap : int
        The number of points of overlap between segments. Will be adjusted
        to be <= n_per_seg. The default value is 0.
    n_per_seg : int | None
        Length of each Welch segment (windowed with a Hamming window). Defaults
        to None, which sets n_per_seg equal to n_fft.
    %(n_jobs)s
    %(average-psd)s

        .. versionadded:: 0.19.0
    %(window-psd)s

        .. versionadded:: 0.22.0
    %(verbose)s

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or (..., n_freqs, n_segments)
        The power spectral densities. If ``average='mean`` or
        ``average='median'``, the returned array will have the same shape
        as the input data plus an additional frequency dimension.
        If ``average=None``, the returned array will have the same shape as
        the input data plus two additional dimensions corresponding to
        frequencies and the unaggregated segments, respectively.
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    Notes
    -----
    .. versionadded:: 0.14.0
    """
    _check_option('average', average, (None, 'mean', 'median'))

    dshape = x.shape[:-1]
    n_times = x.shape[-1]
    x = x.reshape(-1, n_times)

    # Prep the PSD
    n_fft, n_per_seg, n_overlap = _check_nfft(n_times, n_fft, n_per_seg,
                                              n_overlap)
    win_size = n_fft / float(sfreq)
    logger.info("Effective window size : %0.3f (s)" % win_size)
    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if not freq_mask.any():
        raise ValueError(
            f'No frequencies found between fmin={fmin} and fmax={fmax}')
    freq_sl = slice(*(np.where(freq_mask)[0][[0, -1]] + [0, 1]))
    del freq_mask
    freqs = freqs[freq_sl]

    # Parallelize across first N-1 dimensions
    n_jobs = check_n_jobs(n_jobs)
    x_splits = np.array_split(x, n_jobs)
    logger.debug(
        f'Spectogram using {n_fft}-point FFT on {n_per_seg} samples with '
        f'{n_overlap} overlap and {window} window')

    from scipy.signal import spectrogram
    parallel, my_spect_func, n_jobs = parallel_func(_spect_func, n_jobs=n_jobs)
    func = partial(spectrogram, noverlap=n_overlap, nperseg=n_per_seg,
                   nfft=n_fft, fs=sfreq, window=window)
    f_spect = parallel(my_spect_func(d, func=func, freq_sl=freq_sl,
                                     average=average)
                       for d in x_splits)
    psds = np.concatenate(f_spect, axis=0)
    shape = dshape + (len(freqs),)
    if average is None:
        shape = shape + (-1,)
    psds.shape = shape
    return psds, freqs


@verbose
def psd_welch(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None, n_fft=256,
              n_overlap=0, n_per_seg=None, picks=None, proj=False, n_jobs=1,
              reject_by_annotation=True, average='mean', window='hamming',
              verbose=None):
    """Compute the power spectral density (PSD) using Welch's method.

    Calculates periodograms for a sliding window over the time dimension, then
    averages them together for each channel/epoch.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation.
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    tmin : float | None
        Min time of interest.
    tmax : float | None
        Max time of interest.
    n_fft : int
        The length of FFT used, must be ``>= n_per_seg`` (default: 256).
        The segments will be zero-padded if ``n_fft > n_per_seg``.
        If n_per_seg is None, n_fft must be <= number of time points
        in the data.
    n_overlap : int
        The number of points of overlap between segments. Will be adjusted
        to be <= n_per_seg. The default value is 0.
    n_per_seg : int | None
        Length of each Welch segment (windowed with a Hamming window). Defaults
        to None, which sets n_per_seg equal to n_fft.
    %(picks_good_data_noref)s
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    %(n_jobs)s
    %(reject_by_annotation_raw)s

        .. versionadded:: 0.15.0
    %(average-psd)s

        .. versionadded:: 0.19.0
    %(window-psd)s

        .. versionadded:: 0.22.0
    %(verbose)s

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or (..., n_freqs, n_segments)
        The power spectral densities. If ``average='mean`` or
        ``average='median'`` and input is of type Raw or Evoked, then psds will
        be of shape (n_channels, n_freqs); if input is of type Epochs, then
        psds will be of shape (n_epochs, n_channels, n_freqs).
        If ``average=None``, the returned array will have an additional
        dimension corresponding to the unaggregated segments.
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    See Also
    --------
    mne.io.Raw.plot_psd
    mne.Epochs.plot_psd
    psd_multitaper
    psd_array_welch

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj,
                                  reject_by_annotation=reject_by_annotation)
    return psd_array_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,
                           n_overlap=n_overlap, n_per_seg=n_per_seg,
                           average=average, n_jobs=n_jobs, window=window,
                           verbose=verbose)


@verbose
def psd_multitaper(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                   bandwidth=None, adaptive=False, low_bias=True,
                   normalization='length', picks=None, proj=False,
                   n_jobs=1, reject_by_annotation=False, verbose=None):
    """Compute the power spectral density (PSD) using multitapers.

    Calculates spectral density for orthogonal tapers, then averages them
    together for each channel/epoch. See :footcite:`Slepian1978` for a
    description of the tapers and :footcite:`PercivalWalden1993` for the
    general method.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation.
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    tmin : float | None
        Min time of interest.
    tmax : float | None
        Max time of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    %(picks_good_data_noref)s
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    %(n_jobs)s
    %(reject_by_annotation_raw)s
    %(verbose)s

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. If input is of type Raw,
        then psds will be shape (n_channels, n_freqs), if input is type Epochs
        then psds will be shape (n_epochs, n_channels, n_freqs).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    See Also
    --------
    mne.io.Raw.plot_psd
    mne.Epochs.plot_psd
    psd_array_multitaper
    psd_welch
    csd_multitaper

    Notes
    -----
    .. versionadded:: 0.12.0

    References
    ----------
    .. footbibliography::
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj,
                                  reject_by_annotation=reject_by_annotation)
    return psd_array_multitaper(data, sfreq, fmin=fmin, fmax=fmax,
                                bandwidth=bandwidth, adaptive=adaptive,
                                low_bias=low_bias, normalization=normalization,
                                n_jobs=n_jobs, verbose=verbose)
