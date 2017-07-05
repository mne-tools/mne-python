# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
#           Mikolaj Magnuski <mmagnuski@swps.edu.pl>
# License : BSD 3-clause

import numbers
import numpy as np

from ..parallel import parallel_func
from ..io.pick import _pick_data_channels
from ..utils import logger, verbose, _time_mask, _get_reduction
from ..fixes import get_spectrogram
from .multitaper import psd_array_multitaper


def _psd_func(epoch, noverlap, n_per_seg, nfft, fs, freq_mask, func):
    """Aux function."""
    return func(epoch, fs=fs, nperseg=n_per_seg, noverlap=noverlap,
                nfft=nfft, window='hann')[2][..., freq_mask, :]


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
        raise ValueError(('n_overlap cannot be greater than or equal to '
                          'n_per_seg (or n_fft). Got n_overlap of %d while '
                          'n_per_seg is %d.') % (n_overlap, n_per_seg))
    return n_fft, n_per_seg, n_overlap


def _check_psd_data(inst, tmin, tmax, picks, proj, reject_by_annotation=False):
    """Check PSD data / pull arrays from inst."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    if not isinstance(inst, (BaseEpochs, BaseRaw, Evoked)):
        raise ValueError('epochs must be an instance of Epochs, Raw, or'
                         'Evoked. Got type {0}'.format(type(inst)))

    time_mask = _time_mask(inst.times, tmin, tmax, sfreq=inst.info['sfreq'])
    if picks is None:
        picks = _pick_data_channels(inst.info, with_ref_meg=False)
    if isinstance(picks, numbers.Integral):
        # Fix for IndexError when picks is int: in that case inst.data[picks]
        # or epochs.get_data()[:, picks] would drop channels dimension
        picks = [picks]
    if proj:
        # Copy first so it's not modified
        inst = inst.copy().apply_proj()

    sfreq = inst.info['sfreq']
    if isinstance(inst, BaseRaw):
        start, stop = np.where(time_mask)[0][[0, -1]]
        rba = 'NaN' if reject_by_annotation else reject_by_annotation
        data = inst.get_data(picks, start, stop + 1, reject_by_annotation=rba)
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()[:, picks][:, :, time_mask]
    else:  # Evoked
        data = inst.data[picks][:, time_mask]

    return data, sfreq


@verbose
def psd_array_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
                    n_per_seg=None, combine='mean', n_jobs=1, verbose=None):
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
        The number of points of overlap between segments. Must be < n_per_seg.
        The default value is 0.
    n_per_seg : int | None
        Length of each Welch segment. The smaller it is with respect to the
        signal length the smoother are the PSDs. Defaults to None, which sets
        n_per_seg equal to n_fft.
    combine : str | float | callable | None
        The type of reduction to perform on welch windows. If string it can be
        'mean' or 'median'. If float it is understood as the proportion of
        values to trim before performing mean (has to be > 0 and < 0.5) ie.
        trimmed-mean. If function it has to perform the reduction along last
        dimension. If None, no reduction is performed and PSDs for individual
        windows are returned. In such case the output PSDs have one more
        dimension than the input array with windows as the last dimension.

        .. versionadded:: 0.15.0
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. All dimensions up to the last will
        be the same as input unless combine is None. When combine is None
        the PSDs are of shape (..., n_freqs, n_windows).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    Notes
    -----
    .. versionadded:: 0.14.0
    """
    spectrogram = get_spectrogram()
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
    freqs = freqs[freq_mask]

    # Parallelize across first N-1 dimensions (channels or epochs * channels)
    parallel, my_psd_func, n_jobs = parallel_func(_psd_func, n_jobs=n_jobs)
    x_splits = np.array_split(x, n_jobs)
    f_spectrogram = parallel(my_psd_func(d, noverlap=n_overlap, nfft=n_fft,
                                         fs=sfreq, freq_mask=freq_mask,
                                         func=spectrogram, n_per_seg=n_per_seg)
                             for d in x_splits)

    # Combining, reducing windows and reshaping to original data shape
    if combine is not None:
        combine = _get_reduction(combine)
        psds = np.concatenate([combine(f_s) for f_s in f_spectrogram],
                              axis=0).reshape(dshape + (-1,))
    else:
        psds = np.concatenate(f_spectrogram, axis=0)
        n_windows = psds.shape[-1]
        psds = psds.reshape(dshape + (-1, n_windows))
    return psds, freqs


@verbose
def psd_welch(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None, n_fft=256,
              n_overlap=0, n_per_seg=None, picks=None, proj=False, n_jobs=1,
              reject_by_annotation=True, combine='mean', verbose=None):
    """Compute the power spectral density (PSD) using Welch's method.

    Calculates periodigrams for a sliding window over the
    time dimension, then averages them together for each channel/epoch.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    n_fft : int
        The length of FFT used, must be ``>= n_per_seg`` (default: 256).
        The segments will be zero-padded if ``n_fft > n_per_seg``.
        If n_per_seg is None, n_fft must be >= number of time points
        in the data.
    n_overlap : int
        The number of points of overlap between segments. Must be < n_per_seg.
        The default value is 0.
    n_per_seg : int | None
        Length of each Welch segment. The smaller it is with respect to the
        signal length the smoother are the PSDs. Defaults to None, which sets
        n_per_seg equal to n_fft.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    reject_by_annotation : bool
        Whether to omit bad segments from the data while computing the
        PSD. If True, annotated segments with a description that starts
        with 'bad' are omitted. Has no effect if ``inst`` is an Epochs or
        Evoked object. Defaults to True.

        .. versionadded:: 0.15.0
    combine : str | float | callable | None
        The type of reduction to perform on welch windows. If string it can be
        'mean' or 'median'. If float it is understood as the proportion of
        values to trim before performing mean (has to be > 0 and < 0.5) ie.
        trimmed-mean. If function it has to perform the reduction along last
        dimension. If None, no reduction is performed and PSDs for individual
        windows are returned. In such case the output PSDs have one more
        dimension than the input array with windows as the last dimension.

        .. versionadded:: 0.15.0
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or (..., n_freqs, n_windows)
        The power spectral densities. If combine is not None and input is of
        type Raw, returned PSDs are shaped (n_channels, n_freqs), if input is
        of type Epochs returned PSDs are shaped (n_epochs, n_channels,
        n_freqs). When combine is None the PSDs are of shape (..., n_freqs,
        n_windows).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_multitaper,
    csd_epochs, psd_array_welch

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj,
                                  reject_by_annotation=reject_by_annotation)
    psds, freq = psd_array_welch(data, sfreq, fmin=fmin, fmax=fmax,
                                 n_fft=n_fft, n_overlap=n_overlap,
                                 n_per_seg=n_per_seg, combine=combine,
                                 n_jobs=n_jobs, verbose=verbose)
    # reshape if combine is None:
    return psds, freq


@verbose
def psd_multitaper(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                   bandwidth=None, adaptive=False, low_bias=True,
                   normalization='length', picks=None, proj=False,
                   n_jobs=1, verbose=None):
    """Compute the power spectral density (PSD) using multitapers.

    Calculates spectral density for orthogonal tapers, then averages them
    together for each channel/epoch. See [1] for a description of the tapers
    and [2] for the general method.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation.
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. If input is of type Raw,
        then psds will be shape (n_channels, n_freqs), if input is type Epochs
        then psds will be shape (n_epochs, n_channels, n_freqs).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    References
    ----------
    .. [1] Slepian, D. "Prolate spheroidal wave functions, Fourier analysis,
           and uncertainty V: The discrete case." Bell System Technical
           Journal, vol. 57, 1978.

    .. [2] Percival D.B. and Walden A.T. "Spectral Analysis for Physical
           Applications: Multitaper and Conventional Univariate Techniques."
           Cambridge University Press, 1993.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_welch, csd_epochs,
    psd_array_multitaper

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return psd_array_multitaper(data, sfreq, fmin=fmin, fmax=fmax,
                                bandwidth=bandwidth, adaptive=adaptive,
                                low_bias=low_bias, normalization=normalization,
                                n_jobs=n_jobs, verbose=verbose)
