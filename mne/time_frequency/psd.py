# Authors : Alexandre Gramfort, alexandre.gramfort@inria.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD-3-Clause

from functools import partial

import numpy as np

from ..parallel import parallel_func
from ..utils import logger, verbose, _check_option, _ensure_int


# adapted from SciPy
# https://github.com/scipy/scipy/blob/f71e7fad717801c4476312fe1e23f2dfbb4c9d7f/scipy/signal/_spectral_py.py#L2019  # noqa: E501
def _median_biases(n):
    # Compute the biases for 0 to max(n, 1) terms included in a median calc
    biases = np.ones(n + 1)
    # The original SciPy code is:
    #
    # def _median_bias(n):
    #     ii_2 = 2 * np.arange(1., (n - 1) // 2 + 1)
    #     return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)
    #
    # This is a sum over (n-1)//2 terms.
    # The ii_2 terms here for different n are:
    #
    # n=0: []  # 0 terms
    # n=1: []  # 0 terms
    # n=2: []  # 0 terms
    # n=3: [2]  # 1 term
    # n=4: [2]  # 1 term
    # n=5: [2, 4]  # 2 terms
    # n=6: [2, 4]  # 2 terms
    # ...
    #
    # We can get the terms for 0 through n using a cumulative summation and
    # indexing:
    if n >= 3:
        ii_2 = 2 * np.arange(1, (n - 1) // 2 + 1)
        sums = 1 + np.cumsum(1. / (ii_2 + 1) - 1. / ii_2)
        idx = np.arange(2, n) // 2 - 1
        biases[3:] = sums[idx]
    return biases


def _decomp_aggregate_mask(epoch, func, average, freq_sl):
    _, _, spect = func(epoch)
    spect = spect[..., freq_sl, :]
    # Do the averaging here (per epoch) to save memory
    if average == 'mean':
        spect = np.nanmean(spect, axis=-1)
    elif average == 'median':
        biases = _median_biases(spect.shape[-1])
        idx = (~np.isnan(spect)).sum(-1)
        spect = np.nanmedian(spect, axis=-1) / biases[idx]
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


@verbose
def psd_array_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
                    n_per_seg=None, n_jobs=None, average='mean',
                    window='hamming', *, verbose=None):
    """Compute power spectral density (PSD) using Welch's method.

    Welch's method is described in :footcite:t:`Welch1967`.

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
    %(average_psd)s

        .. versionadded:: 0.19.0
    %(window_psd)s

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

    References
    ----------
    .. footbibliography::
    """
    _check_option('average', average, (None, False, 'mean', 'median'))
    n_fft = _ensure_int(n_fft, "n_fft")
    n_overlap = _ensure_int(n_overlap, "n_overlap")
    if n_per_seg is not None:
        n_per_seg = _ensure_int(n_per_seg, "n_per_seg")
    if average is False:
        average = None

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
    logger.debug(
        f'Spectogram using {n_fft}-point FFT on {n_per_seg} samples with '
        f'{n_overlap} overlap and {window} window')

    from scipy.signal import spectrogram
    parallel, my_spect_func, n_jobs = parallel_func(_spect_func, n_jobs=n_jobs)
    func = partial(spectrogram, noverlap=n_overlap, nperseg=n_per_seg,
                   nfft=n_fft, fs=sfreq, window=window)
    x_splits = [arr for arr in np.array_split(x, n_jobs) if arr.size != 0]
    f_spect = parallel(my_spect_func(d, func=func, freq_sl=freq_sl,
                                     average=average)
                       for d in x_splits)
    psds = np.concatenate(f_spect, axis=0)
    shape = dshape + (len(freqs),)
    if average is None:
        shape = shape + (-1,)
    psds.shape = shape
    return psds, freqs
