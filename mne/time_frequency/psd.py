# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np

from ..parallel import parallel_func
from ..io.pick import _pick_data_channels
from ..utils import logger, verbose, _time_mask
from .multitaper import _psd_multitaper


def _pwelch(epoch, noverlap, nfft, fs, freq_mask, welch_fun):
    """Aux function."""
    return welch_fun(epoch, nperseg=nfft, noverlap=noverlap,
                     nfft=nfft, fs=fs)[1][..., freq_mask]


def _compute_psd(data, fmin, fmax, Fs, n_fft, psd, n_overlap, pad_to):
    """Compute the PSD."""
    out = [psd(d, Fs=Fs, NFFT=n_fft, noverlap=n_overlap, pad_to=pad_to)
           for d in data]
    psd = np.array([o[0] for o in out])
    freqs = out[0][1]
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    return psd[:, mask], freqs


def _check_nfft(n, n_fft, n_overlap):
    """Helper to make sure n_fft and n_overlap make sense."""
    n_fft = n if n_fft > n else n_fft
    n_overlap = n_fft - 1 if n_overlap >= n_fft else n_overlap
    return n_fft, n_overlap


def _check_psd_data(inst, tmin, tmax, picks, proj):
    """Helper to do checks on PSD data / pull arrays from inst."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    if not isinstance(inst, (BaseEpochs, BaseRaw, Evoked)):
        raise ValueError('epochs must be an instance of Epochs, Raw, or'
                         'Evoked. Got type {0}'.format(type(inst)))

    time_mask = _time_mask(inst.times, tmin, tmax, sfreq=inst.info['sfreq'])
    if picks is None:
        picks = _pick_data_channels(inst.info, with_ref_meg=False)
    if proj:
        # Copy first so it's not modified
        inst = inst.copy().apply_proj()

    sfreq = inst.info['sfreq']
    if isinstance(inst, BaseRaw):
        start, stop = np.where(time_mask)[0][[0, -1]]
        data, times = inst[picks, start:(stop + 1)]
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()[:, picks][:, :, time_mask]
    elif isinstance(inst, Evoked):
        data = inst.data[picks][:, time_mask]

    return data, sfreq


def _psd_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
               n_jobs=1):
    """Compute power spectral density (PSD) using Welch's method.

    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
    n_jobs : int
        Number of CPUs to use in the computation.

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or
        The power spectral densities. All dimensions up to the last will
        be the same as input.
    freqs : ndarray, shape (n_freqs,)
        The frequencies.
    """
    from scipy.signal import welch
    dshape = x.shape[:-1]
    n_times = x.shape[-1]
    x = x.reshape(-1, n_times)

    # Prep the PSD
    n_fft, n_overlap = _check_nfft(n_times, n_fft, n_overlap)
    win_size = n_fft / float(sfreq)
    logger.info("Effective window size : %0.3f (s)" % win_size)
    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Parallelize across first N-1 dimensions
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs)
    x_splits = np.array_split(x, n_jobs)
    f_psd = parallel(my_pwelch(d, noverlap=n_overlap, nfft=n_fft,
                     fs=sfreq, freq_mask=freq_mask,
                     welch_fun=welch)
                     for d in x_splits)

    # Combining/reshaping to original data shape
    psds = np.concatenate(f_psd, axis=0)
    psds = psds.reshape(np.hstack([dshape, -1]))
    return psds, freqs


@verbose
def psd_welch(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None, n_fft=256,
              n_overlap=0, picks=None, proj=False, n_jobs=1, verbose=None):
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
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
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

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_multitaper,
    csd_epochs

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return _psd_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,
                      n_overlap=n_overlap, n_jobs=n_jobs)


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
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_welch, csd_epochs

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return _psd_multitaper(data, sfreq, fmin=fmin, fmax=fmax,
                           bandwidth=bandwidth, adaptive=adaptive,
                           low_bias=low_bias,
                           normalization=normalization,  n_jobs=n_jobs)
