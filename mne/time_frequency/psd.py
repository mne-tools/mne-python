# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np

from ..parallel import parallel_func
from ..io.proj import make_projector_info
from ..io.pick import pick_types
from ..utils import logger, verbose, deprecated, _time_mask


@verbose
@deprecated('This will be deprecated in release v0.13, see psd_ functions.')
def compute_raw_psd(raw, tmin=0., tmax=None, picks=None, fmin=0,
                    fmax=np.inf, n_fft=2048, n_overlap=0,
                    proj=False, n_jobs=1, verbose=None):
    """Compute power spectral density with average periodograms.
    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    tmin : float
        Minimum time instant to consider (in seconds).
    tmax : float | None
        Maximum time instant to consider (in seconds). None will use the
        end of the file.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    proj : bool
        Apply SSP projection vectors.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    Returns
    -------
    psd : array of float
        The PSD for all channels
    freqs: array of float
        The frequencies
    """
    from scipy.signal import welch
    tmax = raw.times[-1] if tmax is None else tmax
    start, stop = raw.time_as_index([tmin, tmax])
    if picks is not None:
        data, times = raw[picks, start:(stop + 1)]
    else:
        data, times = raw[:, start:(stop + 1)]
    n_fft, n_overlap = _check_nfft(len(times), n_fft, n_overlap)

    if proj:
        proj, _ = make_projector_info(raw.info)
        if picks is not None:
            data = np.dot(proj[picks][:, picks], data)
        else:
            data = np.dot(proj, data)

    n_fft = int(n_fft)
    Fs = raw.info['sfreq']

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(Fs)))

    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)

    freqs = np.arange(n_fft // 2 + 1) * (Fs / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    psds = np.array(parallel(my_pwelch([channel],
                                       noverlap=n_overlap, nfft=n_fft, fs=Fs,
                                       freq_mask=freq_mask, welch_fun=welch)
                             for channel in data))[:, 0, :]

    return psds, freqs


def _pwelch(epoch, noverlap, nfft, fs, freq_mask, welch_fun):
    """Aux function"""
    return welch_fun(epoch, nperseg=nfft, noverlap=noverlap,
                     nfft=nfft, fs=fs)[1][..., freq_mask]


def _compute_psd(data, fmin, fmax, Fs, n_fft, psd, n_overlap, pad_to):
    """Compute the PSD"""
    out = [psd(d, Fs=Fs, NFFT=n_fft, noverlap=n_overlap, pad_to=pad_to)
           for d in data]
    psd = np.array([o[0] for o in out])
    freqs = out[0][1]
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    return psd[:, mask], freqs


def _check_nfft(n, n_fft, n_overlap):
    """Helper to make sure n_fft and n_overlap make sense"""
    n_fft = n if n_fft > n else n_fft
    n_overlap = n_fft - 1 if n_overlap >= n_fft else n_overlap
    return n_fft, n_overlap


def _check_psd_data(inst, tmin, tmax, picks, proj):
    """Helper to do checks on PSD data / pull arrays from inst"""
    from ..epochs import _BaseEpochs
    from ..io.base import _BaseRaw
    if not isinstance(inst, (_BaseEpochs, _BaseRaw)):
        raise ValueError('epochs must be an instance of Epochs or Raw.'
                         ' Got type {0}'.format(type(inst)))

    if tmin is not None or tmax is not None:
        time_mask = _time_mask(inst.times, tmin, tmax)
    else:
        time_mask = slice(None)
    if picks is None:
        picks = pick_types(inst.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    if proj:
        inst = inst.apply_proj()

    sfreq = inst.info['sfreq']
    if isinstance(inst, _BaseRaw):
        start, stop = inst.time_as_index([tmin, tmax])
        data, times = inst[picks, start:(stop + 1)]
    elif isinstance(inst, _BaseEpochs):
        data = inst.get_data()[:, picks][..., time_mask]

    return data, sfreq


@verbose
def _psd_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
               n_jobs=1, verbose=None):
    """Helper function for calculating Welch PSD."""
    from scipy.signal import welch
    dshape = x.shape[:-1]
    n_times = x.shape[-1]
    x = x.reshape(np.product(dshape), -1)

    # Prep the PSD
    n_fft, n_overlap = _check_nfft(n_times, n_fft, n_overlap)
    win_size = n_fft / float(sfreq)
    logger.info("Effective window size : %0.3f (s)" % win_size)
    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Parallelize across first N-1 dimensions
    psds = np.empty(x.shape[:-1] + (freqs.size,))
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)
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
    """Compute the PSD using Welch's method.

    Parameters
    ----------
    inst : instance of Epochs or Raw
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
        to be <= n_fft.
        bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    psds : ndarray, shape ([n_epochs], n_channels, n_freqs)
        The power spectral densities. If Raw is provided,
        then psds will be 2-D.
    freqs : ndarray (n_freqs)
        The frequencies.
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    psds, freqs = _psd_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,
                             n_overlap=n_overlap, n_jobs=n_jobs,
                             verbose=verbose)
    return psds, freqs


def _psd_multitaper(x, sfreq, fmin=0, fmax=np.inf, bandwidth=None,
                    adaptive=False, low_bias=True, normalization='length',
                    n_jobs=1, verbose=None):
    """Helper function for calculating Multitaper PSD."""
    from .multitaper import multitaper_psd
    dshape = x.shape[:-1]
    x = x.reshape(np.product(dshape), -1)

    # Stack data so it's treated separately
    psds, freqs = multitaper_psd(x=x, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                 bandwidth=bandwidth, adaptive=adaptive,
                                 low_bias=low_bias,
                                 normalization=normalization, n_jobs=n_jobs,
                                 verbose=verbose)

    # Combining/reshaping to original data shape
    psds = psds.reshape(np.hstack([dshape, -1]))
    return psds, freqs


@verbose
def psd_multitaper(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                   bandwidth=None, adaptive=False, low_bias=True,
                   normalization='length', picks=None, proj=False,
                   n_jobs=1, verbose=None):
    """Compute the PSD using multitapers.


    Parameters
    ----------
    inst : instance of Epochs or Raw |
           array of shape ([n_epochs], n_channels, n_times)
        The data. If not an instance of Epochs or Raw,
        sfreq must also be supplied.
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
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
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    psds : ndarray, shape ([n_epochs], n_channels, n_freqs)
        The power spectral densities. If Raw is provided,
        then psds will be 2-D.
    freqs : ndarray, shape (n_freqs)
        The frequencies.
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    psds, freqs = _psd_multitaper(data, sfreq, fmin=fmin, fmax=fmax,
                                  bandwidth=bandwidth, adaptive=adaptive,
                                  low_bias=low_bias,
                                  normalization=normalization,  n_jobs=n_jobs,
                                  verbose=verbose)
    return psds, freqs


@verbose
@deprecated('This will be deprecated in release v0.13, see psd_ functions.')
def compute_epochs_psd(epochs, picks=None, fmin=0, fmax=np.inf, tmin=None,
                       tmax=None, n_fft=256, n_overlap=0, proj=False,
                       n_jobs=1, verbose=None):
    """Compute power spectral density with average periodograms.
    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
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
        If ``n_fft > len(epochs.times)``, it will be adjusted down to
        ``len(epochs.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft.
    proj : bool
        Apply SSP projection vectors.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    Returns
    -------
    psds : ndarray (n_epochs, n_channels, n_freqs)
        The power spectral densities.
    freqs : ndarray, shape (n_freqs)
        The frequencies.
    """
    from scipy.signal import welch
    n_fft = int(n_fft)
    Fs = epochs.info['sfreq']
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    n_fft, n_overlap = _check_nfft(len(epochs.times), n_fft, n_overlap)

    if tmin is not None or tmax is not None:
        time_mask = _time_mask(epochs.times, tmin, tmax)
    else:
        time_mask = slice(None)

    data = epochs.get_data()[:, picks][..., time_mask]
    if proj:
        proj, _ = make_projector_info(epochs.info)
        if picks is not None:
            data = np.dot(proj[picks][:, picks], data)
        else:
            data = np.dot(proj, data)

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(Fs)))

    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (Fs / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    psds = np.empty(data.shape[:-1] + (freqs.size,))

    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)

    for idx, fepochs in zip(np.array_split(np.arange(len(data)), n_jobs),
                            parallel(my_pwelch(epoch, noverlap=n_overlap,
                                               nfft=n_fft, fs=Fs,
                                               freq_mask=freq_mask,
                                               welch_fun=welch)
                                     for epoch in np.array_split(data,
                                                                 n_jobs))):
        for i_epoch, f_epoch in zip(idx, fepochs):
            psds[i_epoch, :, :] = f_epoch

    return psds, freqs
