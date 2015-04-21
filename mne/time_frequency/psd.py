# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np

from ..parallel import parallel_func
from ..io.proj import make_projector_info
from ..io.pick import pick_types
from ..utils import logger, verbose


@verbose
def compute_raw_psd(raw, tmin=0., tmax=np.inf, picks=None, fmin=0,
                    fmax=np.inf, n_fft=2048, n_overlap=0,
                    proj=False, n_jobs=1, verbose=None):
    """Compute power spectral density with average periodograms.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    tmin : float
        Minimum time instant to consider (in seconds).
    tmax : float
        Maximum time instant to consider (in seconds).
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


@verbose
def compute_epochs_psd(epochs, picks=None, fmin=0, fmax=np.inf, n_fft=256,
                       n_overlap=0, proj=False, n_jobs=1, verbose=None):
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
    freqs : ndarray (n_freqs)
        The frequencies.
    """
    from scipy.signal import welch
    n_fft = int(n_fft)
    Fs = epochs.info['sfreq']
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    n_fft, n_overlap = _check_nfft(len(epochs.times), n_fft, n_overlap)

    data = epochs.get_data()[:, picks]
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
