# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np

from ..parallel import parallel_func
from ..io.proj import make_projector_info
from ..io.pick import pick_types
from ..utils import logger, verbose, _time_mask
from scipy.signal import welch
from .multitaper import multitaper_psd


@verbose
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
    sfreq = raw.info['sfreq']

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(sfreq)))

    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)

    freqs = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    psds = np.array(parallel(my_pwelch([channel],
                                       noverlap=n_overlap, nfft=n_fft,
                                       fs=sfreq, freq_mask=freq_mask,
                                       welch_fun=welch)
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
def compute_epochs_psd(epochs, picks=None, fmin=0, fmax=np.inf,
                       tmin=None, tmax=None, n_fft=256, n_overlap=0,
                       proj=False, method='welch',  mt_bandwidth=None,
                       mt_adaptive=False, mt_low_bias=True,
                       mt_normalization='length', sfreq=None, n_jobs=1,
                       verbose=None):
    """Compute power spectral density with average periodograms.

    Parameters
    ----------
    epochs : instance of Epochs | array of shape(n_epochs, n_channels, n_times)
        The epochs. If not an Epochs object, it must be a numpy array
        with shape (n_epochs, n_channels, n_times). Sampling frequency must
        also be supplied.
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
    method : 'welch' | 'multitaper'
        The method to use in calculating the PSD.
    sfreq : None | float
        The sampling frequency of the signal if it is passed as an array.
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
    from ..epochs import _BaseEpochs
    if isinstance(epochs, _BaseEpochs):
        if sfreq is not None:
            raise ValueError('If Epochs object supplied, sfreq must be None')
        sfreq = epochs.info['sfreq']
        if picks is None:
            picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                               exclude='bads')

        if tmin is not None or tmax is not None:
            time_mask = _time_mask(epochs.times, tmin, tmax)
        else:
            time_mask = slice(None)

        data = epochs.get_data()[:, picks][..., time_mask]
        if proj:
            proj, _ = make_projector_info(epochs.info)
            if picks is not None:
                proj = proj[picks][:, picks]
            data = np.dot(proj, data)

    elif isinstance(epochs, np.ndarray):
        if not isinstance(sfreq, (int, float)):
            raise ValueError('Must give sampling frequency (sfreq)'
                             ' if epochs is an array.')
        if epochs.ndim != 3:
            raise ValueError('epochs array must have 3 dimensions')
        data = np.atleast_3d(epochs)
    else:
        raise ValueError('epochs must be an instance of Epochs or '
                         'a numpy array. Got type {0}'.format(type(epochs)))

    n_epochs, n_channels, n_times = data.shape
    if method == 'welch':
        n_fft, n_overlap = _check_nfft(n_times, n_fft, n_overlap)
        n_fft = int(n_fft)
        win_size = n_fft / float(sfreq)
        logger.info("Effective window size : %0.3f (s)" % win_size)

        freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        psds = np.empty(data.shape[:-1] + (freqs.size,))

        parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                    verbose=verbose)

        data_splits = np.array_split(data, n_jobs)
        f_epochs = parallel(my_pwelch(d, noverlap=n_overlap, nfft=n_fft,
                                      fs=sfreq, freq_mask=freq_mask,
                                      welch_fun=welch)
                            for d in data_splits)
        psds = np.concatenate(f_epochs, axis=0)

    elif method == 'multitaper':
        # Pass each epoch/channel pair separately
        data = data.reshape(n_epochs * n_channels, n_times)
        psds, freqs = multitaper_psd(x=data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                     bandwidth=mt_bandwidth,
                                     adaptive=mt_adaptive,
                                     low_bias=mt_low_bias,
                                     n_jobs=n_jobs,
                                     normalization=mt_normalization,
                                     verbose=verbose)

        n_freqs = psds.shape[1]
        psds = psds.reshape(n_epochs, n_channels, n_freqs)
    else:
        raise ValueError("Unsupported PSD method: {0}".format(method))

    return psds, freqs
