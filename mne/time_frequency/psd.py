# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np
from scipy.signal import welch

from ..parallel import parallel_func
from ..io.proj import make_projector_info
from ..io.pick import pick_types
from ..utils import logger, verbose


@verbose
def compute_raw_psd(raw, tmin=0., tmax=np.inf, fmin=0, fmax=np.inf,
                    proj=False, n_fft=2048, window_size=2048, n_overlap=0,
                    picks=None, n_jobs=1, verbose=None):
    """Compute power spectral density with average periodograms.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    tmin : float
        Minimum time instant to consider (in seconds).
    tmax : float
        Maximum time instant to consider (in seconds).
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    proj : bool
        Apply SSP projection vectors
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs.
    window_size : int, optional
        Length of each window.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
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
    start, stop = raw.time_as_index([tmin, tmax])
    if picks is not None:
        data, times = raw[picks, start:(stop + 1)]
    else:
        data, times = raw[:, start:(stop + 1)]

    if proj:
        proj, _ = make_projector_info(raw.info)
        if picks is not None:
            data = np.dot(proj[picks][:, picks], data)
        else:
            data = np.dot(proj, data)

    n_fft = int(n_fft)
    Fs = raw.info['sfreq']

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(Fs)))

    import matplotlib.pyplot as plt
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)

    out = np.array(parallel(my_pwelch([channel], window_size=window_size,
                   noverlap=n_overlap, nfft=n_fft, fs=Fs)
                   for channel in data))

    psds = out[:, 0, 1, :]
    freqs = out[0, 0, 0]

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    psds = psds[:, freq_mask]

    return psds, freqs


def _pwelch(epoch, window_size, noverlap, nfft, fs):
    return [welch(channel, nperseg=window_size, noverlap=noverlap,
                  nfft=nfft, fs=fs)
            for channel in epoch]


def _compute_psd(data, fmin, fmax, Fs, n_fft, psd, n_overlap, pad_to):
    """Compute the PSD"""
    out = [psd(d, Fs=Fs, NFFT=n_fft, noverlap=n_overlap, pad_to=pad_to)
           for d in data]
    psd = np.array([o[0] for o in out])
    freqs = out[0][1]
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    return psd[:, mask], freqs


@verbose
def compute_epochs_psd(epochs, picks=None, fmin=0, fmax=np.inf, n_fft=2048,
                       window_size=256, n_overlap=128, proj=False, n_jobs=1,
                       verbose=None):
    """Compute power spectral density with with average periodograms.

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
        it is the smoother are the PSDs. The default value is 2048.
    window_size : int, optional
        Length of each window. The default value is 256.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 128 (window_size // 2).
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

    n_fft = int(n_fft)
    Fs = epochs.info['sfreq']
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

    if proj:
        proj, _ = make_projector_info(raw.info)
        if picks is not None:
            data = np.dot(proj[picks][:, picks], data)
        else:
            data = np.dot(proj, data)

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(Fs)))

    import matplotlib.pyplot as plt
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs,
                                                verbose=verbose)

    psds = np.empty(epochs.get_data()[:, picks].shape[:-1] +
                    (n_fft // 2 + 1,))
    freqs = np.arange(psds.shape[-1]) * (Fs / n_fft)
        
    for i_epoch, fepoch in enumerate(parallel(
        my_pwelch(epoch[:, picks], window_size=window_size,
                  noverlap=n_overlap, nfft=n_fft, fs=Fs)
                  for epoch in np.array_split(epochs.get_data(), n_jobs))):

        for i_channel, fchannel in enumerate(fepoch):
            if i_epoch == 0:
                psds[len(fepoch) * i_epoch + i_channel, :, :] = fchannel[1]
            else:
                psds[i_iter * i_epoch + i_channel, :, :] = fchannel[1]
        i_iter = len(fepoch)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    psds = psds[..., freq_mask]

    return psds, freqs
