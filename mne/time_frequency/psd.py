# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np

from ..parallel import parallel_func
from ..io.proj import make_projector_info
from ..io.pick import pick_types
from ..utils import logger, verbose


@verbose
def compute_raw_psd(raw, tmin=0, tmax=np.inf, picks=None,
                    fmin=0, fmax=np.inf, n_fft=2048, pad_to=None, n_overlap=0,
                    n_jobs=1, plot=False, proj=False, NFFT=None,
                    verbose=None):
    """Compute power spectral density with average periodograms.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
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
    pad_to : int | None
        The number of points to which the data segment is padded when
        performing the FFT. If None, pad_to equals `NFFT`.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    n_jobs : int
        Number of CPUs to use in the computation.
    plot : bool
        Plot each PSD estimates
    proj : bool
        Apply SSP projection vectors
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    psd : array of float
        The PSD for all channels
    freqs: array of float
        The frequencies
    """
    if NFFT is not None:
        n_fft = NFFT
        warnings.warn("`NFFT` is deprecated and will be removed in v0.9. "
                      "Use `n_fft` instead")
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
    parallel, my_psd, n_jobs = parallel_func(plt.psd, n_jobs)
    fig = plt.figure()
    out = parallel(my_psd(d, Fs=Fs, NFFT=n_fft, noverlap=n_overlap,
                          pad_to=pad_to) for d in data)
    if not plot:
        plt.close(fig)
    freqs = out[0][1]
    psd = np.array([o[0] for o in out])

    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    psd = psd[:, mask]

    return psd, freqs


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
def compute_epochs_psd(epochs, picks=None, fmin=0, fmax=np.inf, n_fft=256,
                       pad_to=None, n_overlap=0, n_jobs=1, verbose=None):
    """Compute power spectral density with with average periodograms.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    tmin : float
        Min time instant to consider
    tmax : float
        Max time instant to consider
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
    pad_to : int | None
        The number of points to which the data segment is padded when
        performing the FFT. If None, pad_to equals `n_fft`.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
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

    logger.info("Effective window size : %0.3f (s)" % (n_fft / float(Fs)))
    psds = []
    import matplotlib.pyplot as plt
    parallel, my_psd, n_jobs = parallel_func(_compute_psd, n_jobs)
    fig = plt.figure()  # threading will induce errors otherwise
    out = parallel(my_psd(data[picks], fmin, fmax, Fs, n_fft, plt.psd,
                          n_overlap, pad_to)
                   for data in epochs)
    plt.close(fig)
    psds = [o[0] for o in out]
    freqs = [o[1] for o in out]
    return np.array(psds), freqs[0]
