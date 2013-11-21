# Author : Alexandre Gramfort, gramfort@nmr.mgh.harvard.edu (2011)
# License : BSD 3-clause

import numpy as np

import logging
logger = logging.getLogger('mne')

# XXX : don't import pylab here or you will break the doc

from ..parallel import parallel_func
from ..fiff.proj import make_projector_info
from .. import verbose


@verbose
def compute_raw_psd(raw, tmin=0, tmax=np.inf, picks=None,
                    fmin=0, fmax=np.inf, NFFT=2048, n_jobs=1,
                    plot=False, proj=False, verbose=None):
    """Compute power spectral density with multi-taper

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    tmin : float
        Min time instant to consider

    tmax : float
        Max time instant to consider

    picks : None or array of integers
        The selection of channels to include in the computation.
        If None, take all channels.

    fmin : float
        Min frequency of interest

    fmax : float
        Max frequency of interest

    NFFT : int
        The length of the tappers ie. the windows. The smaller
        it is the smoother are the PSDs.

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

    NFFT = int(NFFT)
    Fs = raw.info['sfreq']

    logger.info("Effective window size : %0.3f (s)" % (NFFT / float(Fs)))

    import pylab as pl
    parallel, my_psd, n_jobs = parallel_func(pl.psd, n_jobs)
    fig = pl.figure()
    out = parallel(my_psd(d, Fs=Fs, NFFT=NFFT) for d in data)
    if not plot:
        pl.close(fig)
    freqs = out[0][1]
    psd = np.array(zip(*out)[0])

    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    psd = psd[:, mask]

    return psd, freqs
