# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn

import numpy as np
from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

from .utils import check_idx
from ..time_frequency.multitaper import dpss_windows, _mt_spectra,\
                                        _psd_from_mt, _csd_from_mt,\
                                        _psd_from_mt_adaptive
from .. import verbose


@verbose
def coherency(data, idx=None, sfreq=2*np.pi, fmin=0, fmax=np.inf,
              bandwidth=None, adaptive=False, low_bias=True,
              verbose=None):

    """Compute coherency between signals using a multi-taper method.

    The computed coherency is given by

    .. math:: C_{XY}(f) = \frac{\langle S_{XY}(f)\rangle}
                               {\sqrt{\langle S_{XX}(f) \rangle
                                      \langle S_{YY}(f) \rangle}}

    Where the cross spectral density :math:`S_{XY}(f)` and the power
    spectral densities :math:`S_{XX}(f), S_{YY}(f)` are computed using
    a multi-taper method. The average :math:`\langle \cdot \rangle` is
    computed over epochs.

    Note: By default, the coherency between all signals is computed. If
    one is only interested in the connectivity between some signals, the
    "idx" parameter can be used. For example, to compute the coherency
    between the signal with index 0 and signals "2, 3, 4" (a total of 3
    connections) one can use the following:

    idx = (np.array([0, 0, 0],    # row indices
           np.array([2, 3, 4])))  # col indices

    coh_flat = coherency(data, idx=idx, ...)

    This is equivalent, but more efficient, to

    coh = coherenct(data, idx=None, ...)
    coh_flat = coh[idx]  # idx defined above

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times)
        The data from which to compute coherency.
    idx : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    coh : array, shape=(n_signals, n_signals, n_freq) or
                 shape=(len(idx[0]), n_freq) if idx is not None
        Computed coherency.
    freqs : array
        Frequency points at which the coherency was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used.
    """

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays

    logger.info('Coherency computation...')
    for epoch_idx, data_i in enumerate(data):

        if epoch_idx == 0:
            # initialize things
            n_signals, n_times = data_i.shape

            # compute standardized half-bandwidth
            if bandwidth is not None:
                half_nbw = float(bandwidth) * n_times / (2 * sfreq)
            else:
                half_nbw = 4

            # compute dpss windows
            n_tapers_max = int(2 * half_nbw)
            dpss, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                         low_bias=low_bias)
            n_tapers = len(eigvals)
            logger.info('    using %d DPSS windows' % n_tapers)

            if adaptive and len(eigvals) < 3:
                warn('Not adaptively combining the spectral estimators '
                     'due to a low number of tapers.')
                adaptive = False

            if idx is None:
                # only compute r for lower-triangular region
                idx_use = np.tril_indices(n_signals, -1)
            else:
                idx_use = check_idx(idx)

            # number of connectivities to compute
            n_con = len(idx_use[0])

            logger.info('    computing coh for %d connections' % n_con)

            # decide which frequencies to keep
            freqs = fftfreq(n_times, 1. / sfreq)
            freqs = freqs[freqs >= 0]
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[freq_mask]
            n_freqs = np.sum(freq_mask)
            logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                        % (freqs[0], freqs[-1], n_freqs))

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[idx_use[0], idx_use[1]])

            # map idx to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in idx_use]

            # space to accumulate psd and csd over epochs
            psd = np.zeros((len(sig_idx), n_freqs))
            csd = np.zeros((n_con, n_freqs), dtype=np.complex128)

        # epoch processing starts here
        if data_i.shape != (n_signals, n_times):
            raise ValueError('all epochs must have the same shape')

        logger.info('    processing epoch %d' % (epoch_idx + 1))

        # compute tapered spectra
        x_mt, _ = _mt_spectra(data_i[sig_idx], dpss, sfreq)

        if not adaptive:
            x_mt = x_mt[:, :, freq_mask]
            weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
            this_psd = _psd_from_mt(x_mt, weights)

            psd += this_psd

            for i in xrange(n_con):
                csd[i] += _csd_from_mt(x_mt[idx_map[0][i]],
                                       x_mt[idx_map[1][i]],
                                       weights, weights)[0]
        else:
            # compute PSD and adaptive weights
            this_psd, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                                      return_weights=True)

            psd += this_psd

            # only keep freqs of interest
            x_mt = x_mt[:, :, freq_mask]

            # compute CSD
            for i in xrange(n_con):
                csd[i] += _csd_from_mt(x_mt[idx_map[0][i]],
                                       x_mt[idx_map[1][i]],
                                       weights[idx_map[0][i]],
                                       weights[idx_map[1][i]])

    # normalize
    n_epochs = epoch_idx + 1
    psd /= n_epochs
    csd /= n_epochs

    # compute coherency
    coh = np.empty((n_con, n_freqs), dtype=np.complex128)
    for i in range(n_con):
        coh[i] = csd[i] / np.sqrt(psd[idx_map[0][i]] * psd[idx_map[1][i]])

    if idx is None:
        # return all-to-all connectivity matrix
        logger.info('    assembling 3D connectivity matrix')
        coh_flat = coh
        coh = np.ones((n_signals, n_signals, n_freqs), dtype=np.complex128)

        coh[idx_use] = coh_flat
        coh[(idx_use[1], idx_use[0])] = coh_flat.conj()  # reverse connections

    logger.info('[done]')

    return coh, freqs, n_epochs, n_tapers
