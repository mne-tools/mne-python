# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
from ..externals.six.moves import zip
import copy

import numpy as np

from ..utils import logger, verbose
from .spectral import spectral_connectivity


@verbose
def phase_slope_index(data, indices=None, sfreq=2 * np.pi,
                      mode='multitaper', fmin=None, fmax=np.inf,
                      tmin=None, tmax=None, mt_bandwidth=None,
                      mt_adaptive=False, mt_low_bias=True,
                      cwt_frequencies=None, cwt_n_cycles=7, block_size=1000,
                      n_jobs=1, verbose=None):
    """Compute the Phase Slope Index (PSI) connectivity measure.

    The PSI is an effective connectivity measure, i.e., a measure which can
    give an indication of the direction of the information flow (causality).
    For two time series, and one computes the PSI between the first and the
    second time series as follows

    indices = (np.array([0]), np.array([1]))
    psi = phase_slope_index(data, indices=indices, ...)

    A positive value means that time series 0 is ahead of time series 1 and
    a negative value means the opposite.

    The PSI is computed from the coherency (see spectral_connectivity), details
    can be found in [1].

    References
    ----------
    [1] Nolte et al. "Robustly Estimating the Flow Direction of Information in
    Complex Physical Systems", Physical Review Letters, vol. 100, no. 23,
    pp. 1-4, Jun. 2008.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times)
        Can also be a list/generator of array, shape =(n_signals, n_times);
        list/generator of SourceEstimate; or Epochs.
        The data from which to compute connectivity. Note that it is also
        possible to combine multiple signals by providing a list of tuples,
        e.g., data = [(arr_0, stc_0), (arr_1, stc_1), (arr_2, stc_2)],
        corresponds to 3 epochs, and arr_* could be an array with the same
        number of time points as stc_*.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    mode : str
        Spectrum estimation mode can be either: 'multitaper', 'fourier', or
        'cwt_morlet'.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
        If None the frequency corresponding to an epoch length of 5 cycles
        is used.
    fmax : float | tuple of floats
        The upper frequency of interest. Multiple bands are dedined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    tmin : float | None
        Time to start connectivity estimation.
    tmax : float | None
        Time to end connectivity estimation.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    cwt_frequencies : array
        Array of frequencies of interest. Only used in 'cwt_morlet' mode.
    cwt_n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency. Only used in
        'cwt_morlet' mode.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psi : array
        Computed connectivity measure(s). The shape of each array is either
        (n_signals, n_signals, n_bands) mode: 'multitaper' or 'fourier'
        (n_signals, n_signals, n_bands, n_times) mode: 'cwt_morlet'
        when "indices" is None, or
        (n_con, n_bands) mode: 'multitaper' or 'fourier'
        (n_con, n_bands, n_times) mode: 'cwt_morlet'
        when "indices" is specified and "n_con = len(indices[0])".
    freqs : array
        Frequency points at which the connectivity was computed.
    times : array
        Time points for which the connectivity was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used. Only defined in 'multitaper' mode.
        Otherwise None is returned.
    """
    logger.info('Estimating phase slope index (PSI)')
    # estimate the coherency
    cohy, freqs_, times, n_epochs, n_tapers = spectral_connectivity(
        data, method='cohy', indices=indices, sfreq=sfreq, mode=mode,
        fmin=fmin, fmax=fmax, fskip=0, faverage=False, tmin=tmin, tmax=tmax,
        mt_bandwidth=mt_bandwidth, mt_adaptive=mt_adaptive,
        mt_low_bias=mt_low_bias, cwt_frequencies=cwt_frequencies,
        cwt_n_cycles=cwt_n_cycles, block_size=block_size, n_jobs=n_jobs,
        verbose=verbose)

    logger.info('Computing PSI from estimated Coherency')
    # compute PSI in the requested bands
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    bands = list(zip(np.asarray((fmin,)).ravel(), np.asarray((fmax,)).ravel()))
    n_bands = len(bands)

    freq_dim = -2 if mode == 'cwt_morlet' else -1

    # allocate space for output
    out_shape = list(cohy.shape)
    out_shape[freq_dim] = n_bands
    psi = np.zeros(out_shape, dtype=np.float)

    # allocate accumulator
    acc_shape = copy.copy(out_shape)
    acc_shape.pop(freq_dim)
    acc = np.empty(acc_shape, dtype=np.complex128)

    freqs = list()
    idx_fi = [slice(None)] * cohy.ndim
    idx_fj = [slice(None)] * cohy.ndim
    for band_idx, band in enumerate(bands):
        freq_idx = np.where((freqs_ > band[0]) & (freqs_ < band[1]))[0]
        freqs.append(freqs_[freq_idx])

        acc.fill(0.)
        for fi, fj in zip(freq_idx, freq_idx[1:]):
            idx_fi[freq_dim] = fi
            idx_fj[freq_dim] = fj
            acc += np.conj(cohy[idx_fi]) * cohy[idx_fj]

        idx_fi[freq_dim] = band_idx
        psi[idx_fi] = np.imag(acc)
    logger.info('[PSI Estimation Done]')

    return psi, freqs, times, n_epochs, n_tapers
