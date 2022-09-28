# Author : Martin Luessi mluessi@nmr.mgh.harvard.edu (2012)
# License : BSD-3-Clause

# Parts of this code were copied from NiTime http://nipy.sourceforge.net/nitime

import operator

import numpy as np

from ..filter import next_fast_len
from ..parallel import parallel_func
from ..utils import sum_squared, warn, verbose, logger, _check_option


def dpss_windows(N, half_nbw, Kmax, low_bias=True, interp_from=None,
                 interp_kind='linear'):
    """Compute Discrete Prolate Spheroidal Sequences.

    Will give of orders [0,Kmax-1] for a given frequency-spacing multiple
    NW and sequence length N.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    N : int
        Sequence length.
    half_nbw : float
        Standardized half bandwidth corresponding to 2 * half_bw = BW*f0
        = BW*N/dt but with dt taken as 1.
    Kmax : int
        Number of DPSS windows to return is Kmax (orders 0 through Kmax-1).
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.

        .. note:: If SciPy 1.1 or greater is available, interpolating
                  is likely not necessary as DPSS computations should be
                  sufficiently fast.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.

    Returns
    -------
    v, e : tuple,
        The v array contains DPSS windows shaped (Kmax, N).
        e are the eigenvalues.

    Notes
    -----
    Tridiagonal form of DPSS calculation from :footcite:`Slepian1978`.

    References
    ----------
    .. footbibliography::
    """
    from scipy import interpolate
    from scipy.fft import rfft, irfft
    from scipy.signal.windows import dpss as sp_dpss

    # This np.int32 business works around a weird Windows bug, see
    # gh-5039 and https://github.com/scipy/scipy/pull/8608
    Kmax = np.int32(operator.index(Kmax))
    N = np.int32(operator.index(N))
    W = float(half_nbw) / N
    nidx = np.arange(N, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % N
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, half_nbw, Kmax, low_bias=False)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            tmp = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = tmp(np.linspace(0, this_d.shape[-1] - 1, N,
                                     endpoint=False))

            # Rescale:
            d_temp = d_temp / np.sqrt(sum_squared(d_temp))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        dpss = sp_dpss(N, half_nbw, Kmax)

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390

    # compute autocorr using FFT (same as nitime.utils.autocorr(dpss) * N)
    rxx_size = 2 * N - 1
    n_fft = next_fast_len(rxx_size)
    dpss_fft = rfft(dpss, n_fft)
    dpss_rxx = irfft(dpss_fft * dpss_fft.conj(), n_fft)
    dpss_rxx = dpss_rxx[:, :N]

    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warn('Could not properly use low_bias, keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def _psd_from_mt_adaptive(x_mt, eigvals, freq_mask, max_iter=150,
                          return_weights=False):
    r"""Use iterative procedure to compute the PSD from tapered spectra.

    .. note:: Modified from NiTime.

    Parameters
    ----------
    x_mt : array, shape=(n_signals, n_tapers, n_freqs)
       The DFTs of the tapered sequences (only positive frequencies)
    eigvals : array, length n_tapers
       The eigenvalues of the DPSS tapers
    freq_mask : array
        Frequency indices to keep
    max_iter : int
       Maximum number of iterations for weight computation
    return_weights : bool
       Also return the weights

    Returns
    -------
    psd : array, shape=(n_signals, np.sum(freq_mask))
        The computed PSDs
    weights : array shape=(n_signals, n_tapers, np.sum(freq_mask))
        The weights used to combine the tapered spectra

    Notes
    -----
    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`
    """
    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError('Need one eigenvalue for each taper')

    if n_tapers < 3:
        raise ValueError('Not enough tapers to compute adaptive weights.')

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = np.trapz(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
    del psd_est

    # allocate space for output
    psd = np.empty((n_signals, np.sum(freq_mask)))

    # only keep the frequencies of interest
    x_mt = x_mt[:, :, freq_mask]

    if return_weights:
        weights = np.empty((n_signals, n_tapers, psd.shape[1]))

    for i, (xk, var) in enumerate(zip(x_mt, x_var)):
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = _psd_from_mt(xk[:2, :], rt_eig[:2, np.newaxis])

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = (psd_iter / (eigvals[:, np.newaxis] * psd_iter +
                               (1 - eigvals[:, np.newaxis]) * var))
            d_k *= rt_eig[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged
            err -= d_k
            if np.max(np.mean(err ** 2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        if n == max_iter - 1:
            warn('Iterative multi-taper PSD computation did not converge.')

        psd[i, :] = psd_iter

        if return_weights:
            weights[i, :, :] = d_k

    if return_weights:
        return psd, weights
    else:
        return psd


def _psd_from_mt(x_mt, weights):
    """Compute PSD from tapered spectra.

    Parameters
    ----------
    x_mt : array
        Tapered spectra
    weights : array
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array
        The computed PSD
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd


def _csd_from_mt(x_mt, y_mt, weights_x, weights_y):
    """Compute CSD from tapered spectra.

    Parameters
    ----------
    x_mt : array
        Tapered spectra for x
    y_mt : array
        Tapered spectra for y
    weights_x : array
        Weights used to combine the tapered spectra of x_mt
    weights_y : array
        Weights used to combine the tapered spectra of y_mt

    Returns
    -------
    psd: array
        The computed PSD
    """
    csd = np.sum(weights_x * x_mt * (weights_y * y_mt).conj(), axis=-2)
    denom = (np.sqrt((weights_x * weights_x.conj()).real.sum(axis=-2)) *
             np.sqrt((weights_y * weights_y.conj()).real.sum(axis=-2)))
    csd *= 2 / denom
    return csd


def _mt_spectra(x, dpss, sfreq, n_fft=None):
    """Compute tapered spectra.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        Input signal
    dpss : array, shape=(n_tapers, n_times)
        The tapers
    sfreq : float
        The sampling frequency
    n_fft : int | None
        Length of the FFT. If None, the number of samples in the input signal
        will be used.

    Returns
    -------
    x_mt : array, shape=(..., n_tapers, n_times)
        The tapered spectra
    freqs : array
        The frequency points in Hz of the spectra
    """
    from scipy.fft import rfft, rfftfreq
    if n_fft is None:
        n_fft = x.shape[-1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = rfftfreq(n_fft, 1. / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)),
                    dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.)
    return x_mt, freqs


@verbose
def _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive,
                       interp_from=None, verbose=None):
    """Triage windowing and multitaper parameters."""
    # Compute standardized half-bandwidth
    from scipy.signal import get_window
    if isinstance(bandwidth, str):
        logger.info('    Using standard spectrum estimation with "%s" window'
                    % (bandwidth,))
        window_fun = get_window(bandwidth, n_times)[np.newaxis]
        return window_fun, np.ones(1), False

    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / (2. * sfreq)
    else:
        half_nbw = 4.
    if half_nbw < 0.5:
        raise ValueError(
            'bandwidth value %s yields a normalized bandwidth of %s < 0.5, '
            'use a value of at least %s'
            % (bandwidth, half_nbw, sfreq / n_times))

    # Compute DPSS windows
    n_tapers_max = int(2 * half_nbw)
    window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                       low_bias=low_bias,
                                       interp_from=interp_from)
    logger.info('    Using multitaper spectrum estimation with %d DPSS '
                'windows' % len(eigvals))

    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators due to a '
             'low number of tapers (%s < 3).' % (len(eigvals),))
        adaptive = False

    return window_fun, eigvals, adaptive


@verbose
def psd_array_multitaper(x, sfreq, fmin=0.0, fmax=np.inf, bandwidth=None,
                         adaptive=False, low_bias=True, normalization='length',
                         output='power', n_jobs=None, verbose=None):
    r"""Compute power spectral density (PSD) using a multi-taper method.

    The power spectral density is computed with DPSS
    tapers\ :footcite:p:`Slepian1978`.

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
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    %(normalization)s
    output : str
        The format of the returned ``psds`` array. Can be either ``'complex'``
        or ``'power'``. If ``'power'``, the power spectral density is returned.
        If ``output='complex'``, the complex fourier coefficients are returned
        per taper.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or (..., n_tapers, n_freqs)
        The power spectral densities. All dimensions up to the last (or the
        last two if ``output='complex'``) will be the same as input.
    freqs : array
        The frequency points in Hz of the PSD.
    weights : ndarray
        The weights used for averaging across tapers. Only returned if
        ``output='complex'``.

    See Also
    --------
    mne.io.Raw.plot_psd
    mne.Epochs.plot_psd
    csd_multitaper
    psd_multitaper

    Notes
    -----
    .. versionadded:: 0.14.0

    References
    ----------
    .. footbibliography::
    """
    from scipy.fft import rfftfreq
    _check_option('normalization', normalization, ['length', 'full'])

    # Reshape data so its 2-D for parallelization
    ndim_in = x.ndim
    x = np.atleast_2d(x)
    n_times = x.shape[-1]
    dshape = x.shape[:-1]
    x = x.reshape(-1, n_times)

    dpss, eigvals, adaptive = _compute_mt_params(
        n_times, sfreq, bandwidth, low_bias, adaptive)
    n_tapers = len(dpss)
    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]

    # decide which frequencies to keep
    freqs = rfftfreq(n_times, 1. / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    if output == 'complex':
        psd = np.zeros((x.shape[0], n_tapers, n_freqs), dtype='complex')
    else:
        psd = np.zeros((x.shape[0], n_freqs))

    # Let's go in up to 50 MB chunks of signals to save memory
    n_chunk = max(50000000 // (len(freq_mask) * len(eigvals) * 16), 1)
    offsets = np.concatenate((np.arange(0, x.shape[0], n_chunk), [x.shape[0]]))
    for start, stop in zip(offsets[:-1], offsets[1:]):
        x_mt = _mt_spectra(x[start:stop], dpss, sfreq)[0]
        if output == 'power':
            if not adaptive:
                psd[start:stop] = _psd_from_mt(x_mt[:, :, freq_mask], weights)
            else:
                parallel, my_psd_from_mt_adaptive, n_jobs = \
                    parallel_func(_psd_from_mt_adaptive, n_jobs)
                n_splits = min(stop - start, n_jobs)
                out = parallel(my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                               for x in np.array_split(x_mt, n_splits))
                psd[start:stop] = np.concatenate(out)
        else:
            psd[start:stop] = x_mt[:, :, freq_mask]

    if normalization == 'full':
        psd /= sfreq

    # Combining/reshaping to original data shape
    last_dims = (n_freqs,) if output == 'power' else (n_tapers, n_freqs)
    psd.shape = dshape + last_dims
    if ndim_in == 1:
        psd = psd[0]

    if output == 'complex':
        return psd, freqs, weights
    else:
        return psd, freqs


@verbose
def tfr_array_multitaper(epoch_data, sfreq, freqs, n_cycles=7.0,
                         zero_mean=True, time_bandwidth=None, use_fft=True,
                         decim=1, output='complex', n_jobs=None,
                         verbose=None):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers.

    Same computation as `~mne.time_frequency.tfr_multitaper`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float | int
        Sampling frequency of the data.
    freqs : array-like of float, shape (n_freqs,)
        The frequencies.
    n_cycles : float | array of float
        Number of cycles in the wavelet. Fixed number or one per
        frequency. Defaults to 7.0.
    zero_mean : bool
        If True, make sure the wavelets have a mean of zero. Defaults to True.
    time_bandwidth : float
        If None, will be set to 4.0 (3 tapers). Time x (Full) Bandwidth
        product. The number of good tapers (low-bias) is chosen automatically
        based on this to equal floor(time_bandwidth - 1). Defaults to None.
    use_fft : bool
        Use the FFT for convolutions or not. Defaults to True.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition. Defaults to 1.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.
    output : str, default 'complex'

        * 'complex' : single trial per taper complex values.
        * 'power' : single trial power.
        * 'phase' : single trial per taper phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels. Defaults to 1.
    %(verbose)s

    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If ``output in ['complex',
        'phase']``, then the shape of ``out`` is ``(n_epochs, n_chans,
        n_tapers, n_freqs, n_times)``; if output is 'power', the shape of
        ``out`` is ``(n_epochs, n_chans, n_freqs, n_times)``, else it is
        ``(n_chans, n_freqs, n_times)``. If output is 'avg_power_itc', the real
        values in ``out`` contain the average power and the imaginary values
        contain the ITC: ``out = avg_power + i * itc``.

    See Also
    --------
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell

    Notes
    -----
    .. versionadded:: 0.14.0
    """
    from .tfr import _compute_tfr
    return _compute_tfr(epoch_data, freqs, sfreq=sfreq,
                        method='multitaper', n_cycles=n_cycles,
                        zero_mean=zero_mean, time_bandwidth=time_bandwidth,
                        use_fft=use_fft, decim=decim, output=output,
                        n_jobs=n_jobs, verbose=verbose)
