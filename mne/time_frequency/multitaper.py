# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Parts of this code were copied from NiTime http://nipy.sourceforge.net/nitime

import numpy as np

from ..parallel import parallel_func
from ..utils import (
    _check_option,
    _verbose_control,
    logger,
    verbose_static,
    warn,
)


def dpss_windows(N, half_nbw, Kmax, *, sym=True, norm=None, low_bias=True):
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
    sym : bool
        Whether to generate a symmetric window (``True``, for filter design) or
        a periodic window (``False``, for spectral analysis). Default is
        ``True``.

        .. versionadded:: 1.3
    norm : 2 | ``'approximate'`` | ``'subsample'`` | None
        Window normalization method. If ``'approximate'`` or ``'subsample'``,
        windows are normalized by the maximum, and a correction scale-factor
        for even-length windows is applied either using
        ``N**2/(N**2+half_nbw)`` ("approximate") or a FFT-based subsample shift
        ("subsample"). ``2`` uses the L2 norm. ``None`` (the default) uses
        ``"approximate"`` when ``Kmax=None`` and ``2`` otherwise.

        .. versionadded:: 1.3
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.

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
    # TODO VERSION can be removed with SciPy 1.16 is min,
    # workaround for https://github.com/scipy/scipy/pull/22344
    from scipy.signal.windows import dpss as sp_dpss

    if N <= 1:
        dpss, eigvals = np.ones((1, 1)), np.ones(1)
    else:
        dpss, eigvals = sp_dpss(
            N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True
        )
    if low_bias:
        idx = eigvals > 0.9
        if not idx.any():
            warn("Could not properly use low_bias, keeping lowest-bias taper")
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def _psd_from_mt_adaptive(x_mt, eigvals, freq_mask, max_iter=250, return_weights=False):
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
        Maximum number of iterations for weight computation.
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
    from scipy.integrate import trapezoid

    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError("Need one eigenvalue for each taper")

    if n_tapers < 3:
        raise ValueError("Not enough tapers to compute adaptive weights.")

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = trapezoid(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
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
            d_k = psd_iter / (
                eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var
            )
            d_k *= rt_eig[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged
            err -= d_k
            if np.max(np.mean(err**2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        if n == max_iter - 1:
            warn("Iterative multi-taper PSD computation did not converge.")

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
    x_mt : array, shape=(..., n_tapers, n_freqs)
        Tapered spectra
    weights : array, shape=(n_tapers,)
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array, shape=(..., n_freqs)
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
    x_mt : array, shape=(..., n_tapers, n_freqs)
        Tapered spectra for x
    y_mt : array, shape=(..., n_tapers, n_freqs)
        Tapered spectra for y
    weights_x : array, shape=(n_tapers,)
        Weights used to combine the tapered spectra of x_mt
    weights_y : array, shape=(n_tapers,)
        Weights used to combine the tapered spectra of y_mt

    Returns
    -------
    csd: array
        The computed CSD
    """
    csd = np.sum(weights_x * x_mt * (weights_y * y_mt).conj(), axis=-2)
    denom = np.sqrt((weights_x * weights_x.conj()).real.sum(axis=-2)) * np.sqrt(
        (weights_y * weights_y.conj()).real.sum(axis=-2)
    )
    csd *= 2 / denom
    return csd


def _mt_spectra(x, dpss, sfreq, n_fft=None, remove_dc=True):
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
    %(remove_dc)s

    Returns
    -------
    x_mt : array, shape=(..., n_tapers, n_freqs)
        The tapered spectra
    freqs : array, shape=(n_freqs,)
        The frequency points in Hz of the spectra
    """
    from scipy.fft import rfft, rfftfreq

    if n_fft is None:
        n_fft = x.shape[-1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    if remove_dc:
        x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = rfftfreq(n_fft, 1.0 / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)), dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.0)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.0)
    return x_mt, freqs


@_verbose_control
def _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive, verbose=None):
    """Triage windowing and multitaper parameters."""
    # Compute standardized half-bandwidth
    from scipy.signal import get_window

    if isinstance(bandwidth, str):
        logger.info(f'    Using standard spectrum estimation with "{bandwidth}" window')
        window_fun = get_window(bandwidth, n_times)[np.newaxis]
        return window_fun, np.ones(1), False

    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / (2.0 * sfreq)
    else:
        half_nbw = 4.0
    if half_nbw < 0.5:
        raise ValueError(
            f"bandwidth value {bandwidth} yields a normalized half-bandwidth of "
            f"{half_nbw} < 0.5, use a value of at least {sfreq / n_times}"
        )

    # Compute DPSS windows
    n_tapers_max = int(2 * half_nbw)
    window_fun, eigvals = dpss_windows(
        n_times, half_nbw, n_tapers_max, sym=False, low_bias=low_bias
    )
    logger.info(
        f"    Using multitaper spectrum estimation with {len(eigvals)} DPSS windows"
    )

    if adaptive and len(eigvals) < 3:
        warn(
            "Not adaptively combining the spectral estimators due to a "
            f"low number of tapers ({len(eigvals)} < 3)."
        )
        adaptive = False

    return window_fun, eigvals, adaptive


@verbose_static(
    "fmin_fmax_psd", "normalization", "remove_dc", "n_jobs", "max_iter_multitaper"
)
def psd_array_multitaper(
    x,
    sfreq,
    fmin=0.0,
    fmax=np.inf,
    bandwidth=None,
    adaptive=False,
    low_bias=True,
    normalization="length",
    remove_dc=True,
    output="power",
    n_jobs=None,
    *,
    max_iter=150,
    verbose=None,
):
    r"""Compute power spectral density (PSD) using a multi-taper method.

    The power spectral density is computed with DPSS
    tapers :footcite:p:`Slepian1978`.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin, fmax : float
        The lower- and upper-bound on frequencies of interest. Default is
        ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
    bandwidth : float
        Frequency bandwidth of the multi-taper window function in Hz. For a
        given frequency, frequencies at ``± bandwidth / 2`` are smoothed
        together. The default value is a bandwidth of
        ``8 * (sfreq / n_times)``.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    normalization : 'full' | 'length'
        Normalization strategy. If "full", the PSD will be normalized by the
        sampling rate as well as the length of the signal (as in
        :ref:`Nitime <nitime:users-guide>`). Default is ``'length'``.
    remove_dc : bool
        If ``True``, the mean is subtracted from each segment before computing
        its spectrum.
    output : str
        The format of the returned ``psds`` array, ``'complex'`` or
        ``'power'``:

        * ``'power'`` : the power spectral density is returned.
        * ``'complex'`` : the complex fourier coefficients are returned per
          taper.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
    max_iter : int
        Maximum number of iterations to reach convergence when combining the
        tapered spectra with adaptive weights (see argument ``adaptive``). This
        argument has not effect if ``adaptive`` is set to ``False``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

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
    csd_multitaper
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd

    Notes
    -----
    .. versionadded:: 0.14.0

    References
    ----------
    .. footbibliography::
    """
    from scipy.fft import rfftfreq

    _check_option("normalization", normalization, ["length", "full"])

    # Reshape data so its 2-D for parallelization
    ndim_in = x.ndim
    x = np.atleast_2d(x)
    n_times = x.shape[-1]
    dshape = x.shape[:-1]
    x = x.reshape(-1, n_times)

    dpss, eigvals, adaptive = _compute_mt_params(
        n_times, sfreq, bandwidth, low_bias, adaptive
    )
    n_tapers = len(dpss)
    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]

    # decide which frequencies to keep
    freqs = rfftfreq(n_times, 1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    if output == "complex":
        psd = np.zeros((x.shape[0], n_tapers, n_freqs), dtype="complex")
    else:
        psd = np.zeros((x.shape[0], n_freqs))

    # Let's go in up to 50 MB chunks of signals to save memory
    n_chunk = max(50000000 // (len(freq_mask) * len(eigvals) * 16), 1)
    offsets = np.concatenate((np.arange(0, x.shape[0], n_chunk), [x.shape[0]]))
    for start, stop in zip(offsets[:-1], offsets[1:]):
        x_mt = _mt_spectra(x[start:stop], dpss, sfreq, remove_dc=remove_dc)[0]
        if output == "power":
            if not adaptive:
                psd[start:stop] = _psd_from_mt(x_mt[:, :, freq_mask], weights)
            else:
                parallel, my_psd_from_mt_adaptive, n_jobs = parallel_func(
                    _psd_from_mt_adaptive, n_jobs
                )
                n_splits = min(stop - start, n_jobs)
                out = parallel(
                    my_psd_from_mt_adaptive(x, eigvals, freq_mask, max_iter)
                    for x in np.array_split(x_mt, n_splits)
                )
                psd[start:stop] = np.concatenate(out)
        else:
            psd[start:stop] = x_mt[:, :, freq_mask]

    if normalization == "full":
        psd /= sfreq

    # Combining/reshaping to original data shape
    last_dims = (n_freqs,) if output == "power" else (n_tapers, n_freqs)
    psd = psd.reshape(dshape + last_dims, copy=False)
    if ndim_in == 1:
        psd = psd[0]

    if output == "complex":
        return psd, freqs, weights
    else:
        return psd, freqs


@verbose_static(
    "freqs_tfr_array",
    "n_cycles_tfr",
    "time_bandwidth_tfr",
    "decim_tfr",
    "n_jobs",
    "temporal_window_tfr_intro",
    "temporal_window_tfr_multitaper_notes",
    "time_bandwidth_tfr_notes",
)
def tfr_array_multitaper(
    data,
    sfreq,
    freqs,
    n_cycles=7.0,
    zero_mean=True,
    time_bandwidth=4.0,
    use_fft=True,
    decim=1,
    output="complex",
    n_jobs=None,
    *,
    return_weights=False,
    verbose=None,
):
    r"""Compute Time-Frequency Representation (TFR) using DPSS tapers.

    Same computation as `~mne.time_frequency.tfr_multitaper`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` or
    `~mne.Evoked` objects.

    Parameters
    ----------
    data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float
        Sampling frequency of the data in Hz.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : int | array of int, shape (n_freqs,)
        Number of cycles in the wavelet, either a fixed number or one per
        frequency. The number of cycles ``n_cycles`` and the frequencies of
        interest ``freqs`` define the temporal window length. See notes for
        additional information about the relationship between those arguments
        and about time and frequency smoothing.
    zero_mean : bool
        If True, make sure the wavelets have a mean of zero. Defaults to True.
    time_bandwidth : float ``≥ 2.0``
        Product between the temporal window length (in seconds) and the *full*
        frequency bandwidth (in Hz). This product can be seen as the surface of the
        window on the time/frequency plane and controls the frequency bandwidth
        (thus the frequency resolution) and the number of good tapers. See notes
        for additional information.
    use_fft : bool
        Use the FFT for convolutions or not. Defaults to True.
    decim : int | slice
        Decimation factor, applied *after* time-frequency decomposition.

        - if :class:`int`, returns ``tfr[..., ::decim]`` (keep only every Nth
          sample along the time axis).
        - if :class:`slice`, returns ``tfr[..., decim]`` (keep only the specified
          slice along the time axis).

        .. note::
            Decimation is done after convolutions and may create aliasing
            artifacts.
    output : str

        * ``'complex'`` : single trial per taper complex values.
        * ``'power'`` : single trial power.
        * ``'phase'`` : single trial per taper phase.
        * ``'avg_power'`` : average of single trial power.
        * ``'itc'`` : inter-trial coherence.
        * ``'avg_power_itc'`` : average of single trial power and inter-trial
          coherence across trials.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
        The parallelization is implemented across channels.
    return_weights : bool
        If True, return the taper weights. Only applies if ``output='complex'`` or
        ``'phase'``.

        .. versionadded:: 1.10.0
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    out : array
        Time frequency transform of ``data``.

        - if ``output in ('complex',' 'phase')``, array of shape
          ``(n_epochs, n_chans, n_tapers, n_freqs, n_times)``
        - if ``output`` is ``'power'``, array of shape ``(n_epochs, n_chans,
          n_freqs, n_times)``
        - else, array of shape ``(n_chans, n_freqs, n_times)``

        If ``output`` is ``'avg_power_itc'``, the real values in ``out``
        contain the average power and the imaginary values contain the
        inter-trial coherence: :math:`out = power_{avg} + i * ITC`.
    weights : array of shape (n_tapers, n_freqs)
        The taper weights. Only returned if ``output='complex'`` or ``'phase'`` and
        ``return_weights=True``.

    See Also
    --------
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell

    Notes
    -----
    In spectrotemporal analysis (as with traditional fourier methods),
    the temporal and spectral resolution are interrelated: longer temporal windows
    allow more precise frequency estimates; shorter temporal windows "smear"
    frequency estimates while providing more precise timing information.

    Time-frequency representations are computed using a sliding temporal window.
    Either the temporal window has a fixed length independent of frequency, or the
    temporal window decreases in length with increased frequency.

    .. image:: https://www.fieldtriptoolbox.org/assets/img/tutorial/timefrequencyanalysis/figure1.png

    *Figure: Time and frequency smoothing. (a) For a fixed length temporal window
    the time and frequency smoothing remains fixed. (b) For temporal windows that
    decrease with frequency, the temporal smoothing decreases and the frequency
    smoothing increases with frequency.* Source: `FieldTrip tutorial: Time-frequency
    analysis using Hanning window, multitapers and wavelets
    <https://www.fieldtriptoolbox.org/tutorial/timefrequencyanalysis>`_.

    In MNE-Python, the multitaper temporal window length is defined by the arguments
    ``freqs`` and ``n_cycles``, respectively defining the frequencies of interest
    and the number of cycles: :math:`T = \frac{\mathtt{n\_cycles}}{\mathtt{freqs}}`

    A fixed number of cycles for all frequencies will yield a temporal window which
    decreases with frequency. For example, ``freqs=np.arange(1, 6, 2)`` and
    ``n_cycles=2`` yields ``T=array([2., 0.7, 0.4])``.

    To use a temporal window with fixed length, the number of cycles has to be
    defined based on the frequency. For example, ``freqs=np.arange(1, 6, 2)`` and
    ``n_cycles=freqs / 2`` yields ``T=array([0.5, 0.5, 0.5])``.

    In MNE-Python's multitaper functions, the frequency bandwidth is
    additionally affected by the parameter ``time_bandwidth``.
    The ``n_cycles`` parameter determines the temporal window length based on the
    frequencies of interest: :math:`T = \frac{\mathtt{n\_cycles}}{\mathtt{freqs}}`.
    The ``time_bandwidth`` parameter defines the "time-bandwidth product", which is
    the product of the temporal window length (in seconds) and the frequency
    bandwidth (in Hz). Thus once ``n_cycles`` has been set, frequency bandwidth is
    determined by :math:`\frac{\mathrm{time~bandwidth}}{\mathrm{time~window}}`, and
    thus passing a larger ``time_bandwidth`` value will increase the frequency
    bandwidth (thereby decreasing the frequency *resolution*).

    The increased frequency bandwidth is reached by averaging spectral estimates
    obtained from multiple tapers. Thus, ``time_bandwidth`` also determines the
    number of tapers used. MNE-Python uses only "good" tapers (tapers with minimal
    leakage from far-away frequencies); the number of good tapers is
    ``floor(time_bandwidth - 1)``. This means there is another trade-off at play,
    between frequency resolution and the variance reduction that multitaper
    analysis provides. Striving for finer frequency resolution (by setting
    ``time_bandwidth`` low) means fewer tapers will be used, which undermines what
    is unique about multitaper methods — namely their ability to improve accuracy /
    reduce noise in the power estimates by using several (orthogonal) tapers.

    .. warning::

        In `~mne.time_frequency.tfr_array_multitaper` and
        `~mne.time_frequency.tfr_multitaper`, ``time_bandwidth`` defines the
        product of the temporal window length with the *full* frequency bandwidth
        For example, a full bandwidth of 4 Hz at a frequency of interest of 10 Hz
        will "smear" the frequency estimate between 8 Hz and 12 Hz.

        This is not the case for `~mne.time_frequency.psd_array_multitaper` where
        the argument ``bandwidth`` defines the *half* frequency bandwidth. In the
        example above, the half-frequency bandwidth is 2 Hz.

    .. versionadded:: 0.14.0
    """  # noqa: E501
    from .tfr import _compute_tfr

    return _compute_tfr(
        data,
        freqs,
        sfreq=sfreq,
        method="multitaper",
        n_cycles=n_cycles,
        zero_mean=zero_mean,
        time_bandwidth=time_bandwidth,
        use_fft=use_fft,
        decim=decim,
        output=output,
        return_weights=return_weights,
        n_jobs=n_jobs,
        verbose=verbose,
    )
