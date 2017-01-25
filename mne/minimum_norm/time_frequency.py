# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg, fftpack

from ..io.constants import FIFF
from ..source_estimate import _make_stc
from ..time_frequency.tfr import cwt, morlet
from ..time_frequency.multitaper import (dpss_windows, _psd_from_mt,
                                         _psd_from_mt_adaptive, _mt_spectra)
from ..baseline import rescale, _log_rescale
from .inverse import (combine_xyz, prepare_inverse_operator, _assemble_kernel,
                      _pick_channels_inverse_operator, _check_method,
                      _check_ori, _subject_from_inverse)
from ..parallel import parallel_func
from ..utils import logger, verbose, warn
from ..externals import six


def _prepare_source_params(inst, inverse_operator, label=None,
                           lambda2=1.0 / 9.0, method="dSPM", nave=1,
                           decim=1, pca=True, pick_ori="normal",
                           prepared=False, verbose=None):
    """Prepare inverse operator and params for spectral / TFR analysis."""
    if not prepared:
        inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    else:
        inv = inverse_operator
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(inst.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_ori)

    if pca:
        U, s, Vh = linalg.svd(K, full_matrices=False)
        rank = np.sum(s > 1e-8 * s[0])
        K = s[:rank] * U[:, :rank]
        Vh = Vh[:rank]
        logger.info('Reducing data rank to %d' % rank)
    else:
        Vh = None
    is_free_ori = inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    return K, sel, Vh, vertno, is_free_ori, noise_norm


@verbose
def source_band_induced_power(epochs, inverse_operator, bands, label=None,
                              lambda2=1.0 / 9.0, method="dSPM", nave=1,
                              n_cycles=5, df=1, use_fft=False, decim=1,
                              baseline=None, baseline_mode='logratio',
                              pca=True, n_jobs=1, prepared=False,
                              verbose=None):
    """Compute source space induced power in given frequency bands.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    inverse_operator : instance of inverse operator
        The inverse operator.
    bands : dict
        Example : bands = dict(alpha=[8, 9]).
    label : Label
        Restricts the source estimates to a given label.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    df : float
        delta frequency within bands.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    decim : int
        Temporal decimation factor.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used.
    baseline_mode : None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    n_jobs : int
        Number of jobs to run in parallel.
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stcs : dict with a SourceEstimate (or VolSourceEstimate) for each band
        The estimated source space induced power estimates.
    """
    method = _check_method(method)

    frequencies = np.concatenate([np.arange(band[0], band[1] + df / 2.0, df)
                                 for _, band in six.iteritems(bands)])

    powers, _, vertno = _source_induced_power(
        epochs, inverse_operator, frequencies, label=label, lambda2=lambda2,
        method=method, nave=nave, n_cycles=n_cycles, decim=decim,
        use_fft=use_fft, pca=pca, n_jobs=n_jobs, with_plv=False,
        prepared=prepared)

    Fs = epochs.info['sfreq']  # sampling in Hz
    stcs = dict()

    subject = _subject_from_inverse(inverse_operator)
    _log_rescale(baseline, baseline_mode)  # for early failure
    for name, band in six.iteritems(bands):
        idx = [k for k, f in enumerate(frequencies) if band[0] <= f <= band[1]]

        # average power in band + mean over epochs
        power = np.mean(powers[:, idx, :], axis=1)

        # Run baseline correction
        power = rescale(power, epochs.times[::decim], baseline, baseline_mode,
                        copy=False, verbose=False)

        tmin = epochs.times[0]
        tstep = float(decim) / Fs
        stc = _make_stc(power, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)
        stcs[name] = stc

        logger.info('[done]')

    return stcs


def _prepare_tfr(data, decim, pick_ori, Ws, K, source_ori):
    """Prepare TFR source localization."""
    n_times = data[:, :, ::decim].shape[2]
    n_freqs = len(Ws)
    n_sources = K.shape[0]
    is_free_ori = False
    if (source_ori == FIFF.FIFFV_MNE_FREE_ORI and pick_ori is None):
        is_free_ori = True
        n_sources //= 3

    shape = (n_sources, n_freqs, n_times)
    return shape, is_free_ori


@verbose
def _compute_pow_plv(data, K, sel, Ws, source_ori, use_fft, Vh,
                     with_power, with_plv, pick_ori, decim, verbose=None):
    """Aux function for induced power and PLV."""
    shape, is_free_ori = _prepare_tfr(data, decim, pick_ori, Ws, K, source_ori)
    n_sources, n_times = shape[:2]
    power = np.zeros(shape, dtype=np.float)  # power or raw TFR
    # phase lock
    plv = np.zeros(shape, dtype=np.complex) if with_plv else None

    for epoch in data:
        epoch = epoch[sel]  # keep only selected channels

        if Vh is not None:
            epoch = np.dot(Vh, epoch)  # reducing data rank

        power_e, plv_e = _single_epoch_tfr(
            data=epoch, is_free_ori=is_free_ori, K=K, Ws=Ws, use_fft=use_fft,
            decim=decim, shape=shape, with_plv=with_plv, with_power=with_power)

        power += power_e
        if with_plv:
            plv += plv_e

    return power, plv


def _single_epoch_tfr(data, is_free_ori, K, Ws, use_fft, decim, shape,
                      with_plv, with_power):
    """Compute single trial TFRs, either ITC, power or raw TFR."""
    tfr_e = np.zeros(shape, dtype=np.float)  # power or raw TFR
    # phase lock
    plv_e = np.zeros(shape, dtype=np.complex) if with_plv else None
    n_sources, _, n_times = shape
    for f, w in enumerate(Ws):
        tfr_ = cwt(data, [w], use_fft=use_fft, decim=decim)
        tfr_ = np.asfortranarray(tfr_.reshape(len(data), -1))

        # phase lock and power at freq f
        if with_plv:
            plv_f = np.zeros((n_sources, n_times), dtype=np.complex)

        tfr_f = np.zeros((n_sources, n_times), dtype=np.float)

        for k, t in enumerate([np.real(tfr_), np.imag(tfr_)]):
            sol = np.dot(K, t)

            sol_pick_normal = sol
            if is_free_ori:
                sol_pick_normal = sol[2::3]

            if with_plv:
                if k == 0:  # real
                    plv_f += sol_pick_normal
                else:  # imag
                    plv_f += 1j * sol_pick_normal

            if is_free_ori:
                logger.debug('combining the current components...')
                sol = combine_xyz(sol, square=with_power)
            elif with_power:
                sol *= sol
            tfr_f += sol
            del sol

        tfr_e[:, f, :] += tfr_f
        del tfr_f

        if with_plv:
            plv_f /= np.abs(plv_f)
            plv_e[:, f, :] += plv_f
            del plv_f

    return tfr_e, plv_e


@verbose
def _source_induced_power(epochs, inverse_operator, frequencies, label=None,
                          lambda2=1.0 / 9.0, method="dSPM", nave=1, n_cycles=5,
                          decim=1, use_fft=False, pca=True, pick_ori="normal",
                          n_jobs=1, with_plv=True, zero_mean=False,
                          prepared=False, verbose=None):
    """Aux function for source induced power."""
    epochs_data = epochs.get_data()
    K, sel, Vh, vertno, is_free_ori, noise_norm = _prepare_source_params(
        inst=epochs, inverse_operator=inverse_operator, label=label,
        lambda2=lambda2, method=method, nave=nave, pca=pca, pick_ori=pick_ori,
        prepared=prepared, verbose=verbose)

    inv = inverse_operator
    parallel, my_compute_source_tfrs, n_jobs = parallel_func(
        _compute_pow_plv, n_jobs)
    Fs = epochs.info['sfreq']  # sampling in Hz

    logger.info('Computing source power ...')

    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    n_jobs = min(n_jobs, len(epochs_data))
    out = parallel(my_compute_source_tfrs(data=data, K=K, sel=sel, Ws=Ws,
                                          source_ori=inv['source_ori'],
                                          use_fft=use_fft, Vh=Vh,
                                          with_plv=with_plv, with_power=True,
                                          pick_ori=pick_ori, decim=decim)
                   for data in np.array_split(epochs_data, n_jobs))
    power = sum(o[0] for o in out)
    power /= len(epochs_data)  # average power over epochs

    if with_plv:
        plv = sum(o[1] for o in out)
        plv = np.abs(plv)
        plv /= len(epochs_data)  # average power over epochs
    else:
        plv = None

    if method != "MNE":
        power *= noise_norm.ravel()[:, None, None] ** 2

    return power, plv, vertno


@verbose
def source_induced_power(epochs, inverse_operator, frequencies, label=None,
                         lambda2=1.0 / 9.0, method="dSPM", nave=1, n_cycles=5,
                         decim=1, use_fft=False, pick_ori=None,
                         baseline=None, baseline_mode='logratio', pca=True,
                         n_jobs=1, zero_mean=False, prepared=False,
                         verbose=None):
    """Compute induced power and phase lock.

    Computation can optionaly be restricted in a label.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    frequencies : array
        Array of frequencies of interest.
    label : Label
        Restricts the source estimates to a given label.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim : int
        Temporal decimation factor.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    n_jobs : int
        Number of jobs to run in parallel.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori)

    power, plv, vertno = _source_induced_power(epochs,
                                               inverse_operator, frequencies,
                                               label=label, lambda2=lambda2,
                                               method=method, nave=nave,
                                               n_cycles=n_cycles, decim=decim,
                                               use_fft=use_fft,
                                               pick_ori=pick_ori,
                                               pca=pca, n_jobs=n_jobs,
                                               prepared=False)

    # Run baseline correction
    power = rescale(power, epochs.times[::decim], baseline, baseline_mode,
                    copy=False)
    return power, plv


@verbose
def compute_source_psd(raw, inverse_operator, lambda2=1. / 9., method="dSPM",
                       tmin=None, tmax=None, fmin=0., fmax=200.,
                       n_fft=2048, overlap=0.5, pick_ori=None, label=None,
                       nave=1, pca=True, prepared=False, verbose=None):
    """Compute source power spectrum density (PSD).

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    inverse_operator : instance of InverseOperator
        The inverse operator
    lambda2: float
        The regularization parameter
    method: "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA
    tmin : float | None
        The beginning of the time interval of interest (in seconds). If None
        start from the beginning of the file.
    tmax : float | None
        The end of the time interval of interest (in seconds). If None
        stop at the end of the file.
    fmin : float
        The lower frequency of interest
    fmax : float
        The upper frequency of interest
    n_fft: int
        Window size for the FFT. Should be a power of 2.
    overlap: float
        The overlap fraction between windows. Should be between 0 and 1.
        0 means no overlap.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    label: Label
        Restricts the source estimates to a given label
    nave : int
        The number of averages used to scale the noise covariance matrix.
    pca: bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The PSD (in dB) of each of the sources.
    """
    from scipy.signal import hanning
    pick_ori = _check_ori(pick_ori)

    logger.info('Considering frequencies %g ... %g Hz' % (fmin, fmax))

    K, sel, Vh, vertno, is_free_ori, noise_norm = _prepare_source_params(
        inst=raw, inverse_operator=inverse_operator, label=label,
        lambda2=lambda2, method=method, nave=nave, pca=pca, pick_ori=pick_ori,
        prepared=prepared, verbose=verbose)

    start, stop = 0, raw.last_samp + 1 - raw.first_samp
    if tmin is not None:
        start = raw.time_as_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_as_index(tmax)[0] + 1
    n_fft = int(n_fft)
    Fs = raw.info['sfreq']
    window = hanning(n_fft)
    freqs = fftpack.fftfreq(n_fft, 1. / Fs)
    freqs_mask = (freqs >= 0) & (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freqs_mask]
    fstep = np.mean(np.diff(freqs))
    if is_free_ori and pick_ori is None:
        psd = np.zeros((K.shape[0] // 3, np.sum(freqs_mask)))
    else:
        psd = np.zeros((K.shape[0], np.sum(freqs_mask)))
    n_windows = 0

    for this_start in np.arange(start, stop, int(n_fft * (1. - overlap))):
        data, _ = raw[sel, this_start:this_start + n_fft]
        if data.shape[1] < n_fft:
            logger.info("Skipping last buffer")
            break

        if Vh is not None:
            data = np.dot(Vh, data)  # reducing data rank

        data *= window[None, :]

        data_fft = fftpack.fft(data)[:, freqs_mask]
        sol = np.dot(K, data_fft)

        if is_free_ori and pick_ori is None:
            sol = combine_xyz(sol, square=True)
        else:
            sol = (sol * sol.conj()).real

        if method != "MNE":
            sol *= noise_norm ** 2

        psd += sol
        n_windows += 1

    psd /= n_windows

    psd = 10 * np.log10(psd)

    subject = _subject_from_inverse(inverse_operator)
    stc = _make_stc(psd, vertices=vertno, tmin=fmin * 1e-3,
                    tstep=fstep * 1e-3, subject=subject)
    return stc


@verbose
def _compute_source_psd_epochs(epochs, inverse_operator, lambda2=1. / 9.,
                               method="dSPM", fmin=0., fmax=200.,
                               pick_ori=None, label=None, nave=1,
                               pca=True, inv_split=None, bandwidth=4.,
                               adaptive=False, low_bias=True, n_jobs=1,
                               prepared=False, verbose=None):
    """Generator for compute_source_psd_epochs."""
    logger.info('Considering frequencies %g ... %g Hz' % (fmin, fmax))

    K, sel, Vh, vertno, is_free_ori, noise_norm = _prepare_source_params(
        inst=epochs, inverse_operator=inverse_operator, label=label,
        lambda2=lambda2, method=method, nave=nave, pca=pca, pick_ori=pick_ori,
        prepared=prepared, verbose=verbose)

    # split the inverse operator
    if inv_split is not None:
        K_split = np.array_split(K, inv_split)
    else:
        K_split = [K]

    # compute DPSS windows
    n_times = len(epochs.times)
    sfreq = epochs.info['sfreq']

    # compute standardized half-bandwidth
    half_nbw = float(bandwidth) * n_times / (2 * sfreq)
    if half_nbw < 0.5:
        warn('Bandwidth too small, using minimum (normalized 0.5)')
        half_nbw = 0.5
    n_tapers_max = int(2 * half_nbw)

    dpss, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                 low_bias=low_bias)
    n_tapers = len(dpss)

    logger.info('Using %d tapers with bandwidth %0.1fHz'
                % (n_tapers, bandwidth))

    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators '
             'due to a low number of tapers.')
        adaptive = False

    if adaptive:
        parallel, my_psd_from_mt_adaptive, n_jobs = \
            parallel_func(_psd_from_mt_adaptive, n_jobs)
    else:
        weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]

    subject = _subject_from_inverse(inverse_operator)
    for k, e in enumerate(epochs):
        logger.info("Processing epoch : %d" % (k + 1))
        data = e[sel]

        if Vh is not None:
            data = np.dot(Vh, data)  # reducing data rank

        # compute tapered spectra in sensor space
        x_mt, freqs = _mt_spectra(data, dpss, sfreq)

        if k == 0:
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            fstep = np.mean(np.diff(freqs))

        # allocate space for output
        psd = np.empty((K.shape[0], np.sum(freq_mask)))

        # Optionally, we split the inverse operator into parts to save memory.
        # Without splitting the tapered spectra in source space have size
        # (n_vertices x n_tapers x n_times / 2)
        pos = 0
        for K_part in K_split:
            # allocate space for tapered spectra in source space
            x_mt_src = np.empty((K_part.shape[0], x_mt.shape[1],
                                x_mt.shape[2]), dtype=x_mt.dtype)

            # apply inverse to each taper
            for i in range(n_tapers):
                x_mt_src[:, i, :] = np.dot(K_part, x_mt[:, i, :])

            # compute the psd
            if adaptive:
                out = parallel(my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                               for x in np.array_split(x_mt_src,
                                                       min(n_jobs,
                                                           len(x_mt_src))))
                this_psd = np.concatenate(out)
            else:
                x_mt_src = x_mt_src[:, :, freq_mask]
                this_psd = _psd_from_mt(x_mt_src, weights)

            psd[pos:pos + K_part.shape[0], :] = this_psd
            pos += K_part.shape[0]

        # combine orientations
        if is_free_ori and pick_ori is None:
            psd = combine_xyz(psd, square=False)

        if method != "MNE":
            psd *= noise_norm ** 2

        stc = _make_stc(psd, tmin=fmin, tstep=fstep, vertices=vertno,
                        subject=subject)

        # we return a generator object for "stream processing"
        yield stc


@verbose
def compute_source_psd_epochs(epochs, inverse_operator, lambda2=1. / 9.,
                              method="dSPM", fmin=0., fmax=200.,
                              pick_ori=None, label=None, nave=1,
                              pca=True, inv_split=None, bandwidth=4.,
                              adaptive=False, low_bias=True,
                              return_generator=False, n_jobs=1,
                              prepared=False, verbose=None):
    """Compute source power spectrum density (PSD) from Epochs.

    This uses the multi-taper method to compute the PSD.

    Parameters
    ----------
    epochs : instance of Epochs
        The raw data.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    label : Label
        Restricts the source estimates to a given label.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    pca : bool
        If True, the true dimension of data is estimated before running
        the time-frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    inv_split : int or None
        Split inverse operator into inv_split parts in order to save memory.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stcs : list (or generator object) of SourceEstimate | VolSourceEstimate
        The source space PSDs for each epoch.
    """
    # use an auxiliary function so we can either return a generator or a list
    stcs_gen = _compute_source_psd_epochs(epochs, inverse_operator,
                                          lambda2=lambda2, method=method,
                                          fmin=fmin, fmax=fmax,
                                          pick_ori=pick_ori, label=label,
                                          nave=nave, pca=pca,
                                          inv_split=inv_split,
                                          bandwidth=bandwidth,
                                          adaptive=adaptive,
                                          low_bias=low_bias, n_jobs=n_jobs,
                                          prepared=prepared)

    if return_generator:
        # return generator object
        return stcs_gen
    else:
        # return a list
        stcs = list()
        for stc in stcs_gen:
            stcs.append(stc)

        return stcs
