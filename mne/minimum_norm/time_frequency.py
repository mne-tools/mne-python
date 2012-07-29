# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg, signal, fftpack

from ..fiff.constants import FIFF
from ..time_frequency.tfr import cwt, morlet
from ..baseline import rescale
from .inverse import combine_xyz, prepare_inverse_operator, _assemble_kernel, \
                     _make_stc, _pick_channels_inverse_operator, _check_method
from ..parallel import parallel_func


def source_band_induced_power(epochs, inverse_operator, bands, label=None,
                              lambda2=1.0 / 9.0, method="dSPM", n_cycles=5,
                              df=1, use_fft=False, decim=1, baseline=None,
                              baseline_mode='logratio', pca=True,
                              n_jobs=1, dSPM=None):
    """Compute source space induced power in given frequency bands

    Parameters
    ----------
    epochs: instance of Epochs
        The epochs
    inverse_operator: instance of inverse operator
        The inverse operator
    bands: dict
        Example : bands = dict(alpha=[8, 9])
    label: Label
        Restricts the source estimates to a given label
    lambda2: float
        The regularization parameter of the minimum norm
    method: "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    df: float
        delta frequency within bands
    decim: int
        Temporal decimation factor
    use_fft: bool
        Do convolutions in time or frequency domain with FFT
    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode: None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    pca: bool
        If True, the true dimension of data is estimated before running
        the time frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64)
    n_jobs: int
        Number of jobs to run in parallel
    """
    method = _check_method(method, dSPM)

    frequencies = np.concatenate([np.arange(band[0], band[1] + df / 2.0, df)
                                 for _, band in bands.iteritems()])

    powers, _, vertno = _source_induced_power(epochs,
                                      inverse_operator, frequencies,
                                      label=label,
                                      lambda2=lambda2, method=method,
                                      n_cycles=n_cycles, decim=decim,
                                      use_fft=use_fft, pca=pca, n_jobs=n_jobs,
                                      with_plv=False)

    Fs = epochs.info['sfreq']  # sampling in Hz
    stcs = dict()

    for name, band in bands.iteritems():
        idx = [k for k, f in enumerate(frequencies) if band[0] <= f <= band[1]]

        # average power in band + mean over epochs
        power = np.mean(powers[:, idx, :], axis=1)

        # Run baseline correction
        power = rescale(power, epochs.times[::decim], baseline, baseline_mode,
                        verbose=True, copy=False)

        tstep = float(decim) / Fs
        stc = _make_stc(power, epochs.times[0], tstep, vertno)
        stcs[name] = stc

        print '[done]'

    return stcs


def _compute_pow_plv(data, K, sel, Ws, source_ori, use_fft, Vh, with_plv,
                     pick_normal, decim):
    """Aux function for source_induced_power"""
    n_times = data[:, :, ::decim].shape[2]
    n_freqs = len(Ws)
    n_sources = K.shape[0]
    is_free_ori = False
    if (source_ori == FIFF.FIFFV_MNE_FREE_ORI and not pick_normal):
        is_free_ori = True
        n_sources /= 3

    shape = (n_sources, n_freqs, n_times)
    power = np.zeros(shape, dtype=np.float)  # power
    if with_plv:
        shape = (n_sources, n_freqs, n_times)
        plv = np.zeros(shape, dtype=np.complex)  # phase lock
    else:
        plv = None

    for e in data:
        e = e[sel]  # keep only selected channels

        if Vh is not None:
            e = np.dot(Vh, e)  # reducing data rank

        for f, w in enumerate(Ws):
            tfr = cwt(e, [w], use_fft=use_fft)[:, :, ::decim]
            tfr = np.asfortranarray(tfr.reshape(len(e), -1))

            # phase lock and power at freq f
            if with_plv:
                plv_f = np.zeros((n_sources, n_times), dtype=np.complex)
            pow_f = np.zeros((n_sources, n_times), dtype=np.float)

            for k, t in enumerate([np.real(tfr), np.imag(tfr)]):
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
                    # print 'combining the current components...',
                    sol = combine_xyz(sol, square=True)
                else:
                    np.power(sol, 2, sol)
                pow_f += sol
                del sol

            power[:, f, :] += pow_f
            del pow_f

            if with_plv:
                plv_f /= np.abs(plv_f)
                plv[:, f, :] += plv_f
                del plv_f

    return power, plv


def _source_induced_power(epochs, inverse_operator, frequencies, label=None,
                          lambda2=1.0 / 9.0, method="dSPM", n_cycles=5,
                          decim=1, use_fft=False, pca=True, pick_normal=True,
                          n_jobs=1, with_plv=True, zero_mean=False):
    """Aux function for source_induced_power
    """
    parallel, my_compute_pow_plv, n_jobs = parallel_func(_compute_pow_plv,
                                                         n_jobs)
    #
    #   Set up the inverse according to the parameters
    #
    epochs_data = epochs.get_data()

    nave = len(epochs_data)  # XXX : can do better when no preload

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(epochs.ch_names, inv)
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_normal)

    if pca:
        U, s, Vh = linalg.svd(K, full_matrices=False)
        rank = np.sum(s > 1e-8 * s[0])
        K = s[:rank] * U[:, :rank]
        Vh = Vh[:rank]
        print 'Reducing data rank to %d' % rank
    else:
        Vh = None

    Fs = epochs.info['sfreq']  # sampling in Hz

    print 'Computing source power ...'

    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    n_jobs = min(n_jobs, len(epochs_data))
    out = parallel(my_compute_pow_plv(data, K, sel, Ws,
                                      inv['source_ori'], use_fft, Vh,
                                      with_plv, pick_normal, decim)
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


def source_induced_power(epochs, inverse_operator, frequencies, label=None,
                         lambda2=1.0 / 9.0, method="dSPM", n_cycles=5, decim=1,
                         use_fft=False, pick_normal=False, baseline=None,
                         baseline_mode='logratio', pca=True, n_jobs=1,
                         dSPM=None, zero_mean=False):
    """Compute induced power and phase lock

    Computation can optionaly be restricted in a label.

    Parameters
    ----------
    epochs: instance of Epochs
        The epochs
    inverse_operator: instance of inverse operator
        The inverse operator
    label: Label
        Restricts the source estimates to a given label
    frequencies: array
        Array of frequencies of interest
    lambda2: float
        The regularization parameter of the minimum norm
    method: "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim: int
        Temporal decimation factor
    use_fft: bool
        Do convolutions in time or frequency domain with FFT
    pick_normal: bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode: None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    pca: bool
        If True, the true dimension of data is estimated before running
        the time frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64)
    n_jobs: int
        Number of jobs to run in parallel
    zero_mean: bool
        Make sure the wavelets are zero mean.
    """
    method = _check_method(method, dSPM)

    power, plv, vertno = _source_induced_power(epochs,
                            inverse_operator, frequencies,
                            label, lambda2, method, n_cycles, decim,
                            use_fft, pick_normal=pick_normal, pca=pca,
                            n_jobs=n_jobs)

    # Run baseline correction
    if baseline is not None:
        power = rescale(power, epochs.times[::decim], baseline, baseline_mode,
                        verbose=True, copy=False)

    return power, plv


def compute_source_psd(raw, inverse_operator, lambda2=1. / 9., method="dSPM",
                       tmin=None, tmax=None, fmin=0., fmax=200.,
                       NFFT=2048, overlap=0.5, pick_normal=False, label=None,
                       nave=1, pca=True):
    """Compute source power spectrum density (PSD)

    Parameters
    ----------
    raw : instance of Raw
        The raw data
    inverse_operator : dict
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
    NFFT: int
        Window size for the FFT. Should be a power of 2.
    overlap: float
        The overlap fraction between windows. Should be between 0 and 1.
        0 means no overlap.
    pick_normal : bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    label: Label
        Restricts the source estimates to a given label
    nave : int
        The number of averages used to scale the noise covariance matrix.
    pca: bool
        If True, the true dimension of data is estimated before running
        the time frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64)

    Returns
    -------
    stc : SourceEstimate
        The PSD (in dB) of each of the sources.
    """

    print 'Considering frequencies %g ... %g Hz' % (fmin, fmax)

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    is_free_ori = inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(raw.ch_names, inv)
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_normal)

    if pca:
        U, s, Vh = linalg.svd(K, full_matrices=False)
        rank = np.sum(s > 1e-8 * s[0])
        K = s[:rank] * U[:, :rank]
        Vh = Vh[:rank]
        print 'Reducing data rank to %d' % rank
    else:
        Vh = None

    start, stop = 0, raw.last_samp + 1 - raw.first_samp
    if tmin is not None:
        start = raw.time_to_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_to_index(tmax)[0] + 1
    NFFT = int(NFFT)
    Fs = raw.info['sfreq']
    window = signal.hanning(NFFT)
    freqs = fftpack.fftfreq(NFFT, 1. / Fs)
    freqs_mask = (freqs >= 0) & (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freqs_mask]
    fstep = np.mean(np.diff(freqs))
    psd = np.zeros((noise_norm.size, np.sum(freqs_mask)))
    n_windows = 0

    for this_start in np.arange(start, stop, int(NFFT * (1. - overlap))):
        data, _ = raw[sel, this_start:this_start + NFFT]
        if data.shape[1] < NFFT:
            print "Skipping last buffer"
            break

        if Vh is not None:
            data = np.dot(Vh, data)  # reducing data rank

        data *= window[None, :]

        data_fft = fftpack.fft(data)[:, freqs_mask]
        sol = np.dot(K, data_fft)

        if is_free_ori and not pick_normal:
            sol = combine_xyz(sol, square=True)
        else:
            sol = np.abs(sol) ** 2

        if method != "MNE":
            sol *= noise_norm ** 2

        psd += sol
        n_windows += 1

    psd /= n_windows

    psd = 10 * np.log10(psd)

    stc = _make_stc(psd, fmin * 1e-3, fstep * 1e-3, vertno)
    return stc
