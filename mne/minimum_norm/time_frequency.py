# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..time_frequency.tfr import cwt, morlet
from ..baseline import rescale
from .inverse import combine_xyz, prepare_inverse_operator, _assemble_kernel, \
                     _make_stc, _pick_channels_inverse_operator
from ..parallel import parallel_func


def source_band_induced_power(epochs, inverse_operator, bands, label=None,
                              lambda2=1.0 / 9.0, dSPM=True, n_cycles=5, df=1,
                              use_fft=False, decim=1, baseline=None,
                              baseline_mode='logratio', pca=True,
                              n_jobs=1):
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
    dSPM: bool
        Do dSPM or not?
    n_cycles: int
        Number of cycles
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
    frequencies = np.concatenate([np.arange(band[0], band[1] + df / 2.0, df)
                                 for _, band in bands.iteritems()])

    powers, _, vertno = _source_induced_power(epochs,
                                      inverse_operator, frequencies,
                                      label=label,
                                      lambda2=lambda2, dSPM=dSPM,
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
                          lambda2=1.0 / 9.0, dSPM=True, n_cycles=5, decim=1,
                          use_fft=False, pca=True, pick_normal=True,
                          n_jobs=1, with_plv=True):
    """Aux function for source_induced_power
    """
    parallel, my_compute_pow_plv, n_jobs = parallel_func(_compute_pow_plv,
                                                         n_jobs)
    #
    #   Set up the inverse according to the parameters
    #
    epochs_data = epochs.get_data()

    nave = len(epochs_data)  # XXX : can do better when no preload

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
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
    K, noise_norm, vertno = _assemble_kernel(inv, label, dSPM, pick_normal)

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

    Ws = morlet(Fs, frequencies, n_cycles=n_cycles)

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

    if dSPM:
        power *= noise_norm.ravel()[:, None, None] ** 2

    return power, plv, vertno


def source_induced_power(epochs, inverse_operator, frequencies, label=None,
                         lambda2=1.0 / 9.0, dSPM=True, n_cycles=5, decim=1,
                         use_fft=False, pick_normal=False, baseline=None,
                         baseline_mode='logratio', pca=True, n_jobs=1):
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
    dSPM: bool
        Do dSPM or not?
    n_cycles: int
        Number of cycles
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
    """
    power, plv, vertno = _source_induced_power(epochs,
                            inverse_operator, frequencies,
                            label, lambda2, dSPM, n_cycles, decim,
                            use_fft, pick_normal=pick_normal, pca=pca,
                            n_jobs=n_jobs)

    # Run baseline correction
    if baseline is not None:
        power = rescale(power, epochs.times[::decim], baseline, baseline_mode,
                        verbose=True, copy=False)

    return power, plv
