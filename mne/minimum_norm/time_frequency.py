# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..fiff.constants import FIFF
from ..stc import SourceEstimate
from ..time_frequency.tfr import cwt, morlet
from .inverse import combine_xyz, prepare_inverse_operator


def _compute_power(data, K, sel, Ws, source_ori, use_fft, Vh):
    """Aux function for source_induced_power"""
    power = 0

    for e in data:
        e = e[sel]  # keep only selected channels

        if Vh is not None:
            e = np.dot(Vh, e)  # reducing data rank

        for w in Ws:
            tfr = cwt(e, [w], use_fft=use_fft)
            tfr = np.asfortranarray(tfr.reshape(len(e), -1))

            for t in [np.real(tfr), np.imag(tfr)]:
                sol = np.dot(K, t)

                if source_ori == FIFF.FIFFV_MNE_FREE_ORI:
                    # print 'combining the current components...',
                    sol = combine_xyz(sol, square=True)
                else:
                    np.power(sol, 2, sol)

                power += sol
                del sol

    return power


def source_induced_power(epochs, inverse_operator, bands, lambda2=1.0 / 9.0,
                         dSPM=True, n_cycles=5, df=1, use_fft=False,
                         baseline=None, baseline_mode='logratio', pca=True,
                         subtract_evoked=False, n_jobs=1):
    """Compute source space induced power

    Parameters
    ----------
    epochs: instance of Epochs
        The epochs
    inverse_operator: instance of inverse operator
        The inverse operator
    bands: dict
        Example : bands = dict(alpha=[8, 9])
    lambda2: float
        The regularization parameter of the minimum norm
    dSPM: bool
        Do dSPM or not?
    n_cycles: int
        Number of cycles
    df: float
        delta frequency within bands
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
    subtract_evoked: bool
        If True, the evoked component (average of all epochs) if subtracted
        from each epochs.
    n_jobs: int
        Number of jobs to run in parallel
    """

    if n_jobs == -1:
        try:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        except ImportError:
            print "multiprocessing not installed. Cannot run in parallel."
            n_jobs = 1

    try:
        from scikits.learn.externals.joblib import Parallel, delayed
        parallel = Parallel(n_jobs)
        my_compute_power = delayed(_compute_power)
    except ImportError:
        print "joblib not installed. Cannot run in parallel."
        n_jobs = 1
        my_compute_power = _compute_power
        parallel = list

    #
    #   Set up the inverse according to the parameters
    #
    epochs_data = epochs.get_data()

    if subtract_evoked:  # subtract with a copy not to touch epochs
        epochs_data = epochs_data - np.mean(epochs_data, axis=0)

    nave = len(epochs_data)  # XXX : can do better when no preload

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, dSPM)
    #
    #   Pick the correct channels from the data
    #
    sel = [epochs.ch_names.index(name) for name in inv['noise_cov']['names']]
    print 'Picked %d channels from the data' % len(sel)
    print 'Computing inverse...',
    #
    #   Simple matrix multiplication followed by combination of the
    #   three current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    K = inv['reginv'][:, None] * reduce(np.dot,
                                           [inv['eigen_fields']['data'],
                                           inv['whitener'],
                                           inv['proj']])

    if pca:
        U, s, Vh = linalg.svd(K)
        rank = np.sum(s > 1e-8*s[0])
        K = np.dot(K, s[:rank] * U[:, :rank])
        Vh = Vh[:rank]
        print 'Reducing data rank to %d' % rank
    else:
        Vh = None

    #
    #   Transformation into current distributions by weighting the
    #   eigenleads with the weights computed above
    #
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        # print '(eigenleads already weighted)...',
        K = np.dot(inv['eigen_leads']['data'], K)
    else:
        #
        #     R^0.5 has to factored in
        #
        # print '(eigenleads need to be weighted)...',
        K = np.sqrt(inv['source_cov']['data'])[:, None] * \
                             np.dot(inv['eigen_leads']['data'], K)

    Fs = epochs.info['sfreq']  # sampling in Hz

    stcs = dict()
    src = inv['src']

    for name, band in bands.iteritems():
        print 'Computing power in band %s [%s, %s] Hz...' % (name, band[0],
                                                             band[1])

        freqs = np.arange(band[0], band[1] + df / 2.0, df)  # frequencies
        Ws = morlet(Fs, freqs, n_cycles=n_cycles)

        power = sum(parallel(my_compute_power(data, K, sel, Ws,
                                            inv['source_ori'], use_fft, Vh)
                            for data in np.array_split(epochs_data, n_jobs)))

        if dSPM:
            # print '(dSPM)...',
            power *= inv['noisenorm'][:, None] ** 2

        # average power in band + mean over epochs
        power /= len(epochs_data) * len(freqs)

        # Run baseline correction
        if baseline is not None:
            print "Applying baseline correction ..."
            times = epochs.times
            bmin, bmax = baseline
            if bmin is None:
                imin = 0
            else:
                imin = int(np.where(times >= bmin)[0][0])
            if bmax is None:
                imax = len(times)
            else:
                imax = int(np.where(times <= bmax)[0][-1]) + 1
            mean_baseline_power = np.mean(power[:, imin:imax], axis=1)
            if baseline_mode is 'logratio':
                power /= mean_baseline_power[:, None]
                power = np.log(power)
            elif baseline_mode is 'zscore':
                power -= mean_baseline_power[:, None]
                power /= np.std(power[:, imin:imax], axis=1)[:, None]
        else:
            print "No baseline correction applied..."

        stc = SourceEstimate(None)
        stc.data = power
        stc.tmin = epochs.times[0]
        stc.tstep = 1.0 / Fs
        stc.lh_vertno = src[0]['vertno']
        stc.rh_vertno = src[1]['vertno']
        stc._init_times()

        stcs[name] = stc

        print '[done]'

    return stcs
