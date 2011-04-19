# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

from ..fiff.constants import FIFF
from ..stc import SourceEstimate
from ..time_frequency.tfr import cwt, morlet
from .inverse import combine_xyz, prepare_inverse_operator


def source_induced_power(epochs, inverse_operator, bands, lambda2=1.0 / 9.0,
                         dSPM=True, n_cycles=5, df=1, use_fft=False,
                         baseline=None, baseline_mode='ratio'):
    """XXX for source_induced_power

    Parameters
    ----------
    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'ratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))


    """

    #
    #   Set up the inverse according to the parameters
    #
    epochs_data = epochs.get_data()
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
    n_times = len(epochs.times)

    for name, band in bands.iteritems():
        print 'Computing power in band %s [%s, %s] Hz...' % (name, band[0],
                                                             band[1])

        freqs = np.arange(band[0], band[1] + df / 2.0, df)  # frequencies
        n_freqs = len(freqs)
        Ws = morlet(Fs, freqs, n_cycles=n_cycles)

        power = 0

        for e in epochs_data:
            e = e[sel]  # keep only selected channels
            tfr = cwt(e, Ws, use_fft=use_fft)
            tfr = tfr.reshape(len(e), -1)

            sol = np.dot(K, tfr)

            if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
                # print 'combining the current components...',
                sol = combine_xyz(sol)

            if dSPM:
                # print '(dSPM)...',
                sol *= inv['noisenorm'][:, None]

            # average power in band
            sol = np.mean(np.reshape(sol ** 2, (-1, n_freqs, n_times)), axis=1)
            power += sol
            del sol

        power /= len(epochs_data)

        # Run baseline correction
        if baseline is not None:
            print "Applying baseline correction ..."
            times = epochs.times
            bmin = baseline[0]
            bmax = baseline[1]
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
