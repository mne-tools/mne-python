#!/usr/bin/env python

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy.signal import hilbert

import mne

from ..time_frequency import morlet
from ..time_frequency.tfr import cwt
from ..preprocessing.peak_finder import peak_finder
from ..filter import band_pass_filter
from ..parallel import parallel_func
from ..utils import logger


def abs_cwt(x, W):
    return np.abs(cwt(x, [W], use_fft=False).ravel()).astype(np.float32)


def cross_frequency_coupling(x, sfreq, phase_freq, n_samples,
                             n_cycles, fa_high, fa_low,
                             f_n, n_jobs=1, alpha=0.001, n_surrogates=10 ** 4):
    ''' Compute the cross frequency coupling.

    References:
    High gamma power is phase-locked to theta oscillations in human neocortex.
    Canolty RT1, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT.
    (Science. 2006)
    '''

    Ws = morlet(sfreq, [phase_freq], n_cycles=n_cycles,
                sigma=None, zero_mean=True)
    x_low = cwt(x, Ws, use_fft=False).ravel()

    phases = np.angle(x_low)
    trigger_inds, _ = peak_finder(phases)

    # freqs = np.logspace(1., 250., 70)
    freqs = np.logspace(np.log10(fa_low), np.log10(fa_high), f_n)
    n_freqs = len(freqs)

    # define epoch for averaging:
    n_times = np.round(sfreq)  # +/- 1 second around trigger

    # makes sure all triggers allow full epoch, get rid of those that don't:
    trigger_inds = trigger_inds[trigger_inds > n_times]
    trigger_inds = trigger_inds[trigger_inds < n_samples - n_times]

    # Compute filtered data
    Ws = morlet(sfreq, freqs, n_cycles=n_cycles, sigma=None,
                zero_mean=True)

    parallel, my_abs_cwt, _ = parallel_func(abs_cwt, n_jobs)
    ampmat = np.empty((n_freqs, n_samples), dtype=np.float)

    if n_jobs == 1:
        for e, W in enumerate(Ws):
            ampmat[e] = np.array(abs_cwt(x, W))
    else:
        ampmat = np.array(parallel(my_abs_cwt(x, W) for W in Ws))

    # zscore for normalization
    ampmat -= np.mean(ampmat, axis=1)[:, None]
    ampmat /= np.std(ampmat, axis=1)[:, None]

    # phase-triggered ERP of raw signal:
    # and phase-triggered time-frequency amplitude values (normalized):
    erp = np.zeros(2 * n_times)
    traces = np.zeros((n_freqs, 2 * n_times), dtype=np.float32)
    for ind in trigger_inds:
        erp += x[0, ind - n_times: ind + n_times]
        traces += ampmat[:, ind - n_times: ind + n_times]
    erp /= len(trigger_inds)
    traces /= len(trigger_inds)

    # permutation resampling for statistical significance:
    # only need to compute mean and variance of surrogate distributions at one
    # time point for all frequencies, since the surrogate trigger events could
    # have occurred at any point
    shifts = np.floor(np.random.rand(2 * n_surrogates) * n_samples)
    # get rid of trigger shifts within one second of actual occurances:
    shifts = shifts[shifts > sfreq]
    shifts = shifts[shifts < n_samples - sfreq]

    sur_distributions = np.zeros((n_freqs, n_surrogates), dtype=np.float32)
    for sur in range(n_surrogates):
        if (n_surrogates - sur + 1 % 100) == 0:
            logger.info(n_surrogates - sur + 1)
        # circular shift
        sur_triggers = np.mod(trigger_inds + shifts[sur], n_samples)
        sur_triggers = sur_triggers.astype(np.int)
        sur_distributions[:, sur] = np.mean(ampmat[:, sur_triggers], axis=1)

    sur_fits_mean = np.mean(sur_distributions, axis=1)
    sur_fits_std = np.std(sur_distributions, axis=1)

    ztraces = (traces - sur_fits_mean[:, None]) / sur_fits_std[:, None]

    # Compute FDR threshold
    from scipy import stats
    p_values = stats.norm.pdf(ztraces)
    accept, _ = mne.stats.fdr_correction(p_values, alpha=alpha)
    z_threshold = np.abs(ztraces[accept]).min()

    times = np.arange(-n_times, n_times, dtype=np.float) / sfreq

    return times, freqs, traces, ztraces, z_threshold, erp


def phase_amplitude_coupling(data, fs, fp_low, fp_high,
                             fa_low, fa_high, bin_size_angle=18, n_jobs=1):
    ''' Compute modulation index for the given data.


    References:
    Measuring phase-amplitude coupling between neuronal oscillations
    of different frequencies.
    Tort ABL, Komorowski R, Eichenbaum H, Kopell N., (J Neurophysiol. 2010)
    '''

    # Filter data into phase modulating freq and amplitude modulated freq.
    # Fp = 5-10 Hz and Fa = 30-100Hz
    x_fp = band_pass_filter(data, fs, fp_low, fp_high,
                            method='fft', n_jobs=n_jobs)
    x_fa = band_pass_filter(data, fs, fa_low, fa_high,
                            method='fft', n_jobs=n_jobs)

    # Calculate phase series of phase modulating signal
    phase_series_fp = np.unwrap(abs(np.angle(hilbert(x_fp))))

    # Calculate amplitude envelope of amplitude modulated signal
    amplitude_envelope_fa = np.abs(hilbert(x_fa))

    # Bin the phases
    # bin size is 18 degrees. (18 * 10 = 180)
    bin_size = np.deg2rad(bin_size_angle)

    phase_bins = np.arange(phase_series_fp.min(), phase_series_fp.max(),
                           bin_size)

    N = len(phase_bins)  # Number of phase bins
    assert N == 10, '10 bins are expected for a bin_size of 18 degrees.'

    digitized = np.digitize(phase_series_fp, phase_bins)

    # Calculate mean amplitude at each phase bin
    amplitude_bin_means = [amplitude_envelope_fa[digitized == i].mean()
                           for i in range(1, len(phase_bins))]

    # Calculate normalized mean amplitude.
    normalized_amplitude = amplitude_bin_means / np.sum(amplitude_bin_means)
    assert (np.round(normalized_amplitude.sum()) == 1,
            'Normalized amplitudes are incorrect')

    # Calculate KL distance using Shannon entropy.
    h = 0
    for i in range(0, N - 1):
        h += -1.0 * (normalized_amplitude[i] * np.log(normalized_amplitude[i]))

    # Kullback-Liebler distance
    kl_distance = np.log(N) - h

    # Calculate Modulation Index
    mi = (kl_distance / np.log(N))
    assert 0 <= mi <= 1, 'MI is normalised and should lie between 0 and 1.'

    logger.info('The modulation index found is %s' % (mi))
    return mi, phase_bins, normalized_amplitude
