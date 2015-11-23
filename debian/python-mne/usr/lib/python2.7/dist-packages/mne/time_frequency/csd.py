# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import warnings
import copy as cp

import numpy as np
from scipy.fftpack import fftfreq

from ..io.pick import pick_types
from ..utils import logger, verbose
from ..time_frequency.multitaper import (dpss_windows, _mt_spectra,
                                         _csd_from_mt, _psd_from_mt_adaptive)


class CrossSpectralDensity(object):
    """Cross-spectral density

    Parameters
    ----------
    data : array of shape (n_channels, n_channels)
        The cross-spectral density matrix.
    ch_names : list of string
        List of channels' names.
    projs :
        List of projectors used in CSD calculation.
    bads :
        List of bad channels.
    frequencies : float | list of float
        Frequency or frequencies for which the CSD matrix was calculated. If a
        list is passed, data is a sum across CSD matrices for all frequencies.
    n_fft : int
        Length of the FFT used when calculating the CSD matrix.
    """
    def __init__(self, data, ch_names, projs, bads, frequencies, n_fft):
        self.data = data
        self.dim = len(data)
        self.ch_names = cp.deepcopy(ch_names)
        self.projs = cp.deepcopy(projs)
        self.bads = cp.deepcopy(bads)
        self.frequencies = np.atleast_1d(np.copy(frequencies))
        self.n_fft = n_fft

    def __repr__(self):
        s = 'frequencies : %s' % self.frequencies
        s += ', size : %s x %s' % self.data.shape
        s += ', data : %s' % self.data
        return '<CrossSpectralDensity  |  %s>' % s


@verbose
def compute_epochs_csd(epochs, mode='multitaper', fmin=0, fmax=np.inf,
                       fsum=True, tmin=None, tmax=None, n_fft=None,
                       mt_bandwidth=None, mt_adaptive=False, mt_low_bias=True,
                       projs=None, verbose=None):
    """Estimate cross-spectral density from epochs

    Note: Baseline correction should be used when creating the Epochs.
          Otherwise the computed cross-spectral density will be inaccurate.

    Note: Results are scaled by sampling frequency for compatibility with
          Matlab.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    mode : str
        Spectrum estimation mode can be either: 'multitaper' or 'fourier'.
    fmin : float
        Minimum frequency of interest.
    fmax : float | np.inf
        Maximum frequency of interest.
    fsum : bool
        Sum CSD values for the frequencies of interest. Summing is performed
        instead of averaging so that accumulated power is comparable to power
        in the time domain. If True, a single CSD matrix will be returned. If
        False, the output will be a list of CSD matrices.
    tmin : float | None
        Minimum time instant to consider. If None start at first sample.
    tmax : float | None
        Maximum time instant to consider. If None end at last sample.
    n_fft : int | None
        Length of the FFT. If None the exact number of samples between tmin and
        tmax will be used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    projs : list of Projection | None
        List of projectors to use in CSD calculation, or None to indicate that
        the projectors from the epochs should be inherited.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.
    """
    # Portions of this code adapted from mne/connectivity/spectral.py

    # Check correctness of input data and parameters
    if fmax < fmin:
        raise ValueError('fmax must be larger than fmin')
    tstep = epochs.times[1] - epochs.times[0]
    if tmin is not None and tmin < epochs.times[0] - tstep:
        raise ValueError('tmin should be larger than the smallest data time '
                         'point')
    if tmax is not None and tmax > epochs.times[-1] + tstep:
        raise ValueError('tmax should be smaller than the largest data time '
                         'point')
    if tmax is not None and tmin is not None:
        if tmax < tmin:
            raise ValueError('tmax must be larger than tmin')
    if epochs.baseline is None:
        warnings.warn('Epochs are not baseline corrected, cross-spectral '
                      'density may be inaccurate')

    if projs is None:
        projs = cp.deepcopy(epochs.info['projs'])
    else:
        projs = cp.deepcopy(projs)

    picks_meeg = pick_types(epochs[0].info, meg=True, eeg=True, eog=False,
                            ref_meg=False, exclude='bads')
    ch_names = [epochs.ch_names[k] for k in picks_meeg]

    # Preparing time window slice
    tstart, tend = None, None
    if tmin is not None:
        tstart = np.where(epochs.times >= tmin)[0][0]
    if tmax is not None:
        tend = np.where(epochs.times <= tmax)[0][-1] + 1
    tslice = slice(tstart, tend, None)
    n_times = len(epochs.times[tslice])
    n_fft = n_times if n_fft is None else n_fft

    # Preparing frequencies of interest
    sfreq = epochs.info['sfreq']
    orig_frequencies = fftfreq(n_fft, 1. / sfreq)
    freq_mask = (orig_frequencies > fmin) & (orig_frequencies < fmax)
    frequencies = orig_frequencies[freq_mask]
    n_freqs = len(frequencies)

    if n_freqs == 0:
        raise ValueError('No discrete fourier transform results within '
                         'the given frequency window. Please widen either '
                         'the frequency window or the time window')

    # Preparing for computing CSD
    logger.info('Computing cross-spectral density from epochs...')
    if mode == 'multitaper':
        # Compute standardized half-bandwidth
        if mt_bandwidth is not None:
            half_nbw = float(mt_bandwidth) * n_times / (2 * sfreq)
        else:
            half_nbw = 2

        # Compute DPSS windows
        n_tapers_max = int(2 * half_nbw)
        window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                           low_bias=mt_low_bias)
        n_tapers = len(eigvals)
        logger.info('    using multitaper spectrum estimation with %d DPSS '
                    'windows' % n_tapers)

        if mt_adaptive and len(eigvals) < 3:
            warnings.warn('Not adaptively combining the spectral estimators '
                          'due to a low number of tapers.')
            mt_adaptive = False
    elif mode == 'fourier':
        logger.info('    using FFT with a Hanning window to estimate spectra')
        window_fun = np.hanning(n_times)
        mt_adaptive = False
        eigvals = 1.
        n_tapers = None
    else:
        raise ValueError('Mode has an invalid value.')

    csds_mean = np.zeros((len(ch_names), len(ch_names), n_freqs),
                         dtype=complex)

    # Picking frequencies of interest
    freq_mask_mt = freq_mask[orig_frequencies >= 0]

    # Compute CSD for each epoch
    n_epochs = 0
    for epoch in epochs:
        epoch = epoch[picks_meeg][:, tslice]

        # Calculating Fourier transform using multitaper module
        x_mt, _ = _mt_spectra(epoch, window_fun, sfreq, n_fft)

        if mt_adaptive:
            # Compute adaptive weights
            _, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                               return_weights=True)
            # Tiling weights so that we can easily use _csd_from_mt()
            weights = weights[:, np.newaxis, :, :]
            weights = np.tile(weights, [1, x_mt.shape[0], 1, 1])
        else:
            # Do not use adaptive weights
            if mode == 'multitaper':
                weights = np.sqrt(eigvals)[np.newaxis, np.newaxis, :,
                                           np.newaxis]
            else:
                # Hack so we can sum over axis=-2
                weights = np.array([1.])[:, None, None, None]

        x_mt = x_mt[:, :, freq_mask_mt]

        # Calculating CSD
        # Tiling x_mt so that we can easily use _csd_from_mt()
        x_mt = x_mt[:, np.newaxis, :, :]
        x_mt = np.tile(x_mt, [1, x_mt.shape[0], 1, 1])
        y_mt = np.transpose(x_mt, axes=[1, 0, 2, 3])
        weights_y = np.transpose(weights, axes=[1, 0, 2, 3])
        csds_epoch = _csd_from_mt(x_mt, y_mt, weights, weights_y)

        # Scaling by number of samples and compensating for loss of power due
        # to windowing (see section 11.5.2 in Bendat & Piersol).
        if mode == 'fourier':
            csds_epoch /= n_times
            csds_epoch *= 8 / 3.

        # Scaling by sampling frequency for compatibility with Matlab
        csds_epoch /= sfreq

        csds_mean += csds_epoch
        n_epochs += 1

    csds_mean /= n_epochs

    logger.info('[done]')

    # Summing over frequencies of interest or returning a list of separate CSD
    # matrices for each frequency
    if fsum is True:
        csd_mean_fsum = np.sum(csds_mean, 2)
        csd = CrossSpectralDensity(csd_mean_fsum, ch_names, projs,
                                   epochs.info['bads'],
                                   frequencies=frequencies, n_fft=n_fft)
        return csd
    else:
        csds = []
        for i in range(n_freqs):
            csds.append(CrossSpectralDensity(csds_mean[:, :, i], ch_names,
                                             projs, epochs.info['bads'],
                                             frequencies=frequencies[i],
                                             n_fft=n_fft))
        return csds
