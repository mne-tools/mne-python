# Authors: Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from numpy.random import shuffle
from scipy import stats
from scipy.signal import hilbert, gaussian

from ..stats import fdr_correction
from ..time_frequency import morlet
from ..time_frequency.tfr import cwt
from ..preprocessing.peak_finder import peak_finder
from ..filter import band_pass_filter
from ..parallel import parallel_func
from ..utils import logger, check_random_state


def _abs_cwt(x, W):
    return np.abs(cwt(x, [W], use_fft=False).ravel()).astype(np.float32)


def make_surrogate_data(data):
    """
    Returns a copy of the shuffled surrogate data.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_times)
        Data to be shuffled.

    Returns
    -------
    shuffled_data : ndarray, shape (n_epochs, n_times)
        The shuffled data, with each epoch shuffled across n_times.
    """
    shuffled_data = data.copy()
    map(shuffle, shuffled_data)
    return shuffled_data


def generate_pac_signal(sfreq, duration, n_epochs, f_phase, f_amplitude,
                        sigma=5, max_amp=1, random_state=None,
                        mean=0, std=0.2):
    """
    Generate a phase amplitude coupled signal based on the given parameters.

    A phase amplitude coupled signal of given duration and n_epochs with
    varying amounts of phase amplitude coupling based on a gaussian window
    is returned.

    Parameters
    ----------
    sfreq : float
        Sampling frequency.
    duration : float
        Time in seconds.
    n_epochs : int
        Number of epochs to be generated.
    f_phase : float
        Phase modulating frequency in Hz (eg. 8.).
    f_amplitude : float
        Amplitude modulated frequency eg. gamma
    sigma : float
        Standard deviation of gaussian window used. The gaussian window
        determines various levels of modulation in the trials.
        Default value is 5.
    max_amp : float
        Constant that determines maximum amplitude of both signals.
        Default value is 1.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the noise estimation.
        Default value is None.
    mean : float
        Mean of the Gaussian distribution (np.random.normal)
        used to generate white noise. Default value is 0.
    std : float
        Standard deviation of the Gaussian distribution
        (np.random.normal) used to generate white noise.
        Default value is 0.2.

    Returns
    -------
    pac_signal : ndarray, shape (n_epochs, n_times)
        Phase amplitude coupled signal.
    """

    times = np.linspace(0, duration, sfreq * duration)
    n_times = len(times)

    if n_epochs <= 1:
        raise RuntimeError(
            'Number of trials too low, please provide atleast two.')

    rng = check_random_state(random_state)
    # generate white noise
    white_noise = rng.normal(mean, std, n_times * n_epochs)
    white_noise = np.reshape(white_noise, (n_epochs, n_times))

    if sigma == 0:
        raise ValueError(
            'Sigma value cannot be 0. Please choose a higher value.')

    # make a gaussian window
    win = gaussian(n_epochs, sigma, sym=False)

    if np.max(win) == np.min(win):
        raise RuntimeError(
            'win values lead to divide by 0.')

    # normalize the gaussian window
    win = (win - np.min(win)) / (np.max(win) - np.min(win))

    # Construct the signal with many levels of modulation
    pac_signal = np.zeros((n_epochs, n_times))

    for sig in range(n_epochs):
        # Generate amplitude envelope of modulated signal
        amp_fa = (max_amp * ((1. - win[sig]) *
                             np.sin(2 * np.pi * f_phase * times)
                             + 1. + win[sig]) / 2.)
        # Generate the phase amplitude coupled signal
        pac_signal[sig] = (amp_fa * np.sin(2. * np.pi * f_amplitude * times) +
                           max_amp * np.sin(2. * np.pi * f_phase * times) +
                           white_noise[sig])

    if len(pac_signal) != n_epochs:
        raise RuntimeError('Data not equal to epochs.')

    # mix the trials
    order = np.argsort(rng.randn(len(pac_signal)))
    pac_signal = pac_signal[order, :]

    return pac_signal


def cross_frequency_coupling(data, sfreq, phase_freq, n_cycles,
                             l_amp_freq, h_amp_freq, n_freqs, alpha=0.001,
                             n_surrogates=10 ** 4, random_state=None,
                             n_jobs=1):
    """
    Compute the cross frequency coupling.

    Parameters
    ----------
    data : array, shape (n_times,)
        Signal time series.
    sfreq : float
        Sampling frequency.
    phase_freq : float
        The phase modulating frequency in Hz. (eg. 10.)
    n_cycles : float | array of float
        Number of cycles used to compute morlet wavelets.
    l_amp_freq : float
        Lower edge of amplitude modulated frequency band in Hz. (eg. 60.)
    h_amp_freq : float
        Upper edge of amplitude modulated frequency band in Hz. (eg. 100.)
    n_freqs : int
        Number of frequency points to compute. (eg. 100)
    alpha : float
        Error rate allowed. Default 0.001.
    n_surrogates : int
        Number of surrogates. Default 10 ** 4.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the noise estimation.
        Default value is None.
    n_jobs : int
        Number of jobs to run in parallel. Default 1.

    Returns
    -------
    times : array, shape (n_times,)
        Time points for signal plotting.
    freqs : array, shape (n_freqs,)
        Frequency points across range at which amplitudes are computed.
    traces : ndarray, shape (n_epochs, n_times)
        Normalized amplitude traces.
    ztraces : ndarray, shape (n_epochs, n_times)
        Statistically significant amplitude traces.
    z_threshold : float
        Threshold of statistically significant amplitude traces.
    erp : array, shape (n_times,)
        Evoked related potential.

    References:
    High gamma power is phase-locked to theta oscillations in human neocortex.
    Canolty RT1, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT.
    (Science. 2006)
    """

    if data.ndim != 1:
        raise ValueError('Dimensions of data incorrect. Please use 1d array.')

    Ws = morlet(sfreq, [phase_freq], n_cycles=n_cycles,
                sigma=None, zero_mean=True)
    data = data.reshape(1, data.size)
    x_low = cwt(data, Ws, use_fft=False).ravel()

    phases = np.angle(x_low)
    trigger_inds, _ = peak_finder(phases)

    freqs = np.logspace(np.log10(l_amp_freq), np.log10(h_amp_freq), n_freqs)

    n_samples = data.size

    # define epoch for averaging:
    n_times = int(np.round(sfreq))  # +/- 1 second around trigger

    # makes sure all triggers allow full epoch, get rid of those that don't:
    trigger_inds = trigger_inds[trigger_inds > n_times]
    trigger_inds = trigger_inds[trigger_inds < n_samples - n_times]

    # Compute filtered data
    Ws = morlet(sfreq, freqs, n_cycles=n_cycles, sigma=None,
                zero_mean=True)

    parallel, my_abs_cwt, _ = parallel_func(_abs_cwt, n_jobs)

    if n_jobs == 1:
        ampmat = np.empty((n_freqs, n_samples), dtype=np.float)
        for e, W in enumerate(Ws):
            ampmat[e] = np.array(_abs_cwt(data, W))
    else:
        ampmat = np.array(parallel(my_abs_cwt(data, W) for W in Ws))

    # zscore for normalization
    ampmat -= np.mean(ampmat, axis=1)[:, None]
    ampmat /= np.std(ampmat, axis=1)[:, None]

    # phase-triggered ERP of raw signal:
    # and phase-triggered time-frequency amplitude values (normalized):
    erp = np.zeros(2 * n_times)
    traces = np.zeros((n_freqs, 2 * n_times), dtype=np.float32)
    for ind in trigger_inds:
        erp += data[0, ind - n_times: ind + n_times]
        traces += ampmat[:, ind - n_times: ind + n_times]
    erp /= len(trigger_inds)
    traces /= len(trigger_inds)

    # permutation resampling for statistical significance:
    # only need to compute mean and variance of surrogate distributions at one
    # time point for all frequencies, since the surrogate trigger events could
    # have occurred at any point
    rng = check_random_state(random_state)
    shifts = np.floor(rng.rand(2 * n_surrogates) * n_samples)
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
    p_values = stats.norm.pdf(ztraces)
    accept, _ = fdr_correction(p_values, alpha=alpha)
    z_threshold = np.abs(ztraces[accept]).min()

    # time points used to plot the ERP signal
    times = np.arange(-n_times, n_times, dtype=np.float) / sfreq

    return times, freqs, traces, ztraces, z_threshold, erp


def phase_amplitude_coupling(data, sfreq, l_phase_freq, h_phase_freq,
                             l_amp_freq, h_amp_freq, bin_num=18, method='iir',
                             surrogates=False, n_jobs=1):
    """
    Compute modulation index for the given data.

    Parameters
    ----------
    data : ndarray (n_epochs, n_times)
        Signal time series.
    sfreq : float
        Sampling frequency in Hz.
    l_phase_freq : float
        Lower edge of phase modulating frequency.
    h_phase_freq : float
        Upper edge of phase modulating frequency.
    l_amp_freq : float
        Lower edge of amplitude modulated frequency.
    h_amp_freq : float
        Upper edge of amplitude modulated frequency.
    bin_num : int
        Number of phase bins. Default is 18. (20 degrees * 18 = 360)
    method : str
        Filter method used. Default is 'iir'.
    n_jobs : int
        Number of jobs used for parallelization. Default 1.
    surrogates : bool
        Return surrogates values for data specific to PAC,
        the phase series and amplitude envelope are shuffled for every trial.
        Default value is False.

    Returns
    -------
    normalized_amplitude : ndarray, shape (n_epochs, bin_num)
        The normalized amplitude values across each bin.
    phase_bins : array, shape (number of bins + 1,)
        Binned phase points. Used for plotting.

    References:
    Measuring phase-amplitude coupling between neuronal oscillations
    of different frequencies.
    Tort ABL, Komorowski R, Eichenbaum H, Kopell N., (J Neurophysiol. 2010)
    """

    # obtained number of trials
    n_trials = len(data)

    # Filter data into phase modulating freq and amplitude modulated freq.
    x_fp = band_pass_filter(data, sfreq, l_phase_freq, h_phase_freq,
                            method=method, n_jobs=n_jobs)
    x_fa = band_pass_filter(data, sfreq, l_amp_freq, h_amp_freq,
                            method=method, n_jobs=n_jobs)

    if len(x_fp) != len(x_fa) or x_fp.ndim != x_fa.ndim:
        raise RuntimeError(
            'Phase and amplitude filtered data does not have same dimensions')

    # Calculate phase series of phase modulating signal
    phase_series_fp = np.angle(hilbert(x_fp)) + np.pi

    # Calculate amplitude envelope of amplitude modulated signal
    amp_envelope_fa = np.abs(hilbert(x_fa))

    if surrogates:
        phase_series_fp = make_surrogate_data(phase_series_fp)
        amp_envelope_fa = make_surrogate_data(amp_envelope_fa)

    # Bin the phases
    bin_size = 2. * np.pi / bin_num  # 360 degrees divided by number of bins
    phase_bins = np.arange(phase_series_fp.min(), phase_series_fp.max() +
                           bin_size, bin_size)
    if len(phase_bins) - 1 != bin_num:
        raise RuntimeError(
            'Phase bins are incorrect, please check the number of bins used.')

    # Initialize the arrays
    digitized = np.zeros(phase_series_fp.shape)
    amplitude_bin_means = normalized_amplitude = np.zeros((n_trials, bin_num))

    # Calculate the amplitude distribution for every trial
    for trial in range(n_trials):
        digitized[trial] = np.digitize(phase_series_fp[trial], phase_bins,
                                       right=False)

        # Calculate mean amplitude at each phase bin
        amplitude_bin_means[trial] = [amp_envelope_fa[trial][digitized[trial] == i].mean()
                                      for i in range(1, len(phase_bins))]
        if np.isnan(np.sum(amplitude_bin_means[trial])):
            raise ValueError('Encountered nan when calculating mean\
                               amplitude for bins.')

        # Calculate normalized mean amplitude.
        normalized_amplitude[trial] = (amplitude_bin_means[trial] /
                                       np.sum(amplitude_bin_means[trial]))
        assert np.round(normalized_amplitude[trial].sum()) == 1, ('Normalized\
                        amplitudes are incorrect')

    return normalized_amplitude, phase_bins


def modulation_index(amplitude_distribution):
    """
    Calculates the normalized modulation index from the amplitude distribution.

    Parameters
    ----------
    amplitude_distribution : ndarray, shape (n_epochs, bin_num)
        Amplitude distribution across phase bins.

    Returns
    -------
    mi : array, shape (n_epochs,)
        Modulation index values.
    """
    n_trial = len(amplitude_distribution)
    mi = np.zeros(n_trial)
    for trial in range(n_trial):
        # Calculate the modulation index for every trial
        # (modulation index calculated from Kullback Liebler
        #  distance and Shannon entropy)
        if len(amplitude_distribution[trial]) == 0:
            raise ValueError('Length of amplitude distribution is 0.')

        if np.sum(amplitude_distribution[trial] > 1):
            raise ValueError('Amplitude distribution should be normalized.')

        mi[trial] = 1. - (stats.entropy(amplitude_distribution[trial]) /
                          np.log(len(amplitude_distribution[trial])))
        if mi[trial] > 1 or mi[trial] < 0:
            raise ValueError(
                'MI is normalized and should lie between 0 and 1.')

    return mi
