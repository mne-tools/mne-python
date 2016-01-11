"""Functions to visualize phase/amplitude relationships in signals"""

import numpy as np
from matplotlib import pyplot as plt


def plot_phase_locked_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, tmin=-.5, tmax=.5):
    """Make a phase-locked amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    tmin : float
        The time to include before each phase peak
    tmax : float
        The time to include after each phase peak

    Returns
    -------
    axs : array of matplotlib axes
        The axes used for plotting.
    """
    from ..connectivity import phase_locked_amplitude
    data_amp, data_phase, times = phase_locked_amplitude(
        epochs, freqs_phase, freqs_amp,
        ix_ph, ix_amp, tmin=tmin, tmax=tmax)

    # Plotting
    f, axs = plt.subplots(2, 1)
    ax = axs[0]
    ax.pcolormesh(times, freqs_amp, data_amp)

    ax = axs[1]
    ax.plot(times, data_phase)

    plt.setp(axs, xlim=(times[0], times[-1]))
    return axs


def plot_phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, normalize=True,
                                n_bins=20):
    """Make a circular phase-binned amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    normalize : bool
        Whether amplitudes are normalized before averaging together. Helps
        if some frequencies have a larger mean amplitude than others.
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size (2 * np.pi) / n_bins.

    Returns
    -------
    ax : matplotlib axis
        The axis used for plotting.
    """
    from ..connectivity import phase_binned_amplitude
    from sklearn.preprocessing import MinMaxScaler
    amps, bins = phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                        ix_ph, ix_amp, n_bins=n_bins)
    if normalize is True:
        amps = MinMaxScaler().fit_transform(amps)
    plt.figure()
    ax = plt.subplot(111, polar=True)
    bins_plt = bins[:-1]  # Because there is 1 more bins than amps
    width = 2 * np.pi / len(bins_plt)
    ax.bar(bins_plt + np.pi, amps, color='r', width=width)
    return ax
