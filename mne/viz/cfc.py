"""Functions to visualize phase/amplitude relationships in signals"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_phase_locked_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, tmin=-.5, tmax=.5):
    """Make a phase-locked amplitude plot."""
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
    """Make a circular phase-binned amplitude plot"""
    from ..connectivity import phase_binned_amplitude
    amps, bins = phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                        ix_ph, ix_amp, n_bins=n_bins)
    if normalize is True:
        amps = MinMaxScaler().fit_transform(amps)
    f = plt.figure()
    ax = plt.subplot(111, polar=True)
    bins_plt = bins[:-1]  # Because there is 1 more bins than amps
    width = 2 * np.pi / len(bins_plt)
    ax.bar(bins_plt + np.pi, amps, color='r', width=width)
    return ax
