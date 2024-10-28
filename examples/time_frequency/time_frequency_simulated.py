"""
.. _ex-tfr-comparison:

==================================================================================
Time-frequency on simulated data (Multitaper vs. Morlet vs. Stockwell vs. Hilbert)
==================================================================================

This example demonstrates the different time-frequency estimation methods
on simulated data. It shows the time-frequency resolution trade-off
and the problem of estimation variance. In addition it highlights
alternative functions for generating TFRs without averaging across
trials, or by operating on numpy arrays.
"""  # noqa E501
# Authors: Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import numpy as np
from matplotlib import pyplot as plt

from mne import Epochs, create_info
from mne.io import RawArray
from mne.time_frequency import AverageTFRArray, EpochsTFRArray, tfr_array_morlet

print(__doc__)

# %%
# Simulate data
# -------------
#
# We'll simulate data with a known spectro-temporal structure.

sfreq = 1000.0
ch_names = ["SIM0001", "SIM0002"]
ch_types = ["grad", "grad"]
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = 1024  # Just over 1 second epochs
n_epochs = 40
seed = 42
rng = np.random.RandomState(seed)
data = rng.randn(len(ch_names), n_times * n_epochs + 200)  # buffer

# Add a 50 Hz sinusoidal burst to the noise and ramp it.
t = np.arange(n_times, dtype=np.float64) / sfreq
signal = np.sin(np.pi * 2.0 * 50.0 * t)  # 50 Hz sinusoid signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.0  # hard windowing
on_time = np.logical_and(t >= 0.45, t <= 0.55)
signal[on_time] *= np.hanning(on_time.sum())  # ramping
data[:, 100:-100] += np.tile(signal, n_epochs)  # add signal

raw = RawArray(data, info)
events = np.zeros((n_epochs, 3), dtype=int)
events[:, 0] = np.arange(n_epochs) * n_times
epochs = Epochs(
    raw,
    events,
    dict(sin50hz=0),
    tmin=0,
    tmax=n_times / sfreq,
    reject=dict(grad=4000),
    baseline=None,
)

epochs.average().plot()

# %%
# Calculate a time-frequency representation (TFR)
# -----------------------------------------------
#
# Below we'll demonstrate the output of several TFR functions in MNE:
#
# * :func:`mne.time_frequency.tfr_multitaper`
# * :func:`mne.time_frequency.tfr_stockwell`
# * :func:`mne.time_frequency.tfr_morlet`
# * :meth:`mne.Epochs.filter` and :meth:`mne.Epochs.apply_hilbert`
#
# Multitaper transform
# ====================
# First we'll use the multitaper method for calculating the TFR.
# This creates several orthogonal tapering windows in the TFR estimation,
# which reduces variance. We'll also show some of the parameters that can be
# tweaked (e.g., ``time_bandwidth``) that will result in different multitaper
# properties, and thus a different TFR. You can trade time resolution or
# frequency resolution or both in order to get a reduction in variance.

freqs = np.arange(5.0, 100.0, 3.0)
vmin, vmax = -3.0, 3.0  # Define our color limits.

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
for n_cycles, time_bandwidth, ax, title in zip(
    [freqs / 2, freqs, freqs / 2],  # number of cycles
    [2.0, 4.0, 8.0],  # time bandwidth
    axs,
    [
        "Sim: Least smoothing, most variance",
        "Sim: Less frequency smoothing,\nmore time smoothing",
        "Sim: Less time smoothing,\nmore frequency smoothing",
    ],
):
    power = epochs.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=True,
    )
    ax.set_title(title)
    # Plot results. Baseline correct based on first 100 ms.
    power.plot(
        [0],
        baseline=(0.0, 0.1),
        mode="mean",
        vlim=(vmin, vmax),
        axes=ax,
        show=False,
        colorbar=False,
    )

##############################################################################
# Stockwell (S) transform
# =======================
#
# Stockwell uses a Gaussian window to balance temporal and spectral resolution.
# Importantly, frequency bands are phase-normalized, hence strictly comparable
# with regard to timing, and, the input signal can be recoverd from the
# transform in a lossless way if we disregard numerical errors. In this case,
# we control the spectral / temporal resolution by specifying different widths
# of the gaussian window using the ``width`` parameter.

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
fmin, fmax = freqs[[0, -1]]
for width, ax in zip((0.2, 0.7, 3.0), axs):
    power = epochs.compute_tfr(method="stockwell", freqs=(fmin, fmax), width=width)
    power.plot(
        [0], baseline=(0.0, 0.1), mode="mean", axes=ax, show=False, colorbar=False
    )
    ax.set_title(f"Sim: Using S transform, width = {width:0.1f}")

# %%
# Morlet Wavelets
# ===============
#
# Next, we'll show the TFR using morlet wavelets, which are a sinusoidal wave
# with a gaussian envelope. We can control the balance between spectral and
# temporal resolution with the ``n_cycles`` parameter, which defines the
# number of cycles to include in the window.

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
all_n_cycles = [1, 3, freqs / 2.0]
for n_cycles, ax in zip(all_n_cycles, axs):
    power = epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True
    )
    power.plot(
        [0],
        baseline=(0.0, 0.1),
        mode="mean",
        vlim=(vmin, vmax),
        axes=ax,
        show=False,
        colorbar=False,
    )
    n_cycles = "scaled by freqs" if not isinstance(n_cycles, int) else n_cycles
    ax.set_title(f"Sim: Using Morlet wavelet, n_cycles = {n_cycles}")

# %%
# Narrow-bandpass Filter and Hilbert Transform
# ============================================
#
# Finally, we'll show a time-frequency representation using a narrow bandpass
# filter and the Hilbert transform. Choosing the right filter parameters is
# important so that you isolate only one oscillation of interest, generally
# the width of this filter is recommended to be about 2 Hz.

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
bandwidths = [1.0, 2.0, 4.0]
for bandwidth, ax in zip(bandwidths, axs):
    data = np.zeros(
        (len(epochs), len(ch_names), freqs.size, epochs.times.size), dtype=complex
    )
    for idx, freq in enumerate(freqs):
        # Filter raw data and re-epoch to avoid the filter being longer than
        # the epoch data for low frequencies and short epochs, such as here.
        raw_filter = raw.copy()
        # NOTE: The bandwidths of the filters are changed from their defaults
        # to exaggerate differences. With the default transition bandwidths,
        # these are all very similar because the filters are almost the same.
        # In practice, using the default is usually a wise choice.
        raw_filter.filter(
            l_freq=freq - bandwidth / 2,
            h_freq=freq + bandwidth / 2,
            # no negative values for large bandwidth and low freq
            l_trans_bandwidth=min([4 * bandwidth, freq - bandwidth]),
            h_trans_bandwidth=4 * bandwidth,
        )
        raw_filter.apply_hilbert()
        epochs_hilb = Epochs(
            raw_filter, events, tmin=0, tmax=n_times / sfreq, baseline=(0, 0.1)
        )
        data[:, :, idx] = epochs_hilb.get_data()
    power = EpochsTFRArray(epochs.info, data, epochs.times, freqs, method="hilbert")
    power.average().plot(
        [0],
        baseline=(0.0, 0.1),
        mode="mean",
        vlim=(0, 0.1),
        axes=ax,
        show=False,
        colorbar=False,
    )
    n_cycles = "scaled by freqs" if not isinstance(n_cycles, int) else n_cycles
    ax.set_title(
        "Sim: Using narrow bandpass filter Hilbert,\n"
        f"bandwidth = {bandwidth}, "
        f"transition bandwidth = {4 * bandwidth}"
    )

# %%
# Calculating a TFR without averaging over epochs
# -----------------------------------------------
#
# It is also possible to calculate a TFR without averaging across trials.
# We can do this by using ``average=False``. In this case, an instance of
# :class:`mne.time_frequency.EpochsTFR` is returned.

n_cycles = freqs / 2.0
power = epochs.compute_tfr(
    method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False
)
print(type(power))
avgpower = power.average()
avgpower.plot(
    [0],
    baseline=(0.0, 0.1),
    mode="mean",
    vlim=(vmin, vmax),
    title="Using Morlet wavelets and EpochsTFR",
    show=False,
)

# %%
# Operating on arrays
# -------------------
#
# MNE-Python also has functions that operate on :class:`NumPy arrays <numpy.ndarray>`
# instead of MNE-Python objects. These are :func:`~mne.time_frequency.tfr_array_morlet`
# and :func:`~mne.time_frequency.tfr_array_multitaper`. They expect inputs of the shape
# ``(n_epochs, n_channels, n_times)`` and return an array of shape
# ``(n_epochs, n_channels, n_freqs, n_times)`` (or optionally, can collapse the epochs
# dimension if you want average power or inter-trial coherence; see ``output`` param).

power = tfr_array_morlet(
    epochs.get_data(),
    sfreq=epochs.info["sfreq"],
    freqs=freqs,
    n_cycles=n_cycles,
    output="avg_power",
    zero_mean=False,
)
# Put it into a TFR container for easy plotting
tfr = AverageTFRArray(
    info=epochs.info, data=power, times=epochs.times, freqs=freqs, nave=len(epochs)
)
tfr.plot(
    baseline=(0.0, 0.1),
    picks=[0],
    mode="mean",
    vlim=(vmin, vmax),
    title="TFR calculated on a NumPy array",
)
