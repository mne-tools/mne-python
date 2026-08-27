"""
.. _tut-variable-duration-epochs:

Epochs whose trials have different durations
============================================

Most epoching starts from an event and takes the same window around every one of
them, which gives a rectangular ``(n_epochs, n_channels, n_times)`` array and one
time axis shared by every trial. Some experiments do not fit that shape. A gait
cycle, a spoken word, a reaching movement and a sleep stage all last as long as
they last, and the duration is often the thing being studied.

The usual way to handle this is to pick a fixed window and accept the
consequences: a window long enough for the longest trial pads the short ones, and
a window short enough for the shortest one truncates the rest. This tutorial
shows the other option, keeping each trial at the length it actually had, and
what the resulting object can and cannot do.

We use the Sleep Physionet data, where the hypnogram annotations mark sleep stage
bouts and each bout carries its own duration.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets.sleep_physionet.age import fetch_data

psg_file, hypnogram_file = fetch_data(subjects=[0], recording=[1])[0]

raw = mne.io.read_raw_edf(
    psg_file,
    stim_channel=False,
    preload=True,
    verbose="error",  # ignore issues with stored filter settings
)
raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])

annotations = mne.read_annotations(hypnogram_file)
raw.set_annotations(annotations, emit_warning=False)

# %%
# The annotations already carry durations
# ---------------------------------------
#
# Each hypnogram entry marks one bout of a sleep stage, and
# :class:`~mne.Annotations` stores its ``duration`` alongside its ``onset``. That
# duration is not a constant.

stages = {
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # stages 3 and 4 are conventionally merged
    "Sleep stage R": 4,
}
event_id = {"N1": 1, "N2": 2, "N3/4": 3, "REM": 4}

onset = annotations.onset
duration = annotations.duration
description = np.array(annotations.description)

keep = np.array([desc in stages for desc in description])
# a five minute cap keeps the padded array in the last section small; nothing
# about the container requires it
keep &= duration <= 300.0

onset, duration = onset[keep], duration[keep]
description = description[keep]

print(f"{len(duration)} bouts, {duration.min():.0f} to {duration.max():.0f} s")
print(f"median {np.median(duration):.0f} s")

# %%
# Building epochs that keep those durations
# -----------------------------------------
#
# ``tmin`` and ``tmax`` accept one value per event as well as a single number.
# Here every bout starts at its own onset, so ``tmin`` is zero throughout and
# ``tmax`` is the bout's own length. The last sample is included, which is why
# ``tmax`` is one sample short of the full duration.

sfreq = raw.info["sfreq"]
events = np.column_stack(
    [
        np.round((onset - raw.first_time) * sfreq).astype(int),
        np.zeros(len(onset), int),
        np.array([stages[desc] for desc in description]),
    ]
)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=np.zeros(len(events)),
    tmax=duration - 1.0 / sfreq,
    baseline=None,
    preload=True,
)
print(epochs)

# %%
# The object reports that its trials are not all the same length, and the
# durations it holds are the ones the annotations described.

print(f"variable_duration: {epochs.variable_duration}")
print(f"durations: {epochs.durations.min():.0f} to {epochs.durations.max():.0f} s")

# %%
# Because bounds that carry no variation collapse back to a single value, this
# only changes behaviour when the durations really do differ. Passing equal
# bounds gives an ordinary fixed-duration ``Epochs``.

n2_events = events[events[:, 2] == event_id["N2"]][:5]
fixed_bounds = mne.Epochs(
    raw,
    n2_events,
    {"N2": event_id["N2"]},
    tmin=np.zeros(len(n2_events)),
    tmax=np.full(len(n2_events), 29.99),
    baseline=None,
    preload=True,
    verbose=False,
)
print(f"equal bounds -> variable_duration: {fixed_bounds.variable_duration}")

# %%
# Getting the data out
# --------------------
#
# There is no rectangular array to return, so :meth:`~mne.Epochs.get_data` gives
# a list with one ``(n_channels, n_times)`` array per epoch. Nothing is padded
# and nothing is cut: each array holds exactly the samples that the bout covered
# in the continuous recording.

data = epochs.get_data()
print(f"{len(data)} arrays, first four shapes {[d.shape for d in data[:4]]}")

lengths = np.array([d.shape[-1] for d in data])
print(f"total samples held: {lengths.sum()}")
print(f"a rectangular array would hold: {lengths.max() * len(lengths)}")

# %%
# For the same reason there is no single ``times`` attribute. Each epoch has its
# own time axis, which :meth:`~mne.Epochs.get_times` returns.

for idx in (0, 1):
    t = epochs.get_times(idx)
    print(f"epoch {idx}: {len(t)} samples, {t[0]:.2f} to {t[-1]:.2f} s")

# %%
# Asking for ``epochs.times`` raises rather than inventing an axis. Returning the
# longest epoch's axis would make ``len(epochs.times)`` disagree with the data
# for every other epoch while looking perfectly normal.

try:
    epochs.times
except RuntimeError as err:
    print(f"RuntimeError: {err}")

# %%
# Operations that do not touch the time axis
# ------------------------------------------
#
# Selecting epochs, selecting channels and dropping epochs all work as usual,
# because none of them care how long each trial is. The per-epoch bounds travel
# with the epochs they belong to.

n2 = epochs["N2"]
print(
    f"epochs['N2']: {len(n2)} epochs, "
    f"{n2.durations.min():.0f} to {n2.durations.max():.0f} s"
)

first_ten = epochs[:10]
print(f"epochs[:10]: durations {first_ten.durations.round(0)}")

one_channel = epochs.copy().pick(["EEG Pz-Oz"])
print(
    f"after pick: {one_channel.ch_names}, durations unchanged: "
    f"{np.array_equal(one_channel.durations, epochs.durations)}"
)

# %%
# Browsing them
# -------------
#
# :meth:`~mne.Epochs.plot` shows each bout at the length it really has. The
# browser lays the variable-length blocks end to end and rules a line between
# them, so the vertical boundaries are unevenly spaced: a 30 second bout takes a
# fifth of the width of a 150 second one. Nothing is padded or truncated to make
# the picture rectangular, and :meth:`~mne.Epochs.as_fixed` is not involved.
#
# Pick a handful of bouts with genuinely different lengths, taking the first
# occurrence of each distinct duration rather than trusting the first few epochs
# to differ.

_, first_of_each = np.unique(epochs.durations, return_index=True)
browse_idx = np.sort(first_of_each[:5])
browse_epochs = epochs[browse_idx]
print(f"browsing durations: {browse_epochs.durations.round(0)} s")

# the browser's time axis is the real samples, laid end to end
n_browser_samples = sum(
    len(browse_epochs.get_times(ii)) for ii in range(len(browse_epochs))
)
print(f"{n_browser_samples} samples in total, none of them padding")

# %%
# Browsing variable-duration epochs currently needs the Matplotlib backend; the
# PyQtGraph one does not handle ragged epochs yet.

with mne.viz.use_browser_backend("matplotlib"):
    browse_epochs.plot(n_epochs=len(browse_epochs), picks="eeg")

# %%
# Operations that need one time axis
# ----------------------------------
#
# Averaging is the clearest case. :class:`~mne.Evoked` holds one array and one
# ``nave``, and there is no honest way to fill either when the trials stop at
# different times. Rather than pad quietly, the reduction refuses and says what
# it would need.

try:
    epochs.average()
except NotImplementedError as err:
    print(f"NotImplementedError: {err}")

# %%
# Making the padding explicit
# ---------------------------
#
# When a rectangular array is genuinely what you want,
# :meth:`~mne.Epochs.as_fixed` produces one. It returns the padded
# :class:`~mne.EpochsArray` together with the number of epochs contributing at
# each sample, so the cost of the padding is visible rather than implied.

padded, n_contributing = epochs.as_fixed()
print(f"padded shape: {padded.get_data().shape}")
print(
    f"contributing: {n_contributing.max()} at the start, "
    f"{n_contributing.min()} at the end"
)

held = lengths.sum() * len(epochs.ch_names)
allocated = padded.get_data().size
print(f"padding waste: {100 * (1 - held / allocated):.1f}%")

# %%
# That second return value is the point of the method. Plotted against time it
# shows how quickly the epochs stop contributing, which is exactly the
# information an averaged :class:`~mne.Evoked` cannot carry.

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
times = padded.times
ax.fill_between(times, n_contributing, step="post", alpha=0.25)
ax.plot(times, n_contributing, drawstyle="steps-post")

half = len(epochs) / 2
crossing = times[np.argmax(n_contributing < half)]
ax.axhline(half, color="0.4", ls=":", lw=1)
ax.axvline(crossing, color="0.4", ls=":", lw=1)
ax.annotate(
    f"half the epochs have ended by {crossing:.0f} s",
    xy=(crossing, half),
    xytext=(crossing + 20, len(epochs) * 0.7),
    arrowprops=dict(arrowstyle="->", color="0.4"),
)

ax.set(
    xlabel="Time (s)",
    ylabel="Epochs contributing",
    title="How many sleep-stage bouts are still running",
    xlim=(0, times[-1]),
    ylim=(0, len(epochs) * 1.05),
)

# %%
# Reading the figure from left to right: every bout contributes at the start,
# and by the end a single long bout is holding up the whole window. An average
# over this padded array would combine all of them at ``t = 0`` and one of them
# at the right-hand edge, while reporting one ``nave`` for the lot. Keeping the
# count alongside the data is what makes that visible.
#
# When fixed windows are the right choice
# ---------------------------------------
#
# None of this argues against fixed-length epochs. Sleep staging is a good
# example of when they are correct: :ref:`tut-sleep-stage-classif` classifies 30
# second windows, so it passes ``chunk_duration=30.`` to
# :func:`mne.events_from_annotations` and deliberately turns each bout into a
# series of equal windows. That is the right representation when the window is
# the unit of analysis.
#
# Variable-duration epochs are for the other case, when the bout itself is the
# unit and its length is part of what is being measured.
