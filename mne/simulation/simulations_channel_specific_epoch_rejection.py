"""
Channel-specific epoch rejection simulation.

.. _tut-brainstorm-elekta-phantom:

==========================================
Phantom dipole simulations
==========================================
"""


# Authors:
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# The data were collected with an Elekta Neuromag VectorView system
# at 1000 Hz and low-pass filtered at 330 Hz.
# Here the medium-amplitude (200 nAm, amplitudes can be seen in raw data)
# data are read to construct instances of :class:`mne.io.Raw`.
data_path = bst_phantom_elekta.data_path(verbose=True)

raw_fname = data_path / "kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif"
raw = read_raw_fif(raw_fname)

# %%
# The data channel array consisted of 204 MEG planor gradiometers,
# 102 axial magnetometers, and 3 stimulus channels.

# Next, let's look at the events in the phantom data for one stimulus channel:
events = find_events(raw, "STI201")
raw.info["bads"] = ["MEG1933", "MEG2421"]  # known bad channels

# setup headmodel
subjects_dir = data_path
fetch_phantom("otaniemi", subjects_dir=subjects_dir)
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)
subject = "phantom_otaniemi"
trans = mne.transforms.Transform("head", "mri", np.eye(4))

# We can see that the simulated dipoles produce sinusoidal bursts at 20 Hz
# %%
# Next, we epoch the the data based on the dipoles events (1:32)
# We select 100 ms before and 100ms after the event trigger
# and baseline correct the epochs from -100 ms to - 0.05 before stimulus onset

tmin, tmax = -0.1, 0.1
bmax = -0.05  # Avoid capture filter ringing into baseline

# loop over all dipoles (32)
dip_epochs = []

for dipole in list(range(1, 5)):
    epochs = mne.Epochs(
        raw, events, dipole, tmin, tmax, baseline=(None, bmax), preload=True
    )

    n_epochs = len(epochs.events)
    n_channels = len(epochs.ch_names)

    epochs_per_channel_array = np.zeros((n_epochs, n_channels))

    # set half of epochs for half of channels to bad
    epochs_per_channel_array[: n_epochs // 2, : n_channels // 2] = 1

    epochs.drop_bad_epochs_by_channel(epochs_per_channel_array)

    # calculate covariance for test epochs
    cov = mne.compute_covariance(epochs, tmax=bmax)

    # Next, we fit the dipoles for the evoked data.
    # We choose the timepoint which maximises global field power
    t_peak = 0.036  # true for Elekta phantom
    evoked = epochs.average().crop(t_peak, t_peak)
    dip, _ = fit_dipole(evoked, cov, sphere, n_jobs=None)

    dip_epochs.append(epochs)

# dip epochs is a list that contains a list with 32 dipoles and
# a list for each dipole with 18 epochs

# get all true dipole positions
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()

for idx, dipole in enumerate(dip_epochs[:5]):
    # select simulated dipoles (we skipped first and last dipole)
    true_pos = actual_pos[idx, :]
    true_ori = actual_ori[idx, :]

    true_amp = 100.0  # nAm

    dip_pos = [dip.pos for dip in dipole]
    dip_ori = [dip.ori for dip in dipole]
    dip_amp = [dip.amplitude for dip in dipole]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(10, 7), layout="constrained"
    )

    n_epochs = list(range(1, 19))[::-1]

    pos_diff = [
        1000 * np.sqrt(np.sum((pos - true_pos) ** 2, axis=-1)) for pos in dip_pos
    ]
    ax1.plot(n_epochs, pos_diff)
    ax1.set_xlabel("Number of epochs")
    ax1.set_ylabel("Loc. error (mm)")
    ax1.set_title(f"Dipole {idx}")

    angle_diff = [
        np.rad2deg(np.arccos(np.abs(np.sum(ori * true_ori, axis=1)))) for ori in dip_ori
    ]
    ax2.plot(n_epochs, angle_diff)
    ax2.set_xlabel("Number of epochs")
    ax2.set_ylabel("Angle error (°)")

    amp_diff = [true_amp - amp / 1e-9 for amp in dip_amp]
    ax3.plot(n_epochs, amp_diff)
    ax3.set_xlabel("Number of epochs")
    ax3.set_ylabel("Amplitude error (nAm)")
