"""Phantom dipole simulations.

.. _tut-brainstorm-elekta-phantom:

==========================================
Phantom dipole simulations
==========================================

"""
# sphinx_gallery_thumbnail_number = 9

# Authors: Carina Forster <carinaforster0611@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

# %%
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
for dipole in list(range(1, 33)):
    epochs = mne.Epochs(
        raw, events, dipole, tmin, tmax, baseline=(None, bmax), preload=True
    )

    n_epochs, n_channels, _ = epochs.get_data().shape

    # create channel mask
    mask = np.zeros((n_epochs, n_channels))

    # mark fifth epoch channel 0 for NaN
    mask[5, 0] = True

    # set first epoch for first channel to NaN
    epochs.drop_bad_epochs_by_channel(mask)

    # make sure function sets first epoch first channel to NaN
    assert np.isnan(epochs.get_data()[5, 0]).all()

    # store the dipole estimations
    onedipole_epochs = []

    # estimate dipoles n_epochs
    for a in range(18):
        # ignore first and last epoch (corrupted)
        test_epochs = epochs[(1 + a) : -1]

        # calculate covariance for test epochs
        cov = mne.compute_covariance(test_epochs, tmax=bmax)
        # covariance estimation error: ValueError: array must not contain infs or NaNs

        # let's drop the whole epoch
        epochs_for_cov = epochs.copy().drop([5])

        # but this defeats the purpose
        # (then we can just drop the whole epoch from the start)
        cov = mne.compute_covariance(epochs_for_cov, tmax=bmax)


# Not sure if this addresses the issue, but we can calculate covariance on
# full epochs and then evoked on less epochs for one channel and see what
# happens to the dipole fitting error

# parameters
t_peak = 0.036
imbalance_levels = [
    1.0,
    0.75,
    0.5,
    0.25,
    0.1,
]  # fraction of epochs kept for one channel
n_repeats = 10  # random resampling for stability

dipole_errors = []

for dipole in range(1, 33):
    epochs = mne.Epochs(
        raw, events, dipole, tmin, tmax, baseline=(None, bmax), preload=True
    )

    # remove first and last epoch
    data = epochs.get_data()[1:-1, :, :]
    n_epochs, n_channels, n_times = data.shape

    # compute covariance from full data
    cov = mne.compute_covariance(epochs, tmax=bmax)

    dipole_errors_per_level = []

    for frac in imbalance_levels:
        n_keep_ch0 = int(frac * n_epochs)
        n_keep_other = n_epochs

        errors = []

        for _ in range(n_repeats):
            # random epoch selection
            keep_ch0 = np.random.choice(n_epochs, n_keep_ch0, replace=False)
            keep_other = np.arange(n_epochs)

            # construct unequal averaged evoked
            evoked_data = np.zeros((n_channels, n_times))

            # channel 0 averaged with fewer epochs
            evoked_data[0] = data[keep_ch0, 0].mean(axis=0)

            # other channels averaged normally
            for ch in range(1, n_channels):
                evoked_data[ch] = data[keep_other, ch].mean(axis=0)

            evoked = mne.EvokedArray(evoked_data, epochs.info, tmin=epochs.times[0])

            evoked_peak = evoked.copy().crop(t_peak, t_peak)

            dip, _ = fit_dipole(evoked_peak, cov, sphere)

            # compute localization error
            actual_pos, _ = mne.dipole.get_phantom_dipoles()
            true_pos = actual_pos[dipole - 1]

            loc_error = 1000 * np.linalg.norm(dip.pos[0] - true_pos)
            errors.append(loc_error)

        dipole_errors_per_level.append(np.mean(errors))

    dipole_errors.append(dipole_errors_per_level)

# convert to mm
imbalance_percent = [100 * f for f in imbalance_levels]

mean_errors = np.mean(dipole_errors, axis=0)

plt.figure()
plt.plot(imbalance_percent, mean_errors)
plt.gca().invert_xaxis()
plt.xlabel("Epochs retained for channel 0 (%)")
plt.ylabel("Localization error (mm)")
plt.title("Effect of Unequal Epoch Counts on Dipole Fitting")
plt.show()
