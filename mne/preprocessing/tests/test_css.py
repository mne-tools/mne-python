# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from mne import pick_types, read_evokeds
from mne.datasets import testing
from mne.preprocessing._css import cortical_signal_suppression

data_path = testing.data_path(download=False)
fname_evoked = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"


@testing.requires_testing_data
def test_cortical_signal_suppression():
    """Test that CSS is dampening the cortical signal and has right shape."""
    ave = read_evokeds(fname_evoked)[0]
    eeg_ind = pick_types(ave.info, eeg=True)
    mag_ind = pick_types(ave.info, meg="mag")
    grad_ind = pick_types(ave.info, meg="grad")
    ave.data[mag_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * np.mean(
        np.abs(ave.data[mag_ind][0, :])
    )
    ave.data[mag_ind][1, :] = np.sin(2 * np.pi * 239 * ave.times) * np.mean(
        np.abs(ave.data[mag_ind][1, :])
    )
    ave.data[grad_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * np.mean(
        np.abs(ave.data[grad_ind][0, :])
    )
    ave.data[eeg_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * np.mean(
        np.abs(ave.data[eeg_ind][0, :])
    )
    ave.data[eeg_ind][1, :] = np.sin(2 * np.pi * 239 * ave.times) * np.mean(
        np.abs(ave.data[eeg_ind][1, :])
    )
    # include test for gh-12373, that you can use MAG+EEG if you want
    for mag_picks, ind in ((None, eeg_ind), ("eeg", mag_ind)):
        ave_f = cortical_signal_suppression(ave, mag_picks=mag_picks)
        assert ave_f.data.shape == ave.data.shape
        cort_power = np.linalg.norm(ave.data[ind][0, :])
        deep_power = np.linalg.norm(ave.data[ind][1, :])
        cort_power_f = np.linalg.norm(ave_f.data[ind][0, :])
        deep_power_f = np.linalg.norm(ave_f.data[ind][1, :])
        rel_SNR_gain = (deep_power_f / deep_power) / (cort_power_f / cort_power)
        assert rel_SNR_gain > 3, mag_picks
