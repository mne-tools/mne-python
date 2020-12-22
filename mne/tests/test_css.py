# Author: John G Samuelsson <johnsam@mit.edu>

from mne.preprocessing._css import cortical_signal_suppression
from mne import pick_types, read_evokeds
from mne.datasets import sample
import numpy as np
import pytest


def test_cortical_signal_suppression():
    """Test that CSS is dampening the cortical signal and has right shape."""
    data_path = sample.data_path()
    ave = read_evokeds(data_path + '/MEG/sample/sample_audvis-ave.fif')[0]
    eeg_ind = pick_types(ave.info, meg=False, eeg=True)
    mag_ind = pick_types(ave.info, meg='mag', eeg=False)
    grad_ind = pick_types(ave.info, meg='grad', eeg=False)
    ave.data[mag_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * \
        np.mean(np.abs(ave.data[mag_ind][0, :]))
    ave.data[mag_ind][1, :] = np.sin(2 * np.pi * 239 * ave.times) * \
        np.mean(np.abs(ave.data[mag_ind][1, :]))
    ave.data[grad_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * \
        np.mean(np.abs(ave.data[grad_ind][0, :]))
    ave.data[eeg_ind][0, :] = np.sin(2 * np.pi * 40 * ave.times) * \
        np.mean(np.abs(ave.data[eeg_ind][0, :]))
    ave.data[eeg_ind][1, :] = np.sin(2 * np.pi * 239 * ave.times) * \
        np.mean(np.abs(ave.data[eeg_ind][1, :]))
    ave_f = cortical_signal_suppression(ave)
    cort_power = np.sum(np.abs(ave.data[eeg_ind][0, :]))
    deep_power = np.sum(np.abs(ave.data[eeg_ind][1, :]))
    cort_power_f = np.sum(np.abs(ave_f.data[eeg_ind][0, :]))
    deep_power_f = np.sum(np.abs(ave_f.data[eeg_ind][1, :]))
    rel_SNR_gain = (deep_power_f / deep_power) / (cort_power_f / cort_power)
    assert rel_SNR_gain > 0
    assert ave_f.data.shape == ave.data.shape
    pytest.raises(ValueError, cortical_signal_suppression, mag_ind)
