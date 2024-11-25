"""Test the ieeg projection functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import firls

from mne.io import read_raw_fif
from mne.preprocessing import apply_pca_obs
from mne.preprocessing.ecg import find_ecg_events

data_path = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_path / "test_raw.fif"


@pytest.fixture()
def short_raw_data():
    """Create a short, picked raw instance."""
    return read_raw_fif(raw_fname, preload=True).crop(0, 7)


@pytest.mark.parametrize(
    # TODO: Are there any parameters we can cycle through to
    # test multiple? Different fs, windows, highpass freqs, etc.?
    # TODO: how do we determine qrs and filter_coords? What are these?
    ("fs", "highpass_freq", "qrs", "filter_coords"),
    [
        (0.2, 1.0, 100, 200),
        (0.1, 2.0, 100, 200),
    ],
)
def test_heart_artifact_removal(short_raw, fs, highpass_freq, qrs, filter_coords):
    """Test PCA-OBS analysis and heart artifact removal of ECG datasets."""
    # get the sampling frequency of the test data and
    # generate the filter coords as in our example
    fs = short_raw.info["sfreq"]
    a = [0, 0, 1, 1]
    f = [0, 0.4 / (fs / 2), 0.9 / (fs / 2), 1]  # 0.9 Hz highpass filter
    ord_ = round(3 * fs / 0.5)
    filter_coords = firls(ord_ + 1, f, a)

    # extract the QRS
    ecg_events, _, _ = find_ecg_events(short_raw, ch_name=None)
    ecg_event_samples = np.asarray([[ecg_event[0] for ecg_event in ecg_events]])

    # copy the original raw and remove the heart artifact in-place
    raw_orig = copy.deepcopy(short_raw)
    apply_pca_obs(
        raw=short_raw,
        picks=["eeg"],
        qrs=ecg_event_samples,
        filter_coords=filter_coords,
    )
    # raw.get_data() ? to get shapes to compare

    assert raw_orig != short_raw

    # # Do something with fs and highpass as processing of the data?

    # # call pca_obs algorithm
    # result = pca_obs(raw, qrs=qrs, filter_coords=filter_coords)

    # # assert results
    # assert result is not None
    # assert result.shape == (100, 100)
    # assert result.shape == raw.shape  # is this a condition we can test?
    # assert result[0, 0] == 1.0


if __name__ == "__main__":
    pytest.main(["mne/preprocessing/tests/test_pca_obs.py"])
