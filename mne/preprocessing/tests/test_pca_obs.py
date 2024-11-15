"""Test the ieeg projection functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# TODO: migrate this structure to test out function

import pytest

from mne.io import read_raw_fif
from mne.preprocessing.pca_obs import pca_obs
from mne.datasets.testing import data_path, requires_testing_data

# TODO: Where are the test files we want to use located?
fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"

@requires_testing_data
@pytest.mark.parametrize(
    # TODO: Are there any parameters we can cycle through to 
    # test multiple? Different fs, windows, highpass freqs, etc.?
    # TODO: how do we determine qrs and filter_coords? What are these?
    "fs, highpass_freq, qrs, filter_coords",
    [
        (0.2, 1.0, 100, 200),
        (0.1, 2.0, 100, 200),
    ],
)
def test_heart_artifact_removal(fs, highpass_freq, qrs, filter_coords):
    """Test PCA-OBS analysis and heart artifact removal of ECG datasets."""
    raw = read_raw_fif(fname)

    # Do something with fs and highpass as processing of the data?
    ...

    # call pca_obs algorithm
    result = pca_obs(raw, qrs=qrs, filter_coords=filter_coords)

    # assert results
    assert result is not None
    assert result.shape == (100, 100)
    assert result.shape == raw.shape  # is this a condition we can test? 
    assert result[0, 0] == 1.0
    ... 