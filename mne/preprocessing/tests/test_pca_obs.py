"""Test the ieeg projection functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mne.io import read_raw_fif
from mne.io.fiff.raw import Raw
from mne.preprocessing import apply_pca_obs

data_path = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_path / "test_raw.fif"


@pytest.fixture()
def short_raw_data():
    """Create a short, picked raw instance."""
    return read_raw_fif(raw_fname, preload=True)


def test_heart_artifact_removal(short_raw_data: Raw):
    """Test PCA-OBS analysis and heart artifact removal of ECG datasets."""
    # fake some random qrs events
    ecg_event_samples = np.arange(0, len(short_raw_data.times), 1400) + 1430

    # copy the original raw. heart artifact is removed in-place
    orig_df: pd.DataFrame = short_raw_data.to_data_frame().copy(deep=True)

    # perform heart artifact removal
    apply_pca_obs(raw=short_raw_data, picks=["eeg"], qrs=ecg_event_samples, n_jobs=1)

    # compare processed df to original df
    removed_heart_artifact_df: pd.DataFrame = short_raw_data.to_data_frame()

    # ensure all column names remain the same
    pd.testing.assert_index_equal(
        orig_df.columns,
        removed_heart_artifact_df.columns,
    )

    # ensure every column starting with EEG has been altered
    altered_cols = [c for c in orig_df.columns if c.startswith("EEG")]
    for col in altered_cols:
        with pytest.raises(
            AssertionError
        ):  # make sure that error is raised when we check equal
            pd.testing.assert_series_equal(
                orig_df[col],
                removed_heart_artifact_df[col],
            )

    # ensure every column not starting with EEG has not been altered
    unaltered_cols = [c for c in orig_df.columns if not c.startswith("EEG")]
    pd.testing.assert_frame_equal(
        orig_df[unaltered_cols],
        removed_heart_artifact_df[unaltered_cols],
    )
