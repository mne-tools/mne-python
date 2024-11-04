"""Test the fot_ecg_template function."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from mne.io import read_raw_fif
from mne.preprocessing.pca_obs import fit_ecg_template
from mne.datasets.testing import data_path, requires_testing_data

# TODO: Where are the test files we want to use located?
fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@requires_testing_data
def test_fit_ecg_template():
    """Test PCA-OBS analysis and heart artifact removal of ECG datasets."""
    raw = read_raw_fif(fname)

    # Somehow have to "fake" all these inputs to the function
    result = fit_ecg_template(
        data=None,
        pca_template=None,
        aPeak_idx=None,
        peak_range=None,
        pre_range=None,
        post_range=None,
        midP=None,
        fitted_art=None,
        post_idx_previousPeak=None,
        n_samples_fit=None,
    )

    # assert results
    assert result is not None
    assert result.shape == (100, 100)
    assert result.shape == raw.shape  # is this a condition we can test? 
    assert result[0, 0] == 1.0
    ... 