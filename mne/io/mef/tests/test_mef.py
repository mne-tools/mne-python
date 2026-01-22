# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

pymef = pytest.importorskip("pymef")


def _get_mef_test_file():
    """Find a MEF3 test file if available."""
    from pathlib import Path

    # Check common locations for test data
    test_paths = [
        Path("/private/tmp/test_micromed"),
        Path.home() / "mne_data",
    ]

    for base in test_paths:
        if base.exists():
            mef_dirs = list(base.rglob("*.mefd"))
            if mef_dirs:
                return mef_dirs[0]
    return None


@pytest.fixture
def mef_file():
    """Get a MEF3 test file or skip."""
    fpath = _get_mef_test_file()
    if fpath is None:
        pytest.skip("No MEF3 test file available")
    return fpath


def test_mef_reading(mef_file):
    """Test reading MEF3 file."""
    from mne.io import read_raw_mef

    raw = read_raw_mef(mef_file, preload=False)

    assert raw.info["sfreq"] > 0
    assert len(raw.ch_names) > 0
    assert raw.n_times > 0

    # Test lazy loading
    data, times = raw[:, :100]
    assert data.shape[1] == 100

    # Test full load
    raw.load_data()
    assert raw.preload


def test_mef_channel_types(mef_file):
    """Test that channel types are set to sEEG."""
    from mne.io import read_raw_mef

    raw = read_raw_mef(mef_file, preload=False)
    ch_types = set(raw.get_channel_types())

    # Default should be sEEG
    assert "seeg" in ch_types


def test_mef_data_types(mef_file):
    """Test that data is returned as float64."""
    from mne.io import read_raw_mef

    raw = read_raw_mef(mef_file, preload=True)
    data = raw.get_data()

    assert data.dtype == np.float64
