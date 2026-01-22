# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

neo = pytest.importorskip("neo")


def _get_trc_test_file():
    """Find a Micromed TRC test file if available."""
    from pathlib import Path

    # Check common locations for test data
    test_paths = [
        Path("/private/tmp/test_micromed"),
        Path.home() / "mne_data",
    ]

    for base in test_paths:
        if base.exists():
            trc_files = list(base.rglob("*.trc")) + list(base.rglob("*.TRC"))
            if trc_files:
                return trc_files[0]
    return None


@pytest.fixture
def trc_file():
    """Get a TRC test file or skip."""
    fpath = _get_trc_test_file()
    if fpath is None:
        pytest.skip("No Micromed TRC test file available")
    return fpath


def test_micromed_reading(trc_file):
    """Test reading Micromed TRC file."""
    from mne.io import read_raw_micromed

    raw = read_raw_micromed(trc_file, preload=False)

    assert raw.info["sfreq"] > 0
    assert len(raw.ch_names) > 0
    assert raw.n_times > 0

    # Test lazy loading
    data, times = raw[:, :100]
    assert data.shape[1] == 100

    # Test full load
    raw.load_data()
    assert raw.preload


def test_micromed_channel_types(trc_file):
    """Test that channel types are set to sEEG."""
    from mne.io import read_raw_micromed

    raw = read_raw_micromed(trc_file, preload=False)
    ch_types = set(raw.get_channel_types())

    # Default should be sEEG
    assert "seeg" in ch_types


def test_micromed_data_types(trc_file):
    """Test that data is returned as float64."""
    from mne.io import read_raw_micromed

    raw = read_raw_micromed(trc_file, preload=True)
    data = raw.get_data()

    assert data.dtype == np.float64
