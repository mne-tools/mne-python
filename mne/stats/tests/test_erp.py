# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import pytest

from mne import Epochs, read_events
from mne.io import read_raw_fif
from mne.stats.erp import compute_sme

base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw = read_raw_fif(base_dir / "test_raw.fif")
events = read_events(base_dir / "test-eve.fif")


def test_compute_sme():
    """Test SME computation."""
    epochs = Epochs(raw, events)
    sme = compute_sme(epochs, start=0, stop=0.1)
    assert sme.shape == (376,)

    with pytest.raises(TypeError, match="int or float"):
        compute_sme(epochs, "0", 0.1)
    with pytest.raises(TypeError, match="int or float"):
        compute_sme(epochs, 0, "0.1")
    with pytest.raises(ValueError, match="out of bounds"):
        compute_sme(epochs, -1.2, 0.3)
    with pytest.raises(ValueError, match="out of bounds"):
        compute_sme(epochs, -0.1, 0.8)
