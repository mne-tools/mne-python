# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import pytest

from mne import Epochs, read_events
from mne.io import read_raw_fif
from mne.stats.erp import (
    compute_sme,
    get_area,
    get_frac_area_latency,
    get_frac_peak_latency,
    get_peak,
)

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


def test_get_peak():
    """Test peak computation."""
    epochs = Epochs(raw, events)
    evoked = epochs.average()
    peak_df = get_peak(evoked, tmin=0, tmax=0.1)
    evoked_pos = evoked.copy()
    evoked_pos.data = abs(evoked_pos.data)
    get_peak(evoked_pos, tmin=0, tmax=0.1, mode="neg", strict=False)

    evoked_neg = evoked.copy()
    evoked_neg.data = -abs(evoked_neg.data)
    get_peak(evoked_neg, tmin=0, tmax=0.1, mode="pos", strict=False)

    assert peak_df.shape == (366, 3)

    with pytest.raises(ValueError, match="No negative values encountered"):
        evoked_pos.data = abs(evoked_pos.data)
        get_peak(evoked_pos, tmin=0, tmax=0.1, mode="neg", strict=True)
    with pytest.raises(ValueError, match="No positive values encountered"):
        get_peak(evoked_neg, tmin=0, tmax=0.1, mode="pos", strict=True)


def test_get_area():
    """Test area computation."""
    epochs = Epochs(raw, events)
    evoked = epochs.average()
    area_df = get_area(evoked, tmin=0, tmax=0.1)
    assert area_df.shape == (366, 2)


def test_get_frac_peak_latency():
    """Test fractional peak latency computation."""
    epochs = Epochs(raw, events)
    evoked = epochs.average()
    frac_peak_df = get_frac_peak_latency(evoked, frac=0.5, tmin=0, tmax=0.1)
    assert frac_peak_df.shape == (366, 4)

    with pytest.raises(ValueError, match="No negative values encountered"):
        evoked_pos = evoked.copy()
        evoked_pos.data = abs(evoked_pos.data)
        get_frac_peak_latency(
            evoked, frac=0.5, tmin=0, tmax=0.1, mode="neg", strict=True
        )
    with pytest.raises(ValueError, match="No positive values encountered"):
        evoked_neg = evoked.copy()
        evoked_neg.data = -abs(evoked_neg.data)
        get_frac_peak_latency(
            evoked, frac=0.5, tmin=0, tmax=0.1, mode="pos", strict=True
        )


def test_get_frac_area_latency():
    """Test fractional area latency computation."""
    epochs = Epochs(raw, events)
    evoked = epochs.average()
    frac_area_df = get_frac_area_latency(evoked, frac=0.5, tmin=0, tmax=0.1)
    assert frac_area_df.shape == (366, 3)
