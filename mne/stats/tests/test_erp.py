# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import Epochs, EvokedArray, create_info, read_events
from mne.io import read_raw_fif
from mne.stats.erp import (
    compute_area,
    compute_frac_area_latency,
    compute_frac_peak_latency,
    compute_peak,
    compute_sme,
)

pytest.importorskip("pandas")

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


def _triangle_evoked(sfreq=1000.0):
    """Return an Evoked with triangular peaks of exactly known area and latency.

    Each triangle has its apex at 0.3 s and a base spanning 0.2-0.4 s, so the
    area is 0.5 * 0.2 * height, and the half-amplitude points are at 0.25 and
    0.35 s.
    """
    times = np.arange(-0.1, 0.6, 1 / sfreq)
    triangle = np.clip(1 - np.abs(times - 0.3) / 0.1, 0, None)
    data = np.array([triangle * 1e-6, triangle * 2e-6, triangle * -1e-6])
    info = create_info(["ch0", "ch1", "ch2"], sfreq, "eeg")
    return EvokedArray(data, info, tmin=times[0])


def test_compute_peak():
    """Test peak amplitude and latency against a known triangular peak."""
    evoked = _triangle_evoked()
    peaks = compute_peak(evoked, start=0.2, stop=0.4, mode="pos")
    assert list(peaks["channel"]) == ["ch0", "ch1", "ch2"]
    assert_allclose(peaks["latency"][:2], [0.3, 0.3], atol=2e-3)
    assert_allclose(peaks["amplitude"][:2], [1e-6, 2e-6], rtol=1e-3)


def test_compute_area():
    """Test area computation against a triangle of known area."""
    evoked = _triangle_evoked()
    areas = compute_area(evoked, start=0.2, stop=0.4, mode="pos")
    assert_allclose(areas["area"][:2], [1e-7, 2e-7], rtol=1e-3)

    signed = compute_area(evoked, start=0.2, stop=0.4, mode="intg")
    assert_allclose(signed["area"][2], -1e-7, rtol=1e-3)


def test_compute_frac_peak_latency():
    """Test fractional peak latency against known half-amplitude crossings."""
    evoked = _triangle_evoked()
    latencies = compute_frac_peak_latency(
        evoked, frac=0.5, start=0.2, stop=0.4, mode="pos"
    )
    assert_allclose(latencies["fractional_peak_onset"][0], 0.25, atol=2e-3)
    assert_allclose(latencies["fractional_peak_offset"][0], 0.35, atol=2e-3)


def test_compute_frac_area_latency():
    """Test fractional area latency on a symmetric triangle."""
    evoked = _triangle_evoked()
    latencies = compute_frac_area_latency(
        evoked, frac=0.5, start=0.2, stop=0.4, mode="pos", picks=["ch0"]
    )
    assert_allclose(latencies["fractional_area_latency"][0], 0.3, atol=2e-3)

    # ch2 is negative-going, so mode="pos" accumulates no area for it and the
    # result is NaN with a warning
    with pytest.warns(RuntimeWarning, match="No area was accumulated"):
        nan_latency = compute_frac_area_latency(
            evoked, frac=0.5, start=0.2, stop=0.4, mode="pos", picks=["ch2"]
        )
    assert np.isnan(nan_latency["fractional_area_latency"][0])
