# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (
    _validate_nirs_info,
    beer_lambert_law,
    motion_correct_spline,
    optical_density,
    spline,
)

fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_spline_reduces_step_od(fname):
    """Test spline correction reduces a step artefact in OD data."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]

    # Save clean signal for comparison
    original = raw_od._data[0].copy()

    # Inject a step artefact in the first 30 samples of channel 0
    max_shift = np.max(np.abs(np.diff(raw_od._data[0])))
    shift_amp = 20 * max_shift
    raw_od._data[0, 0:30] -= shift_amp

    # Build per-channel motion mask: first 30 samples are artefact
    tIncCh = np.ones((len(picks), n_times), dtype=bool)
    tIncCh[:, 0:30] = False

    raw_od_corr = motion_correct_spline(raw_od, p=0.01, tIncCh=tIncCh)

    # Corrected signal should be closer to original than corrupted signal
    mse_before = np.mean((raw_od._data[0] - original) ** 2)
    mse_after = np.mean((raw_od_corr._data[0] - original) ** 2)
    assert mse_after < mse_before


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_spline_reduces_step_hb(fname):
    """Test spline correction works on haemoglobin concentration data."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    raw_hb = beer_lambert_law(raw_od)
    picks = _validate_nirs_info(raw_hb.info)
    n_times = raw_hb._data.shape[1]

    max_shift = np.max(np.diff(raw_hb._data[0]))
    shift_amp = 5 * max_shift
    raw_hb._data[0, 0:30] -= shift_amp

    tIncCh = np.ones((len(picks), n_times), dtype=bool)
    tIncCh[:, 0:30] = False

    raw_hb_corr = motion_correct_spline(raw_hb, p=0.01, tIncCh=tIncCh)
    assert np.max(np.diff(raw_hb_corr._data[0])) < shift_amp


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_spline_constant_channels(fname):
    """Test spline correction does not crash on constant channels."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]

    raw_od._data[picks[0]] = 0.0
    raw_od._data[picks[1]] = 1.0

    tIncCh = np.ones((len(picks), n_times), dtype=bool)
    tIncCh[:, 10:20] = False

    raw_od_corr = motion_correct_spline(raw_od, p=0.01, tIncCh=tIncCh)

    assert_allclose(raw_od_corr._data[picks[0]], 0.0)
    assert_allclose(raw_od_corr._data[picks[1]], 1.0)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_spline_returns_copy(fname):
    """Test spline correction does not modify the input Raw in place."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]
    original = raw_od._data[picks[0]].copy()

    tIncCh = np.ones((len(picks), n_times), dtype=bool)
    tIncCh[0, 10:30] = False

    _ = motion_correct_spline(raw_od, p=0.01, tIncCh=tIncCh)
    assert_allclose(raw_od._data[picks[0]], original)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_spline_no_artifacts(fname):
    """Test with tIncCh=None the function runs without raising."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)

    raw_od_corr = motion_correct_spline(raw_od, p=0.01, tIncCh=None)
    assert raw_od_corr._data.shape == raw_od._data.shape


def test_spline_alias():
    """Test spline is an alias for motion_correct_spline."""
    assert spline is motion_correct_spline


def test_motion_correct_spline_wrong_type():
    """Test passing a non-Raw object raises TypeError."""
    with pytest.raises(TypeError):
        motion_correct_spline(np.zeros((10, 100)), p=0.01)
