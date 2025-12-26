# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest as pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import BaseRaw, read_raw_nirx
from mne.preprocessing.nirs import optical_density
from mne.utils import _validate_type

fname_nirx = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)


@testing.requires_testing_data
def test_optical_density():
    """Test return type for optical density."""
    raw = read_raw_nirx(fname_nirx, preload=False)
    assert "fnirs_cw_amplitude" in raw
    assert "fnirs_od" not in raw
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, "raw")
    assert "fnirs_cw_amplitude" not in raw
    assert "fnirs_od" in raw
    with pytest.raises(RuntimeError, match="on continuous wave"):
        optical_density(raw)


@pytest.mark.parametrize("multi_wavelength_raw", [2], indirect=True)
def test_optical_density_multi_wavelength(multi_wavelength_raw):
    """Ensure OD can process >=3 wavelengths and preserves channels."""
    # Validate original CW data
    raw = multi_wavelength_raw.copy()
    assert len(raw.ch_names) == 2 * 3
    assert raw.ch_names[0] == "S1_D1 700"
    assert raw.ch_names[5] == "S2_D2 850"
    assert set(raw.get_channel_types()) == {"fnirs_cw_amplitude"}

    # Validate that data has been converted to OD, number of channels preserved
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, "raw")
    assert len(raw.ch_names) == 2 * 3
    assert raw.ch_names[0] == "S1_D1 700"
    assert raw.ch_names[5] == "S2_D2 850"
    assert set(raw.get_channel_types()) == {"fnirs_od"}


@testing.requires_testing_data
def test_optical_density_zeromean():
    """Test that optical density can process zero mean data."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    raw._data[4] -= np.mean(raw._data[4])
    raw._data[4, -1] = 0
    with np.errstate(invalid="raise", divide="raise"):
        with pytest.warns(RuntimeWarning, match="Negative"):
            raw = optical_density(raw)
    assert "fnirs_od" in raw


@testing.requires_testing_data
def test_optical_density_manual():
    """Test optical density on known values."""
    test_tol = 0.01
    raw = read_raw_nirx(fname_nirx, preload=True)
    # log(1) = 0
    raw._data[4] = np.ones(145)
    # log(0.5)/-1 = 0.69
    # log(1.5)/-1 = -0.40
    test_data = np.tile([0.5, 1.5], 73)[:145]
    raw._data[5] = test_data

    od = optical_density(raw)
    assert_allclose(od.get_data([4]), 0.0)
    assert_allclose(od.get_data([5])[0, :2], [0.69, -0.4], atol=test_tol)
