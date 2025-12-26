# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import BaseRaw, read_raw_fif, read_raw_nirx
from mne.preprocessing.nirs import beer_lambert_law, optical_density
from mne.utils import _validate_type

testing_path = data_path(download=False)
fname_nirx_15_0 = testing_path / "NIRx" / "nirscout" / "nirx_15_0_recording"
fname_nirx_15_2 = testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording"
fname_nirx_15_2_short = (
    testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname", ([fname_nirx_15_2_short, fname_nirx_15_2, fname_nirx_15_0])
)
@pytest.mark.parametrize("fmt", ("nirx", "fif"))
def test_beer_lambert(fname, fmt, tmp_path):
    """Test converting NIRX files."""
    assert fmt in ("nirx", "fif")
    raw = read_raw_nirx(fname)
    if fmt == "fif":
        raw.save(tmp_path / "test_raw.fif")
        raw = read_raw_fif(tmp_path / "test_raw.fif")
    assert "fnirs_cw_amplitude" in raw
    assert "fnirs_od" not in raw
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, "raw")
    assert "fnirs_cw_amplitude" not in raw
    assert "fnirs_od" in raw
    assert "hbo" not in raw
    raw = beer_lambert_law(raw)
    _validate_type(raw, BaseRaw, "raw")
    assert "fnirs_cw_amplitude" not in raw
    assert "fnirs_od" not in raw
    assert "hbo" in raw
    assert "hbr" in raw


@testing.requires_testing_data
def test_beer_lambert_unordered_errors():
    """NIRS data requires specific ordering and naming of channels."""
    raw = read_raw_nirx(fname_nirx_15_0)
    raw_od = optical_density(raw)
    raw_od.pick([0, 1, 2])
    with pytest.raises(ValueError, match="NIRS channels not ordered correctly."):
        beer_lambert_law(raw_od)

    # Test that an error is thrown if channel naming frequency doesn't match
    # what is stored in loc[9], which should hold the light frequency too.
    # Introduce 2 new frequencies to make it 4 in total vs 2 stored in loc[9].
    # This way the bad data will have 20 channels and 4 wavelengths, so as not
    # to get caught by the check for divisibility (channel % wavelength == 0).
    raw_od = optical_density(raw)
    assert raw.ch_names[0] == "S1_D1 760" and raw.ch_names[1] == "S1_D1 850"
    assert (
        raw_od.ch_names.index(raw.ch_names[0]) == 0
        and raw_od.ch_names.index(raw.ch_names[1]) == 1
    )
    raw_od.rename_channels(
        {
            raw.ch_names[0]: raw.ch_names[0].replace("760", "770"),
            raw.ch_names[1]: raw.ch_names[1].replace("850", "840"),
        }
    )
    assert raw_od.ch_names[0] == "S1_D1 770" and raw_od.ch_names[1] == "S1_D1 840"
    with pytest.raises(ValueError, match="NIRS channels not ordered correctly."):
        beer_lambert_law(raw_od)


@testing.requires_testing_data
def test_beer_lambert_v_matlab():
    """Compare MNE results to MATLAB toolbox."""
    pymatreader = pytest.importorskip("pymatreader")
    raw = read_raw_nirx(fname_nirx_15_0)
    raw = optical_density(raw)
    raw = beer_lambert_law(raw, ppf=(0.121, 0.121))
    raw._data *= 1e6  # Scale to uM for comparison to MATLAB

    matlab_fname = (
        testing_path / "NIRx" / "nirscout" / "validation" / "nirx_15_0_recording_bl.mat"
    )
    matlab_data = pymatreader.read_mat(matlab_fname)

    for idx in range(raw.get_data().shape[0]):
        mean_error = np.mean(matlab_data["data"][:, idx] - raw._data[idx])
        assert mean_error < 0.1
        matlab_name = (
            "S"
            + str(int(matlab_data["sources"][idx]))
            + "_D"
            + str(int(matlab_data["detectors"][idx]))
            + " "
            + matlab_data["type"][idx]
        )
        assert raw.info["ch_names"][idx] == matlab_name


@pytest.mark.parametrize("multi_wavelength_raw", [2], indirect=True)
def test_beer_lambert_multi_wavelength(multi_wavelength_raw):
    """Ensure Beer-Lambert can process >=3 wavelengths and reduces to 2 channels."""
    # Verify original CW data
    raw = multi_wavelength_raw.copy()
    assert len(raw.ch_names) == 2 * 3
    assert raw.ch_names[0] == "S1_D1 700"
    assert raw.ch_names[5] == "S2_D2 850"
    assert set(raw.get_channel_types()) == {"fnirs_cw_amplitude"}

    # Convert to OD (tested elsewhere)
    raw = optical_density(raw)

    # Verify data after conversion to Hb; channel numbers reduced to 2 per pair
    raw = beer_lambert_law(raw)
    _validate_type(raw, BaseRaw, "raw")
    assert len(raw.ch_names) == 2 * 2
    assert all(name.endswith(" hbo") or name.endswith(" hbr") for name in raw.ch_names)
    assert raw.ch_names[0] == "S1_D1 hbo"
    assert raw.ch_names[3] == "S2_D2 hbr"
    assert set(raw.get_channel_types()) == {"hbo", "hbr"}
