# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import BaseRaw, read_raw_fif, read_raw_nirx, read_raw_snirf
from mne.preprocessing.nirs import (
    _channel_frequencies,
    beer_lambert_law,
    optical_density,
)
from mne.utils import _validate_type

testing_path = data_path(download=False)
fname_nirx_15_0 = testing_path / "NIRx" / "nirscout" / "nirx_15_0_recording"
fname_nirx_15_2 = testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording"
fname_nirx_15_2_short = (
    testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)
fname_labnirs_multi_wavelength = (
    testing_path / "SNIRF" / "Labnirs" / "labnirs_3wl_raw_recording.snirf"
)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,fmt",
    (
        [
            (fname_nirx_15_2_short, "nirx"),
            (fname_nirx_15_2_short, "fif"),
            (fname_nirx_15_2, "nirx"),
            (fname_nirx_15_2, "fif"),
            (fname_nirx_15_0, "nirx"),
            (fname_nirx_15_0, "fif"),
            (fname_labnirs_multi_wavelength, "snirf"),
        ]
    ),
)
def test_beer_lambert(fname, fmt, tmp_path):
    """Test converting raw CW amplitude files."""
    if fname.suffix == ".snirf":
        pytest.importorskip("h5py")
    match fmt:
        case "nirx":
            raw_volt = read_raw_nirx(fname)
        case "fif":
            raw_nirx = read_raw_nirx(fname)
            raw_nirx.save(tmp_path / "test_raw.fif")
            raw_volt = read_raw_fif(tmp_path / "test_raw.fif")
        case "snirf":
            raw_volt = read_raw_snirf(fname)
        case _:
            raise ValueError(
                f"fmt expected to be one of 'nirx', 'fif' or 'snirf', got {fmt}"
            )

    raw_od = optical_density(raw_volt)
    _validate_type(raw_od, BaseRaw, "raw")

    raw_hb = beer_lambert_law(raw_od)
    _validate_type(raw_hb, BaseRaw, "raw")

    # Verify channel numbers (multi-wavelength aware)
    # Raw voltage has: optode pairs * number of wavelengths
    # OD must have the same number as raw voltage
    # Hb data must have: number of optode pairs * 2
    nfreqs = len(set(_channel_frequencies(raw_volt.info)))
    assert len(raw_volt.ch_names) % nfreqs == 0
    npairs = len(raw_volt.ch_names) // nfreqs
    assert len(raw_hb.ch_names) % npairs == 0
    assert len(raw_hb.ch_names) // npairs == 2.0

    # Verify data types
    assert set(raw_volt.get_channel_types()) == {"fnirs_cw_amplitude"}
    assert set(raw_hb.get_channel_types()) == {"hbo", "hbr"}

    # Verify that pair ordering did not change just channel name suffixes
    old_prefixes = [name.split(" ")[0] for name in raw_volt.ch_names[::nfreqs]]
    new_prefixes = [name.split(" ")[0] for name in raw_hb.ch_names[::2]]
    assert old_prefixes == new_prefixes
    assert all([name.split(" ")[1] in {"hbo", "hbr"} for name in raw_hb.ch_names])


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
