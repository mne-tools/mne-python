# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import create_info
from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import BaseRaw, RawArray, read_raw_fif, read_raw_nirx, read_raw_snirf
from mne.preprocessing.nirs import (
    _channel_frequencies,
    beer_lambert_law,
    optical_density,
    source_detector_distances,
)
from mne.preprocessing.nirs._beer_lambert_law import _get_sd_distances
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


def test_beer_lambert_sd_distances():
    """Test Beer-Lambert conversion with explicit source-detector distances."""
    data = np.array(
        [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.4, 0.5, 0.6], [0.45, 0.55, 0.65]]
    )
    # Ch names chosen to test reordered indices
    ch_names = ["S1_D1 760", "S1_D1 850", "S10_D10 760", "S10_D10 850"]

    # Case 1: valid locations, sd_distances=None
    raw = RawArray(data, create_info(ch_names, sfreq=1.0, ch_types="fnirs_od"))
    sd_distances = [0.03, 0.03, 0.03, 0.03]
    for idx, (freq, distance) in enumerate(zip([760, 850, 760, 850], sd_distances)):
        raw.info["chs"][idx]["loc"][3:6] = [0.0, 0.0, 0.0]
        raw.info["chs"][idx]["loc"][6:9] = [distance, 0.0, 0.0]
        raw.info["chs"][idx]["loc"][9] = freq
    expected = beer_lambert_law(raw)

    # Case 2: valid locations, sd_distances=<arr>
    with pytest.warns(RuntimeWarning, match=r"(?i)will be overridden"):
        actual = beer_lambert_law(raw, sd_distances=sd_distances)
    assert actual.ch_names == expected.ch_names
    assert_allclose(actual.get_data(), expected.get_data(), rtol=1e-12, atol=0)

    # Case 3: no locations, sd_distances=None
    for idx in range(len(raw.info["chs"])):
        raw.info["chs"][idx]["loc"][3:9] = np.nan
    assert np.isnan(source_detector_distances(raw.info)).all()
    with pytest.raises(
        ValueError, match=r"(?i)source-detector distances are all zero or NaN"
    ):
        beer_lambert_law(raw)

    # Case 4: no locations, sd_distances=<arr>
    actual = beer_lambert_law(raw, sd_distances=sd_distances)
    assert actual.ch_names == expected.ch_names
    assert_allclose(actual.get_data(), expected.get_data(), rtol=1e-12, atol=0)

    # Case 5: no locations, sd_distances=<scalar>
    actual = beer_lambert_law(raw, sd_distances=sd_distances[0])
    assert actual.ch_names == expected.ch_names
    assert_allclose(actual.get_data(), expected.get_data(), rtol=1e-12, atol=0)


def test_get_sd_distances():
    """Test source-detector distance selection and validation."""
    raw = RawArray(
        np.zeros((4, 3)),
        create_info(
            ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"], 1.0, "fnirs_od"
        ),
    )
    expected = np.array([0.03, 0.03, 0.04, 0.04])
    for idx, (freq, distance) in enumerate(zip([760, 850, 760, 850], expected)):
        raw.info["chs"][idx]["loc"][3:6] = [0.0, 0.0, 0.0]
        raw.info["chs"][idx]["loc"][6:9] = [distance, 0.0, 0.0]
        raw.info["chs"][idx]["loc"][9] = freq

    assert_allclose(_get_sd_distances(raw, None), expected, rtol=1e-12, atol=0)
    with pytest.warns(RuntimeWarning, match=r"(?i)will be overridden"):
        assert_allclose(_get_sd_distances(raw, expected), expected, rtol=1e-12, atol=0)
    with pytest.warns(RuntimeWarning, match=r"(?i)will be overridden"):
        assert_allclose(
            _get_sd_distances(raw, 0.05), np.full(4, 0.05), rtol=1e-12, atol=0
        )

    for idx in range(len(raw.info["chs"])):
        raw.info["chs"][idx]["loc"][3:9] = np.nan
    assert_allclose(_get_sd_distances(raw, expected), expected, rtol=1e-12, atol=0)

    with pytest.raises(ValueError, match=r"1D array-like"):
        _get_sd_distances(raw, np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"length matching"):
        _get_sd_distances(raw, [0.03, 0.03])
    with pytest.raises(TypeError, match=r"sd_distances"):
        _get_sd_distances(raw, "foo")
