# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import shutil
from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

from mne._fiff.constants import FIFF
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nirx, read_raw_snirf
from mne.io.tests.test_raw import _test_raw_reader
from mne.preprocessing.nirs import (
    _channel_frequencies,
    _reorder_nirx,
    beer_lambert_law,
    optical_density,
    short_channels,
    source_detector_distances,
)
from mne.transforms import _get_trans, apply_trans
from mne.utils import _chmod_rw_R, catch_logging

testing_path = data_path(download=False)
# SfNIRS files
sfnirs_homer_103_wShort = (
    testing_path
    / "SNIRF"
    / "SfNIRS"
    / "snirf_homer3"
    / "1.0.3"
    / "snirf_1_3_nirx_15_2_recording_w_short.snirf"
)
sfnirs_homer_103_wShort_original = (
    testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)
sfnirs_homer_103_153 = (
    testing_path
    / "SNIRF"
    / "SfNIRS"
    / "snirf_homer3"
    / "1.0.3"
    / "nirx_15_3_recording.snirf"
)

# NIRSport2 files
nirx_nirsport2_103 = (
    testing_path / "SNIRF" / "NIRx" / "NIRSport2" / "1.0.3" / "2021-04-23_005.snirf"
)
nirx_nirsport2_103_2 = (
    testing_path / "SNIRF" / "NIRx" / "NIRSport2" / "1.0.3" / "2021-05-05_001.snirf"
)
snirf_nirsport2_20219 = (
    testing_path / "SNIRF" / "NIRx" / "NIRSport2" / "2021.9" / "2021-10-01_002.snirf"
)

# Kernel
kernel_flow2_path = testing_path / "SNIRF" / "Kernel" / "Flow2" / "Portal_2024_10_23"
kernel_td_gated = kernel_flow2_path / "c345d04_2.snirf"  # Type 201 (TD Gated, 201)
kernel_td_moments = kernel_flow2_path / "c345d04_3.snirf"  # Type 202 (TD Moments, 301)
kernel_hb = kernel_flow2_path / "c345d04_5.snirf"  # Type 203 (Hb, 99999)


h5py = pytest.importorskip("h5py")  # module-level

# Fieldtrip
ft_od = testing_path / "SNIRF" / "FieldTrip" / "220307_opticaldensity.snirf"

# GowerLabs
lumo110 = testing_path / "SNIRF" / "GowerLabs" / "lumomat-1-1-0.snirf"

# Shimadzu Labnirs 3-wavelength converted to snirf using custom tool
labnirs_multi_wavelength = (
    testing_path / "SNIRF" / "Labnirs" / "labnirs_3wl_raw_recording.snirf"
)


def _get_loc(raw, ch_name):
    return raw.copy().pick(ch_name).info["chs"][0]["loc"]


@requires_testing_data
@pytest.mark.filterwarnings("ignore:.*contains 2D location.*:")
@pytest.mark.filterwarnings("ignore:.*measurement date.*:")
@pytest.mark.parametrize(
    "fname",
    (
        [
            sfnirs_homer_103_wShort,
            nirx_nirsport2_103,
            sfnirs_homer_103_153,
            nirx_nirsport2_103,
            nirx_nirsport2_103_2,
            nirx_nirsport2_103_2,
            pytest.param(kernel_td_gated, id=f"kernel: {kernel_td_gated.stem}"),
            pytest.param(kernel_td_moments, id=f"kernel: {kernel_td_moments.stem}"),
            pytest.param(kernel_hb, id=f"kernel: {kernel_hb.stem}"),
            lumo110,
            labnirs_multi_wavelength,
        ]
    ),
)
def test_basic_reading_and_min_process(fname):
    """Test reading SNIRF files and minimum typical processing."""
    raw = read_raw_snirf(fname, preload=True)
    # SNIRF data can contain several types, so only apply appropriate functions
    kinds = [
        "fnirs_cw_amplitude",
        "fnirs_od",
        "fnirs_td_gated_amplitude",
        "fnirs_td_moments_intensity",
        "hbo",
        # TODO: add fd_*
    ]
    ch_types = raw.get_channel_types(unique=True)
    got_kinds = [kind for kind in kinds if kind in raw]
    assert len(got_kinds) == 1, f"Need one data type, {got_kinds=} and {ch_types=}"
    if "fnirs_cw_amplitude" in raw:
        raw = optical_density(raw)
    elif "fnirs_od" in raw:
        raw = beer_lambert_law(raw, ppf=6)
    elif "fnirs_td_gated_amplitude" in raw:
        pass
    elif "fnirs_td_moments_intensity" in raw:
        assert "fnirs_td_moments_mean" in raw
        assert "fnirs_td_moments_variance" in raw
    else:
        assert "hbo" in raw
        assert "hbr" in raw


@requires_testing_data
@pytest.mark.filterwarnings("ignore:.*measurement date.*:")
def test_snirf_gowerlabs():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(lumo110, preload=True)

    assert raw._data.shape == (216, 274)
    assert raw.info["dig"][0]["coord_frame"] == FIFF.FIFFV_COORD_HEAD
    assert len(raw.ch_names) == 216
    assert_allclose(raw.info["sfreq"], 10.0)
    # we don't force them to be sorted according to a naive split
    assert raw.ch_names != sorted(raw.ch_names)
    # ... but this file does have a nice logical ordering already
    print(raw.ch_names)
    assert raw.ch_names == sorted(
        raw.ch_names,
        # use a key which is (src triplet, freq, src, freq, det)
        key=lambda name: (
            (int(name.split()[0].split("_")[0][1:]) - 1) // 3,
            int(name.split()[1]),
            int(name.split()[0].split("_")[0][1:]),
            int(name.split()[0].split("_")[1][1:]),
        ),
    )


@requires_testing_data
def test_snirf_basic():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(sfnirs_homer_103_wShort, preload=True)
    assert raw.info["subject_info"]["his_id"] == "default"

    # Test data import
    assert raw._data.shape == (26, 145)
    assert raw.info["sfreq"] == 12.5

    # Test channel naming
    assert raw.info["ch_names"][:4] == [
        "S1_D1 760",
        "S1_D9 760",
        "S2_D3 760",
        "S2_D10 760",
    ]
    assert raw.info["ch_names"][24:26] == ["S5_D8 850", "S5_D13 850"]

    # Test frequency encoding
    assert raw.info["chs"][0]["loc"][9] == 760
    assert raw.info["chs"][24]["loc"][9] == 850

    # Test source locations
    assert_allclose(
        [-8.6765 * 1e-2, 0.0049 * 1e-2, -2.6167 * 1e-2],
        _get_loc(raw, "S1_D1 760")[3:6],
        rtol=0.02,
    )
    assert_allclose(
        [7.9579 * 1e-2, -2.7571 * 1e-2, -2.2631 * 1e-2],
        _get_loc(raw, "S2_D3 760")[3:6],
        rtol=0.02,
    )
    assert_allclose(
        [-2.1387 * 1e-2, -8.8874 * 1e-2, 3.8393 * 1e-2],
        _get_loc(raw, "S3_D2 760")[3:6],
        rtol=0.02,
    )
    assert_allclose(
        [1.8602 * 1e-2, 9.7164 * 1e-2, 1.7539 * 1e-2],
        _get_loc(raw, "S4_D4 760")[3:6],
        rtol=0.02,
    )
    assert_allclose(
        [-0.1108 * 1e-2, 0.7066 * 1e-2, 8.9883 * 1e-2],
        _get_loc(raw, "S5_D5 760")[3:6],
        rtol=0.02,
    )

    # Test detector locations
    assert_allclose(
        [-8.0409 * 1e-2, -2.9677 * 1e-2, -2.5415 * 1e-2],
        _get_loc(raw, "S1_D1 760")[6:9],
        rtol=0.02,
    )
    assert_allclose(
        [-8.7329 * 1e-2, 0.7577 * 1e-2, -2.7980 * 1e-2],
        _get_loc(raw, "S1_D9 850")[6:9],
        rtol=0.02,
    )
    assert_allclose(
        [9.2027 * 1e-2, 0.0161 * 1e-2, -2.8909 * 1e-2],
        _get_loc(raw, "S2_D3 850")[6:9],
        rtol=0.02,
    )
    assert_allclose(
        [7.7548 * 1e-2, -3.5901 * 1e-2, -2.3179 * 1e-2],
        _get_loc(raw, "S2_D10 850")[6:9],
        rtol=0.02,
    )

    assert "fnirs_cw_amplitude" in raw


@requires_testing_data
def test_snirf_against_nirx():
    """Test Homer generated against file snirf was created from."""
    raw_homer = read_raw_snirf(sfnirs_homer_103_wShort, preload=True)
    _reorder_nirx(raw_homer)
    raw_orig = read_raw_nirx(sfnirs_homer_103_wShort_original, preload=True)

    # Check annotations are the same
    assert_allclose(raw_homer.annotations.onset, raw_orig.annotations.onset)
    assert_allclose(
        [float(d) for d in raw_homer.annotations.description],
        [float(d) for d in raw_orig.annotations.description],
    )
    # Homer writes durations as 5s regardless of the true duration.
    # So we will not test that the nirx file stim durations equal
    # the homer file stim durations.

    # Check names are the same
    assert raw_homer.info["ch_names"] == raw_orig.info["ch_names"]

    # Check frequencies are the same
    num_chans = len(raw_homer.ch_names)
    new_chs = raw_homer.info["chs"]
    ori_chs = raw_orig.info["chs"]
    assert_allclose(
        [new_chs[idx]["loc"][9] for idx in range(num_chans)],
        [ori_chs[idx]["loc"][9] for idx in range(num_chans)],
    )

    # Check data is the same
    assert_allclose(raw_homer.get_data(), raw_orig.get_data())


@requires_testing_data
def test_snirf_nonstandard(tmp_path):
    """Test custom tags."""
    fname = str(tmp_path) + "/mod.snirf"
    shutil.copy(sfnirs_homer_103_wShort, fname)
    _chmod_rw_R(tmp_path)
    # Manually mark up the file to match MNE-NIRS custom tags
    with h5py.File(fname, "r+") as f:
        f.create_dataset("nirs/metaDataTags/middleName", data=[b"X"])
        f.create_dataset("nirs/metaDataTags/lastName", data=[b"Y"])
        f.create_dataset("nirs/metaDataTags/sex", data=[b"1"])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["first_name"] == "default"  # pull from his_id
    with h5py.File(fname, "r+") as f:
        f.create_dataset("nirs/metaDataTags/firstName", data=[b"W"])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["first_name"] == "W"
    assert raw.info["subject_info"]["middle_name"] == "X"
    assert raw.info["subject_info"]["last_name"] == "Y"
    assert raw.info["subject_info"]["sex"] == 1
    assert raw.info["subject_info"]["his_id"] == "default"
    with h5py.File(fname, "r+") as f:
        del f["nirs/metaDataTags/sex"]
        f.create_dataset("nirs/metaDataTags/sex", data=[b"2"])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["sex"] == 2
    with h5py.File(fname, "r+") as f:
        del f["nirs/metaDataTags/sex"]
        f.create_dataset("nirs/metaDataTags/sex", data=[b"0"])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["sex"] == 0

    with h5py.File(fname, "r+") as f:
        f.create_dataset("nirs/metaDataTags/MNE_coordFrame", data=[1])


@requires_testing_data
def test_snirf_empty_landmark_labels(tmp_path):
    """Test reading SNIRF files with empty landmarkLabels (gh-13627)."""
    fname = tmp_path / "empty_labels.snirf"
    shutil.copy(sfnirs_homer_103_wShort, fname)
    _chmod_rw_R(tmp_path)

    # Modify file to have landmarkPos3D but empty/scalar landmarkLabels
    with h5py.File(fname, "r+") as f:
        # Remove existing landmark data if present
        if "landmarkPos3D" in f["nirs/probe"]:
            del f["nirs/probe/landmarkPos3D"]
        if "landmarkLabels" in f["nirs/probe"]:
            del f["nirs/probe/landmarkLabels"]

        # Add non-empty landmarkPos3D
        f.create_dataset(
            "nirs/probe/landmarkPos3D",
            data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        # Add empty scalar landmarkLabels (this triggers the bug in gh-13627)
        f.create_dataset("nirs/probe/landmarkLabels", data=b"")

    # This should not raise "TypeError: iteration over a 0-d array"
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["dig"] is not None


@requires_testing_data
def test_snirf_nirsport2():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(nirx_nirsport2_103, preload=True)

    # Test data import
    assert raw._data.shape == (92, 84)
    assert_almost_equal(raw.info["sfreq"], 7.6, decimal=1)

    # Test channel naming
    assert raw.info["ch_names"][:4] == [
        "S1_D1 760",
        "S1_D3 760",
        "S1_D9 760",
        "S1_D16 760",
    ]
    assert raw.info["ch_names"][24:26] == ["S8_D15 760", "S8_D20 760"]

    # Test frequency encoding
    assert raw.info["chs"][0]["loc"][9] == 760
    assert raw.info["chs"][-1]["loc"][9] == 850

    assert sum(short_channels(raw.info)) == 16


@requires_testing_data
def test_snirf_coordframe():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="head").info["chs"][3][
        "coord_frame"
    ]
    assert raw == FIFF.FIFFV_COORD_HEAD

    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="mri").info["chs"][3][
        "coord_frame"
    ]
    assert raw == FIFF.FIFFV_COORD_HEAD

    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="unknown").info["chs"][3][
        "coord_frame"
    ]
    assert raw == FIFF.FIFFV_COORD_UNKNOWN


@requires_testing_data
def test_snirf_nirsport2_w_positions():
    """Test reading SNIRF files with known positions."""
    raw = read_raw_snirf(nirx_nirsport2_103_2, preload=True, optode_frame="mri")
    _reorder_nirx(raw)

    # Test data import
    assert raw._data.shape == (40, 128)
    assert_almost_equal(raw.info["sfreq"], 10.2, decimal=1)

    # Test channel naming
    assert raw.info["ch_names"][:4] == [
        "S1_D1 760",
        "S1_D1 850",
        "S1_D6 760",
        "S1_D6 850",
    ]
    assert raw.info["ch_names"][24:26] == ["S6_D4 760", "S6_D4 850"]

    # Test frequency encoding
    assert raw.info["chs"][0]["loc"][9] == 760
    assert raw.info["chs"][1]["loc"][9] == 850

    assert sum(short_channels(raw.info)) == 16

    # Test distance between optodes matches values from
    # nirsite https://github.com/mne-tools/mne-testing-data/pull/86
    # figure 3
    allowed_distance_error = 0.005
    assert_allclose(
        source_detector_distances(raw.copy().pick("S1_D1 760").info),
        [0.0304],
        atol=allowed_distance_error,
    )
    assert_allclose(
        source_detector_distances(raw.copy().pick("S2_D2 760").info),
        [0.0400],
        atol=allowed_distance_error,
    )

    # Test location of detectors
    # The locations of detectors can be seen in the first
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/86
    allowed_dist_error = 0.0002
    locs = [ch["loc"][6:9] for ch in raw.info["chs"]]
    head_mri_t, _ = _get_trans("fsaverage", "head", "mri")
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info["ch_names"][0][3:5] == "D1"
    assert_allclose(mni_locs[0], [-0.0841, -0.0464, -0.0129], atol=allowed_dist_error)

    assert raw.info["ch_names"][2][3:5] == "D6"
    assert_allclose(mni_locs[2], [-0.0841, -0.0138, 0.0248], atol=allowed_dist_error)

    assert raw.info["ch_names"][34][3:5] == "D5"
    assert_allclose(mni_locs[34], [0.0845, -0.0451, -0.0123], atol=allowed_dist_error)

    # Test location of sensors
    # The locations of sensors can be seen in the second
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/86
    allowed_dist_error = 0.0002
    locs = [ch["loc"][3:6] for ch in raw.info["chs"]]
    head_mri_t, _ = _get_trans("fsaverage", "head", "mri")
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info["ch_names"][0][:2] == "S1"
    assert_allclose(mni_locs[0], [-0.0848, -0.0162, -0.0163], atol=allowed_dist_error)

    assert raw.info["ch_names"][9][:2] == "S2"
    assert_allclose(mni_locs[9], [-0.0, -0.1195, 0.0142], atol=allowed_dist_error)

    assert raw.info["ch_names"][34][:2] == "S8"
    assert_allclose(mni_locs[34], [0.0828, -0.046, 0.0285], atol=allowed_dist_error)

    mon = raw.get_montage()
    assert len(mon.dig) == 27


@requires_testing_data
def test_snirf_fieldtrip_od():
    """Test reading FieldTrip SNIRF files with optical density data."""
    raw = read_raw_snirf(ft_od, preload=True)

    # Test data import
    assert raw._data.shape == (72, 500)
    assert raw.copy().pick("fnirs")._data.shape == (72, 500)
    assert raw.copy().pick("fnirs_od")._data.shape == (72, 500)
    with pytest.raises(ValueError, match="not be interpreted as channel"):
        raw.copy().pick("hbo")
    with pytest.raises(ValueError, match="not be interpreted as channel"):
        raw.copy().pick("hbr")

    assert_allclose(raw.info["sfreq"], 50)


@requires_testing_data
@pytest.mark.parametrize(
    "kind, shape, fname",
    [
        pytest.param("hb", (4, 38), kernel_hb, id="hb"),
        pytest.param("td moments", (12, 38), kernel_td_moments, id="td moments"),
        pytest.param("td gated", (100, 38), kernel_td_gated, id="td gated"),
    ],
)
def test_snirf_kernel_basic(kind, shape, fname):
    """Test reading Kernel SNIRF files with haemoglobin or TD data."""
    raw = read_raw_snirf(fname, preload=True)
    if kind == "hb":
        # Test data import
        assert raw._data.shape == shape
        hbo_data = raw.get_data("hbo")
        hbr_data = raw.get_data("hbr")
        assert hbo_data.shape == hbr_data.shape == (shape[0] // 2, shape[1])
        hbo_norm = np.nanmedian(np.linalg.norm(hbo_data, axis=-1))
        hbr_norm = np.nanmedian(np.linalg.norm(hbr_data, axis=-1))
        assert 1e-5 < hbr_norm < hbo_norm < 1e-4
    elif kind == "td moments":
        assert raw._data.shape == shape
        n_ch = 0
        lims = dict(intensity=(1e4, 1e7), mean=(1e-9, 1e-8), variance=(1e-19, 1e-16))
        for key, val in lims.items():
            data = raw.get_data(f"fnirs_td_moments_{key}")
            assert data.shape[1] == len(raw.times)
            norm = np.nanmedian(np.linalg.norm(data, axis=-1))
            min_, max_ = val
            assert min_ < norm < max_, key
            n_ch += data.shape[0]
        assert raw._data.shape[0] == len(raw.ch_names) == n_ch
        mean_ch = raw.copy().pick("fnirs_td_moments_mean").info["chs"][0]
        assert mean_ch["unit"] == FIFF.FIFF_UNIT_SEC
        var_ch = raw.copy().pick("fnirs_td_moments_variance").info["chs"][0]
        assert var_ch["unit"] == FIFF.FIFF_UNIT_SEC2
    else:
        assert kind == "td gated"
        assert raw._data.shape == shape
        data = raw.get_data("fnirs_td_gated_amplitude")
        assert data.shape == shape
        assert np.max(np.abs(data)) > 1e4
        # Channel names should include wavelength and bin info
        assert all("bin" in ch for ch in raw.ch_names)
        # Check channel metadata
        ch = raw.info["chs"][0]
        assert ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_TD_GATED_AMPLITUDE
        assert ch["loc"][9] > 0  # wavelength
        assert ch["loc"][10] > 0  # time_delay * time_delay_width

    assert_allclose(raw.info["sfreq"], 3.759351, atol=1e-5)

    bad_nans = np.isnan(raw.get_data()).any(axis=1)
    assert np.sum(bad_nans) == 0

    assert len(raw.annotations.description) == 8
    assert raw.annotations.onset[0] == 4.988107
    assert raw.annotations.onset[1] == 5.988107
    assert raw.annotations.description[0] == "StartBlock"
    assert raw.annotations.description[1] == "StartTrial"


@requires_testing_data
@pytest.mark.parametrize(
    "sfreq,context",
    (
        [3.75, nullcontext()],  # sfreq estimated from file is 3.759351
        [22, pytest.warns(RuntimeWarning, match="User-supplied sampling frequency")],
    ),
)
def test_user_set_sfreq(sfreq, context):
    """Test manually setting sfreq."""
    with context:
        # both sfreqs are far enough from true rate to yield >1% jitter
        with pytest.warns(RuntimeWarning, match=r"jitter of \d+\.\d*% in sample times"):
            raw = read_raw_snirf(kernel_hb, preload=False, sfreq=sfreq)
    assert raw.info["sfreq"] == sfreq


@requires_testing_data
@pytest.mark.parametrize(
    "fname, boundary_decimal, test_scaling, test_rank",
    (
        [sfnirs_homer_103_wShort, 0, True, True],
        [nirx_nirsport2_103, 0, True, False],  # strange rank behavior
        [nirx_nirsport2_103_2, 0, False, True],  # weirdly small values
        [snirf_nirsport2_20219, 0, True, True],
    ),
)
def test_snirf_standard(fname, boundary_decimal, test_scaling, test_rank):
    """Test standard operations."""
    _test_raw_reader(
        read_raw_snirf,
        fname=fname,
        boundary_decimal=boundary_decimal,
        test_scaling=test_scaling,
        test_rank=test_rank,
    )  # low fs


@requires_testing_data
def test_annotation_description_from_stim_groups():
    """Test annotation descriptions parsed from stim group names."""
    raw = read_raw_snirf(nirx_nirsport2_103_2, preload=True)
    expected_descriptions = ["1", "2", "6"]
    assert_equal(expected_descriptions, raw.annotations.description)


@requires_testing_data
def test_annotation_duration_from_stim_groups():
    """Test annotation durations extracted correctly from stim group."""
    raw = read_raw_snirf(snirf_nirsport2_20219, preload=True)
    # Specify the expected SNIRF stim durations.
    # We can verify these values should be 10 by using the official
    # SNIRF package pysnirf2 and running the following script.
    # You will see that the print statement shows the middle column,
    # which represents duration, will be all 10s.
    # from snirf import Snirf
    # a = Snirf(snirf_nirsport2_20219, "r+"); print(a.nirs[0].stim[0].data)
    expected_durations = np.full((10,), 10.0)
    assert_equal(expected_durations, raw.annotations.duration)


def test_birthday(tmp_path, monkeypatch):
    """Test birthday parsing."""
    try:
        snirf = pytest.importorskip("snirf")
    except AttributeError as exc:
        # Until https://github.com/BUNPC/pysnirf2/pull/43 is released
        pytest.skip(f"snirf import error: {exc}")
    fname = tmp_path / "test.snirf"
    with snirf.Snirf(str(fname), "w") as a:
        a.nirs.appendGroup()
        a.nirs[0].data.appendGroup()
        a.nirs[0].data[0].dataTimeSeries = np.zeros((2, 2))
        a.nirs[0].data[0].time = [0, 1]
        for i in range(2):
            a.nirs[0].data[0].measurementList.appendGroup()
            a.nirs[0].data[0].measurementList[i].sourceIndex = 1
            a.nirs[0].data[0].measurementList[i].detectorIndex = 1
            a.nirs[0].data[0].measurementList[i].wavelengthIndex = 1
            a.nirs[0].data[0].measurementList[i].dataType = 99999
            a.nirs[0].data[0].measurementList[i].dataTypeIndex = 0
        a.nirs[0].data[0].measurementList[0].dataTypeLabel = "HbO"
        a.nirs[0].data[0].measurementList[1].dataTypeLabel = "HbR"
        a.nirs[0].metaDataTags.SubjectID = "0"
        a.nirs[0].metaDataTags.MeasurementDate = "2000-01-01"
        a.nirs[0].metaDataTags.MeasurementTime = "00:00:00"
        a.nirs[0].metaDataTags.LengthUnit = "m"
        a.nirs[0].metaDataTags.TimeUnit = "s"
        a.nirs[0].metaDataTags.FrequencyUnit = "Hz"
        a.nirs[0].metaDataTags.add("DateOfBirth", "1950-01-01")
        a.nirs[0].probe.wavelengths = [0, 0]
        a.nirs[0].probe.sourcePos3D = np.zeros((1, 3))
        a.nirs[0].probe.detectorPos3D = np.zeros((1, 3))
        # Until https://github.com/BUNPC/pysnirf2/pull/39 is released
        monkeypatch.setattr(a._cfg.logger, "info", lambda *args, **kwargs: None)
        a.save()

    raw = read_raw_snirf(fname)
    assert raw.info["subject_info"]["birthday"] == datetime.date(1950, 1, 1)
    # TODO: trigger some setting checkers that should maybe be in the reader (like
    # those for subject_info)
    raw.info.copy()


@requires_testing_data
def test_sample_rate_jitter(tmp_path):
    """Test handling of jittered sample times."""
    from shutil import copy2

    # Create a clean copy and ensure it loads without error
    new_file = tmp_path / "snirf_nirsport2_2019.snirf"
    copy2(snirf_nirsport2_20219, new_file)
    _chmod_rw_R(tmp_path)
    read_raw_snirf(new_file)

    # Edit the file and add jitter within tolerance (0.99%)
    with h5py.File(new_file, "r+") as f:
        orig_time = np.array(f.get("nirs/data1/time"))
        acceptable_time_jitter = orig_time.copy()
        mean_period = np.mean(np.diff(orig_time))
        acceptable_time_jitter[-1] += 0.0099 * mean_period
        del f["nirs/data1/time"]
        f.flush()
        f.create_dataset("nirs/data1/time", data=acceptable_time_jitter)
    with catch_logging("info") as log:
        read_raw_snirf(new_file)
    lines = "\n".join(line for line in log.getvalue().splitlines() if "jitter" in line)
    assert "Found jitter of 0.9" in lines

    # Add jitter of 1.02%, which is greater than allowed tolerance
    with h5py.File(new_file, "r+") as f:
        unacceptable_time_jitter = orig_time
        unacceptable_time_jitter[-1] = unacceptable_time_jitter[-1] + (
            0.0102 * mean_period
        )
        del f["nirs/data1/time"]
        f.flush()
        f.create_dataset("nirs/data1/time", data=unacceptable_time_jitter)
    with pytest.warns(RuntimeWarning, match="non-uniformly-sampled data"):
        read_raw_snirf(new_file, verbose=True)


def test_get_dataunit_scaling():
    """Test CMIXF-12 and legacy Hb unit scaling."""
    from mne.io.snirf._snirf import _get_dataunit_scaling

    # Legacy shorthand: [prefix]M
    assert _get_dataunit_scaling("M") == 1.0
    assert _get_dataunit_scaling("mM") == 1e-3
    assert _get_dataunit_scaling("uM") == 1e-6
    assert _get_dataunit_scaling("nM") == 1e-9
    # CMIXF-12: [prefix]mol / L
    assert _get_dataunit_scaling("mol/L") == 1.0
    assert _get_dataunit_scaling("mmol/L") == 1e-3
    assert _get_dataunit_scaling("umol/L") == 1e-6
    assert _get_dataunit_scaling("nmol/L") == 1e-9
    assert _get_dataunit_scaling("pmol/L") == 1e-12
    # CMIXF-12: [prefix]mol / dm^3  (dm^3 = L)
    assert _get_dataunit_scaling("mol/dm^3") == 1.0
    assert _get_dataunit_scaling("mmol/dm^3") == 1e-3
    assert _get_dataunit_scaling("umol/dm^3") == 1e-6
    # CMIXF-12: [prefix]mol / m^3  (1 m^3 = 1000 L)
    assert _get_dataunit_scaling("mol/m^3") == 1e-3
    assert _get_dataunit_scaling("mmol/m^3") == 1e-6
    # Non-standard lowercase liter
    assert _get_dataunit_scaling("mol/l") == 1.0
    assert _get_dataunit_scaling("mmol/l") == 1e-3
    # Empty string default
    assert _get_dataunit_scaling("") == 1.0
    # Unsupported unit
    with pytest.raises(RuntimeError, match="not supported"):
        _get_dataunit_scaling("bad_unit")


@requires_testing_data
def test_snirf_multiple_wavelengths():
    """Test importing synthetic SNIRF files with >=3 wavelengths."""
    raw = read_raw_snirf(labnirs_multi_wavelength, preload=True)
    assert raw._data.shape == (45, 250)
    assert raw.info["sfreq"] == pytest.approx(19.6, abs=0.01)
    assert raw.info["ch_names"][:3] == ["S2_D2 780", "S2_D2 805", "S2_D2 830"]
    assert len(raw.ch_names) == 45
    freqs = np.unique(_channel_frequencies(raw.info))
    assert_array_equal(freqs, [780, 805, 830])
    distances = source_detector_distances(raw.info)
    assert len(distances) == len(raw.ch_names)
