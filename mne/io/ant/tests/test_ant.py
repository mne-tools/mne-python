# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import Annotations
from mne._fiff.constants import FIFF
from mne.datasets import testing
from mne.io import BaseRaw, read_raw, read_raw_ant, read_raw_brainvision
from mne.io.ant.ant import RawANT

pytest.importorskip("antio", minversion="0.5.0")
data_path = testing.data_path(download=False) / "antio"


TypeDataset = dict[
    str, dict[str, Path] | str | int | tuple[str, str, str] | dict[str, str] | None
]


def read_raw_bv(fname: Path) -> BaseRaw:
    """Read a brainvision file exported from eego.

    For some reason, the first impedance measurement is annotated at sample 0. But since
    BrainVision files are 1-indexed, the reader removes '1' to create 0-indexed
    annotations. Thus, the first impedance measurement annotation ends up with an onset
    1 sample before the start of the recording.
    This is not really an issue as the annotation duration is sufficient to make sure
    that MNE does not drop it entirely as 'outside of the data range'.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Limited .* annotation.*outside the data range.",
            category=RuntimeWarning,
        )
        raw_bv = read_raw_brainvision(fname)
    return raw_bv


@pytest.fixture(scope="module")
def ca_208() -> TypeDataset:
    """Return the paths to the CA_208 dataset containing 64 channel gel recordings."""
    cnt = {
        "short": data_path / "CA_208" / "test_CA_208.cnt",
        "amp-dc": data_path / "CA_208" / "test_CA_208_amp_disconnection.cnt",
        "start-stop": data_path / "CA_208" / "test_CA_208_start_stop.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 64,
        "n_misc": 24,
        "meas_date": "2024-08-14-10-44-47+0000",
        "patient_info": {
            "name": "antio test",
            "his_id": "",
            "birthday": "2024-08-14",
            "sex": 0,
        },
        "machine_info": ("eego", "EE_225", ""),
        "hospital": "",
    }


@pytest.fixture(scope="module")
def ca_208_refs() -> TypeDataset:
    """Return the paths and info to the CA_208_refs dataset.

    The following montage was applid on export:
    - highpass: 0.3 Hz - lowpass: 30 Hz
    - Fp1, Fpz, Fp2 referenced to Fz
    - CP3, CP4 referenced to Cz
    - others to CPz
    """
    cnt = {
        "short": data_path / "CA_208_refs" / "test-ref.cnt",
        "legacy": data_path / "CA_208_refs" / "test-ref-legacy.cnt",
    }
    bv = {
        "short": cnt["short"].with_suffix(".vhdr"),
    }
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 64,
        "n_misc": 0,
        "meas_date": "2024-09-09-10-57-44+0000",
        "patient_info": {
            "name": "antio test",
            "his_id": "",
            "birthday": "2024-08-14",
            "sex": 0,
        },
        "machine_info": ("eego", "EE_225", ""),
        "hospital": "",
    }


@pytest.fixture(scope="module")
def andy_101() -> TypeDataset:
    """Return the path and info to the andy_101 dataset."""
    cnt = {
        "short": data_path / "andy_101" / "Andy_101-raw.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 128,
        "n_misc": 0,
        "meas_date": "2024-08-19-16-17-07+0000",
        "patient_info": {
            "name": "Andy test_middle_name EEG_Exam",
            "his_id": "test_subject_code",
            "birthday": "2024-08-19",
            "sex": 2,
        },
        "machine_info": ("eego", "EE_226", ""),
        "hospital": "",
    }


@pytest.fixture(scope="module")
def na_271() -> TypeDataset:
    """Return the path to a dataset containing 128 channel recording.

    The recording was done with an NA_271 net dipped in saline solution.
    """
    cnt = {
        "short": data_path / "NA_271" / "test-na-271.cnt",
        "legacy": data_path / "NA_271" / "test-na-271-legacy.cnt",
    }
    bv = {
        "short": cnt["short"].with_suffix(".vhdr"),
    }
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 128,
        "n_misc": 0,
        "meas_date": "2024-09-06-10-45-07+0000",
        "patient_info": {
            "name": "antio test",
            "his_id": "",
            "birthday": "2024-08-14",
            "sex": 0,
        },
        "machine_info": ("eego", "EE_226", ""),
        "hospital": "",
    }


@pytest.fixture(scope="module")
def na_271_bips() -> TypeDataset:
    """Return the path to a dataset containing 128 channel recording.

    The recording was done with an NA_271 net dipped in saline solution and includes
    bipolar channels.
    """
    cnt = {
        "short": data_path / "NA_271_bips" / "test-na-271.cnt",
        "legacy": data_path / "NA_271_bips" / "test-na-271-legacy.cnt",
    }
    bv = {
        "short": cnt["short"].with_suffix(".vhdr"),
    }
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 128,
        "n_misc": 6,
        "meas_date": "2024-09-06-10-37-23+0000",
        "patient_info": {
            "name": "antio test",
            "his_id": "",
            "birthday": "2024-08-14",
            "sex": 0,
        },
        "machine_info": ("eego", "EE_226", ""),
        "hospital": "",
    }


@pytest.fixture(scope="module")
def user_annotations() -> TypeDataset:
    """Return the path to a dataset containing user annotations with floating pins."""
    cnt = {
        "short": data_path / "user_annotations" / "test-user-annotation.cnt",
        "legacy": data_path / "user_annotations" / "test-user-annotation-legacy.cnt",
    }
    bv = {
        "short": cnt["short"].with_suffix(".vhdr"),
    }
    return {
        "cnt": cnt,
        "bv": bv,
        "n_eeg": 64,
        "n_misc": 0,
        "meas_date": "2024-08-29-16-15-44+0000",
        "patient_info": {
            "name": "test test",
            "his_id": "",
            "birthday": "2024-02-06",
            "sex": 0,
        },
        "machine_info": ("eego", "EE_225", ""),
        "hospital": "",
    }


@testing.requires_testing_data
@pytest.mark.parametrize("dataset", ["ca_208", "andy_101", "na_271"])
def test_io_data(dataset, request):
    """Test loading of .cnt file."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])  # preload=False
    raw_bv = read_raw_bv(dataset["bv"]["short"])
    cnt = raw_cnt.get_data()
    bv = raw_bv.get_data()
    assert cnt.shape == bv.shape
    assert_allclose(cnt, bv, atol=1e-8)

    # check preload=False and preload=False with raw.load_data()
    raw_cnt.crop(0.05, 1.05)
    raw_cnt2 = read_raw_ant(dataset["cnt"]["short"], preload=False)
    raw_cnt2.crop(0.05, 1.05).load_data()
    assert_allclose(raw_cnt.get_data(), raw_cnt2.get_data())

    # check preload=False vs Brainvision file
    raw_bv.crop(0.05, 1.05)
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)

    # check preload=False vs BrainVision file after dropping channels
    raw_cnt.pick(raw_cnt.ch_names[::2])
    raw_bv.pick(raw_bv.ch_names[::2])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)

    # check after raw_cnt.load_data()
    raw_cnt.load_data()
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)

    # check preload True vs False
    raw_cnt = read_raw_ant(dataset["cnt"]["short"], preload=False)
    raw_cnt2 = read_raw_ant(dataset["cnt"]["short"], preload=True)
    bads = [raw_cnt.ch_names[idx] for idx in (1, 5, 10)]
    assert_allclose(
        raw_cnt.drop_channels(bads).get_data(), raw_cnt2.drop_channels(bads).get_data()
    )
    raw_bv = read_raw_bv(dataset["bv"]["short"]).drop_channels(bads)
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
    assert_allclose(raw_cnt2.get_data(), raw_bv.get_data(), atol=1e-8)


@testing.requires_testing_data
@pytest.mark.parametrize("dataset", ["ca_208", "andy_101", "na_271"])
def test_io_info(dataset, request):
    """Test the ifo loaded from a .cnt file."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])  # preload=False
    raw_bv = read_raw_bv(dataset["bv"]["short"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert (
        raw_cnt.get_channel_types()
        == ["eeg"] * dataset["n_eeg"] + ["misc"] * dataset["n_misc"]
    )
    assert_allclose(
        (raw_bv.info["meas_date"] - raw_cnt.info["meas_date"]).total_seconds(),
        0,
        atol=1e-3,
    )


@testing.requires_testing_data
def test_io_info_parse_misc(ca_208: TypeDataset):
    """Test parsing misc channels from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"])
    with pytest.warns(
        RuntimeWarning,
        match="All EEG channels are not referenced to the same electrode.",
    ):
        raw_cnt = read_raw_ant(ca_208["cnt"]["short"], misc=None)
    assert len(raw_cnt.ch_names) == ca_208["n_eeg"] + ca_208["n_misc"]
    assert raw_cnt.get_channel_types() == ["eeg"] * len(raw_cnt.ch_names)


def test_io_info_parse_non_standard_misc(na_271_bips: TypeDataset):
    """Test parsing misc channels with modified names from a .cnt file."""
    with pytest.warns(
        RuntimeWarning, match="EEG channels are not referenced to the same electrode"
    ):
        raw = read_raw_ant(na_271_bips["cnt"]["short"], misc=None)
    assert raw.get_channel_types() == ["eeg"] * (
        na_271_bips["n_eeg"] + na_271_bips["n_misc"]
    )
    raw = read_raw_ant(
        na_271_bips["cnt"]["short"], preload=False, misc=r".{0,1}E.{1}G|Aux|Audio"
    )
    assert (
        raw.get_channel_types()
        == ["eeg"] * na_271_bips["n_eeg"] + ["misc"] * na_271_bips["n_misc"]
    )


@testing.requires_testing_data
def test_io_info_parse_eog(ca_208: TypeDataset):
    """Test parsing EOG channels from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"], eog="EOG")
    assert len(raw_cnt.ch_names) == ca_208["n_eeg"] + ca_208["n_misc"]
    idx = raw_cnt.ch_names.index("EOG")
    ch_types = ["eeg"] * ca_208["n_eeg"] + ["misc"] * ca_208["n_misc"]
    ch_types[idx] = "eog"
    assert raw_cnt.get_channel_types() == ch_types


@testing.requires_testing_data
@pytest.mark.parametrize(
    "dataset", ["andy_101", "ca_208", "na_271", "user_annotations"]
)
def test_subject_info(dataset, request):
    """Test reading the subject info."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])
    subject_info = raw_cnt.info["subject_info"]
    assert subject_info["his_id"] == dataset["patient_info"]["his_id"]
    assert subject_info["first_name"] == dataset["patient_info"]["name"]
    assert subject_info["sex"] == dataset["patient_info"]["sex"]
    assert (
        subject_info["birthday"].strftime("%Y-%m-%d")
        == dataset["patient_info"]["birthday"]
    )


@testing.requires_testing_data
@pytest.mark.parametrize(
    "dataset", ["andy_101", "ca_208", "na_271", "user_annotations"]
)
def test_machine_info(dataset, request):
    """Test reading the machine info."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])
    device_info = raw_cnt.info["device_info"]
    make, model, serial = dataset["machine_info"]
    assert device_info["type"] == make
    assert device_info["model"] == model
    assert device_info["serial"] == serial


@testing.requires_testing_data
def test_io_amp_disconnection(ca_208: TypeDataset):
    """Test loading of .cnt file with amplifier disconnection."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"])
    raw_bv = read_raw_bv(ca_208["bv"]["amp-dc"])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
    assert (
        raw_cnt.get_data(reject_by_annotation="omit").shape != raw_bv.get_data().shape
    )
    # create annotation on the BV file
    idx = [
        k
        for k, elt in enumerate(raw_bv.annotations.description)
        if any(code in elt for code in ("9001", "9002"))
    ]
    assert len(idx) == 2
    start = raw_bv.annotations.onset[idx[0]]
    stop = raw_bv.annotations.onset[idx[1]]
    annotations = Annotations(
        onset=start,
        duration=stop - start + 1 / raw_bv.info["sfreq"],  # estimate is 1 sample short
        description="BAD_segment",
    )
    raw_bv.set_annotations(annotations)
    assert_allclose(
        raw_cnt.get_data(reject_by_annotation="omit"),
        raw_bv.get_data(reject_by_annotation="omit"),
        atol=1e-8,
    )


@testing.requires_testing_data
@pytest.mark.parametrize("description", ["impedance", "test"])
def test_io_impedance(ca_208: TypeDataset, description: str):
    """Test loading of impedances from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"], impedance_annotation=description)
    annotations = [
        annot for annot in raw_cnt.annotations if annot["description"] == description
    ]
    assert len(annotations) != 0


@testing.requires_testing_data
def test_io_segments(ca_208: TypeDataset):
    """Test reading a .cnt file with segents (start/stop)."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["start-stop"])
    raw_bv = read_raw_bv(ca_208["bv"]["start-stop"])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)


@testing.requires_testing_data
def test_annotations_and_preload(ca_208: TypeDataset):
    """Test annotation loading with preload True/False."""
    raw_cnt_preloaded = read_raw_ant(ca_208["cnt"]["short"], preload=True)
    assert len(raw_cnt_preloaded.annotations) == 2  # impedance measurements, start/end
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"], preload=False)
    assert len(raw_cnt.annotations) == 2
    raw_cnt.crop(2, 3)
    assert len(raw_cnt.annotations) == 0
    raw_cnt.load_data()
    assert len(raw_cnt.annotations) == 0

    raw_cnt_preloaded = read_raw_ant(ca_208["cnt"]["amp-dc"], preload=True)
    assert len(raw_cnt_preloaded.annotations) == 5  # 4 impedances, 1 disconnection
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"], preload=False)
    assert len(raw_cnt.annotations) == 5
    idx = np.where(raw_cnt.annotations.description == "BAD_disconnection")[0]
    onset = raw_cnt.annotations.onset[idx][0]
    raw_cnt.crop(0, onset - 1)
    assert len(raw_cnt.annotations) == 1  # initial impedance measurement
    assert raw_cnt.annotations.description[0] == "impedance"


@testing.requires_testing_data
def test_read_raw(ca_208: TypeDataset):
    """Test loading through read_raw."""
    raw = read_raw(ca_208["cnt"]["short"])
    assert isinstance(raw, RawANT)


@testing.requires_testing_data
@pytest.mark.parametrize("preload", [True, False])
def test_read_raw_with_user_annotations(user_annotations: TypeDataset, preload: bool):
    """Test reading raw objects which have user annotations."""
    raw = read_raw_ant(user_annotations["cnt"]["short"], preload=preload)
    assert raw.annotations
    assert "1000/user-annot" in raw.annotations.description
    assert "1000/user-annot-2" in raw.annotations.description


@testing.requires_testing_data
@pytest.mark.parametrize("dataset", ["na_271", "user_annotations"])
def test_read_raw_legacy_format(dataset, request):
    """Test reading the legacy CNT format."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])  # preload=False
    raw_bv = read_raw_bv(dataset["bv"]["short"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert (
        raw_cnt.get_channel_types()
        == ["eeg"] * dataset["n_eeg"] + ["misc"] * dataset["n_misc"]
    )
    assert_allclose(
        (raw_bv.info["meas_date"] - raw_cnt.info["meas_date"]).total_seconds(),
        0,
        atol=1e-3,
    )


@testing.requires_testing_data
def test_read_raw_custom_reference(ca_208_refs: TypeDataset):
    """Test reading a CNT file with custom EEG references."""
    with pytest.warns(
        RuntimeWarning, match="EEG channels are not referenced to the same electrode"
    ):
        raw = read_raw_ant(ca_208_refs["cnt"]["short"], preload=False)
    for ch in raw.info["chs"]:
        assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
    bipolars = ("Fp1-Fz", "Fpz-Fz", "Fp2-Fz", "CP3-Cz", "CP4-Cz")
    with pytest.warns(
        RuntimeWarning, match="EEG channels are not referenced to the same electrode"
    ):
        raw = read_raw_ant(
            ca_208_refs["cnt"]["short"], preload=False, bipolars=bipolars
        )
    assert all(elt in raw.ch_names for elt in bipolars)
    for ch in raw.info["chs"]:
        if ch["ch_name"] in bipolars:
            assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG_BIPOLAR
        else:
            assert ch["coil_type"] == FIFF.FIFFV_COIL_EEG
