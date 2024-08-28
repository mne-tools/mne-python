# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy.testing import assert_allclose

from mne import Annotations
from mne.datasets import testing
from mne.io import BaseRaw, read_raw_ant, read_raw_brainvision

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("antio", minversion="0.3.0")
data_path = testing.data_path(download=False) / "antio"


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
def ca_208() -> dict[str, dict[str, Path] | str | int | dict[str, str | int]]:
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
def andy_101() -> dict[str, dict[str, Path] | str | int | dict[str, str | int]]:
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


@pytest.mark.parametrize("dataset", ["ca_208", "andy_101"])
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


@pytest.mark.parametrize("dataset", ["ca_208", "andy_101"])
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


def test_io_info_parse_misc(
    ca_208: dict[str, dict[str, Path] | str | int | dict[str, str | int]],
):
    """Test parsing misc channels from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"])
    with pytest.warns(
        RuntimeWarning,
        match="All EEG channels are not referenced to the same electrode.",
    ):
        raw_cnt = read_raw_ant(ca_208["cnt"]["short"], misc=None)
    assert len(raw_cnt.ch_names) == ca_208["n_eeg"] + ca_208["n_misc"]
    assert raw_cnt.get_channel_types() == ["eeg"] * len(raw_cnt.ch_names)


def test_io_info_parse_eog(
    ca_208: dict[str, dict[str, Path] | str | int | dict[str, str | int]],
):
    """Test parsing EOG channels from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"], eog="EOG")
    assert len(raw_cnt.ch_names) == ca_208["n_eeg"] + ca_208["n_misc"]
    idx = raw_cnt.ch_names.index("EOG")
    ch_types = ["eeg"] * ca_208["n_eeg"] + ["misc"] * ca_208["n_misc"]
    ch_types[idx] = "eog"
    assert raw_cnt.get_channel_types() == ch_types


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
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


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_machine_info(dataset, request):
    """Test reading the machine info."""
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])
    device_info = raw_cnt.info["device_info"]
    make, model, serial = dataset["machine_info"]
    assert device_info["type"] == make
    assert device_info["model"] == model
    assert device_info["serial"] == serial


def test_io_amp_disconnection(ca_208: dict[str, dict[str, Path]]) -> None:
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


@pytest.mark.parametrize("description", ["impedance", "test"])
def test_io_impedance(ca_208: dict[str, dict[str, Path]], description: str) -> None:
    """Test loading of impedances from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"], impedance_annotation=description)
    assert isinstance(raw_cnt.impedances, list)
    for elt in raw_cnt.impedances:
        assert isinstance(elt, dict)
        assert list(elt) == raw_cnt.ch_names
        assert all(isinstance(val, float) for val in elt.values())
    annotations = [
        annot for annot in raw_cnt.annotations if annot["description"] == description
    ]
    assert len(annotations) == len(raw_cnt.impedances)


def test_io_segments(ca_208: dict[str, dict[str, Path]]) -> None:
    """Test reading a .cnt file with segents (start/stop)."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["start-stop"])
    raw_bv = read_raw_bv(ca_208["bv"]["start-stop"])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
