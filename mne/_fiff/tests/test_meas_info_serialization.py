"""Test Info serialization to dict and JSON."""
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import Info, create_info
from mne.channels import make_standard_montage
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.transforms import Transform


def test_basic_info_to_dict():
    """Test basic Info.to_data_dict() method."""
    ch_names = ["MEG1", "MEG2", "EEG1", "EOG1"]
    ch_types = ["mag", "grad", "eeg", "eog"]
    sfreq = 1000.0

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info["description"] = "Test recording"
    info["bads"] = ["MEG2"]

    # Convert to dict
    info_dict = info.to_data_dict()

    # Check it's JSON-serializable
    json_str = json.dumps(info_dict)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Check metadata
    assert "_mne_version" in info_dict
    assert isinstance(info_dict["_mne_version"], str)


def test_basic_info_from_dict():
    """Test basic Info.from_data_dict() method."""
    ch_names = ["MEG1", "MEG2"]
    sfreq = 500.0

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="mag")
    info["description"] = "Restored test"
    info["bads"] = ["MEG1"]

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify
    assert info_restored["sfreq"] == sfreq
    assert info_restored["ch_names"] == ch_names
    assert info_restored["nchan"] == len(ch_names)
    assert info_restored["description"] == "Restored test"
    assert list(info_restored["bads"]) == ["MEG1"]


def test_info_roundtrip_json():
    """Test Info roundtrip through JSON."""
    info = create_info(ch_names=["EEG1", "EEG2"], sfreq=1000.0, ch_types="eeg")
    info["description"] = "Roundtrip test"

    # Convert to JSON and back
    json_str = json.dumps(info.to_data_dict())
    info_restored = Info.from_data_dict(json.loads(json_str))

    # Verify
    assert info_restored["sfreq"] == info["sfreq"]
    assert info_restored["ch_names"] == info["ch_names"]
    assert info_restored["description"] == info["description"]


def test_info_with_transform():
    """Test Info serialization with Transform objects."""
    info = create_info(ch_names=["MEG1"], sfreq=1000.0, ch_types="mag")

    trans = Transform("meg", "head", np.eye(4))
    with info._unlock():
        info["dev_head_t"] = trans

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify transform
    assert info_restored["dev_head_t"] is not None
    assert isinstance(info_restored["dev_head_t"], Transform)
    assert info_restored["dev_head_t"]["from"] == info["dev_head_t"]["from"]
    assert info_restored["dev_head_t"]["to"] == info["dev_head_t"]["to"]
    assert_allclose(info_restored["dev_head_t"]["trans"], info["dev_head_t"]["trans"])


def test_info_with_montage():
    """Test Info serialization with digitization points."""
    montage = make_standard_montage("standard_1020")
    ch_names = ["Fp1", "Fp2", "F3", "F4"]
    info = create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage)

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify digitization
    assert info_restored["dig"] is not None
    assert len(info_restored["dig"]) == len(info["dig"])

    # Check a few dig points
    for orig_dig, restored_dig in zip(info["dig"][:3], info_restored["dig"][:3]):
        assert orig_dig["kind"] == restored_dig["kind"]
        assert_allclose(orig_dig["r"], restored_dig["r"])


def test_info_with_subject_info():
    """Test Info serialization with subject_info."""
    info = create_info(ch_names=["EEG1"], sfreq=1000.0, ch_types="eeg")

    info["subject_info"] = {
        "id": 1,
        "his_id": "SUBJ001",
        "first_name": "John",
        "last_name": "Doe",
        "sex": 1,
        "birthday": date(1990, 1, 15),
    }

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify subject info
    assert info_restored["subject_info"]["id"] == 1
    assert info_restored["subject_info"]["his_id"] == "SUBJ001"
    assert info_restored["subject_info"]["birthday"] == date(1990, 1, 15)
    assert isinstance(info_restored["subject_info"]["birthday"], date)


def test_info_with_meas_date():
    """Test Info serialization with meas_date."""
    info = create_info(ch_names=["EEG1"], sfreq=1000.0, ch_types="eeg")

    meas_date = datetime(2023, 11, 13, 10, 30, 0, tzinfo=timezone.utc)
    with info._unlock():
        info["meas_date"] = meas_date

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify meas_date
    assert info_restored["meas_date"] == meas_date
    assert isinstance(info_restored["meas_date"], datetime)


def test_info_with_projections():
    """Test Info serialization with SSP projections."""
    # Use real data that has projections
    data_path = testing.data_path(download=False)
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    if not raw_fname.exists():
        pytest.skip("Sample data not available")

    raw = read_raw_fif(raw_fname, preload=False, verbose=False)
    info = raw.info

    # Verify we have projections
    assert len(info["projs"]) > 0

    # Serialize and restore
    info_dict = info.to_data_dict()
    json_str = json.dumps(info_dict)
    info_restored = Info.from_data_dict(json.loads(json_str))

    # Verify projections
    assert len(info_restored["projs"]) == len(info["projs"])
    for orig_proj, restored_proj in zip(info["projs"], info_restored["projs"]):
        assert orig_proj["desc"] == restored_proj["desc"]
        assert orig_proj["active"] == restored_proj["active"]
        assert_allclose(orig_proj["data"]["data"], restored_proj["data"]["data"])


def test_channel_locations_preserved():
    """Test that channel locations are preserved in serialization."""
    montage = make_standard_montage("standard_1020")
    ch_names = ["Fp1", "Fp2", "C3", "C4"]
    info = create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage)

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Check channel locations
    for i, ch in enumerate(info["chs"]):
        orig_loc = ch["loc"]
        restored_loc = info_restored["chs"][i]["loc"]
        assert_allclose(orig_loc, restored_loc, rtol=1e-6)


def test_info_file_roundtrip():
    """Test roundtrip save/load to JSON file."""
    info = create_info(
        ch_names=["EEG1", "EEG2", "EOG"], sfreq=500.0, ch_types=["eeg", "eeg", "eog"]
    )
    info["description"] = "File roundtrip test"
    info["bads"] = ["EEG2"]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "info.json"

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(info.to_data_dict(), f)

        # Load from JSON
        with open(json_path, "r") as f:
            loaded_dict = json.load(f)

        info_restored = Info.from_data_dict(loaded_dict)

        # Verify
        assert info_restored["sfreq"] == info["sfreq"]
        assert info_restored["description"] == info["description"]
        assert list(info_restored["bads"]) == list(info["bads"])


def test_numpy_array_conversion():
    """Test that numpy arrays are properly converted to lists."""
    info = create_info(ch_names=["MEG1"], sfreq=1000.0, ch_types="mag")

    # Add a transform with numpy arrays
    trans_array = np.random.randn(4, 4)
    trans = Transform("meg", "head", trans_array)

    with info._unlock():
        info["dev_head_t"] = trans

    # Serialize
    info_dict = info.to_data_dict()

    # Check that trans is now a list
    assert isinstance(info_dict["dev_head_t"]["trans"], list)
    assert isinstance(info_dict["dev_head_t"]["trans"][0], list)

    # Restore and verify values match
    info_restored = Info.from_data_dict(info_dict)
    assert_allclose(info_restored["dev_head_t"]["trans"], trans_array)


def test_empty_fields():
    """Test serialization with empty/None fields."""
    info = create_info(ch_names=["EEG1"], sfreq=1000.0, ch_types="eeg")

    # These should be None or empty
    assert info["dig"] is None
    assert info["dev_head_t"] is None
    assert len(info["projs"]) == 0

    # Serialize and restore
    info_dict = info.to_data_dict()
    info_restored = Info.from_data_dict(info_dict)

    # Verify empty fields are preserved
    assert info_restored["dig"] is None
    assert info_restored["dev_head_t"] is None
    assert len(info_restored["projs"]) == 0


def test_info_dict_is_json_serializable():
    """Test that to_dict() produces truly JSON-serializable output."""
    # Use channel names that exist in the standard montage
    montage = make_standard_montage("standard_1020")
    ch_names = ["Fp1", "Fp2"]
    
    info = create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage)

    info["description"] = "JSON test"
    info["subject_info"] = {
        "id": 1,
        "birthday": date(1990, 1, 1),
    }

    # Convert to dict
    info_dict = info.to_data_dict()

    # This should not raise any errors
    json_str = json.dumps(info_dict)
    assert isinstance(json_str, str)

    # Should be able to load it back
    loaded = json.loads(json_str)
    assert isinstance(loaded, dict)
