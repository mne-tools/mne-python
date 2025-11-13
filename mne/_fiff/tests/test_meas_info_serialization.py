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
from numpy.testing import assert_allclose

from mne import Info, create_info
from mne.channels import make_standard_montage
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.transforms import Transform


def test_basic_info_to_dict():
    """Test basic Info.to_dict() method."""
    ch_names = ["MEG1", "MEG2", "EEG1", "EOG1"]
    ch_types = ["mag", "grad", "eeg", "eog"]
    sfreq = 1000.0

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info["description"] = "Test recording"
    info["bads"] = ["MEG2"]

    # Convert to dict
    info_dict = info.to_dict()

    # Check it's JSON-serializable
    json_str = json.dumps(info_dict)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Check metadata
    assert "_mne_version" in info_dict
    assert isinstance(info_dict["_mne_version"], str)


def test_basic_info_from_dict():
    """Test basic Info.from_dict() method."""
    ch_names = ["MEG1", "MEG2"]
    sfreq = 500.0

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="mag")
    info["description"] = "Restored test"
    info["bads"] = ["MEG1"]

    # Serialize and restore
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    json_str = json.dumps(info.to_dict())
    info_restored = Info.from_dict(json.loads(json_str))

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
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    info_dict = info.to_dict()
    json_str = json.dumps(info_dict)
    info_restored = Info.from_dict(json.loads(json_str))

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
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
            json.dump(info.to_dict(), f)

        # Load from JSON
        with open(json_path) as f:
            loaded_dict = json.load(f)

        info_restored = Info.from_dict(loaded_dict)

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
    info_dict = info.to_dict()

    # Check that trans is now a list
    assert isinstance(info_dict["dev_head_t"]["trans"], list)
    assert isinstance(info_dict["dev_head_t"]["trans"][0], list)

    # Restore and verify values match
    info_restored = Info.from_dict(info_dict)
    assert_allclose(info_restored["dev_head_t"]["trans"], trans_array)


def test_empty_fields():
    """Test serialization with empty/None fields."""
    info = create_info(ch_names=["EEG1"], sfreq=1000.0, ch_types="eeg")

    # These should be None or empty
    assert info["dig"] is None
    assert info["dev_head_t"] is None
    assert len(info["projs"]) == 0

    # Serialize and restore
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

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
    info_dict = info.to_dict()

    # This should not raise any errors
    json_str = json.dumps(info_dict)
    assert isinstance(json_str, str)

    # Should be able to load it back
    loaded = json.loads(json_str)
    assert isinstance(loaded, dict)


def test_named_int_preservation():
    """Test that NamedInt types are preserved with their names."""
    from mne.utils._bunch import NamedInt

    info = create_info(ch_names=["EEG1"], sfreq=1000.0, ch_types="eeg")

    # custom_ref_applied is a NamedInt field
    assert isinstance(info["custom_ref_applied"], NamedInt)
    original_repr = repr(info["custom_ref_applied"])

    # Serialize and restore
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

    # Verify NamedInt is preserved
    assert isinstance(info_restored["custom_ref_applied"], NamedInt)
    assert isinstance(
        info["custom_ref_applied"], type(info_restored["custom_ref_applied"])
    )
    assert repr(info_restored["custom_ref_applied"]) == original_repr
    assert info["custom_ref_applied"] == info_restored["custom_ref_applied"]


def test_complex_nested_structures():
    """Test serialization of complex nested structures like hpi_meas and hpi_results."""
    data_path = testing.data_path(download=False)
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    if not raw_fname.exists():
        pytest.skip("Sample data not available")

    raw = read_raw_fif(raw_fname, preload=False, verbose=False)
    info = raw.info

    # Verify we have complex nested structures
    assert len(info["hpi_meas"]) > 0
    assert len(info["hpi_results"]) > 0

    # Serialize and restore
    info_dict = info.to_dict()
    json_str = json.dumps(info_dict)
    info_restored = Info.from_dict(json.loads(json_str))

    # Verify hpi_meas structure
    assert len(info_restored["hpi_meas"]) == len(info["hpi_meas"])
    orig_hpi = info["hpi_meas"][0]
    rest_hpi = info_restored["hpi_meas"][0]

    assert orig_hpi["creator"] == rest_hpi["creator"]
    assert orig_hpi["sfreq"] == rest_hpi["sfreq"]
    assert orig_hpi["ncoil"] == rest_hpi["ncoil"]

    # Check nested hpi_coils arrays
    assert len(orig_hpi["hpi_coils"]) == len(rest_hpi["hpi_coils"])
    orig_coil = orig_hpi["hpi_coils"][0]
    rest_coil = rest_hpi["hpi_coils"][0]
    assert_allclose(orig_coil["epoch"], rest_coil["epoch"])
    assert_allclose(orig_coil["slopes"], rest_coil["slopes"])

    # Verify hpi_results with Transform in coord_trans
    assert len(info_restored["hpi_results"]) == len(info["hpi_results"])
    if info["hpi_results"][0]["coord_trans"] is not None:
        orig_trans = info["hpi_results"][0]["coord_trans"]
        rest_trans = info_restored["hpi_results"][0]["coord_trans"]
        assert isinstance(rest_trans, Transform)
        assert orig_trans["from"] == rest_trans["from"]
        assert orig_trans["to"] == rest_trans["to"]
        assert_allclose(orig_trans["trans"], rest_trans["trans"])


def test_all_field_types_comprehensive():
    """Comprehensive test ensuring all Info field types serialize correctly."""
    data_path = testing.data_path(download=False)
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    if not raw_fname.exists():
        pytest.skip("Sample data not available")

    raw = read_raw_fif(raw_fname, preload=False, verbose=False)
    info = raw.info

    # Add additional fields to maximize coverage
    info["subject_info"] = {
        "id": 1,
        "his_id": "SUBJ001",
        "first_name": "Test",
        "last_name": "Subject",
        "birthday": date(1990, 5, 15),
        "sex": 1,
        "hand": 1,
    }

    # Serialize and restore
    info_dict = info.to_dict()
    json_str = json.dumps(info_dict)
    info_restored = Info.from_dict(json.loads(json_str))

    # Test all critical field types
    field_tests = {
        # Scalar types
        "sfreq": (float, lambda o, r: o == r),
        "nchan": (int, lambda o, r: o == r),
        "highpass": (float, lambda o, r: o == r),
        "lowpass": (float, lambda o, r: o == r),
        # String types
        "description": ((str, type(None)), lambda o, r: o == r),
        "experimenter": ((str, type(None)), lambda o, r: o == r),
        # List types
        "ch_names": (list, lambda o, r: o == r),
        "bads": (list, lambda o, r: list(o) == list(r)),
        "chs": (list, lambda o, r: len(o) == len(r)),
        "projs": (list, lambda o, r: len(o) == len(r)),
        "dig": (list, lambda o, r: len(o) == len(r)),
        # Dict types
        "file_id": ((dict, type(None)), lambda o, r: True),
        "meas_id": ((dict, type(None)), lambda o, r: True),
        # Special MNE types
        "dev_head_t": (
            (Transform, type(None)),
            lambda o, r: type(o) is type(r),
        ),
        "custom_ref_applied": (
            int,
            lambda o, r: o == r,
        ),  # NamedInt, but subclass of int
    }

    failures = []
    for field, (expected_type, validator) in field_tests.items():
        orig = info[field]
        rest = info_restored[field]

        # Type check
        if not isinstance(rest, expected_type):
            failures.append(
                f"{field}: type mismatch - expected {expected_type}, got {type(rest)}"
            )
            continue

        # Value check
        try:
            if not validator(orig, rest):
                failures.append(f"{field}: value mismatch")
        except Exception as e:
            failures.append(f"{field}: validation error - {e}")

    # Special checks for complex types
    # Check Transform preservation
    if info["dev_head_t"] is not None:
        assert isinstance(info_restored["dev_head_t"], Transform)
        assert_allclose(
            info["dev_head_t"]["trans"], info_restored["dev_head_t"]["trans"]
        )

    # Check Projection preservation
    if len(info["projs"]) > 0:
        from mne.proj import Projection

        assert isinstance(info_restored["projs"][0], Projection)
        assert info["projs"][0]["desc"] == info_restored["projs"][0]["desc"]

    # Check DigPoint preservation
    if info["dig"] is not None and len(info["dig"]) > 0:
        from mne._fiff._digitization import DigPoint

        assert isinstance(info_restored["dig"][0], DigPoint)
        assert info["dig"][0]["kind"] == info_restored["dig"][0]["kind"]

    # Check NamedInt preservation
    from mne.utils._bunch import NamedInt

    assert isinstance(info_restored["custom_ref_applied"], NamedInt)
    assert repr(info["custom_ref_applied"]) == repr(info_restored["custom_ref_applied"])

    # Check subject_info with date
    assert info_restored["subject_info"]["birthday"] == date(1990, 5, 15)
    assert isinstance(info_restored["subject_info"]["birthday"], date)

    # Check datetime
    assert isinstance(info_restored["meas_date"], datetime)
    assert info_restored["meas_date"] == info["meas_date"]

    if failures:
        pytest.fail("Field validation failures:\n" + "\n".join(failures))


def test_channel_info_preservation():
    """Test that channel info with locations is fully preserved."""
    montage = make_standard_montage("standard_1020")
    ch_names = ["Fp1", "Fp2", "C3", "C4", "Oz"]
    info = create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage)

    # Serialize and restore
    info_dict = info.to_dict()
    info_restored = Info.from_dict(info_dict)

    # Verify all channel properties
    for i, ch in enumerate(info["chs"]):
        rest_ch = info_restored["chs"][i]

        assert ch["ch_name"] == rest_ch["ch_name"]
        assert ch["kind"] == rest_ch["kind"]
        assert ch["unit"] == rest_ch["unit"]
        assert ch["cal"] == rest_ch["cal"]
        assert_allclose(ch["loc"], rest_ch["loc"], rtol=1e-6)


def test_datetime_explicit_tagging():
    """Test that datetime objects use explicit _mne_type tagging."""
    from mne._fiff.meas_info import _make_serializable, _restore_objects

    # Test datetime tagging
    dt = datetime(2023, 11, 13, 10, 30, 0, tzinfo=timezone.utc)
    serialized_dt = _make_serializable(dt)

    assert isinstance(serialized_dt, dict)
    assert serialized_dt["_mne_type"] == "datetime"
    assert serialized_dt["value"] == "2023-11-13T10:30:00+00:00"

    # Test date tagging
    d = date(1990, 1, 15)
    serialized_date = _make_serializable(d)

    assert isinstance(serialized_date, dict)
    assert serialized_date["_mne_type"] == "date"
    assert serialized_date["value"] == "1990-01-15"

    # Test restoration
    restored_dt = _restore_objects(serialized_dt)
    assert isinstance(restored_dt, datetime)
    assert restored_dt == dt

    restored_date = _restore_objects(serialized_date)
    assert isinstance(restored_date, date)
    assert restored_date == d

    # Test that strings with date-like patterns are NOT converted
    # This prevents false positives
    date_like_strings = [
        "2023-01-01-my-file-name",
        "2024-05-15-experiment",
        "1990-12-31",  # Plain string, not tagged
    ]

    for s in date_like_strings:
        # If not tagged, it should remain a string
        assert _restore_objects(s) == s
        assert isinstance(_restore_objects(s), str)
