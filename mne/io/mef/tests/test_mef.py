# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from types import SimpleNamespace

import numpy as np
import pytest

from mne.datasets import testing
from mne.io import read_raw_mef
from mne.io.mef._utils import (
    _UUTC_NO_ENTRY,
    _get_mef_units_scale,
    _mef_time_metadata_extras,
    _records_to_annotations,
    _toc_to_gap_annotations,
)

pymef = pytest.importorskip("pymef")
data_path = testing.data_path(download=False)
mef_file_path = (
    data_path / "MEF" / "sub-ieegModulator_ses-ieeg01_task-photicstim_run-01_ieeg.mefd"
)


@pytest.fixture
def mef_file():
    """Get a MEF3 test file or skip."""
    if not mef_file_path.exists():
        pytest.skip(f"MEF3 test directory not found: {mef_file_path}")
    return mef_file_path


@testing.requires_testing_data
def test_mef_reading(mef_file):
    """Test reading MEF3 file."""
    raw = read_raw_mef(mef_file, preload=False)

    assert raw.info["sfreq"] > 0
    assert len(raw.ch_names) > 0
    assert raw.n_times > 0

    # Test lazy loading
    data, times = raw[:, :100]
    assert data.shape[1] == 100

    # Test full load
    raw.load_data()
    assert raw.preload


@testing.requires_testing_data
def test_mef_channel_types(mef_file):
    """Test that channel types default to sEEG."""
    raw = read_raw_mef(mef_file, preload=False)
    ch_types = set(raw.get_channel_types())

    # All channels should default to sEEG
    assert ch_types == {"seeg"}


@testing.requires_testing_data
def test_mef_data_types(mef_file):
    """Test that data is returned as float64."""
    raw = read_raw_mef(mef_file, preload=True)
    data = raw.get_data()

    assert data.dtype == np.float64


def test_mef_units_scale_helper():
    """Test unit scaling helper against known cases."""
    scale, _, unit_desc_norm, ufact_valid, unit_known = _get_mef_units_scale("uV", 2.0)
    assert scale == pytest.approx(2e-6)
    assert unit_desc_norm == "uv"
    assert ufact_valid
    assert unit_known

    scale, _, unit_desc_norm, ufact_valid, unit_known = _get_mef_units_scale(
        "furlong", 2.0
    )
    assert scale == pytest.approx(2.0)
    assert unit_desc_norm == "furlong"
    assert ufact_valid
    assert not unit_known

    scale, _, unit_desc_norm, ufact_valid, unit_known = _get_mef_units_scale("mV", 0.0)
    assert scale == pytest.approx(1e-3)
    assert unit_desc_norm == "mv"
    assert not ufact_valid
    assert unit_known


def test_mef_time_metadata_extras():
    """Test session time metadata extraction."""
    md3 = dict(
        recording_time_offset=123456,
        DST_start_time=_UUTC_NO_ENTRY,
        DST_end_time=789,
    )
    extras = _mef_time_metadata_extras(md3)
    assert extras == {"recording_time_offset": 123456, "dst_end_time": 789}


def test_mef_record_annotations():
    """Test record annotation conversion."""
    start_uutc = 1_000_000
    session = SimpleNamespace(
        session_md={
            "records_info": {
                "records": [
                    dict(type="Note", time=start_uutc + 2_000_000, text="hello")
                ]
            }
        }
    )
    ts_channels = {
        "CH01": {
            "records_info": {
                "records": [dict(type="SyLg", time=start_uutc + 1_000_000, text="chan")]
            },
            "segments": {
                "seg-000001": {
                    "records_info": {
                        "records": [
                            dict(
                                type="EDFA",
                                time=start_uutc + 3_000_000,
                                text="seg",
                                duration=500_000,
                            )
                        ]
                    }
                }
            },
        }
    }

    onsets, durations, desc, ch_names, extras = _records_to_annotations(
        session, ts_channels, start_uutc
    )
    order = np.argsort(onsets)
    desc = [desc[ii] for ii in order]
    ch_names = [ch_names[ii] for ii in order]
    extras = [extras[ii] for ii in order]
    durations = [durations[ii] for ii in order]

    assert desc[0] == "SyLg: chan"
    assert ch_names[0] == ["CH01"]
    assert extras[0]["channel"] == "CH01"
    assert desc[1] == "Note: hello"
    assert ch_names[1] == []
    assert desc[2] == "EDFA: seg"
    assert ch_names[2] == ["CH01"]
    assert extras[2]["segment"] == "seg-000001"
    assert durations[2] == pytest.approx(0.5)


def test_mef_toc_gap_annotations():
    """Test TOC gap annotations."""
    toc = np.array(
        [
            [1, 0, 1],
            [10, 10, 10],
            [0, 10, 25],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    onsets, durations = _toc_to_gap_annotations(toc, sfreq=10.0)
    assert onsets == [2.0]
    assert durations == [0.5]


@testing.requires_testing_data
def test_mef_scaling_matches_pymef(mef_file):
    """Test that MNE scaling matches pymef data plus metadata scaling."""
    raw = read_raw_mef(mef_file, preload=False)
    session = pymef.mef_session.MefSession(str(mef_file), "")
    ts_channels = session.session_md["time_series_channels"]
    if not ts_channels:
        pytest.skip("No MEF time series channels available")

    scales = []
    for ch_name in raw.ch_names:
        ch_md = ts_channels[ch_name]["section_2"]
        scale, _, _, _, _ = _get_mef_units_scale(
            ch_md["units_description"], ch_md["units_conversion_factor"]
        )
        scales.append(scale)
    scales = np.array(scales)
    if np.allclose(scales, 1.0):
        pytest.skip("MEF test data uses unit scaling factor 1 for all channels")

    start, stop = 0, 10
    pymef_data = session.read_ts_channels_sample(raw.ch_names, [start, stop])
    pymef_data = np.array(pymef_data, dtype=np.float64)
    expected = pymef_data * scales[:, np.newaxis]

    data = raw.get_data(start=start, stop=stop)
    np.testing.assert_allclose(data, expected, rtol=1e-7, atol=0.0)
