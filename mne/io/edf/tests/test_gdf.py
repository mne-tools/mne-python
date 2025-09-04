# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import shutil
from datetime import date, datetime, timedelta, timezone

import numpy as np
import scipy.io as sio
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal

from mne import events_from_annotations, find_events, pick_types
from mne.datasets import testing
from mne.io import read_raw_gdf
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)
gdf1_path = data_path / "GDF" / "test_gdf_1.25"
gdf2_path = data_path / "GDF" / "test_gdf_2.20"
gdf_1ch_path = data_path / "GDF" / "test_1ch.gdf"


@testing.requires_testing_data
def test_gdf_data():
    """Test reading raw GDF 1.x files."""
    raw = read_raw_gdf(
        gdf1_path.with_name(gdf1_path.name + ".gdf"),
        eog=None,
        misc=None,
        preload=True,
    )
    picks = pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    data, _ = raw[picks]

    # Test Status is added as event
    EXPECTED_EVS_ONSETS = raw._raw_extras[0]["events"][1]
    EXPECTED_EVS_ID = {
        f"{evs}": i
        for i, evs in enumerate(
            [
                32769,
                32770,
                33024,
                33025,
                33026,
                33027,
                33028,
                33029,
                33040,
                33041,
                33042,
                33043,
                33044,
                33045,
                33285,
                33286,
            ],
            1,
        )
    }
    evs, evs_id = events_from_annotations(raw)
    assert_array_equal(evs[:, 0], EXPECTED_EVS_ONSETS)
    assert evs_id == EXPECTED_EVS_ID

    # this .npy was generated using the official biosig python package
    raw_biosig = np.load(gdf1_path.with_name(gdf1_path.name + "_biosig.npy"))
    raw_biosig = raw_biosig * 1e-6  # data are stored in microvolts
    data_biosig = raw_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Test for events
    assert len(raw.annotations.duration == 963)

    # gh-5604
    assert raw.info["meas_date"] is None


@testing.requires_testing_data
def test_gdf2_birthday(tmp_path):
    """Test reading raw GDF 2.x files."""
    new_fname = tmp_path / "temp.gdf"
    shutil.copyfile(gdf2_path.with_name(gdf2_path.name + ".gdf"), new_fname)
    # go back 44.5 years so the subject should show up as 44
    offset_edf = datetime.now(tz=timezone.utc) - datetime(  # to their ref
        1, 1, 1, tzinfo=timezone.utc
    )
    offset_44_yr = offset_edf - timedelta(days=int(365 * 44.5))  # 44.5 yr ago
    offset_44_yr_days = offset_44_yr.total_seconds() / (24 * 60 * 60)  # days
    d = (int(offset_44_yr_days) + 367) * 2**32  # with their conversion
    with open(new_fname, "r+b") as fid:
        fid.seek(176, 0)
        assert np.fromfile(fid, np.uint64, 1)[0] == 0
        fid.seek(176, 0)
        fid.write(np.array([d], np.uint64).tobytes())
        fid.seek(176, 0)
        assert np.fromfile(fid, np.uint64, 1)[0] == d
    raw = read_raw_gdf(new_fname, eog=None, misc=None, preload=True)
    assert "subject_info" not in raw._raw_extras[0]
    assert raw.info["subject_info"] is not None
    birthdate = datetime(1, 1, 1, tzinfo=timezone.utc) + offset_44_yr
    assert raw.info["subject_info"]["birthday"] == date(
        birthdate.year,
        birthdate.month,
        birthdate.day,
    )


@testing.requires_testing_data
def test_gdf2_data():
    """Test reading raw GDF 2.x files."""
    raw = read_raw_gdf(
        gdf2_path.with_name(gdf2_path.name + ".gdf"),
        eog=None,
        misc=None,
        preload=True,
    )
    assert raw.info["subject_info"]["birthday"] == date(1, 1, 1)

    picks = pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    data, _ = raw[picks]

    # This .mat was generated using the official biosig matlab package
    mat = sio.loadmat(gdf2_path.with_name(gdf2_path.name + "_biosig.mat"))
    data_biosig = mat["dat"] * 1e-6  # data are stored in microvolts
    data_biosig = data_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Find events
    events = find_events(raw, verbose=1)
    events[:, 2] >>= 8  # last 8 bits are system events in biosemi files
    assert_equal(events.shape[0], 2)  # 2 events in file
    assert_array_equal(events[:, 2], [20, 28])

    # gh-5604
    assert raw.info["meas_date"] is None
    _test_raw_reader(
        read_raw_gdf,
        input_fname=gdf2_path.with_name(gdf2_path.name + ".gdf"),
        eog=None,
        misc=None,
        test_scaling=False,  # XXX this should be True
    )


@testing.requires_testing_data
def test_one_channel_gdf():
    """Test a one-channel GDF file."""
    ecg = read_raw_gdf(gdf_1ch_path, preload=True)
    assert ecg["ECG"][0].shape == (1, 4500)
    assert 150.0 == ecg.info["sfreq"]


@testing.requires_testing_data
def test_gdf_exclude_channels():
    """Test reading GDF data with excluded channels."""
    raw = read_raw_gdf(
        gdf1_path.with_name(gdf1_path.name + ".gdf"), exclude=("FP1", "O1")
    )
    assert "FP1" not in raw.ch_names
    assert "O1" not in raw.ch_names
    raw = read_raw_gdf(
        gdf2_path.with_name(gdf2_path.name + ".gdf"), exclude=("Fp1", "O1")
    )
    assert "Fp1" not in raw.ch_names
    assert "O1" not in raw.ch_names
    raw = read_raw_gdf(gdf2_path.with_name(gdf2_path.name + ".gdf"), exclude=".+z$")
    assert "AFz" not in raw.ch_names
    assert "Cz" not in raw.ch_names
    assert "Pz" not in raw.ch_names
    assert "Oz" not in raw.ch_names


@testing.requires_testing_data
def test_gdf_include():
    """Test reading GDF data with include."""
    raw = read_raw_gdf(
        gdf1_path.with_name(gdf1_path.name + ".gdf"), include=("FP1", "O1")
    )
    assert sorted(raw.ch_names) == ["FP1", "O1"]
