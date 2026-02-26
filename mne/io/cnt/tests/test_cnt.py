# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne import pick_types
from mne.annotations import read_annotations
from mne.datasets import testing
from mne.io.cnt import read_raw_cnt
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)
fname = data_path / "CNT" / "scan41_short.cnt"
# Contains bad spans and could not be read properly before PR #12393
fname_bad_spans = data_path / "CNT" / "test_CNT_events_mne_JWoess_clipped.cnt"


_no_parse = pytest.warns(RuntimeWarning, match="Could not parse")
inconsistent = pytest.warns(RuntimeWarning, match="Inconsistent file information")
outside = pytest.warns(RuntimeWarning, match="outside the data")
omitted = pytest.warns(RuntimeWarning, match="Omitted 4 annot")


@testing.requires_testing_data
def test_old_data():
    """Test reading raw cnt files."""
    with _no_parse, pytest.raises(RuntimeError, match="number of bytes"):
        read_raw_cnt(input_fname=fname, eog="auto", misc=["NA1", "LEFT_EAR"])
    with _no_parse:
        raw = _test_raw_reader(
            read_raw_cnt,
            input_fname=fname,
            eog="auto",
            misc=["NA1", "LEFT_EAR"],
            data_format="int16",
        )

    # make sure we use annotations event if we synthesized stim
    assert len(raw.annotations) == 6

    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info["bads"] == ["LEFT_EAR", "VEOGR"]  # test bads

    # the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info["meas_date"] is None


@testing.requires_testing_data
def test_new_data():
    """Test reading raw cnt files with different header."""
    with inconsistent, outside, omitted:
        raw = read_raw_cnt(
            input_fname=fname_bad_spans, header="new", data_format="int32"
        )

    assert raw.info["bads"] == ["F8"]  # test bads


@testing.requires_testing_data
def test_auto_data():
    """Test reading raw cnt files with automatic header."""
    with inconsistent, outside, omitted:
        raw = read_raw_cnt(
            input_fname=fname_bad_spans, data_format="int16", verbose="debug"
        )
    # Test that responses are read properly
    assert "KeyPad Response 1" in raw.annotations.description
    assert raw.info["bads"] == ["F8"]

    with _no_parse:
        raw = _test_raw_reader(
            read_raw_cnt,
            input_fname=fname,
            eog="auto",
            misc=["NA1", "LEFT_EAR"],
            data_format="int16",
        )

    # make sure we use annotations event if we synthesized stim
    assert len(raw.annotations) == 6

    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info["bads"] == ["LEFT_EAR", "VEOGR"]  # test bads

    # the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info["meas_date"] is None


@testing.requires_testing_data
def test_compare_events_and_annotations():
    """Test comparing annotations and events."""
    with _no_parse:
        raw = read_raw_cnt(fname, data_format="int16")
    events = np.array(
        [[333, 0, 7], [1010, 0, 7], [1664, 0, 109], [2324, 0, 7], [2984, 0, 109]]
    )

    annot = read_annotations(fname, data_format="int16")
    assert len(annot) == 6
    assert_array_equal(annot.onset[:-1], events[:, 0] / raw.info["sfreq"])
    assert "STI 014" not in raw.info["ch_names"]


@testing.requires_testing_data
def test_reading_bytes():
    """Test reading raw cnt files with different header."""
    with _no_parse:
        raw_16 = read_raw_cnt(fname, preload=True, data_format="int16")
    with inconsistent, outside, omitted:
        raw_32 = read_raw_cnt(fname_bad_spans, preload=True, data_format="int32")

    # Verify that the number of bytes read is correct
    assert len(raw_16) == 3070
    assert len(raw_32) == 143765  # TODO: Used to be 90000! Need to eyeball


@testing.requires_testing_data
def test_bad_spans():
    """Test reading raw cnt files with bad spans."""
    with inconsistent:
        annot = read_annotations(fname_bad_spans, data_format="int32")
    temp = "\t".join(annot.description)
    assert "BAD" in temp
