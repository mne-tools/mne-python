# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import find_events
from mne._fiff.constants import FIFF
from mne._fiff.pick import _DATA_CH_TYPES_SPLIT
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_eyelink
from mne.io.eyelink._utils import _adjust_times, _find_overlaps
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _record_warnings

pd = pytest.importorskip("pandas")

MAPPING = {
    "left": ["xpos_left", "ypos_left", "pupil_left"],
    "right": ["xpos_right", "ypos_right", "pupil_right"],
    "both": [
        "xpos_left",
        "ypos_left",
        "pupil_left",
        "xpos_right",
        "ypos_right",
        "pupil_right",
    ],
}

testing_path = data_path(download=False)
fname = testing_path / "eyetrack" / "test_eyelink.asc"
fname_href = testing_path / "eyetrack" / "test_eyelink_HREF.asc"


def test_eyetrack_not_data_ch():
    """Eyetrack channels are not data channels."""
    msg = (
        "eyetrack channels are not data channels. Refer to MNE definition"
        " of data channels in the glossary section of the documentation."
    )
    assert "eyegaze" not in _DATA_CH_TYPES_SPLIT, msg
    assert "pupil" not in _DATA_CH_TYPES_SPLIT, msg


@requires_testing_data
@pytest.mark.parametrize(
    "fname, create_annotations, find_overlaps, apply_offsets",
    [
        (fname, False, False, False),
        (
            fname,
            False,
            False,
            False,
        ),
        (
            fname,
            True,
            False,
            False,
        ),
        (
            fname,
            True,
            True,
            True,
        ),
        (
            fname,
            ["fixations", "saccades", "blinks"],
            True,
            False,
        ),
    ],
)
def test_eyelink(fname, create_annotations, find_overlaps, apply_offsets):
    """Test reading eyelink asc files."""
    raw = read_raw_eyelink(
        fname,
        create_annotations=create_annotations,
        find_overlaps=find_overlaps,
        apply_offsets=apply_offsets,
    )

    # First, tests that shouldn't change based on function arguments
    assert raw.info["sfreq"] == 500  # True for this file
    assert raw.info["meas_date"].month == 3
    assert raw.info["meas_date"].day == 10
    assert raw.info["meas_date"].year == 2022

    assert len(raw.info["ch_names"]) == 6
    assert raw.info["chs"][0]["kind"] == FIFF.FIFFV_EYETRACK_CH
    assert raw.info["chs"][0]["coil_type"] == FIFF.FIFFV_COIL_EYETRACK_POS
    raw.info["chs"][2]["coil_type"] == FIFF.FIFFV_COIL_EYETRACK_PUPIL

    # x_left
    assert all(raw.info["chs"][0]["loc"][3:5] == [-1, -1])
    # pupil_left
    assert raw.info["chs"][2]["loc"][3] == -1
    assert np.isnan(raw.info["chs"][2]["loc"][4])
    # y_right
    assert all(raw.info["chs"][4]["loc"][3:5] == [1, 1])
    assert "RawEyelink" in repr(raw)

    # Test some annotation values for accuracy.
    if create_annotations is True and find_overlaps:
        orig = raw.info["meas_date"]
        df = raw.annotations.to_data_frame()
        # Convert annot onset datetimes to seconds, relative to orig_time
        df["time_in_sec"] = df["onset"].apply(
            lambda x: x.timestamp() - orig.timestamp()
        )
        # There is a blink in this data at 8.9 seconds
        cond = (df["time_in_sec"] > 8.899) & (df["time_in_sec"] < 8.95)
        assert df[cond]["description"].values[0].startswith("BAD_blink")

        # Check that the annotation ch_names are set correctly
        assert np.array_equal(raw.annotations[0]["ch_names"], MAPPING["both"])

    if isinstance(create_annotations, list) and find_overlaps:
        # the last pytest parametrize condition should hit this
        assert np.array_equal(raw.annotations[0]["ch_names"], MAPPING["both"])


@requires_testing_data
@pytest.mark.parametrize("fname_href", [(fname_href)])
def test_radian(fname_href):
    """Test converting HREF position data to radians."""
    with pytest.warns(RuntimeWarning, match="Annotations for"):
        raw = read_raw_eyelink(fname_href, create_annotations=["blinks"])
    # Test channel types
    assert raw.get_channel_types() == ["eyegaze", "eyegaze", "pupil"]

    # Test that eyegaze channels have a radian unit
    assert raw.info["chs"][0]["unit"] == FIFF.FIFF_UNIT_RAD
    assert raw.info["chs"][1]["unit"] == FIFF.FIFF_UNIT_RAD

    # Data in radians should range between -1 and 1
    # Test first channel (xpos_right)
    assert raw.get_data()[0].min() > -1
    assert raw.get_data()[0].max() < 1


@requires_testing_data
@pytest.mark.parametrize("fname", [(fname)])
def test_fill_times(fname):
    """Test use of pd.merge_asof in _fill_times.

    We are merging on floating
    point values. pd.merge_asof is used so that any differences in floating
    point precision between df['samples']['times'] and the times generated
    with np.arange don't result in the time columns not merging
    correctly - i.e. 1560687.0 and 1560687.000001 should merge.
    """
    raw = read_raw_eyelink(fname, create_annotations=False)
    sfreq = raw.info["sfreq"]
    # just take first 1000 points for testing
    df = raw.to_data_frame()[:1000]
    # even during blinks, pupil val is 0, so there should be no nans
    # in this column
    assert not df["pupil_left"].isna().sum()
    nan_count = df["pupil_left"].isna().sum()  # i.e 0
    df_merged = _adjust_times(df, sfreq)
    # If times dont merge correctly, there will be additional rows in
    # in df_merged with all nan values
    assert df_merged["pupil_left"].isna().sum() == nan_count  # i.e. 0


def test_find_overlaps():
    """Test finding overlapping ocular events between the left and right eyes.

    In the simulated blink df below, the first two rows
    will be considered an overlap because the diff() of both the 'time' and
    'end_time' values is <.05 (50ms). the 3rd and 4th rows will not be
    considered an overlap because the diff() of the 'time' values is > .05
    (4.20 - 4.14 = .06). The 5th and 6th rows will not be considered an
    overlap because they are both left eye events.
    """
    blink_df = pd.DataFrame(
        {
            "eye": ["L", "R", "L", "R", "L", "L"],
            "time": [0.01, 0.04, 4.14, 4.20, 6.50, 6.504],
            "end_time": [0.05, 0.08, 4.18, 4.22, 6.60, 6.604],
        }
    )
    overlap_df = _find_overlaps(blink_df)
    assert len(overlap_df["eye"].unique()) == 3  # ['both', 'left', 'right']
    assert len(overlap_df) == 5  # ['both', 'L', 'R', 'L', 'L']
    assert overlap_df["eye"].iloc[0] == "both"


@requires_testing_data
@pytest.mark.parametrize("fname", [fname])
def test_bino_to_mono(tmp_path, fname):
    """Test a file that switched from binocular to monocular mid-recording."""
    out_file = tmp_path / "tmp_eyelink.asc"
    in_file = Path(fname)
    lines = in_file.read_text("utf-8").splitlines()
    # We'll also add some binocular velocity data to increase our testing coverage.
    start_idx = [li for li, line in enumerate(lines) if line.startswith("START")][0]
    for li, line in enumerate(lines[start_idx:-2], start=start_idx):
        tokens = line.split("\t")
        event_type = tokens[0]
        if event_type == "SAMPLES":
            tokens.insert(3, "VEL")
            lines[li] = "\t".join(tokens)
        elif event_type.isnumeric():
            # fake velocity values for x/y left/right
            tokens[4:4] = ["999.1", "999.2", "999.3", "999.4"]
            lines[li] = "\t".join(tokens)
    end_line = lines[-2]
    end_ts = int(end_line.split("\t")[1])
    # Now only left eye data
    second_block = []
    new_ts = end_ts + 1
    info = [
        "GAZE",
        "LEFT",
        "VEL",
        "RATE",
        "500.00",
        "TRACKING",
        "CR",
        "FILTER",
        "2",
    ]
    start = ["START", f"{new_ts}", "LEFT", "SAMPLES", "EVENTS"]
    pupil = ["PUPIL", "DIAMETER"]
    samples = ["SAMPLES"] + info
    events = ["EVENTS"] + info
    second_block.append("\t".join(start) + "\n")
    second_block.append("\t".join(pupil) + "\n")
    second_block.append("\t".join(samples) + "\n")
    second_block.append("\t".join(events) + "\n")
    # Some fake data.. # x, y, pupil, velicty x/y status
    left = ["960", "540", "0.0", "999.1", "999.2", "..."]
    NUM_FAKE_SAMPLES = 4000
    for ii in range(NUM_FAKE_SAMPLES):
        ts = new_ts + ii
        tokens = [f"{ts}"] + left
        second_block.append("\t".join(tokens) + "\n")
    # interleave some events into the second block
    duration = 500
    blink_ts = new_ts + 500
    end_blink = ["EBLINK", "L", f"{blink_ts}", f"{blink_ts + 50}", "106"]
    fix_ts = new_ts + 1500
    end_fix = [
        "EFIX",
        "L",
        f"{fix_ts}",
        f"{fix_ts + duration}",
        "1616",
        "1025.1",
        "580.9",
        "1289",
    ]
    sacc_ts = new_ts + 2500
    end_sacc = [
        "ESACC",
        "L",
        f"{sacc_ts}",
        f"{sacc_ts + duration}",
        "52",
        "1029.6",
        "582.3",
        "581.7",
        "292.5",
        "10.30",
        "387",
    ]
    second_block.append("\t".join(end_blink) + "\n")
    second_block.append("\t".join(end_fix) + "\n")
    second_block.append("\t".join(end_sacc) + "\n")
    end_ts = ts + 1
    end_block = ["END", f"{end_ts}", "SAMPLES", "EVENTS", "RES", "45", "45"]
    second_block.append("\t".join(end_block))
    lines += second_block
    out_file.write_text("\n".join(lines), encoding="utf-8")

    with pytest.warns(
        RuntimeWarning, match="This recording switched between monocular and binocular"
    ):
        raw = read_raw_eyelink(out_file)
    want_channels = [
        "xpos_left",
        "ypos_left",
        "pupil_left",
        "xpos_right",
        "ypos_right",
        "pupil_right",
        "xvel_left",
        "yvel_left",
        "xvel_right",
        "yvel_right",
    ]
    assert len(set(raw.info["ch_names"]).difference(set(want_channels))) == 0


def _simulate_eye_tracking_data(in_file, out_file):
    out_file = Path(out_file)

    new_samples_line = (
        "SAMPLES\tPUPIL\tLEFT\tVEL\tRES\tHTARGET\tRATE\t1000.00"
        "\tTRACKING\tCR\tFILTER\t2\tINPUT"
    )

    # Define your known BUTTON events
    button_events = [
        (7453390, 1, 1),
        (7453410, 1, 0),
        (7453420, 1, 1),
        (7453430, 1, 0),
        (7453440, 1, 1),
        (7453450, 1, 0),
    ]
    button_idx = 0

    with out_file.open("w") as fp:
        in_recording_block = False
        events = []

        for line in Path(in_file).read_text().splitlines():
            if line.startswith("START"):
                in_recording_block = True
            if in_recording_block:
                tokens = line.split()
                event_type = tokens[0]
                if event_type.isnumeric():  # samples
                    tokens[4:4] = ["100", "20", "45", "45", "127.0"]  # vel, res, DIN
                    tokens.extend(["1497.0", "5189.0", "512.5", "............."])
                elif event_type in ("EFIX", "ESACC"):
                    if event_type == "ESACC":
                        tokens[5:7] = [".", "."]  # pretend start pos is unknown
                    tokens.extend(["45", "45"])  # resolution
                elif event_type == "SAMPLES":
                    tokens[1] = "PUPIL"  # simulate raw coordinate data
                    tokens[3:3] = ["VEL", "RES", "HTARGET"]
                    tokens.append("INPUT")
                elif event_type == "EBLINK":
                    continue  # simulate no blink events

                elif event_type == "END":
                    pass
                else:
                    fp.write(f"{line}\n")
                    continue
                events.append("\t".join(tokens))
                if event_type == "END":
                    fp.write("\n".join(events) + "\n")
                    events.clear()
                    in_recording_block = False
            else:
                fp.write(f"{line}\n")

        fp.write("START\t7452389\tRIGHT\tSAMPLES\tEVENTS\n")
        fp.write(f"{new_samples_line}\n")

        # simulate a second block that starts with empty samples
        for timestamp in np.arange(7452389, 7453154):
            # samples without last status column
            # see https://github.com/mne-tools/mne-python/issues/13567
            fp.write(f"{timestamp}\t.\t.\t0.0\t.\t.\t.\t.\t0.0\t...\t.\t.\t.\n")

        for timestamp in np.arange(7453154, 7453390):  # second block continues
            fp.write(
                f"{timestamp}\t-2434.0\t-1760.0\t840.0\t100\t20\t45\t45\t127.0\t"
                "...\t1497\t5189\t512.5\t.............\n"
            )

        for timestamp in np.arange(7453390, 7453490):  # 100 samples 7453100
            fp.write(
                f"{timestamp}\t-2434.0\t-1760.0\t840.0\t100\t20\t45\t45\t127.0\t"
                "...\t1497\t5189\t512.5\t.............\n"
            )
            # Check and insert button events at this timestamp
            if (
                button_idx < len(button_events)
                and button_events[button_idx][0] == timestamp
            ):
                t, btn_id, state = button_events[button_idx]
                fp.write(
                    f"BUTTON\t{t}\t{btn_id}\t{state}\t100\t20\t45\t45\t127.0\t"
                    "1497.0\t5189.0\t512.5\t.............\n"
                )
                button_idx += 1
        fp.write("END\t7453390\tRIGHT\tSAMPLES\tEVENTS\n")

        # simulate a third block with no samples at all
        fp.write("START\t7454390\tRIGHT\tSAMPLES\tEVENTS\n")
        fp.write(f"{new_samples_line}\n")
        for timestamp in np.arange(7454390, 7454990):
            fp.write(
                f"{timestamp}\t.\t.\t0.0\t.\t.\t.\t.\t0.0\t...\t.\t.\t.\n"  # All empty
            )
        fp.write("END\t7454990\tRIGHT\tSAMPLES\tEVENTS\n")


@requires_testing_data
@pytest.mark.parametrize("fname", [fname_href])
def test_multi_block_misc_channels(fname, tmp_path):
    """Test a file with many edge cases.

    This file has multiple acquisition blocks, each tracking a different eye.
    The coordinates are in raw units (not pixels or radians).
    It has some misc channels (head position, saccade velocity, etc.)
    """
    out_file = tmp_path / "tmp_eyelink.asc"
    _simulate_eye_tracking_data(fname, out_file)

    with (
        _record_warnings(),
        pytest.warns(RuntimeWarning, match="Raw pupil position data detected"),
        pytest.warns(RuntimeWarning, match="The eye being tracked changed"),
    ):
        raw = read_raw_eyelink(out_file, apply_offsets=True)

    chs_in_file = [
        "xpos_right",
        "ypos_right",
        "pupil_right",
        "xvel_right",
        "yvel_right",
        "xres",
        "yres",
        "DIN",
        "x_head",
        "y_head",
        "distance",
        "xpos_left",
        "ypos_left",
        "pupil_left",
        "xvel_left",
        "yvel_left",
    ]

    assert raw.ch_names == chs_in_file
    assert raw.annotations.description[1] == "SYNCTIME"

    assert raw.annotations.description[-8] == "BAD_ACQ_SKIP"
    assert np.isclose(raw.annotations.onset[-8], 1.001)
    assert np.isclose(raw.annotations.duration[-8], 0.1)

    data, times = raw.get_data(return_times=True)
    assert not np.isnan(data[0, np.where(times < 1)[0]]).any()
    assert np.isnan(data[0, np.logical_and(times > 1, times <= 1.1)]).all()

    assert raw.annotations.description[-7] == "button_1_press"
    button_idx = [
        ii
        for ii, desc in enumerate(raw.annotations.description)
        if "button" in desc.lower()
    ]
    assert len(button_idx) == 6
    assert_allclose(raw.annotations.onset[button_idx[0]], 2.102, atol=1e-3)

    # smoke test for reading events with missing samples (should not emit a warning)
    find_events(raw, verbose=True)


@requires_testing_data
@pytest.mark.parametrize("this_fname", (fname, fname_href))
def test_basics(this_fname):
    """Test basics of reading."""
    _test_raw_reader(read_raw_eyelink, fname=this_fname, test_preloading=False)


@requires_testing_data
def test_annotations_without_offset(tmp_path):
    """Test read of annotations without offset."""
    out_file = tmp_path / "tmp_eyelink.asc"

    # create fake dataset
    with open(fname_href) as file:
        lines = file.readlines()
    ts = lines[-3].split("\t")[0]
    line = f"MSG\t{ts} test string\n"
    lines = lines[:-3] + [line] + lines[-3:]
    with open(out_file, "w") as file:
        file.writelines(lines)

    raw = read_raw_eyelink(out_file, apply_offsets=False)
    assert raw.annotations[-1]["description"] == "test string"
    onset1 = raw.annotations[-1]["onset"]
    assert raw.annotations[1]["description"] == "-2 SYNCTIME"
    onset2 = raw.annotations[1]["onset"]

    raw = read_raw_eyelink(out_file, apply_offsets=True)
    assert raw.annotations[-1]["description"] == "test string"
    assert raw.annotations[1]["description"] == "SYNCTIME"
    assert_allclose(raw.annotations[-1]["onset"], onset1)
    assert_allclose(raw.annotations[1]["onset"], onset2 - 2 / raw.info["sfreq"])


@requires_testing_data
def test_no_datetime(tmp_path):
    """Test reading a file with no datetime."""
    out_file = tmp_path / "tmp_eyelink.asc"
    with open(fname) as file:
        lines = file.readlines()
    # remove the timestamp from the datetime line
    lines[1] = lines[1].split(":")[0] + ":"
    with open(out_file, "w") as file:
        file.writelines(lines)
    raw = read_raw_eyelink(out_file)
    assert raw.info["meas_date"] is None
    # Sanity check that a None meas_date doesn't change annotation times
    # First annotation in this file is a fixation at 0.004 seconds
    np.testing.assert_allclose(raw.annotations.onset[0], 0.004)


@requires_testing_data
def test_href_eye_events(tmp_path):
    """Test Parsing file where Eye Event Data option was set to 'HREF'."""
    out_file = tmp_path / "tmp_eyelink.asc"
    lines = fname_href.read_text("utf-8").splitlines()

    for li, line in enumerate(lines):
        if not line.startswith(("ESACC", "EFIX")):
            continue
        tokens = line.split()
        if line.startswith("ESACC"):
            href_sacc_vals = ["9999", "9999", "9999", "9999", "99.99", "999"]
            tokens[5:5] = href_sacc_vals  # add href saccade values
        elif line.startswith("EFIX"):
            tokens = line.split()
            href_fix_vals = ["9999.9", "9999.9", "999"]
            tokens[5:3] = href_fix_vals
        new_line = "\t".join(tokens) + "\n"
        lines[li] = new_line
    out_file.write_text("\n".join(lines), encoding="utf-8")

    raw = read_raw_eyelink(out_file)
    # Just check that we actually parsed the Saccade and Fixation events
    assert "saccade" in raw.annotations.description
    assert "fixation" in raw.annotations.description
