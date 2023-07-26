import pytest

import numpy as np

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_eyelink
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.constants import FIFF
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.utils import _check_pandas_installed, requires_pandas

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
@requires_pandas
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
@requires_pandas
@pytest.mark.parametrize("fname_href", [(fname_href)])
def test_radian(fname_href):
    """Test converting HREF position data to radians."""
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
@requires_pandas
@pytest.mark.parametrize("fname", [(fname)])
def test_fill_times(fname):
    """Test use of pd.merge_asof in _fill_times.

    We are merging on floating
    point values. pd.merge_asof is used so that any differences in floating
    point precision between df['samples']['times'] and the times generated
    with np.arange don't result in the time columns not merging
    correctly - i.e. 1560687.0 and 1560687.000001 should merge.
    """
    from ..eyelink import _fill_times

    raw = read_raw_eyelink(fname, create_annotations=False)
    sfreq = raw.info["sfreq"]
    # just take first 1000 points for testing
    df = raw.dataframes["samples"].iloc[:1000].reset_index(drop=True)
    # even during blinks, pupil val is 0, so there should be no nans
    # in this column
    assert not df["pupil_left"].isna().sum()
    nan_count = df["pupil_left"].isna().sum()  # i.e 0
    df_merged = _fill_times(df, sfreq)
    # If times dont merge correctly, there will be additional rows in
    # in df_merged with all nan values
    assert df_merged["pupil_left"].isna().sum() == nan_count  # i.e. 0


@requires_pandas
def test_find_overlaps():
    """Test finding overlapping occular events between the left and right eyes.

    In the simulated blink df below, the first two rows
    will be considered an overlap because the diff() of both the 'time' and
    'end_time' values is <.05 (50ms). the 3rd and 4th rows will not be
    considered an overlap because the diff() of the 'time' values is > .05
    (4.20 - 4.14 = .06). The 5th and 6th rows will not be considered an
    overlap because they are both left eye events.
    """
    from ..eyelink import _find_overlaps

    pd = _check_pandas_installed()
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


@pytest.mark.xfail(reason="Attributes and test_preloading fail")
@requires_testing_data
@pytest.mark.parametrize("this_fname", (fname, fname_href))
def test_basics(this_fname):
    """Test basics of reading."""
    _test_raw_reader(read_raw_eyelink, fname=this_fname)
