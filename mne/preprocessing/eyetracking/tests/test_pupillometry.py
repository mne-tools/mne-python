# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne import Annotations, create_info
from mne.annotations import _annotations_starts_stops
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import RawArray, read_raw_eyelink
from mne.preprocessing.eyetracking import interpolate_blinks

fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"
pytest.importorskip("pandas")


@requires_testing_data
@pytest.mark.parametrize(
    "buffer, match, cause_error, interpolate_gaze, crop",
    [
        (0.025, "BAD_blink", False, False, False),
        (0.025, "BAD_blink", False, True, True),
        ((0.025, 0.025), ["random_annot"], False, False, False),
        (0.025, "BAD_blink", True, False, False),
    ],
)
def test_interpolate_blinks(buffer, match, cause_error, interpolate_gaze, crop):
    """Test interpolating pupil data during blinks."""
    raw = read_raw_eyelink(fname, create_annotations=["blinks"], find_overlaps=True)
    if crop:
        raw.crop(tmin=2)
        assert raw.first_time == 2.0
    # Create a dummy stim channel
    # this will hit a certain line in the interpolate_blinks function
    info = create_info(["STI"], raw.info["sfreq"], ["stim"])
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)

    # Get the indices of the first blink
    blink_starts, blink_ends = _annotations_starts_stops(raw, "BAD_blink")
    blink_starts = np.divide(blink_starts, raw.info["sfreq"])
    blink_ends = np.divide(blink_ends, raw.info["sfreq"])
    first_blink_start = blink_starts[0]
    first_blink_end = blink_ends[0]
    if match == ["random_annot"]:
        msg = "No annotations matching"
        with pytest.warns(RuntimeWarning, match=msg):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return

    if cause_error:
        # Make an annotation without ch_names info
        raw.annotations.append(onset=1, duration=1, description="BAD_blink")
        with pytest.raises(ValueError):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return
    else:
        interpolate_blinks(
            raw, buffer=buffer, match=match, interpolate_gaze=interpolate_gaze
        )

    # Now get the data and check that the blinks are interpolated
    data, times = raw.get_data(return_times=True)
    # Get the indices of the first blink
    blink_ind = np.where((times >= first_blink_start) & (times <= first_blink_end))[0]
    # pupil data during blinks are zero, check that interpolated data are not zeros
    assert not np.any(data[2, blink_ind] == 0)  # left eye
    assert not np.any(data[5, blink_ind] == 0)  # right eye
    if interpolate_gaze:
        assert not np.isnan(data[0, blink_ind]).any()  # left eye
        assert not np.isnan(data[1, blink_ind]).any()  # right eye


def _pupil_raw_with_gaps():
    """Synthetic pupil Raw with two missing segments at known sample ranges."""
    info = create_info(["pupil_left", "pupil_right"], 100.0, ["pupil", "pupil"])
    data = np.ones((2, 1000)) * 5.0
    data[:, 100:150] = 0.0  # a blink
    data[:, 600:650] = 0.0  # a non-blink gap (e.g. NaN/dropout)
    raw = RawArray(data, info)
    raw.set_annotations(
        Annotations(
            onset=[1.0, 6.0],
            duration=[0.5, 0.5],
            description=["BAD_blink", "BAD_NAN"],
            ch_names=[("pupil_left", "pupil_right"), ("pupil_left", "pupil_right")],
        )
    )
    return raw


def test_interpolate_blinks_respects_match():
    """All descriptions in ``match`` are interpolated, not just ``BAD_blink``.

    Regression test for #13880: ``interpolate_blinks`` exposes a ``match``
    parameter, but the segment start/stop times were derived from a hardcoded
    ``"BAD_blink"`` query, so any other matched annotation (e.g. ``BAD_NAN``
    from :func:`mne.preprocessing.annotate_nan`) was silently ignored.
    """
    # default match="BAD_blink": only the blink gap is filled, the other remains
    raw = _pupil_raw_with_gaps()
    interpolate_blinks(raw, buffer=(0.0, 0.0))
    data = raw.get_data()
    assert not np.any(data[:, 100:150] == 0.0)  # blink interpolated
    assert np.all(data[:, 600:650] == 0.0)  # unmatched gap untouched

    # match both descriptions: both gaps are interpolated
    raw = _pupil_raw_with_gaps()
    interpolate_blinks(raw, buffer=(0.0, 0.0), match=["BAD_blink", "BAD_NAN"])
    data = raw.get_data()
    assert not np.any(data[:, 100:150] == 0.0)
    assert not np.any(data[:, 600:650] == 0.0)


def test_interpolate_blinks_synthetic():
    """Test interpolate_blinks with synthetic data across channels and buffers."""
    info = create_info(
        ["pupil_left", "pupil_right", "gaze_x"],
        100.0,
        ["pupil", "pupil", "eyegaze"],
    )
    data = np.ones((3, 1000)) * 10.0
    data[0, 100:150] = 0.0
    data[1, 100:150] = 0.0
    data[2, 100:150] = 0.0
    raw = RawArray(data, info)
    raw.set_annotations(
        Annotations(
            onset=[1.0],
            duration=[0.5],
            description=["BAD_blink"],
            ch_names=[("pupil_left", "pupil_right", "gaze_x")],
        )
    )

    # 1. By default, only pupil channels are interpolated
    raw_pupil = raw.copy()
    interpolate_blinks(raw_pupil, buffer=0.05, interpolate_gaze=False)
    d = raw_pupil.get_data()
    assert not np.any(d[0, 95:155] == 0.0)
    assert not np.any(d[1, 95:155] == 0.0)
    assert np.all(d[2, 100:150] == 0.0)

    # 2. With interpolate_gaze=True, gaze is also interpolated
    raw_gaze = raw.copy()
    interpolate_blinks(raw_gaze, buffer=0.05, interpolate_gaze=True)
    d_gaze = raw_gaze.get_data()
    assert not np.any(d_gaze[2, 95:155] == 0.0)

    # 3. Channel selectivity: annotation for only pupil_left
    raw_left = raw.copy()
    raw_left.set_annotations(
        Annotations(
            onset=[1.0],
            duration=[0.5],
            description=["BAD_blink"],
            ch_names=[("pupil_left",)],
        )
    )
    interpolate_blinks(raw_left, buffer=(0.0, 0.0))
    d_left = raw_left.get_data()
    assert not np.any(d_left[0, 100:150] == 0.0)
    assert np.all(d_left[1, 100:150] == 0.0)
