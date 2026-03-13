# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne import create_info
from mne.annotations import _annotations_starts_stops
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import RawArray, read_raw_eyelink
from mne.preprocessing.eyetracking import interpolate_blinks

fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"
pytest.importorskip("pandas")


@requires_testing_data
@pytest.mark.parametrize(
    "buffer, description_prefix, match, cause_error, interpolate_gaze, crop",
    [
        (0.025, "BAD_", "BAD_blink", False, False, False),
        (0.025, "BAD_", "BAD_blink", False, True, True),
        (0.025, "BAD_", "BAD_blink", True, False, False),
        ((0.025, 0.025), "BAD_", ["random_annot"], False, False, False),
        (0.025, "", "blink", False, False, False),
        (0.025, "", "blink", False, True, True),
        (0.025, "", "blink", True, False, False),
    ],
)
def test_interpolate_bad_blinks(
    description_prefix, buffer, match, cause_error, interpolate_gaze, crop
):
    """Test interpolating pupil data during blinks."""
    raw = read_raw_eyelink(fname, create_annotations=["blinks"], find_overlaps=True)

    # read_raw_eyelink prefixes any blink description with "BAD_" but we want to
    # test interpolate_blinks does not depend on this convention
    if description_prefix != "BAD_":
        blink_description = f"{description_prefix}blink"
        raw.annotations.rename({"BAD_blink": blink_description})
    else:
        blink_description = "BAD_blink"

    # we add a set of events with a description starting as the blink annotations as
    # well in case interpolate_blinks picks them up. If so, the test is expected to
    # fail.
    blinking_light_description = f"{description_prefix}blinking_light"
    # the light switches on every second for 1/10th second.
    blinking_lights_onsets = raw.times[:: int(raw.info["sfreq"])]
    blinking_lights_durations = np.full_like(
        blinking_lights_onsets,
        0.1,
    )
    raw.annotations.append(
        blinking_lights_onsets,
        blinking_lights_durations,
        blinking_light_description,
        [raw.ch_names] * blinking_lights_onsets.size,
    )

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
    blink_starts, blink_ends = _annotations_starts_stops(raw, blink_description)
    blink_starts = np.divide(blink_starts, raw.info["sfreq"])
    blink_ends = np.divide(blink_ends, raw.info["sfreq"])
    if match == ["random_annot"]:
        msg = "No annotations matching"
        with pytest.warns(RuntimeWarning, match=msg):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return

    if cause_error:
        # Make an annotation without ch_names info
        raw.annotations.append(onset=1, duration=1, description=blink_description)
        with pytest.raises(ValueError):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return
    else:
        interpolate_blinks(
            raw, buffer=buffer, match=match, interpolate_gaze=interpolate_gaze
        )

    # Now get the data and check that the blinks are interpolated
    data, times = raw.get_data(return_times=True)
    for blink_start, blink_end in zip(blink_starts, blink_ends):
        # Get the indices of the first blink
        blink_ind = np.where((times >= blink_start) & (times <= blink_end))[0]
        # pupil data during blinks are zero, check that interpolated data are not zeros
        assert not np.any(data[2, blink_ind] == 0)  # left eye
        assert not np.any(data[5, blink_ind] == 0)  # right eye
        if interpolate_gaze:
            assert not np.isnan(data[0, blink_ind]).any()  # left eye
            assert not np.isnan(data[1, blink_ind]).any()  # right eye
