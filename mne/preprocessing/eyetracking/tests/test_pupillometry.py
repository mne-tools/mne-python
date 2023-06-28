# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD

import numpy as np
import pytest

from mne import create_info
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_eyelink, RawArray
from mne.preprocessing.eyetracking import interpolate_blinks
from mne.utils import requires_pandas


fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@requires_testing_data
@requires_pandas
@pytest.mark.parametrize(
    "buffer, match, cause_error, interpolate_gaze",
    [
        (0.025, "BAD_blink", False, False),
        (0.025, "BAD_blink", False, True),
        ((0.025, 0.025), ["random_annot"], False, False),
        (0.025, "BAD_blink", True, False),
    ],
)
def test_interpolate_blinks(buffer, match, cause_error, interpolate_gaze):
    """Test interpolating pupil data during blinks."""
    raw = read_raw_eyelink(
        fname, preload=True, create_annotations=["blinks"], find_overlaps=True
    )
    # Create a dummy stim channel
    # this will hit a certain line in the interpolate_blinks function
    info = create_info(["STI"], raw.info["sfreq"], ["stim"])
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)

    # Get the indices of the first blink
    first_blink_start = raw.annotations[0]["onset"]
    first_blink_end = raw.annotations[0]["onset"] + raw.annotations[0]["duration"]
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
