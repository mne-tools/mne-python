# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD

import numpy as np
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_eyelink
from mne.preprocessing.eyetracking import interpolate_blinks
from mne.utils import requires_pandas


fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@requires_testing_data
@requires_pandas
@pytest.mark.xfail(reason="This will fail until PR 11746 is merged")
@pytest.mark.parametrize(
    "buffer, match",
    [(0.025, "BAD_blink"), ((0.025, 0.025), ["BAD_blink", "blink_manual"])],
)
def test_interpolate_blinks(buffer, match):
    """Test interpolating pupil data during blinks."""
    raw = read_raw_eyelink(
        fname, preload=True, create_annotations=["blinks"], find_overlaps=True
    )
    first_blink_start = raw.annotations[0]["onset"]
    first_blink_end = raw.annotations[0]["onset"] + raw.annotations[0]["duration"]
    interpolate_blinks(raw, buffer=buffer, match=match)

    # Now get the data and check that the blinks are interpolated
    data, times = raw.get_data(return_times=True)
    # Get the indices of the first blink
    first_blink_indices = np.where(
        np.logical_and((times > first_blink_start), (times < first_blink_end))
    )[0]
    # pupil data during blinks are zero, check that interpolated data are not zeros
    assert not np.any(data[2, first_blink_indices] == 0)  # left eye
    assert not np.any(data[5, first_blink_indices] == 0)  # right eye
