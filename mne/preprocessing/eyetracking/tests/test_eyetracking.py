# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mne
from mne._fiff.constants import FIFF
from mne.utils import _record_warnings


def test_set_channel_types_eyetrack(eyetrack_raw):
    """Test that set_channel_types_eyetrack worked on the fixture."""
    assert eyetrack_raw.info["chs"][0]["kind"] == FIFF.FIFFV_EYETRACK_CH
    assert eyetrack_raw.info["chs"][1]["coil_type"] == FIFF.FIFFV_COIL_EYETRACK_POS
    assert eyetrack_raw.info["chs"][0]["unit"] == FIFF.FIFF_UNIT_PX
    assert eyetrack_raw.info["chs"][2]["unit"] == FIFF.FIFF_UNIT_NONE


def test_convert_units(eyetrack_raw, eyetrack_cal):
    """Test unit conversion."""
    raw, cal = eyetrack_raw, eyetrack_cal  # shorter names

    # roundtrip conversion should be identical to original data
    data_orig = raw.get_data(picks=[0])  # take the first x-coord channel
    mne.preprocessing.eyetracking.convert_units(raw, cal, "radians")
    assert raw.info["chs"][0]["unit"] == FIFF.FIFF_UNIT_RAD
    # Gaze was to center of screen, so x-coord and y-coord should now be 0 radians
    assert_allclose(raw.get_data(picks=[0, 1]), 0)

    # Should raise an error if we try to convert to radians again
    with pytest.raises(ValueError, match="Data must be in"):
        mne.preprocessing.eyetracking.convert_units(raw, cal, "radians")

    # Convert back to pixels
    mne.preprocessing.eyetracking.convert_units(raw, cal, "pixels")
    assert raw.info["chs"][1]["unit"] == FIFF.FIFF_UNIT_PX
    data_new = raw.get_data(picks=[0])
    assert_allclose(data_orig, data_new)

    # Should raise an error if we try to convert to pixels again
    with pytest.raises(ValueError, match="Data must be in"):
        mne.preprocessing.eyetracking.convert_units(raw, cal, "pixels")

    # Finally, check that we raise other errors or warnings when we should
    # warn if no eyegaze channels found
    raw_misc = raw.copy()
    with _record_warnings():  # channel units change warning
        raw_misc.set_channel_types({ch: "misc" for ch in raw_misc.ch_names})
    with pytest.warns(UserWarning, match="Could not"):
        mne.preprocessing.eyetracking.convert_units(raw_misc, cal, "radians")

    # raise an error if the calibration is missing a key
    bad_cal = cal.copy()
    bad_cal.pop("screen_size")
    bad_cal["screen_distance"] = None
    with pytest.raises(KeyError, match="Calibration object must have the following"):
        mne.preprocessing.eyetracking.convert_units(raw, bad_cal, "radians")

    # warn if visual angle is too large
    cal_tmp = cal.copy()
    cal_tmp["screen_distance"] = 0.1
    raw_tmp = raw.copy()
    raw_tmp._data[0, :10] = 1900  # gaze to extremity of screen
    with pytest.warns(UserWarning, match="Some visual angle values"):
        mne.preprocessing.eyetracking.convert_units(raw_tmp, cal_tmp, "radians")

    # raise an error if channel locations not set
    raw_missing = raw.copy()
    raw_missing.info["chs"][0]["loc"] = np.zeros(12)
    with pytest.raises(ValueError, match="loc array not set"):
        mne.preprocessing.eyetracking.convert_units(raw_missing, cal, "radians")


def test_get_screen_visual_angle(eyetrack_cal):
    """Test calculating the radians of visual angle for a screen."""
    # Our toy calibration should subtend .56 x .32 radians i.e 31.5 x 18.26 degrees
    viz_angle = mne.preprocessing.eyetracking.get_screen_visual_angle(eyetrack_cal)
    assert viz_angle.shape == (2,)
    np.testing.assert_allclose(np.round(viz_angle, 2), (0.56, 0.32))
