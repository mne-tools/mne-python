# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import matplotlib.pyplot as plt
import pytest

import mne
from mne._fiff.constants import FIFF


@pytest.mark.parametrize("axes, unit", [(None, "px"), (True, "rad")])
def test_plot_heatmap(eyetrack_raw, eyetrack_cal, axes, unit):
    """Test plot_gaze."""
    epochs = mne.make_fixed_length_epochs(eyetrack_raw, duration=1.0)
    epochs.load_data()
    width, height = eyetrack_cal["screen_resolution"]  # 1920, 1080
    if unit == "rad":
        mne.preprocessing.eyetracking.convert_units(epochs, eyetrack_cal, to="radians")

    if axes:
        axes = plt.subplot()

    # First check that we raise errors when we should
    with pytest.raises(ValueError, match="If no calibration is provided"):
        mne.viz.eyetracking.plot_gaze(epochs)

    with pytest.raises(ValueError, match="If a calibration is provided"):
        mne.viz.eyetracking.plot_gaze(
            epochs, width=width, height=height, calibration=eyetrack_cal
        )

    with pytest.raises(ValueError, match="Invalid unit"):
        ep_bad = epochs.copy()
        ep_bad.info["chs"][0]["unit"] = FIFF.FIFF_UNIT_NONE
        mne.viz.eyetracking.plot_gaze(ep_bad, calibration=eyetrack_cal)

    # raise an error if no calibration object is provided for radian data
    if unit == "rad":
        with pytest.raises(ValueError, match="If gaze data are in Radians"):
            mne.viz.eyetracking.plot_gaze(epochs, axes=axes, width=1, height=1)

    # Now check that we get the expected output
    if unit == "px":
        fig = mne.viz.eyetracking.plot_gaze(
            epochs, width=width, height=height, axes=axes, cmap="Greys", sigma=None
        )
    elif unit == "rad":
        fig = mne.viz.eyetracking.plot_gaze(
            epochs,
            calibration=eyetrack_cal,
            axes=axes,
            cmap="Greys",
            sigma=None,
        )
    img = fig.axes[0].images[0].get_array()
    # We simulated a 2D histogram where only the central pixel (960, 540) was active
    # so regardless of the unit, we should have a heatmap with the central bin active
    assert img.T[width // 2, height // 2] == 1
