# Authors: Scott Huberty <seh33@uw.edu>
#
# License: Simplified BSD

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mne


@pytest.mark.parametrize("axes", [None, True])
def test_plot_heatmap(axes):
    """Test plot_gaze."""
    # Create a toy epochs instance
    info = info = mne.create_info(
        ch_names=["xpos", "ypos"], sfreq=100, ch_types="eyegaze"
    )
    # simulate a steady fixation at the center of the screen
    width, height = (1920, 1080)
    shape = (1, 100)  # x or y, time
    data = np.vstack([np.full(shape, width / 2), np.full(shape, height / 2)])
    epochs = mne.EpochsArray(data[None, ...], info)
    epochs.info["chs"][0]["loc"][4] = -1
    epochs.info["chs"][1]["loc"][4] = 1

    if axes:
        axes = plt.subplot()
    fig = mne.viz.eyetracking.plot_gaze(
        epochs, width=width, height=height, axes=axes, cmap="Greys", sigma=None
    )
    img = fig.axes[0].images[0].get_array()
    # We simulated a 2D histogram where only the central pixel (960, 540) was active
    assert img.T[width // 2, height // 2] == 1  # central pixel is active
    assert np.sum(img) == 1  # only the central pixel should be active
