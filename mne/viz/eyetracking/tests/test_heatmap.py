import pytest

import matplotlib.pyplot as plt
import numpy as np

import mne


@pytest.mark.parametrize(
    "axes",
    [(None), (True)],
)
def test_plot_heatmap(axes):
    """Test plot_gaze."""
    # Create a toy epochs instance
    info = info = mne.create_info(
        ch_names=["xpos", "ypos"], sfreq=100, ch_types="eyegaze"
    )
    # here we pretend that the subject was looking at the center of the screen
    # we limit the gaze data between 860-1060px horizontally and 440-640px vertically
    data = np.vstack([np.full((1, 100), 1920 / 2), np.full((1, 100), 1080 / 2)])
    epochs = mne.EpochsArray(data[None, ...], info)
    epochs.info["chs"][0]["loc"][4] = -1
    epochs.info["chs"][1]["loc"][4] = 1

    if axes:
        axes = plt.subplot()
    fig = mne.viz.eyetracking.plot_gaze(
        epochs, width=1920, height=1080, axes=axes, cmap="Greys"
    )
    img = fig.axes[0].images[0].get_array()
    # the pixels in the center of canvas
    assert 960 in np.where(img)[1]
    assert np.isclose(np.min(np.where(img)[1]), 860)
    assert np.isclose(np.max(np.where(img)[1]), 1060)
    assert 540 in np.where(img)[0]
    assert np.isclose(np.min(np.where(img)[0]), 440)
    assert np.isclose(np.max(np.where(img)[0]), 640)
