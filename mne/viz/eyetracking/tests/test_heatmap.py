import matplotlib.pyplot as plt
import pytest

from mne import make_fixed_length_epochs
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_eyelink
from mne.preprocessing.eyetracking import interpolate_blinks
from mne.utils import requires_pandas
from mne.viz.eyetracking import plot_gaze

fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@requires_testing_data
@requires_pandas
@pytest.mark.parametrize(
    "fname, axes",
    [(fname, None), (fname, True)],
)
def test_plot_heatmap(fname, axes):
    """Test plot_gaze."""
    raw = read_raw_eyelink(fname, find_overlaps=True)
    interpolate_blinks(raw, interpolate_gaze=True, buffer=(0.05, 0.2))
    epochs = make_fixed_length_epochs(raw, duration=5)

    if axes:
        axes = plt.subplot()
    plot_gaze(epochs, width=1920, height=1080, axes=axes)
