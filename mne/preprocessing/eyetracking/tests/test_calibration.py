import pytest

import numpy as np

from mne.datasets.testing import data_path, requires_testing_data
from ..calibration import Calibration, read_eyelink_calibration

# for test_read_eylink_calibration
testing_path = data_path(download=False)
fname = testing_path / "eyetrack" / "test_eyelink.asc"

# for test_create_calibration
test_points = [
    (115.0, 540.0, 0.42, 101.5, 554.8),
    (960.0, 540.0, 0.23, 9.9, -4.1),
    (1804.0, 540.0, 0.17, 1795.9, 539.0),
]
field_names = ["point_x", "point_y", "offset", "gaze_x", "gaze_y"]
dtypes = [(name, float) for name in field_names]
test_structured = np.array(test_points, dtype=dtypes)
test_lists = [list(point) for point in test_points]
test_array_2d = np.array(test_lists)

expected_repr = (
    "Calibration |\n"
    "  onset: 0 seconds\n"
    "  model: H3\n"
    "  eye: right\n"
    "  average error: 0.5 degrees\n"
    "  max error: 1.0 degrees\n"
    "  screen size: (0.531, 0.298) meters\n"
    "  screen distance: 0.065 meters\n"
    "  screen resolution: (1920, 1080) pixels\n"
)


@pytest.mark.parametrize(
    (
        "onset, model, eye, avg_error, max_error, points, screen_size, screen_distance,"
        " screen_resolution"
    ),
    [
        (0, "H3", "right", 0.5, 1.0, test_points, (0.531, 0.298), 0.065, (1920, 1080)),
        (
            0,
            "H3",
            "right",
            0.5,
            1.0,
            test_structured,
            (0.531, 0.298),
            0.065,
            (1920, 1080),
        ),
        (0, "H3", "right", 0.5, 1.0, test_lists, (0.531, 0.298), 0.065, (1920, 1080)),
        (
            0,
            "H3",
            "right",
            0.5,
            1.0,
            test_array_2d,
            (0.531, 0.298),
            0.065,
            (1920, 1080),
        ),
        (None, None, None, None, None, None, None, None, None),
    ],
)
def test_create_calibration(
    onset,
    model,
    eye,
    avg_error,
    max_error,
    points,
    screen_size,
    screen_distance,
    screen_resolution,
):
    """Test creating a Calibration object."""
    kwargs = dict(
        onset=onset,
        model=model,
        eye=eye,
        avg_error=avg_error,
        max_error=max_error,
        points=points,
        screen_size=screen_size,
        screen_distance=screen_distance,
        screen_resolution=screen_resolution,
    )
    cal = Calibration(**kwargs)
    assert cal["onset"] == onset
    assert cal["model"] == model
    assert cal["eye"] == eye
    assert cal["avg_error"] == avg_error
    assert cal["max_error"] == max_error
    if points is not None:
        assert np.array_equal(cal["points"], test_structured)
    else:
        assert cal["points"] is None
    assert cal["screen_size"] == screen_size
    assert cal["screen_distance"] == screen_distance
    assert cal["screen_resolution"] == screen_resolution
    # test __getattr__
    assert cal.onset == cal["onset"]
    with pytest.raises(AttributeError):
        assert cal.fake_key
    # test copy method
    copied_obj = cal.copy()
    # Check if the copied object is an instance of Calibration
    assert isinstance(copied_obj, Calibration)
    # Check if the an attribute of the copied object is equal to the original object
    assert copied_obj["onset"] == cal["onset"]
    # Modify the copied object and check if it is independent from the original object
    copied_obj["onset"] = 20
    assert copied_obj["onset"] != cal["onset"]
    # test __repr__
    if cal["onset"] is not None:
        assert repr(cal) == expected_repr  # test __repr__


@requires_testing_data
@pytest.mark.parametrize("fname", [(fname)])
def test_read_calibration(fname):
    """Test reading calibration data from an eyelink asc file."""
    calibrations = read_eyelink_calibration(fname)
    expected_x_left = np.array(
        [
            960.0,
            960.0,
            960.0,
            115.0,
            1804.0,
            216.0,
            1703.0,
            216.0,
            1703.0,
            537.0,
            1382.0,
            537.0,
            1382.0,
        ]
    )
    expected_y_right = np.array(
        [
            540.0,
            92.0,
            987.0,
            540.0,
            540.0,
            145.0,
            145.0,
            934.0,
            934.0,
            316.0,
            316.0,
            763.0,
            763.0,
        ]
    )
    expected_gaze_y_left = np.array(
        [
            544.1,
            76.0,
            1001.2,
            554.8,
            539.0,
            160.4,
            146.4,
            927.1,
            962.1,
            308.4,
            313.9,
            765.0,
            754.6,
        ]
    )
    expected_offset_right = np.array(
        [0.36, 0.5, 0.2, 0.1, 0.3, 0.38, 0.13, 0.33, 0.22, 0.18, 0.34, 0.52, 0.21]
    )

    assert len(calibrations) == 2  # calibration[0] is left, calibration[1] is right
    assert calibrations[0]["onset"] == 0
    assert calibrations[1]["onset"] == 0
    assert calibrations[0]["model"] == "HV13"
    assert calibrations[1]["model"] == "HV13"
    assert calibrations[0]["eye"] == "left"
    assert calibrations[1]["eye"] == "right"
    assert calibrations[0]["avg_error"] == 0.30
    assert calibrations[0]["max_error"] == 0.90
    assert calibrations[1]["avg_error"] == 0.31
    assert calibrations[1]["max_error"] == 0.52
    assert calibrations[0]["points"]["point_x"] == pytest.approx(expected_x_left)
    assert calibrations[1]["points"]["point_y"] == pytest.approx(expected_y_right)
    assert calibrations[0]["points"]["gaze_y"] == pytest.approx(expected_gaze_y_left)
    assert calibrations[1]["points"]["offset"] == pytest.approx(expected_offset_right)


@requires_testing_data
@pytest.mark.parametrize("fname", [(fname)])
def test_plot_calibration(fname):
    """Test plotting calibration data."""
    import matplotlib.pyplot as plt

    # Set the non-interactive backend
    plt.switch_backend("agg")

    calibrations = read_eyelink_calibration(fname)
    cal_left = calibrations[0]
    fig = cal_left.plot(show=True, show_offsets=True)
    ax = fig.axes[0]

    scatter1 = ax.collections[0]
    scatter2 = ax.collections[1]
    px, py = cal_left.points["point_x"], cal_left.points["point_y"]
    gaze_x, gaze_y = cal_left.points["gaze_x"], cal_left.points["gaze_y"]

    assert ax.title.get_text() == f"Calibration ({cal_left.eye} eye)"
    assert len(ax.collections) == 2  # Two scatter plots

    assert np.allclose(scatter1.get_offsets(), np.column_stack((px, py)))
    assert np.allclose(scatter2.get_offsets(), np.column_stack((gaze_x, gaze_y)))
    plt.close(fig)
