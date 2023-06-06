import pytest

import numpy as np

from ..calibration import Calibration


test_points = [(960.0, 540.0, 0.23, 9.9, -4.1), (960.0, 92.0, 0.38, -7.8, 16.0)]
field_names = ["point_x", "point_y", "offset", "diff_x", "diff_y"]
dtypes = [(name, float) for name in field_names]
test_array = np.array(test_points, dtype=dtypes)


@pytest.mark.parametrize(
    (
        "onset, model, eye, avg_error, max_error, points, screen_size, screen_distance,"
        " screen_resolution"
    ),
    [
        (0, "H3", "right", 0.5, 1.0, test_points, (0.531, 0.298), 0.065, (1920, 1080)),
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
        assert np.array_equal(cal["points"], test_array)
    else:
        assert cal["points"] is None
    assert cal["screen_size"] == screen_size
    assert cal["screen_distance"] == screen_distance
    assert cal["screen_resolution"] == screen_resolution
