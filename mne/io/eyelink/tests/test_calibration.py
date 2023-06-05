import pytest
from ..calibration import Calibration


@pytest.mark.parametrize(
    (
        "onset, model, eye, avg_error, max_error, points, screen_size, screen_distance,"
        " screen_resolution"
    ),
    [
        (0, "H3", "right", 0.5, 1.0, 3, (0.531, 0.298), 0.065, (1920, 1080)),
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
    if all([kwarg is None for kwarg in kwargs]):
        for kwarg in kwargs:
            assert cal[kwarg] is None
    else:
        assert cal["onset"] == onset
        assert cal["model"] == model
        assert cal["eye"] == eye
        assert cal["avg_error"] == avg_error
        assert cal["max_error"] == max_error
        assert cal["points"] == points
        assert cal["screen_size"] == screen_size
        assert cal["screen_distance"] == screen_distance
        assert cal["screen_resolution"] == screen_resolution
