# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne.datasets.testing import data_path, requires_testing_data

from ..calibration import Calibration, read_eyelink_calibration

# for test_read_eylink_calibration
testing_path = data_path(download=False)
fname = testing_path / "eyetrack" / "test_eyelink.asc"

# for test_create_calibration
POSITIONS = np.array([[115.0, 540.0], [960.0, 540.0], [1804.0, 540.0]])
OFFSETS = np.array([0.42, 0.23, 0.17])
GAZES = np.array([[101.5, 554.8], [9.9, -4.1], [1795.9, 539.0]])

EXPECTED_REPR = (
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
        "onset, model, eye, avg_error, max_error, positions, offsets, gaze,"
        " screen_size, screen_distance, screen_resolution"
    ),
    [
        (
            0,
            "H3",
            "right",
            0.5,
            1.0,
            POSITIONS,
            OFFSETS,
            GAZES,
            (0.531, 0.298),
            0.065,
            (1920, 1080),
        ),
        (None, None, None, None, None, None, None, None, None, None, None),
    ],
)
def test_create_calibration(
    onset,
    model,
    eye,
    avg_error,
    max_error,
    positions,
    offsets,
    gaze,
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
        positions=positions,
        offsets=offsets,
        gaze=gaze,
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
    if positions is not None:
        assert isinstance(cal["positions"], np.ndarray)
        assert np.array_equal(cal["positions"], np.array(POSITIONS))
    else:
        assert cal["positions"] is None
    if offsets is not None:
        assert isinstance(cal["offsets"], np.ndarray)
        assert np.array_equal(cal["offsets"], np.array(OFFSETS))
    if gaze is not None:
        assert isinstance(cal["gaze"], np.ndarray)
        assert np.array_equal(cal["gaze"], np.array(GAZES))
    assert cal["screen_size"] == screen_size
    assert cal["screen_distance"] == screen_distance
    assert cal["screen_resolution"] == screen_resolution
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
        assert repr(cal) == EXPECTED_REPR  # test __repr__


@requires_testing_data
@pytest.mark.parametrize("fname", [(fname)])
def test_read_calibration(fname):
    """Test reading calibration data from an eyelink asc file."""
    calibrations = read_eyelink_calibration(fname)
    # These numbers were pulled from the file and confirmed.
    POSITIONS_L = (
        [960, 540],
        [960, 92],
        [960, 987],
        [115, 540],
        [1804, 540],
        [216, 145],
        [1703, 145],
        [216, 934],
        [1703, 934],
        [537, 316],
        [1382, 316],
        [537, 763],
        [1382, 763],
    )

    DIFF_L = (
        [9.9, -4.1],
        [-7.8, 16.0],
        [-1.9, -14.2],
        [13.5, -14.8],
        [8.1, 1.0],
        [-7.0, -15.4],
        [-10.1, -1.4],
        [-0.3, 6.9],
        [-32.3, -28.1],
        [8.2, 7.6],
        [9.6, 2.1],
        [-10.6, -2.0],
        [-11.8, 8.4],
    )
    GAZE_L = np.array(POSITIONS_L) + np.array(DIFF_L)

    POSITIONS_R = (
        [960, 540],
        [960, 92],
        [960, 987],
        [115, 540],
        [1804, 540],
        [216, 145],
        [1703, 145],
        [216, 934],
        [1703, 934],
        [537, 316],
        [1382, 316],
        [537, 763],
        [1382, 763],
    )
    DIFF_R = (
        [-5.2, -16.1],
        [23.7, 1.3],
        [2.0, -9.3],
        [4.4, 1.5],
        [-6.5, -12.7],
        [16.6, -7.5],
        [5.7, -1.8],
        [15.4, -3.5],
        [-2.0, -10.2],
        [0.1, 8.3],
        [1.9, -15.8],
        [-24.8, -2.3],
        [3.2, -9.2],
    )
    GAZE_R = np.array(POSITIONS_R) + np.array(DIFF_R)

    OFFSETS_R = [
        0.36,
        0.50,
        0.20,
        0.10,
        0.30,
        0.38,
        0.13,
        0.33,
        0.22,
        0.18,
        0.34,
        0.52,
        0.21,
    ]

    assert len(calibrations) == 2  # calibration[0] is left, calibration[1] is right
    np.testing.assert_allclose(calibrations[0]["onset"], -6.85)
    np.testing.assert_allclose(calibrations[1]["onset"], -6.85)
    assert calibrations[0]["model"] == "HV13"
    assert calibrations[1]["model"] == "HV13"
    assert calibrations[0]["eye"] == "left"
    assert calibrations[1]["eye"] == "right"
    assert calibrations[0]["avg_error"] == 0.30
    assert calibrations[0]["max_error"] == 0.90
    assert calibrations[1]["avg_error"] == 0.31
    assert calibrations[1]["max_error"] == 0.52
    np.testing.assert_array_equal(POSITIONS_L, calibrations[0]["positions"])
    np.testing.assert_array_equal(POSITIONS_R, calibrations[1]["positions"])
    np.testing.assert_array_equal(GAZE_L, calibrations[0]["gaze"])
    np.testing.assert_array_equal(GAZE_R, calibrations[1]["gaze"])
    np.testing.assert_array_equal(OFFSETS_R, calibrations[1]["offsets"])


@requires_testing_data
@pytest.mark.parametrize(
    "fname, axes",
    [(fname, None), (fname, True)],
)
def test_plot_calibration(fname, axes):
    """Test plotting calibration data."""
    import matplotlib.pyplot as plt

    # Set the non-interactive backend
    plt.switch_backend("agg")

    if axes:
        axes = plt.subplot()
    calibrations = read_eyelink_calibration(fname)
    cal_left = calibrations[0]
    fig = cal_left.plot(show=True, show_offsets=True, axes=axes)
    ax = fig.axes[0]

    scatter1 = ax.collections[0]
    scatter2 = ax.collections[1]
    px, py = cal_left["positions"].T
    gaze_x, gaze_y = cal_left["gaze"].T

    assert ax.title.get_text() == f"Calibration ({cal_left['eye']} eye)"
    assert len(ax.collections) == 2  # Two scatter plots

    np.testing.assert_allclose(scatter1.get_offsets(), np.column_stack((px, py)))
    np.testing.assert_allclose(
        scatter2.get_offsets(), np.column_stack((gaze_x, gaze_y))
    )
    plt.close(fig)
