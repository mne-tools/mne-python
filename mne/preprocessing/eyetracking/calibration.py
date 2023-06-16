"""Eyetracking Calibration(s) class constructor."""

# Authors: Scott Huberty <seh33@uw.edu>
#          Eric Larson <larson.eric.d@gmail>
#          Adapted from: https://github.com/pyeparse/pyeparse
# License: BSD-3-Clause

from copy import deepcopy

import numpy as np

from ...utils import _check_fname, fill_doc, logger
from ...viz.utils import plt_show


@fill_doc
class Calibration(dict):
    """Eye-tracking calibration info.

    This data structure behaves like a dictionary. It contains information regarding a
    calibration that was conducted during an eye-tracking recording.

    .. note::
        When possible, a Calibration instance should be created with a helper function,
        such as :func:`~mne.preprocessing.eyetracking.read_eyelink_calibration`.

    Parameters
    ----------
    onset : float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started, then the onset should be
        set to ``0`` seconds.
    model : str
        A string, which is the model of the eye-tracking calibration that was applied.
        For example ``'H3'`` for a horizontal only 3-point calibration, or ``'HV3'``
        for a horizontal and vertical 3-point calibration.
    eye : str
        The eye that was calibrated. For example, ``'left'``, or ``'right'``.
    avg_error : float
        The average error in degrees between the calibration positions and the
        actual gaze position.
    max_error : float
        The maximum error in degrees that occurred between the calibration
        positions and the actual gaze position.
    positions : array-like of float, shape ``(n_calibration_points, 2)``
        The x and y coordinates of the calibration points.
    offsets : array-like of float, shape ``(n_calibration_points,)``
        The error in degrees between the calibration position and the actual
        gaze position for each calibration point.
    gaze : array-like of float, shape ``(n_calibration_points, 2)``
        The x and y coordinates of the actual gaze position for each calibration point.
    screen_size : array-like of shape ``(2,)``
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example ``(.531, .298)`` for a monitor with
        a display area of 531 x 298 mm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : array-like of shape ``(2,)``
        The resolution (in pixels) of the screen that the eyetracking data
        was collected with. For example, ``(1920, 1080)`` for a 1920x1080
        resolution display.

    Attributes
    ----------
    onset : float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started, the onset will be ``0`` seconds.
    model : str
        A string, which is the model of the calibration that was applied. For
        example ``'H3'`` for a horizontal only 3-point calibration, or ``'HV3'`` for a
        horizontal and vertical 3-point calibration.
    eye : str
        The eye that was calibrated. For example, ``'left'``, or ``'right'``.
    avg_error : float
        The average error in degrees between the calibration points and the actual gaze
        position.
    max_error : float
        The maximum error in degrees that occurred between the calibration points and
        the actual gaze position.
    positions : ndarray of float, shape ``(n_calibration_points, 2)``
        The x and y coordinates of the calibration points.
    offsets : ndarray of float, shape ``(n_calibration_points,)``
        The error in degrees between the calibration position and the actual
        gaze position for each calibration point.
    gaze : ndarray of float, shape ``(n_calibration_points, 2)``
        The x and y coordinates of the actual gaze position for each calibration point.
    screen_size : array-like
        The width and height (in meters) of the screen that the eyetracking data was
        collected  with. For example ``(.531, .298)`` for a monitor with a display area
        of 531 x 298 mm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : array-like
        The resolution (in pixels) of the screen that the eyetracking data was
        collected with. For example, ``(1920, 1080)`` for a 1920x1080 resolution
        display.
    """

    def __init__(
        self,
        onset=None,
        model=None,
        eye=None,
        avg_error=None,
        max_error=None,
        positions=None,
        offsets=None,
        gaze=None,
        screen_size=None,
        screen_distance=None,
        screen_resolution=None,
    ):
        super().__init__(
            onset=onset,
            model=model,
            eye=eye,
            avg_error=avg_error,
            max_error=max_error,
            screen_size=screen_size,
            screen_distance=screen_distance,
            screen_resolution=screen_resolution,
        )
        self["positions"] = positions
        self["offsets"] = offsets
        self["gaze"] = gaze

    def __repr__(self):
        """Return a summary of the Calibration object."""
        onset = self.get("onset", "N/A")
        model = self.get("model", "N/A")
        eye = self.get("eye", "N/A")
        avg_error = self.get("avg_error", "N/A")
        max_error = self.get("max_error", "N/A")
        screen_size = self.get("screen_size", "N/A")
        screen_distance = self.get("screen_distance", "N/A")
        screen_resolution = self.get("screen_resolution", "N/A")
        return (
            f"Calibration |\n"
            f"  onset: {onset} seconds\n"
            f"  model: {model}\n"
            f"  eye: {eye}\n"
            f"  average error: {avg_error} degrees\n"
            f"  max error: {max_error} degrees\n"
            f"  screen size: {screen_size} meters\n"
            f"  screen distance: {screen_distance} meters\n"
            f"  screen resolution: {screen_resolution} pixels\n"
        )

    def copy(self):
        """Copy the instance.

        Returns
        -------
        cal : instance of Calibration
            The copied Calibration.
        """
        return deepcopy(self)

    def __setitem__(self, key, value):
        """Make sure that some keys are caste as numpy arrays.

        Because methods like plot expect numpy arrays.
        """
        if key in ("positions", "offsets", "gaze") and isinstance(value, (tuple, list)):
            logger.info("Converting %s to numpy array", key)
            value = np.array(value)
        super().__setitem__(key, value)

    def plot(self, title=None, show_offsets=True, origin="top-left", show=True):
        """Visualize calibration.

        Parameters
        ----------
        title : str
            The title to be displayed. Defaults to ``None``, which uses a generic title.
        show_offsets : bool
            Whether to display the offset (in visual degrees) of each calibration
            point or not. Defaults to ``False``.
        origin : str
            What should be considered the origin of the screen. Can be ``'top-left'``,
            ``'top-right'``, ``'bottom-left'``, or ``'bottom-right'``. Defaults to
            ``'top-left'`` because for most monitors, pixel coordinate ``(0,0)``, often
            referred to as origin, is at the top left of corner of the screen.
        show : bool
            Whether to show the figure or not. Defaults to ``True``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object for the calibration plot.
        """
        import matplotlib.pyplot as plt

        msg = "positions and gaze keys must both be 2D numpy arrays."
        assert isinstance(self["positions"], np.ndarray), msg
        assert isinstance(self["gaze"], np.ndarray), msg

        fig, ax = plt.subplots(constrained_layout=True)
        px, py = self["positions"].T
        gaze_x, gaze_y = self["gaze"].T

        if title is None:
            ax.set_title(f"Calibration ({self['eye']} eye)")
        else:
            ax.set_title(title)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        # Display avg_error and max_error in the top left corner
        text = (
            f"avg_error: {self['avg_error']} deg.\nmax_error: {self['max_error']} deg."
        )
        ax.text(
            0,
            1.01,
            text,
            transform=ax.transAxes,
            verticalalignment="baseline",
            fontsize=8,
        )

        msg = (
            "origin must be 'top-left', 'top-right', 'bottom-left', or 'bottom-right."
            f" got {origin}"
        )
        assert origin in ("top-left", "top-right", "bottom-left", "bottom-right"), msg
        if origin == "top-left":
            # Invert the y-axis because origin is at the top left corner for most
            # monitors
            ax.invert_yaxis()
        elif origin == "top-right":
            ax.invert_yaxis()
            ax.invert_xaxis()
        elif origin == "bottom-right":
            ax.invert_xaxis()
        # if origin is 'bottom-left' no need to do anything
        ax.scatter(px, py, color="gray")
        ax.scatter(gaze_x, gaze_y, color="red", alpha=0.5)

        if show_offsets:
            for i in range(len(px)):
                x_offset = 0.01 * gaze_x[i]  # 1% to the right of the gazepoint
                text = ax.text(
                    x=gaze_x[i] + x_offset,
                    y=gaze_y[i],
                    s=self["offsets"][i],
                    fontsize=8,
                    ha="left",
                    va="center",
                )

        plt_show(show)
        return fig


@fill_doc
def read_eyelink_calibration(
    fname, screen_size=None, screen_distance=None, screen_resolution=None
):
    """Return info on calibrations collected in an eyelink file.

    Parameters
    ----------
    fname : path-like
        Path to the eyelink file (.asc).
    screen_size : array-like of shape ``(2,)``
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example ``(.531, .298)`` for a monitor with
        a display area of 531 x 298 mm. Defaults to ``None``.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
        Defaults to ``None``.
    screen_resolution : array-like of shape ``(2,)``
        The resolution (in pixels) of the screen that the eyetracking data
        was collected with. For example, ``(1920, 1080)`` for a 1920x1080
        resolution display. Defaults to ``None``.

    Returns
    -------
    calibrations : list
        A list of :class:`~mne.preprocessing.eyetracking.Calibration` instances, one for
        each eye of every calibration that was performed during the recording session.
    """
    from ...io.eyelink._utils import _parse_calibration

    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")
    logger.info("Reading calibration data from {}".format(fname))
    lines = fname.read_text(encoding="ASCII").splitlines()
    return _parse_calibration(lines, screen_size, screen_distance, screen_resolution)
