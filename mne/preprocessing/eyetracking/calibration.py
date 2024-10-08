"""Eyetracking Calibration(s) class constructor."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy

import numpy as np

from ...io.eyelink._utils import _parse_calibration
from ...utils import _check_fname, _validate_type, fill_doc, logger
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
        performed before the recording started, the the onset can be
        negative.
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
    """

    def __init__(
        self,
        *,
        onset,
        model,
        eye,
        avg_error,
        max_error,
        positions,
        offsets,
        gaze,
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
            positions=positions,
            offsets=offsets,
            gaze=gaze,
        )

    def __repr__(self):
        """Return a summary of the Calibration object."""
        return (
            f"Calibration |\n"
            f"  onset: {self['onset']} seconds\n"
            f"  model: {self['model']}\n"
            f"  eye: {self['eye']}\n"
            f"  average error: {self['avg_error']} degrees\n"
            f"  max error: {self['max_error']} degrees\n"
            f"  screen size: {self['screen_size']} meters\n"
            f"  screen distance: {self['screen_distance']} meters\n"
            f"  screen resolution: {self['screen_resolution']} pixels\n"
        )

    def copy(self):
        """Copy the instance.

        Returns
        -------
        cal : instance of Calibration
            The copied Calibration.
        """
        return deepcopy(self)

    def plot(self, show_offsets=True, axes=None, show=True):
        """Visualize calibration.

        Parameters
        ----------
        show_offsets : bool
            Whether to display the offset (in visual degrees) of each calibration
            point or not. Defaults to ``True``.
        axes : instance of matplotlib.axes.Axes | None
            Axes to draw the calibration positions to. If ``None`` (default), a new axes
            will be created.
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

        if axes is not None:
            from matplotlib.axes import Axes

            _validate_type(axes, Axes, "axes")
            ax = axes
            fig = ax.get_figure()
        else:  # create new figure and axes
            fig, ax = plt.subplots(layout="constrained")
        px, py = self["positions"].T
        gaze_x, gaze_y = self["gaze"].T

        ax.set_title(f"Calibration ({self['eye']} eye)")
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

        # Invert y-axis because the origin is in the top left corner
        ax.invert_yaxis()
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
    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")
    logger.info(f"Reading calibration data from {fname}")
    lines = fname.read_text(encoding="ASCII").splitlines()
    return _parse_calibration(lines, screen_size, screen_distance, screen_resolution)
