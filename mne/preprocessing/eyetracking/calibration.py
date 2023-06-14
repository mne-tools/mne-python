"""Eyetracking Calibration(s) class constructor."""

# Authors: Scott Huberty <seh33@uw.edu>
#          Eric Larson <larson.eric.d@gmail>
#          Adapted from: https://github.com/pyeparse/pyeparse
# License: BSD-3-Clause

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ...utils import _check_fname, fill_doc, logger


@fill_doc
class Calibration(dict):
    """Eye-tracking calibration info.

    This data structure behaves like a dictionary. It contains information regarding a
    calibration that was conducted during an eye-tracking recording.

    .. note::
        When possible, this class should be instantiated via a helper function,
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
        The average error in degrees between the calibration points and the
        actual gaze position.
    max_error : float
        The maximum error in degrees that occurred between the calibration
        points and the actual gaze position.
    points : array-like of float, shape ``(n_calibration_points, 5)``
        The data for the positions, actual gaze, and offsets for each calibration point.
        Each row should contain data for 1 calibration point. The columns should be
        of shape ``(5,)`` and contain ``(point_x, point_y, offset, gaze_x, gaze_y)``,
        where:

            - point_x: the x-coordinate of the calibration point
            - point_y: the y-coordinate of the calibration point
            - offset: the error in degrees between the calibration position and the
                actual gaze position
            - gaze_x: the x-coordinate of the actual gaze position
            - gaze_y: the y-coordinate of the actual gaze position

        If the value for a field is not available, use ``np.nan``. See the example below
        for more details.

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
    points : ndarray
        a 1D structured numpy array of shape ``(n_calibration_points,)``, which contains
        the data for each calibration point.
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

    Examples
    --------
    Below is an example of data that can be passed to the points parameter:
    ``>>> data = [(960., 540., 0.23, 950.1,  544.1), (960., 92., 0.38, 967.8, 76. )]``
    """

    def __init__(
        self,
        onset=None,
        model=None,
        eye=None,
        avg_error=None,
        max_error=None,
        points=None,
        screen_size=None,
        screen_distance=None,
        screen_resolution=None,
    ):
        super().__init__()
        self["onset"] = onset
        self["model"] = model
        self["eye"] = eye
        self["avg_error"] = avg_error
        self["max_error"] = max_error
        if points is not None:
            self.set_calibration_array(points)
        else:
            self["points"] = points
        self["screen_size"] = screen_size
        self["screen_distance"] = screen_distance
        self["screen_resolution"] = screen_resolution

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

    def __getattr__(self, name):
        """Allow dot indexing of dict keys."""
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def copy(self):
        """Copy the instance.

        Returns
        -------
        info : instance of Calibration
            The copied Calibration.
        """
        return deepcopy(self)

    def set_calibration_array(self, data):
        """
        Create a Numpy Array containing data regarding each calibration point.

        This method takes an array-like objects and converts it into a structured numpy
        array, with field names ``'point_x'``, ``'point_y'``, ``'offset'``,
        ``'gaze_x'``, and ``'gaze_y'``.

        Parameters
        ----------
        data : array-like of float, shape ``(n_calibration_points, 5)``
            The data for the positions, actual gaze, and offsets for each calibration
            point. Each row should contain data for 1 calibration point. The columns
            should be of shape ``(5,)`` and contain
            ``(point_x, point_y, offset, gaze_x, gaze_y)``, where:

                - point_x: the x-coordinate of the calibration point
                - point_y: the y-coordinate of the calibration point
                - offset: the error in degrees between the calibration position and the
                    actual gaze position
                - gaze_x: the x-coordinate of the actual gaze position
                - gaze_y: the y-coordinate of the actual gaze position

            If the value for a field is not available, use ``np.nan``. See the example
            below for more details.

        Returns
        -------
        self: instance of Calibration
            The Calibration instance, with the points attribute containing a structured
            numpy array, with field names ``'point_x'``, ``'point_y'``, ``'offset'``,
            ``'diff_x'``, and ``'diff_y'``.

        Examples
        --------
        Below is an example of a list of tuples that can be passed to this method:
        ``>>> data = [(960., 540., 0.23, 950.1, 544.1), (960., 92., 0.38, 967.8, 76.)]``
        """
        from numpy.lib.recfunctions import unstructured_to_structured

        field_names = ("point_x", "point_y", "offset", "gaze_x", "gaze_y")
        dtypes = np.dtype([(name, float) for name in field_names])
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float64)

        if not isinstance(data, np.ndarray):
            raise TypeError(
                "data must be array-like of shape (n_points, 5). got {data}"
            )
        if data.dtype.names == field_names:
            # already a structured array
            structured_array = data
        else:
            structured_array = unstructured_to_structured(data, dtype=dtypes)
        assert structured_array.ndim == 1
        if not len(structured_array[0]) == 5:
            raise ValueError(
                f"Each column in data must have have 5 elements: got {data}"
            )
        self["points"] = structured_array

    def plot(self, title=None, show_offsets=False, invert_y_axis=True, show=True):
        """Visualize calibration.

        Parameters
        ----------
        title : str
            The title to be displayed. Defaults to ``None``, which uses a generic title.
        show_offsets : bool
            Whether to display the offset (in visual degrees) of each calibration
            point or not. Defaults to ``False``.
        invert_y_axis : bool
            Whether to invert the y-axis or not. In many monitors, pixel coordinate
            (0,0), which is often referred to as origin, is at the top left of corner
            of the screen. Defaults to ``True``.
        show : bool
            Whether to show the figure or not. Defaults to ``True``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object for the calibration plot.
        """
        import matplotlib.pyplot as plt

        if not isinstance(self["points"], np.ndarray):
            raise TypeError(
                "Calibration points must be a numpy array. Use "
                "set_calibration_array() to set calibration data."
            )
        fig, ax = plt.subplots()
        px, py = self["points"]["point_x"], self["points"]["point_y"]
        gaze_x, gaze_y = self["points"]["gaze_x"], self["points"]["gaze_y"]

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

        if invert_y_axis:
            # Invert the y-axis because origin is at the top left corner for most
            # monitors
            ax.invert_yaxis()
        ax.scatter(px, py, color="gray")
        ax.scatter(gaze_x, gaze_y, color="red", alpha=0.5)

        if show_offsets:
            for i in range(len(px)):
                x_offset = 0.01 * gaze_x[i]  # 1% to the right of the gazepoint
                text = ax.text(
                    x=gaze_x[i] + x_offset,
                    y=gaze_y[i],
                    s=self["points"]["offset"][i],
                    fontsize=8,
                    ha="left",
                    va="center",
                )

        fig.tight_layout()
        fig.show() if show else None
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
    screen_size : array-like of shape (2,)
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example ``(.531, .298)`` for a monitor with
        a display area of 531 x 298 mm. Defaults to ``None``.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
        Defaults to ``None``.
    screen_resolution : array-like of shape (2,)
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
