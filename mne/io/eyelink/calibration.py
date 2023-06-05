"""Eyetracking Calibration(s) class constructor."""

# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause

from collections import OrderedDict

import numpy as np

from ...utils import fill_doc


@fill_doc
class Calibration(OrderedDict):
    """A dictionary containing calibration data.

    Parameters
    ----------
    onset : float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started, then the onset should be
        set to 0 seconds.
    model : str
        A string, which is the model of the eyetracker. For example H3 for
        a horizontal only 3-point calibration, or HV3 for a horizontal and
        vertical 3-point calibration.
    eye : str
        The eye that was calibrated. For example, 'left',
        'right', or 'both'.
    avg_error : float
        The average error in degrees between the calibration points and the
        actual gaze position.
    max_error : float
        The maximum error in degrees that occurred between the calibration
        points and the actual gaze position.
    points : list
        List of tuples, contaiing the data for each individual calibration point.
        Each tuple should represent data for 1 calibration point. The elements
        within each tuple should be as follows:
            - (point_x, point_y, offset, diff_x, diff_y)
        where:
            - point_x: the x pixel-coordinate of the calibration point
            - point_y: the y pixel-coordinate of the calibration point
            - offset: the error in degrees between the calibration point and the
                actual gaze position
            - diff_x: the difference in x pixel coordinates between the calibration
                point and the actual gaze position
            - diff_y: the difference in y pixel coordinates between the calibration
                point and the actual gaze position
    screen_size : tuple
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example (.531, .298) for a monitor with
        a display area of 531 x 298 mm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : tuple
        The resolution (in pixels) of the screen that the eyetracking data
        was collected with. For example, (1920, 1080) for a 1920x1080
        resolution display.

    Attributes
    ----------
    onset : float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started.
    model : str
        A string, which is the model of the calibration that was administerd. For
        example 'H3' for a horizontal only 3-point calibration, or 'HV3' for a
        horizontal and vertical 3-point calibration.
    eye : str
        The eye that was calibrated. For example, 'left', or 'right'.
    avg_error : float
        The average error in degrees between the calibration points and the actual gaze
        position.
    max_error : float
        The maximum error in degrees that occurred between the calibration points and
        the actual gaze position.
    points : ndarray
        a 2D numpy array, which are the data for each calibration point.
    screen_size : tuple
        The width and height (in meters) of the screen that the eyetracking data was
        collected  with. For example (.531, .298) for a monitor with a display area of
        531 x 298 mm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : tuple
        The resolution (in pixels) of the screen that the eyetracking data was
        collected with.
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
        if points is not None and isinstance(points, list):
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

    def set_calibration_array(self, data):
        """
        Convert a list of tuples to a structured array with calibration field names.

        Parameters
        ----------
        data : list
           List of tuples, containing the data for each individual calibration point.
           Each tuple should represent data for 1 calibration point. The elements
           within each tuple should be as follows:
                - (point_x, point_y, offset, diff_x, diff_y)
            where:
                - point_x: the x pixel-coordinate of the calibration point
                - point_y: the y pixel-coordinate of the calibration point
                - offset: the error in degrees between the calibration point and the
                    actual gaze position
                - diff_x: the difference in x pixel coordinates between the calibration
                    point and the actual gaze position
                - diff_y: the difference in y pixel coordinates between the calibration
                    point and the actual gaze position

        Returns
        -------
        self: instance of Calibration
            The Calibration instance with the points attribute set as a structured numpy
            array.

        Examples
        --------
        Below is an example of a list of tuples that can be passed to this method:
        >>> data = [(960., 540., 0.23, 9.9, -4.1), (960., 92., 0.38, -7.8, 16.)]
        """
        field_names = ["point_x", "point_y", "offset", "diff_x", "diff_y"]
        dtype = [(name, float) for name in field_names]
        if isinstance(data, list):
            if not all([len(elem) == len(field_names) for elem in data]):
                raise ValueError(
                    f"Each tuple in the data list must have have 5 elements: "
                    f"Got {data}"
                )
            structured_array = np.array(data, dtype=dtype)
            self["points"] = structured_array
        else:
            raise TypeError(
                f"Data must be a list. got {data} which is of type {type(data)}"
            )

    def plot(self, title=None, show=True):
        """Visualize calibration.

        Parameters
        ----------
        title : str
            The title to be displayed. Defaults to None, which uses a generic title.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object for the calibration plot.
        """
        import matplotlib.pyplot as plt

        if not len(self["points"]):
            raise ValueError(
                "No calibration data to plot. Use set_calibration_array()"
                " to set calibration data."
            )
        if not isinstance(self["points"], np.ndarray):
            raise TypeError(
                "Calibration points must be a numpy array. Use "
                "set_calibration_array() to set calibration data."
            )
        fig, ax = plt.subplots()
        px, py = self["points"]["point_x"], self["points"]["point_y"]
        dx, dy = self["points"]["diff_x"], self["points"]["diff_y"]

        if title is None:
            ax.set_title(f"Calibration ({self['eye']} eye)")
        else:
            ax.set_title(title)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        ax.scatter(px, py, color="gray")
        ax.scatter(px - dx, py - dy, color="red")
        fig.show() if show else None
        return fig
