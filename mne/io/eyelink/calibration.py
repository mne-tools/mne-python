"""Eyetracking Calibration(s) class constructor."""

# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause

from collections import OrderedDict
from ...utils import fill_doc


@fill_doc
class Calibrations(list):
    """A list of Calibration objects.

    Parameters
    ----------
    onset: float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started, then the onset should be
        set to 0 seconds.
    model: str
        A string, which is the model of the eyetracker. For example H3 for
        a horizontal only 3-point calibration, or HV3 for a horizontal and
        vertical 3-point calibration.
    eye: str
        the eye that was calibrated. For example, 'left',
        'right', or 'both'.
    avg_error: float
        The average error in degrees between the calibration points and the
        actual gaze position. If 'eye' is 'both', then a dict can be passed
        with the average error for each eye. For example, {'left': 0.5, 'right': 0.6}.
    max_error: float
        The maximum error in degrees that occurred between the calibration
        points and the actual gaze position. If 'eye' is 'both', then a dict
        can be passed with the maximum error for each eye. For example,
        {'left': 0.5, 'right': 0.6}.
    points: ndarray
        a 2D numpy array, which are the data for each calibration point.
        Each row contains the x and y pixel-coordinates of the actual gaze position
        to the calibration point, the error in degrees between the calibration point
        and the actual gaze position, and the difference in x and y pixel coordinates
        between the calibration point and the actual gaze position. If 'eye' is 'both',
        then a dict can be passed with a separate 2D numpy array for each eye.
    screen_size : tuple
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example (.531, .298) for a monitor with
        a display area of 531 x 298 cm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : tuple
        The resolution (in pixels) of the screen that the eyetracking data
        was collected with. For example, (1920, 1080) for a 1920x1080
        resolution display.

    Returns
    -------
    calibrations: Calibrations
        A Calibrations instance, which is a list of Calibration objects.
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
        if any(
            arg is not None
            for arg in (
                onset,
                model,
                eye,
                avg_error,
                max_error,
                points,
                screen_size,
                screen_distance,
                screen_resolution,
            )
        ):
            calibration = Calibration(
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
            self.append(calibration)

    def __repr__(self):
        """Return the number of calibration objects in this instance."""
        num_calibrations = len(self)
        return f"Calibrations | {num_calibrations} calibration(s)"


@fill_doc
class Calibration(OrderedDict):
    """A dictionary containing calibration data.

    Parameters
    ----------
    onset: float
        The onset of the calibration in seconds. If the calibration was
        performed before the recording started, then the onset should be
        set to 0 seconds.
    model: str
        A string, which is the model of the eyetracker. For example H3 for
        a horizontal only 3-point calibration, or HV3 for a horizontal and
        vertical 3-point calibration.
    eye: str
        the eye that was calibrated. For example, 'left',
        'right', or 'both'.
    avg_error: float
        The average error in degrees between the calibration points and the
        actual gaze position. If 'eye' is 'both', then a dict can be passed
        with the average error for each eye. For example, {'left': 0.5, 'right': 0.6}.
    max_error: float
        The maximum error in degrees that occurred between the calibration
        points and the actual gaze position. If 'eye' is 'both', then a dict
        can be passed with the maximum error for each eye. For example,
        {'left': 0.5, 'right': 0.6}.
    points: ndarray
        a 2D numpy array, which are the data for each calibration point.
        Each row contains the x and y pixel-coordinates of the actual gaze position
        to the calibration point, the error in degrees between the calibration point
        and the actual gaze position, and the difference in x and y pixel coordinates
        between the calibration point and the actual gaze position. If 'eye' is 'both',
        then a dict can be passed with a separate 2D numpy array for each eye.
    screen_size : tuple
        The width and height (in meters) of the screen that the eyetracking
        data was collected with. For example (.531, .298) for a monitor with
        a display area of 531 x 298 cm.
    screen_distance : float
        The distance (in meters) from the participant's eyes to the screen.
    screen_resolution : tuple
        The resolution (in pixels) of the screen that the eyetracking data
        was collected with. For example, (1920, 1080) for a 1920x1080
        resolution display.
    """

    def __init__(
        self,
        onset=None,
        model=None,
        avg_error=None,
        max_error=None,
        points=None,
        eye=None,
        screen_size=None,
        screen_distance=None,
        screen_resolution=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self["onset"] = onset
        self["model"] = {} if model is None else model
        self["eye"] = {} if eye is None else eye
        self["avg_error"] = {} if avg_error is None else avg_error
        self["max_error"] = {} if max_error is None else max_error
        self["points"] = [] if points is None else points
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
