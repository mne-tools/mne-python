"""Helper functions for reading eyelink ASCII files."""
# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause

import re
import numpy as np


def _find_recording_start(lines):
    """Return the first START line in an SR Research EyeLink ASCII file.

    Parameters
    ----------
        lines: A list of strings, which are The lines in an eyelink ASCII file.

    Returns
    -------
        The line that contains the info on the start of the recording.
    """
    for line in lines:
        if line.startswith("START"):
            return line
    raise ValueError("Could not find the start of the recording.")


def _parse_validation_line(line):
    """Parse a single line of eyelink validation data.

    Parameters
    ----------
        line: A string containing a line of validation data from an eyelink
        ASCII file.

    Returns
    -------
        A list of tuples containing the validation data.
    """
    tokens = line.split()
    xy = tokens[-6].strip("[]").split(",")  # e.g. '960, 540'
    xy_diff = tokens[-2].strip("[]").split(",")  # e.g. '-1.5, -2.8'
    vals = [float(v) for v in [*xy, tokens[-4], *xy_diff]]
    vals[3] += vals[0]  # pos_x + eye_x i.e. 960 + -1.5
    vals[4] += vals[1]  # pos_y + eye_y

    return tuple(vals)


def _parse_calibration(
    lines, screen_size=None, screen_distance=None, screen_resolution=None
):
    """Parse the lines in the given list and returns a list of Calibration instances.

    Parameters
    ----------
        lines: A list of strings, which are The lines in an eyelink ASCII file.

    Returns
    -------
        A list containing one or more Calibration instances,
        one for each calibration that was recorded in the eyelink ASCII file
        data.
    """
    from ...preprocessing.eyetracking.calibration import Calibration

    regex = re.compile(r"\d+")  # for finding numeric characters
    calibrations = list()
    rec_start = float(_find_recording_start(lines).split()[1])

    for line_number, line in enumerate(lines):
        if (
            "!CAL VALIDATION " in line and "ABORTED" not in line
        ):  # Start of a calibration
            tokens = line.split()
            model = tokens[4]  # e.g. 'HV13'
            this_eye = tokens[6].lower()  # e.g. 'left'
            timestamp = float(tokens[1])
            onset = (timestamp - rec_start) / 1000.0  # in seconds
            avg_error = float(line.split("avg.")[0].split()[-1])  # e.g. 0.3
            max_error = float(line.split("max")[0].split()[-1])  # e.g. 0.9

            n_points = int(regex.search(model).group())  # e.g. 13
            n_points *= 2 if "LR" in line else 1  # one point per eye if "LR"
            # The next n_point lines contain the validation data
            points = []
            for validation_index in range(n_points):
                subline = lines[line_number + validation_index + 1]
                if "!CAL VALIDATION" in subline:
                    continue  # for bino mode, skip the second eye's validation summary
                subline_eye = subline.split("at")[0].split()[-1].lower()  # e.g. 'left'
                if subline_eye != this_eye:
                    continue  # skip the validation lines for the other eye
                point_info = _parse_validation_line(subline)
                points.append(point_info)
            # Convert the list of validation data into a numpy array
            positions = np.array([point[:2] for point in points])
            offsets = np.array([point[2] for point in points])
            gaze = np.array([point[3:] for point in points])
            # create the Calibration instance
            calibration = Calibration(
                onset=max(0.0, onset),  # 0 if calibrated before recording
                model=model,
                eye=this_eye,
                avg_error=avg_error,
                max_error=max_error,
                positions=positions,
                offsets=offsets,
                gaze=gaze,
                screen_size=screen_size,
                screen_distance=screen_distance,
                screen_resolution=screen_resolution,
            )
            calibrations.append(calibration)
    return calibrations
