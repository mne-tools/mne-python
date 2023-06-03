"""Helper functions for reading eyelink ASCII files."""
# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause

import re
import numpy as np

from .calibration import Calibration, Calibrations


def _find_recording_start(lines):
    """Return the first START line in an eyelink ASCII file.

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
        A dictionary containing the validation data.
    """
    keys = ["point_x", "point_y", "offset", "diff_x", "diff_y"]
    dtype = [(key, "f8") for key in keys]
    parsed_data = np.empty(1, dtype=dtype)

    tokens = line.split()
    xy = tokens[-6].strip("[]").split(",")  # e.g. '960, 540'
    xy_diff = tokens[-2].strip("[]").split(",")  # e.g. '-1.5, -2.8'
    vals = [float(v) for v in [*xy, tokens[-4], *xy_diff]]

    for key, data in zip(keys, vals):
        parsed_data[0][key] = data

    return parsed_data


def _parse_calibration(
    lines, screen_size=None, screen_distance=None, screen_resolution=None
):
    """Parse the lines in the given list and returns a Calibrations instance.

    Parameters
    ----------
        lines: A list of strings, which are The lines in an eyelink ASCII file.

    Returns
    -------
        A Calibrations instance containing one or more Calibration instances,
        one for each calibration that was recorded in the eyelink ASCII file
        data.
    """
    regex = re.compile(r"\d+")  # for finding numeric characters
    calibrations = Calibrations()
    rec_start = float(_find_recording_start(lines).split()[1])

    for line_number, line in enumerate(lines):
        if (
            "!CAL VALIDATION " in line and "ABORTED" not in line
        ):  # Start of a calibration
            tokens = line.split()
            this_eye = tokens[6].lower()
            assert this_eye in ["left", "right"]
            if "LR" not in line or ("LR" in line and this_eye == "left"):
                # for binocular calibrations, there are two '!CAL VALIDATION' lines
                # Create a single calibration instance for both eyes.
                calibration = Calibration(
                    screen_size=screen_size,
                    screen_distance=screen_distance,
                    screen_resolution=screen_resolution,
                )
            calibration["model"] = tokens[4]  # e.g. 'HV13'
            assert calibration["model"].startswith("H")
            calibration["eye"] = "both" if "LR" in line else this_eye
            timestamp = float(tokens[1])
            onset = timestamp - rec_start
            calibration["onset"] = 0 if onset < 0 else onset

            avg_error = float(line.split("avg.")[0].split()[-1])  # e.g. 0.3
            max_error = float(line.split("max")[0].split()[-1])  # e.g. 0.9
            if calibration["eye"] == "both":
                if not isinstance(calibration["points"], dict):
                    # don't overwrite dict if it was set in previous line
                    calibration["points"] = {"left": [], "right": []}
                calibration["avg_error"][this_eye] = avg_error
                calibration["max_error"][this_eye] = max_error
            else:
                calibration["avg_error"] = avg_error
                calibration["max_error"] = max_error

            n_points = int(regex.search(calibration["model"]).group())  # e.g. 9
            n_points *= 2 if "LR" in line else 1  # one point per eye if "LR"
            # The next n_point lines contain the validation data
            for validation_index in range(n_points):
                subline = lines[line_number + validation_index + 1]
                subline_eye = subline.split("at")[0].split()[-1].lower()
                if subline_eye != this_eye:
                    continue  # skip the validation lines for the other eye
                point_info = _parse_validation_line(subline)
                if calibration["eye"] == "both":
                    calibration["points"][this_eye].append(point_info)
                else:
                    calibration["points"].append(point_info)
            # Convert the list of validation data into a numpy array
            if calibration["eye"] == "both":
                calibration["points"][this_eye] = np.concatenate(
                    calibration["points"][this_eye], axis=0
                )
            else:
                calibration["points"] = np.concatenate(calibration["points"], axis=0)

    calibrations.append(calibration)
    return calibrations
