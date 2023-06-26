"""Helper functions for reading eyelink ASCII files."""
# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause

import re
import numpy as np

from ...utils import _check_pandas_installed, _validate_type


def _isfloat(token):
    """Boolean test for whether string can be of type float.

    Parameters
    ----------
    token : str
        Single element from tokens list.
    """
    _validate_type(token, str, "token")
    try:
        float(token)
    except ValueError:
        return False
    else:
        return True


def _convert_types(tokens):
    """Convert the type of each token in list.

    The tokens input is a list of string elements.
    Posix timestamp strings can be integers, eye gaze position and
    pupil size can be floats. flags token ("...") remains as string.
    Missing eye/head-target data (indicated by '.' or 'MISSING_DATA')
    are replaced by np.nan.

    Parameters
    ----------
    Tokens : list
        List of string elements.

    Returns
    -------
        Tokens list with elements of various types.
    """
    return [
        int(token)
        if token.isdigit()  # execute this before _isfloat()
        else float(token)
        if _isfloat(token)
        else np.nan
        if token in (".", "MISSING_DATA")
        else token  # remains as string
        for token in tokens
    ]


def _parse_line(line):
    """Parse tab delminited string from eyelink ASCII file.

    Takes a tab deliminited string from eyelink file,
    splits it into a list of tokens, and converts the type
    for each token in the list.
    """
    tokens = line.split()
    return _convert_types(tokens)


def _is_sys_msg(line):
    """Flag lines from eyelink ASCII file that contain a known system message.

    Some lines in eyelink files are system outputs usually
    only meant for Eyelinks DataViewer application to read.
    These shouldn't need to be parsed.

    Parameters
    ----------
    line : string
        single line from Eyelink asc file

    Returns
    -------
    bool :
        True if any of the following strings that are
        known to indicate a system message are in the line

    Notes
    -----
    Examples of eyelink system messages:
    - ;Sess:22Aug22;Tria:1;Tri2:False;ESNT:182BFE4C2F4;
    - ;NTPT:182BFE55C96;SMSG:__NTP_CLOCK_SYNC__;DIFF:-1;
    - !V APLAYSTART 0 1 library/audio
    - !MODE RECORD CR 500 2 1 R
    """
    return "!V" in line or "!MODE" in line or ";" in line


def _get_sfreq(rec_info):
    """Get sampling frequency from Eyelink ASCII file.

    Parameters
    ----------
    rec_info : list
        the first list in self._event_lines['SAMPLES'].
        The sfreq occurs after RATE: i.e. [..., RATE, 1000, ...].

    Returns
    -------
    sfreq : int | float
    """
    return rec_info[rec_info.index("RATE") + 1]


def _sort_by_time(df, col="time"):
    df.sort_values(col, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _convert_times(df, first_samp, col="time"):
    """Set initial time to 0, converts from ms to seconds in place.

    Parameters
    ----------
       df pandas.DataFrame:
           One of the dataframes in the self.dataframes dict.

       first_samp int:
           timestamp of the first sample of the recording. This should
           be the first sample of the first recording block.
        col str (default 'time'):
            column name to sort pandas.DataFrame by

    Notes
    -----
    Each sample in an Eyelink file has a posix timestamp string.
    Subtracts the "first" sample's timestamp from each timestamp.
    The "first" sample is inferred to be the first sample of
    the first recording block, i.e. the first "START" line.
    """
    _sort_by_time(df, col)
    for col in df.columns:
        if col.endswith("time"):  # 'time' and 'end_time' cols
            df[col] -= first_samp
            df[col] /= 1000
        if col in ["duration", "offset"]:
            df[col] /= 1000


def _fill_times(
    df,
    sfreq,
    time_col="time",
):
    """Fill missing timestamps if there are multiple recording blocks.

    Parameters
    ----------
    df : pandas.DataFrame:
        dataframe of the eyetracking data samples, BEFORE
        _convert_times() is applied to the dataframe

    sfreq : int | float:
        sampling frequency of the data

    time_col : str (default 'time'):
        name of column with the timestamps (e.g. 9511881, 9511882, ...)

    Returns
    -------
    %(df_return)s

    Notes
    -----
    After _parse_recording_blocks, Files with multiple recording blocks will
    have missing timestamps for the duration of the period between the blocks.
    This would cause the occular annotations (i.e. blinks) to not line up with
    the signal.
    """
    pd = _check_pandas_installed()

    first, last = df[time_col].iloc[[0, -1]]
    step = 1000 / sfreq
    df[time_col] = df[time_col].astype(float)
    new_times = pd.DataFrame(
        np.arange(first, last + step / 2, step), columns=[time_col]
    )
    return pd.merge_asof(
        new_times, df, on=time_col, direction="nearest", tolerance=step / 10
    )


def _find_overlaps(df, max_time=0.05):
    """Merge left/right eye events with onset/offset diffs less than max_time.

    df : pandas.DataFrame
        Pandas DataFrame with occular events (fixations, saccades, blinks)
    max_time : float (default 0.05)
        Time in seconds. Defaults to .05 (50 ms)

    Returns
    -------
    DataFrame: %(df_return)s
        :class:`pandas.DataFrame` specifying overlapped eye events, if any
    Notes
    -----
    The idea is to cumulative sum the boolean values for rows with onset and
    offset differences (against the previous row) that are greater than the
    max_time. If onset and offset diffs are less than max_time then no_overlap
    will become False. Alternatively, if either the onset or offset diff is
    greater than max_time, no_overlap becomes True. Cumulatively summing over
    these boolean values will leave rows with no_overlap == False unchanged
    and hence with the same group number.
    """
    pd = _check_pandas_installed()

    df = df.copy()
    df["overlap_start"] = df.sort_values("time")["time"].diff().lt(max_time)

    df["overlap_end"] = df["end_time"].diff().abs().lt(max_time)

    df["no_overlap"] = ~(df["overlap_end"] & df["overlap_start"])
    df["group"] = df["no_overlap"].cumsum()

    # now use groupby on 'group'. If one left and one right eye in group
    # the new start/end times are the mean of the two eyes
    ovrlp = pd.concat(
        [
            pd.DataFrame(g[1].drop(columns="eye").mean()).T
            if (len(g[1]) == 2) and (len(g[1].eye.unique()) == 2)
            else g[1]  # not an overlap, return group unchanged
            for g in df.groupby("group")
        ]
    )
    # overlapped events get a "both" value in the "eye" col
    if "eye" in ovrlp.columns:
        ovrlp["eye"] = ovrlp["eye"].fillna("both")
    else:
        ovrlp["eye"] = "both"
    tmp_cols = ["overlap_start", "overlap_end", "no_overlap", "group"]
    return ovrlp.drop(columns=tmp_cols).reset_index(drop=True)


# Used by read_eyelinke_calibration


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
