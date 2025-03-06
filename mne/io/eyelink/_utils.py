"""Helper functions for reading eyelink ASCII files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from datetime import datetime, timedelta, timezone

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ...annotations import Annotations
from ...utils import _check_pandas_installed, logger, warn

EYELINK_COLS = {
    "timestamp": ("time",),
    "pos": {
        "left": ("xpos_left", "ypos_left", "pupil_left"),
        "right": ("xpos_right", "ypos_right", "pupil_right"),
    },
    "velocity": {
        "left": ("xvel_left", "yvel_left"),
        "right": ("xvel_right", "yvel_right"),
    },
    "resolution": ("xres", "yres"),
    "input": ("DIN",),
    "remote": ("x_head", "y_head", "distance"),
    "block_num": ("block",),
    "eye_event": ("eye", "time", "end_time", "duration"),
    "fixation": ("fix_avg_x", "fix_avg_y", "fix_avg_pupil_size"),
    "saccade": (
        "sacc_start_x",
        "sacc_start_y",
        "sacc_end_x",
        "sacc_end_y",
        "sacc_visual_angle",
        "peak_velocity",
    ),
}


def _parse_eyelink_ascii(
    fname, find_overlaps=True, overlap_threshold=0.05, apply_offsets=False
):
    # ======================== Parse ASCII File =========================
    raw_extras = dict()
    raw_extras.update(_parse_recording_blocks(fname))
    raw_extras.update(_get_metadata(raw_extras))
    raw_extras["dt"] = _get_recording_datetime(fname)
    _validate_data(raw_extras)

    # ======================== Create DataFrames ========================
    raw_extras["dfs"] = _create_dataframes(raw_extras, apply_offsets)
    del raw_extras["sample_lines"]  # free up memory
    # add column names to dataframes and set the dtype of each column
    col_names, ch_names = _infer_col_names(raw_extras)
    raw_extras["dfs"] = _assign_col_names(col_names, raw_extras["dfs"])
    raw_extras["dfs"] = _set_df_dtypes(raw_extras["dfs"])  # set dtypes for dataframes
    # if HREF data, convert to radians
    if "HREF" in raw_extras["rec_info"]:
        raw_extras["dfs"]["samples"] = _convert_href_samples(
            raw_extras["dfs"]["samples"]
        )
    # fill in times between recording blocks with BAD_ACQ_SKIP
    if raw_extras["n_blocks"] > 1:
        logger.info(
            f"There are {raw_extras['n_blocks']} recording blocks in this file."
            f" Times between blocks will be annotated with BAD_ACQ_SKIP."
        )
        raw_extras["dfs"]["samples"] = _adjust_times(
            raw_extras["dfs"]["samples"], raw_extras["sfreq"]
        )
    # Convert timestamps to seconds
    for df in raw_extras["dfs"].values():
        df = _convert_times(df, raw_extras["first_samp"])
    # Find overlaps between left and right eye events
    if find_overlaps:
        for key in raw_extras["dfs"]:
            if key not in ["blinks", "fixations", "saccades"]:
                continue
            raw_extras["dfs"][key] = _find_overlaps(
                raw_extras["dfs"][key], max_time=overlap_threshold
            )
    # ======================== Info for BaseRaw ========================
    eye_ch_data = raw_extras["dfs"]["samples"][ch_names].to_numpy().T
    info = _create_info(ch_names, raw_extras)

    return eye_ch_data, info, raw_extras


def _parse_recording_blocks(fname):
    """Parse Eyelink ASCII file.

    Eyelink samples occur within START and END blocks.
    samples lines start with a posix-like string,
    and contain eyetracking sample info. Event Lines
    start with an upper case string and contain info
    about occular events (i.e. blink/saccade), or experiment
    messages sent by the stimulus presentation software.
    """
    with fname.open() as file:
        data_dict = dict()
        data_dict["sample_lines"] = []
        data_dict["event_lines"] = {
            "START": [],
            "END": [],
            "SAMPLES": [],
            "EVENTS": [],
            "ESACC": [],
            "EBLINK": [],
            "EFIX": [],
            "MSG": [],
            "INPUT": [],
            "BUTTON": [],
            "PUPIL": [],
        }

        is_recording_block = False
        for line in file:
            if line.startswith("START"):  # start of recording block
                is_recording_block = True
            if is_recording_block:
                tokens = line.split()
                if not tokens:
                    continue  # skip empty lines
                if tokens[0][0].isnumeric():  # Samples
                    data_dict["sample_lines"].append(tokens)
                elif tokens[0] in data_dict["event_lines"].keys():
                    if _is_sys_msg(line):
                        continue  # system messages don't need to be parsed.
                    event_key, event_info = tokens[0], tokens[1:]
                    data_dict["event_lines"][event_key].append(event_info)
                    if tokens[0] == "END":  # end of recording block
                        is_recording_block = False
        if not data_dict["sample_lines"]:  # no samples parsed
            raise ValueError(f"Couldn't find any samples in {fname}")
        return data_dict


def _validate_data(raw_extras):
    """Check the incoming data for some known problems that can occur."""
    # Detect the datatypes that are in file.
    if "GAZE" in raw_extras["rec_info"]:
        logger.info(
            "Pixel coordinate data detected."
            "Pass `scalings=dict(eyegaze=1e3)` when using plot"
            " method to make traces more legible."
        )

    elif "HREF" in raw_extras["rec_info"]:
        logger.info("Head-referenced eye-angle (HREF) data detected.")
    elif "PUPIL" in raw_extras["rec_info"]:
        warn("Raw eyegaze coordinates detected. Analyze with caution.")
    if "AREA" in raw_extras["pupil_info"]:
        logger.info("Pupil-size area detected.")
    elif "DIAMETER" in raw_extras["pupil_info"]:
        logger.info("Pupil-size diameter detected.")
    # If more than 1 recording period, check whether eye being tracked changed.
    if raw_extras["n_blocks"] > 1:
        if raw_extras["tracking_mode"] == "monocular":
            blocks_list = raw_extras["event_lines"]["SAMPLES"]
            eye_per_block = [block_info[1].lower() for block_info in blocks_list]
            if not all([this_eye == raw_extras["eye"] for this_eye in eye_per_block]):
                warn(
                    "The eye being tracked changed during the"
                    " recording. The channel names will reflect"
                    " the eye that was tracked at the start of"
                    " the recording."
                )


def _get_recording_datetime(fname):
    """Create a datetime object from the datetime in ASCII file."""
    # create a timezone object for UTC
    tz = timezone(timedelta(hours=0))
    in_header = False
    with fname.open() as file:
        for line in file:
            # header lines are at top of file and start with **
            if line.startswith("**"):
                in_header = True
            if in_header:
                if line.startswith("** DATE:"):
                    dt_str = line.replace("** DATE:", "").strip()
                    fmt = "%a %b %d %H:%M:%S %Y"
                    # Eyelink measdate timestamps are timezone naive.
                    # Force datetime to be in UTC.
                    # Even though dt is probably in local time zone.
                    try:
                        dt_naive = datetime.strptime(dt_str, fmt)
                    except ValueError:
                        # date string is missing or in an unexpected format
                        logger.info(
                            "Could not detect date from file with date entry: "
                            f"{repr(dt_str)}"
                        )
                        return
                    else:
                        return dt_naive.replace(tzinfo=tz)  # make it dt aware
        return


def _get_metadata(raw_extras):
    """Get tracking mode, sfreq, eye tracked, pupil metric, etc.

    Don't call this until after _parse_recording_blocks.
    """
    meta_data = dict()
    meta_data["rec_info"] = raw_extras["event_lines"]["SAMPLES"][0]
    if ("LEFT" in meta_data["rec_info"]) and ("RIGHT" in meta_data["rec_info"]):
        meta_data["tracking_mode"] = "binocular"
        meta_data["eye"] = "both"
    else:
        meta_data["tracking_mode"] = "monocular"
        meta_data["eye"] = meta_data["rec_info"][1].lower()
    meta_data["first_samp"] = float(raw_extras["event_lines"]["START"][0][0])
    meta_data["sfreq"] = _get_sfreq_from_ascii(meta_data["rec_info"])
    meta_data["pupil_info"] = raw_extras["event_lines"]["PUPIL"][0]
    meta_data["n_blocks"] = len(raw_extras["event_lines"]["START"])
    return meta_data


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


def _get_sfreq_from_ascii(rec_info):
    """Get sampling frequency from Eyelink ASCII file.

    Parameters
    ----------
    rec_info : list
        the first list in raw_extras["event_lines"]['SAMPLES'].
        The sfreq occurs after RATE: i.e. [..., RATE, 1000, ...].

    Returns
    -------
    sfreq : float
    """
    return float(rec_info[rec_info.index("RATE") + 1])


def _create_dataframes(raw_extras, apply_offsets):
    """Create pandas.DataFrame for Eyelink samples and events.

    Creates a pandas DataFrame for sample_lines and for each
    non-empty key in event_lines.
    """
    pd = _check_pandas_installed()
    df_dict = dict()

    # dataframe for samples
    df_dict["samples"] = pd.DataFrame(raw_extras["sample_lines"])
    df_dict["samples"] = _drop_status_col(df_dict["samples"])  # drop STATUS col

    # dataframe for each type of occular event
    for event, label in zip(
        ["EFIX", "ESACC", "EBLINK"], ["fixations", "saccades", "blinks"]
    ):
        if raw_extras["event_lines"][event]:  # an empty list returns False
            df_dict[label] = pd.DataFrame(raw_extras["event_lines"][event])
        else:
            logger.info(
                f"No {label} were found in this file. "
                f"Not returning any info on {label}."
            )

    # make dataframe for experiment messages
    if raw_extras["event_lines"]["MSG"]:
        msgs = []
        for token in raw_extras["event_lines"]["MSG"]:
            if apply_offsets and len(token) == 2:
                ts, msg = token
                offset = np.nan
            elif apply_offsets:
                ts = token[0]
                try:
                    offset = float(token[1])
                    msg = " ".join(str(x) for x in token[2:])
                except ValueError:
                    offset = np.nan
                    msg = " ".join(str(x) for x in token[1:])
            else:
                ts, offset = token[0], np.nan
                msg = " ".join(str(x) for x in token[1:])
            msgs.append([ts, offset, msg])
        df_dict["messages"] = pd.DataFrame(msgs)

    # make dataframe for recording block start, end times
    i = 1
    blocks = list()
    for bgn, end in zip(
        raw_extras["event_lines"]["START"], raw_extras["event_lines"]["END"]
    ):
        blocks.append((float(bgn[0]), float(end[0]), i))
        i += 1
    cols = ["time", "end_time", "block"]
    df_dict["recording_blocks"] = pd.DataFrame(blocks, columns=cols)

    # TODO: Make dataframes for other eyelink events (Buttons)
    return df_dict


def _drop_status_col(samples_df):
    """Drop STATUS column from samples dataframe.

    see https://github.com/mne-tools/mne-python/issues/11809, and section 4.9.2.1 of
    the Eyelink 1000 Plus User Manual, version 1.0.19. We know that the STATUS
    column is either 3, 5, 13, or 17 characters long, i.e. "...", ".....", ".C."
    """
    status_cols = []
    # we know the first 3 columns will be the time, xpos, ypos
    for col in samples_df.columns[3:]:
        if samples_df[col][0][0].isnumeric():
            # if the value is numeric, it's not a status column
            continue
        if len(samples_df[col][0]) in [3, 5, 13, 17]:
            status_cols.append(col)
    return samples_df.drop(columns=status_cols)


def _infer_col_names(raw_extras):
    """Build column and channel names for data from Eyelink ASCII file.

    Returns the expected column names for the sample lines and event
    lines, to be passed into pd.DataFrame. The columns present in an eyelink ASCII
    file can vary. The order that col_names are built below should NOT change.
    """
    col_names = {}
    # initiate the column names for the sample lines
    col_names["samples"] = list(EYELINK_COLS["timestamp"])

    # and for the eye message lines
    col_names["blinks"] = list(EYELINK_COLS["eye_event"])
    col_names["fixations"] = list(EYELINK_COLS["eye_event"] + EYELINK_COLS["fixation"])
    col_names["saccades"] = list(EYELINK_COLS["eye_event"] + EYELINK_COLS["saccade"])

    # Recording was either binocular or monocular
    # If monocular, find out which eye was tracked and append to ch_name
    if raw_extras["tracking_mode"] == "monocular":
        eye = raw_extras["eye"]
        ch_names = list(EYELINK_COLS["pos"][eye])
    elif raw_extras["tracking_mode"] == "binocular":
        ch_names = list(EYELINK_COLS["pos"]["left"] + EYELINK_COLS["pos"]["right"])
    col_names["samples"].extend(ch_names)

    # The order of these if statements should not be changed.
    if "VEL" in raw_extras["rec_info"]:  # If velocity data are reported
        if raw_extras["tracking_mode"] == "monocular":
            ch_names.extend(EYELINK_COLS["velocity"][eye])
            col_names["samples"].extend(EYELINK_COLS["velocity"][eye])
        elif raw_extras["tracking_mode"] == "binocular":
            ch_names.extend(
                EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
            )
            col_names["samples"].extend(
                EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
            )
    # if resolution data are reported
    if "RES" in raw_extras["rec_info"]:
        ch_names.extend(EYELINK_COLS["resolution"])
        col_names["samples"].extend(EYELINK_COLS["resolution"])
        col_names["fixations"].extend(EYELINK_COLS["resolution"])
        col_names["saccades"].extend(EYELINK_COLS["resolution"])
    # if digital input port values are reported
    if "INPUT" in raw_extras["rec_info"]:
        ch_names.extend(EYELINK_COLS["input"])
        col_names["samples"].extend(EYELINK_COLS["input"])

    # if head target info was reported, add its cols
    if "HTARGET" in raw_extras["rec_info"]:
        ch_names.extend(EYELINK_COLS["remote"])
        col_names["samples"].extend(EYELINK_COLS["remote"])

    return col_names, ch_names


def _assign_col_names(col_names, df_dict):
    """Assign column names to dataframes.

    Parameters
    ----------
    col_names : dict
        Dictionary of column names for each dataframe.
    """
    for key, df in df_dict.items():
        if key in ("samples", "blinks", "fixations", "saccades"):
            df.columns = col_names[key]
        elif key == "messages":
            cols = ["time", "offset", "event_msg"]
            df.columns = cols
    return df_dict


def _set_df_dtypes(df_dict):
    from mne.utils import _set_pandas_dtype

    for key, df in df_dict.items():
        if key in ["samples"]:
            # convert missing position values to NaN
            _set_missing_values(df, df.columns[1:])
            _set_pandas_dtype(df, df.columns, float, verbose="warning")
        elif key in ["blinks", "fixations", "saccades"]:
            _set_missing_values(df, df.columns[1:])
            _set_pandas_dtype(df, df.columns[1:], float, verbose="warning")
        elif key == "messages":
            _set_pandas_dtype(df, ["time"], float, verbose="warning")  # timestamp
    return df_dict


def _set_missing_values(df, columns):
    """Set missing values to NaN. operates in-place."""
    missing_vals = (".", "MISSING_DATA")
    for col in columns:
        # we explicitly use numpy instead of pd.replace because it is faster
        # if a stim channel (DIN) we should use zero so it can cast to int properly
        # in find_events
        replacement = 0 if col == "DIN" else np.nan
        df[col] = np.where(df[col].isin(missing_vals), replacement, df[col])


def _sort_by_time(df, col="time"):
    df.sort_values(col, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _convert_times(df, first_samp, col="time"):
    """Set initial time to 0, converts from ms to seconds in place.

    Parameters
    ----------
       df pandas.DataFrame:
           One of the dataframes in raw_extras["dfs"] dict.

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
    return df


def _adjust_times(
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
    df = pd.merge_asof(
        new_times, df, on=time_col, direction="nearest", tolerance=step / 2
    )
    # fix DIN NaN values
    if "DIN" in df.columns:
        df["DIN"] = df["DIN"].fillna(0)
    return df


def _find_overlaps(df, max_time=0.05):
    """Merge left/right eye events with onset/offset diffs less than max_time.

    Parameters
    ----------
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

    if not len(df):
        return
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


def _convert_href_samples(samples_df):
    """Convert HREF eyegaze samples to radians."""
    # grab the xpos and ypos channel names
    pos_names = EYELINK_COLS["pos"]["left"][:-1] + EYELINK_COLS["pos"]["right"][:-1]
    for col in samples_df.columns:
        if col not in pos_names:  # 'xpos_left' ... 'ypos_right'
            continue
        series = _href_to_radian(samples_df[col])
        samples_df[col] = series
    return samples_df


def _href_to_radian(opposite, f=15_000):
    """Convert HREF eyegaze samples to radians.

    Parameters
    ----------
    opposite : int
        The x or y coordinate in an HREF gaze sample.
    f : int (default 15_000)
        distance of plane from the eye. Defaults to 15,000 units, which was taken
        from the Eyelink 1000 plus user manual.

    Returns
    -------
    x or y coordinate in radians

    Notes
    -----
    See section 4.4.2.2 in the Eyelink 1000 Plus User Manual
    (version 1.0.19) for a detailed description of HREF data.
    """
    return np.arcsin(opposite / f)


def _create_info(ch_names, raw_extras):
    """Create info object for RawEyelink."""
    # assign channel type from ch_name
    pos_names = EYELINK_COLS["pos"]["left"][:-1] + EYELINK_COLS["pos"]["right"][:-1]
    pupil_names = EYELINK_COLS["pos"]["left"][-1] + EYELINK_COLS["pos"]["right"][-1]
    ch_types = [
        "eyegaze"
        if ch in pos_names
        else "pupil"
        if ch in pupil_names
        else "stim"
        if ch == "DIN"
        else "misc"
        for ch in ch_names
    ]
    info = create_info(ch_names, raw_extras["sfreq"], ch_types)
    # set correct loc for eyepos and pupil channels
    for ch_dict in info["chs"]:
        # loc index 3 can indicate left or right eye
        if ch_dict["ch_name"].endswith("left"):  # [x,y,pupil]_left
            ch_dict["loc"][3] = -1  # left eye
        elif ch_dict["ch_name"].endswith("right"):  # [x,y,pupil]_right
            ch_dict["loc"][3] = 1  # right eye
        else:
            logger.debug(
                f"leaving index 3 of loc array as"
                f" {ch_dict['loc'][3]} for {ch_dict['ch_name']}"
            )
        # loc index 4 can indicate x/y coord
        if ch_dict["ch_name"].startswith("x"):
            ch_dict["loc"][4] = -1  # x-coord
        elif ch_dict["ch_name"].startswith("y"):
            ch_dict["loc"][4] = 1  # y-coord
        else:
            logger.debug(
                f"leaving index 4 of loc array as"
                f" {ch_dict['loc'][4]} for {ch_dict['ch_name']}"
            )
        if "HREF" in raw_extras["rec_info"]:
            if ch_dict["ch_name"].startswith(("xpos", "ypos")):
                ch_dict["unit"] = FIFF.FIFF_UNIT_RAD
    return info


def _make_eyelink_annots(df_dict, create_annots, apply_offsets):
    """Create Annotations for each df in raw_extras."""
    eye_ch_map = {
        "L": ("xpos_left", "ypos_left", "pupil_left"),
        "R": ("xpos_right", "ypos_right", "pupil_right"),
        "both": (
            "xpos_left",
            "ypos_left",
            "pupil_left",
            "xpos_right",
            "ypos_right",
            "pupil_right",
        ),
    }
    valid_descs = ["blinks", "saccades", "fixations", "messages"]
    msg = (
        "create_annotations must be True or a list containing one or"
        f" more of {valid_descs}."
    )
    wrong_type = msg + f" Got a {type(create_annots)} instead."
    if create_annots is True:
        descs = valid_descs
    else:
        if not isinstance(create_annots, list):
            raise TypeError(wrong_type)
        for desc in create_annots:
            if desc not in valid_descs:
                raise ValueError(msg + f" Got '{desc}' instead")
        descs = create_annots

    annots = None
    for key, df in df_dict.items():
        eye_annot_cond = (key in ["blinks", "fixations", "saccades"]) and (key in descs)
        if eye_annot_cond:
            onsets = df["time"]
            durations = df["duration"]
            # Create annotations for both eyes
            descriptions = key[:-1]  # i.e "blink", "fixation", "saccade"
            if key == "blinks":
                descriptions = "BAD_" + descriptions
            ch_names = df["eye"].map(eye_ch_map).tolist()
            this_annot = Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                ch_names=ch_names,
            )
        elif (key in ["messages"]) and (key in descs):
            if apply_offsets:
                # If df['offset] is all NaNs, time is not changed
                onsets = df["time"] + df["offset"].fillna(0)
            else:
                onsets = df["time"]
            durations = [0] * onsets
            descriptions = df["event_msg"]
            this_annot = Annotations(
                onset=onsets, duration=durations, description=descriptions
            )
        else:
            continue  # TODO make df and annotations for Buttons
        if not annots:
            annots = this_annot
        elif annots:
            annots += this_annot
    if not annots:
        warn(f"Annotations for {descs} were requested but none could be made.")
        return
    return annots


def _make_gap_annots(raw_extras, key="recording_blocks"):
    """Create Annotations for gap periods between recording blocks."""
    df = raw_extras["dfs"][key]
    onsets = df["end_time"].iloc[:-1]
    diffs = df["time"].shift(-1) - df["end_time"]
    durations = diffs.iloc[:-1]
    descriptions = ["BAD_ACQ_SKIP"] * len(onsets)
    return Annotations(onset=onsets, duration=durations, description=descriptions)


# ======================== Used by read_eyelink-calibration ===========================


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
                onset=onset,
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
