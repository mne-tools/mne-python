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
    "messages": ("time", "offset", "event_msg"),
}


def _parse_eyelink_ascii(
    fname, find_overlaps=True, overlap_threshold=0.05, apply_offsets=False
):
    # ======================== Parse ASCII File =========================
    raw_extras = dict()
    raw_extras["dt"] = _get_recording_datetime(fname)
    data_blocks: list[dict] = _parse_recording_blocks(fname)
    _validate_data(data_blocks)

    # ======================== Create DataFrames ========================
    # Process each block individually, then combine
    processed_blocks = _create_dataframes(data_blocks, apply_offsets)
    raw_extras["dfs"], ch_names = _combine_block_dataframes(processed_blocks)
    del processed_blocks  # free memory
    for block in data_blocks:
        del block["samples"]  # remove samples from block to save memory

    first_block = data_blocks[0]
    raw_extras["pos_unit"] = first_block["info"]["unit"]
    raw_extras["sfreq"] = first_block["info"]["sfreq"]
    raw_extras["first_timestamp"] = first_block["info"]["first_timestamp"]
    raw_extras["n_blocks"] = len(data_blocks)
    # if HREF data, convert to radians
    if raw_extras["pos_unit"] == "HREF":
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
        df = _convert_times(df, raw_extras["first_timestamp"])
    # Find overlaps between left and right eye events
    if find_overlaps:
        for key in raw_extras["dfs"]:
            if key not in ["blinks", "fixations", "saccades"]:
                continue
            raw_extras["dfs"][key] = _find_overlaps(
                raw_extras["dfs"][key], max_time=overlap_threshold
            )
    # ======================== Info for BaseRaw ========================
    dfs = raw_extras["dfs"]

    if "samples" not in dfs or dfs["samples"].empty:
        logger.info("No sample data found, creating empty Raw object.")
        eye_ch_data = np.empty((len(ch_names), 0))
    else:
        eye_ch_data = dfs["samples"][ch_names].to_numpy().T

    info = _create_info(ch_names, raw_extras)

    return eye_ch_data, info, raw_extras


def _parse_recording_blocks(fname):
    """Parse Eyelink ASCII file.

    Eyelink samples occur within START and END blocks.
    samples lines start with a posix-like string,
    and contain eyetracking sample info. Event Lines
    start with an upper case string and contain info
    about ocular events (i.e. blink/saccade), or experiment
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
        data_blocks = []

        is_recording_block = False
        for line in file:
            if line.startswith("START"):  # start of recording block
                is_recording_block = True
                # Initialize container for new block data
                current_block = {
                    "samples": [],
                    "events": {
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
                    },
                    "info": None,
                }
            if is_recording_block:
                tokens = line.split()
                if not tokens:
                    continue  # skip empty lines
                if tokens[0][0].isnumeric():  # Samples
                    current_block["samples"].append(tokens)
                elif tokens[0] in current_block["events"].keys():
                    if _is_sys_msg(line):
                        continue  # system messages don't need to be parsed.
                    event_key, event_info = tokens[0], tokens[1:]
                    current_block["events"][event_key].append(event_info)
                    if tokens[0] == "END":  # end of recording block
                        current_block["info"] = _get_metadata(current_block)
                        data_blocks.append(current_block)
                        is_recording_block = False
        if not data_blocks:  # no samples parsed
            raise ValueError(f"Couldn't find any samples in {fname}")
        return data_blocks


def _validate_data(data_blocks: list):
    """Check the incoming data for some known problems that can occur."""
    # Detect the datatypes that are in file.
    units = []
    pupil_units = []
    modes = []
    eyes = []
    sfreqs = []
    for block in data_blocks:
        units.append(block["info"]["unit"])
        modes.append(block["info"]["tracking_mode"])
        eyes.append(block["info"]["eye"])
        sfreqs.append(block["info"]["sfreq"])
        pupil_units.append(block["info"]["pupil_unit"])
    if "GAZE" in units:
        logger.info(
            "Pixel coordinate data detected. "
            "Pass `scalings=dict(eyegaze=1e3)` when using plot"
            " method to make traces more legible."
        )
    if "HREF" in units:
        logger.info("Head-referenced eye-angle (HREF) data detected.")
    elif "PUPIL" in units:
        warn("Raw eyegaze coordinates detected. Analyze with caution.")
    if "AREA" in pupil_units:
        logger.info("Pupil-size area detected.")
    elif "DIAMETER" in pupil_units:
        logger.info("Pupil-size diameter detected.")

    if len(set(modes)) > 1:
        warn(
            "This recording switched between monocular and binocular tracking. "
            f"In order of acquisition blocks, tracking modes were {modes}. Data "
            "for the missing eye during monocular tracking will be filled with NaN."
        )
    # Monocular tracking but switched between left/right eye
    elif len(set(eyes)) > 1:
        warn(
            "The eye being tracked changed during the recording. "
            f"In order of acquisition blocks, they were {eyes}. "
            "Missing data for each eye will be filled with NaN."
        )
    if len(set(sfreqs)) > 1:
        raise RuntimeError(
            "The sampling frequency changed during the recording. "
            f"In order of acquisition blocks, they were {sfreqs}. "
            "please notify MNE-Python developers"
        )  # pragma: no cover
    if len(set(units)) > 1:
        raise RuntimeError(
            "The unit of measurement for x/y coordinates changed during the recording. "
            f"In order of acquisition blocks, they were {units}. "
            "please notify MNE-Python developers"
        )  # pragma: no cover


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


def _get_metadata(data_block: dict):
    """Get tracking mode, sfreq, eye tracked, pupil metric, etc. for one data block."""
    meta_data = dict()
    rec_info = data_block["events"]["SAMPLES"][0]
    meta_data["unit"] = rec_info[0]

    # If the file doesn't have pupil data, i'm not sure if there will be any PUPIL info?
    if not data_block["events"]["PUPIL"]:
        ps_unit = None
    else:
        ps_unit = data_block["events"]["PUPIL"][0][0]
    meta_data["pupil_unit"] = ps_unit
    if ("LEFT" in rec_info) and ("RIGHT" in rec_info):
        meta_data["tracking_mode"] = "binocular"
        meta_data["eye"] = "both"
    else:
        meta_data["tracking_mode"] = "monocular"
        meta_data["eye"] = rec_info[1].lower()
    meta_data["first_timestamp"] = float(data_block["events"]["START"][0][0])
    meta_data["last_timestamp"] = float(data_block["events"]["END"][0][0])
    meta_data["sfreq"] = _get_sfreq_from_ascii(rec_info)
    meta_data["rec_info"] = data_block["events"]["SAMPLES"][0]
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


def _create_dataframes(data_blocks, apply_offsets):
    """Create and process pandas DataFrames for each recording block.

    Processes each block individually with its own column structure,
    then returns a list of processed block dataframes.
    """
    processed_blocks = []

    for block_idx, block in enumerate(data_blocks):
        # Create dataframes for this block
        block_dfs = _create_dataframes_for_block(block, apply_offsets)

        # Infer column names for this specific block
        col_names, ch_names = _infer_col_names_for_block(block)

        # Assign column names and set dtypes for this block
        block_dfs = _assign_col_names(col_names, block_dfs)
        block_dfs = _set_df_dtypes(block_dfs)

        processed_blocks.append(
            {
                "block_idx": block_idx,
                "dfs": block_dfs,
                "ch_names": ch_names,
                "info": block["info"],
            }
        )
    return processed_blocks


def _create_dataframes_for_block(block, apply_offsets):
    """Create pandas.DataFrame for one recording block's samples and events.

    Creates a pandas DataFrame for sample_lines and for each
    non-empty key in event_lines for a single recording block.
    No column names are assigned at this point.
    This also returns the MNE channel names needed to represent this block of data.
    """
    pd = _check_pandas_installed()
    df_dict = dict()

    # dataframe for samples in this block
    if block["samples"]:
        df_dict["samples"] = pd.DataFrame(block["samples"])
        df_dict["samples"] = _drop_status_col(df_dict["samples"])  # drop STATUS col

    # dataframe for each type of ocular event in this block
    for event, label in zip(
        ["EFIX", "ESACC", "EBLINK"], ["fixations", "saccades", "blinks"]
    ):
        if block["events"][event]:  # an empty list returns False
            df_dict[label] = pd.DataFrame(block["events"][event])
        else:
            # Changed this from info to debug level to avoid spamming the log
            logger.debug(f"No {label} events found in block")

    # make dataframe for experiment messages in this block
    if block["events"]["MSG"]:
        msgs = []
        for token in block["events"]["MSG"]:
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

    # make dataframes for other button events
    if block["events"]["BUTTON"]:
        button_events = block["events"]["BUTTON"]
        parsed = []
        for entry in button_events:
            parsed.append(
                {
                    "time": float(entry[0]),  # onset
                    "button_id": int(entry[1]),
                    "button_pressed": int(entry[2]),  # 1 = press, 0 = release
                }
            )
        df_dict["buttons"] = pd.DataFrame(parsed)
        n_button = len(df_dict.get("buttons", []))
        logger.info(f"Found {n_button} button event(s) in this file.")
    else:
        logger.info("No button events found in this file.")

    return df_dict


def _infer_col_names_for_block(block: dict) -> tuple[dict[str, list], list]:
    """Build column and channel names for data from one Eyelink recording block.

    Returns the expected column names for the sample lines and event
    lines for a single recording block. The columns present can vary
    between blocks if tracking mode changes.
    """
    col_names = {}
    block_info = block["info"]

    # initiate the column names for the sample lines
    col_names["samples"] = list(EYELINK_COLS["timestamp"])
    col_names["messages"] = list(EYELINK_COLS["messages"])

    # and for the eye message lines
    col_names["blinks"] = list(EYELINK_COLS["eye_event"])
    col_names["fixations"] = list(EYELINK_COLS["eye_event"] + EYELINK_COLS["fixation"])
    col_names["saccades"] = list(EYELINK_COLS["eye_event"] + EYELINK_COLS["saccade"])

    # Get block-specific tracking info
    tracking_mode = block_info["tracking_mode"]
    eye = block_info["eye"]
    rec_info = block["events"]["SAMPLES"][0]  # SAMPLES line for this block

    # Recording was either binocular or monocular for this block
    if tracking_mode == "monocular":
        ch_names = list(EYELINK_COLS["pos"][eye])
    elif tracking_mode == "binocular":
        ch_names = list(EYELINK_COLS["pos"]["left"] + EYELINK_COLS["pos"]["right"])
    col_names["samples"].extend(ch_names)

    # The order of these if statements should not be changed.
    if "VEL" in rec_info:  # If velocity data are reported
        if tracking_mode == "monocular":
            ch_names.extend(EYELINK_COLS["velocity"][eye])
            col_names["samples"].extend(EYELINK_COLS["velocity"][eye])
        elif tracking_mode == "binocular":
            ch_names.extend(
                EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
            )
            col_names["samples"].extend(
                EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
            )
    # if resolution data are reported
    if "RES" in rec_info:
        ch_names.extend(EYELINK_COLS["resolution"])
        col_names["samples"].extend(EYELINK_COLS["resolution"])
        col_names["fixations"].extend(EYELINK_COLS["resolution"])
        col_names["saccades"].extend(EYELINK_COLS["resolution"])
    # if digital input port values are reported
    if "INPUT" in rec_info:
        ch_names.extend(EYELINK_COLS["input"])
        col_names["samples"].extend(EYELINK_COLS["input"])

    # if head target info was reported, add its cols
    if "HTARGET" in rec_info:
        ch_names.extend(EYELINK_COLS["remote"])
        col_names["samples"].extend(EYELINK_COLS["remote"])

    return col_names, ch_names


def _combine_block_dataframes(processed_blocks: list[dict]):
    """Combine dataframes across acquisition blocks.

    Handles cases where blocks have different columns/data in them
    (e.g. binocular vs monocular tracking, or switching between the left and right eye).
    """
    pd = _check_pandas_installed()

    # Determine unified column structure by collecting all unique column names
    # across all acquisition blocks
    all_ch_names = []
    all_samples_cols = set()
    all_df_types = set()

    for block in processed_blocks:
        # The tests assume a certain order of channel names.
        # so we can't use a set like we do for the columns.
        # bc it randomly orders the channel names.
        for ch_name in block["ch_names"]:
            if ch_name not in all_ch_names:
                all_ch_names.append(ch_name)
        if "samples" in block["dfs"]:
            all_samples_cols.update(block["dfs"]["samples"].columns)
        all_df_types.update(block["dfs"].keys())

    # The sets randomly ordered the column names.
    all_samples_cols = sorted(all_samples_cols)

    # Combine dataframes by type
    combined_dfs = {}

    for df_type in all_df_types:
        block_dfs = []
        for block in processed_blocks:
            if df_type in block["dfs"]:
                # We will update the dfs in-place to conserve memory
                block_df = block["dfs"][df_type]

                # For samples dataframes, ensure all have the same columns
                if df_type == "samples":
                    for col in all_samples_cols:
                        if col not in block_df.columns:
                            block_df[col] = np.nan

                    # Reorder columns
                    block_df = block_df[all_samples_cols]

                block_dfs.append(block_df)

        if block_dfs:
            # Concatenate all blocks for this dataframe type
            combined_dfs[df_type] = pd.concat(block_dfs, ignore_index=True)

    # Create recording blocks dataframe from block info
    blocks_data = []
    for i, block in enumerate(processed_blocks):
        start_time = block["info"]["first_timestamp"]
        end_time = block["info"]["last_timestamp"]
        blocks_data.append((start_time, end_time, i + 1))
    combined_dfs["recording_blocks"] = pd.DataFrame(
        blocks_data, columns=["time", "end_time", "block"]
    )

    return combined_dfs, all_ch_names


def _drop_status_col(samples_df):
    """Drop STATUS column from samples dataframe.

    see https://github.com/mne-tools/mne-python/issues/11809, and section 4.9.2.1 of
    the Eyelink 1000 Plus User Manual, version 1.0.19. We know that the STATUS
    column is either 3, 5, 13, or 17 characters long, i.e. "...", ".....", ".C."
    """
    status_cols = []
    # we know the first 3 columns will be the time, xpos, ypos
    for col in samples_df.columns[3:]:
        # use first valid index and value to ignore leading empty values
        # see https://github.com/mne-tools/mne-python/issues/13567
        first_valid_index = samples_df[col].first_valid_index()
        if first_valid_index is None:
            # The entire column is NaN, so we can drop it
            status_cols.append(col)
            continue
        first_value = samples_df.loc[first_valid_index, col]
        try:
            float(first_value)
            continue  # if the value is numeric, it's not a status column
        except (ValueError, TypeError):
            # cannot convert to float, so it might be a status column
            # further check the length of the string value
            if len(first_value) in [3, 5, 13, 17]:
                status_cols.append(col)
    return samples_df.drop(columns=status_cols)


def _assign_col_names(col_names, df_dict):
    """Assign column names to dataframes.

    Parameters
    ----------
    col_names : dict of str to list
        Dictionary of column names for each dataframe.
    df_dict : dict of str to pandas.DataFrame
        Dictionary of dataframes to assign column names to.
    """
    skipped_types = []
    for key, df in df_dict.items():
        if key in ("samples", "blinks", "fixations", "saccades", "messages"):
            cols = col_names[key]
        else:
            skipped_types.append(key)
            continue
        max_cols = len(cols)
        if len(df.columns) != len(cols):
            if key in ("saccades", "fixations") and len(df.columns) >= 4:
                # see https://github.com/mne-tools/mne-python/pull/13357
                logger.debug(
                    f"{key} events have more columns ({len(df.columns)}) than  "
                    f"expected ({len(cols)}). Using first 4 (eye, time, end_time, "
                    "duration)."
                )
                max_cols = 4
            else:
                raise ValueError(
                    f"Expected the {key} data in this file to have {len(cols)} columns "
                    f"of data, but got {len(df.columns)}. Expected columns: {cols}."
                )
        new_col_names = {
            old: new for old, new in zip(df.columns[:max_cols], cols[:max_cols])
        }
        df.rename(columns=new_col_names, inplace=True)
    logger.debug(f"Skipped assigning column names to {skipped_types} dataframes.")
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
        if str(col).endswith("time"):  # 'time' and 'end_time' cols
            df[col] -= first_samp
            df[col] /= 1000
        if str(col) in ["duration", "offset"]:
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
    This would cause the ocular annotations (i.e. blinks) to not line up with
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
        Pandas DataFrame with ocular events (fixations, saccades, blinks)
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
        if raw_extras["pos_unit"] == "HREF":
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
    valid_descs = ["blinks", "saccades", "fixations", "buttons", "messages"]
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
        elif (key == "buttons") and (key in descs):
            required_cols = {"time", "button_id", "button_pressed"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing column: {required_cols - set(df.columns)}")
            # Give user a hint
            n_presses = df["button_pressed"].sum()
            logger.info("Found %d button press events.", n_presses)

            df = df.sort_values("time")
            onsets = df["time"]
            durations = np.zeros_like(onsets)
            descriptions = df.apply(_get_button_description, axis=1)

            this_annot = Annotations(
                onset=onsets, duration=durations, description=descriptions
            )
        else:
            continue
        if not annots:
            annots = this_annot
        elif annots:
            annots += this_annot
    if not annots:
        warn(f"Annotations for {descs} were requested but none could be made.")
        return

    return annots


def _get_button_description(row):
    button_id = int(row["button_id"])
    action = "press" if row["button_pressed"] == 1 else "release"
    return f"button_{button_id}_{action}"


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
            line_idx = line_number + 1
            read_points = 0
            while read_points < n_points and line_idx < len(lines):
                subline = lines[line_idx].strip()
                line_idx += 1

                if not subline or "!CAL VALIDATION" in subline:
                    continue  # for bino mode, skip the second eye's validation summary

                subline_eye = subline.split("at")[0].split()[-1].lower()  # e.g. 'left'
                if subline_eye != this_eye:
                    continue  # skip the validation lines for the other eye
                point_info = _parse_validation_line(subline)
                points.append(point_info)
                read_points += 1
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
