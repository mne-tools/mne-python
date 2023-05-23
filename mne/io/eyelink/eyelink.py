# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: BSD-3-Clause

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from ..constants import FIFF
from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, _check_pandas_installed

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
    "flags": ("flags",),
    "remote": ("x_head", "y_head", "distance"),
    "remote_flags": ("head_flags",),
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


def _isfloat(token):
    """Boolean test for whether string can be of type float.

    Parameters
    ----------
    token : str
        Single element from tokens list.
    """
    if isinstance(token, str):
        try:
            float(token)
            return True
        except ValueError:
            return False
    else:
        raise ValueError(
            "input should be a string," f" but {token} is of type {type(token)}"
        )


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
    if len(line):
        tokens = line.split()
        return _convert_types(tokens)
    else:
        raise ValueError("line is empty, nothing to parse")


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
    return any(["!V" in line, "!MODE" in line, ";" in line])


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
    for i, token in enumerate(rec_info):
        if token == "RATE":
            # sfreq is the first token after RATE
            return rec_info[i + 1]


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


@fill_doc
def read_raw_eyelink(
    fname,
    preload=False,
    verbose=None,
    create_annotations=True,
    apply_offsets=False,
    find_overlaps=False,
    overlap_threshold=0.05,
    gap_description="bad_rec_gap",
):
    """Reader for an Eyelink .asc file.

    Parameters
    ----------
    fname : str
        Path to the eyelink file (.asc).
    %(preload)s
    %(verbose)s
    create_annotations : bool | list (default True)
        Whether to create mne.Annotations from occular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ['fixations', 'saccades',' blinks', messages'].
        If True, creates mne.Annotations for both occular events and experiment
        messages.
    apply_offsets : bool (default False)
        Adjusts the onset time of the mne.Annotations created from Eyelink
        experiment messages, if offset values exist in
        self.dataframes['messages'].
    find_overlaps : bool (default False)
        Combine left and right eye :class:`mne.Annotations` (blinks, fixations,
        saccades) if their start times and their stop times are both not
        separated by more than overlap_threshold.
    overlap_threshold : float (default 0.05)
        Time in seconds. Threshold of allowable time-gap between the start and
        stop times of the left and right eyes. If gap is larger than threshold,
        the :class:`mne.Annotations` will be kept separate (i.e. "blink_L",
        "blink_R"). If the gap is smaller than the threshold, the
        :class:`mne.Annotations` will be merged (i.e. "blink_both").
    gap_description : str (default 'bad_rec_gap')
        If there are multiple recording blocks in the file, the description of
        the annotation that will span across the gap period between the
        blocks. Uses 'bad_rec_gap' by default so that these time periods will
        be considered bad by MNE and excluded from operations like epoching.

    Returns
    -------
    raw : instance of RawEyelink
        A Raw object containing eyetracker data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    extension = Path(fname).suffix
    if extension not in ".asc":
        raise ValueError(
            "This reader can only read eyelink .asc files."
            f" Got extension {extension} instead. consult eyelink"
            " manual for converting eyelink data format (.edf)"
            " files to .asc format."
        )

    return RawEyelink(
        fname,
        preload=preload,
        verbose=verbose,
        create_annotations=create_annotations,
        apply_offsets=apply_offsets,
        find_overlaps=find_overlaps,
        overlap_threshold=overlap_threshold,
        gap_desc=gap_description,
    )


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    fname : str
        Path to the data file (.XXX).
    create_annotations : bool | list (default True)
        Whether to create mne.Annotations from occular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ['fixations', 'saccades',' blinks', messages'].
        If True, creates mne.Annotations for both occular events and experiment
        messages.
    apply_offsets : bool (default False)
        Adjusts the onset time of the mne.Annotations created from Eyelink
        experiment messages, if offset values exist in
        raw.dataframes['messages'].
     find_overlaps : boolean (default False)
        Combine left and right eye :class:`mne.Annotations` (blinks, fixations,
        saccades) if their start times and their stop times are both not
        separated by more than overlap_threshold.
    overlap_threshold : float (default 0.05)
        Time in seconds. Threshold of allowable time-gap between the start and
        stop times of the left and right eyes. If gap is larger than threshold,
        the :class:`mne.Annotations` will be kept separate (i.e. "blink_L",
        "blink_R"). If the gap is smaller than the threshold, the
        :class:`mne.Annotations` will be merged (i.e. "blink_both").
    gap_desc : str (default 'bad_rec_gap')
        If there are multiple recording blocks in the file, the description of
        the annotation that will span across the gap period between the
        blocks. Uses 'bad_rec_gap' by default so that these time periods will
        be considered bad by MNE and excluded from operations like epoching.
    %(preload)s
    %(verbose)s

    Attributes
    ----------
    fname : pathlib.Path
        Eyelink filename
    dataframes : dict
        Dictionary of pandas DataFrames. One for eyetracking samples,
        and one for each type of eyelink event (blinks, messages, etc)
    _sample_lines : list
        List of lists, each list is one sample containing eyetracking
        X/Y and pupil channel data (+ other channels, if they exist)
    _event_lines : dict
        Each key contains a list of lists, for an event-type that occurred
        during the recording period. Events can vary, from occular events
        (blinks, saccades, fixations), to messages from the stimulus
        presentation software, or info from a response controller.
    _system_lines : list
        List of tab delimited strings. Each string is a system message,
        that in most cases aren't needed. System messages occur for
        Eyelinks DataViewer application.
    _tracking_mode : str
        Whether whether a single eye was tracked ('monocular'), or both
        ('binocular').
    _gap_desc : str
        The description to be used for annotations returned by _make_gap_annots

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(
        self,
        fname,
        preload=False,
        verbose=None,
        create_annotations=True,
        apply_offsets=False,
        find_overlaps=False,
        overlap_threshold=0.05,
        gap_desc="bad_rec_gap",
    ):
        logger.info("Loading {}".format(fname))

        self.fname = Path(fname)
        self._sample_lines = None
        self._event_lines = None
        self._system_lines = None
        self._tracking_mode = None  # assigned in self._infer_col_names
        self._meas_date = None
        self._rec_info = None
        self._gap_desc = gap_desc
        self.dataframes = {}

        self._get_recording_datetime()  # sets self._meas_date
        self._parse_recording_blocks()  # sets sample, event, & system lines

        sfreq = _get_sfreq(self._event_lines["SAMPLES"][0])
        col_names, ch_names = self._infer_col_names()
        self._create_dataframes(
            col_names, sfreq, find_overlaps=find_overlaps, threshold=overlap_threshold
        )
        info = self._create_info(ch_names, sfreq)
        eye_ch_data = self.dataframes["samples"][ch_names]
        eye_ch_data = eye_ch_data.to_numpy().T

        # create mne object
        super(RawEyelink, self).__init__(
            info, preload=eye_ch_data, filenames=[self.fname], verbose=verbose
        )
        # set meas_date
        self.set_meas_date(self._meas_date)

        # Make Annotations
        gap_annots = None
        if len(self.dataframes["recording_blocks"]) > 1:
            gap_annots = self._make_gap_annots()
        eye_annots = None
        if create_annotations:
            eye_annots = self._make_eyelink_annots(
                self.dataframes, create_annotations, apply_offsets
            )
        if gap_annots and eye_annots:  # set both
            self.set_annotations(gap_annots + eye_annots)
        elif gap_annots:
            self.set_annotations(gap_annots)
        elif eye_annots:
            self.set_annotations(eye_annots)
        else:
            logger.info("Not creating any annotations")

    def _parse_recording_blocks(self):
        """Parse Eyelink ASCII file.

        Eyelink samples occur within START and END blocks.
        samples lines start with a posix-like string,
        and contain eyetracking sample info. Event Lines
        start with an upper case string and contain info
        about occular events (i.e. blink/saccade), or experiment
        messages sent by the stimulus presentation software.
        """
        with self.fname.open() as file:
            block_num = 1
            self._sample_lines = []
            self._event_lines = {
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
            self._system_lines = []

            is_recording_block = False
            for line in file:
                if line.startswith("START"):  # start of recording block
                    is_recording_block = True
                if is_recording_block:
                    if _is_sys_msg(line):
                        self._system_lines.append(line)
                        continue  # system messages don't need to be parsed.
                    tokens = _parse_line(line)
                    tokens.append(block_num)  # add current block number
                    if isinstance(tokens[0], (int, float)):  # Samples
                        self._sample_lines.append(tokens)
                    elif tokens[0] in self._event_lines.keys():
                        event_key, event_info = tokens[0], tokens[1:]
                        self._event_lines[event_key].append(event_info)
                    if tokens[0] == "END":  # end of recording block
                        is_recording_block = False
                        block_num += 1
            if not self._event_lines["START"]:
                raise ValueError(
                    "Could not determine the start of the"
                    " recording. When converting to ASCII, START"
                    " events should not be suppressed."
                )
            if not self._sample_lines:  # no samples parsed
                raise ValueError(f"Couldn't find any samples in {self.fname}")
            self._validate_data()

    def _validate_data(self):
        """Check the incoming data for some known problems that can occur."""
        self._rec_info = self._event_lines["SAMPLES"][0]
        pupil_info = self._event_lines["PUPIL"][0]
        n_blocks = len(self._event_lines["START"])
        sfreq = int(_get_sfreq(self._rec_info))
        first_samp = self._event_lines["START"][0][0]
        if ("LEFT" in self._rec_info) and ("RIGHT" in self._rec_info):
            self._tracking_mode = "binocular"
        else:
            self._tracking_mode = "monocular"
        # Detect the datatypes that are in file.
        if "GAZE" in self._rec_info:
            logger.info("Pixel coordinate data detected.")
            logger.warning(
                "Pass `scalings=dict(eyegaze=1e3)` when using plot"
                " method to make traces more legible."
            )
        elif "HREF" in self._rec_info:
            logger.info("Head-referenced eye angle data detected.")
        elif "PUPIL" in self._rec_info:
            logger.warning("Raw eyegaze coordinates detected. Analyze with" " caution.")
        if "AREA" in pupil_info:
            logger.info("Pupil-size area reported.")
        elif "DIAMETER" in pupil_info:
            logger.info("Pupil-size diameter reported.")
        # Check sampling frequency.
        if sfreq == 2000 and isinstance(first_samp, int):
            raise ValueError(
                f"The sampling rate is {sfreq}Hz but the"
                " timestamps were not output as float values."
                " Check the settings in the EDF2ASC application."
            )
        elif sfreq != 2000 and isinstance(first_samp, float):
            raise ValueError(
                "For recordings with a sampling rate less than"
                " 2000Hz, timestamps should not be output to the"
                " ASCII file as float values. Check the"
                " settings in the EDF2ASC application. Got a"
                f" sampling rate of {sfreq}Hz."
            )
        # If more than 1 recording period, make sure sfreq didn't change.
        if n_blocks > 1:
            err_msg = (
                "The sampling frequency changed during the recording."
                " This file cannot be read into MNE."
            )
            for block_info in self._event_lines["SAMPLES"][1:]:
                block_sfreq = int(_get_sfreq(block_info))
                if block_sfreq != sfreq:
                    raise ValueError(
                        err_msg + f" Got both {sfreq} and {block_sfreq} Hz."
                    )
            if self._tracking_mode == "monocular":
                assert self._rec_info[1] in ["LEFT", "RIGHT"]
                eye = self._rec_info[1]
                blocks_list = self._event_lines["SAMPLES"]
                eye_per_block = [block_info[1] for block_info in blocks_list]
                if not all([this_eye == eye for this_eye in eye_per_block]):
                    logger.warning(
                        "The eye being tracked changed during the"
                        " recording. The channel names will reflect"
                        " the eye that was tracked at the start of"
                        " the recording."
                    )

    def _get_recording_datetime(self):
        """Create a datetime object from the datetime in ASCII file."""
        # create a timezone object for UTC
        tz = timezone(timedelta(hours=0))
        in_header = False
        with self.fname.open() as file:
            for line in file:
                # header lines are at top of file and start with **
                if line.startswith("**"):
                    in_header = True
                if in_header:
                    if line.startswith("** DATE:"):
                        dt_str = line.replace("** DATE:", "").strip()
                        fmt = "%a %b %d %H:%M:%S %Y"
                        try:
                            # Eyelink measdate timestamps are timezone naive.
                            # Force datetime to be in UTC.
                            # Even though dt is probably in local time zone.
                            dt_naive = datetime.strptime(dt_str, fmt)
                            dt_aware = dt_naive.replace(tzinfo=tz)
                            self._meas_date = dt_aware
                        except Exception:
                            msg = (
                                "Extraction of measurement date failed."
                                " Please report this as a github issue."
                                " The date is being set to None"
                            )
                            logger.warning(msg)
                        break

    def _href_to_radian(self, opposite, f=15_000):
        """Convert HREF eyegaze samples to radians.

        Parameters
        ----------
        opposite : int
            The x or y coordinate in an HREF gaze sample.
        f : int (default 15_000)
            distance of plane from the eye.

        Returns
        -------
        x or y coordinate in radians

        Notes
        -----
        See section 4.4.2.2 in the Eyelink 1000 Plus User Manual
        (version 1.0.19) for a detailed description of HREF data.
        """
        return np.arcsin(opposite / f)

    def _infer_col_names(self):
        """Build column and channel names for data from Eyelink ASCII file.

        Returns the expected column names for the sample lines and event
        lines, to be passed into pd.DataFrame. Sample and event lines in
        eyelink files have a fixed order of columns, but the columns that
        are present can vary. The order that col_names is built below should
        NOT change.
        """
        col_names = {}
        # initiate the column names for the sample lines
        col_names["sample"] = list(EYELINK_COLS["timestamp"])

        # and for the eye message lines
        col_names["blink"] = list(EYELINK_COLS["eye_event"])
        col_names["fixation"] = list(
            EYELINK_COLS["eye_event"] + EYELINK_COLS["fixation"]
        )
        col_names["saccade"] = list(EYELINK_COLS["eye_event"] + EYELINK_COLS["saccade"])

        # Recording was either binocular or monocular
        # If monocular, find out which eye was tracked and append to ch_name
        if self._tracking_mode == "monocular":
            assert self._rec_info[1] in ["LEFT", "RIGHT"]
            eye = self._rec_info[1].lower()
            ch_names = list(EYELINK_COLS["pos"][eye])
        elif self._tracking_mode == "binocular":
            ch_names = list(EYELINK_COLS["pos"]["left"] + EYELINK_COLS["pos"]["right"])
        col_names["sample"].extend(ch_names)

        # The order of these if statements should not be changed.
        if "VEL" in self._rec_info:  # If velocity data are reported
            if self._tracking_mode == "monocular":
                ch_names.extend(EYELINK_COLS["velocity"][eye])
                col_names["sample"].extend(EYELINK_COLS["velocity"][eye])
            elif self._tracking_mode == "binocular":
                ch_names.extend(
                    EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
                )
                col_names["sample"].extend(
                    EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
                )
        # if resolution data are reported
        if "RES" in self._rec_info:
            ch_names.extend(EYELINK_COLS["resolution"])
            col_names["sample"].extend(EYELINK_COLS["resolution"])
            col_names["fixation"].extend(EYELINK_COLS["resolution"])
            col_names["saccade"].extend(EYELINK_COLS["resolution"])
        # if digital input port values are reported
        if "INPUT" in self._rec_info:
            ch_names.extend(EYELINK_COLS["input"])
            col_names["sample"].extend(EYELINK_COLS["input"])

        # add flags column
        col_names["sample"].extend(EYELINK_COLS["flags"])

        # if head target info was reported, add its cols after flags col.
        if "HTARGET" in self._rec_info:
            ch_names.extend(EYELINK_COLS["remote"])
            col_names["sample"].extend(
                EYELINK_COLS["remote"] + EYELINK_COLS["remote_flags"]
            )

        # finally add a column for recording block number
        # FYI this column does not exist in the asc file..
        # but it is added during _parse_recording_blocks
        for col in col_names.values():
            col.extend(EYELINK_COLS["block_num"])

        return col_names, ch_names

    def _create_dataframes(self, col_names, sfreq, find_overlaps=False, threshold=0.05):
        """Create pandas.DataFrame for Eyelink samples and events.

        Creates a pandas DataFrame for self._sample_lines and for each
        non-empty key in self._event_lines.
        """
        pd = _check_pandas_installed()

        # First sample should be the first line of the first recording block
        first_samp = self._event_lines["START"][0][0]

        # dataframe for samples
        self.dataframes["samples"] = pd.DataFrame(
            self._sample_lines, columns=col_names["sample"]
        )
        if "HREF" in self._rec_info:
            pos_names = (
                EYELINK_COLS["pos"]["left"][:-1] + EYELINK_COLS["pos"]["right"][:-1]
            )
            for col in self.dataframes["samples"].columns:
                if col not in pos_names:  # 'xpos_left' ... 'ypos_right'
                    continue
                series = self._href_to_radian(self.dataframes["samples"][col])
                self.dataframes["samples"][col] = series

        n_block = len(self._event_lines["START"])
        if n_block > 1:
            logger.info(
                f"There are {n_block} recording blocks in this"
                " file. Times between blocks will be annotated with"
                f" {self._gap_desc}."
            )
            # if there is more than 1 recording block we must account for
            # the missing timestamps and samples bt the blocks
            self.dataframes["samples"] = _fill_times(
                self.dataframes["samples"], sfreq=sfreq
            )
        _convert_times(self.dataframes["samples"], first_samp)

        # dataframe for each type of occular event
        for event, columns, label in zip(
            ["EFIX", "ESACC", "EBLINK"],
            [col_names["fixation"], col_names["saccade"], col_names["blink"]],
            ["fixations", "saccades", "blinks"],
        ):
            if self._event_lines[event]:  # an empty list returns False
                self.dataframes[label] = pd.DataFrame(
                    self._event_lines[event], columns=columns
                )
                _convert_times(self.dataframes[label], first_samp)

                if find_overlaps is True:
                    if self._tracking_mode == "monocular":
                        raise ValueError(
                            "find_overlaps is only valid with"
                            " binocular recordings, this file is"
                            f" {self._tracking_mode}"
                        )
                    df = _find_overlaps(self.dataframes[label], max_time=threshold)
                    self.dataframes[label] = df

            else:
                logger.info(
                    f"No {label} were found in this file. "
                    f"Not returning any info on {label}."
                )

        # make dataframe for experiment messages
        if self._event_lines["MSG"]:
            msgs = []
            for tokens in self._event_lines["MSG"]:
                timestamp = tokens[0]
                block = tokens[-1]
                # if offset token exists, it will be the 1st index
                # and is an int or float
                if isinstance(tokens[1], (int, float)):
                    offset = tokens[1]
                    msg = " ".join(str(x) for x in tokens[2:-1])
                else:
                    # there is no offset token
                    offset = np.nan
                    msg = " ".join(str(x) for x in tokens[1:-1])
                msgs.append([timestamp, offset, msg, block])

            cols = ["time", "offset", "event_msg", "block"]
            self.dataframes["messages"] = pd.DataFrame(msgs, columns=cols)
            _convert_times(self.dataframes["messages"], first_samp)

        # make dataframe for recording block start, end times
        assert len(self._event_lines["START"]) == len(self._event_lines["END"])
        blocks = [
            [bgn[0], end[0], bgn[-1]]  # start, end, block_num
            for bgn, end in zip(self._event_lines["START"], self._event_lines["END"])
        ]
        cols = ["time", "end_time", "block"]
        self.dataframes["recording_blocks"] = pd.DataFrame(blocks, columns=cols)
        _convert_times(self.dataframes["recording_blocks"], first_samp)

        # make dataframe for digital input port
        if self._event_lines["INPUT"]:
            cols = ["time", "DIN", "block"]
            self.dataframes["DINS"] = pd.DataFrame(
                self._event_lines["INPUT"], columns=cols
            )
            _convert_times(self.dataframes["DINS"], first_samp)

        # TODO: Make dataframes for other eyelink events (Buttons)

    def _create_info(self, ch_names, sfreq):
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
        info = create_info(ch_names, sfreq, ch_types)
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
            if "HREF" in self._rec_info:
                if ch_dict["ch_name"].startswith(("xpos", "ypos")):
                    ch_dict["unit"] = FIFF.FIFF_UNIT_RAD
        return info

    def _make_gap_annots(self, key="recording_blocks"):
        """Create Annotations for gap periods between recording blocks."""
        df = self.dataframes[key]
        gap_desc = self._gap_desc
        onsets = df["end_time"].iloc[:-1]
        diffs = df["time"].shift(-1) - df["end_time"]
        durations = diffs.iloc[:-1]
        descriptions = [gap_desc] * len(onsets)
        return Annotations(onset=onsets, duration=durations, description=descriptions)

    def _make_eyelink_annots(self, df_dict, create_annots, apply_offsets):
        """Create Annotations for each df in self.dataframes."""
        valid_descs = ["blinks", "saccades", "fixations", "messages"]
        msg = (
            "create_annotations must be True or a list containing one or"
            f" more of {valid_descs}."
        )
        wrong_type = msg + f" Got a {type(create_annots)} instead."
        if create_annots is True:
            descs = valid_descs
        else:
            assert isinstance(create_annots, list), wrong_type
            for desc in create_annots:
                assert desc in valid_descs, msg + f" Got '{desc}' instead"
            descs = create_annots

        annots = None
        for key, df in df_dict.items():
            eye_annot_cond = (key in ["blinks", "fixations", "saccades"]) and (
                key in descs
            )
            if eye_annot_cond:
                onsets = df["time"]
                durations = df["duration"]
                # Create annotations for both eyes
                descriptions = f"{key[:-1]}_" + df["eye"]  # i.e "blink_r"
                this_annot = Annotations(
                    onset=onsets, duration=durations, description=descriptions
                )
            elif (key in ["messages"]) and (key in descs):
                if apply_offsets:
                    if df["offset"].isnull().all():
                        logger.warning(
                            "There are no offsets for the messages"
                            f" in {self.fname}. Not applying any"
                            " offset"
                        )
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
            logger.warning(
                f"Annotations for {descs} were requested but" " none could be made."
            )
            return
        return annots
