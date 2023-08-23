"""SR Research Eyelink Load Function."""

# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
from ._utils import (
    _convert_times,
    _adjust_times,
    _get_recording_datetime,
    _find_overlaps,
    _get_sfreq_from_ascii,
    _is_sys_msg,
)  # helper functions
from ..constants import FIFF
from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations
from ...utils import (
    _check_fname,
    _check_pandas_installed,
    fill_doc,
    logger,
    verbose,
    warn,
)

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


@fill_doc
def read_raw_eyelink(
    fname,
    preload=False,
    verbose=None,
    create_annotations=True,
    apply_offsets=False,
    find_overlaps=False,
    overlap_threshold=0.05,
    gap_description=None,
):
    """Reader for an Eyelink .asc file.

    Parameters
    ----------
    fname : path-like
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
        Time in seconds. Threshold of allowable time-gap between both the start and
        stop times of the left and right eyes. If the gap is larger than the threshold,
        the :class:`mne.Annotations` will be kept separate (i.e. ``"blink_L"``,
        ``"blink_R"``). If the gap is smaller than the threshold, the
        :class:`mne.Annotations` will be merged and labeled as ``"blink_both"``.
        Defaults to ``0.05`` seconds (50 ms), meaning that if the blink start times of
        the left and right eyes are separated by less than 50 ms, and the blink stop
        times of the left and right eyes are separated by less than 50 ms, then the
        blink will be merged into a single :class:`mne.Annotations`.
    gap_description : str (default 'BAD_ACQ_SKIP')
        Label for annotations that span across the gap period between the
        blocks. Uses ``'BAD_ACQ_SKIP'`` by default so that these time periods will
        be considered bad by MNE and excluded from operations like epoching.

        .. deprecated:: 1.5

           This parameter is deprecated and will be removed in version 1.6. Use
           :meth:`mne.Annotations.rename` if you want something other than
           ``BAD_ACQ_SKIP`` as the annotation label.

    Returns
    -------
    raw : instance of RawEyelink
        A Raw object containing eyetracker data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    It is common for SR Research Eyelink eye trackers to only record data during trials.
    To avoid frequent data discontinuities and to ensure that the data is continuous
    so that it can be aligned with EEG and MEG data (if applicable), this reader will
    preserve the times between recording trials and annotate them with
    ``'BAD_ACQ_SKIP'``.
    """
    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")

    raw_eyelink = RawEyelink(
        fname,
        preload=preload,
        verbose=verbose,
        create_annotations=create_annotations,
        apply_offsets=apply_offsets,
        find_overlaps=find_overlaps,
        overlap_threshold=overlap_threshold,
        gap_desc=gap_description,
    )
    return raw_eyelink


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    fname : path-like
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
    gap_desc : str
        If there are multiple recording blocks in the file, the description of
        the annotation that will span across the gap period between the
        blocks. Default is ``None``, which uses 'BAD_ACQ_SKIP' by default so that these
        timeperiods will be considered bad by MNE and excluded from operations like
        epoching. Note that this parameter is deprecated and will be removed in 1.6.
        Use ``mne.annotations.rename`` instead.


    %(preload)s
    %(verbose)s

    Attributes
    ----------
    fname : pathlib.Path
        Eyelink filename

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
        gap_desc=None,
    ):
        logger.info("Loading {}".format(fname))

        raw_extras = dict()  # extra info from file
        self.fname = Path(fname)
        self._sample_lines = None  # sample lines from file
        self._event_lines = None  # event messages from file
        self._tracking_mode = None  # assigned in self._infer_col_names
        self._rec_info = None
        self._ascii_sfreq = None
        if gap_desc is None:
            gap_desc = "BAD_ACQ_SKIP"
        else:
            warn(
                "gap_description is deprecated in 1.5 and will be removed in 1.6, "
                "use raw.annotations.rename to use a description other than "
                "'BAD_ACQ_SKIP'",
                FutureWarning,
            )
        self._gap_desc = gap_desc
        self.dataframes = {}

        # ======================== Parse ASCII File =========================
        raw_extras["dt"] = _get_recording_datetime(self.fname)
        self._parse_recording_blocks()  # sets sample, event, & system lines

        # ======================== Create DataFrames ========================
        self._create_dataframes()
        del self._sample_lines  # free up memory
        # add column names to dataframes
        col_names, ch_names = self._infer_col_names()
        self._assign_col_names(col_names)
        self._set_df_dtypes()  # set dtypes for each dataframe
        if "HREF" in self._rec_info:
            self._convert_href_samples()
        # fill in times between recording blocks with BAD_ACQ_SKIP
        n_blocks = len(self._event_lines["START"])
        if n_blocks > 1:
            logger.info(
                f"There are {n_blocks} recording blocks in this file. Times between"
                f"  blocks will be annotated with {self._gap_desc}."
            )
            self.dataframes["samples"] = _adjust_times(
                self.dataframes["samples"], self._ascii_sfreq
            )
        # Convert timestamps to seconds
        for df in self.dataframes.values():
            first_samp = float(self._event_lines["START"][0][0])
            _convert_times(df, first_samp)
        # Find overlaps between left and right eye events
        if find_overlaps:
            for key in self.dataframes:
                if key not in ["blinks", "fixations", "saccades"]:
                    continue
                self.dataframes[key] = _find_overlaps(
                    self.dataframes[key], max_time=overlap_threshold
                )

        # ======================== Create Raw Object =========================
        info = self._create_info(ch_names, self._ascii_sfreq)
        eye_ch_data = self.dataframes["samples"][ch_names]
        eye_ch_data = eye_ch_data.to_numpy().T
        super(RawEyelink, self).__init__(
            info,
            preload=eye_ch_data,
            filenames=[self.fname],
            verbose=verbose,
            raw_extras=[raw_extras],
        )
        self.set_meas_date(self._raw_extras[0]["dt"])

        # ======================== Make Annotations =========================
        gap_annots = None
        if len(self._event_lines["START"]) > 1:
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

        # Free up memory
        del self.dataframes
        del self._event_lines

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
                    tokens = line.split()
                    if tokens[0][0].isnumeric():  # Samples
                        self._sample_lines.append(tokens)
                    elif tokens[0] in self._event_lines.keys():
                        if _is_sys_msg(line):
                            continue  # system messages don't need to be parsed.
                        event_key, event_info = tokens[0], tokens[1:]
                        self._event_lines[event_key].append(event_info)
                        if tokens[0] == "END":  # end of recording block
                            is_recording_block = False
            if not self._sample_lines:  # no samples parsed
                raise ValueError(f"Couldn't find any samples in {self.fname}")
            self._validate_data()

    def _validate_data(self):
        """Check the incoming data for some known problems that can occur."""
        self._rec_info = self._event_lines["SAMPLES"][0]
        self._pupil_info = self._event_lines["PUPIL"][0]
        self._n_blocks = len(self._event_lines["START"])
        self._ascii_sfreq = _get_sfreq_from_ascii(self._event_lines["SAMPLES"][0])
        if ("LEFT" in self._rec_info) and ("RIGHT" in self._rec_info):
            self._tracking_mode = "binocular"
        else:
            self._tracking_mode = "monocular"
        # Detect the datatypes that are in file.
        if "GAZE" in self._rec_info:
            logger.info(
                "Pixel coordinate data detected."
                "Pass `scalings=dict(eyegaze=1e3)` when using plot"
                " method to make traces more legible."
            )

        elif "HREF" in self._rec_info:
            logger.info("Head-referenced eye-angle (HREF) data detected.")
        elif "PUPIL" in self._rec_info:
            warn("Raw eyegaze coordinates detected. Analyze with caution.")
        if "AREA" in self._pupil_info:
            logger.info("Pupil-size area detected.")
        elif "DIAMETER" in self._pupil_info:
            logger.info("Pupil-size diameter detected.")
        # If more than 1 recording period, check whether eye being tracked changed.
        if self._n_blocks > 1:
            if self._tracking_mode == "monocular":
                eye = self._rec_info[1]
                blocks_list = self._event_lines["SAMPLES"]
                eye_per_block = [block_info[1] for block_info in blocks_list]
                if not all([this_eye == eye for this_eye in eye_per_block]):
                    warn(
                        "The eye being tracked changed during the"
                        " recording. The channel names will reflect"
                        " the eye that was tracked at the start of"
                        " the recording."
                    )

    def _convert_href_samples(self):
        """Convert HREF eyegaze samples to radians."""
        # grab the xpos and ypos channel names
        pos_names = EYELINK_COLS["pos"]["left"][:-1] + EYELINK_COLS["pos"]["right"][:-1]
        for col in self.dataframes["samples"].columns:
            if col not in pos_names:  # 'xpos_left' ... 'ypos_right'
                continue
            series = self._href_to_radian(self.dataframes["samples"][col])
            self.dataframes["samples"][col] = series

    def _href_to_radian(self, opposite, f=15_000):
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

    def _infer_col_names(self):
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
        col_names["fixations"] = list(
            EYELINK_COLS["eye_event"] + EYELINK_COLS["fixation"]
        )
        col_names["saccades"] = list(
            EYELINK_COLS["eye_event"] + EYELINK_COLS["saccade"]
        )

        # Recording was either binocular or monocular
        # If monocular, find out which eye was tracked and append to ch_name
        if self._tracking_mode == "monocular":
            assert self._rec_info[1] in ["LEFT", "RIGHT"]
            eye = self._rec_info[1].lower()
            ch_names = list(EYELINK_COLS["pos"][eye])
        elif self._tracking_mode == "binocular":
            ch_names = list(EYELINK_COLS["pos"]["left"] + EYELINK_COLS["pos"]["right"])
        col_names["samples"].extend(ch_names)

        # The order of these if statements should not be changed.
        if "VEL" in self._rec_info:  # If velocity data are reported
            if self._tracking_mode == "monocular":
                ch_names.extend(EYELINK_COLS["velocity"][eye])
                col_names["samples"].extend(EYELINK_COLS["velocity"][eye])
            elif self._tracking_mode == "binocular":
                ch_names.extend(
                    EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
                )
                col_names["samples"].extend(
                    EYELINK_COLS["velocity"]["left"] + EYELINK_COLS["velocity"]["right"]
                )
        # if resolution data are reported
        if "RES" in self._rec_info:
            ch_names.extend(EYELINK_COLS["resolution"])
            col_names["samples"].extend(EYELINK_COLS["resolution"])
            col_names["fixations"].extend(EYELINK_COLS["resolution"])
            col_names["saccades"].extend(EYELINK_COLS["resolution"])
        # if digital input port values are reported
        if "INPUT" in self._rec_info:
            ch_names.extend(EYELINK_COLS["input"])
            col_names["samples"].extend(EYELINK_COLS["input"])

        # if head target info was reported, add its cols
        if "HTARGET" in self._rec_info:
            ch_names.extend(EYELINK_COLS["remote"])
            col_names["samples"].extend(EYELINK_COLS["remote"])

        return col_names, ch_names

    def _create_dataframes(self):
        """Create pandas.DataFrame for Eyelink samples and events.

        Creates a pandas DataFrame for self._sample_lines and for each
        non-empty key in self._event_lines.
        """
        pd = _check_pandas_installed()

        # dataframe for samples
        self.dataframes["samples"] = pd.DataFrame(self._sample_lines)
        self._drop_status_col()  # Remove STATUS column

        # dataframe for each type of occular event
        for event, label in zip(
            ["EFIX", "ESACC", "EBLINK"], ["fixations", "saccades", "blinks"]
        ):
            if self._event_lines[event]:  # an empty list returns False
                self.dataframes[label] = pd.DataFrame(self._event_lines[event])
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
                # if offset token exists, it will be the 1st index and is numeric
                if tokens[1].lstrip("-").replace(".", "", 1).isnumeric():
                    offset = float(tokens[1])
                    msg = " ".join(str(x) for x in tokens[2:])
                else:
                    # there is no offset token
                    offset = np.nan
                    msg = " ".join(str(x) for x in tokens[1:])
                msgs.append([timestamp, offset, msg])
            self.dataframes["messages"] = pd.DataFrame(msgs)

        # make dataframe for recording block start, end times
        i = 1
        blocks = list()
        for bgn, end in zip(self._event_lines["START"], self._event_lines["END"]):
            blocks.append((float(bgn[0]), float(end[0]), i))
            i += 1
        cols = ["time", "end_time", "block"]
        self.dataframes["recording_blocks"] = pd.DataFrame(blocks, columns=cols)

        # make dataframe for digital input port
        if self._event_lines["INPUT"]:
            cols = ["time", "DIN"]
            self.dataframes["DINS"] = pd.DataFrame(self._event_lines["INPUT"])

        # TODO: Make dataframes for other eyelink events (Buttons)

    def _drop_status_col(self):
        """Drop STATUS column from samples dataframe.

        see https://github.com/mne-tools/mne-python/issues/11809, and section 4.9.2.1 of
        the Eyelink 1000 Plus User Manual, version 1.0.19. We know that the STATUS
        column is either 3, 5, 13, or 17 characters long, i.e. "...", ".....", ".C."
        """
        status_cols = []
        # we know the first 3 columns will be the time, xpos, ypos
        for col in self.dataframes["samples"].columns[3:]:
            if self.dataframes["samples"][col][0][0].isnumeric():
                # if the value is numeric, it's not a status column
                continue
            if len(self.dataframes["samples"][col][0]) in [3, 5, 13, 17]:
                status_cols.append(col)
        self.dataframes["samples"].drop(columns=status_cols, inplace=True)

    def _assign_col_names(self, col_names):
        """Assign column names to dataframes.

        Parameters
        ----------
        col_names : dict
            Dictionary of column names for each dataframe.
        """
        for key, df in self.dataframes.items():
            if key in ("samples", "blinks", "fixations", "saccades"):
                df.columns = col_names[key]
            elif key == "messages":
                cols = ["time", "offset", "event_msg"]
                df.columns = cols
            elif key == "DINS":
                cols = ["time", "DIN"]
                df.columns = cols

    def _set_df_dtypes(self):
        from ...utils import _set_pandas_dtype

        for key, df in self.dataframes.items():
            if key in ["samples", "DINS"]:
                # convert missing position values to NaN
                self._set_missing_values(df)
                _set_pandas_dtype(df, df.columns, float, verbose="warning")
            elif key in ["blinks", "fixations", "saccades"]:
                _set_pandas_dtype(df, df.columns[1:], float, verbose="warning")
            elif key == "messages":
                _set_pandas_dtype(df, ["time"], float, verbose="warning")  # timestamp

    def _set_missing_values(self, df):
        """Set missing values to NaN. operates in-place."""
        missing_vals = (".", "MISSING_DATA")
        for col in df.columns:
            if col.startswith(("xpos", "ypos")):
                # we explicitly use numpy instead of pd.replace because it is faster
                df[col] = np.where(df[col].isin(missing_vals), np.nan, df[col])

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
