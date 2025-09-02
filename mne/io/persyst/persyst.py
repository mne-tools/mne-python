# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...annotations import Annotations
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw


@fill_doc
def read_raw_persyst(fname, preload=False, verbose=None) -> "RawPersyst":
    """Reader for a Persyst (.lay/.dat) recording.

    Parameters
    ----------
    fname : path-like
        Path to the Persyst header ``.lay`` file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawPersyst
        A Raw object containing Persyst data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawPersyst.

    Notes
    -----
    It is assumed that the ``.lay`` and ``.dat`` file
    are in the same directory. To get the correct file path to the
    ``.dat`` file, ``read_raw_persyst`` will get the corresponding dat
    filename from the lay file, and look for that file inside the same
    directory as the lay file.
    """
    return RawPersyst(fname, preload, verbose)


@fill_doc
class RawPersyst(BaseRaw):
    """Raw object from a Persyst file.

    Parameters
    ----------
    fname : path-like
        Path to the Persyst header (.lay) file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        fname = str(_check_fname(fname, "read", True, "fname"))
        logger.info(f"Loading {fname}")

        # make sure filename is the Lay file
        if not fname.endswith(".lay"):
            fname = fname + ".lay"
        # get the current directory and Lay filename
        curr_path, lay_fname = op.dirname(fname), op.basename(fname)
        if not op.exists(fname):
            raise FileNotFoundError(
                f'The path you specified, "{lay_fname}",does not exist.'
            )

        # sections and subsections currently unused
        keys, data, sections = _read_lay_contents(fname)

        # these are the section headers in the Persyst file layout
        # Note: We do not make use of "SampleTimes" yet
        fileinfo_dict = OrderedDict()
        channelmap_dict = OrderedDict()
        patient_dict = OrderedDict()
        comments_dict = OrderedDict()

        # keep track of total number of comments
        num_comments = 0

        # loop through each line in the lay file
        for key, val, section in zip(keys, data, sections):
            if key == "":
                continue

            # Make sure key are lowercase for everything, but electrodes.
            # We also do not want to lower-case comments because those
            # are free-form text where casing may matter.
            if key is not None and section not in ["channelmap", "comments"]:
                key = key.lower()

            # FileInfo
            if section == "fileinfo":
                # extract the .dat file name
                if key == "file":
                    dat_fname = op.basename(val)
                    dat_fpath = op.join(curr_path, op.basename(dat_fname))

                    # determine if .dat file exists where it should
                    error_msg = (
                        f"The data path you specified "
                        f"does not exist for the lay path, "
                        f"{lay_fname}. Make sure the dat file "
                        f"is in the same directory as the lay "
                        f"file, and the specified dat filename "
                        f"matches."
                    )
                    if not op.exists(dat_fpath):
                        raise FileNotFoundError(error_msg)
                fileinfo_dict[key] = val
            # ChannelMap
            elif section == "channelmap":
                # channel map has <channel_name>=<number> for <key>=<val>
                channelmap_dict[key] = val
            # Patient (All optional)
            elif section == "patient":
                patient_dict[key] = val
            # Comments (turned into mne.Annotations)
            elif section == "comments":
                comments_dict[key] = comments_dict.get(key, list()) + [val]
                num_comments += 1

        # get numerical metadata
        # datatype is either 7 for 32 bit, or 0 for 16 bit
        datatype = fileinfo_dict.get("datatype")
        cal = float(fileinfo_dict.get("calibration"))
        n_chs = int(fileinfo_dict.get("waveformcount"))

        # Store subject information from lay file in mne format
        # Note: Persyst also records "Physician", "Technician",
        #       "Medications", "History", and "Comments1" and "Comments2"
        #       and this information is currently discarded
        subject_info = _get_subjectinfo(patient_dict)

        # set measurement date
        testdate = patient_dict.get("testdate")
        if testdate is not None:
            # TODO: Persyst may change its internal date schemas
            #  without notice
            # These are the 3 "so far" possible datatime storage
            # formats in Persyst .lay
            if "/" in testdate:
                testdate = datetime.strptime(testdate, "%m/%d/%Y")
            elif "-" in testdate:
                testdate = datetime.strptime(testdate, "%d-%m-%Y")
            elif "." in testdate:
                testdate = datetime.strptime(testdate, "%Y.%m.%d")

            if not isinstance(testdate, datetime):
                warn(
                    "Cannot read in the measurement date due "
                    "to incompatible format. Please set manually "
                    f"for {lay_fname} "
                )
                meas_date = None
            else:
                testtime = datetime.strptime(patient_dict.get("testtime"), "%H:%M:%S")
                meas_date = datetime(
                    year=testdate.year,
                    month=testdate.month,
                    day=testdate.day,
                    hour=testtime.hour,
                    minute=testtime.minute,
                    second=testtime.second,
                    tzinfo=timezone.utc,
                )

        # Create mne structure
        ch_names = list(channelmap_dict.keys())
        if n_chs != len(ch_names):
            raise RuntimeError(
                "Channels in lay file do not "
                "match the number of channels "
                "in the .dat file."
            )  # noqa
        # get rid of the "-Ref" in channel names
        ch_names = [ch.upper().split("-REF")[0] for ch in ch_names]

        # get the sampling rate and default channel types to EEG
        sfreq = fileinfo_dict.get("samplingrate")
        ch_types = "eeg"
        info = create_info(ch_names, sfreq, ch_types=ch_types)
        info.update(subject_info=subject_info)
        with info._unlock():
            for idx in range(n_chs):
                # calibration brings to uV then 1e-6 brings to V
                info["chs"][idx]["cal"] = cal * 1.0e-6
            info["meas_date"] = meas_date

        # determine number of samples in file
        # Note: We do not use the lay file to do this
        # because clips in time may be generated by Persyst that
        # DO NOT modify the "SampleTimes" section
        with open(dat_fpath, "rb") as f:
            # determine the precision
            if int(datatype) == 7:
                # 32 bit
                dtype = np.dtype("i4")
            elif int(datatype) == 0:
                # 16 bit
                dtype = np.dtype("i2")
            else:
                raise RuntimeError(f"Unknown format: {datatype}")

            # allow offset to occur
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
            n_samples = n_samples // (dtype.itemsize * n_chs)

            logger.debug(f"Loaded {n_samples} samples for {n_chs} channels.")

        raw_extras = {"dtype": dtype, "n_chs": n_chs, "n_samples": n_samples}
        # create Raw object
        super().__init__(
            info,
            preload,
            filenames=[dat_fpath],
            last_samps=[n_samples - 1],
            raw_extras=[raw_extras],
            verbose=verbose,
        )

        # set annotations based on the comments read in
        onset = np.zeros(num_comments, float)
        duration = np.zeros(num_comments, float)
        description = [""] * num_comments

        # loop through comments dictionary, which may contain
        # multiple events for the same "text" annotation
        t_idx = 0
        for _description, event_tuples in comments_dict.items():
            for _onset, _duration in event_tuples:
                # extract the onset, duration, description to
                # create an Annotations object
                onset[t_idx] = _onset
                duration[t_idx] = _duration
                description[t_idx] = _description
                t_idx += 1
        annot = Annotations(onset, duration, description)
        self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The Persyst software  records raw data in either 16 or 32 bit
        binary files. In addition, it stores the calibration to convert
        data to uV in the lay file.
        """
        dtype = self._raw_extras[fi]["dtype"]
        n_chs = self._raw_extras[fi]["n_chs"]
        dat_fname = self.filenames[fi]

        # compute samples count based on start and stop
        time_length_samps = stop - start

        # read data from .dat file into array of correct size, then calibrate
        # records = recnum rows x inf columns
        count = time_length_samps * n_chs

        # seek the dat file
        with open(dat_fname, "rb") as dat_file_ID:
            # allow offset to occur
            dat_file_ID.seek(n_chs * dtype.itemsize * start, 1)

            # read in the actual record starting at possibly offset
            record = np.fromfile(dat_file_ID, dtype=dtype, count=count)

        # chs * rows
        # cast as float32; more than enough precision
        record = np.reshape(record, (n_chs, -1), order="F").astype(np.float32)

        # calibrate to convert to V and handle mult
        _mult_cal_one(data, record, idx, cals, mult)


def _get_subjectinfo(patient_dict):
    # attempt to parse out the birthdate, but if it doesn't
    # meet spec, then it will set to None
    birthdate = patient_dict.get("birthdate")
    if "/" in birthdate:
        try:
            birthdate = datetime.strptime(birthdate, "%m/%d/%y")
        except ValueError:
            birthdate = None
            print(f"Unable to process birthdate of {birthdate} ")
    elif "-" in birthdate:
        try:
            birthdate = datetime.strptime(birthdate, "%d-%m-%y")
        except ValueError:
            birthdate = None
            print(f"Unable to process birthdate of {birthdate} ")

    subject_info = {
        "first_name": patient_dict.get("first"),
        "middle_name": patient_dict.get("middle"),
        "last_name": patient_dict.get("last"),
        "sex": patient_dict.get("sex"),
        "hand": patient_dict.get("hand"),
        "his_id": patient_dict.get("id"),
        "birthday": birthdate,
    }
    subject_info = {key: val for key, val in subject_info.items() if val is not None}

    # Recode sex values
    sex_dict = dict(
        m=FIFF.FIFFV_SUBJ_SEX_MALE,
        male=FIFF.FIFFV_SUBJ_SEX_MALE,
        f=FIFF.FIFFV_SUBJ_SEX_FEMALE,
        female=FIFF.FIFFV_SUBJ_SEX_FEMALE,
    )
    subject_info["sex"] = sex_dict.get(subject_info["sex"], FIFF.FIFFV_SUBJ_SEX_UNKNOWN)

    # Recode hand values
    hand_dict = dict(
        r=FIFF.FIFFV_SUBJ_HAND_RIGHT,
        right=FIFF.FIFFV_SUBJ_HAND_RIGHT,
        l=FIFF.FIFFV_SUBJ_HAND_LEFT,
        left=FIFF.FIFFV_SUBJ_HAND_LEFT,
        a=FIFF.FIFFV_SUBJ_HAND_AMBI,
        ambidextrous=FIFF.FIFFV_SUBJ_HAND_AMBI,
        ambi=FIFF.FIFFV_SUBJ_HAND_AMBI,
    )
    # no handedness is set when unknown
    try:
        subject_info["hand"] = hand_dict[subject_info["hand"]]
    except KeyError:
        subject_info.pop("hand")

    return subject_info


def _read_lay_contents(fname):
    """Lay file are laid out like a INI file."""
    # keep track of sections, keys and data
    sections = []
    keys, data = [], []

    # initialize all section to empty str
    section = ""
    with open(fname) as fin:
        for line in fin:
            # break a line into a status, key and value
            status, key, val = _process_lay_line(line, section)

            # handle keys and values if they are
            # Section, Subsections, or Line items
            if status == 1:  # Section was found
                section = val.lower()
                continue

            # keep track of all sections, subsections,
            # keys and the data of the file
            sections.append(section)
            data.append(val)
            keys.append(key)

    return keys, data, sections


def _process_lay_line(line, section):
    """Process a line read from the Lay (INI) file.

    Each line in the .lay file will be processed
    into a structured ``status``, ``key`` and ``value``.

    Parameters
    ----------
    line : str
        The actual line in the Lay file.
    section : str
        The section in the Lay file.

    Returns
    -------
    status : int
        Returns the following integers based on status.
        -1  => unknown string found
        0   => empty line found
        1   => section found
        2   => key-value pair found
    key : str
        The string before the ``'='`` character. If section is "Comments",
        then returns the text comment description.
    value : str
        The string from the line after the ``'='`` character. If section is
        "Comments", then returns the onset and duration as a tuple.

    Notes
    -----
    The lay file comprises of multiple "sections" that are documented with
    bracket ``[]`` characters. For example, ``[FileInfo]`` and the lines
    afterward indicate metadata about the data file itself. Within
    each section, there are multiple lines in the format of
    ``<key>=<value>``.

    For ``FileInfo``, ``Patient`` and ``ChannelMap``
    each line will be denoted with a ``key`` and a ``value`` that
    can be represented as a dictionary. The keys describe what sort
    of data that line holds, while the values contain the corresponding
    value. In some cases, the ``value``.

    For ``SampleTimes``, the ``key`` and ``value`` pair indicate the
    start and end time in seconds of the original data file.

    For ``Comments`` section, this denotes an area where users through
    Persyst actually annotate data in time. These are instead
    represented as 5 data points that are ``,`` delimited. These
    data points are ordered as:

        1. time (in seconds) of the annotation
        2. duration (in seconds) of the annotation
        3. state (unused)
        4. variable type (unused)
        5. free-form text describing the annotation
    """
    key = ""  # default; only return value possibly not set
    line = line.strip()  # remove leading and trailing spaces
    end_idx = len(line) - 1  # get the last index of the line

    # empty sequence evaluates to false
    if not line:
        status = 0
        key = ""
        value = ""
        return status, key, value
    # section found
    elif (line[0] == "[") and (line[end_idx] == "]") and (end_idx + 1 >= 3):
        status = 1
        value = line[1:end_idx].lower()
    # key found
    else:
        # handle Comments section differently from all other sections
        # TODO: utilize state and var_type in code.
        #  Currently not used
        if section == "comments":
            # Persyst Comments output 5 variables "," separated
            time_sec, duration, state, var_type, text = line.split(",", 4)
            del var_type, state
            status = 2
            key = text
            value = (time_sec, duration)
        # all other sections
        else:
            if "=" not in line:
                raise RuntimeError(
                    f"The line {line} does not conform "
                    "to the standards. Please check the "
                    ".lay file."
                )  # noqa
            pos = line.index("=")
            status = 2

            # the line now is composed of a
            # <key>=<value>
            key = line[0:pos]
            key.strip()
            value = line[pos + 1 : end_idx + 1]
            value.strip()
    return status, key, value
