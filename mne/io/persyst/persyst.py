# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _read_segments_file
from ...annotations import Annotations
from ...utils import (
    _check_fname,
    _pl,
    _verbose_control,
    fill_doc_static,
    logger,
    warn,
)
from ..base import BaseRaw


@fill_doc_static("preload", "verbose")
def read_raw_persyst(
    fname: Path | str,
    preload: bool | str = False,
    verbose: bool | str | int | None = None,
) -> "RawPersyst":
    """Reader for a Persyst (.lay/.dat) recording.

    Parameters
    ----------
    fname : path-like
        Path to the Persyst header ``.lay`` file.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

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

    Recording gaps described in the ``[SampleTimes]`` section of the ``.lay`` file
    are filled with zeros and annotated as ``BAD_ACQ_SKIP``.
    """
    return RawPersyst(fname, preload, verbose)


@fill_doc_static("preload", "verbose")
class RawPersyst(BaseRaw):
    """Raw object from a Persyst file.

    Parameters
    ----------
    fname : path-like
        Path to the Persyst header (.lay) file.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @_verbose_control
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
        fileinfo_dict = OrderedDict()
        channelmap_dict = OrderedDict()
        patient_dict = OrderedDict()
        comments_dict = OrderedDict()
        sample_times = list()

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
            # SampleTimes (recording gaps)
            elif section == "sampletimes":
                sample_times.append((key, val))

        # get numerical metadata
        # datatype is either 7 for 32 bit, or 0 for 16 bit
        datatype = fileinfo_dict["datatype"]
        cal = float(fileinfo_dict["calibration"])
        n_chs = int(fileinfo_dict["waveformcount"])

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
                testtime = datetime.strptime(patient_dict["testtime"], "%H:%M:%S")
                meas_date = datetime(
                    year=testdate.year,
                    month=testdate.month,
                    day=testdate.day,
                    hour=testtime.hour,
                    minute=testtime.minute,
                    second=testtime.second,
                    tzinfo=UTC,
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
        sfreq = fileinfo_dict["samplingrate"]
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

        # contiguous blocks of the .dat file, separated by zero-filled gaps
        file_starts, out_starts = _get_data_blocks(
            sample_times, n_samples, info["sfreq"]
        )
        n_gap_samples = out_starts[-1] - file_starts[-1]

        # Cache-sized blocks are 2.3x faster on a 107 MB file.
        raw_extras = {"dtype": dtype, "n_chs": n_chs, "n_samples": n_samples}
        raw_extras["max_block_samples"] = max(
            1, 16 * 1024**2 // dtype.itemsize // n_chs
        )
        raw_extras["file_starts"] = file_starts
        raw_extras["out_starts"] = out_starts
        # create Raw object
        super().__init__(
            info,
            preload,
            filenames=[dat_fpath],
            last_samps=[n_samples + n_gap_samples - 1],
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
        sfreq = self.info["sfreq"]
        onset = _shift_onsets_by_gaps(onset, file_starts, out_starts, sfreq)
        annot = Annotations(onset, duration, description)

        # annotate the zero-filled gaps
        gap_n_samples = np.diff(out_starts - file_starts)
        gap_starts = out_starts[1:] - gap_n_samples
        n_gaps = len(gap_starts)
        if n_gaps:
            logger.info(
                f"Found {n_gaps} acquisition gap{_pl(n_gaps)} in [SampleTimes], "
                "filling with zeros and annotating as BAD_ACQ_SKIP."
            )
            annot.append(
                gap_starts / sfreq, gap_n_samples / sfreq, ["BAD_ACQ_SKIP"] * n_gaps
            )
        self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The Persyst software  records raw data in either 16 or 32 bit
        binary files. In addition, it stores the calibration to convert
        data to uV in the lay file. Recording gaps (from the [SampleTimes]
        section) are not stored in the file and are filled with zeros.
        """
        extras = self._raw_extras[fi]
        file_starts, out_starts = extras["file_starts"], extras["out_starts"]
        out_stops = out_starts + np.diff(np.append(file_starts, extras["n_samples"]))
        # gap samples are left as is, the output buffer is already zeroed
        for file_start, out_start, out_stop in zip(file_starts, out_starts, out_stops):
            read_start, read_stop = max(out_start, start), min(out_stop, stop)
            if read_start >= read_stop:
                continue
            _read_segments_file(
                self,
                data[:, read_start - start : read_stop - start],
                idx,
                fi,
                file_start + read_start - out_start,
                file_start + read_stop - out_start,
                cals,
                mult,
                dtype=extras["dtype"],
                n_channels=extras["n_chs"],
                max_block_samples=extras["max_block_samples"],
            )


def _get_data_blocks(sample_times, n_samples, sfreq):
    """Get the first sample of each contiguous block in the .dat and output data.

    Each [SampleTimes] ``<sample>=<time>`` entry gives the time (in seconds since
    midnight) of that .dat sample; a time later than expected marks a gap.
    """
    file_starts, out_starts = [0], [0]
    try:
        entries = sorted((int(sample), float(time)) for sample, time in sample_times)
    except ValueError:
        warn("Could not parse the [SampleTimes] section, ignoring recording gaps.")
        return np.array(file_starts), np.array(out_starts)
    n_beyond = sum(sample >= n_samples for sample, _ in entries)
    if n_beyond:
        warn(
            f"Ignoring {n_beyond} [SampleTimes] entr{_pl(n_beyond, 'y', 'ies')} at "
            f"or beyond the end of the data ({n_samples} samples)."
        )
        entries = entries[: len(entries) - n_beyond]
    # times are stored with millisecond precision, so gaps shorter than 1 ms (i.e., a
    # few samples at high sfreq) cannot be distinguished from timestamp rounding
    tol = max(0.5 / sfreq, 1e-3)
    n_gap_total = 0
    for (prev_sample, prev_time), (sample, time) in zip(entries[:-1], entries[1:]):
        jump = time - prev_time - (sample - prev_sample) / sfreq
        # times restart at 0 after midnight, so use whichever of the two possible
        # jumps is closer to 0 (backward steps of more than 12 h are midnight wraps)
        if abs(jump + 86400) < abs(jump):
            jump += 86400
        if jump < -tol:
            warn(f"Ignoring negative [SampleTimes] jump at sample {sample}.")
            continue
        n_gap = round(jump * sfreq) if jump > tol else 0
        if n_gap:
            n_gap_total += n_gap
            file_starts.append(sample)
            out_starts.append(sample + n_gap_total)
    return np.array(file_starts), np.array(out_starts)


def _shift_onsets_by_gaps(onset, file_starts, out_starts, sfreq):
    """Shift [Comments] onsets past the recording gaps that precede them."""
    # This keeps each comment on the same data samples as before gaps were inserted,
    # which is correct if Persyst stores data (not wall-clock) time (gh-12339).
    block = np.searchsorted(file_starts / sfreq, onset, side="right") - 1
    return onset + (out_starts - file_starts)[np.maximum(block, 0)] / sfreq


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

    subject_info: dict[str, Any] = {
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

    For ``SampleTimes``, the ``key`` is a sample index in the .dat file and the
    ``value`` is the time (in seconds since midnight of the test date) at which
    that sample was recorded. There is an entry for the first sample and after
    each recording gap.

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
