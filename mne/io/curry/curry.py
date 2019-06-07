# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os
import re
import numpy as np
import mne

from ..base import BaseRaw, _check_update_montage
from ..meas_info import create_info
from ..utils import _read_segments_file, warn, _find_channels, _create_chs


def _read_curry_annotations(fname_base, curry_vers):
    """read annotations from curry event files"""


    EVENT_FILE_EXTENSION = {7: ['.cef', '.ceo'], 8: ['.cdt.cef', '.cdt.ceo']}

    event_file = None
    for ext in EVENT_FILE_EXTENSION[curry_vers]:
        if os.path.isfile(fname_base + ext):
            event_file = fname_base + ext

    if event_file is not None:
        annotations_dict = _read_curry_lines(event_file, ["NUMBER_LIST "])
        curry_annotations = np.array(annotations_dict["NUMBER_LIST "], dtype=int)

    else:
        curry_annotations = None

    # TODO: This returns annotations in a curry specific format. This might be reformatted to fit mne
    return curry_annotations


def _get_curry_version(file_extension):
    """check out the curry file version"""

    if 'cdt' in file_extension:
        curry_vers = 8

    else:
        curry_vers = 7

    return curry_vers


def _check_missing_files(full_fname, fname_base, curry_vers):
    """
    Check if all neccessary files exist.
     """

    if curry_vers == 8:
        for check_ext in [".cdt", ".cdt.dpa"]:
            if not os.path.isfile(fname_base + check_ext):
                raise FileNotFoundError("The following required file cannot be"
                                        " found: %s. Please make sure it is "
                                        "located in the same directory as %s."
                                        % (fname_base + check_ext, full_fname))

    if curry_vers == 7:
        for check_ext in [".dap", ".dat", ".rs3"]:
            if not os.path.isfile(fname_base + check_ext):
                raise FileNotFoundError("The following required file cannot be"
                                        " found: %s. Please make sure it is "
                                        "located in the same directory as %s."
                                        % (fname_base + check_ext, full_fname))


def _read_curry_lines(fname, regex_list):
    """
    Read through the lines of a curry parameter files and save data.

    Parameters
    ----------
    fname : str
        Path to a curry file.
    regex_list : list of str
        A list of strings or regular expressions to search within the file.
        Each element `regex` in `regex_list` must be formulated so that
        `regex + "START_LIST"` initiates the start and `regex + "END_LIST"`
        initiates the end of the elements that should be saved.

    Returns
    -------
    data_dict : dict
        A dictionary containing the extracted data. For each element `regex`
        in `regex_list` a dictionary key `data_dict[regex]` is created, which
        contains a list of the according data.

    """

    save_lines = {}
    data_dict = {}

    for regex in regex_list:
        save_lines[regex] = False
        data_dict[regex] = []

    with open(fname) as fid:
        for line in fid:
            for regex in regex_list:
                if re.match(regex + "END_LIST", line):
                    save_lines[regex] = False

                if save_lines[regex] and line != "\n":
                    result = line.replace("\n", "")
                    if "\t" in result:
                        result = result.split("\t")
                    data_dict[regex].append(result)

                if re.match(regex + "START_LIST", line):
                    save_lines[regex] = True

    return data_dict


def _read_curry_info(fname_base, curry_vers):

    #####################################
    # read parameters from the param file

    CAL = 1e-6
    INFO_FILE_EXTENSION = {7: '.dap', 8: '.cdt.dpa'}
    LABEL_FILE_EXTENSION = {7: '.rs3', 8: '.cdt.dpa'}


    var_names = ['NumSamples', 'NumChannels', 'NumTrials', 'SampleFreqHz',
                 'TriggerOffsetUsec', 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'NUM_CHANNELS', 'NUM_TRIALS', 'SAMPLE_FREQ_HZ',
                 'TRIGGER_OFFSET_USEC', 'DATA_FORMAT', 'SAMPLE_TIME_USEC']

    param_dict = dict()
    with open(fname_base + INFO_FILE_EXTENSION[curry_vers]) as fid:
        for line in fid:
            if any(var_name in line for var_name in var_names):
                key, val = line.replace(" ", "").replace("\n", "").split("=")
                param_dict[key.lower().replace("_", "")] = val

    for var in var_names[:7]:
        if var.lower() not in param_dict:
            raise KeyError("Variable %s cannot be found in the parameter file."
                           % var)

    n_samples = int(param_dict["numsamples"])
    # n_ch = int(param_dict["numchannels"])
    n_trials = int(param_dict["numtrials"])
    sfreq = float(param_dict["samplefreqhz"])
    offset = float(param_dict["triggeroffsetusec"]) * CAL
    time_step = float(param_dict["sampletimeusec"]) * CAL
    data_format = param_dict["dataformat"]

    if (sfreq == 0) and (time_step != 0):
        sfreq = 1. / time_step

    #####################################
    # read labels from label files

    data_dict = _read_curry_lines(fname_base + LABEL_FILE_EXTENSION[curry_vers],
                                  ["LABELS.*?", "SENSORS.*?"])

    ch_names = data_dict["LABELS.*?"]

    ch_pos = np.array(data_dict["SENSORS.*?"], dtype=float)
    # TODO: include this in ch_dict

    # ch_names = list(reversed(ch_names))
    info = create_info(ch_names, sfreq)

    # TODO; There's still a lot more information that can be brought into info["chs"]. However i'm not sure what to do with MEG chans here
    for ch_dict in info["chs"]:
        ch_dict["cal"] = CAL

    return info, n_trials, n_samples, curry_vers, data_format


def read_raw_curry(input_fname, preload=False):
    """
    Read raw data from Curry files.

    Parameters
    ----------
    input_fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). If the curry file
        is stored in ASCII data format, then preload must be `True`.

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing CURRY data.

    """

    DATA_FILE_EXTENSION = {7: '.dat', 8: '.cdt'}

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    fname_base, ext = input_fname.split(".", maxsplit=1)

    curry_vers = _get_curry_version(ext)
    _check_missing_files(input_fname, fname_base, curry_vers)

    info, n_trials, n_samples, curry_vers, data_format = _read_curry_info(fname_base, curry_vers)
    annotations = _read_curry_annotations(fname_base, curry_vers)
    info["events"] = annotations

    raw = RawCurry(fname_base + DATA_FILE_EXTENSION[curry_vers], info, n_samples, data_format)

    return raw


class RawCurry(BaseRaw):
    """"""

    def __init__(self, data_fname, info, n_samples, data_format, montage=None, eog=(), ecg=(),
                 emg=(), misc=(), preload=False, verbose=None):

        data_fname = os.path.abspath(data_fname)

        last_samps = [n_samples - 1]

        if preload == False and data_format == "ASCII":
            warn('Got ASCII format data as input. Data will be preloaded.')

            cals = [[ch_dict["cal"]] for ch_dict in info["chs"]]
            preload = np.loadtxt(data_fname).T * cals


        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""

        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")
