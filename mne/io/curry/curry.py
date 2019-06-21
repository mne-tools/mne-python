# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os
import re
from pathlib import Path
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _read_segments_file, warn
from ...utils import check_fname
from ..constants import FIFF

INFO_FILE_EXTENSION = {7: '.dap', 8: '.cdt.dpa'}
LABEL_FILE_EXTENSION = {7: '.rs3', 8: '.cdt.dpa'}
DATA_FILE_EXTENSION = {7: '.dat', 8: '.cdt'}
CHANTYPES = {"meg": "_MAG1", "eeg": "", "misc": "_OTHERS"}
FIFFV_CHANTYPES = {"meg": FIFF.FIFFV_MEG_CH, "eeg": FIFF.FIFFV_EEG_CH,
                   "misc": FIFF.FIFFV_MISC_CH}
SI_UNITS = dict(V=FIFF.FIFF_UNIT_V, T=FIFF.FIFF_UNIT_T)
SI_UNIT_SCALE = dict(c=1e-2, m=1e-3, u=1e-6, μ=1e-6, n=1e-9, p=1e-12, f=1e-15)


def _get_curry_version(file_extension):
    """check out the curry file version"""

    return 8 if 'cdt' in file_extension else 7


def _check_missing_files(fname_base, curry_vers):
    """Check if all neccessary files exist."""

    _msg = "The following required files cannot be found: {0}.\nPlease make " \
           "sure all required files are located in the same directory."

    missing = [str(Path(fname_base).with_suffix(ext))
               for ext in [DATA_FILE_EXTENSION[curry_vers],
                           INFO_FILE_EXTENSION[curry_vers],
                           LABEL_FILE_EXTENSION[curry_vers]]
               if not Path(fname_base).with_suffix(ext).is_file()]

    if missing:
        raise FileNotFoundError(_msg.format(missing))


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
        `regex + " START_LIST"` initiates the start and `regex + " END_LIST"`
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
                if re.match(regex + " END_LIST", line):
                    save_lines[regex] = False

                if save_lines[regex] and line != "\n":
                    result = line.replace("\n", "")
                    if "\t" in result:
                        result = result.split("\t")
                    data_dict[regex].append(result)

                if re.match(regex + " START_LIST", line):
                    save_lines[regex] = True

    return data_dict


def _read_curry_info(fname_base, curry_vers):
    """extract info from curry parameter files"""

    var_names = ['NumSamples', 'NumChannels', 'NumTrials', 'SampleFreqHz',
                 'TriggerOffsetUsec', 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'NUM_CHANNELS', 'NUM_TRIALS', 'SAMPLE_FREQ_HZ',
                 'TRIGGER_OFFSET_USEC', 'DATA_FORMAT', 'SAMPLE_TIME_USEC']


    param_dict = dict()
    unit_dict = dict()
    with open(fname_base + INFO_FILE_EXTENSION[curry_vers]) as fid:
        for line in iter(fid):
            if any(var_name in line for var_name in var_names):
                key, val = line.replace(" ", "").replace("\n", "").split("=")
                param_dict[key.lower().replace("_", "")] = val
            for type in CHANTYPES:
                if "DEVICE_PARAMETERS" + CHANTYPES[type] + " START" in line:
                    data_unit = next(fid)
                    unit_dict[type] = data_unit.replace(" ", "").replace("\n", "").split("=")[-1]

    for var in var_names[:7]:
        if var.lower() not in param_dict:
            raise KeyError("Variable %s cannot be found in the parameter file."
                           % var)

    n_samples = int(param_dict["numsamples"])
    # n_ch = int(param_dict["numchannels"])
    n_trials = int(param_dict["numtrials"])
    sfreq = float(param_dict["samplefreqhz"])
    offset = float(param_dict["triggeroffsetusec"]) * 1e-6
    time_step = float(param_dict["sampletimeusec"]) * 1e-6
    ascii = param_dict["dataformat"] == "ASCII"

    if (sfreq == 0) and (time_step != 0):
        sfreq = 1. / time_step

    # read labels from label files
    labels = _read_curry_lines(fname_base + LABEL_FILE_EXTENSION[curry_vers],
                               ["LABELS" + CHANTYPES[key] for key in ["meg", "eeg", "misc"]])

    sensors = _read_curry_lines(fname_base + LABEL_FILE_EXTENSION[curry_vers],
                                ["SENSORS" + CHANTYPES[key] for key in ["meg", "eeg", "misc"]])

    all_chans = list()
    for key in ["meg", "eeg", "misc"]:
        for ind, chan in enumerate(labels["LABELS" + CHANTYPES[key]]):
            ch = {"ch_name": chan,
                  "unit": unit_dict[key],
                  "kind": FIFFV_CHANTYPES[key]}
            if key in ("meg", "eeg"):
                loc = sensors["SENSORS" + CHANTYPES[key]][ind]
                ch["loc"] = np.array(loc, dtype=float)

            all_chans.append(ch)

    ch_names = [chan["ch_name"] for chan in all_chans]
    info = create_info(ch_names, sfreq)

    for ind, ch_dict in enumerate(info["chs"]):
        ch_dict["kind"] = all_chans[ind]["kind"]
        ch_dict['unit'] = SI_UNITS[all_chans[ind]['unit'][1]]
        ch_dict['cal'] = SI_UNIT_SCALE[all_chans[ind]['unit'][0]]
        if ch_dict["kind"] in (FIFF.FIFFV_MEG_CH,
                               FIFF.FIFFV_EEG_CH):
            ch_dict["loc"] = all_chans[ind]["loc"]

    return info, n_trials, n_samples, curry_vers, ascii


def read_events_curry(fname, event_ids=None):
    """
    Read events from Curry event files.

    Parameters
    ----------
    input_fname : str
        Path to a curry event file with extensions .cef, .ceo,
        .cdt.cef, or .cdt.ceo
    event_ids : tuple, list or None (default None)
        If tuple or list, only the event IDs in event_ids
        will be read. If None, all event IDs will be read.

    Returns
    -------
    events : ndarray
        An array of shape (n_events,3) containing MNE events.
    """

    check_fname(fname, 'curry event', ('.cef', '.ceo', '.cdt.cef', '.cdt.ceo'),
                endings_err=('.cef', '.ceo', '.cdt.cef', '.cdt.ceo'))

    events_dict = _read_curry_lines(fname, ["NUMBER_LIST"])
    # The first 3 column seem to contain the event information
    curry_events = np.array(events_dict["NUMBER_LIST"], dtype=int)[:, 0:3]

    if event_ids is not None:
        idx = [i in event_ids for i in curry_events[:, -1]]
        curry_events = curry_events[idx]

    return curry_events


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
        is stored in ASCII data format, then preload will automatically be
        set to True.

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing Curry data.

    """

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    fname_base, ext = input_fname.split(".", maxsplit=1)

    curry_vers = _get_curry_version(ext)
    _check_missing_files(fname_base, curry_vers)

    info, n_trials, n_samples, curry_vers, ascii = _read_curry_info(fname_base, curry_vers)

    raw = RawCurry(fname_base + DATA_FILE_EXTENSION[curry_vers], info, n_samples, ascii, preload)

    return raw


class RawCurry(BaseRaw):
    """
    Raw object from Curry file.

    Parameters
    ----------
    data_fname : str
        Path to the Curry data file (.dat or .cdt).
    info : instance of mne.Info
        An instance of mne.Info as returned by mne.io.curry._read_curry_info.
    n_samples : int
        The number of samples of the curry data file.
    ascii : bool (default False)
        If the loaded data file is coded in ASCII, ascii must be set to True.
        Else must be False.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory) If ascii is True,
        preload will automatically be set to True.
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    """

    def __init__(self, data_fname, info, n_samples, ascii,
                 preload=False, verbose=None):

        data_fname = os.path.abspath(data_fname)

        last_samps = [n_samples - 1]

        if ascii:
            if preload == False:
                warn('Got ASCII format data as input. Data will be preloaded.')

            cals = [[ch_dict["cal"]] for ch_dict in info["chs"]]
            preload = np.loadtxt(data_fname).T * cals


        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""

        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")
