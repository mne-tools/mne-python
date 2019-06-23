# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os
import re
from pathlib import Path
from packaging import version
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _read_segments_file, _mult_cal_one
from ..constants import FIFF
from ...utils import check_fname
from ...annotations import Annotations

INFO_FILE_EXTENSION = {7: '.dap', 8: '.cdt.dpa'}
LABEL_FILE_EXTENSION = {7: '.rs3', 8: '.cdt.dpa'}
DATA_FILE_EXTENSION = {7: '.dat', 8: '.cdt'}
CHANTYPES = {"meg": "_MAG1", "eeg": "", "misc": "_OTHERS"}
FIFFV_CHANTYPES = {"meg": FIFF.FIFFV_MEG_CH, "eeg": FIFF.FIFFV_EEG_CH,
                   "misc": FIFF.FIFFV_MISC_CH}
SI_UNITS = dict(V=FIFF.FIFF_UNIT_V, T=FIFF.FIFF_UNIT_T)
SI_UNIT_SCALE = dict(c=1e-2, m=1e-3, u=1e-6, μ=1e-6, n=1e-9, p=1e-12, f=1e-15)


def _get_curry_version(file_extension):
    """Check out the curry file version."""
    return 8 if 'cdt' in file_extension else 7


def _check_missing_files(fname_base, curry_vers):
    """Check if all necessary files exist."""
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
    """Read through the lines of a curry parameter files and save data.

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
    """Extract info from curry parameter files."""
    var_names = ['NumSamples', 'SampleFreqHz',
                 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'SAMPLE_FREQ_HZ',
                 'DATA_FORMAT', 'SAMPLE_TIME_USEC']

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
                    unit_dict[type] = data_unit.replace(" ", "") \
                        .replace("\n", "").split("=")[-1]

    n_samples = int(param_dict["numsamples"])
    sfreq = float(param_dict["samplefreqhz"])
    time_step = float(param_dict["sampletimeusec"]) * 1e-6
    is_ascii = param_dict["dataformat"] == "ASCII"

    if (sfreq == 0) and (time_step != 0):
        sfreq = 1. / time_step

    # read labels from label files
    labels = _read_curry_lines(fname_base + LABEL_FILE_EXTENSION[curry_vers],
                               ["LABELS" + CHANTYPES[key] for key in
                                ["meg", "eeg", "misc"]])

    sensors = _read_curry_lines(fname_base + LABEL_FILE_EXTENSION[curry_vers],
                                ["SENSORS" + CHANTYPES[key] for key in
                                 ["meg", "eeg", "misc"]])

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

    return info, n_samples, is_ascii


def _read_events_curry(fname, event_ids=None):
    """Read events from Curry event files.

    Parameters
    ----------
    fname : str
        Path to a curry event file with extensions .cef, .ceo,
        .cdt.cef, or .cdt.ceo
    event_ids : tuple, list or None (default None)
        If tuple or list, only the event IDs in event_ids
        will be read. If None, all event IDs will be read.

    Returns
    -------
    events : ndarray, shape (n_events, 3)
        The array of events.
    """
    check_fname(fname, 'curry event', ('.cef', '.cdt.cef'),
                endings_err=('.cef', '.cdt.cef'))

    events_dict = _read_curry_lines(fname, ["NUMBER_LIST"])
    # The first 3 column seem to contain the event information
    curry_events = np.array(events_dict["NUMBER_LIST"], dtype=int)[:, 0:3]

    if event_ids is not None:
        idx = np.array([i in event_ids for i in curry_events[:, -1]])
        curry_events = curry_events[idx]

    return curry_events


def _read_annotations_curry(fname, sfreq='auto'):
    r"""Read events from Curry event files.

    Parameters
    ----------
    fname : str
        The filename.
    sfreq : float | 'auto'
        The sampling frequency in the file. If set to 'auto' then the
        ``sfreq`` is taken from the respective info file of the same name with
        according file extension (\*.dap for Curry 7; \*.cdt.dpa for Curry8).
         So data.cef looks in data.dap and data.cdt.cef looks in data.cdt.dpa.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    events = _read_events_curry(fname)

    if sfreq == 'auto':
        fname_base, ext = fname.split(".", maxsplit=1)
        curry_vers = _get_curry_version(ext)
        with open(fname_base + INFO_FILE_EXTENSION[curry_vers]) as fid:
            for line in fid:
                if ('SampleFreqHz' or 'SAMPLE_FREQ_HZ') in line:
                    sfreq = float(line.split("=")[1])
    else:
        sfreq = sfreq

    onset = events[:, 0] / sfreq
    duration = np.zeros(events.shape[0])
    description = events[:, 2]

    return Annotations(onset, duration, description)


def read_raw_curry(input_fname, preload=False, verbose=None):
    """Read raw data from Curry files.

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
        on the hard drive (slower, requires less memory).
    %(verbose)s

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing Curry data.

    """
    return RawCurry(input_fname, preload, verbose)


class RawCurry(BaseRaw):
    """Raw object from Curry file.

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
        on the hard drive (slower, requires less memory).
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    """

    def __init__(self, input_fname, preload=False, verbose=None):

        # we don't use os.path.splitext to also handle extensions like .cdt.dpa
        fname_base, ext = input_fname.split(".", maxsplit=1)
        curry_vers = _get_curry_version(ext)
        _check_missing_files(fname_base, curry_vers)
        data_fname = os.path.abspath(fname_base +
                                     DATA_FILE_EXTENSION[curry_vers])

        info, n_samples, is_ascii = _read_curry_info(fname_base, curry_vers)

        last_samps = [n_samples - 1]
        self._is_ascii = is_ascii

        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps,
            orig_format='int', verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        if self._is_ascii:
            ch_idx = range(0, len(self.ch_names))
            if version.parse(np.version.version) >= version.parse("1.16.0"):
                block = np.loadtxt(self.filenames[0],
                                   skiprows=start,
                                   usecols=ch_idx[idx],
                                   max_rows=stop - start).T
            else:
                block = np.loadtxt(self.filenames[0],
                                   skiprows=start,
                                   usecols=ch_idx[idx]).T
                block = block[:, :stop - start]

            data_view = data[:, 0:block.shape[1]]
            _mult_cal_one(data_view, block, idx, cals, mult)

        else:
            _read_segments_file(self, data, idx, fi, start, stop, cals,
                                mult, dtype="<f4")
