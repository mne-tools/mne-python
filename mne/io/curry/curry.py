# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os.path as op
from collections import namedtuple
import re
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..utils import _read_segments_file, _mult_cal_one
from ..constants import FIFF
from ...utils import check_fname, check_version, logger, verbose, warn
from ...annotations import Annotations

FILE_EXTENSIONS = {"Curry 7": {"info": ".dap",
                               "data": ".dat",
                               "labels": ".rs3",
                               "events": ".cef"},
                   "Curry 8": {"info": ".cdt.dpa",
                               "data": ".cdt",
                               "labels": ".cdt.dpa",
                               "events": ".cdt.cef"}}
CHANTYPES = {"meg": "_MAG1", "eeg": "", "misc": "_OTHERS"}
FIFFV_CHANTYPES = {"meg": FIFF.FIFFV_MEG_CH, "eeg": FIFF.FIFFV_EEG_CH,
                   "misc": FIFF.FIFFV_MISC_CH}
SI_UNITS = dict(V=FIFF.FIFF_UNIT_V, T=FIFF.FIFF_UNIT_T)
SI_UNIT_SCALE = dict(c=1e-2, m=1e-3, u=1e-6, μ=1e-6, n=1e-9, p=1e-12, f=1e-15)

CurryFileStructure = namedtuple('CurryFileStructure',
                                'info, data, labels, events')
CurryParameters = namedtuple('CurryParameters',
                             'n_samples, sfreq, is_ascii, unit_dict')


def _get_curry_version(file_extension):
    """Check out the curry file version."""
    return "Curry 8" if "cdt" in file_extension else "Curry 7"


def _get_curry_file_structure(fname, required=[]):
    """Store paths to a CurryFileStructure and check for required files."""
    _msg = "The following required files cannot be found: {0}.\nPlease make " \
           "sure all required files are located in the same directory as {1}."

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    fname_base, ext = fname.split(".", maxsplit=1)
    version = _get_curry_version(ext)

    info = fname_base + FILE_EXTENSIONS[version]["info"]
    data = fname_base + FILE_EXTENSIONS[version]["data"]
    labels = fname_base + FILE_EXTENSIONS[version]["labels"]
    events = fname_base + FILE_EXTENSIONS[version]["events"]
    my_curry = CurryFileStructure(
        info=info if op.isfile(info) else None,
        data=data if op.isfile(data) else None,
        labels=labels if op.isfile(labels) else None,
        events=events if op.isfile(events) else None)

    missing = [fname_base + FILE_EXTENSIONS[version][field]
               for field in required if getattr(my_curry, field) is None]
    if missing:
        raise FileNotFoundError(_msg.format(np.unique(missing), fname))

    return my_curry


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


def _read_curry_parameters(fname):
    """Extract Curry params from a Curry info file."""
    _msg_match = "The sampling frequency and the time steps extracted from " \
                 "the parameter file do not match."
    _msg_invalid = "sfreq must be greater than 0. Got sfreq = {0}"

    var_names = ['NumSamples', 'SampleFreqHz',
                 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'SAMPLE_FREQ_HZ',
                 'DATA_FORMAT', 'SAMPLE_TIME_USEC']

    param_dict = dict()
    unit_dict = dict()
    with open(fname) as fid:
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

    if time_step == 0:
        true_sfreq = sfreq
    elif sfreq == 0:
        true_sfreq = 1 / time_step
    elif not np.isclose(sfreq, 1 / time_step):
        raise ValueError(_msg_match)
    else:  # they're equal and != 0
        true_sfreq = sfreq
    if true_sfreq <= 0:
        raise ValueError(_msg_invalid.format(true_sfreq))

    return CurryParameters(n_samples, true_sfreq, is_ascii, unit_dict)


def _read_curry_info(info_fname, label_fname):
    """Extract info from curry parameter files."""
    curry_params = _read_curry_parameters(info_fname)

    # read labels from label files
    labels = _read_curry_lines(label_fname,
                               ["LABELS" + CHANTYPES[key] for key in
                                ["meg", "eeg", "misc"]])

    sensors = _read_curry_lines(label_fname,
                                ["SENSORS" + CHANTYPES[key] for key in
                                 ["meg", "eeg", "misc"]])

    all_chans = list()
    for key in ["meg", "eeg", "misc"]:
        for ind, chan in enumerate(labels["LABELS" + CHANTYPES[key]]):
            ch = {"ch_name": chan,
                  "unit": curry_params.unit_dict[key],
                  "kind": FIFFV_CHANTYPES[key]}
            if key in ("meg", "eeg"):
                loc = sensors["SENSORS" + CHANTYPES[key]][ind]
                ch["loc"] = np.array(loc, dtype=float)

            all_chans.append(ch)

    ch_names = [chan["ch_name"] for chan in all_chans]
    info = create_info(ch_names, curry_params.sfreq)

    for ind, ch_dict in enumerate(info["chs"]):
        ch_dict["kind"] = all_chans[ind]["kind"]
        ch_dict['unit'] = SI_UNITS[all_chans[ind]['unit'][1]]
        ch_dict['cal'] = SI_UNIT_SCALE[all_chans[ind]['unit'][0]]
        if ch_dict["kind"] in (FIFF.FIFFV_MEG_CH,
                               FIFF.FIFFV_EEG_CH):
            ch_dict["loc"] = all_chans[ind]["loc"]

    return info, curry_params.n_samples, curry_params.is_ascii


def _read_events_curry(fname):
    """Read events from Curry event files.

    Parameters
    ----------
    fname : str
        Path to a curry event file with extensions .cef, .ceo,
        .cdt.cef, or .cdt.ceo

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
    required = ["events", "info"] if sfreq == 'auto' else ["events"]
    curry_paths = _get_curry_file_structure(fname, required)
    events = _read_events_curry(curry_paths.events)

    if sfreq == 'auto':
        sfreq = _read_curry_parameters(curry_paths.info).sfreq

    onset = events[:, 0] / sfreq
    duration = np.zeros(events.shape[0])
    description = events[:, 2]

    return Annotations(onset, duration, description)


@verbose
def read_raw_curry(fname, preload=False, verbose=None):
    """Read raw data from Curry files.

    Parameters
    ----------
    fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing Curry data.
    """
    return RawCurry(fname, preload, verbose)


class RawCurry(BaseRaw):
    """Raw object from Curry file.

    Parameters
    ----------
    fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):

        curry_paths = _get_curry_file_structure(fname, required=["info",
                                                                 "data",
                                                                 "labels"])

        data_fname = op.abspath(curry_paths.data)

        info, n_samples, is_ascii = _read_curry_info(curry_paths.info,
                                                     curry_paths.labels)

        last_samps = [n_samples - 1]
        self._is_ascii = is_ascii

        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps,
            orig_format='int', verbose=verbose)

        if curry_paths.events is not None:
            logger.info('Event file found. Extracting Annotations from'
                        ' %s...' % curry_paths.events)
            annots = _read_annotations_curry(curry_paths.events,
                                             sfreq=self.info["sfreq"])
            self.set_annotations(annots)
        else:
            logger.info('Event file not found. No Annotations set.')

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        if self._is_ascii:
            ch_idx = range(0, len(self.ch_names))
            if check_version("numpy", "1.16.0"):
                block = np.loadtxt(self.filenames[0],
                                   skiprows=start,
                                   usecols=ch_idx[idx],
                                   max_rows=stop - start).T
            else:
                warn("Data reading might take longer for ASCII files. Update "
                     "numpy to version 1.16.0 or greater for more efficient "
                     "data reading.")
                block = np.loadtxt(self.filenames[0],
                                   skiprows=start,
                                   usecols=ch_idx[idx]).T
                block = block[:, :stop - start]
            data_view = data[:, :block.shape[1]]
            _mult_cal_one(data_view, block, idx, cals, mult)

        else:
            _read_segments_file(self, data, idx, fi, start, stop, cals,
                                mult, dtype="<f4")
