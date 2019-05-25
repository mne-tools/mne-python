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
from ..utils import _read_segments_file, _find_channels, _create_chs


def _read_curry_events(fname_base, curry_vers):
    """
    Read curry files and convert the data for mne.
    Inspired by Matt Pontifex' EEGlab loadcurry() extension.

    :param full_fname:
    :return:
    """

    #####################################
    # read events from cef/ceo files

    curry_events = []

    if curry_vers == 7:
        if os.path.isfile(fname_base + '.cef'):
            file_extension = '.cef'
        elif os.path.isfile(fname_base + '.ceo'):
            file_extension = '.ceo'
        else:
            curry_events = None
    else:
        if os.path.isfile(fname_base + '.cdt.cef'):
            file_extension = '.cdt.cef'
        elif os.path.isfile(fname_base + '.cdt.ceo'):
            file_extension = '.cdt.ceo'
        else:
            curry_events = None

    if curry_events is not None:

        save_events = False
        with open(fname_base + file_extension) as fid:
            for line in fid:

                if "NUMBER_LIST END_LIST" in line:
                    save_events = False

                if save_events:
                    # print(line)
                    curry_events.append(line.split("\t"))

                if "NUMBER_LIST START_LIST" in line:
                    save_events = True

        curry_events = np.array(curry_events, dtype=int)

    # TODO: This returns events in a curry specific format. This might be reformatted to fit mne
    return curry_events


def _check_curry_file(full_fname):
    """
    Check if all neccessary files exist and return the path without extension
     and its CURRY version
     """

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    fname_base, ext = full_fname.split(".", maxsplit=1)

    if 'cdt' in ext:
        curry_vers = 8
        for check_ext in [".cdt", ".cdt.dpa"]:
            if not os.path.isfile(fname_base + check_ext):
                raise FileNotFoundError("The following required file cannot be"
                                        " found: %s. Please make sure it is "
                                        "located in the same directory as %s."
                                        % (fname_base + check_ext, full_fname))

    else:
        curry_vers = 7
        for check_ext in [".dap", ".dat", ".rs3"]:
            if not os.path.isfile(fname_base + check_ext):
                raise FileNotFoundError("The following required file cannot be"
                                        " found: %s. Please make sure it is "
                                        "located in the same directory as %s."
                                        % (fname_base + check_ext, full_fname))

    return fname_base, curry_vers


def _read_curry_info(fname_base, curry_vers):
    #####################################
    # read parameters from the param file

    if curry_vers == 7:
        file_extension = '.dap'
    else:
        file_extension = '.cdt.dpa'

    var_names = ['NumSamples', 'NumChannels', 'NumTrials', 'SampleFreqHz',
                 'TriggerOffsetUsec', 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'NUM_CHANNELS', 'NUM_TRIALS', 'SAMPLE_FREQ_HZ',
                 'TRIGGER_OFFSET_USEC', 'DATA_FORMAT', 'SAMPLE_TIME_USEC']

    param_dict = dict()
    with open(fname_base + file_extension) as fid:
        for line in fid:
            if any(var_name in line for var_name in var_names):
                key, val = line.replace(" ", "").replace("\n", "").split("=")
                param_dict[key] = val

    if not len(param_dict) == 7:
        raise KeyError("Some variables cannot be found in the parameter file.")

    if "NumSamples" in param_dict:
        n_samples = int(param_dict["NumSamples"])
        # n_ch = int(param_dict["NumChannels"])
        n_trials = int(param_dict["NumTrials"])
        sfreq = float(param_dict["SampleFreqHz"])
        offset = float(param_dict["TriggerOffsetUsec"]) / 1e6  # convert to s
        time_step = float(param_dict["SampleTimeUsec"]) / 1e6
        # data_format = param_dict["DataFormat"]

    else:
        n_samples = int(param_dict["NUM_SAMPLES"])
        # n_ch = int(param_dict["NUM_CHANNELS"])
        n_trials = int(param_dict["NUM_TRIALS"])
        sfreq = float(param_dict["SAMPLE_FREQ_HZ"])
        offset = float(param_dict["TRIGGER_OFFSET_USEC"]) / 1e6
        time_step = float(param_dict["SAMPLE_TIME_USEC"]) / 1e6
        # data_format = param_dict["DATA_FORMAT"]

    if (sfreq == 0) and (time_step != 0):
        sfreq = 1. / time_step

    #####################################
    # read labels from label files

    if curry_vers == 7:
        file_extension = '.rs3'
    else:
        file_extension = '.cdt.dpa'

    ch_names = []
    ch_pos = []

    save_labels = False
    save_ch_pos = False
    with open(fname_base + file_extension) as fid:
        for line in fid:

            if re.match("LABELS.*? END_LIST", line):
                save_labels = False

            # if "SENSORS END_LIST" in line:
            if re.match("SENSORS.*? END_LIST", line):
                save_ch_pos = False

            if save_labels:
                if line != "\n":
                    ch_names.append(line.replace("\n", ""))

            if save_ch_pos:
                ch_pos.append(line.split("\t"))

            if re.match("LABELS.*? START_LIST", line):
                save_labels = True

            # if "SENSORS START_LIST" in line:
            if re.match("SENSORS.*? START_LIST", line):
                save_ch_pos = True

    ch_pos = np.array(ch_pos, dtype=float)
    # TODO find a good method to set montage (do it in read_montage instead?)

    info = create_info(ch_names, sfreq)

    # TODO; There's still a lot more information that can be brought into info["chs"]. However i'm not sure what to do with MEG chans here
    for ch_dict in info["chs"]:
        ch_dict["cal"] = 1e-6

    return info, n_trials, n_samples, offset, curry_vers


def read_raw_curry(input_fname):
    """
    Read raw data from Curry files.

    Parameters
    ----------
    input_fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing CURRY data.

    """

    fname_base, curry_vers = _check_curry_file(input_fname)

    info, n_trials, n_samples, offset, curry_vers = _read_curry_info(fname_base, curry_vers)

    events = _read_curry_events(fname_base, curry_vers)

    info["events"] = events

    if curry_vers == 7:
        file_extension = ".dat"
    else:  # curry_vers == 8
        file_extension = ".cdt"

    raw = RawCurry(fname_base + file_extension, info, n_samples)

    return raw


class RawCurry(BaseRaw):
    """"""

    def __init__(self, data_fname, info, n_samples, montage=None, eog=(), ecg=(),
                 emg=(), misc=(), preload=False, verbose=None):  # noqa: D102

        data_fname = os.path.abspath(data_fname)

        last_samps = [n_samples - 1]

        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps, orig_format='int',
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="float32")
