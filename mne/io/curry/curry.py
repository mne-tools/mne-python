# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)


import numpy as np
import mne
import os
import re


def _read_curry(full_filename):
    '''
    Read curry into python.
    Inspired by Matt Pontifex' EEGlab loadcurry() extension

    :param full_filename:
    :return:
    '''

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    file, ext = full_filename.split(".", maxsplit=1)

    if 'cdt' in ext:
        curry_vers = 8
        if not (os.path.isfile(file + ".cdt") and
                os.path.isfile(file + ".cdt.dpa")):
            raise FileNotFoundError("The requested filename %s does not have "
                                    "both file components (.cdt, .cdt.dpa) "
                                    "created by Curry 8." % file)

    else:
        curry_vers = 7
        if not (os.path.isfile(file + ".dap") and
                os.path.isfile(file + ".dat") and
                os.path.isfile(file + ".rs3")):
            raise FileNotFoundError("The requested filename %s does not have "
                                    "all file components (.dap, .dat, .rs3) "
                                    "created by Curry 6 and 7." % file)

    #####################################
    # read parameters from the param file

    if (curry_vers == 7):
        file_extension = '.dap'
    elif (curry_vers == 8):
        file_extension = '.cdt.dpa'

    var_names = ['NumSamples', 'NumChannels', 'NumTrials', 'SampleFreqHz',
                 'TriggerOffsetUsec', 'DataFormat', 'SampleTimeUsec',
                 'NUM_SAMPLES', 'NUM_CHANNELS', 'NUM_TRIALS', 'SAMPLE_FREQ_HZ',
                 'TRIGGER_OFFSET_USEC', 'DATA_FORMAT', 'SAMPLE_TIME_USEC']

    param_dict = dict()
    with open(file + file_extension) as f:
        for line in f:
            if any(var_name in line for var_name in var_names):
                key, val = line.replace(" ", "").replace("\n", "").split("=")
                param_dict[key] = val

    if not len(param_dict) == 7:
        raise KeyError("Some variables cannot be found in the parameter file.")

    if "NumSamples" in param_dict.keys():
        n_samples = int(param_dict["NumSamples"])
        n_ch = int(param_dict["NumChannels"])
        n_trials = int(param_dict["NumTrials"])
        sfreq = float(param_dict["SampleFreqHz"])
        offset = float(param_dict["TriggerOffsetUsec"]) / 1e6  # convert to s
        time_step = float(param_dict["SampleTimeUsec"]) / 1e6
        data_format = param_dict["DataFormat"]

    else:
        n_samples = int(param_dict["NUM_SAMPLES"])
        n_ch = int(param_dict["NUM_CHANNELS"])
        n_trials = int(param_dict["NUM_TRIALS"])
        sfreq = float(param_dict["SAMPLE_FREQ_HZ"])
        offset = float(param_dict["TRIGGER_OFFSET_USEC"]) / 1e6
        time_step = float(param_dict["SAMPLE_TIME_USEC"]) / 1e6
        data_format = param_dict["DATA_FORMAT"]

    if (sfreq == 0) and (time_step != 0):
        sfreq = 1 / time_step

    #####################################
    # read labels from label files

    if (curry_vers == 7):
        file_extension = '.rs3'
    elif (curry_vers == 8):
        file_extension = '.cdt.dpa'

    ch_names = []
    ch_pos = []

    save_labels = False
    save_ch_pos = False
    with open(file + file_extension) as f:
        for line in f:

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

    #####################################
    # read events from cef/ceo files

    # TODO: actually read the event files
    try:
        if (curry_vers == 7):
            if os.path.isfile(file + '.cef'):
                file_extension = '.cef'
            else:
                file_extension = '.ceo'
        elif (curry_vers == 8):
            if os.path.isfile(file + '.cdt.cef'):
                file_extension = '.cdt.cef'
            else:
                file_extension = '.cdt.ceo'

        events = []

        save_events = False
        with open(file + file_extension) as f:
            for line in f:

                if "NUMBER_LIST END_LIST" in line:
                    save_events = False

                if save_events:
                    # print(line)
                    events.append(line)

                if "NUMBER_LIST START_LIST" in line:
                    save_events = True

    # .cef or ceo files don't always exist, so don't read events here
    except FileNotFoundError:
        pass

    #####################################
    # read data from dat/cdt files

    if (curry_vers == 7):
        file_extension = '.dat'
    elif (curry_vers == 8):
        file_extension = '.cdt'

    if data_format == "ASCII":
        with open(file + file_extension) as f:
            data = np.loadtxt(f).T

    else:
        with open(file + file_extension) as f:
            data = np.fromfile(f, dtype='float32')
            data = np.reshape(data, [n_ch, n_samples * n_trials])

    info = mne.create_info(ch_names, sfreq)

    return data, info, n_trials, offset


def read_raw_curry(input_fname):
    '''
    Create a mne.io.RawArray from curry files.

    :param input_fname:
    :return:
    '''

    data, info, n_trials, offset = _read_curry(input_fname)

    # TODO: create a RawCurry class instead of RawArray?
    return mne.io.RawArray(data, info, first_samp=offset)
