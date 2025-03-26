import logging as log
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np


def read(inputfilename="", plotdata=1, verbosity=2):
    """Curry Reader Help.

    Usage:
    currydata = read(inputfilename = '', plotdata = 1, verbosity = 2)

    Inputs:
    inputfilename:      if left empty, reader will prompt user with file selection box,
                        otherwise specify filename with path;
                        supported files are: raw float (cdt), ascii (cdt), legacy raw
                        float (dat) and legacy ascii (dat)
    plotdata:           plotdata = 0, don't show plot
                        plotdata = 1, show plot (default)
                        plotdata = x, with x > 1, shows and automatically closes plot
                        after x seconds
    verbosity:          1 is low, 2 is medium (default) and 3 is high

    Output as dictionary with keys:
    'data'              functional data matrix (e.g. EEG, MEG) with dimensions (samples,
                        channels)
    'info'              data information with keys:
                        {'samples', 'channels', 'trials', 'samplingfreq'}
    'labels'            channel labels list
    'sensorpos'         channel locations matrix [x,y,z]
    'events'            events matrix where every row corresponds to:
                        [event latency, event type, event start, event stop]
    'annotations'       events annotation list
    'epochinfo'         epochs matrix where every row corresponds to:
                        [number of averages, total epochs, type, accept, correct,
                        response, response time]
    'epochlabels'       epoch labels list
    'impedancematrix'   impedance matrix with max size (channels, 10), corresponding to
                        last ten impedance measurements
    'landmarks'         functional, HPI or headshape landmarks locations
    'landmarkslabels'   labels for functional (e.g. LPA, Nasion,...), HPI (e.g. HPI 1,
                        HPI 2,...) or headshape (e.g. H1, H2,...) landmarks
    'hpimatrix'         HPI-coil measurements matrix (Orion-MEG only) where every row is
                        [measurementsample, dipolefitflag, x, y, z, deviation]

    2021 - Compumedics Neuroscan
    """
    # configure verbosity logging
    verbositylevel = (
        log.WARNING
        if verbosity == 1
        else log.INFO
        if verbosity == 2
        else log.DEBUG
        if verbosity == 3
        else log.INFO
    )

    log.basicConfig(format="%(levelname)s: %(message)s", level=verbositylevel)

    if inputfilename == "":
        try:
            # create root window for filedialog
            root = tk.Tk()
            root.withdraw()

            # check if last used directory was kept
            lastdirfilename = "lastdir.txt"
            if os.path.isfile(lastdirfilename):
                lastdirfile = open(lastdirfilename)
                initdir = lastdirfile.read().strip()
                lastdirfile.close()
            else:
                initdir = "/"

            filepath = filedialog.askopenfilename(
                initialdir=initdir,
                title="Open Curry Data File",
                filetypes=(
                    ("All Curry files", "*.cdt *.dat"),
                    ("cdt files", "*.cdt"),
                    ("dat files", "*.dat"),
                    ("all files", "*.*"),
                ),
            )
            root.destroy()

            lastdirfile = open(lastdirfilename, "w")
            lastdirfile.write(os.path.dirname(filepath))
            lastdirfile.close()

            # handle cancel
            if not filepath:
                raise Exception
        except Exception as _:
            raise Exception("Unable to open file")
    else:
        filepath = inputfilename

    # pathname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    try:
        basename, extension = filepath.split(".", maxsplit=1)
    except Exception as _:
        raise Exception("Unsupported file, choose a cdt or dat file")

    parameterfile = ""
    parameterfile2 = ""
    labelfile = ""
    # labelfile2 = ""
    eventfile = ""
    eventfile2 = ""
    hpifile = ""

    if extension == "dat":
        parameterfile = basename + ".dap"
        labelfile = basename + ".rs3"
        eventfile = basename + ".cef"
        eventfile2 = basename + ".ceo"
    elif extension == "cdt":
        parameterfile = filepath + ".dpa"
        parameterfile2 = filepath + ".dpo"
        eventfile = filepath + ".cef"
        eventfile2 = filepath + ".ceo"
        hpifile = filepath + ".hpi"
    else:
        raise Exception("Unsupported extension, choose a cdt or dat file")

    log.info("Reading file %s ...", filename)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # open parameter file

    contents = []

    try:
        fid = open(parameterfile)
        contents = fid.read()
    except Exception as _:
        log.debug("Could not open parameter file, trying alternative extension...")

    # open alternative parameter file
    if not contents:
        try:
            fid = open(parameterfile2)
            contents = fid.read()
        except FileNotFoundError:
            raise FileNotFoundError("Parameter file not found")
        except Exception as _:
            raise Exception("Could not open alternative parameter file")

    fid.close()

    if not contents:
        raise Exception("Parameter file is empty")

    # check for compressed file format
    ctok = "DataGuid"
    ix = contents.find(ctok)
    ixstart = contents.find("=", ix) + 1
    ixstop = contents.find("\n", ix)

    if ix != -1:
        text = contents[ixstart:ixstop].strip()
        if text == "{2912E8D8-F5C8-4E25-A8E7-A1385967DA09}":
            raise Exception(
                "Unsupported compressed data format, use Curry to convert file to raw "
                "float format"
            )

    # read parameters from parameter file
    # tokens (second line is for Curry 6 notation)
    tok = [
        "NumSamples",
        "NumChannels",
        "NumTrials",
        "SampleFreqHz",
        "TriggerOffsetUsec",
        "DataFormat",
        "DataSampOrder",
        "SampleTimeUsec",
        "NUM_SAMPLES",
        "NUM_CHANNELS",
        "NUM_TRIALS",
        "SAMPLE_FREQ_HZ",
        "TRIGGER_OFFSET_USEC",
        "DATA_FORMAT",
        "DATA_SAMP_ORDER",
        "SAMPLE_TIME_USEC",
    ]

    # scan keywords - all keywords must exist!
    nt = len(tok)
    a = [0] * nt  # initialize
    for i in range(nt):
        ctok = tok[i]
        ix = contents.find(ctok)
        ixstart = contents.find("=", ix) + 1  # skip =
        ixstop = contents.find("\n", ix)
        if ix != -1:
            text = contents[ixstart:ixstop].strip()
            if text == "ASCII" or text == "CHAN":  # test for alphanumeric values
                a[i] = 1
            elif text.isnumeric():
                a[i] = float(text)  # assign if it was a number

    # derived variables.  numbers (1) (2) etc are the token numbers
    nSamples = int(a[0] + a[int(0 + nt / 2)])
    nChannels = int(a[1] + a[int(1 + nt / 2)])
    nTrials = int(a[2] + a[int(2 + nt / 2)])
    fFrequency = a[3] + a[int(3 + nt / 2)]
    fOffsetUsec = a[4] + a[int(4 + nt / 2)]
    nASCII = int(a[5] + a[int(5 + nt / 2)])
    nMultiplex = int(a[6] + a[int(6 + nt / 2)])
    fSampleTime = a[7] + a[int(7 + nt / 2)]

    datainfo = {
        "samples": nSamples,
        "channels": nChannels,
        "trials": nTrials,
        "samplingfreq": fFrequency,
    }
    log.info(
        "Number of samples = %s, number of channels = %s, number of trials/epochs = %s,"
        " sampling frequency = %s Hz",
        str(nSamples),
        str(nChannels),
        str(nTrials),
        str(fFrequency),
    )

    if fFrequency == 0 or fSampleTime != 0:
        fFrequency = 1000000 / fSampleTime

    # try to guess number of samples based on datafile size
    if nSamples < 0:
        if nASCII == 1:
            raise Exception(
                "Number of samples cannot be guessed from ASCII data file. "
                "Use Curry to convert this file to Raw Float format"
            )
        else:
            log.warning(
                "Number of samples not present in parameter file. It will be estimated "
                "from size of data file"
            )
            fileSize = os.path.getsize(filepath)
            nSamples = fileSize / (4 * nChannels * nTrials)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # search for Impedance Values
    tixstart = contents.find("IMPEDANCE_VALUES START_LIST")
    tixstart = contents.find("\n", tixstart)
    tixstop = contents.find("IMPEDANCE_VALUES END_LIST")

    impedancelist = []

    if tixstart != -1 and tixstop != 1:
        text = contents[tixstart : tixstop - 1].split()
        for imp in text:
            if int(imp) != -1:  # skip?
                impedancelist.append(float(imp))

        # Curry records last 10 impedances
        try:
            impedancematrix = np.asarray(impedancelist, dtype=float).reshape(
                int(len(impedancelist) / nChannels), nChannels
            )
        except ValueError:
            impedancematrix = np.empty((int(len(impedancelist) / nChannels), nChannels))

    if impedancematrix.any():
        log.info("Found impedance matrix")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # open label file
    if extension == "dat":
        try:
            fid = open(labelfile)
            contents = fid.read()
            fid.close()
        except Exception as _:
            log.warning("Found no label file")

    # read labels from label file
    # initialize labels
    labels = [""] * nChannels

    for i in range(nChannels):
        labels[i] = "EEG" + str(i + 1)

    # scan for LABELS (occurs four times per channel group)
    ix = findtokens("\nLABELS", contents)
    nc = 0

    if ix:
        for i in range(3, len(ix), 4):  # loop over channel groups
            text = contents[ix[i - 1] : ix[i]]
            text = text[text.find("\n", 1) :].split()
            last = nChannels - nc
            numLabels = min(last, len(text))
            for j in range(numLabels):  # loop over labels
                labels[nc] = text[j]
                nc += 1
        log.info("Found channel labels")
    else:
        log.warning("Using dummy labels (EEG1, EEG2, ...)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # search for landmarks
    landmarks = []
    landmarkslabels = []

    # scan for LANDMARKS (occurs four times per channel group)
    ix = findtokens("\nLANDMARKS", contents)
    nc = 0
    totallandmarks = 0
    numlandmarksgroup = []  # number of landmarks per group

    if ix:
        for i in range(
            3, len(ix), 4
        ):  # first pass over groups to find total of landmarks
            text = contents[ix[i - 1] : ix[i]]
            text = text[text.find("\n", 1) :].splitlines()[1:]
            totallandmarks += len(text)
            numlandmarksgroup.append(len(text))

        lmpositions = np.zeros([totallandmarks, 3])
        for i in range(3, len(ix), 4):  # loop over channel groups
            text = contents[ix[i - 1] : ix[i]]
            text = text[text.find("\n", 1) :].split()
            last = totallandmarks - nc
            numlandmarks = min(last, int(len(text) / 3))
            for j in range(0, numlandmarks * 3, 3):
                lmpositions[nc][:] = np.array(text[j : j + 3])
                nc += 1

        landmarks = lmpositions
        log.info("Found landmarks")

    # landmark labels
    ix = findtokens("\nLM_REMARKS", contents)
    landmarkslabels = [""] * totallandmarks
    startindex = 0
    count = 0

    if ix and totallandmarks:
        for i in range(3, len(ix), 4):  # loop over channel groups
            text = contents[ix[i - 1] : ix[i]]
            text = text[text.find("\n", 1) :].splitlines()[1:]
            landmarkslabels[startindex : startindex + len(text)] = text
            startindex += numlandmarksgroup[count]
            count += 1

    ##########################################################################
    # read sensor locations from label file
    sensorpos = []

    # scan for SENSORS (occurs four times per channel group)
    ix = findtokens("\nSENSORS", contents)
    nc = 0

    if ix:
        grouppospersensor = []
        maxpersensor = 0
        numchanswithpos = 0
        for i in range(
            3, len(ix), 4
        ):  # first pass over groups to determine sensorpos and maxpersensor sizes
            text = contents[ix[i - 1] : ix[i]]
            text = text[text.find("\n", 1) :].splitlines()[1:]
            numchanswithpos += len(text)
            pospersensor = len(text[0].split())
            maxpersensor = max(pospersensor, maxpersensor)
            grouppospersensor.append(pospersensor)

        if (
            (maxpersensor == 3 or maxpersensor == 6)
            # 3 is (x,y,z) per sensor (EEG,MEG)
            # 6 is (x,y,z,x1,y1,z1) per sensor (MEG)
            and numchanswithpos > 0
            and numchanswithpos <= nChannels
        ):
            positions = np.zeros((numchanswithpos, maxpersensor))

            for group, i in enumerate(range(3, len(ix), 4)):  # loop over channel groups
                text = contents[ix[i - 1] : ix[i]]
                text = text[text.find("\n", 1) :].split()
                last = nChannels - nc
                pospersensor = grouppospersensor[group]
                numchannels = min(last, int(len(text) / pospersensor))
                for j in range(0, numchannels * pospersensor, pospersensor):
                    positions[nc][:pospersensor] = np.array(text[j : j + pospersensor])
                    nc += 1

            sensorpos = positions
            log.info("Found sensor positions")
        else:
            log.warning("Reading sensor positions failed (dimensions inconsistency)")
    else:
        log.warning("No sensor positions were found")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # search for epoch labels
    epochlabelslist = []

    if extension == "dat":
        try:
            fid = open(parameterfile)
            contents = fid.read()
            fid.close()
        except Exception as _:
            log.warning("Found no parameter file")

    ctok = "\nEPOCH_LABELS"
    if ctok in contents:
        tixstart = contents.find("EPOCH_LABELS START_LIST")
        tixstart = contents.find("\n", tixstart)
        tixstop = contents.find("EPOCH_LABELS END_LIST")

        if tixstart != -1 and tixstop != 1:
            epochlabelslist = contents[tixstart : tixstop - 1].split()

    if epochlabelslist:
        log.info("Found epoch labels")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # search for epoch information
    tixstart = contents.find("EPOCH_INFORMATION START_LIST")
    tixstart = contents.find("\n", tixstart)
    tixstop = contents.find("EPOCH_INFORMATION END_LIST")
    infoelements = 7
    epochinformation = []

    if tixstart != -1 and tixstop != 1:
        epochinformation = np.zeros((len(epochlabelslist), infoelements))
        text = contents[tixstart : tixstop - 1].split()
        for i in range(0, len(text), infoelements):
            for j in range(infoelements):
                epochinformation[int(i / infoelements)][j] = int(text[i + j])

    if epochinformation.any():
        log.info("Found epoch information")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # read events from event file
    # initialize events
    events = []
    annotations = []
    contents = []

    try:
        fid = open(eventfile)
        contents = fid.read()
    except Exception as _:
        log.debug("Trying event file alternative extension...")

    # open alternative event file
    if fid.closed:
        try:
            fid = open(eventfile2)
            contents = fid.read()
        except Exception as _:
            log.debug("Found no event file")

    fid.close()

    if contents:
        # scan for NUMBER_LIST (occurs five times)
        tixstart = contents.find("NUMBER_LIST START_LIST")
        tixstart = contents.find("\n", tixstart)
        tixstop = contents.find("NUMBER_LIST END_LIST")
        numberelements = 11
        numbereventprops = 4

        text = contents[tixstart : tixstop - 1].split()
        events = np.zeros((0, numbereventprops))

        for i in range(0, len(text), numberelements):
            sample = int(text[i])
            etype = int(text[i + 2])
            startsample = int(text[i + 4])
            endsample = int(text[i + 5])
            newevent = np.array([sample, etype, startsample, endsample])
            events = np.vstack([events, newevent])  # concat new event in events matrix

        # scan for REMARK_LIST (occurs five times)
        tixstart = contents.find("REMARK_LIST START_LIST")
        tixstart = contents.find("\n", tixstart)
        tixstop = contents.find("REMARK_LIST END_LIST")

        if tixstart != -1 and tixstop != 1:
            annotations = contents[tixstart : tixstop - 1].splitlines()

        log.info("Found events")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # read HPI coils (only Orion-MEG) if present

    hpimatrix = []
    contents = []

    try:
        fid = open(hpifile)
        contents = fid.read()
    except Exception as _:
        log.debug("Found no HPI file")

    fid.close()

    if contents:
        # get file version and number of coils
        tixstart = contents.find("FileVersion")
        tixstop = contents.find("\n", tixstart)
        text = contents[tixstart:tixstop].split()
        # hpifileversion = text[1]

        tixstart = contents.find("NumCoils")
        tixstop = contents.find("\n", tixstart)
        text = contents[tixstart:tixstop].split()
        # numberofcoils = text[1]

        hpimatrix = np.loadtxt(hpifile, dtype=np.float32, skiprows=3)
        log.info("Found HPI matrix")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % read data file

    data = []

    try:
        itemstoread = nSamples * nTrials * nChannels
        if nASCII == 1:
            data = np.fromfile(
                filepath, dtype=np.float32, count=itemstoread, sep=" "
            ).reshape(nSamples * nTrials, nChannels)
        else:
            data = np.fromfile(
                filepath,
                dtype=np.float32,
                count=itemstoread,
            ).reshape(nSamples * nTrials, nChannels)
    except FileNotFoundError:
        raise FileNotFoundError("Data file not found")
    except Exception as _:
        raise Exception("Could not open data file")

    if nSamples * nTrials != data.shape[0]:
        log.warning(
            "Inconsistent number of samples. File may be displayed incompletely"
        )
        nSamples = data.shape[0] / nTrials

    # transpose?
    if nMultiplex == 1:
        data = data.transpose()

    if plotdata > 0 and data.any():
        time = np.linspace(
            fOffsetUsec / 1000,
            fOffsetUsec / 1000 + (nSamples * nTrials - 1) * 1000 / fFrequency,
            nSamples * nTrials,
            dtype=np.float32,
        )
        # avoid logging output from matplotlib
        log.getLogger("matplotlib.font_manager").disabled = True
        # stacked plot
        amprange = max(abs(data.min()), abs(data.max()))
        shift = np.linspace(
            (nChannels - 1) * amprange * 0.3, 0, nChannels, dtype=np.float32
        )
        data += np.tile(shift, (nSamples * nTrials, 1))
        fig, ax = plt.subplots()
        ax.plot(time, data)
        ax.set_yticks(shift)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time [ms]")
        ax.set_title(filename)
        log.info("Found data file")
        if plotdata == 1:
            plt.show()
        elif plotdata > 1:
            plt.show(block=False)
            plt.pause(plotdata)
            plt.close()
        else:
            log.warning("Invalid plotdata input: please see description in help")

    # assemble output dict
    output = {
        "data": data,
        "info": datainfo,
        "labels": labels,
        "sensorpos": sensorpos,
        "events": events,
        "annotations": annotations,
        "epochinfo": epochinformation,
        "epochlabels": epochlabelslist,
        "impedances": impedancematrix,
        "landmarks": landmarks,
        "landmarkslabels": landmarkslabels,
        "hpimatrix": hpimatrix,
    }

    return output


def findtokens(token, contents):
    """Findtoken.

    Returns indices of token occurrences in input string contents.
    """
    if not token or not contents:
        raise Exception("Invalid input for finding token")

    tokenindices = []
    index = 0
    while index < len(contents):
        index = contents.find(token, index)
        if index == -1:
            break
        tokenindices.append(index)
        index += len(token)
    return tokenindices
