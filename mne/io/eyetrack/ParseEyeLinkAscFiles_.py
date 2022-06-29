# ParseEyeLinkAsc.py
# - Reads in .asc data files from EyeLink and produces pandas dataframes for
# further analysis
#
# Created 7/31/18-8/15/18 by DJ.
# Updated 7/4/19 by DJ - detects and handles monocular sample data.


def ParseEyeLinkAsc_(elFilename):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable
    # dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file from an AX-CPT
    # task in the current path.
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus
    # software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples
    #
    # Created 7/31/18-8/15/18 by DJ.
    # Updated 11/12/18 by DJ - switched from "trials" to "recording periods"
    # for experiments with continuous recording
    # Updated 9/??/19 by Dominik Welke - fixed read-in of data

    # Import packages
    import numpy as np
    import pandas as pd
    import time

    # ===== READ IN FILES ===== #
    # Read in EyeLink file
    print('Reading in EyeLink file %s...' % elFilename)
    t = time.time()
    with open(elFilename, "r+") as f:
        fileTxt0 = (line.rstrip() for line in f)
        # fileTxt0 = [line for line in fileTxt0 if line]  # Non-blank lines in
        # a list
        fileTxt0 = [line for line in fileTxt0]  # lines in a list
        fileTxt0 = np.array(fileTxt0)

    print('Done! Took %f seconds.' % (time.time() - t))

    # Separate lines into samples and messages
    print('Sorting lines...')
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER'] * nLines, dtype='object')
    # iStartRec = None
    iStartRec = []  # DW: make a list of all rec-starts
    t = time.time()
    for iLine in range(nLines):
        if len(fileTxt0[iLine]) < 3:
            lineType[iLine] = 'EMPTY'
        elif (
                fileTxt0[iLine].startswith('*') or
                fileTxt0[iLine].startswith('>>>>>')):
            lineType[iLine] = 'COMMENT'
        elif (
                fileTxt0[iLine].split()[0][0].isdigit() or
                fileTxt0[iLine].split()[0].startswith('-')):
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        # TODO: Find more general way of determining if recording has started
        # if '!CAL' in fileTxt0[iLine]:
        #    iStartRec = iLine + 1
        if 'START' in fileTxt0[iLine]:
            # DW: more general way of determining if recording has started..
            iStartRec.append(iLine + 1)
    print('Done! Took %f seconds.' % (time.time() - t))

    # ===== PARSE EYELINK FILE ===== #
    t = time.time()
    # Trials
    print('Parsing recording markers...')
    iNotStart = np.nonzero(lineType != 'START')[0]
    dfRecStart = pd.read_csv(elFilename, skiprows=iNotStart, header=None,
                             delim_whitespace=True, usecols=[1])
    dfRecStart.columns = ['tStart']
    iNotEnd = np.nonzero(lineType != 'END')[0]
    dfRecEnd = pd.read_csv(elFilename, skiprows=iNotEnd, header=None,
                           delim_whitespace=True, usecols=[1, 5, 6])
    dfRecEnd.columns = ['tEnd', 'xRes', 'yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart, dfRecEnd], axis=1)
    nRec = dfRec.shape[0]
    print('%d recording periods found.' % nRec)

    # Import Messages
    print('Parsing stimulus messages...')
    t = time.time()
    iMsg = np.nonzero(lineType == 'MSG')[0]
    # set up
    tMsg = []
    txtMsg = []
    t = time.time()
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    # Convert dict to dataframe
    dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})
    print('Done! Took %f seconds.' % (time.time() - t))

    # Import Fixations
    print('Parsing fixations...')
    t = time.time()
    iNotEfix = np.nonzero(lineType != 'EFIX')[0]
    dfFix = pd.read_csv(elFilename, skiprows=iNotEfix, header=None,
                        delim_whitespace=True, usecols=range(1, 8))
    dfFix.columns = ['eye', 'tStart', 'tEnd', 'duration',
                     'xAvg', 'yAvg', 'pupilAvg']
    nFix = dfFix.shape[0]
    print('Done! Took %f seconds.' % (time.time() - t))

    # Saccades
    print('Parsing saccades...')
    t = time.time()
    iNotEsacc = np.nonzero(lineType != 'ESACC')[0]
    dfSacc = pd.read_csv(elFilename, skiprows=iNotEsacc, header=None,
                         delim_whitespace=True, usecols=range(1, 11))
    dfSacc.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xStart',
                      'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']
    print('Done! Took %f seconds.' % (time.time() - t))

    # Blinks
    print('Parsing blinks...')
    iNotEblink = np.nonzero(lineType != 'EBLINK')[0]
    dfBlink = pd.read_csv(elFilename, skiprows=iNotEblink, header=None,
                          delim_whitespace=True, usecols=range(1, 5))
    dfBlink.columns = ['eye', 'tStart', 'tEnd', 'duration']
    print('Done! Took %f seconds.' % (time.time() - t))

    # determine sample columns based on eyes recorded in file
    eyesInFile = np.unique(dfFix.eye)
    if eyesInFile.size == 2:
        print('binocular data detected.')
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile[0]
        print('monocular data detected (%c eye).' % eye)
        cols = ['tSample', '%cX' % eye, '%cY' % eye, '%cPupil' % eye]
    # Import samples
    print('Parsing samples...')
    t = time.time()
    # iNotSample = np.nonzero(np.logical_or(
    #    lineType != 'SAMPLE', np.arange(nLines) < iStartRec))[0]
    iNotSample = np.nonzero(  # DW: try this, to get ALL data
        np.logical_or(
            lineType != 'SAMPLE', np.arange(nLines) < iStartRec[0]))[0]
    dfSamples = pd.read_csv(elFilename, skiprows=iNotSample, header=None,
                            delim_whitespace=True, usecols=range(0, len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyesInFile:
            dfSamples['%cX' % eye] = pd.to_numeric(dfSamples['%cX' % eye],
                                                   errors='coerce')
            dfSamples['%cY' % eye] = pd.to_numeric(dfSamples['%cY' % eye],
                                                   errors='coerce')
            dfSamples['%cPupil' % eye] = pd.to_numeric(
                dfSamples['%cPupil' % eye], errors='coerce')
        else:
            dfSamples['%cX' % eye] = np.nan
            dfSamples['%cY' % eye] = np.nan
            dfSamples['%cPupil' % eye] = np.nan

    print('Done! Took %.1f seconds.' % (time.time() - t))

    # Return new compilation dataframe
    return dfRec, dfMsg, dfFix, dfSacc, dfBlink, dfSamples
