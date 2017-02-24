"""Function for read the headers."""


def read_mff_header(filepath):
    """Header reader Function.

    Parameters:
    filepath = str
    """
    import numpy as np
    from xml.dom.minidom import parse
    from .general import _get_signal_bl, _get_ep_inf, _extract
#    from mff_getSummaryInfo import mff_getSummaryInfo
    # ----------------------------------------------------
    signal_blocks = _get_signal_bl(filepath)
    samprate = signal_blocks['sampRate']
    numblocks = signal_blocks['blocks']
    blocknumsamps = np.array(signal_blocks['binObj'])
    #  -----------------------------------------------------
    pibhasref = False
    pibnchans = 0
    if signal_blocks['pibSignalFile'] != []:
        pnssetfile = filepath + '/pnsSet.xml'
        pnssetobj = parse(pnssetfile)
        pnssensors = pnssetobj.getElementsByTagName('sensor')
        pibnchans = pnssensors.length
        if signal_blocks['npibChan'] - pibnchans == 1:
            pibhasref = True
    #  -----------------------------------------------------
    epoch_info = _get_ep_inf(filepath, samprate)
    #  -----------------------------------------------------
    blockbeginsamps = np.zeros((numblocks), dtype='i8')
    for x in range(0, (numblocks - 1)):
        blockbeginsamps[x + 1] = blockbeginsamps[x] + blocknumsamps[x]
    #  -----------------------------------------------------
    summaryinfo = dict(blocks=signal_blocks['blocks'],
                       eegFilename=signal_blocks['eegFile'],
                       infoFile=signal_blocks['infoFile'],
                       sampRate=signal_blocks['sampRate'],
                       nChans=signal_blocks['nChan'],
                       pibBinObj=signal_blocks['pibBinObj'],
                       pibBlocks=signal_blocks['pibBlocks'],
                       pibFilename=signal_blocks['pibSignalFile'],
                       pibNChans=pibnchans,
                       pibHasRef=pibhasref,
                       epochType=epoch_info['epochType'],
                       epochBeginSamps=epoch_info['epochBeginSamps'],
                       epochNumSamps=epoch_info['epochNumSamps'],
                       epochFirstBlocks=epoch_info['epochFirstBlocks'],
                       epochLastBlocks=epoch_info['epochLastBlocks'],
                       epochLabels=epoch_info['epochLabels'],
                       epochTime0=epoch_info['epochTime0'],
                       multiSubj=epoch_info['multiSubj'],
                       epochSubjects=epoch_info['epochSubjects'],
                       epochFilenames=epoch_info['epochFilenames'],
                       epochSegStatus=epoch_info['epochSegStatus'],
                       blockBeginSamps=blockbeginsamps,
                       blockNumSamps=blocknumsamps)
    # ----------------------------------------------------
    # Pull header info from the summary info.
    nsamplespre = 0
    if summaryinfo['epochType'] == 'seg':
        nsamples = summaryinfo['epochNumSamps'][0]
        ntrials = len(summaryinfo['epochNumSamps'])
#        ntrials = summaryInfo['blocks']
        # if Time0 is the same for all segments...
        if len(set(summaryinfo['epochTime0'])) == 1:
            nsamplespre = summaryinfo['epochTime0'][0]
    else:
        nsamples = sum(summaryinfo['blockNumSamps'])
        ntrials = 1
    # --------------------------------------------------------------------------
    # Add the sensor info.
    sensor_layout_file = filepath + '/sensorLayout.xml'
    sensor_layout_obj = parse(sensor_layout_file)
    sensors = sensor_layout_obj.getElementsByTagName('sensor')
    label = []
    chantype = []
    chanunit = []
    tmp_label = []
    n_chans = 0
    for sensor in sensors:
        sensortype = int(sensor.getElementsByTagName('type')[0]
                         .firstChild.data)
        if sensortype == 0 or sensortype == 1:
            if sensor.getElementsByTagName('name')[0].firstChild is None:
                sn = sensor.getElementsByTagName('number')[0].firstChild.data
                sn = sn.encode()
                tmp_label = 'E' + sn.decode()
            else:
                sn = sensor.getElementsByTagName('name')[0].firstChild.data
                sn = sn.encode()
                tmp_label = sn.decode()
            label.append(tmp_label)
            chantype.append('eeg')
            chanunit.append('uV')
            n_chans = n_chans + 1
    if n_chans != summaryinfo['nChans']:
        print("Error. Should never occur.")
    # ---------------------------------------------------------------------------
    if summaryinfo['pibNChans'] > 0:
        pns_set_file = filepath + '/pnsSet.xml'
        pns_set_obj = parse(pns_set_file)
        pns_sensors = pns_set_obj.getElementsByTagName('sensor')
        for p in range(summaryinfo['pibNChans']):
            tmp_label = 'pib' + str(p + 1)
            label.append(tmp_label)
            pns_sensor_obj = pns_sensors[p]
            chantype.append(pns_sensor_obj.getElementsByTagName('name')[0]
                            .firstChild.data.encode())
            chanunit.append(pns_sensor_obj.getElementsByTagName('unit')[0]
                            .firstChild.data.encode())

    n_chans = n_chans + summaryinfo['pibNChans']
    # ------------------------------------------------------------------------------
    info_filepath = filepath + "/" + "info.xml"  # add with filepath
    tags = ['mffVersion', 'recordTime']
    version_and_date = _extract(tags, filepath=info_filepath)
    header = dict(Fs=summaryinfo['sampRate'],
                  version=version_and_date['mffVersion'][0],
                  date=version_and_date['recordTime'][0],
                  nChans=n_chans,
                  nSamplesPre=nsamplespre,
                  nSamples=nsamples,
                  nTrials=ntrials,
                  label=label,
                  chantype=chantype,
                  chanunit=chanunit,
                  orig=summaryinfo)
    return header
