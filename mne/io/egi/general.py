"""General functions."""

import numpy as np
import os
from xml.dom.minidom import parse
import re


def _get_signal_bl(filepath):
    pib_signal_file, list_infofile = _get_signalfname(filepath, 'PNSData')
    eeg_info = _get_blocks(filepath, pib_signal_file[0])

    signal_blocks = dict(npibChan=eeg_info['nC'],
                         pibBinObj=eeg_info['blockNumSamps'],
                         pibBlocks=eeg_info['blocks'],
                         pibSignalFile=[],
                         sfreq=eeg_info['sfreq'],
                         nChan=eeg_info['nC'],
                         binObj=eeg_info['blockNumSamps'],
                         blocks=eeg_info['blocks'],
                         eegFile=pib_signal_file,
                         infoFile=list_infofile)
    return signal_blocks


def _extract(tags, filepath=None, obj=None):
    if obj is not None:
        fileobj = obj
    elif filepath is not None:
        fileobj = parse(filepath)
    else:
        raise ValueError('There is not object or file to extract data')
    infoxml = dict()
    for tag in tags:
        value = fileobj.getElementsByTagName(tag)
        infoxml[tag] = []
        for i in range(len(value)):
            infoxml[tag].append(value[i].firstChild.data)
    return infoxml


def _get_gains(filepath):
    file_obj = parse(filepath)
    objects = file_obj.getElementsByTagName('calibration')
    gains = dict()
    for ob in objects:
        value = ob.getElementsByTagName('type')
        if value[0].firstChild.data == 'GCAL':
            data_g = _extract(['ch'], obj=ob)['ch']
            gains.update(gcal=np.asarray(data_g, dtype=np.float64))
        elif value[0].firstChild.data == 'ICAL':
            data_g = _extract(['ch'], obj=ob)['ch']
            gains.update(ical=np.asarray(data_g, dtype=np.float64))
    return gains


def _get_ep_inf(filepath, samprate):
    epochfile = filepath + '/epochs.xml'
    epochlist = parse(epochfile)
    epochs = epochlist.getElementsByTagName('epoch')
    numepochs = epochs.length
    ep_begins = np.zeros((numepochs), dtype='i8')
    epochnumsamps = np.zeros((numepochs), dtype='i8')
    epochfirstblocks = np.zeros((numepochs), dtype='i8')
    epochlastblocks = np.zeros((numepochs), dtype='i8')
    epochtime0 = np.zeros((numepochs), dtype='i8')
    epochlabels = [None] * numepochs    # np.zeros((numEpochs), dtype='S50')
    epochsubjects = []
    epochfilenames = []
    epochsegstatus = [None] * numepochs  # np.zeros((numEpochs), dtype='S50')
    multisubj = False

    for p in range(numepochs):
        anepoch = epochs[p]
        epochbegin = int(anepoch.getElementsByTagName('beginTime')[0]
                         .firstChild.data)
        epochend = int(anepoch.getElementsByTagName('endTime')[0]
                       .firstChild.data)
        ep_begins[p] = _u2sample(epochbegin, samprate)[0]
        epochtime0[p] = ep_begins[p]
        epochnumsamps[p] = _u2sample(epochend, samprate)[0] - ep_begins[p]
        epochfirstblocks[p] = int(anepoch.getElementsByTagName('firstBlock')[0]
                                  .firstChild.data)
        epochlastblocks[p] = int(anepoch.getElementsByTagName('lastBlock')[0]
                                 .firstChild.data)
        epochlabels[p] = 'epoch'
    epochtype = 'cnt'
    totalnumsegs = 0

    categfile = os.path.join(filepath, 'categories.xml')
    if os.path.isfile(categfile):
        epochtype = 'seg'
        categlist = parse(categfile)
        cats = categlist.getElementsByTagName('cat')
        numcategs = cats.length

        multisubj = False
        if os.path.exists(filepath + '/subjects'):
            multisubj = True
            for p in range(numcategs):
                acateg = cats[p]
                seglist = acateg.getElementsByTagName('segments')
                numsegs = seglist.length
                for q in range(numsegs):
                    aseg = seglist[q]
                    keylistarray = aseg.getElementsByTagName('keyCode')
                    numkeys = keylistarray.length
                    numsubjs = 0
                    for r in range(numkeys):
                        akey = keylistarray[r]
                        if akey.firstChild.data.encode() == 'subj':
                            numsubjs = numsubjs + 1

        if multisubj:
            epochsubjects = [None] * numepochs
            # np.zeros((numEpo), dtype='S50')
            epochfilenames = [None] * numepochs
            # np.zeros((numEpochs), dtype='S50')
        for p in range(numcategs):
            acateg = cats[p]
            categlabel = acateg.getElementsByTagName('name')[0].firstChild.data
            seglist = acateg.getElementsByTagName('seg')
            numsegs = seglist.length
            totalnumsegs = totalnumsegs + numsegs
            for q in range(numsegs):
                aseg = seglist[q]
                segbegin = int(aseg.getElementsByTagName('beginTime')[0]
                               .firstChild.data)
                segbeginsamp = int(_u2sample(segbegin, samprate)[0])
                segind = np.where(ep_begins == segbeginsamp)[0]
                epochsegstatus[segind[0]] = aseg.getAttribute('status')
                epochlabels[segind[0]] = categlabel
                time0 = int(aseg.getElementsByTagName('evtBegin')[0]
                            .firstChild.data)
                time0samp = _u2sample(time0, samprate)[0]
                time0samp = time0samp - segbeginsamp
                epochtime0[segind[0]] = time0samp
                if multisubj:
                    keylistarray = aseg.getElementsByTagName('keyCode')
                    numkeys = keylistarray.length
                    datalistarray = aseg.getElementsByTagName('data')
                    for r in range(numkeys):
                        akey = keylistarray[r]
                        adata = datalistarray[r]
                        subject = None
                        filename = None
                        if adata.firstChild is not None:
                            if akey.firstChild.data.encode() == 'subj':
                                subject = adata.firstChild.data
                            elif akey.firstChild.data.encode() == 'FILE':
                                filename = adata.firstChild.data.encode()
                        epochsubjects[segind[0]] = subject
                        epochfilenames[segind[0]] = filename
    l_epfilen = len(set(epochfilenames))
    l_epsubjects = len(set(epochsubjects))
    if (multisubj and l_epfilen == 1 and l_epsubjects == 1):
        epochsubjects = []
        epochfilenames = []
        multisubj = False
    #  --------------------------------------------------------------------
    epoch_info = dict(epochType=epochtype,
                      epochBeginSamps=ep_begins,
                      epochNumSamps=epochnumsamps,
                      epochFirstBlocks=epochfirstblocks,
                      epochLastBlocks=epochlastblocks,
                      epochLabels=epochlabels,
                      epochTime0=epochtime0,
                      multiSubj=multisubj,
                      epochSubjects=epochsubjects,
                      epochFilenames=epochfilenames,
                      epochSegStatus=epochsegstatus)
    return epoch_info


def _get_blocks(filepath):
    """Get info from meta data blocks."""
    binfile = os.path.join(filepath)
    n_blocks = 0
    samples_block = []
    header_sizes = []
    with open(binfile, 'rb') as fid:
        fid.seek(0, 2)  # go to end of file
        file_length = fid.tell()
        block_size = file_length
        fid.seek(0)
        position = 0
        while position < file_length:
            block = _block_r(fid)
            if block is None:
                samples_block.append(samples_block[n_blocks])
                n_blocks += 1
                fid.seek(block_size, 1)
                position = fid.tell()
                continue
            block_size = block['block_size']
            header_size = block['header_size']
            header_sizes.append(header_size)
            samples_block.append(block['nsamples'])
            fid.seek(block_size, 1)
            sfreq = block['sfreq']
            n_channels = block['nc']
            position = fid.tell()

    samples_block = np.array(samples_block)
    signal_blocks = dict(n_channels=n_channels, sfreq=sfreq, n_blocks=n_blocks,
                         blockNumSamps=samples_block,
                         header_sizes=header_sizes)
    return signal_blocks


def _get_signalfname(filepath, infontype):
    listfiles = os.listdir(filepath)
    binfiles = list(f for f in listfiles if 'signal' in f and f[-4:] == '.bin')
    signalfile = []
    infofiles = []
    for binfile in binfiles:
        bin_num_str = re.search(r'\d+', binfile).group()
        infofile = 'info' + bin_num_str + '.xml'
        infobjfile = os.path.join(filepath, infofile)
        infobj = parse(infobjfile)
        if infobj.getElementsByTagName(infontype) is not None:
            signalfile.append('signal' + bin_num_str + '.bin')
            infofiles.append(infofile)
    return signalfile, infofiles


def _u2sample(microsecs, samprate):
    sampduration = 1000000. / samprate
    samplenum = np.float(microsecs) / sampduration
    reminder = np.float(microsecs) % sampduration
    samplenum = np.fix(samplenum)
    out = [samplenum, reminder]
    return out


def _bls2blns(n_samples, bn_sample):
    blockn = 0
    #  blockNumSamps(blockNum)
    n_sample = bn_sample[blockn]
    while n_samples > n_sample:
        blockn = blockn + 1
        # blockNumSamps(blockNum)
        n_sample = n_sample + bn_sample[blockn]
    # blockNumSamps(epochNum)
    n_sample = n_sample - bn_sample[blockn]
    sample = n_samples - n_sample
    return blockn, sample


def _block_r(fid):
    """Read meta data."""
    if np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0] != 1:  # not metadata
        return None
    header_size = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    block_size = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    hl = int(block_size / 4)
    nc = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    nsamples = int(hl / nc)
    np.fromfile(fid, dtype=np.dtype('i4'), count=nc)  # sigoffset
    sigfreq = np.fromfile(fid, dtype=np.dtype('i4'), count=nc)
    sfreq = (sigfreq[0] - 32) / ((nc - 1) * 2)
    count = int(header_size / 4 - (4 + 2 * nc))
    np.fromfile(fid, dtype=np.dtype('i4'), count=count)  # sigoffset
    # position = fid.tell()
    block = dict(nc=nc,
                 hl=hl,
                 nsamples=nsamples,
                 block_size=block_size,
                 header_size=header_size,
                 sfreq=sfreq)
    return block
