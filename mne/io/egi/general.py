"""General functions."""

import numpy as np


def _get_signal_bl(filepath):
    pib_signal_file, list_infofile = _get_signalfname(filepath, 'PNSData')
    eeg_info = _get_signal_nbin(filepath, pib_signal_file[0])
    # All the files should be the same parameters, i guess
    if pib_signal_file == []:
        pib_info = dict(nC=0,
                        sampRate=0,
                        blocks=0,
                        blockNumSamps=[],
                        signalFile=[])
    else:
        pib_info = _get_signal_nbin(filepath, pib_signal_file[0])

    signal_blocks = dict(npibChan=pib_info['nC'],
                         pibBinObj=pib_info['blockNumSamps'],
                         pibBlocks=pib_info['blocks'],
                         pibSignalFile=[],
                         sampRate=eeg_info['sampRate'],
                         nChan=eeg_info['nC'],
                         binObj=eeg_info['blockNumSamps'],
                         blocks=eeg_info['blocks'],
                         eegFile=pib_signal_file,
                         infoFile=list_infofile)
    return signal_blocks


def _extract(tags, filepath=None, obj=None):
    from xml.dom.minidom import parse
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
    import numpy as np
    from xml.dom.minidom import parse
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
    import numpy as np
    from xml.dom.minidom import parse
    import os.path
    #  -----------------------------------------------------------------
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
                                subject = adata.firstChild.data   # .encode()
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


def _get_signal_nbin(filepath, signalnbin):
    import numpy as np

    binfile = filepath + '/' + signalnbin
    with open(binfile, 'rb') as fid:
        data = fid.read()
        fid.seek(0, 0)
        version = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
        if version != 0:     # siempre el primer valor version es 1?
            block = _block_r(fid)
            blocksize = block['blocksize']
            position = block['position']
            nsamples = block['nsamples']
        else:
            raise NotImplementedError('Only continuos files are supported')
        numblocks = int(len(data) // float(blocksize))
        samples_block = [nsamples]
        fid.seek(position + blocksize)
        if (len(data) / float(blocksize)) - numblocks > 0:
            numblocks = numblocks + 1
        for i in range(1, numblocks):
            version = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
            if version == 0:
                position = fid.tell()
                fid.seek(position + blocksize)
                position = fid.tell()
                samples_block.append(nsamples)
            else:
                block = _block_r(fid)
                blocksize = block['blocksize']
                position = block['position']
                nsamples = block['nsamples']
                samples_block.append(nsamples)
        samples_block = np.array(samples_block)
        signal_blocks = dict(nC=block['nc'],
                             sampRate=block['samprate'],
                             blocks=numblocks,
                             blockNumSamps=samples_block,
                             signalFile=signalnbin)
        return signal_blocks


def _get_signalfname(filepath, infontype):
    import os
    from xml.dom.minidom import parse
    import re
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


#  --------------------------------------------------------------------------
#  infoNType ='EEG' 'PNSData' 'Spectral' 'sourceData' 'JTF' 'TValues' 'Filter'
#  _u2sample.py
#  Python File
#  author Jasmine Song
#  date 7/21/2014
#  Copyright 2014 EGI. All rights reserved.
#  Support routine for MFF Python code. Not intended to be called directly.
#
#  Converts from nanoseconds to samples, given the sampling rate.
##

def _u2sample(microsecs, samprate):
    import numpy as np

    sampduration = 1000000. / samprate
    samplenum = np.float(microsecs) / sampduration
    reminder = np.float(microsecs) % sampduration
    samplenum = np.fix(samplenum)
#    out = {'sampleNum':sampleNum, 'remainder':remainder}
    out = [samplenum, reminder]
    return out
#  function [sampleNum, remainder] = _u2sample(microsecs, sampRate)
#  microsecs = double(microsecs);
#  sampDuration = 1000000/sampRate;
#  sampleNum = microsecs/sampDuration;
#  remainder = uint64(rem(microsecs, sampDuration));
#  sampleNum = fix(sampleNum);


def _read_signaln(filepath, nbinfile, blocksinepoch):
    """Read data signal."""
    import numpy as np

    binfile = filepath + '/' + nbinfile
    with open(binfile, 'rb') as fid:
        fid.seek(0, 0)
        position = fid.tell()
        datalist = []
        blocki = 0
        for b in range(len(blocksinepoch)):
            version = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
            if version != 0:
                block = _block_r(fid)
                blocksize = block['blocksize']
                nsamples = block['nsamples']
                nc = block['nc']
                hl = block['hl']
                if blocki in blocksinepoch:
                    data1 = np.fromfile(fid, dtype=np.dtype('f4'), count=hl)
                    data1 = data1.reshape((nc, nsamples))
                    datalist.append(data1)
                else:
                    position = fid.tell()
                    fid.seek(position + blocksize)
            else:
                if blocki in blocksinepoch:
                    data1 = np.fromfile(fid, dtype=np.dtype('f4'), count=hl)
                    data1 = data1.reshape((nc, nsamples))
                    datalist.append(data1)
                else:
                    position = fid.tell()
                    fid.seek(position + blocksize)
            blocki = blocki + 1
    return datalist


# sampleNum = 700000
# blockNumSamps = summaryInfo['blockNumSamps']
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
    fid.seek(4)
    headersize = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    blocksize = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    hl = int(blocksize / 4)  # como sabes que son 4
    nc = np.fromfile(fid, dtype=np.dtype('i4'), count=1)[0]
    nsamples = int(hl / nc)
    np.fromfile(fid, dtype=np.dtype('i4'), count=nc)  # sigoffset
    sigfreq = np.fromfile(fid, dtype=np.dtype('i4'), count=nc)
    samprate = (sigfreq[0] - 32) / ((nc - 1) * 2)
    count = int(headersize / 4 - (4 + 2 * nc))
    np.fromfile(fid, dtype=np.dtype('i4'), count=count)  # sigoffset
    position = fid.tell()
    block = dict(nc=nc,
                 hl=hl,
                 nsamples=nsamples,
                 blocksize=blocksize,
                 position=position,
                 samprate=samprate)
    return block
