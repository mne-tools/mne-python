"""Read the data."""


def read_mff_data(filepath, indtype, startind, lastind, hdr):
    """Function for load the data in a list.

    Parameters:
    filepath = str
    indtype = "sample" or "epoch"
    startind = the start index (int)
    lastind = the last index (int)
    hdr = the header array from read_mff_header
    """
    import numpy as np
    import os
    from .general import _bls2blns, _read_signaln, _get_gains

    suminfo = hdr['orig']
    blockns = suminfo['blockNumSamps']
    if len(suminfo['infoFile']) == 1:
        info_fp = os.path.join(filepath, suminfo['infoFile'][0])
        gains = _get_gains(info_fp)
    else:
        print('Multisubject not suported')
    if indtype == 'sample':
        st_blockn, st_sample = _bls2blns(startind - 1, blockns)
        ls_blockn, ls_sample = _bls2blns(lastind, blockns)
        blocksinepoch = range(st_blockn, ls_blockn + 1)
    elif indtype == 'epoch':
        blocksinepoch = []
        for i in range(startind - 1, lastind):
            blocksind = range(suminfo['epochFirstBlocks'][i] - 1,
                              suminfo['epochLastBlocks'][i])
            blocksinepoch.extend(blocksind)
    else:
        raise NameError("Indtype must to be either 'epoch' or 'sample'")
    # -------------------------------------------------
    dataeeg = _read_signaln(filepath, suminfo['eegFilename'][0], blocksinepoch)
    if suminfo['pibFilename'] != []:
        datapib = _read_signaln(filepath,
                                suminfo['pibFilename'][0],
                                blocksinepoch)
        datalist = []
        for i in range(len(dataeeg)):
            datai = np.concatenate((dataeeg[i], datapib[i]), axis=0)
            datalist.append(datai)
    else:
        datalist = dataeeg
    # --------------------------------------------------
    if indtype == 'sample':
        if st_blockn == ls_blockn:
            data = datalist[0][:, st_sample:ls_sample]
        elif ls_blockn - st_blockn == 1:
            datalists = datalist[0][:, st_sample:]
            datalistl = datalist[1][:, :ls_sample]
            data = np.concatenate((datalists, datalistl), axis=1)
        else:
            datalists = datalist[0][:, st_sample:]
            for j in range(1, len(datalist) - 1):
                datalistj = datalist[j]
                datalists = np.concatenate((datalists, datalistj), axis=1)
            datalistl = datalist[-1][:, :ls_sample]
            data = np.concatenate((datalists, datalistl), axis=1)
    elif indtype == 'epoch':
        if len(datalist) > 1:
            if len(set(blockns)) > 1:
                data = datalist[0]
                for j in range(1, len(datalist)):
                    datalistj = datalist[j]
                    data = np.concatenate((data, datalistj), axis=1)
            else:
                data = np.zeros((datalist[0].shape[0],
                                 datalist[0].shape[1],
                                 len(datalist)))
                for j in range(len(datalist)):
                    data[:, :, j] = datalist[j]
        else:
            data = datalist[0]
    if 'gcal' in gains:
        for f in range(data.shape[1]):
            data[:, f] = data[:, f] * gains['gcal']  # Check the speed
    return data
