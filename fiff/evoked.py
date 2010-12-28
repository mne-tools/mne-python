import numpy as np

from .constants import FIFF
from .open import fiff_open
from .tag import read_tag
from .tree import dir_tree_find
from .meas_info import read_meas_info


def read_evoked(fname, setno=0):
    """
    [data] = fiff_read_evoked(fname,setno)

    Read one evoked data set

    """

    if setno < 0:
       raise ValueError, 'Data set selector must be positive'

    print 'Reading %s ...\n' % fname
    fid, tree, _ = fiff_open(fname);

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)
    info['filename'] = fname

    #   Locate the data of interest
    processed = dir_tree_find(meas,FIFF.FIFFB_PROCESSED_DATA);
    if len(processed) == 0:
        fid.close()
        raise ValueError, 'Could not find processed data'

    evoked = dir_tree_find(meas,FIFF.FIFFB_EVOKED)
    if len(evoked) == 0:
        fid.close(fid)
        raise ValueError, 'Could not find evoked data'

    #   Identify the aspects
    #
    naspect = 0
    is_smsh = None
    sets = []
    for k in range(len(evoked)):
        aspects_k = dir_tree_find(evoked[k], FIFF.FIFFB_ASPECT)
        set_k = dict(aspects=aspects_k,
                     naspect=len(aspects_k))
        sets.append(set_k)

        if sets[k]['naspect'] > 0:
            if is_smsh is None:
                is_smsh = np.zeros((1, sets[k]['naspect']))
            else:
                is_smsh = np.c_[is_smsh, np.zeros((1, sets[k]['naspect']))]
            naspect += sets[k]['naspect']

        saspects  = dir_tree_find(evoked[k], FIFF.FIFFB_SMSH_ASPECT)
        nsaspects = len(saspects)
        if nsaspects > 0:
            sets[k]['naspect'] += nsaspects
            sets[k]['naspect'] = [sets[k]['naspect'], saspects] # XXX : potential bug
            is_smsh = np.c_[is_smsh, np.ones(1, sets[k]['naspect'])]
            naspect += nsaspects;

    print '\t%d evoked data sets containing a total of %d data aspects' \
          ' in %s\n' % (len(evoked), naspect, fname)

    if setno >= naspect or setno < 0:
        fid.close()
        raise ValueError, 'Data set selector out of range'

    #   Next locate the evoked data set
    #
    p = 0
    goon = True
    for k in range(len(evoked)):
        for a in range(sets[k]['naspect']):
            if p == setno:
                my_evoked = evoked[k]
                my_aspect = sets[k]['aspects'][a]
                goon = False
                break
            p += 1
        if not goon:
            break
    else:
        #   The desired data should have been found but better to check
        fid.close(fid)
        raise ValueError, 'Desired data set not found'

    #   Now find the data in the evoked block
    nchan = 0
    sfreq = -1
    q = 0
    chs = []
    comment = None
    for k in range(my_evoked.nent):
        kind = my_evoked.directory[k].kind
        pos  = my_evoked.directory[k].pos
        if kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, pos)
            first = tag.data
        elif kind == FIFF.FIFF_LAST_SAMPLE:
            tag = read_tag(fid, pos)
            last = tag.data
        elif kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = tag.data
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = tag.data
        elif kind ==FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
            q += 1

    if comment is None:
       comment = 'No comment'

    #   Local channel information?
    if nchan > 0:
        if chs is None:
            fid.close()
            raise ValueError, 'Local channel information was not found when it was expected.'

        if len(chs) != nchan:
            fid.close()
            raise ValueError, 'Number of channels and number of channel definitions are different'

        info.chs = chs
        info.nchan = nchan
        print '\tFound channel information in evoked data. nchan = %d\n' % nchan
        if sfreq > 0:
            info['sfreq'] = sfreq

    nsamp = last - first + 1
    print '\tFound the data of interest:\n'
    print '\t\tt = %10.2f ... %10.2f ms (%s)\n' % (
         1000*first/info['sfreq'], 1000*last/info['sfreq'], comment)
    if info['comps'] is not None:
        print '\t\t%d CTF compensation matrices available\n' % len(info['comps'])

    # Read the data in the aspect block
    nave = 1
    epoch = []
    for k in range(my_aspect.nent):
        kind = my_aspect.directory[k].kind
        pos  = my_aspect.directory[k].pos
        if kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_ASPECT_KIND:
            tag = read_tag(fid, pos)
            aspect_kind = tag.data
        elif kind == FIFF.FIFF_NAVE:
            tag = read_tag(fid, pos)
            nave = tag.data
        elif kind == FIFF.FIFF_EPOCH:
            tag = read_tag(fid, pos)
            epoch.append(tag)

    print '\t\tnave = %d aspect type = %d\n' % (nave, aspect_kind)

    nepoch = len(epoch)
    if nepoch != 1 and nepoch != info.nchan:
        fid.close()
        raise ValueError, 'Number of epoch tags is unreasonable '\
                          '(nepoch = %d nchan = %d)' % (nepoch, info.nchan)

    if nepoch == 1:
       #   Only one epoch
       all_data = epoch[0].data
       #   May need a transpose if the number of channels is one
       if all_data.shape[1] == 1 and info.nchan == 1:
          all_data = all_data.T

    else:
        #   Put the old style epochs together
        all_data = epoch[0].data.T
        for k in range(1, nepoch):
            all_data = np.r_[all_data, epoch[k].data.T]

    if all_data.shape[1] != nsamp:
        fid.close()
        raise ValueError, 'Incorrect number of samples (%d instead of %d)' % (
                                    all_data.shape[1], nsamp)

    #   Calibrate
    cals = np.array([info['chs'][k].cal for k in range(info['nchan'])])
    all_data = np.dot(np.diag(cals.ravel()), all_data)

    #   Put it all together
    data = dict(info=info, evoked=dict(aspect_kind=aspect_kind,
                                   is_smsh=is_smsh[setno],
                                   nave=nave, first=first,
                                   last=last, comment=comment,
                                   times=np.arange(first, last+1,
                                            dtype=np.float) / info['sfreq'],
                                   epochs=all_data))

    fid.close()

    return data
