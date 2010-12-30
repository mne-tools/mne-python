import numpy as np

from .constants import FIFF
from .open import fiff_open
from .tag import read_tag
from .tree import dir_tree_find
from .meas_info import read_meas_info


def read_evoked(fname, setno=0):
    """Read an evoked dataset

    Parameters
    ----------
    fname: string
        The file name.

    setno: int
        The index of the evoked dataset to read. FIF
        file can contain multiple datasets.

    Returns
    -------
    data: dict
        The evoked dataset

    """
    if setno < 0:
        raise ValueError, 'Data set selector must be positive'

    print 'Reading %s ...\n' % fname
    fid, tree, _ = fiff_open(fname)

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)
    info['filename'] = fname

    #   Locate the data of interest
    processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
    if len(processed) == 0:
        fid.close()
        raise ValueError, 'Could not find processed data'

    evoked = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
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

        saspects = dir_tree_find(evoked[k], FIFF.FIFFB_SMSH_ASPECT)
        nsaspects = len(saspects)
        if nsaspects > 0:
            sets[k]['naspect'] += nsaspects
            sets[k]['naspect'] = [sets[k]['naspect'], saspects] # XXX : potential bug
            is_smsh = np.c_[is_smsh, np.ones(1, sets[k]['naspect'])]
            naspect += nsaspects

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
        pos = my_evoked.directory[k].pos
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
            raise ValueError, 'Local channel information was not found ' \
                              'when it was expected.'

        if len(chs) != nchan:
            fid.close()
            raise ValueError, 'Number of channels and number of channel ' \
                              'definitions are different'

        info.chs = chs
        info.nchan = nchan
        print '\tFound channel information in evoked data. nchan = %d' % nchan
        if sfreq > 0:
            info['sfreq'] = sfreq

    nsamp = last - first + 1
    print '\tFound the data of interest:'
    print '\t\tt = %10.2f ... %10.2f ms (%s)' % (
         1000*first/info['sfreq'], 1000*last/info['sfreq'], comment)
    if info['comps'] is not None:
        print '\t\t%d CTF compensation matrices available' % len(info['comps'])

    # Read the data in the aspect block
    nave = 1
    epoch = []
    for k in range(my_aspect.nent):
        kind = my_aspect.directory[k].kind
        pos = my_aspect.directory[k].pos
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

###############################################################################
# Writing

from .tree import copy_tree
from .write import start_file, start_block, end_file, end_block, write_id, \
                   write_float, write_int, write_coord_trans, write_ch_info, \
                   write_dig_point, write_name_list, write_string, \
                   write_float_matrix
from .proj import write_proj
from .ctf import write_ctf_comp


def write_evoked(name, data):
    """Write an evoked dataset to a file

    Parameters
    ----------
    name: string
        The file name.

    data: dict
        The evoked dataset obtained with read_evoked
    """

    #  Create the file and save the essentials
    fid = start_file(name)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if data['info']['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, data['info']['meas_id'])

    #    Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    # Blocks from the original
    blocks = [FIFF.FIFFB_SUBJECT, FIFF.FIFFB_HPI_MEAS, FIFF.FIFFB_HPI_RESULT,
               FIFF.FIFFB_ISOTRAK, FIFF.FIFFB_PROCESSING_HISTORY]

    have_hpi_result = False
    have_isotrak = False
    if len(blocks) > 0 and 'filename' in data['info'] and \
            data['info']['filename'] is not None:
        fid2, tree, _ = fiff_open(data['info']['filename'])
        for block in blocks:
            nodes = dir_tree_find(tree, block)
            copy_tree(fid2, tree['id'], nodes, fid)
            if block == FIFF.FIFFB_HPI_RESULT and len(nodes) > 0:
                have_hpi_result = True
            if block == FIFF.FIFFB_ISOTRAK and len(nodes) > 0:
                have_isotrak = True
        fid2.close()


    #    General
    write_float(fid, FIFF.FIFF_SFREQ, data['info']['sfreq'])
    write_float(fid, FIFF.FIFF_HIGHPASS, data['info']['highpass'])
    write_float(fid, FIFF.FIFF_LOWPASS, data['info']['lowpass'])
    write_int(fid, FIFF.FIFF_NCHAN, data['info']['nchan'])
    if data['info']['meas_date'] is not None:
        write_int(fid, FIFF.FIFF_MEAS_DATE, data['info']['meas_date'])

    #    Coordinate transformations if the HPI result block was not there
    if not have_hpi_result:
        if data['info']['dev_head_t'] is not None:
            write_coord_trans(fid, data['info']['dev_head_t'])

        if data['info']['ctf_head_t'] is not None:
            write_coord_trans(fid, data['info']['ctf_head_t'])

    #  Channel information
    for k in range(data['info']['nchan']):
        #   Scan numbers may have been messed up
        data['info']['chs'][k]['scanno'] = k
        write_ch_info(fid, data['info']['chs'][k])

    #    Polhemus data
    if data['info']['dig'] is not None and not have_isotrak:
        start_block(fid, FIFF.FIFFB_ISOTRAK)
        for d in data['info']['dig']:
            write_dig_point(fid, d)

        end_block(fid, FIFF.FIFFB_ISOTRAK)

    #    Projectors
    write_proj(fid, data['info']['projs'])

    #    CTF compensation info
    write_ctf_comp(fid, data['info']['comps'])

    #    Bad channels
    if len(data['info']['bads']) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, data['info']['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    end_block(fid, FIFF.FIFFB_MEAS_INFO)

    # One or more evoked data sets
    start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    data_evoked = data['evoked']
    if not isinstance(data_evoked, list):
        data_evoked = [data_evoked]

    for evoked in data_evoked:
        start_block(fid, FIFF.FIFFB_EVOKED)

        # Comment is optional
        if len(evoked['comment']) > 0:
            write_string(fid, FIFF.FIFF_COMMENT, evoked['comment'])

        # First and last sample
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, evoked['first'])
        write_int(fid, FIFF.FIFF_LAST_SAMPLE, evoked['last'])

        # The epoch itself
        start_block(fid, FIFF.FIFFB_ASPECT)

        write_int(fid, FIFF.FIFF_ASPECT_KIND, evoked['aspect_kind'])
        write_int(fid, FIFF.FIFF_NAVE, evoked['nave'])

        decal = np.zeros((data['info']['nchan'], data['info']['nchan']))
        for k in range(data['info']['nchan']): # XXX : can be improved
            decal[k, k] = 1.0 / data['info']['chs'][k]['cal']

        write_float_matrix(fid, FIFF.FIFF_EPOCH,
                                np.dot(decal, evoked['epochs']))
        end_block(fid, FIFF.FIFFB_ASPECT)
        end_block(fid, FIFF.FIFFB_EVOKED)

    end_block(fid, FIFF.FIFFB_PROCESSED_DATA)

    end_block(fid, FIFF.FIFFB_MEAS)

    end_file(fid)
