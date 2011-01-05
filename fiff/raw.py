from math import floor, ceil
import numpy as np

from .constants import FIFF
from .open import fiff_open
from .evoked import read_meas_info
from .tree import dir_tree_find
from .tag import read_tag


def setup_read_raw(fname, allow_maxshield=False):
    """Read information about raw data file

    Parameters
    ----------
    fname: string
        The name of the raw file

    allow_maxshield: bool, (default False)
        allow_maxshield if True XXX ???

    Returns
    -------
    data: dict
        Infos about raw data
    """

    #   Open the file
    print 'Opening raw data file %s...' % fname
    fid, tree, _ = fiff_open(fname)

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)

    #   Locate the data of interest
    raw = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
    if raw is None:
        raw = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
        if allow_maxshield:
            raw = dir_tree_find(meas, FIFF.FIFFB_SMSH_RAW_DATA)
            if raw is None:
                raise ValueError, 'No raw data in %s' % fname
        else:
            if raw is None:
                raise ValueError, 'No raw data in %s' % fname

    if len(raw) == 1:
        raw = raw[0]

    #   Set up the output structure
    info['filename'] = fname

    data = dict(fid=fid, info=info, first_samp=0, last_samp=0)

    #   Process the directory
    directory = raw['directory']
    nent = raw['nent']
    nchan = int(info['nchan'])
    first = 0
    first_samp = 0
    first_skip = 0

    #  Get first sample tag if it is there
    if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
        tag = read_tag(fid, directory[first].pos)
        first_samp = int(tag.data)
        first += 1

    #  Omit initial skip
    if directory[first].kind == FIFF.FIFF_DATA_SKIP:
        #  This first skip can be applied only after we know the buffer size
        tag = read_tag(fid, directory[first].pos)
        first_skip = int(tag.data)
        first += 1

    data['first_samp'] = first_samp

    #   Go through the remaining tags in the directory
    rawdir = list()
    nskip = 0
    for k in range(first, nent):
        ent = directory[k]
        if ent.kind == FIFF.FIFF_DATA_SKIP:
            tag = read_tag(fid, ent.pos)
            nskip = int(tag.data)
            print nskip
        elif ent.kind == FIFF.FIFF_DATA_BUFFER:
            #   Figure out the number of samples in this buffer
            if ent.type == FIFF.FIFFT_DAU_PACK16:
                nsamp = ent.size / (2*nchan)
            elif ent.type == FIFF.FIFFT_SHORT:
                nsamp = ent.size / (2*nchan)
            elif ent.type == FIFF.FIFFT_FLOAT:
                nsamp = ent.size / (4*nchan)
            elif ent.type == FIFF.FIFFT_INT:
                nsamp = ent.size / (4*nchan)
            else:
                fid.close()
                raise ValueError, 'Cannot handle data buffers of type %d' % (
                                                                    ent.type)

            #  Do we have an initial skip pending?
            if first_skip > 0:
                first_samp += nsamp*first_skip
                data['first_samp'] = first_samp
                first_skip = 0

            #  Do we have a skip pending?
            if nskip > 0:
                import pdb; pdb.set_trace()
                rawdir.append(dict(ent=None, first=first_samp,
                                   last=first_samp + nskip*nsamp - 1,
                                   nsamp=nskip*nsamp))
                first_samp += nskip*nsamp
                nskip = 0

            #  Add a data buffer
            rawdir.append(dict(ent=ent, first=first_samp,
                               last=first_samp + nsamp -1,
                               nsamp=nsamp))
            first_samp += nsamp

    data['last_samp'] = first_samp - 1

    #   Add the calibration factors
    cals = np.zeros(data['info']['nchan'])
    for k in range(data['info']['nchan']):
        cals[k] = data['info']['chs'][k]['range']*data['info']['chs'][k]['cal']

    data['cals'] = cals
    data['rawdir'] = rawdir
    data['proj'] = None
    data['comp'] = None
    print '\tRange : %d ... %d =  %9.3f ... %9.3f secs' % (
               data['first_samp'], data['last_samp'],
               float(data['first_samp']) / data['info']['sfreq'],
               float(data['last_samp']) / data['info']['sfreq'])
    print 'Ready.\n'

    return data


def read_raw_segment(raw, from_=None, to=None, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: dict
        The dict returned by setup_read_raw

    from_: int
        first sample to include. If omitted, defaults to the first
        sample in data

    to: int
        Last sample to include. If omitted, defaults to the last sample in data

    sel: array, optional
        Indices of channels to select

    node: tree node
        The node of the tree where to look

    Returns
    -------
    data: array, [channels x samples]
       the data matrix (channels x samples)

    times: array, [samples]
        returns the time values corresponding to the samples

    """

    if to is None:
        to = raw['last_samp']
    if from_ is None:
        from_ = raw['first_samp']

    #  Initial checks
    from_ = int(from_)
    to = int(to)
    if from_ < raw['first_samp']:
        from_ = raw['first_samp']

    if to > raw['last_samp']:
        to = raw['last_samp']

    if from_ > to:
        raise ValueError, 'No data in this range'

    print 'Reading %d ... %d  =  %9.3f ... %9.3f secs...' % (
                       from_, to, from_ / float(raw['info']['sfreq']),
                       to / float(raw['info']['sfreq']))

    #  Initialize the data and calibration vector
    nchan = raw['info']['nchan']
    dest = 0
    cal = np.diag(raw['cals'].ravel())

    if sel is None:
        data = np.empty((nchan, to - from_))
        if raw['proj'] is None and raw['comp'] is None:
            mult = None
        else:
            if raw['proj'] is None:
                mult = raw['comp'] * cal
            elif raw['comp'] is None:
                mult = raw['proj'] * cal
            else:
                mult = raw['proj'] * raw['comp'] * cal

    else:
        data = np.empty((len(sel), to - from_))
        if raw['proj'] is None and raw['comp'] is None:
            mult = None
            cal = np.diag(raw['cals'][sel].ravel())
        else:
            if raw['proj'] is None:
                mult = raw['comp'][sel,:] * cal
            elif raw['comp'] is None:
                mult = raw['proj'][sel,:] * cal
            else:
                mult = raw['proj'][sel,:] * raw['comp'] * cal

    do_debug = False
    if cal is not None:
        from scipy import sparse
        cal = sparse.csr_matrix(cal)

    if mult is not None:
        from scipy import sparse
        mult = sparse.csr_matrix(mult)

    for k in range(len(raw['rawdir'])):
        this = raw['rawdir'][k]

        #  Do we need this buffer
        if this['last'] > from_:
            if this['ent'] is None:
                #  Take the easy route: skip is translated to zeros
                if do_debug:
                    print 'S'
                if sel is None:
                    one = np.zeros((nchan, this['nsamp']))
                else:
                    one = np.zeros((len(sel), this['nsamp']))
            else:
                tag = read_tag(raw['fid'], this['ent'].pos)

                #   Depending on the state of the projection and selection
                #   we proceed a little bit differently
                if mult is None:
                    if sel is None:
                        one = cal * tag.data.reshape(this['nsamp'],
                                                     nchan).astype(np.float).T
                    else:
                        one = tag.data.reshape(this['nsamp'],
                                               nchan).astype(np.float).T
                        one = cal * one[sel,:]
                else:
                    one = mult * tag.data.reshape(this['nsamp'],
                                                  nchan).astype(np.float).T

            #  The picking logic is a bit complicated
            if to > this['last'] and from_ <= this['first']:
                #    We need the whole buffer
                first_pick = 0
                last_pick = this['nsamp']
                if do_debug:
                    print 'W'

            elif from_ > this['first']:
                first_pick = from_ - this['first']
                if to < this['last']:
                    #   Something from the middle
                    last_pick = this['nsamp'] + to - this['last']
                    if do_debug:
                        print 'M'
                else:
                    #   From the middle to the end
                    last_pick = this['nsamp'] - 1
                    if do_debug:
                        print 'E'
            else:
                #    From the beginning to the middle
                first_pick = 0
                last_pick = to - this['first']
                if do_debug:
                    print 'B'

            #   Now we are ready to pick
            picksamp = last_pick - first_pick
            if picksamp > 0:
                data[:, dest:dest+picksamp] = one[:, first_pick:last_pick]
                dest += picksamp

       #   Done?
        if this['last'] >= to:
            print ' [done]\n'
            break

    times = np.arange(from_, to) / raw['info']['sfreq']

    return data, times


def read_raw_segment_times(raw, from_, to, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: dict
        The dict returned by setup_read_raw

    from_: float
        Starting time of the segment in seconds

    to: float
        End time of the segment in seconds

    sel: array, optional
        Indices of channels to select

    node: tree node
        The node of the tree where to look

    Returns
    -------
    data: array, [channels x samples]
       the data matrix (channels x samples)

    times: array, [samples]
        returns the time values corresponding to the samples
    """
    #   Convert to samples
    from_ = floor(from_ * raw['info']['sfreq'])
    to = ceil(to * raw['info']['sfreq'])

    #   Read it
    return read_raw_segment(raw, from_, to, sel)

###############################################################################
# Writing

from .write import start_file, start_block, write_id, write_string, \
                   write_ch_info, end_block, write_coord_trans, \
                   write_float, write_int, write_dig_point, \
                   write_name_list, end_file
from .ctf import write_ctf_comp
from .proj import write_proj
from .tree import copy_tree


def start_writing_raw(name, info, sel=None):
    """Start write raw data in file

    Parameters
    ----------
    name : string
        Name of the file to create.

    info : dict
        Measurement info

    sel : array of int, optional
        Indices of channels to include. By default all channels are included.

    Returns
    -------
    fid : file
        The file descriptor

    cals : list
        calibration factors
    """
    #
    #   We will always write floats
    #
    if sel is None:
        sel = np.arange(info['nchan'])
    data_type = 4
    chs = [info['chs'][k] for k in sel]
    nchan = len(chs)
    #
    #  Create the file and save the essentials
    #
    fid = start_file(name)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if info['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info['meas_id'])
    #
    #    Measurement info
    #
    start_block(fid, FIFF.FIFFB_MEAS_INFO)
    #
    #    Blocks from the original
    #
    blocks = [FIFF.FIFFB_SUBJECT, FIFF.FIFFB_HPI_MEAS, FIFF.FIFFB_HPI_RESULT,
              FIFF.FIFFB_ISOTRAK, FIFF.FIFFB_PROCESSING_HISTORY]
    have_hpi_result = False
    have_isotrak = False
    if len(blocks) > 0 and 'filename' in info and info['filename'] is not None:
        fid2, tree, _ = fiff_open(info['filename'])
        for b in blocks:
            nodes = dir_tree_find(tree, b)
            copy_tree(fid2, tree.id, nodes, fid)
            if b == FIFF.FIFFB_HPI_RESULT and len(nodes) > 0:
                have_hpi_result = True
            if b == FIFF.FIFFB_ISOTRAK and len(nodes) > 0:
                have_isotrak = True
        fid2.close()

    #
    #    megacq parameters
    #
    if info['acq_pars'] is not None or info['acq_stim'] is not None:
        start_block(fid, FIFF.FIFFB_DACQ_PARS)
        if info['acq_pars'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_PARS, info['acq_pars'])

        if info['acq_stim'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_STIM, info['acq_stim'])

        end_block(fid, FIFF.FIFFB_DACQ_PARS)
    #
    #    Coordinate transformations if the HPI result block was not there
    #
    if not have_hpi_result:
        if info['dev_head_t'] is not None:
            write_coord_trans(fid, info['dev_head_t'])

        if info['ctf_head_t'] is not None:
            write_coord_trans(fid, info['ctf_head_t'])
    #
    #    Polhemus data
    #
    if info['dig'] is not None and not have_isotrak:
        start_block(fid, FIFF.FIFFB_ISOTRAK)
        for dig_point in info['dig']:
            write_dig_point(fid, dig_point)
            end_block(fid, FIFF.FIFFB_ISOTRAK)
    #
    #    Projectors
    #
    write_proj(fid, info['projs'])
    #
    #    CTF compensation info
    #
    write_ctf_comp(fid, info['comps'])
    #
    #    Bad channels
    #
    if len(info['bads']) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, info['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
    #
    #    General
    #
    write_float(fid, FIFF.FIFF_SFREQ, info['sfreq'])
    write_float(fid, FIFF.FIFF_HIGHPASS, info['highpass'])
    write_float(fid, FIFF.FIFF_LOWPASS, info['lowpass'])
    write_int(fid, FIFF.FIFF_NCHAN, nchan)
    write_int(fid, FIFF.FIFF_DATA_PACK, data_type)
    if info['meas_date'] is not None:
        write_int(fid, FIFF.FIFF_MEAS_DATE, info['meas_date'])
    #
    #    Channel info
    #
    cals = []
    for k in range(nchan):
        #
        #   Scan numbers may have been messed up
        #
        chs[k].scanno = k + 1 # scanno starts at 1 in FIF format
        chs[k].range = 1.0
        cals.append(chs[k]['cal'])
        write_ch_info(fid, chs[k])

    end_block(fid, FIFF.FIFFB_MEAS_INFO)
    #
    # Start the raw data
    #
    start_block(fid, FIFF.FIFFB_RAW_DATA)

    return fid, cals


def write_raw_buffer(fid, buf, cals):
    """Write raw buffer

    Parameters
    ----------
    fid : file descriptor
        an open raw data file

    buf : array
        The buffer to write

    cals : array
        Calibration factors
    """
    if buf.shape[0] != len(cals):
        raise ValueError, 'buffer and calibration sizes do not match'

    write_float(fid, FIFF.FIFF_DATA_BUFFER,
                                    np.dot(np.diag(1.0 / np.ravel(cals)), buf))


def finish_writing_raw(fid):
    """Finish writing raw FIF file

    Parameters
    ----------
    fid : file descriptor
        an open raw data file
    """
    end_block(fid, FIFF.FIFFB_RAW_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)
