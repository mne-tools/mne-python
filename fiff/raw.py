from math import floor, ceil
import numpy as np

from .constants import FIFF
from .open import fiff_open
from .evoked import read_meas_info
from .tree import dir_tree_find
from .tag import read_tag


class Raw(dict):
    """Raw data set"""
    def __getitem__(self, item):
        """getting raw data content with python slicing"""
        if isinstance(item, tuple): # slicing required
            if len(item) == 2: # channels and time instants
                time_slice = item[1]
                sel = item[0]
            else:
                time_slice = item[0]
                sel = None
            start, stop, step = time_slice.start, time_slice.stop, time_slice.step
            if step is not None:
                raise ValueError('step needs to be 1 : %d given' % step)
            return read_raw_segment(self, start=start, stop=stop, sel=sel)
        else:
            return super(Raw, self).__getitem__(item)

    def time_to_index(self, *args):
        indices = []
        for time in args:
            ind = int(time * self['info']['sfreq'])
            indices.append(ind)
        return indices

    def close(self):
        self['fid'].close()


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

    data = Raw(fid=fid, info=info, first_samp=0, last_samp=0)

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
                first_samp += nsamp * first_skip
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
                               last=first_samp + nsamp - 1,
                               nsamp=nsamp))
            first_samp += nsamp

    data['last_samp'] = first_samp - 1

    #   Add the calibration factors
    cals = np.zeros(data['info']['nchan'])
    for k in range(data['info']['nchan']):
        cals[k] = data['info']['chs'][k]['range'] * \
                  data['info']['chs'][k]['cal']

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


def read_raw_segment(raw, start=None, stop=None, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: dict
        The dict returned by setup_read_raw

    start: int, (optional)
        first sample to include (first is 0). If omitted, defaults to the first
        sample in data

    stop: int, (optional)
        First sample to not include.
        If omitted, data is included to the end.

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

    if stop is None:
        stop = raw['last_samp'] + 1
    if start is None:
        start = raw['first_samp']

    #  Initial checks
    start = int(start)
    stop = int(stop)
    if start < raw['first_samp']:
        start = raw['first_samp']

    if stop >= raw['last_samp']:
        stop = raw['last_samp'] + 1

    if start >= stop:
        raise ValueError, 'No data in this range'

    print 'Reading %d ... %d  =  %9.3f ... %9.3f secs...' % (
                       start, stop - 1, start / float(raw['info']['sfreq']),
                       (stop - 1) / float(raw['info']['sfreq'])),

    #  Initialize the data and calibration vector
    nchan = raw['info']['nchan']
    dest = 0
    cal = np.diag(raw['cals'].ravel())

    if sel is None:
        data = np.empty((nchan, stop - start))
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
        data = np.empty((len(sel), stop - start))
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
    # do_debug = True
    if cal is not None:
        from scipy import sparse
        cal = sparse.csr_matrix(cal)

    if mult is not None:
        from scipy import sparse
        mult = sparse.csr_matrix(mult)

    for k in range(len(raw['rawdir'])):
        this = raw['rawdir'][k]

        #  Do we need this buffer
        if this['last'] >= start:
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
            if stop - 1 > this['last'] and start < this['first']:
                #    We need the whole buffer
                first_pick = 0
                last_pick = this['nsamp']
                if do_debug:
                    print 'W'

            elif start >= this['first']:
                first_pick = start - this['first']
                if stop - 1 <= this['last']:
                    #   Something from the middle
                    last_pick = this['nsamp'] + stop - this['last'] - 1
                    if do_debug:
                        print 'M'
                else:
                    #   From the middle to the end
                    last_pick = this['nsamp']
                    if do_debug:
                        print 'E'
            else:
                #    From the beginning to the middle
                first_pick = 0
                last_pick = stop - this['first']
                if do_debug:
                    print 'B'

            #   Now we are ready to pick
            picksamp = last_pick - first_pick
            if picksamp > 0:
                data[:, dest:dest+picksamp] = one[:, first_pick:last_pick]
                dest += picksamp

        #   Done?
        if this['last'] >= stop-1:
            print ' [done]'
            break

    times = np.arange(start, stop) / raw['info']['sfreq']

    raw['fid'].seek(0, 0) # Go back to beginning of the file

    return data, times


def read_raw_segment_times(raw, start, stop, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: dict
        The dict returned by setup_read_raw

    start: float
        Starting time of the segment in seconds

    stop: float
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
    start = floor(start * raw['info']['sfreq'])
    stop = ceil(stop * raw['info']['sfreq'])

    #   Read it
    return read_raw_segment(raw, start, stop, sel)

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

###############################################################################
# misc

def findall(L, value, start=0):
    """Returns indices of all occurence of value in list L starting from start
    """
    c = L.count(value)
    if c == 0:
        return list()
    else:
        ind = list()
        i = start-1
        for _ in range(c):
            i = L.index(value, i+1)
            ind.append(i)
        return ind


def _make_compensator(info, kind):
    """Auxiliary function for make_compensator
    """
    for k in range(len(info['comps'])):
        if info['comps'][k]['kind'] == kind:
            this_data = info['comps'][k]['data'];

            #   Create the preselector
            presel = np.zeros((this_data['ncol'], info['nchan']))
            for col, col_name in enumerate(this_data['col_names']):
                ind = findall(info['ch_names'], col_name)
                if len(ind) == 0:
                    raise ValueError, 'Channel %s is not available in data' % \
                                                                      col_name
                elif len(ind) > 1:
                    raise ValueError, 'Ambiguous channel %s' % col_name
                presel[col, ind] = 1.0

            #   Create the postselector
            postsel = np.zeros((info['nchan'], this_data['nrow']))
            for c, ch_name in enumerate(info['ch_names']):
                ind = findall(this_data['row_names'], ch_name)
                if len(ind) > 1:
                    raise ValueError, 'Ambiguous channel %s' % ch_name
                elif len(ind) == 1:
                    postsel[c, ind] = 1.0

            this_comp = postsel*this_data['data']*presel;
            return this_comp

    return []


def make_compensator(info, from_, to, exclude_comp_chs=False):
    """
    %
    % [comp] = mne_make_compensator(info,from,to,exclude_comp_chs)
    %
    % info              - measurement info as returned by the fif reading routines
    % from              - compensation in the input data
    % to                - desired compensation in the output
    % exclude_comp_chs  - exclude compensation channels from the output (optional)
    %

    %
    % Create a compensation matrix to bring the data from one compensation
    % state to another
    %
    """

    if from_ == to:
        comp = np.zeros((info['nchan'], info['nchan']))
        return comp

    if from_ == 0:
        C1 = np.zeros((info['nchan'], info['nchan']))
    else:
        try:
            C1 = _make_compensator(info, from_)
        except Exception as inst:
            raise ValueError, 'Cannot create compensator C1 (%s)' % inst

        if len(C1) == 0:
            raise ValueError, 'Desired compensation matrix (kind = %d) not found' % from_


    if to == 0:
       C2 = np.zeros((info['nchan'], info['nchan']))
    else:
        try:
            C2 = _make_compensator(info, to)
        except Exception as inst:
            raise ValueError, 'Cannot create compensator C2 (%s)' % inst

        if len(C2) == 0:
            raise ValueError, 'Desired compensation matrix (kind = %d) not found' % to


    #   s_orig = s_from + C1*s_from = (I + C1)*s_from
    #   s_to   = s_orig - C2*s_orig = (I - C2)*s_orig
    #   s_to   = (I - C2)*(I + C1)*s_from = (I + C1 - C2 - C2*C1)*s_from
    comp = np.eye(info['nchan']) + C1 - C2 - C2*C1;

    if exclude_comp_chs:
        pick = np.zeros((info['nchan'], info['nchan']))
        npick = 0
        for k, chan in info['chs']:
            if chan['kind'] != FIFF.FIFFV_REF_MEG_CH:
                npick += 1
                pick[npick] = k

        if npick == 0:
            raise ValueError, 'Nothing remains after excluding the compensation channels'

        comp = comp[pick[1:npick], :]

    return comp
