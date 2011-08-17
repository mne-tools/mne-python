# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from math import floor, ceil
import copy
import numpy as np

from .constants import FIFF
from .open import fiff_open
from .meas_info import read_meas_info, write_meas_info
from .tree import dir_tree_find
from .tag import read_tag


class Raw(dict):
    """Raw data set

    Parameters
    ----------
    fname: string
        The name of the raw file

    allow_maxshield: bool, (default False)
        allow_maxshield if True XXX ???

    info: dict
        Infos about raw data

    """

    def __init__(self, fname, allow_maxshield=False):
        """
        Parameters
        ----------
        fname: string
            The name of the raw file

        allow_maxshield: bool, (default False)
            allow_maxshield if True XXX ???

        """

        #   Open the file
        print 'Opening raw data file %s...' % fname
        fid, tree, _ = fiff_open(fname)

        #   Read the measurement info
        info, meas = read_meas_info(fid, tree)

        #   Locate the data of interest
        raw_node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
        if len(raw_node) == 0:
            raw_node = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
            if allow_maxshield:
                raw_node = dir_tree_find(meas, FIFF.FIFFB_SMSH_RAW_DATA)
                if len(raw_node) == 0:
                    raise ValueError('No raw data in %s' % fname)
            else:
                if len(raw_node) == 0:
                    raise ValueError('No raw data in %s' % fname)

        if len(raw_node) == 1:
            raw_node = raw_node[0]

        #   Set up the output structure
        info['filename'] = fname

        #   Process the directory
        directory = raw_node['directory']
        nent = raw_node['nent']
        nchan = int(info['nchan'])
        first = 0
        first_samp = 0
        first_skip = 0

        #   Get first sample tag if it is there
        if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, directory[first].pos)
            first_samp = int(tag.data)
            first += 1

        #   Omit initial skip
        if directory[first].kind == FIFF.FIFF_DATA_SKIP:
            # This first skip can be applied only after we know the buffer size
            tag = read_tag(fid, directory[first].pos)
            first_skip = int(tag.data)
            first += 1

        self.first_samp = first_samp

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
                    nsamp = ent.size / (2 * nchan)
                elif ent.type == FIFF.FIFFT_SHORT:
                    nsamp = ent.size / (2 * nchan)
                elif ent.type == FIFF.FIFFT_FLOAT:
                    nsamp = ent.size / (4 * nchan)
                elif ent.type == FIFF.FIFFT_INT:
                    nsamp = ent.size / (4 * nchan)
                else:
                    fid.close()
                    raise ValueError('Cannot handle data buffers of type %d' %
                                                                      ent.type)

                #  Do we have an initial skip pending?
                if first_skip > 0:
                    first_samp += nsamp * first_skip
                    self.first_samp = first_samp
                    first_skip = 0

                #  Do we have a skip pending?
                if nskip > 0:
                    import pdb; pdb.set_trace()
                    rawdir.append(dict(ent=None, first=first_samp,
                                       last=first_samp + nskip * nsamp - 1,
                                       nsamp=nskip * nsamp))
                    first_samp += nskip * nsamp
                    nskip = 0

                #  Add a data buffer
                rawdir.append(dict(ent=ent, first=first_samp,
                                   last=first_samp + nsamp - 1,
                                   nsamp=nsamp))
                first_samp += nsamp

        self.last_samp = first_samp - 1

        #   Add the calibration factors
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * \
                      info['chs'][k]['cal']

        self.cals = cals
        self.rawdir = rawdir
        self.proj = None
        self.comp = None
        print '\tRange : %d ... %d =  %9.3f ... %9.3f secs' % (
                   self.first_samp, self.last_samp,
                   float(self.first_samp) / info['sfreq'],
                   float(self.last_samp) / info['sfreq'])
        print 'Ready.'

        self.fid = fid
        self.info = info

    def __getitem__(self, item):
        """getting raw data content with python slicing"""
        if isinstance(item, tuple):  # slicing required
            if len(item) == 2:  # channels and time instants
                time_slice = item[1]
                if isinstance(item[0], slice):
                    start = item[0].start if item[0].start is not None else 0
                    nchan = self.info['nchan']
                    stop = item[0].stop if item[0].stop is not None else nchan
                    step = item[0].step if item[0].step is not None else 1
                    sel = range(start, stop, step)
                else:
                    sel = item[0]
            else:
                time_slice = item[0]
                sel = None
            start, stop, step = time_slice.start, time_slice.stop, \
                                time_slice.step
            if start is None:
                start = 0
            if step is not None:
                raise ValueError('step needs to be 1 : %d given' % step)

            if isinstance(sel, int):
                sel = np.array([sel])

            if sel is not None and len(sel) == 0:
                raise Exception("Empty channel list")

            return read_raw_segment(self, start=start, stop=stop, sel=sel)
        else:
            return super(Raw, self).__getitem__(item)

    def save(self, fname, picks=None, tmin=0, tmax=None, buffer_size_sec=10,
             drop_small_buffer=False):
        """Save raw data to file

        Parameters
        ----------
        fname : string
            File name of the new dataset.

        picks : list of int
            Indices of channels to include

        tmin : float
            Time in seconds of first sample to save

        tmax : int
            Time in seconds of last sample to save

        buffer_size_sec : float
            Size of data chuncks in seconds.

        drop_small_buffer: bool
            Drop or not the last buffer. It is required by maxfilter (SSS)
            that only accepts raw files with buffers of the same size.

        """
        outfid, cals = start_writing_raw(fname, self.info, picks)
        #
        #   Set up the reading parameters
        #

        #   Convert to samples
        start = int(floor(tmin * self.info['sfreq']))

        if tmax is None:
            stop = self.last_samp + 1 - self.first_samp
        else:
            stop = int(floor(tmax * self.info['sfreq']))

        buffer_size = int(ceil(buffer_size_sec * self.info['sfreq']))
        #
        #   Read and write all the data
        #
        write_int(outfid, FIFF.FIFF_FIRST_SAMPLE, start)
        for first in range(start, stop, buffer_size):
            last = first + buffer_size
            if last >= stop:
                last = stop + 1

            if picks is None:
                data, times = self[:, first:last]
            else:
                data, times = self[picks, first:last]

            if (drop_small_buffer and (first > start)
                                            and (len(times) < buffer_size)):
                print 'Skipping data chunk due to small buffer ... [done]\n'
                break

            print 'Writing ... ',
            write_raw_buffer(outfid, data, cals)
            print '[done]'

        finish_writing_raw(outfid)

    def time_to_index(self, *args):
        indices = []
        for time in args:
            ind = int(time * self.info['sfreq'])
            indices.append(ind)
        return indices

    def close(self):
        self.fid.close()

    def __repr__(self):
        s = "n_channels x n_times : %s x %s" % (len(self.info['ch_names']),
                                       self.last_samp - self.first_samp + 1)
        return "Raw (%s)" % s

    @property
    def ch_names(self):
        return self.info['ch_names']


def read_raw_segment(raw, start=0, stop=None, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: Raw object
        An instance of Raw

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
        stop = raw.last_samp + 1

    #  Initial checks
    start = int(start + raw.first_samp)
    stop = int(stop + raw.first_samp)

    if stop >= raw.last_samp:
        stop = raw.last_samp + 1

    if start >= stop:
        raise ValueError('No data in this range')

    print 'Reading %d ... %d  =  %9.3f ... %9.3f secs...' % (
                       start, stop - 1, start / float(raw.info['sfreq']),
                       (stop - 1) / float(raw.info['sfreq'])),

    #  Initialize the data and calibration vector
    nchan = raw.info['nchan']
    dest = 0
    cal = np.diag(raw.cals.ravel())

    if sel is None:
        data = np.empty((nchan, stop - start))
        if raw.proj is None and raw.comp is None:
            mult = None
        else:
            if raw.proj is None:
                mult = raw.comp * cal
            elif raw.comp is None:
                mult = raw.proj * cal
            else:
                mult = raw.proj * raw.comp * cal

    else:
        data = np.empty((len(sel), stop - start))
        if raw.proj is None and raw.comp is None:
            mult = None
            cal = np.diag(raw.cals[sel].ravel())
        else:
            if raw.proj is None:
                mult = raw.comp[sel, :] * cal
            elif raw.comp is None:
                mult = raw.proj[sel, :] * cal
            else:
                mult = raw.proj[sel, :] * raw.comp * cal

    do_debug = False
    # do_debug = True
    if cal is not None:
        from scipy import sparse
        cal = sparse.csr_matrix(cal)

    if mult is not None:
        from scipy import sparse
        mult = sparse.csr_matrix(mult)

    for this in raw.rawdir:

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
                tag = read_tag(raw.fid, this['ent'].pos)

                #   Depending on the state of the projection and selection
                #   we proceed a little bit differently
                if mult is None:
                    if sel is None:
                        one = cal * tag.data.reshape(this['nsamp'],
                                                     nchan).astype(np.float).T
                    else:
                        one = tag.data.reshape(this['nsamp'],
                                               nchan).astype(np.float).T
                        one = cal * one[sel, :]
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
                data[:, dest:(dest + picksamp)] = one[:, first_pick:last_pick]
                dest += picksamp

        #   Done?
        if this['last'] >= stop - 1:
            print ' [done]'
            break

    times = (np.arange(start, stop) - raw.first_samp) / raw.info['sfreq']

    raw.fid.seek(0, 0)  # Go back to beginning of the file

    return data, times


def read_raw_segment_times(raw, start, stop, sel=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw: Raw object
        An instance of Raw

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
    start = floor(start * raw.info['sfreq'])
    stop = ceil(stop * raw.info['sfreq'])

    #   Read it
    return read_raw_segment(raw, start, stop, sel)

###############################################################################
# Writing

from .write import start_file, end_file, start_block, end_block, \
                   write_float, write_int, write_id


def start_writing_raw(name, info, sel=None):
    """Start write raw data in file

    Data will be written in float

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
    if sel is not None:
        info = copy.deepcopy(info)
        info['chs'] = [info['chs'][k] for k in sel]
        info['nchan'] = len(sel)

        ch_names = [c['ch_name'] for c in info['chs']]  # name of good channels
        comps = copy.deepcopy(info['comps'])
        for c in comps:
            row_idx = [k for k, n in enumerate(c['data']['row_names'])
                                                            if n in ch_names]
            row_names = [c['data']['row_names'][i] for i in row_idx]
            rowcals = c['rowcals'][row_idx]
            c['rowcals'] = rowcals
            c['data']['nrow'] = len(row_names)
            c['data']['row_names'] = row_names
            c['data']['data'] = c['data']['data'][row_idx]
        info['comps'] = comps

    cals = []
    for k in range(info['nchan']):
        #
        #   Scan numbers may have been messed up
        #
        info['chs'][k]['scanno'] = k + 1  # scanno starts at 1 in FIF format
        info['chs'][k]['range'] = 1.0
        cals.append(info['chs'][k]['cal'])

    write_meas_info(fid, info, data_type=4)

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
        raise ValueError('buffer and calibration sizes do not match')

    write_float(fid, FIFF.FIFF_DATA_BUFFER, buf / np.ravel(cals)[:, None])


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
