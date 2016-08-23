"""Conversion tool from ITAB to FIF
"""

# Author: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

import os
from os import path as op

import numpy as np

from ...utils import verbose, logger
from ...externals.six import string_types

from ..base import _BaseRaw
from ..utils import _mult_cal_one, _blk_read_lims

from .mhd import _read_mhd
#from .info import _compose_meas_info
from info import _mhd2info
from .constants import ITAB

class RawITAB(_BaseRaw):
    """Raw object from ITAB directory

    Parameters
    ----------
    fname : str
        The raw file to load. Filename should end with *.raw
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    
    @verbose
    def __init__(self, fname, preload=False, verbose=True):


#        if preload:
#            self._preload_data(preload)
#        else:
#            self.preload = False
        
        file_name = list()
        file_name.append(fname)
        
        fname_mhd = fname + ".mhd"
        mhd = _read_mhd(fname_mhd)  # Read the mhd file
        info = _mhd2info(mhd)
        info['buffer_size_sec'] = info['n_samp'] / info['sfreq']
        print(info['buffer_size_sec'])

        pass
        if info.get('buffer_size_sec', None) is None:
            raise RuntimeError('Reader error, notify mne-python developers')
        self.info = info
#        self.n_times = info['n_samp']
#        self.times = info['n_samp']
        info._check_consistency()
        
        first_samps = list()
        first_samps.append(0)
        
        last_samps = list()
        last_samps.append(info['n_samp'] - 1)
        
#        self._update_times()
        super(RawITAB, self).__init__(
            info, preload, last_samps=last_samps, filenames=file_name,
            verbose=verbose)
 

    @verbose
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data"""
        
        #  Initial checks
        start = int(start)
        if stop is None:
            stop = self.info['n_samp']
#        else:
#            min([int(stop), self.n_times])

        if start >= stop:
            raise ValueError('No data in this range')


        offset = 0
        with open(self._filenames[fi], 'rb') as fid:

        #position  file pointer
            data_offset = self.info['start_data']
            fid.seek(data_offset + start * self.info['nchan'], 0)            

        # read data                
            n_read = self.info['n_chan']*self.info['n_samp']
            this_data = np.fromfile(fid, '>i4', count=n_read)
            this_data.shape = (self.info['n_samp'], self.info['nchan'])
           
            data_view = data[:, 0:self.info['n_samp']]
          
        # calibrate data                                
            _mult_cal_one(data_view, this_data.transpose(), idx, cals, mult)
            
            pass
  

def read_raw_itab(fname, preload=False, verbose=None):
    """Raw object from ITAB directory

    Parameters
    ----------
    fname : str
        The raw file to load. Filename should end with *.raw
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of RawITAB
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.01
    """
    
    a = RawITAB(fname, preload=preload, verbose=verbose)
    pass
    return a
              
#                
#    @verbose
#    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
#        """Read a chunk of raw data"""
#        nchan = self._raw_extras[fi]['nchan']
#        data_left = (stop - start) * nchan
#        # amplifier applies only to the sensor channels
#        n_sens = self._raw_extras[fi]['n_sens']
#        sensor_gain = self._raw_extras[fi]['sensor_gain'].copy()
#        sensor_gain[:n_sens] = (sensor_gain[:n_sens] /
#                                self._raw_extras[fi]['amp_gain'])
#        conv_factor = np.array((KIT.VOLTAGE_RANGE /
#                                self._raw_extras[fi]['DYNAMIC_RANGE']) *
#                               sensor_gain)
#        n_bytes = 2
#        # Read up to 100 MB of data at a time.
#        blk_size = min(data_left, (100000000 // n_bytes // nchan) * nchan)
#        with open(self._filenames[fi], 'rb', buffering=0) as fid:
#            # extract data
#            data_offset = KIT.RAW_OFFSET
#            fid.seek(data_offset)
#            # data offset info
#            data_offset = unpack('i', fid.read(KIT.INT))[0]
#            pointer = start * nchan * KIT.SHORT
#            fid.seek(data_offset + pointer)
#            for blk_start in np.arange(0, data_left, blk_size) // nchan:
#                blk_size = min(blk_size, data_left - blk_start * nchan)
#                block = np.fromfile(fid, dtype='h', count=blk_size)
#                block = block.reshape(nchan, -1, order='F').astype(float)
#                blk_stop = blk_start + block.shape[1]
#                data_view = data[:, blk_start:blk_stop]
#                block *= conv_factor[:, np.newaxis]
#
#                # Create a synthetic channel
#                if self._raw_extras[fi]['stim'] is not None:
#                    trig_chs = block[self._raw_extras[fi]['stim'], :]
#                    if self._raw_extras[fi]['slope'] == '+':
#                        trig_chs = trig_chs > self._raw_extras[0]['stimthresh']
#                    elif self._raw_extras[fi]['slope'] == '-':
#                        trig_chs = trig_chs < self._raw_extras[0]['stimthresh']
#                    else:
#                        raise ValueError("slope needs to be '+' or '-'")
#                    # trigger value
#                    if self._raw_extras[0]['stim_code'] == 'binary':
#                        ntrigchan = len(self._raw_extras[0]['stim'])
#                        trig_vals = np.array(2 ** np.arange(ntrigchan),
#                                             ndmin=2).T
#                    else:
#                        trig_vals = np.reshape(self._raw_extras[0]['stim'],
#                                               (-1, 1))
#                    trig_chs = trig_chs * trig_vals
#                    stim_ch = np.array(trig_chs.sum(axis=0), ndmin=2)
#                    block = np.vstack((block, stim_ch))
#                _mult_cal_one(data_view, block, idx, None, mult)
#        # cals are all unity, so can be ignored
#
#    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
#        """Read a segment of data from a file"""
#        stop -= 1
#        offset = 0
#        with _fiff_get_fid(self._filenames[fi]) as fid:
#            for this in self._raw_extras[fi]:
#                #  Do we need this buffer
#                if this['last'] >= start:
#                    #  The picking logic is a bit complicated
#                    if stop > this['last'] and start < this['first']:
#
#
#        data : ndarray, shape (len(idx), stop - start + 1)
#            The data array. Should be modified inplace.
#        idx : ndarray | slice
#            The requested channel indices.
#        fi : int
#            The file index that must be read from.
#        start : int
#            The start sample in the given file.
#        stop : int
#            The stop sample in the given file (inclusive).
#        cals : ndarray, shape (len(idx), 1)
#            Channel calibrations (already sub-indexed).
#        mult : ndarray, shape (len(idx), len(info['chs']) | None
#            The compensation + projection + cals matrix, if applicable.
#    def _read_raw_file(self, info, fname, preload, do_check_fname=True,
#                       verbose=None):
#        """Read in header information from a raw file"""
#        logger.info('Opening raw data file %s...' % fname)
#
##        if do_check_fname:
##            check_fname(fname, 'raw', ('raw.raw'))
#
#        #   Read in the whole file if preload is on
#        whole_file = preload if '.raw' in ext else False
#        ff, tree, _ = fiff_open(fname, preload=whole_file)
#
#        with ff as fid:
#            
#        #   Locate the data of interest
#
#            if len(raw_node) == 1:
#                raw_node = raw_node[0]
#
#            #   Set up the output structure
#            info['filename'] = fname
#
#            #   Process the directory
#            directory = raw_node['directory']
#            nent = raw_node['nent']
#            nchan = int(info['nchan'])
#            first = 0
#            first_samp = 0
#            first_skip = 0
#
#            #   Get first sample tag if it is there
#            if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
#                tag = read_tag(fid, directory[first].pos)
#                first_samp = int(tag.data)
#                first += 1
#                _check_entry(first, nent)
#
#            #   Omit initial skip
#            if directory[first].kind == FIFF.FIFF_DATA_SKIP:
#                # This first skip can be applied only after we know the bufsize
#                tag = read_tag(fid, directory[first].pos)
#                first_skip = int(tag.data)
#                first += 1
#                _check_entry(first, nent)
#
#            raw = _RawShell()
#            raw.filename = fname
#            raw.first_samp = first_samp
#
#            #   Go through the remaining tags in the directory
#            raw_extras = list()
#            nskip = 0
#            orig_format = None
#
#            for k in range(first, nent):
#                ent = directory[k]
#                if ent.kind == FIFF.FIFF_DATA_SKIP:
#                    tag = read_tag(fid, ent.pos)
#                    nskip = int(tag.data)
#                elif ent.kind == FIFF.FIFF_DATA_BUFFER:
#                    #   Figure out the number of samples in this buffer
#                    if ent.type == FIFF.FIFFT_DAU_PACK16:
#                        nsamp = ent.size // (2 * nchan)
#                    elif ent.type == FIFF.FIFFT_SHORT:
#                        nsamp = ent.size // (2 * nchan)
#                    elif ent.type == FIFF.FIFFT_FLOAT:
#                        nsamp = ent.size // (4 * nchan)
#                    elif ent.type == FIFF.FIFFT_DOUBLE:
#                        nsamp = ent.size // (8 * nchan)
#                    elif ent.type == FIFF.FIFFT_INT:
#                        nsamp = ent.size // (4 * nchan)
#                    elif ent.type == FIFF.FIFFT_COMPLEX_FLOAT:
#                        nsamp = ent.size // (8 * nchan)
#                    elif ent.type == FIFF.FIFFT_COMPLEX_DOUBLE:
#                        nsamp = ent.size // (16 * nchan)
#                    else:
#                        raise ValueError('Cannot handle data buffers of type '
#                                         '%d' % ent.type)
#                    if orig_format is None:
#                        if ent.type == FIFF.FIFFT_DAU_PACK16:
#                            orig_format = 'short'
#                        elif ent.type == FIFF.FIFFT_SHORT:
#                            orig_format = 'short'
#                        elif ent.type == FIFF.FIFFT_FLOAT:
#                            orig_format = 'single'
#                        elif ent.type == FIFF.FIFFT_DOUBLE:
#                            orig_format = 'double'
#                        elif ent.type == FIFF.FIFFT_INT:
#                            orig_format = 'int'
#                        elif ent.type == FIFF.FIFFT_COMPLEX_FLOAT:
#                            orig_format = 'single'
#                        elif ent.type == FIFF.FIFFT_COMPLEX_DOUBLE:
#                            orig_format = 'double'
#
#                    #  Do we have an initial skip pending?
#                    if first_skip > 0:
#                        first_samp += nsamp * first_skip
#                        raw.first_samp = first_samp
#                        first_skip = 0
#
#                    #  Do we have a skip pending?
#                    if nskip > 0:
#                        raw_extras.append(dict(
#                            ent=None, first=first_samp, nsamp=nskip * nsamp,
#                            last=first_samp + nskip * nsamp - 1))
#                        first_samp += nskip * nsamp
#                        nskip = 0
#
#                    #  Add a data buffer
#                    raw_extras.append(dict(ent=ent, first=first_samp,
#                                           last=first_samp + nsamp - 1,
#                                           nsamp=nsamp))
#                    first_samp += nsamp
#
#            next_fname = _get_next_fname(fid, fname, tree)
#
#        raw.last_samp = first_samp - 1
#        raw.orig_format = orig_format
#
#        #   Add the calibration factors
#        cals = np.zeros(info['nchan'])
#        for k in range(info['nchan']):
#            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']
#
#        raw._cals = cals
#        raw._raw_extras = raw_extras
#        raw.comp = None
#        raw._orig_comp_grade = None
#
#
#        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
#                    raw.first_samp, raw.last_samp,
#                    float(raw.first_samp) / info['sfreq'],
#                    float(raw.last_samp) / info['sfreq']))
#
#        # store the original buffer size
#        info['buffer_size_sec'] = (np.median([r['nsamp']
#                                              for r in raw_extras]) /
#                                   info['sfreq'])
#
#        raw.info = info
#        raw.verbose = verbose
#
#        logger.info('Ready.')
#
#        return raw, next_fname

#    @verbose
#    def _read_rawdata(self, data, idx, fi, start, stop, cals, mult):
#        """Read a chunk of raw data"""
#        si = self._raw_extras[fi]
#        offset = 0
#        trial_start_idx, r_lims, d_lims = _blk_read_lims(start, stop,
#                                                         int(si['block_size']))
#        with open(self._filenames[fi], 'rb') as fid:
#            logger.info('Opening raw data file %s...' % self._filenames[fi])
#            for bi in range(len(r_lims)):
#                samp_offset = (bi + trial_start_idx) * si['res4_nsamp']
#                n_read = info['n_chan']*info['n_samp']
#                # read the chunk of data
#                pos = CTF.HEADER_SIZE
#                pos += samp_offset * si['n_chan'] * 4
#                fid.seek(pos, 0)
#                this_data = np.fromfile(fid, '<i4', count=n_read)
#                                        
#                return np.fromfile(fid, '<i4', 1)[0]
#                this_data.shape = (si['n_chan'], n_read)
#                this_data = this_data[:, r_lims[bi, 0]:r_lims[bi, 1]]
#                data_view = data[:, d_lims[bi, 0]:d_lims[bi, 1]]
#                _mult_cal_one(data_view, this_data, idx, cals, mult)
#                offset += n_read