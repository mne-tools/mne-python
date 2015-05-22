"""Conversion tool from GDF

"""


import os
import calendar
import datetime
import re
import warnings
from math import ceil, floor

import numpy as np

from ...utils import verbose, logger
from ..base import _BaseRaw, _check_update_montage
from ..meas_info import _empty_info
from ..pick import pick_types
from ..constants import FIFF
from ...externals.six.moves import zip

__GDFTYP_BYTE = (1, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8, 0, 0, 0, 0, 0, 4, 8, 16)
__GDFTYP_NAME = [None]
__GDFTYP_NAME.append(np.int8)
__GDFTYP_NAME.append(np.uint8)
__GDFTYP_NAME.append(np.int16)
__GDFTYP_NAME.append(np.uint16)
__GDFTYP_NAME.append(np.int32)
__GDFTYP_NAME.append(np.uint32)
__GDFTYP_NAME.append(np.int64)
__GDFTYP_NAME.append(np.uint64)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(None)
__GDFTYP_NAME.append(np.float32)
__GDFTYP_NAME.append(np.float64)

def __gdf_time2py_time(t):
    print t
    """ Convert gdf time to python datetime"""
    if t == '                ':
		date = datetime.datetime(2000,1,1)
    else:
        if t[14:16] == '  ':
		    t = t[:14] + '00' + t[16:]
        
        date =  (datetime.datetime(int(t[0:4]),int(t[4:6]),int(t[6:8]),int(t[8:10]),int(t[10:12]),int(t[12:14]),int(t[14:16])*pow(10,4)))
    
    return date


class RawGDF(_BaseRaw):
    """Raw object from GDF file

    Parameters
    ----------
    input_fname : str
        Path to the GDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel (default).
        If None, there will be no stim channel added.
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
    def __init__(self, input_fname, montage, eog=None, misc=None,
                 stim_channel=-1,
                 preload=False, verbose=None):
        logger.info('Extracting gdf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        info, self._gdf_info = _get_gdf_info(input_fname, stim_channel,
                                             eog, misc, preload)
        logger.info('Creating Raw.info structure...')
        _check_update_montage(info, montage)
        
        # Raw attributes
        last_samps = [self._gdf_info['nsamples'] - 1]
        super(RawGDF, self).__init__(
            info, preload, last_samps=last_samps, orig_format='int',
            verbose=verbose)

        logger.info('Ready.')

    def __repr__(self):
        n_chan = self.info['nchan']
        data_range = self.last_samp - self.first_samp + 1
        s = ('%r' % os.path.basename(self.info['filename']),
             "n_channels x n_times : %s x %s" % (n_chan, data_range))
        return "<RawGDF  |  %s>" % ', '.join(s)

    @verbose
    def _read_segment(self, start=0, stop=None, sel=None, data_buffer=None,
                      projector=None, verbose=None):
        """Read a chunk of raw data"""
        from scipy.interpolate import interp1d
        if sel is None:
            sel = np.arange(self.info['nchan'])
        if projector is not None:
            raise NotImplementedError('Currently does not handle projections.')
        if stop is None:
            stop = self.last_samp + 1
        elif stop > self.last_samp + 1:
            stop = self.last_samp + 1
        sel = np.array(sel)

        #  Initial checks
        start = int(start)
        stop = int(stop)

        n_samps = self._gdf_info['n_samps']
        buf_len = self._gdf_info['max_samp']
        sfreq = self.info['sfreq']
        n_chan = self.info['nchan']
        data_size = self._gdf_info['bpb']
        data_offset = self._gdf_info['data_offset']
        stim_channel = self._gdf_info['stim_channel']

        # this is used to deal with indexing in the middle of a sampling period
        blockstart = int(floor(float(start) / buf_len) * buf_len)
        blockstop = int(ceil(float(stop) / buf_len) * buf_len)
        if blockstop > self.last_samp:
            blockstop = self.last_samp + 1

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(sfreq),
                     (stop - 1) / float(sfreq)))

        # gain constructor
        physical_range = np.array([ch['range'] for ch in self.info['chs']])
        cal = np.array([ch['cal'] for ch in self.info['chs']])
        cal = (physical_range)/(cal)
        gains = np.atleast_2d(self._gdf_info['units'] * (cal))
        # physical dimension in uV
        physical_min = self._gdf_info['physical_min'] 
        digital_min = self._gdf_info['digital_min']
        offsets = np.atleast_2d(physical_min-cal*digital_min).T
                
        picks = [stim_channel]
        offsets[picks] = 0
        
        # set up output array
        data_shape = (len(sel), stop - start)
        if isinstance(data_buffer, np.ndarray):
            if data_buffer.shape != data_shape:
                raise ValueError('data_buffer has incorrect shape')
            data = data_buffer
        else:
            data = np.empty(data_shape, dtype=float)

        read_size = blockstop - blockstart
        this_data = np.empty((len(sel), buf_len))

        # does not support multiple data type 
        if len(np.unique(self._gdf_info['gdf_type_np']))>1:
            raise("Multiple data type not supported")

        with open(self.info['filename'], 'rb', buffering=0) as fid:
            # extract data
            fid.seek(data_offset + blockstart * data_size)
            #n_blk = int(ceil(float(read_size) / buf_len))
            #start_offset = start - blockstart
            #end_offset = blockstop - stop
            gdftype = self._gdf_info['gdf_type_np']
            data = np.fromfile(fid,dtype=gdftype[0],count=read_size*n_chan)
            data = data.reshape((read_size,n_chan)).T
            data = np.float64(data)
                
        data *= gains.T[sel]
        data += offsets[sel]

        if stim_channel is not None:
            stim_channel_idx = np.where(sel == stim_channel)[0][0]

            # Allows support for up to 16-bit trigger values (2 ** 16 - 1)
            stim = np.bitwise_and(np.round(data[stim_channel_idx]).astype(int),
                                      65535)
            data[stim_channel_idx, :] = \
                stim[start - blockstart:stop - blockstart]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / self.info['sfreq']
        return data, times
            

def _get_gdf_info(fname, stim_channel,eog, misc, preload):
    """Extracts all the information from the GDF file"""

    if eog is None:
        eog = []
    if misc is None:
        misc = []
    info = _empty_info()
    info['filename'] = fname

    gdf_info = dict()
    gdf_info['events'] = []

    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        version = fid.read(8)
        
        gdf_info['type'] = version[:3]
        gdf_info['number'] = float(version[4:])
        
        if gdf_info['number'] > 1.9 :
            raise('GDF version > 1.9 not implemented')

        fid.read(80)  # subject id
        fid.read(80)  # recording id
        
        #date 
        tm = fid.read(16)
        
        date = __gdf_time2py_time(tm)
        
        info['meas_date'] = calendar.timegm(date.utctimetuple())
        
        gdf_info['data_offset'] = header_nbytes = np.fromstring(fid.read(8), np.int64)[0]
        gdf_info['Equipment']  = np.fromstring(fid.read(8), np.uint8)[0]
        gdf_info['Hospital']   = np.fromstring(fid.read(8), np.uint8)[0]
        gdf_info['Technician'] = np.fromstring(fid.read(8), np.uint8)[0]
        fid.seek(20,1)    #20bytes reserved

        gdf_info['n_records'] = n_records = np.fromstring(fid.read(8), np.int64)[0]
        # record length in seconds
        record_length = np.fromstring(fid.read(8), np.uint32)
        if record_length[0] == 0:
            record_length[0] = 1.
            warnings.warn('Header information is incorrect for record length. '
                          'Default record length set to 1.')
        
        gdf_info['record_length'] = record_length
            
        info['nchan'] =gdf_info['nchan'] = nchan = np.fromstring(fid.read(4), np.uint32)[0]
        channels = list(range(info['nchan']))
        
        
        
        ch_names = [fid.read(16).strip().decode() for ch in channels]
        transducer = [fid.read(80).strip().decode() for ch in channels]  # transducer
        gdf_info['units'] = units = [fid.read(8).strip() for ch in channels]

        for i, unit in enumerate(units):
            if unit[:2] == 'uV':
                units[i] = 1e-6
            else:
                units[i] = 1
        
        physical_min = np.array([np.fromstring(fid.read(8), np.float64)[0]
                                 for ch in channels])
        gdf_info['physical_min'] = physical_min
        physical_max = np.array([np.fromstring(fid.read(8), np.float64)[0]
                                 for ch in channels])
        digital_min = np.array([np.fromstring(fid.read(8), np.int64)[0]
                                for ch in channels])
        gdf_info['digital_min'] = digital_min
        digital_max = np.array([np.fromstring(fid.read(8), np.int64)[0]
                                for ch in channels])
        prefiltering = [fid.read(80).strip().decode() for ch in channels][:-1]
        highpass = np.ravel([re.findall('HP:\s+(\w+)', filt)
                             for filt in prefiltering])
        lowpass = np.ravel([re.findall('LP:\s+(\w+)', filt)
                            for filt in prefiltering])

        high_pass_default = 0.
        if highpass.size == 0:
            info['highpass'] = high_pass_default
        elif all(highpass):
            if highpass[0] == 'NaN':
                info['highpass'] = high_pass_default
            elif highpass[0] == 'DC':
                info['highpass'] = 0.
            else:
                info['highpass'] = float(highpass[0])
        else:
            info['highpass'] = float(np.min(highpass))
            warnings.warn('Channels contain different highpass filters. '
                          'Highest filter setting will be stored.')

        if lowpass.size == 0:
            info['lowpass'] = None
        elif all(lowpass):
            if lowpass[0] == 'NaN':
                info['lowpass'] = None
            else:
                info['lowpass'] = float(lowpass[0])
        else:
            info['lowpass'] = float(np.min(lowpass))
            warnings.warn('%s' % ('Channels contain different lowpass filters.'
                                  ' Lowest filter setting will be stored.'))
        # number of samples per record
        gdf_info['n_samps'] = n_samps = np.array([np.fromstring(fid.read(4), np.int32)[0]
                                for ch in channels])
                            
        gdf_info['gdf_type'] = gdftype = np.array([np.fromstring(fid.read(4), np.int32)[0]
                                for ch in channels])        
        gdf_info['gdf_type_np'] = [__GDFTYP_NAME[t] for t in gdftype]
        fid.read(32 * info['nchan']).decode()  # reserved
        assert fid.tell() == header_nbytes
        
        #total number of byter for data
        gdf_info['bpb'] = np.sum([__GDFTYP_BYTE[t]*n_samps[i] for i,t in enumerate(gdf_info['gdf_type'])])
    
        #EVENT TABLE
        etp = header_nbytes + n_records*gdf_info['bpb']
        #skip data to go to event table
        fid.seek(etp)
        etmode = fid.read(1)
        if etmode != '':
            etmode = np.fromstring(etmode, np.uint8).tolist()[0]
            sr = np.fromstring(fid.read(3), np.uint8)
            EventSampleRate = sr[0]
            events = []
            for i in range(1,len(sr)):
                EventSampleRate = EventSampleRate + sr[i]*256**i
            N = np.fromstring(fid.read(4), np.uint32).tolist()[0]
            POS = np.fromstring(fid.read(N*4), np.uint32)
            TYP = np.fromstring(fid.read(N*2), np.uint16)

            if etmode == 3:
                CHN = np.fromstring(fid.read(N*2), np.uint16)
                DUR = np.fromstring(fid.read(N*4), np.uint32)
                events.append([N,POS,TYP,CHN,DUR])   
            else:
                DUR = np.zeros(N,dtype=np.uint32)
                events.append([N,POS,TYP])
            
            gdf_info['events'] = events
            info['events'] = np.c_[POS,DUR,TYP]

    cals = digital_max-digital_min 
    
    physical_ranges = physical_max - physical_min
    #cals = digital_max - digital_min

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    info['buffer_size_sec'] = 10.

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    info['ch_names'] = ch_names
   
    if stim_channel == -1:
        stim_channel = info['nchan'] - 1
    for idx, ch_info in enumerate(zip(ch_names, physical_ranges, cals)):
        ch_name, physical_range, cal = ch_info
        chan_info = {}
        chan_info['cal'] = cal
        chan_info['logno'] = idx + 1
        chan_info['scanno'] = idx + 1
        chan_info['range'] = physical_range
        chan_info['unit_mul'] = 0.
        chan_info['ch_name'] = ch_name
        chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        chan_info['eeg_loc'] = np.zeros(3)
        chan_info['loc'] = np.zeros(12)
        if ch_name in eog or idx in eog or idx - nchan in eog:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_EOG_CH
        if ch_name in misc or idx in misc or idx - nchan in misc:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
        check1 = stim_channel == ch_name
        check2 = stim_channel == idx
        check3 = info['nchan'] > 1
        stim_check = np.logical_and(np.logical_or(check1, check2), check3)
        if stim_check:
            chan_info['range'] = physical_range
            chan_info['cal'] = cal
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chan_info['ch_name'] = 'STI 014'
            info['ch_names'][idx] = chan_info['ch_name']
            units[idx] = 1
            if isinstance(stim_channel, str):
                stim_channel = idx
        info['chs'].append(chan_info)
    gdf_info['stim_channel'] = stim_channel

    # sfreq defined as the max sampling rate of eeg
    picks = pick_types(info, meg=False, eeg=True)
    if len(picks) == 0:
        gdf_info['max_samp'] = max_samp = n_samps.max()
    else:
        gdf_info['max_samp'] = max_samp = n_samps[picks].max()
    info['sfreq'] = max_samp * float(record_length[1]) / record_length[0]
    gdf_info['nsamples'] = int(n_records * max_samp)

    if info['lowpass'] is None:
        info['lowpass'] = info['sfreq'] / 2.

    return info, gdf_info

def read_raw_gdf(input_fname, montage=None, eog=None, misc=None,
                 stim_channel=-1,preload=False, verbose=None):
    """Reader function for EDF+, BDF conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0).
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes in the
        edf file. Default is None.
    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        -1 corresponds to the last channel (default).
        If None, there will be no stim channel added.
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
    raw : Instance of RawGDF
        A Raw object containing GDF data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawGDF(input_fname=input_fname, montage=montage, eog=eog, misc=misc,
                  stim_channel=stim_channel, preload=preload, verbose=verbose)
