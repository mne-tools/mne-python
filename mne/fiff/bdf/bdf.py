"""Conversion tool from EDF/+,BDF to FIF

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
from os import SEEK_CUR
from struct import unpack
import datetime
import re

import numpy as np
from scipy.linalg import norm

from ...fiff import pick_types
from ...utils import verbose, logger
from ..raw import Raw
from ..constants import FIFF
from ..kit import coreg


class RawBDF(Raw):
    """Raw object from KIT bdf file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        Path to the bdf file.
    input_fname : str
        Path to the hpts file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, hpts, preload=False, verbose=None):
        logger.info('Extracting bdf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self._bdf_params = params = get_bdf_params(input_fname, hpts)
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._preloaded = False
        self.fids = list()
        self._projector = None
        self.first_samp = 0
        self.last_samp = self._bdf_params['nsamples'] - 1
        self.comp = None  # no compensation for KIT
        self.proj = False

        # Create raw.info dict for raw fif object with bdf data
        self.info = {}
        self.info['meas_id'] = None
        self.info['file_id'] = None
        self.info['meas_date'] = params['date']
        self.info['projs'] = []
        self.info['comps'] = []
        self.info['lowpass'] = None
        self.info['highpass'] = None
        self.info['sfreq'] = float(params['sfreq'])
        self.info['nchan'] = params['nchan']
        self.info['bads'] = []
        self.info['acq_pars'], self.info['acq_stim'] = None, None
        self.info['filename'] = None
        self.info['ctf_head_t'] = None
        self.info['dev_ctf_t'] = []
        self.info['filenames'] = []
        self.info['dig'] = None
        self.info['dev_head_t'] = None

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        self.info['ch_names'] = ch_names = params['ch_names']
        chan_locs = coreg.transform_pts(params['sensor_locs'])
        self.info['chs'] = []
        for idx, ch_info in enumerate(zip(ch_names, chan_locs), 1):
            ch_name, ch_loc = ch_info
            chan_info = {}
            chan_info['cal'] = 1.
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = 1
            chan_info['unit_mul'] = 0
            chan_info['ch_name'] = ch_name
            chan_info['unit'] = FIFF.FIFF_UNIT_V
            chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
            chan_info['kind'] = FIFF.FIFFV_EEG_CH
            chan_info['loc'] = ch_loc
            self.info['chs'].append(chan_info)
            if ch_name == 'Status':
                chan_info = {}
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
                chan_info['loc'] = np.zeros(12)
                chan_info['unit'] = FIFF.FIFF_UNIT_NONE
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            elif ch_name.startswith('EX'):
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            self.info['chs'].append(chan_info)

        if preload:
            self._preloaded = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Add time info
            self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
            self._times = np.arange(self.first_samp, self.last_samp + 1,
                                    dtype=np.float64)
            self._times /= self.info['sfreq']
            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           float(self.first_samp) / self.info['sfreq'],
                           float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

    def __repr__(self):
        s = ('%r' % os.path.basename(self._bdf_params['fname']),
             "n_channels x n_times : %s x %s" % (len(self.info['ch_names']),
                                       self.last_samp - self.first_samp + 1))
        return "<RawEDF  |  %s>" % ', '.join(s)

    def read_stim_ch(self, buffer_size=1e5):
        """Read events from data

        Parameter
        ---------
        buffer_size : int
            The size of chunk to by which the data are scanned.

        Returns
        -------
        events : array, [samples]
           The event vector (1 x samples).
        """
        buffer_size = int(buffer_size)
        start = int(self.first_samp)
        stop = int(self.last_samp + 1)

        pick = pick_types(self.info, meg=False, stim=True, exclude=[])
        stim_ch = np.empty((1, stop), dtype=np.int)
        for b_start in range(start, stop, buffer_size):
            b_stop = b_start + buffer_size
            x, _ = self._read_segment(start=b_start, stop=b_stop, sel=pick)
            stim_ch[:, b_start:b_start + x.shape[1]] = x

        return stim_ch

    def _read_segment(self, start=0, stop=None, sel=None, verbose=None,
                      projector=None):
        """Read a chunk of raw data

        Parameters
        ----------
        start : int, (optional)
            first sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int, (optional)
            First sample to not include.
            If omitted, data is included to the end.
        sel : array, optional
            Indices of channels to select.
        projector : array
            SSP operator to apply to the data.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).
        times : array, [samples]
            returns the time values corresponding to the samples.
        """
        if sel is None:
            sel = range(self.info['nchan'])
        elif len(sel) == 1 and sel[0] == 0 and start == 0 and stop == 1:
            return (666, 666)
        if projector is not None:
            raise NotImplementedError('Currently does not handle projections.')
        if stop is None:
            stop = self.last_samp + 1
        elif stop > self.last_samp + 1:
            stop = self.last_samp + 1

        #  Initial checks
        start = int(start)
        stop = int(stop)

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(self.info['sfreq']),
                               (stop - 1) / float(self.info['sfreq'])))
        params = self._bdf_params
        sfreq = params['sfreq']
        data_size = params['data_size']
        data_offset = params['data_offset']

        with open(params['fname'], 'rb') as fid:
            # extract data
            fid.seek(data_offset)
            nchan = params['nchan']
            buffer_size = (stop - start)
            pointer = start * nchan
            fid.seek(data_offset + pointer)
            chan_block = buffer_size / sfreq

            datas = []
            for _ in range(chan_block):
                data = np.empty((nchan, sfreq), dtype=np.int32)
                for chan in range(nchan):
                    chan_data = fid.read(sfreq * data_size)
                    chan_data
                    if type(chan_data) == str:
                        chan_data = np.fromstring(chan_data, np.uint8)
                    else:
                        chan_data = np.asarray(chan_data, np.uint8)
                    chan_data = chan_data.reshape(-1, 3).astype(np.int32)
                    chan_data = (chan_data[:, 0] + (chan_data[:, 1] << 8) +
                                 (chan_data[:, 2] << 16))
                    chan_data[chan_data >= (1 << 23)] -= (1 << 24)
                    data[chan, :] = chan_data
                datas.append(data)
            data = np.hstack(datas)
            data = params['gains'] * data
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop) / self.info['sfreq']

        return data, times


def get_bdf_params(fname, hpts):
    """Extracts all the information from the bdf file.

    Parameters
    ----------
    rawfile : str
        Raw bdf file to be read.

    Returns
    -------
    bdf : dict
        A dict containing all the bdf parameter settings.
    """

    bdf = dict()
    bdf['fname'] = fname
    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        assert(fid.read(8) == '\xffBIOSEMI')

        bdf['subject_id'] = fid.read(80).strip()
        bdf['recording_id'] = fid.read(80).strip()
        (day, month, year) = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        (hour, minute, sec) = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        bdf['date'] = str(datetime.datetime(year + 2000, month, day,
                                            hour, minute, sec))
        bdf['data_offset'] = header_nbytes = int(fid.read(8))
        format = fid.read(44).strip()
        assert format == '24BIT'
        n_records = int(fid.read(8))
        record_length = int(fid.read(8))  # in seconds
        bdf['nchan'] = int(fid.read(4))

        channels = range(bdf['nchan'])
        bdf['ch_names'] = [fid.read(16).strip() for n in channels]
        bdf['transducer_type'] = [fid.read(80).strip() for n in channels]
        bdf['units'] = [fid.read(8).strip() for n in channels]
        units = list(np.unique(bdf['units']))
        if 'Boolean' in units:
            units.remove('Boolean')
        assert len(units) == 1
        if units[0] == 'uV':
            scale = 1e-6
        elif units[0] == 'V':
            scale = 1
        bdf['physical_min'] = np.array([int(fid.read(8)) for n in channels])
        bdf['physical_max'] = np.array([int(fid.read(8)) for n in channels])
        bdf['digital_min'] = np.array([int(fid.read(8)) for n in channels])
        bdf['digital_max'] = np.array([int(fid.read(8)) for n in channels])
        bdf['prefiltering'] = [fid.read(80).strip() for n in channels]
        n_samples_per_record = [int(fid.read(8)) for n in channels]
        assert len(np.unique(n_samples_per_record)) == 1

        n_samples_per_record = n_samples_per_record[0]
        fid.read(32 * bdf['nchan'])  # reserved
        assert fid.tell() == header_nbytes
        physical_diff = bdf['physical_max'] - bdf['physical_min']
        physical_diff = np.array(physical_diff, float)
        digital_diff = bdf['digital_max'] - bdf['digital_min']
        bdf['gains'] = np.array([physical_diff / digital_diff]).T
        bdf['gains'] *= scale
        bdf['sfreq'] = n_samples_per_record / record_length
        bdf['nsamples'] = n_records * n_samples_per_record
        bdf['data_size'] = 3  # 24-bit (3 byte) integers

        locs = open(hpts, 'rb').readlines()
        locs = [x.split() for x in locs]
        locs = {x[1]: tuple(x[2:]) for x in locs}
        locs = [locs[ch_name] if ch_name in locs.keys() else (0, 0, 0)
                              for ch_name in bdf['ch_names']]
        bdf['sensor_locs'] = np.array(locs, int)

    return bdf


def read_raw_bdf(input_fname, mrk=None, elp=None, hsp=None, stim='>',
                 slope='-', stimthresh=1, preload=False, verbose=None):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the bdf file.
    mrk : None | str | array_like, shape = (5, 3)
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
    elp : None | str | array_like, shape = (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10`000 points are in the head shape, they are automatically decimated.
    stim : list of int | '<' | '>'
        Channel-value correspondence when converting KIT trigger channels to a
        Neuromag-style stim channel. For '<', the largest values are assigned
        to the first channel (default). For '>', the largest values are
        assigned to the last channel. Can also be specified as a list of
        trigger channel indexes.
    slope : '+' | '-'
        How to interpret values on KIT trigger channels when synthesizing a
        Neuromag-style stim channel. With '+', a positive slope (low-to-high)
        is interpreted as an event. With '-', a negative slope (high-to-low)
        is interpreted as an event.
    stimthresh : float
        The threshold level for accepting voltage changes in KIT trigger
        channels as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  stim=stim, slope=slope, stimthresh=stimthresh,
                  verbose=verbose, preload=preload)
