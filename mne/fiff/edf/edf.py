"""Conversion tool from EDF+,BDF to FIF

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
import datetime
import re
import numpy as np

from ...fiff import pick_types
from ...utils import verbose, logger
from ..raw import Raw
from ..constants import FIFF
from ..kit import coreg


class RawEDF(Raw):
    """Raw object from EDF+,BDF file

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.

    hpts : str | None
        Path to the hpts file.
        If None, sensor locations are (0,0,0).

    annot : str | None
        Path of the annot file.
        Can be None for BDF only.
        If None for EDF, it will raise an error.

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
    def __init__(self, input_fname, hpts, annot, preload=False, verbose=None):
        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self._bdf_params = params = get_edf_params(input_fname, hpts)
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

        # Create raw.info dict for raw fif object with edf data
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
            chan_info['eeg_loc'] = ch_loc
            chan_info['loc'] = np.zeros(12)
            chan_info['loc'][:3] = ch_loc
            if ch_name == 'Status':
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
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

    def _read_segment(self, start=0, stop=None, sel=None, verbose=None,
                      proj=None):
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

        proj : array
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
        if proj is not None:
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

            if params['subtype'] == '24BIT':
                datas = []
                for _ in range(chan_block):
                    data = np.empty((nchan, sfreq), dtype=np.int32)
                    for chan in range(nchan):
                        chan_data = fid.read(sfreq * data_size)
                        chan_data
                        if isinstance(chan_data, str):
                            chan_data = np.fromstring(chan_data, np.uint8)
                        else:
                            chan_data = np.asarray(chan_data, np.uint8)
                        chan_data = chan_data.reshape(-1, 3)
                        chan_data = chan_data.astype(np.int32)
                        # this converts to 24-bit little endian integer
                        # # no support in numpy
                        chan_data = (chan_data[:, 0] +
                                     (chan_data[:, 1] << 8) +
                                     (chan_data[:, 2] << 16))
                        chan_data[chan_data >= (1 << 23)] -= (1 << 24)
                        data[chan, :] = chan_data
                    datas.append(data)
                data = np.hstack(datas)
                data = params['gains'] * data
            else:
                data = np.fromfile(fid, dtype='<i2', count=buffer_size)
                data = data.reshape((buffer_size, nchan)).T
                data = ((data - params['digital_min']) * params['gains'] +
                        params['physical_min'])
                stim_channel = self._read_annot(params['annot'])
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop) / self.info['sfreq']

        return data, times

    def _read_annot(self, annot):
        """Reads an annotation file and converts it to a stimulus channel
        """

        stim_channel = unicode(annot, 'utf-8').split('\x14') if annot else []

        return stim_channel


def get_edf_params(fname, hpts=None, annot=None):
    """Extracts all the information from the EDF+,BDF file.

    Parameters
    ----------
    rawfile : str
        Raw EDF+,BDF file to be read.

    Returns
    -------
    edf : dict
        A dict containing all the EDF+,BDF parameter settings.
    """

    edf = dict()
    edf['fname'] = fname
    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        assert(fid.read(8) == '\xffBIOSEMI')

        edf['subject_id'] = fid.read(80).strip()
        edf['recording_id'] = fid.read(80).strip()
        day, month, year = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        hour, minute, sec = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        edf['date'] = str(datetime.datetime(year + 2000, month, day,
                                            hour, minute, sec))
        edf['data_offset'] = header_nbytes = int(fid.read(8))
        subtype = fid.read(44).strip()[:5]
        assert subtype in ['EDF+C', 'EDF+D', '24BIT']
        edf['subtype'] = subtype

        n_records = int(fid.read(8))
        record_length = int(fid.read(8))  # in seconds
        edf['nchan'] = int(fid.read(4))

        channels = range(edf['nchan'])
        edf['ch_names'] = [fid.read(16).strip() for n in channels]
        edf['transducer_type'] = [fid.read(80).strip() for n in channels]
        edf['units'] = [fid.read(8).strip() for n in channels]
        not_stim_ch = np.where(np.array(edf['units']) != 'Boolean')[0]
        units = list(np.unique(edf['units']))
        if 'Boolean' in units:
            units.remove('Boolean')
        assert len(units) == 1
        if units[0] == 'uV':
            scale = 1e-6
        elif units[0] == 'V':
            scale = 1
        edf['physical_min'] = np.array([float(fid.read(8)) for n in channels])
        edf['physical_max'] = np.array([float(fid.read(8)) for n in channels])
        edf['digital_min'] = np.array([float(fid.read(8)) for n in channels])
        edf['digital_max'] = np.array([float(fid.read(8)) for n in channels])
        edf['prefiltering'] = [fid.read(80).strip() for n in channels]
        n_samples_per_record = [int(fid.read(8)) for n in channels]
        assert len(np.unique(n_samples_per_record)) == 1

        n_samples_per_record = n_samples_per_record[0]
        fid.read(32 * edf['nchan'])  # reserved
        assert fid.tell() == header_nbytes
    physical_range = edf['physical_max'] - edf['physical_min']
    digital_range = edf['digital_max'] - edf['digital_min']
    edf['gains'] = np.array([physical_range / digital_range]).T
    edf['gains'][not_stim_ch] *= scale
    edf['sfreq'] = n_samples_per_record / record_length
    edf['nsamples'] = n_records * n_samples_per_record
    if edf['subtype'] == '24BIT':
        edf['data_size'] = 3  # 24-bit (3 byte) integers
    else:
        edf['data_size'] = 2  # 16-bit (2 byte) integers
    if os.path.lexists(hpts):
        locs_temp = open(hpts, 'rb').readlines()
        locs_temp = [x.split() for x in locs]
        locs = {}
        for loc in locs_temp:
            locs[loc[1]] = tuple(loc[2:])
    else:
        locs = {}
    locs = [locs[ch_name] if ch_name in locs.keys() else (0, 0, 0)
                          for ch_name in edf['ch_names']]
    edf['sensor_locs'] = np.array(locs, int)

    if edf['subtype'] != '24BIT':
        if os.path.lexists(annot):
            raise ValueError('Missing required annotation file.')

    return edf


def read_raw_edf(input_fname, hpts=None, annot=None, preload=False,
                 verbose=None):
    """Reader function for EDF+, BDF conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.

    hpts : str | None
        Path to the hpts file.
        If None, sensor locations are (0,0,0).

    annot : str | None
        Path of the annot file.
        Can be None for BDF only.
        If None for EDF, it will raise an error.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    """
    return RawEDF(input_fname=input_fname, hpts=hpts, annot=annot,
                  verbose=verbose, preload=preload)
