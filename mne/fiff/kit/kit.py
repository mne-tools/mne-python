'''
Created on Dec 27, 2012

@author: teon
sqd_params class is adapted from Yoshiaki Adachi's Meginfo2.cpp
and sqdread's getdata.m
RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py
'''

from mne.fiff.raw import Raw
from mne import verbose
from mne.fiff.constants import FIFF
from . constants import KIT
from . import coreg
from struct import unpack
import numpy as np
import time
import logging
import re




class sqd_params(object):
    """
    Extracts all the information from the sqd file.

    Parameters
    ----------
    rawfile: str
        raw sqd file to be read
    lowpass: None | int
        value of lowpass filter set in MEG160
    highpass: None | int
        value of highpass filter set in MEG160

        """

    def __init__(self, rawfile, lowpass=None, highpass=None):
        self.dynamic_range = 2 ** 12 / 2  # signed integer. range +/- 2048
        self.voltage_range = 5.
        self.rawfile = rawfile
        fid = open(rawfile, 'r')
        fid.seek(KIT.BASIC_INFO)
        basic_offset = unpack('i', fid.read(KIT.INT))[0]  # integer are 4 bytes
        fid.seek(basic_offset)

        # basic info
        self.version = unpack('i', fid.read(KIT.INT))[0]
        self.revision = unpack('i', fid.read(KIT.INT))[0]
        self.sysid = unpack('i', fid.read(KIT.INT))[0]
        self.sysname = unpack('128s', fid.read(KIT.STRING))[0].split('\n')[0]
        self.modelname = unpack('128s', fid.read(KIT.STRING))[0].split('\n')[0]
        self.nchan = unpack('i', fid.read(KIT.INT))[0]
        self.chan_no = range(self.nchan)
        self.lowpass = lowpass
        self.highpass = highpass

        # amplifier gain
        fid.seek(KIT.AMPLIFIER_INFO)
        amp_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(amp_offset)
        amp_data = unpack('i', fid.read(KIT.INT))[0]
        input_gain_bit = 11  # stored in Bit-11 to 12
        input_gain_mask = 6144  # (0x1800)
        # input_gain: 0:x1, 1:x2, 2:x5, 3:x10
        input_gains = [1, 2, 5, 10]
        self.input_gain = input_gains[(input_gain_mask & amp_data)
                                      >> input_gain_bit]
        # 0:x1, 1:x2, 2:x5, 3:x10, 4:x20, 5:x50, 6:x100, 7:x200
        output_gains = [1, 2, 5, 10, 20, 50, 100, 200]
        output_gain_bit = 0  # stored in Bit-0 to 2
        output_gain_mask = 7  # (0x0007)
        self.output_gain = output_gains[(output_gain_mask & amp_data)
                                        >> output_gain_bit]

        # only channels 0-159 requires gain. the additional channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected

        fid.seek(KIT.CHAN_SENS)
        sens_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=self.nchan * 2)
        self.n_sens = KIT.nmegchan + KIT.nrefchan
        self._sensitivities = (np.reshape(sens, (self.nchan, 2))
                               [:self.n_sens, 0])
        self.sensor_gain = np.ones(self.nchan)
        self.sensor_gain[:self.n_sens] = self._sensitivities

        fid.seek(KIT.SAMPLE_INFO)
        acqcond_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(KIT.INT))[0]
        if acq_type == 1:
            self.sfreq = unpack('d', fid.read(KIT.DOUBLE))[0]
            _ = fid.read(KIT.INT)  # initialized estimate of samples
            self.nsamples = unpack('i', fid.read(KIT.INT))[0]
        else:
            raise NotImplementedError

        fid.seek(KIT.DATA_OFFSET)
        # data offset info
        self.data_offset = unpack('i', fid.read(4))[0]

    def get_data(self):
        """returns an array with data extracted from sqd file"""

        fid = open(self.rawfile, 'r')
        fid.seek(self.data_offset)
        data = np.fromfile(fid, dtype='h', count=self.nsamples * self.nchan)
        data = np.reshape(data, (self.nsamples, self.nchan))
        # amplifier applies only to the sensor channels 0-159
        amp_gain = self.output_gain * self.input_gain
        self.sensor_gain[:self.n_sens] = (self.sensor_gain[:self.n_sens] /
                                          amp_gain)
        conv_factor = np.array((self.voltage_range / self.dynamic_range) *
                               self.sensor_gain, ndmin=2)
        self.x = (conv_factor * data).T

logger = logging.getLogger('mne')


class RawKIT(Raw):
    """
    Raw object from KIT SQD file
    Adapted from mne_bti2fiff.py

    Parameters
    ----------
    data_fname : str
        absolute path to the sqd file.
    mrk_fname : str
        absolute path to marker coils file.
    elp_fname : str
        absolute path to elp digitizer laser points file.
    hsp_fname : str
        absolute path to elp digitizer head shape points file.
    data : bool | array-like
        if array-like custom data matching the header info to be used
        instead of the data from data_fname
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes & Methods
    --------------------
    See documentation for mne.fiff.Raw

        """
    @verbose
    def __init__(self, data_fname, mrk_fname, elp_fname, hsp_fname,
                 sns_fname, data=None, lowpass=None, highpass=None,
                 stim=xrange(160, 168), direction='d',
                 stimthresh, add_chs, verbose=True):

        logger.info('Extracting SQD Parameters from %s...' % data_fname)
        self.params = sqd_params(data_fname, lowpass=lowpass,
                                 highpass=highpass)
        self._data_file = data_fname
        self.mrk_fname = mrk_fname
        self.elp_fname = elp_fname
        self.hsp_fname = hsp_fname
        self.sns_fname = sns_fname
        self.stim = stim
        self.stimthresh = stimthresh
        if direction is 'a':
            self.endian = 1
        elif direction is 'd':
            self.endian = -1
        else:
            raise NotImplementedError

        logger.info('Reading raw data from %s...' % data_fname)
        if not data:
            self.params.get_data()
            self._data = self.params.x
        else:
            self._data = data
        logger.info('Creating Raw.info structure...')
        info = self._create_raw_info()
        self.verbose = verbose
        self._preloaded = True
        self.info = info
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        assert len(self._data) == self.params.nchan
        self._times = np.arange(self.first_samp, \
                                self.last_samp + 1) / info['sfreq']
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                   self.first_samp, self.last_samp,
                   float(self.first_samp) / info['sfreq'],
                   float(self.last_samp) / info['sfreq']))
        # remove subclass helper attributes to create a proper Raw object.
        for attr in self.__dict__:
            if attr not in Raw.__dict__:
                del attr
        logger.info('Ready.')

    @verbose
    def _create_raw_info(self):
        """ Fills list of dicts for initializing empty fif with SQD data"""

        info = {}
        info['meas_id'] = None
        info['file_id'] = None
        info['meas_date'] = time.ctime()
        info['projs'] = []
        info['comps'] = []
        info['lowpass'] = self.params.lowpass
        info['highpass'] = self.params.highpass
        info['sfreq'] = float(self.params.sfreq)
        info['chs'] = self._create_chs_dict()
        info['nchan'] = self.params.nchan
        info['ch_names'] = (self.params.ch_names['MEG']
                            + self.params.ch_names['STIM'])
        info['bads'] = []
        info['acq_pars'], info['acq_stim'] = None, None
        info['filename'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = []
        info['filenames'] = []
        # dev-head transformation dict
        info['dev_head_t'] = {}
        info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD
        # transformation matrix and digitizer points
        # there is some transformation done to the digitization data that
        # I have not figured out yet.
        info['dev_head_t']['trans'], info['dig'] = self._get_coreg()
        logger.info('Done.')
        return info

    def _create_chs_dict(self):
        """creates a list of dicts of meg channels for raw.info"""

        logger.info('Setting channel info structure...')
        self.ch_names = {}
        self.ch_names['MEG'] = ['MEG %03d' % ch for ch
                                in range(1, KIT.nmegchan + 1)]
        self.ch_names['REF'] = ['REF %03d' % ch for ch
                                in range(1, KIT.nrefchan + 1)]
        self.ch_names['TRIG'] = ['TRIG %03d' % ch for ch
                                 in range(1, KIT.ntrigchan + 1)]
        self.ch_names['MISC'] = ['MISC %03d' % ch for ch
                                 in range(1, KIT.nmiscchan + 1)]
        self.ch_names['STIM'] = ['STIM 101']
        p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),([\.\-0-9]+),([\.\-0-9]+)' +
                       r'([\.\-0-9]+),([\.\-0-9]+)')
        locs = np.array(p.findall(open(self.sns_fname).read()), dtype='float')
        # the arrangement of dimensions in current fif is y,x,z in [m].
        # this is to orient in Neuromag coordinates.
        self.chan_locs = locs[:, [1, 0, 2]] / 1000
        self.chan_locs[:, 0] *= -1
        chs = []
        for idx, ch_info in enumerate(zip(self.ch_names['MEG'] +
                                          self.ch_names['REF'],
                                          self.chan_locs), 1):
            ch_name, ch_loc = ch_info
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['eeg_loc'] = None
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = 1.0
            chan_info['unit_mul'] = 0  # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            if ch_name.startswith('MEG'):
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T  # 112 = T
                # this has the coordinates,
                # but the three unit vectors (mne p.273)
                chan_info['loc'] = ch_loc
                # this needs to be sorted out
                chan_info['coil_trans'] = None
            elif ch_name == 'STI 014':
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chs.append(chan_info)

        """create a synthetic channel"""
        trig_chs = self._data[self.stim, :]
        trig_chs = trig_chs > self.stimthresh
        trig_vals = np.array([2 ** (n - 1) for n in
                              xrange(len(self.stim), 0, self.endian)])
        trig_chs = trig_chs * trig_vals
        stim_ch = trig_chs.sum(axis=1)

        # deals with spurious triggering
        idx = np.where(stim_ch > 0)[0]
        idy = idx + KIT.TRIGGER_LENGTH
        diff = stim_ch[idy] - stim_ch[idx]
        index = np.where(diff != 0)
        stim_ch[index] = 0

        self._data = np.vstack((self._data, stim_ch))
        return chs


    def _get_coreg(self):
        """get transformation matrix and hsp points"""

        coreg_data = coreg(mrk_fname=self.mrk_fname, elp_fname=self.elp_fname,
                           hsp_fname=self.hsp_fname)
        dev_head_t = coreg_data.fit()
        return dev_head_t, coreg_data.hsp_points



