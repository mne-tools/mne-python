"""Conversion tool from SQD to FIF

sqd_params class is adapted from Yoshiaki Adachi's Meginfo2.cpp
RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from mne.fiff.raw import Raw
from mne import verbose
from mne.transforms.coreg import fit_matched_pts
from mne.fiff.constants import FIFF
from . constants import KIT
from . import coreg
from struct import unpack
import numpy as np
import time
import logging
import re


class sqd_params(object):
    """Extracts all the information from the sqd file.

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

        self.input_gain = KIT.input_gains[(KIT.input_gain_mask & amp_data)
                                          >> KIT.input_gain_bit]
        self.output_gain = KIT.output_gains[(KIT.output_gain_mask & amp_data)
                                            >> KIT.output_gain_bit]

        # only channels 0-159 requires gain. the additional channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected

        fid.seek(KIT.CHAN_SENS)
        sens_offset = unpack('i', fid.read(KIT.INT))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=self.nchan * 2)
        KIT.n_sens = KIT.nmegchan + KIT.nrefchan
        self._sensitivities = (np.reshape(sens, (self.nchan, 2))
                               [:KIT.n_sens, 0])
        self.sensor_gain = np.ones(self.nchan)
        self.sensor_gain[:KIT.n_sens] = self._sensitivities

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
        self.data_offset = unpack('i', fid.read(KIT.INT))[0]

    def get_data(self):
        """returns an array with data extracted from sqd file"""

        fid = open(self.rawfile, 'r')
        fid.seek(self.data_offset)
        data = np.fromfile(fid, dtype='h', count=self.nsamples * self.nchan)
        data = np.reshape(data, (self.nsamples, self.nchan))
        # amplifier applies only to the sensor channels 0-159
        amp_gain = self.output_gain * self.input_gain
        self.sensor_gain[:KIT.n_sens] = (self.sensor_gain[:KIT.n_sens] /
                                          amp_gain)
        conv_factor = np.array((KIT.VOLTAGE_RANGE / KIT.DYNAMIC_RANGE) *
                               self.sensor_gain, ndmin=2)
        self.x = (conv_factor * data).T

logger = logging.getLogger('mne')


class RawKIT(Raw):
    """Raw object from KIT SQD file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        absolute path to the sqd file.
    mrk_fname : str
        absolute path to marker coils file.
    elp_fname : str
        absolute path to elp digitizer laser points file.
    hsp_fname : str
        absolute path to elp digitizer head shape points file.
    sns_fname : str
        absolute path to sensor information file.
    data : bool | array-like
        if array-like custom data matching the header info to be used
        instead of the data from data_fname
    lowpass : int
        low-pass filter setting of the sqd file.
    highpass : int
        high-pass filter setting of the sqd file.
    stim : list
        list of trigger channels.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes & Methods
    --------------------
    See documentation for mne.fiff.Raw

        """
    @verbose
    def __init__(self, input_fname, mrk_fname, elp_fname, hsp_fname, sns_fname,
                 data=None, lowpass=None, highpass=None, stim=range(167, 159, -1),
                 stimthresh=3.5, verbose=True):

        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        self.params = sqd_params(input_fname, lowpass=lowpass,
                                 highpass=highpass)
        self._data_file = input_fname
        self.mrk_fname = mrk_fname
        self.elp_fname = elp_fname
        self.hsp_fname = hsp_fname
        self.sns_fname = sns_fname
        self.stim = stim
        self.stimthresh = stimthresh
        logger.info('Reading raw data from %s...' % input_fname)
        if not data:
            self.params.get_data()
            self._data = self.params.x
        else:
            self._data = data
        logger.info('Creating Raw.info structure...')
        assert len(self._data) == self.params.nchan
        self._create_raw_info_dict()
        self._create_synth_ch()
        self.verbose = verbose
        self._preloaded = True
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp, self.last_samp + 1)
        self._times /= self.info['sfreq']
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))
        # remove subclass helper attributes to create a proper Raw object.
        for attr in self.__dict__:
            if attr not in Raw.__dict__:
                del attr
        logger.info('Ready.')

    @verbose
    def _create_raw_info_dict(self):
        """Create raw.info dict for raw fif object with SQD data"""

        info = {}
        info['meas_id'] = None
        info['file_id'] = None
        info['meas_date'] = int(time.time())
        info['projs'] = []
        info['comps'] = []
        info['lowpass'] = self.params.lowpass
        info['highpass'] = self.params.highpass
        info['sfreq'] = float(self.params.sfreq)
        info['chs'], info['ch_names'] = self._create_chs_dict()
        info['nchan'] = self.params.nchan + 1  # adds synthetic channel
        info['bads'] = []
        info['acq_pars'], info['acq_stim'] = None, None
        info['filename'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = []
        info['filenames'] = []
        info['dev_head_t'] = {}
        info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD
        info['dev_head_t']['trans'], info['dig'] = self._get_coreg()
        logger.info('Done.')
        self.info = info

    def _create_chs_dict(self):
        """creates a list of dicts of meg channels for raw.info"""

        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                                in range(1, KIT.nmegchan + 1)]
        ch_names['REF'] = ['REF %03d' % ch for ch
                                in range(1, KIT.nrefchan + 1)]
        ch_names['TRIG'] = ['TRIG %03d' % ch for ch
                                 in range(1, KIT.ntrigchan + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                                 in range(1, KIT.nmiscchan + 1)]
        ch_names['STIM'] = ['STI 014']
        p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),([\.\-0-9]+),([\.\-0-9]+)' +
                       r'([\.\-0-9]+),([\.\-0-9]+)')
        locs = np.array(p.findall(open(self.sns_fname).read()), dtype='float')
        chan_locs = coreg.transform_pts(locs[:, [1, 0, 2]])
        chan_angles = locs[:, [3, 4]]
        chs = []
        for idx, ch_info in enumerate(zip(ch_names['MEG'] + ch_names['REF'],
                                          chan_locs, chan_angles), 1):
            ch_name, ch_loc, ch_angles = ch_info
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = KIT.RANGE
            chan_info['unit_mul'] = KIT.UNIT_MUL
            chan_info['ch_name'] = ch_name
            chan_info['unit'] = FIFF.FIFF_UNIT_T
            if ch_name.startswith('MEG'):
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
            if ch_name.startswith('REF'):
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE

            #    create three orthogonal vector
            #    0: theta, 1: phi
            ch_angles = np.radians(ch_angles)
            x = np.sin(ch_angles[0]) * np.cos(ch_angles[1])
            y = np.sin(ch_angles[0]) * np.sin(ch_angles[1])
            z = np.cos(ch_angles[0])
            point = np.array([x, y, z])
            length = np.linalg.norm(point)
            point = point / length
            vec1 = np.empty(point.size, dtype=float)
            vec1 = vec1 - np.sum(vec1 * point) * point
            length1 = np.linalg.norm(vec1)
            vec1 = vec1 / length1
            vec2 = np.cross(point, vec1)
            chan_info['loc'] = np.hstack((ch_loc, point, vec1, vec2))
            chs.append(chan_info)

        #    label trigger and misc channels
        for idy, ch_name in enumerate(ch_names['TRIG'] + ch_names['MISC'] +
                                      ch_names['STIM']):
            idy = idx + idy + 1
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idy
            chan_info['scanno'] = idy
            chan_info['range'] = 1.0
            chan_info['unit_mul'] = 0  # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            if ch_name.startswith('STI'):
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            else:
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            chs.append(chan_info)

        ch_names = (ch_names['MEG'] + ch_names['REF'] +
                    ch_names['TRIG'] + ch_names['MISC'] + ch_names['STIM'])
        return chs, ch_names

    def _create_synth_ch(self):
        """create a synthetic channel"""

        trig_chs = self._data[self.stim, :]
        trig_chs = trig_chs > self.stimthresh
        trig_vals = np.array(2 ** np.arange(len(self.stim)), ndmin=2).T
        trig_chs = trig_chs * trig_vals
        stim_ch = trig_chs.sum(axis=0)
        self._data = np.vstack((self._data, stim_ch))

    def _get_coreg(self):
        """get transformation matrix and hsp points"""

        coreg_data = coreg.coreg(mrk_fname=self.mrk_fname,
                                 elp_fname=self.elp_fname,
                                 hsp_fname=self.hsp_fname)
        dev_head_t = fit_matched_pts(tgt_pts=coreg_data.mrk_points,
                                     src_pts=coreg_data.elp_points)
        return dev_head_t, coreg_data.hsp_points

def read_raw_kit(input_fname, sns_fname, hsp_fname, elp_fname, mrk_fname,
                 stim=range(167, 159, -1), stimthresh=3.5):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        absolute path to the sqd file.
    mrk_fname : str
        absolute path to marker coils file.
    elp_fname : str
        absolute path to elp digitizer laser points file.
    hsp_fname : str
        absolute path to elp digitizer head shape points file.
    sns_fname : str
        absolute path to sensor information file.
    data : bool | array-like
        if array-like custom data matching the header info to be used
        instead of the data from data_fname
    lowpass : int
        low-pass filter setting of the sqd file.
    highpass : int
        high-pass filter setting of the sqd file.
    stim : list
        list of trigger channels.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.

    """
    return RawKIT(input_fname=input_fname, sns_fname=sns_fname,
                  hsp_fname=hsp_fname, elp_fname=elp_fname,
                  mrk_fname=mrk_fname, stim=stim, stimthresh=stimthresh)
