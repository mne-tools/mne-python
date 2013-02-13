"""Conversion tool from SQD to FIF

sqd_params class is adapted from Yoshiaki Adachi's Meginfo2.cpp
RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from ..raw import Raw
from ...utils import verbose
from ...transforms.coreg import fit_matched_pts
from ..constants import FIFF
from .constants import KIT, KIT_NY, KIT_AD
from .import coreg
from struct import unpack
import numpy as np
from scipy.linalg import norm
import time
import logging

logger = logging.getLogger('mne')


class sqd_params(object):
    """Extracts all the information from the sqd file.

    Parameters
    ----------
    rawfile: str
        raw sqd file to be read

    """
    def __init__(self, rawfile):
        self.rawfile = rawfile
        self.KIT = KIT
        fid = open(rawfile, 'r')
        fid.seek(self.KIT.BASIC_INFO)
        basic_offset = unpack('i', fid.read(self.KIT.INT))[0]
        fid.seek(basic_offset)

        # basic info
        self.version = unpack('i', fid.read(self.KIT.INT))[0]
        self.revision = unpack('i', fid.read(self.KIT.INT))[0]
        self.sysid = unpack('i', fid.read(self.KIT.INT))[0]
        self.sysname = unpack('128s', fid.read(self.KIT.STRING))
        self.sysname = self.sysname[0].split('\n')[0]
        self.modelname = unpack('128s', fid.read(self.KIT.STRING))
        self.modelname = self.modelname[0].split('\n')[0]
        self.nchan = unpack('i', fid.read(self.KIT.INT))[0]
        self.chan_no = range(self.nchan)

        if self.sysname == 'New York University Abu Dhabi':
            self.KIT = KIT_AD
        elif self.sysname == 'NYU 160ch System since Jan24 2009':
            self.KIT = KIT_NY
        else:
            raise NotImplementedError

        # amplifier gain
        fid.seek(self.KIT.AMPLIFIER_INFO)
        amp_offset = unpack('i', fid.read(self.KIT.INT))[0]
        fid.seek(amp_offset)
        amp_data = unpack('i', fid.read(self.KIT.INT))[0]

        self.gain1 = self.KIT.GAINS[(self.KIT.GAIN1_MASK & amp_data)
                                    >> self.KIT.GAIN1_BIT]
        self.gain2 = self.KIT.GAINS[(self.KIT.GAIN2_MASK & amp_data)
                                    >> self.KIT.GAIN2_BIT]
        if self.KIT.GAIN3_BIT:
            self.gain3 = self.KIT.GAINS[(self.KIT.GAIN3_MASK & amp_data)
                                        >> self.KIT.GAIN3_BIT]
            self.amp_gain = self.gain1 * self.gain2 * self.gain3
        else:
            self.amp_gain = self.gain1 * self.gain2

        # filter settings
        self.lowpass = self.KIT.LPFS[(self.KIT.LPF_MASK & amp_data)
                                    >> self.KIT.LPF_BIT]
        self.highpass = self.KIT.HPFS[(self.KIT.HPF_MASK & amp_data)
                                    >> self.KIT.HPF_BIT]
        self.notch = self.KIT.BEFS[(self.KIT.BEF_MASK & amp_data)
                                    >> self.KIT.BEF_BIT]

        # only sensor channels requires gain. the additional misc channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected

        fid.seek(self.KIT.CHAN_SENS)
        sens_offset = unpack('i', fid.read(self.KIT.INT))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=self.nchan * 2)
        self._sensitivities = (np.reshape(sens, (self.nchan, 2))
                               [:self.KIT.n_sens, 1])
        self.sensor_gain = np.ones(self.nchan)
        self.sensor_gain[:self.KIT.n_sens] = self._sensitivities

        fid.seek(self.KIT.SAMPLE_INFO)
        acqcond_offset = unpack('i', fid.read(self.KIT.INT))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(self.KIT.INT))[0]
        if acq_type == 1:
            self.sfreq = unpack('d', fid.read(self.KIT.DOUBLE))[0]
            _ = fid.read(self.KIT.INT)  # initialized estimate of samples
            self.nsamples = unpack('i', fid.read(self.KIT.INT))[0]
        else:
            raise NotImplementedError

        fid.seek(self.KIT.DATA_OFFSET)
        # data offset info
        self.data_offset = unpack('i', fid.read(self.KIT.INT))[0]

    def get_data(self):
        """returns an array with data extracted from sqd file"""

        fid = open(self.rawfile, 'r')
        fid.seek(self.data_offset)
        data = np.fromfile(fid, dtype='h', count=self.nsamples * self.nchan)
        data = np.reshape(data, (self.nsamples, self.nchan))
        # amplifier applies only to the sensor channels 0-159
        self.sensor_gain[:self.KIT.n_sens] /= self.amp_gain
        conv_factor = np.array((self.KIT.VOLTAGE_RANGE /
                                self.KIT.DYNAMIC_RANGE) *
                               self.sensor_gain, ndmin=2)
        self.x = (conv_factor * data).T


class RawKIT(Raw):
    """Raw object from KIT SQD file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk_fname : str
        Absolute path to marker coils file.
    elp_fname : str
        Absolute path to elp digitizer laser points file.
    hsp_fname : str
        Absolute path to elp digitizer head shape points file.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list
        List of trigger channels.
    data : bool | array-like
        Array-like data to use in lieu of data from sqd file.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.

    """
    @verbose
    def __init__(self, input_fname, mrk_fname, elp_fname, hsp_fname, sns_fname,
                 stim, data=None, stimthresh=3.5, verbose=True):

        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        params = sqd_params(input_fname)
        mrk_fname = mrk_fname
        elp_fname = elp_fname
        hsp_fname = hsp_fname
        sns_fname = sns_fname
        logger.info('Reading raw data from %s...' % input_fname)
        if not data:
            params.get_data()
            self._data = params.x
        else:
            self._data = data
        logger.info('Creating Raw.info structure...')
        assert len(self._data) == params.nchan

        # Create raw.info dict for raw fif object with SQD data
        self.info = {}
        self.info['meas_id'] = None
        self.info['file_id'] = None
        self.info['meas_date'] = int(time.time())
        self.info['projs'] = []
        self.info['comps'] = []
        self.info['lowpass'] = params.lowpass
        self.info['highpass'] = params.highpass
        self.info['sfreq'] = float(params.sfreq)
        self.info['nchan'] = params.nchan + 1  # adds synthetic channel
        self.info['bads'] = []
        self.info['acq_pars'], self.info['acq_stim'] = None, None
        self.info['filename'] = None
        self.info['ctf_head_t'] = None
        self.info['dev_ctf_t'] = []
        self.info['filenames'] = []
        self.info['dev_head_t'] = {}
        self.info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        self.info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD

        mrk, elp, self.info['dig'] = coreg.coreg(mrk_fname=mrk_fname,
                                                 elp_fname=elp_fname,
                                                 hsp_fname=hsp_fname)
        self.info['dev_head_t']['trans'] = fit_matched_pts(tgt_pts=mrk,
                                                           src_pts=elp)

        # Create a synthetic channel
        trig_chs = self._data[stim, :]
        trig_chs = trig_chs > stimthresh
        trig_vals = np.array(2 ** np.arange(len(stim)), ndmin=2).T
        trig_chs = trig_chs * trig_vals
        stim_ch = trig_chs.sum(axis=0)
        self._data = np.vstack((self._data, stim_ch))

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                                in range(1, params.KIT.n_sens + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                                 in range(1, params.KIT.nmiscchan + 1)]
        ch_names['STIM'] = ['STI 014']
        locs = coreg.read_sns(sns_fname=sns_fname)
        chan_locs = coreg.transform_pts(locs[:, :3])
        chan_angles = locs[:, 3:]
        self.info['chs'] = []
        for idx, ch_info in enumerate(zip(ch_names['MEG'], chan_locs,
                                          chan_angles), 1):
            ch_name, ch_loc, ch_angles = ch_info
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = KIT.RANGE
            chan_info['unit_mul'] = KIT.UNIT_MUL
            chan_info['ch_name'] = ch_name
            chan_info['unit'] = FIFF.FIFF_UNIT_T
            chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
            if idx <= params.KIT.nmegchan:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
            else:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH

            # create three orthogonal vector
            # ch_angles[0]: theta, ch_angles[1]: phi
            ch_angles = np.radians(ch_angles)
            x = np.sin(ch_angles[0]) * np.cos(ch_angles[1])
            y = np.sin(ch_angles[0]) * np.sin(ch_angles[1])
            z = np.cos(ch_angles[0])
            vec_z = np.array([x, y, z])
            length = norm(vec_z)
            vec_z /= length
            vec_x = np.zeros(vec_z.size, dtype=float)
            if vec_z[1] < vec_z[2]:
                if vec_z[0] < vec_z[1]:
                    vec_x[0] = 1.0
                else:
                    vec_x[1] = 1.0
            elif vec_z[0] < vec_z[2]:
                vec_x[0] = 1.0
            else:
                vec_x[2] = 1.0
            vec_x -= np.sum(vec_x * vec_z) * vec_z
            length = norm(vec_x)
            vec_x /= length
            vec_y = np.cross(vec_z, vec_x)
            # transform to Neuromag like coordinate space
            vecs = np.vstack((vec_x, vec_y, vec_z))
            vecs = coreg.transform_pts(vecs, scale=False)
            chan_info['loc'] = np.vstack((ch_loc, vecs)).ravel()
            self.info['chs'].append(chan_info)

        # label trigger and misc channels
        for idy, ch_name in enumerate(ch_names['MISC'] + ch_names['STIM'],
                                      params.KIT.n_sens):
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idy
            chan_info['scanno'] = idy
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_V
            chan_info['unit_mul'] = 0  # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            if ch_name.startswith('STI'):
                chan_info['unit'] = FIFF.FIFF_UNIT_NONE
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            else:
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            self.info['chs'].append(chan_info)

        self.info['ch_names'] = (ch_names['MEG'] + ch_names['MISC'] +
                                 ch_names['STIM'])

        self.verbose = verbose
        self._preloaded = True
        self.fids = list()
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp, self.last_samp + 1)
        self._times /= self.info['sfreq']
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')


def read_raw_kit(input_fname, mrk_fname, elp_fname, hsp_fname, sns_fname,
                 stim, data=None, stimthresh=3.5):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk_fname : str
        Absolute path to marker coils file.
    elp_fname : str
        Absolute path to elp digitizer laser points file.
    hsp_fname : str
        Absolute path to elp digitizer head shape points file.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list
        List of trigger channels.
    data : bool | array-like
        Array-like data to use in lieu of data from sqd file.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.

    """
    return RawKIT(input_fname=input_fname, mrk_fname=mrk_fname,
                  elp_fname=elp_fname, hsp_fname=hsp_fname,
                  sns_fname=sns_fname, stim=stim, data=data,
                  stimthresh=stimthresh)
